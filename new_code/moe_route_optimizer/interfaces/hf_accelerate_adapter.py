"""
Hugging Face Transformers 多进程专家并行 + 数据并行推理适配器

外部只需单线程调用，内部自动spawn EP+DP worker进程：
    adapter = create_hf_accelerate_adapter("path/to/model", ep_size=2)
    outputs = adapter.run_inference("Hello, how are you?")

内部实现：
    - 每个GPU一个独立的worker进程（rank）
    - 数据并行(DP)：输入prompts自动拆分到各worker，每个worker处理不同子集
    - 专家并行(EP)：每个worker只持有部分专家参数（物理分片）
    - MoE层通过all-to-all交换token实现真正的专家并行
    - DP+EP确保all-to-all交换的是不同worker的不同token，而非重复数据
"""

import os
import sys
import gc
import time
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

# 确保导入路径（使用相对路径，兼容任何工作环境）
import os as _os
workspace_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from interfaces.framework_interface import InferenceFrameworkInterface
from config import get_logger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _find_free_port():
    """找到一个可用的TCP端口用于NCCL通信"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _get_num_experts_from_config(config) -> int:
    return (
        getattr(config, 'num_experts', None)
        or getattr(config, 'moe_num_experts', None)
        or getattr(config, 'num_local_experts', None)
        or getattr(config, 'n_routed_experts', None)
        or 8
    )


def _supports_generic_ep_wrapper(config) -> bool:
    model_type = getattr(config, 'model_type', None)
    return model_type not in {'deepseek_v2', 'jetmoe'}


def _should_use_native_ep(config, world_size: int) -> bool:
    if world_size <= 1:
        return False
    # DeepSeek-V2 系列已经自带基于 ep_size 的专家并行实现，
    # 这里直接复用其原生路径，避免套用 Qwen 风格 wrapper。
    return getattr(config, 'model_type', None) in {'deepseek_v2'}


def _should_use_jetmoe_native_ep(config, world_size: int) -> bool:
    if world_size <= 1:
        return False
    return getattr(config, 'model_type', None) == 'jetmoe'


def _extract_gate_info(gate) -> Optional[Dict[str, Any]]:
    if gate is None:
        return None

    gate_for_meta = gate
    if (
        not hasattr(gate_for_meta, 'in_features')
        and hasattr(gate_for_meta, 'layer')
        and isinstance(gate_for_meta.layer, nn.Linear)
    ):
        gate_for_meta = gate_for_meta.layer

    state_dict = gate_for_meta.state_dict()
    weight = state_dict.get('weight')
    in_features = getattr(gate_for_meta, 'in_features', None)
    out_features = getattr(gate_for_meta, 'out_features', None)

    if weight is not None:
        if out_features is None:
            out_features = weight.shape[0]
        if in_features is None and weight.dim() >= 2:
            in_features = weight.shape[1]

    return {
        'state_dict': {k: v.cpu().clone() for k, v in state_dict.items()},
        'in_features': in_features,
        'out_features': out_features,
        'has_bias': getattr(gate, 'bias', None) is not None,
    }


def _extract_block_info(block, num_experts: int) -> Optional[Dict[str, Any]]:
    if block is None:
        return None

    gate = getattr(block, 'gate', None) or getattr(block, 'router', None)
    top_k = getattr(block, 'top_k', None)
    if top_k is None and hasattr(block, 'num_experts_per_tok'):
        top_k = block.num_experts_per_tok
    if top_k is None and gate is not None:
        top_k = getattr(gate, 'top_k', None)
    if top_k is None and hasattr(block, 'config'):
        top_k = getattr(block.config, 'moe_top_k', None)

    norm_topk_prob = getattr(block, 'norm_topk_prob', None)
    if norm_topk_prob is None and gate is not None:
        norm_topk_prob = getattr(gate, 'norm_topk_prob', None)
    if norm_topk_prob is None and hasattr(block, 'config'):
        norm_topk_prob = getattr(block.config, 'norm_topk_prob', True)

    return {
        'num_experts': getattr(block, 'num_experts', None) or num_experts,
        'top_k': top_k if top_k is not None else 2,
        'norm_topk_prob': True if norm_topk_prob is None else norm_topk_prob,
    }


class CUDATimer:
    """精确的CUDA计时器"""
    def __init__(self, device=None):
        self.device = device
        self._start_event = None
        self._end_event = None
        self._elapsed_ms = 0.0

    def start(self):
        if not torch.cuda.is_available():
            return
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop(self):
        if not torch.cuda.is_available() or self._start_event is None:
            return 0.0
        self._end_event.record()
        torch.cuda.synchronize()
        self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        return self._elapsed_ms / 1000.0

    def elapsed_seconds(self):
        return self._elapsed_ms / 1000.0


class ExpertParallelWrapper(nn.Module):
    """
    真正的专家并行包装器（基于torch.distributed all-to-all）

    每个GPU只持有部分专家参数（物理分片）：
    - GPU 0: 专家 0 ~ N/2-1
    - GPU 1: 专家 N/2 ~ N-1

    Forward流程：
    1. Gate计算路由决策
    2. All-to-all dispatch: 将token发送到目标GPU
    3. 本地专家计算
    4. All-to-all combine: 将结果发回原GPU
    """
    def __init__(self, original_moe_layer, world_size=2, rank=0,
                 num_experts=8, layer_idx=0, logger=None):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.experts_per_gpu = num_experts // world_size
        self.layer_idx = layer_idx
        self.logger = logger

        self.expert_start = rank * self.experts_per_gpu
        self.expert_end = self.expert_start + self.experts_per_gpu

        self.gate = getattr(original_moe_layer, 'gate', None)
        if self.gate is None:
            self.gate = getattr(original_moe_layer, 'router', None)

        original_experts = getattr(original_moe_layer, 'experts', None)
        if original_experts is not None and isinstance(original_experts, nn.ModuleList):
            self.local_experts = nn.ModuleList([
                original_experts[i] for i in range(self.expert_start, self.expert_end)
            ])
        else:
            self.local_experts = None
            self._log(f"[EP Layer {layer_idx}] GPU {rank}: experts not found")

        self.shared_expert = getattr(original_moe_layer, 'shared_expert', None)
        self.shared_expert_gate = getattr(original_moe_layer, 'shared_expert_gate', None)

        self._total_comm_delay = 0.0
        self._forward_count = 0

        self.top_k = getattr(original_moe_layer, 'top_k', 2)
        if hasattr(original_moe_layer, 'num_experts_per_tok'):
            self.top_k = original_moe_layer.num_experts_per_tok

        # 是否对topk权重进行重归一化（Qwen1.5-MoE config中 norm_topk_prob=false）
        self.norm_topk_prob = getattr(original_moe_layer, 'norm_topk_prob', True)

        # self._log(
        #     f"[EP Layer {layer_idx}] rank={rank}, world_size={world_size}, "
        #     f"local_experts={self.expert_start}-{self.expert_end-1}, "
        #     f"top_k={self.top_k}, norm_topk_prob={self.norm_topk_prob}"
        # )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def reset_comm_delay(self):
        self._total_comm_delay = 0.0
        self._forward_count = 0

    def get_comm_delay(self):
        return self._total_comm_delay

    def forward(self, hidden_states, *args, **kwargs):
        self._forward_count += 1

        if not dist.is_initialized() or self.world_size <= 1:
            raise RuntimeError("EP requires distributed env with world_size > 1")
        if self.local_experts is None or self.gate is None:
            raise RuntimeError(f"[EP Layer {self.layer_idx}] local_experts or gate is None")

        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_flat = hidden_states.view(num_tokens, hidden_size)

        # 路由（使用float32计算softmax以保证精度，与HF原始实现一致）
        router_logits = self.gate(hidden_flat)
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        expert_to_gpu = topk_indices // self.experts_per_gpu

        send_counts = []
        for gpu_id in range(self.world_size):
            send_counts.append((expert_to_gpu == gpu_id).sum().item())

        send_counts_tensor = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_tensor = torch.zeros_like(send_counts_tensor)

        comm_start = torch.cuda.Event(enable_timing=True)
        comm_mid = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        comm_start.record()

        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)

        recv_counts = recv_counts_tensor.tolist()
        total_recv = sum(recv_counts)
        total_send = sum(send_counts)

        # 准备发送数据
        send_hidden_list, send_weights_list = [], []
        send_expert_ids_list, send_token_ids_list = [], []

        for gpu_id in range(self.world_size):
            mask = (expert_to_gpu == gpu_id)
            if mask.any():
                token_ids, k_ids = torch.where(mask)
                send_hidden_list.append(hidden_flat[token_ids])
                send_weights_list.append(topk_weights[token_ids, k_ids])
                global_eids = topk_indices[token_ids, k_ids]
                send_expert_ids_list.append(global_eids - gpu_id * self.experts_per_gpu)
                send_token_ids_list.append(token_ids)
            else:
                send_hidden_list.append(torch.empty(0, hidden_size, device=device, dtype=dtype))
                send_weights_list.append(torch.empty(0, device=device, dtype=dtype))
                send_expert_ids_list.append(torch.empty(0, device=device, dtype=torch.long))
                send_token_ids_list.append(torch.empty(0, device=device, dtype=torch.long))

        # All-to-all dispatch
        recv_hidden = torch.zeros(total_recv, hidden_size, device=device, dtype=dtype)
        recv_weights = torch.zeros(total_recv, device=device, dtype=dtype)
        recv_eids = torch.zeros(total_recv, device=device, dtype=torch.long)
        recv_tids = torch.zeros(total_recv, device=device, dtype=torch.long)

        if total_send > 0 or total_recv > 0:
            for send_list, recv_ref, dim, dt in [
                (send_hidden_list, 'hidden', hidden_size, dtype),
                (send_weights_list, 'weights', None, dtype),
                (send_expert_ids_list, 'eids', None, torch.long),
                (send_token_ids_list, 'tids', None, torch.long),
            ]:
                s_split = [t.clone() for t in send_list]
                if dim is not None:
                    r_split = [torch.zeros(c, dim, device=device, dtype=dt) for c in recv_counts]
                else:
                    r_split = [torch.zeros(c, device=device, dtype=dt) for c in recv_counts]
                dist.all_to_all(r_split, s_split)
                cat = torch.cat(r_split, dim=0) if total_recv > 0 else None
                if recv_ref == 'hidden' and cat is not None:
                    recv_hidden = cat
                elif recv_ref == 'weights' and cat is not None:
                    recv_weights = cat
                elif recv_ref == 'eids' and cat is not None:
                    recv_eids = cat
                elif recv_ref == 'tids' and cat is not None:
                    recv_tids = cat

        comm_mid.record()

        # 本地专家计算（不计入通信时延）
        recv_output = torch.zeros_like(recv_hidden)
        if total_recv > 0:
            for local_idx in range(self.experts_per_gpu):
                emask = (recv_eids == local_idx)
                if emask.any():
                    recv_output[emask] = self.local_experts[local_idx](recv_hidden[emask])

        recv_output_w = recv_output * recv_weights.unsqueeze(-1)

        # All-to-all combine（单独计时，不含本地专家计算）
        comm_combine_start = torch.cuda.Event(enable_timing=True)
        comm_combine_start.record()

        send_output = torch.zeros(total_send, hidden_size, device=device, dtype=dtype)
        send_back_tids = torch.zeros(total_send, device=device, dtype=torch.long)

        if total_send > 0 or total_recv > 0:
            # 回传output
            r_out_split, offset = [], 0
            for c in recv_counts:
                if c > 0:
                    r_out_split.append(recv_output_w[offset:offset+c].clone())
                    offset += c
                else:
                    r_out_split.append(torch.empty(0, hidden_size, device=device, dtype=dtype))
            s_out_split = [torch.zeros(c, hidden_size, device=device, dtype=dtype) for c in send_counts]
            dist.all_to_all(s_out_split, r_out_split)
            if total_send > 0:
                send_output = torch.cat(s_out_split, dim=0)

            # 回传token ids
            r_tid_split, offset = [], 0
            for c in recv_counts:
                if c > 0:
                    r_tid_split.append(recv_tids[offset:offset+c].clone())
                    offset += c
                else:
                    r_tid_split.append(torch.empty(0, device=device, dtype=torch.long))
            s_tid_split = [torch.zeros(c, device=device, dtype=torch.long) for c in send_counts]
            dist.all_to_all(s_tid_split, r_tid_split)
            if total_send > 0:
                send_back_tids = torch.cat(s_tid_split, dim=0)

        comm_end.record()
        torch.cuda.synchronize()
        dispatch_ms = comm_start.elapsed_time(comm_mid)
        combine_ms = comm_combine_start.elapsed_time(comm_end)
        self._total_comm_delay += (dispatch_ms + combine_ms) / 1000.0

        # 汇聚结果
        output_flat = torch.zeros_like(hidden_flat)
        if total_send > 0:
            output_flat.scatter_add_(
                0, send_back_tids.unsqueeze(-1).expand(-1, hidden_size), send_output
            )

        # 共享专家
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_flat)
            if self.shared_expert_gate is not None:
                sg = torch.sigmoid(self.shared_expert_gate(hidden_flat))
                output_flat = output_flat + sg * shared_out
            else:
                output_flat = output_flat + shared_out

        # if self._forward_count <= 3 or self._forward_count % 50 == 0:
            # self._log(
            #     f"[EP Layer {self.layer_idx}] #{self._forward_count}: "
            #     f"tokens={num_tokens}, send={total_send}, recv={total_recv}, "
            #     f"dispatch={dispatch_ms:.1f}ms, combine={combine_ms:.1f}ms"
            # )

        return output_flat.view(batch_size, seq_len, hidden_size)


class JetMoeNativeEPWrapper(nn.Module):
    """
    JetMoE MLP 专用 EP 包装器。

    JetMoE 的 MLP 专家参数存放在两组并行专家权重中：
    - input_linear.weight: [num_experts, hidden*2, input]
    - output_linear.weight: [num_experts, input, hidden]

    这里按照 expert 维度做物理切分，并通过 all-to-all 分发 token 到目标 rank，
    从而避免对 JetMoE 误用通用 ModuleList-experts wrapper。
    """

    def __init__(self, original_moe_layer, world_size=2, rank=0,
                 num_experts=8, layer_idx=0, logger=None):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.experts_per_gpu = num_experts // world_size
        self.layer_idx = layer_idx
        self.logger = logger

        self.expert_start = rank * self.experts_per_gpu
        self.expert_end = self.expert_start + self.experts_per_gpu

        self.gate = original_moe_layer.router
        self.input_size = original_moe_layer.input_size
        self.hidden_size = original_moe_layer.hidden_size
        self.top_k = getattr(self.gate, 'top_k', 2)
        self.activation = original_moe_layer.activation
        self.bias = original_moe_layer.bias

        self.input_linear_weight = nn.Parameter(
            original_moe_layer.input_linear.weight[self.expert_start:self.expert_end].contiguous()
        )
        self.output_linear_weight = nn.Parameter(
            original_moe_layer.output_linear.weight[self.expert_start:self.expert_end].contiguous()
        )

        self._total_comm_delay = 0.0
        self._forward_count = 0

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def reset_comm_delay(self):
        self._total_comm_delay = 0.0
        self._forward_count = 0

    def get_comm_delay(self):
        return self._total_comm_delay

    def _run_local_expert(self, hidden_states, local_expert_idx):
        hidden_states = F.linear(hidden_states, self.input_linear_weight[local_expert_idx])
        left, right = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(left) * right
        hidden_states = F.linear(hidden_states, self.output_linear_weight[local_expert_idx])
        return hidden_states

    def forward(self, layer_input, *args, **kwargs):
        self._forward_count += 1

        if not dist.is_initialized() or self.world_size <= 1:
            raise RuntimeError("JetMoE native EP requires distributed env with world_size > 1")

        batch_size, seq_len, emb_size = layer_input.size()
        flat_input = layer_input.reshape(-1, emb_size)
        device = flat_input.device
        dtype = flat_input.dtype

        _, batch_index, batch_gates, expert_size, router_logits = self.gate(flat_input)
        if isinstance(expert_size, torch.Tensor):
            expert_size = expert_size.tolist()

        expert_inputs = flat_input[batch_index]
        global_eids = torch.repeat_interleave(
            torch.arange(self.num_experts, device=device, dtype=torch.long),
            torch.tensor(expert_size, device=device, dtype=torch.long),
        )
        expert_to_gpu = global_eids // self.experts_per_gpu

        send_counts = [(expert_to_gpu == gpu_id).sum().item() for gpu_id in range(self.world_size)]
        send_counts_tensor = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_tensor = torch.zeros_like(send_counts_tensor)

        comm_start = torch.cuda.Event(enable_timing=True)
        comm_mid = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        comm_start.record()

        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)

        recv_counts = recv_counts_tensor.tolist()
        total_send = sum(send_counts)
        total_recv = sum(recv_counts)

        send_hidden_list, send_gates_list = [], []
        send_eids_list, send_tids_list = [], []
        for gpu_id in range(self.world_size):
            mask = expert_to_gpu == gpu_id
            if mask.any():
                send_hidden_list.append(expert_inputs[mask])
                send_gates_list.append(batch_gates[mask].to(dtype))
                send_eids_list.append(global_eids[mask] - gpu_id * self.experts_per_gpu)
                send_tids_list.append(batch_index[mask])
            else:
                send_hidden_list.append(torch.empty(0, emb_size, device=device, dtype=dtype))
                send_gates_list.append(torch.empty(0, device=device, dtype=dtype))
                send_eids_list.append(torch.empty(0, device=device, dtype=torch.long))
                send_tids_list.append(torch.empty(0, device=device, dtype=torch.long))

        recv_hidden = torch.zeros(total_recv, emb_size, device=device, dtype=dtype)
        recv_gates = torch.zeros(total_recv, device=device, dtype=dtype)
        recv_eids = torch.zeros(total_recv, device=device, dtype=torch.long)
        recv_tids = torch.zeros(total_recv, device=device, dtype=torch.long)

        if total_send > 0 or total_recv > 0:
            for send_list, recv_ref, dim, dt in [
                (send_hidden_list, 'hidden', emb_size, dtype),
                (send_gates_list, 'gates', None, dtype),
                (send_eids_list, 'eids', None, torch.long),
                (send_tids_list, 'tids', None, torch.long),
            ]:
                send_split = [t.clone() for t in send_list]
                if dim is not None:
                    recv_split = [torch.zeros(c, dim, device=device, dtype=dt) for c in recv_counts]
                else:
                    recv_split = [torch.zeros(c, device=device, dtype=dt) for c in recv_counts]
                dist.all_to_all(recv_split, send_split)
                cat = torch.cat(recv_split, dim=0) if total_recv > 0 else None
                if recv_ref == 'hidden' and cat is not None:
                    recv_hidden = cat
                elif recv_ref == 'gates' and cat is not None:
                    recv_gates = cat
                elif recv_ref == 'eids' and cat is not None:
                    recv_eids = cat
                elif recv_ref == 'tids' and cat is not None:
                    recv_tids = cat

        comm_mid.record()

        recv_output = torch.zeros(total_recv, self.input_size, device=device, dtype=dtype)
        if total_recv > 0:
            for local_idx in range(self.experts_per_gpu):
                mask = recv_eids == local_idx
                if mask.any():
                    recv_output[mask] = self._run_local_expert(recv_hidden[mask], local_idx)
        recv_output = recv_output * recv_gates.unsqueeze(-1)

        send_output = torch.zeros(total_send, self.input_size, device=device, dtype=dtype)
        send_back_tids = torch.zeros(total_send, device=device, dtype=torch.long)
        if total_send > 0 or total_recv > 0:
            recv_output_split, offset = [], 0
            for c in recv_counts:
                if c > 0:
                    recv_output_split.append(recv_output[offset:offset + c].clone())
                    offset += c
                else:
                    recv_output_split.append(torch.empty(0, self.input_size, device=device, dtype=dtype))
            send_output_split = [
                torch.zeros(c, self.input_size, device=device, dtype=dtype) for c in send_counts
            ]
            dist.all_to_all(send_output_split, recv_output_split)
            if total_send > 0:
                send_output = torch.cat(send_output_split, dim=0)

            recv_tid_split, offset = [], 0
            for c in recv_counts:
                if c > 0:
                    recv_tid_split.append(recv_tids[offset:offset + c].clone())
                    offset += c
                else:
                    recv_tid_split.append(torch.empty(0, device=device, dtype=torch.long))
            send_tid_split = [torch.zeros(c, device=device, dtype=torch.long) for c in send_counts]
            dist.all_to_all(send_tid_split, recv_tid_split)
            if total_send > 0:
                send_back_tids = torch.cat(send_tid_split, dim=0)

        comm_end.record()
        torch.cuda.synchronize()
        self._total_comm_delay += (
            comm_start.elapsed_time(comm_mid) + comm_mid.elapsed_time(comm_end)
        ) / 1000.0

        zeros = torch.zeros(batch_size * seq_len, self.input_size, device=device, dtype=dtype)
        if total_send > 0:
            zeros.index_add_(0, send_back_tids, send_output)
        layer_output = zeros.view(batch_size, seq_len, self.input_size)
        layer_output = layer_output + self.bias.to(dtype)
        return layer_output, router_logits


class DeepSeekNativeEPWrapper(nn.Module):
    """
    DeepSeek-V2 风格原生 EP 包装器。

    DeepSeek 模型本身已经支持基于 config.ep_size 的专家切分和 all-to-all 通信，
    这里不再改写 gate 逻辑，只复用其原生 MoE 路径并补充通信时延统计，
    从而兼容训练链路对 get_comm_delay()/hook 的依赖。
    """

    def __init__(self, original_moe_layer, layer_idx=0, logger=None):
        super().__init__()
        self.original_moe_layer = original_moe_layer
        self.config = getattr(original_moe_layer, 'config', None)
        self.layer_idx = layer_idx
        self.logger = logger

        self.gate = getattr(original_moe_layer, 'gate', None)
        self.experts = getattr(original_moe_layer, 'experts', None)
        self.shared_experts = getattr(original_moe_layer, 'shared_experts', None)

        self.num_experts_per_tok = getattr(original_moe_layer, 'num_experts_per_tok', 1)
        self.ep_size = getattr(original_moe_layer, 'ep_size', 1)
        self.experts_per_rank = getattr(original_moe_layer, 'experts_per_rank', 0)
        self.ep_rank = getattr(original_moe_layer, 'ep_rank', 0)

        self._total_comm_delay = 0.0
        self._forward_count = 0

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def reset_comm_delay(self):
        self._total_comm_delay = 0.0
        self._forward_count = 0

    def get_comm_delay(self):
        return self._total_comm_delay

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        if self.experts is None:
            raise RuntimeError(f"[DeepSeek EP Layer {self.layer_idx}] experts is None")

        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        input_split_sizes = None
        output_splits = None
        gathered_idxs = None

        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])

            comm_start = torch.cuda.Event(enable_timing=True)
            comm_mid = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)
            comm_start.record()

            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)

            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(),
                sorted_tokens.shape[1],
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            comm_mid.record()

            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gathered_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            start = 0
            for expert_idx, token_count in enumerate(tokens_per_expert_group.cpu().numpy()):
                gathered_idxs[start:start + token_count] = expert_idx % self.experts_per_rank
                start += token_count
            gathered_idxs = gathered_idxs.argsort()
            sorted_tokens = gathered_tokens[gathered_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        else:
            gathered_tokens = None

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[expert_idx + self.ep_rank * self.experts_per_rank]
            if expert is None:
                raise RuntimeError(
                    f"[DeepSeek EP Layer {self.layer_idx}] expert "
                    f"{expert_idx + self.ep_rank * self.experts_per_rank} is None"
                )
            outputs.append(expert(sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx

        outs = (
            torch.cat(outputs, dim=0)
            if outputs
            else sorted_tokens.new_empty((0, sorted_tokens.shape[-1]))
        )

        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gathered_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            comm_end.record()
            torch.cuda.synchronize()
            self._total_comm_delay += (
                comm_start.elapsed_time(comm_mid) + comm_mid.elapsed_time(comm_end)
            ) / 1000.0
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    def forward(self, hidden_states, *args, **kwargs):
        self._forward_count += 1

        if self.training:
            return self.original_moe_layer(hidden_states, *args, **kwargs)

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class _RemovableHookHandle:
    """可移除的hook句柄，调用remove()时通知所有worker移除对应hook"""

    def __init__(self, cmd_queues, result_queues, hook_id, target_name):
        self._cmd_queues = cmd_queues
        self._result_queues = result_queues
        self._hook_id = hook_id
        self._target_name = target_name

    def remove(self):
        for q in self._cmd_queues:
            q.put({
                'type': 'remove_hook',
                'target': self._target_name,
                'hook_id': self._hook_id,
            })
        for rq in self._result_queues:
            rq.get(timeout=30)


class _WorkerModuleProxy(nn.Module):
    """
    主进程中的模块代理，将hook注册转发到worker进程的实际模块上。

    当用户调用 proxy.register_forward_pre_hook(fn) 时，hook会被发送到
    所有worker进程，注册在实际参与推理的模块上，因此每次推理时hook都会触发。

    注意：hook函数必须是可pickle的（模块级函数或可序列化的类实例）。
    """

    def __init__(self, cmd_queues, result_queues, target_name, adapter=None,
                 gate_state_dict=None, gate_in_features=None, gate_out_features=None,
                 gate_has_bias=False, num_experts=None, top_k=None, norm_topk_prob=None):
        super().__init__()
        self._cmd_queues = cmd_queues
        self._result_queues = result_queues
        self._target_name = target_name  # 'first_moe_block' or 'first_moe_gate'
        self._adapter = adapter  # back-reference to HFAccelerateAdapter
        self._hook_counter = 0

        # 元数据（用于参数检查）
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # 重建gate权重副本（用于参数检查，如 proxy.gate.weight）
        if gate_state_dict is not None and gate_in_features is not None:
            self.gate = nn.Linear(gate_in_features, gate_out_features, bias=gate_has_bias)
            self.gate.load_state_dict(gate_state_dict)
            self.gate.eval()

    def _send_cmd_and_wait(self, cmd):
        """向所有worker发送命令并等待确认"""
        for q in self._cmd_queues:
            q.put(cmd)
        for rq in self._result_queues:
            result = rq.get(timeout=30)
            if result.get('status') != 'ok':
                raise RuntimeError(f"Worker command failed: {result}")

    def register_forward_pre_hook(self, hook, *args, **kwargs):
        """
        在worker进程的实际模块上注册forward pre-hook。
        hook会在每次推理时、该模块forward之前被调用。
        """
        hook_id = f"{self._target_name}_pre_{self._hook_counter}"
        self._hook_counter += 1
        self._send_cmd_and_wait({
            'type': 'register_hook',
            'target': self._target_name,
            'hook_type': 'forward_pre',
            'hook_fn': hook,
            'hook_id': hook_id,
        })
        return _RemovableHookHandle(
            self._cmd_queues, self._result_queues, hook_id, self._target_name
        )

    def register_forward_hook(self, hook, *args, **kwargs):
        """
        在worker进程的实际模块上注册forward hook。
        hook会在每次推理时、该模块forward之后被调用。
        """
        hook_id = f"{self._target_name}_fwd_{self._hook_counter}"
        self._hook_counter += 1
        self._send_cmd_and_wait({
            'type': 'register_hook',
            'target': self._target_name,
            'hook_type': 'forward',
            'hook_fn': hook,
            'hook_id': hook_id,
        })
        return _RemovableHookHandle(
            self._cmd_queues, self._result_queues, hook_id, self._target_name
        )

    def _register_perturbation_generator(self, generator):
        """
        将扰动生成器发送到所有worker进程，在目标模块上注册扰动hook。
        generator必须是nn.Module（可pickle）。
        """
        import copy
        gen_cpu = copy.deepcopy(generator).cpu()
        for q in self._cmd_queues:
            q.put({
                'type': 'register_perturbation_hook',
                'target': self._target_name,
                'generator': gen_cpu,
            })
        for rq in self._result_queues:
            result = rq.get(timeout=120)
            if result.get('status') != 'ok':
                raise RuntimeError(f"Failed to register perturbation hook: {result}")

    def _sync_perturbation_weights(self, state_dict):
        """将更新后的generator权重同步到所有worker"""
        cpu_sd = {k: v.cpu().clone() for k, v in state_dict.items()}
        for q in self._cmd_queues:
            q.put({'type': 'sync_perturbation_weights', 'state_dict': cpu_sd})
        for rq in self._result_queues:
            rq.get(timeout=30)

    def _set_hook_enabled(self, enabled):
        """设置worker中hook的启用/禁用状态"""
        for q in self._cmd_queues:
            q.put({'type': 'set_hook_enabled', 'enabled': enabled})
        for rq in self._result_queues:
            rq.get(timeout=30)

    def _set_hook_training(self, training):
        """设置worker中hook的训练/推理模式"""
        for q in self._cmd_queues:
            q.put({'type': 'set_hook_training', 'training': training})
        for rq in self._result_queues:
            rq.get(timeout=30)

    def _clear_hook_buffer(self):
        """清空worker中hook收集的状态"""
        for q in self._cmd_queues:
            q.put({'type': 'clear_hook_buffer'})
        for rq in self._result_queues:
            rq.get(timeout=30)

    def _get_collected_states(self):
        """获取最近一次推理中hook收集的状态（来自adapter的缓存）"""
        if self._adapter is not None:
            return self._adapter._last_hook_states
        return []

    def _remove_perturbation_hook(self):
        """通知所有worker移除扰动hook"""
        for q in self._cmd_queues:
            q.put({'type': 'remove_perturbation_hook'})
        for rq in self._result_queues:
            rq.get(timeout=30)


# ========== Worker辅助函数 ==========

def _find_moe_layers_in_model(model, logger=None):
    """查找模型中的MoE层"""
    moe_layers, moe_names = [], []
    first_block, first_gate = None, None
    model_type = getattr(getattr(model, 'config', None), 'model_type', None)

    if model_type == 'jetmoe':
        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            if cls == 'jetmoemoe':
                moe_layers.append(module)
                moe_names.append(name)
                if first_block is None:
                    first_block = module
                if first_gate is None:
                    first_gate = getattr(module, 'router', None)
        if logger:
            logger.info(f"Found {len(moe_layers)} JetMoE layers")
        return moe_layers, moe_names, first_block, first_gate

    block_kw = ['sparsemoe', 'moeblock', 'moelayer', 'moe']
    exclude_kw = ['expert', 'fusedmoe', 'sharedfusedmoe', 'gate', 'router']

    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        is_moe = any(k in cls for k in block_kw)
        is_excl = any(k in cls for k in exclude_kw)
        has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
        has_experts = hasattr(module, 'experts') or hasattr(module, 'expert')
        if is_moe and not is_excl and has_gate and has_experts:
            moe_layers.append(module)
            moe_names.append(name)
            if first_block is None:
                first_block = module
            if first_gate is None:
                first_gate = getattr(module, 'gate', None) or getattr(module, 'router', None)

    if not moe_layers:
        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            nm = name.lower()
            if any(k in cls for k in ['moe', 'expert', 'mixture']) or \
               any(k in nm for k in ['moe', 'experts']):
                moe_layers.append(module)
                moe_names.append(name)
                if first_block is None:
                    first_block = module
                if first_gate is None:
                    for attr in ['gate', 'router', 'gating_network', 'w_gate']:
                        if hasattr(module, attr):
                            first_gate = getattr(module, attr)
                            break

    if logger:
        logger.info(f"Found {len(moe_layers)} MoE layers")
    return moe_layers, moe_names, first_block, first_gate


def _tensor_to_queue_payload(tensor: torch.Tensor) -> Dict[str, Any]:
    """Convert a tensor into a multiprocessing.Queue-safe payload."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    array_tensor = cpu_tensor
    if cpu_tensor.dtype == torch.bfloat16:
        array_tensor = cpu_tensor.to(torch.float32)
    return {
        'array': array_tensor.numpy(),
        'dtype': str(cpu_tensor.dtype),
    }


def _serialize_hook_state_for_queue(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Convert collected hook state into payloads that do not rely on torch FD sharing."""
    return {
        key: _tensor_to_queue_payload(value)
        for key, value in state.items()
    }


def _replace_module_in_model(model, module_name, new_module):
    """根据名称路径替换模型中的模块"""
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


# ========== EP Worker进程 ==========

def _ep_worker_fn(rank, world_size, master_port, cmd_queue, result_queue):
    """
    EP worker进程主函数。
    每个worker: 初始化dist → 加载模型 → 应用EP → 进入命令循环。
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    logger = get_logger()
    logger.info(f"[Worker {rank}] Started on {device}")

    model, tokenizer, ep_wrappers = None, None, []
    hook_handles = {}  # hook_id -> RemovableHandle
    hook_targets = {
        'first_moe_block': None,
        'first_moe_gate': None,
    }
    hook_state = {
        'generator': None,
        'enabled': True,
        'training': True,
        'collected_states': [],
        'hook_handle': None,
    }
    hidden_size = 0

    try:
        while True:
            cmd = cmd_queue.get()

            if cmd['type'] == 'load_model':
                model_path = cmd['model_path']
                kw = cmd.get('kwargs', {})
                trust_remote = kw.get('trust_remote_code', True)
                torch_dtype_val = kw.get('torch_dtype', 'auto')

                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

                config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote)
                hidden_size = getattr(config, 'hidden_size', 2048)
                num_experts = _get_num_experts_from_config(config)
                use_native_ep = _should_use_native_ep(config, world_size)
                use_jetmoe_native_ep = _should_use_jetmoe_native_ep(config, world_size)
                use_generic_ep_wrapper = _supports_generic_ep_wrapper(config)
                latency_metric = 'end_to_end'

                if use_native_ep:
                    if num_experts % world_size != 0:
                        raise RuntimeError(
                            f"Native EP model requires num_experts divisible by world_size, "
                            f"got num_experts={num_experts}, world_size={world_size}"
                        )
                    config.ep_size = world_size
                    logger.info(
                        f"[Worker {rank}] Enabling native EP for model_type="
                        f"{getattr(config, 'model_type', 'unknown')} with ep_size={world_size}"
                    )
                if use_jetmoe_native_ep and num_experts % world_size != 0:
                    raise RuntimeError(
                        f"JetMoE native EP requires num_experts divisible by world_size, "
                        f"got num_experts={num_experts}, world_size={world_size}"
                    )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote, padding_side='left'
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if torch_dtype_val == 'auto':
                    model_dtype = torch.float16
                elif isinstance(torch_dtype_val, str):
                    model_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                                   'float32': torch.float32}.get(torch_dtype_val, torch.float16)
                else:
                    model_dtype = torch_dtype_val

                # ====== Step 1: 先加载模型到CPU（避免每个GPU都加载全量专家占满显存） ======
                logger.info(f"[Worker {rank}] Loading model to CPU (dtype={model_dtype})...")
                load_args = {
                    'trust_remote_code': trust_remote,
                    'torch_dtype': model_dtype,
                    'low_cpu_mem_usage': True,
                }
                if use_native_ep:
                    load_args['config'] = config
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_args)

                # ====== Step 2: 在CPU上应用EP，只保留本rank的专家 ======
                moe_layers, moe_names, first_block, first_gate = _find_moe_layers_in_model(model, logger)
                num_detected_moe_layers = len(moe_layers)
                orig_first_block = first_block
                orig_first_gate = first_gate
                experts_per_gpu = num_experts // world_size

                # 捕获gate信息（仅rank 0，通过queue传回主进程用于参数检查）
                gate_info = _extract_gate_info(first_gate) if rank == 0 else None
                block_info = _extract_block_info(first_block, num_experts) if rank == 0 else None

                if world_size > 1 and moe_layers:
                    if use_native_ep:
                        latency_metric = 'ep_comm'
                        logger.info(
                            f"[Worker {rank}] Wrapping native EP MoE layers: {len(moe_layers)} layers, "
                            f"{num_experts} experts total, local experts per rank={experts_per_gpu}"
                        )
                        for idx, (layer, name) in enumerate(zip(moe_layers, moe_names)):
                            wrapper = DeepSeekNativeEPWrapper(
                                layer, layer_idx=idx, logger=logger,
                            )
                            ep_wrappers.append(wrapper)
                            _replace_module_in_model(model, name, wrapper)
                    elif use_jetmoe_native_ep:
                        latency_metric = 'ep_comm'
                        logger.info(
                            f"[Worker {rank}] Wrapping JetMoE native EP layers: {len(moe_layers)} layers, "
                            f"{num_experts} experts total, local experts per rank={experts_per_gpu}"
                        )
                        for idx, (layer, name) in enumerate(zip(moe_layers, moe_names)):
                            wrapper = JetMoeNativeEPWrapper(
                                layer,
                                world_size=world_size,
                                rank=rank,
                                num_experts=num_experts,
                                layer_idx=idx,
                                logger=logger,
                            )
                            ep_wrappers.append(wrapper)
                            _replace_module_in_model(model, name, wrapper)
                    elif use_generic_ep_wrapper:
                        latency_metric = 'ep_comm'
                        logger.info(
                            f"[Worker {rank}] Applying EP on CPU: {len(moe_layers)} layers, "
                            f"{num_experts} experts total, keeping experts "
                            f"{rank * experts_per_gpu}-{(rank + 1) * experts_per_gpu - 1}"
                        )
                        for idx, (layer, name) in enumerate(zip(moe_layers, moe_names)):
                            wrapper = ExpertParallelWrapper(
                                layer, world_size=world_size, rank=rank,
                                num_experts=num_experts, layer_idx=idx, logger=logger,
                            )
                            ep_wrappers.append(wrapper)
                            _replace_module_in_model(model, name, wrapper)

                        # 显式删除对原始MoE层（含全量专家）的引用，使非本地专家可被GC回收
                        logger.info(f"[Worker {rank}] Non-local experts freed from CPU memory")
                    else:
                        logger.info(
                            f"[Worker {rank}] model_type={getattr(config, 'model_type', 'unknown')} "
                            f"does not match generic EP wrapper assumptions; keeping original MoE "
                            f"layers and using replicated DP execution"
                        )
                    del moe_layers, moe_names
                    gc.collect()
                else:
                    del moe_layers, moe_names

                if ep_wrappers:
                    hook_targets['first_moe_block'] = ep_wrappers[0]
                    hook_targets['first_moe_gate'] = ep_wrappers[0].gate
                else:
                    hook_targets['first_moe_block'] = orig_first_block
                    hook_targets['first_moe_gate'] = orig_first_gate

                del first_block, first_gate, orig_first_block, orig_first_gate

                # ====== Step 3: 将修剪后的模型（只含本地专家）移动到GPU ======
                logger.info(f"[Worker {rank}] Moving pruned model to {device}...")
                model = model.to(device)
                model.eval()
                gc.collect()
                torch.cuda.empty_cache()

                dist.barrier()
                logger.info(f"[Worker {rank}] Model loaded and EP applied on {device}")

                result_queue.put({
                    'status': 'ok',
                    'hidden_size': hidden_size,
                    'num_moe_layers': len(ep_wrappers) if ep_wrappers else num_detected_moe_layers,
                    'gate_info': gate_info,
                    'block_info': block_info,
                    'latency_metric': latency_metric,
                })

            elif cmd['type'] == 'register_hook':
                target_name = cmd['target']
                hook_type = cmd['hook_type']
                hook_fn = cmd['hook_fn']
                hook_id = cmd['hook_id']

                # 确定目标模块
                target_module = hook_targets.get(target_name)

                if target_module is not None:
                    if hook_type == 'forward_pre':
                        handle = target_module.register_forward_pre_hook(hook_fn)
                    else:
                        handle = target_module.register_forward_hook(hook_fn)
                    hook_handles[hook_id] = handle
                    logger.info(f"[Worker {rank}] Registered {hook_type} hook '{hook_id}' on {target_name}")
                    result_queue.put({'status': 'ok'})
                else:
                    logger.warning(f"[Worker {rank}] Hook target '{target_name}' not found")
                    result_queue.put({'status': 'ok'})  # 不阻塞主进程

            elif cmd['type'] == 'remove_hook':
                hook_id = cmd['hook_id']
                if hook_id in hook_handles:
                    hook_handles[hook_id].remove()
                    del hook_handles[hook_id]
                    logger.info(f"[Worker {rank}] Removed hook '{hook_id}'")
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'register_perturbation_hook':
                target_name = cmd['target']
                gen = cmd['generator'].to(device)
                hook_state['generator'] = gen
                hook_state['generator'].eval()

                target_module = hook_targets.get(target_name)

                if target_module is not None:
                    if hook_state['hook_handle'] is not None:
                        hook_state['hook_handle'].remove()

                    def _perturbation_hook(module, args):
                        if not hook_state['enabled']:
                            return args
                        if len(args) == 0:
                            return args
                        hidden_states = args[0]
                        if not isinstance(hidden_states, torch.Tensor):
                            return args
                        if hidden_states.dim() >= 2 and hidden_states.shape[1] == 1:
                            return args
                        orig_shape = hidden_states.shape
                        orig_dim = hidden_states.dim()
                        if orig_dim == 2:
                            hidden_states_3d = hidden_states.unsqueeze(0)
                        elif orig_dim == 3:
                            hidden_states_3d = hidden_states
                        else:
                            return args
                        gen_local = hook_state['generator']
                        with torch.no_grad():
                            result = gen_local(
                                hidden_states_3d,
                                deterministic=not hook_state['training'],
                            )
                        perturbed_hidden_3d = result['perturbed_hidden_states']
                        if orig_dim == 2:
                            perturbed_hidden = perturbed_hidden_3d.squeeze(0)
                        else:
                            perturbed_hidden = perturbed_hidden_3d
                        perturbed_hidden = perturbed_hidden.to(hidden_states.dtype)
                        if perturbed_hidden.shape != orig_shape:
                            perturbed_hidden = perturbed_hidden.view(orig_shape)
                        if hook_state['training']:
                            hook_state['collected_states'].append({
                                'hidden_states': hidden_states_3d.detach().cpu().clone(),
                                'selected_indices': result['selected_indices'].detach().cpu().clone(),
                                'perturb_dim_indices': result['perturb_dim_indices'].detach().cpu().clone(),
                                'log_prob': result['log_prob'].detach().cpu().clone(),
                                'perturbation': perturbed_hidden_3d.detach().cpu().clone(),
                            })
                        return (perturbed_hidden,) + args[1:]

                    handle = target_module.register_forward_pre_hook(_perturbation_hook)
                    hook_state['hook_handle'] = handle
                    logger.info(f"[Worker {rank}] Registered perturbation hook on {target_name}")
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'sync_perturbation_weights':
                if hook_state['generator'] is not None:
                    sd = {k: v.to(device) for k, v in cmd['state_dict'].items()}
                    hook_state['generator'].load_state_dict(sd)
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'set_hook_enabled':
                hook_state['enabled'] = cmd['enabled']
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'set_hook_training':
                hook_state['training'] = cmd['training']
                if hook_state['generator'] is not None:
                    if cmd['training']:
                        hook_state['generator'].train()
                    else:
                        hook_state['generator'].eval()
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'clear_hook_buffer':
                hook_state['collected_states'].clear()
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'remove_perturbation_hook':
                if hook_state['hook_handle'] is not None:
                    hook_state['hook_handle'].remove()
                    hook_state['hook_handle'] = None
                hook_state['generator'] = None
                hook_state['collected_states'].clear()
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'run_inference':
                prompts = cmd['inputs']
                gen_config = cmd.get('gen_config', {})

                # 统一随机种子确保所有worker生成一致结果
                seed = gen_config.pop('seed', 42)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                encoded = tokenizer(
                    prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048,
                ).to(device)

                for w in ep_wrappers:
                    w.reset_comm_delay()

                t0 = time.perf_counter()
                use_cache = gen_config.get('use_cache')
                if use_cache is None:
                    model_type = getattr(getattr(model, 'config', None), 'model_type', None)
                    use_cache = False if model_type == 'deepseek_v2' else True
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=gen_config.get('max_new_tokens', 128),
                        do_sample=gen_config.get('do_sample', False),
                        temperature=gen_config.get('temperature', 1.0),
                        top_p=gen_config.get('top_p', 1.0),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=None,
                        use_cache=use_cache,
                    )
                t1 = time.perf_counter()

                ep_comm = sum(w.get_comm_delay() for w in ep_wrappers)
                input_len = encoded['input_ids'].shape[1]
                gen_ids = outputs[:, input_len:]

                results, total_chars = [], 0
                for prompt, gids in zip(prompts, gen_ids):
                    text = tokenizer.decode(gids, skip_special_tokens=True)
                    total_chars += len(text)
                    results.append({'prompt': prompt, 'generated_text': text})

                # 收集hook状态（已在CPU上）用于传回主进程
                hook_states_for_result = [
                    _serialize_hook_state_for_queue(state)
                    for state in hook_state['collected_states']
                ]
                hook_state['collected_states'].clear()

                result_queue.put({
                    'status': 'ok',
                    'outputs': results,
                    'inference_time': t1 - t0,
                    'ep_comm_delay': ep_comm,
                    'generated_chars': total_chars,
                    'hook_states': hook_states_for_result,
                })

            elif cmd['type'] == 'reset_comm_stats':
                for w in ep_wrappers:
                    w.reset_comm_delay()
                result_queue.put({'status': 'ok'})

            elif cmd['type'] == 'shutdown':
                logger.info(f"[Worker {rank}] Shutting down")
                if model is not None:
                    del model
                    model = None
                torch.cuda.empty_cache()
                result_queue.put({'status': 'ok'})
                break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ========== 公共代理适配器 ==========

class HFAccelerateAdapter(InferenceFrameworkInterface):
    """
    单进程接口的多进程 EP+DP 推理适配器

    外部只需单线程调用，内部自动spawn EP+DP worker进程：
    - 每个GPU一个worker进程（一个rank）
    - 数据并行(DP)：输入数据自动拆分，每个worker处理不同子集
    - 专家并行(EP)：MoE层使用all-to-all交换token实现真正的专家并行
    - Worker间通过torch.distributed NCCL通信

    使用方式:
        adapter = create_hf_accelerate_adapter("path/to/model", ep_size=2)
        outputs = adapter.run_inference(["Hello!", "Explain AI."])
        print(adapter.get_generated_texts())
        adapter.cleanup()
    """

    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.tokenizer = None
        self._workers = []
        self._cmd_queues = []
        self._result_queues = []
        self._ep_size = 1
        self._hidden_size = 0
        self._num_moe_layers = 0
        self._last_output = None
        self._last_outputs = []
        self._last_comm_delay = 0.0
        self._last_ep_comm_delay = 0.0
        self._last_generated_chars = 0
        self._is_loaded = False
        self._latency_metric = 'end_to_end'

        # 接口兼容
        self._dp_rank = 0
        self._dp_size = 1
        self._world_size = 1
        self._local_rank = 0
        self._moe_layers = []
        self._ep_wrappers = []
        self._first_moe_gate = None
        self._first_moe_block = None
        self._last_hook_states = []
        self._moe_layer_names = []
        self._cuda_timer = CUDATimer()

        self._generation_config = {
            'max_new_tokens': 128,
            'do_sample': False,
            'temperature': 1.0,
            'top_p': 1.0,
        }

    def load_model(self, model_path: str, **kwargs) -> nn.Module:
        """
        加载模型并启动EP worker进程

        Args:
            model_path: 模型路径
            **kwargs:
                - expert_parallel_size (int): EP GPU数，默认2
                - trust_remote_code (bool): 默认True
                - torch_dtype: 模型精度，默认auto
                - max_new_tokens, temperature, top_p, do_sample: 生成参数
        """
        ep_size = kwargs.get('expert_parallel_size', 2)

        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError("No CUDA GPUs available")
        if ep_size > num_gpus:
            self.logger.warning(f"ep_size={ep_size} > GPUs={num_gpus}, using {num_gpus}")
            ep_size = num_gpus

        self._ep_size = ep_size
        self._world_size = ep_size
        self.logger.info(f"Starting EP with {ep_size} GPUs...")

        # 主进程加载tokenizer（用于tokenize/decode方法）
        from transformers import AutoTokenizer
        trust_remote = kwargs.get('trust_remote_code', True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 找到可用端口
        master_port = _find_free_port()
        self.logger.info(f"NCCL master port: {master_port}")

        # 创建spawn上下文（CUDA安全）
        ctx = mp.get_context('spawn')
        self._cmd_queues = [ctx.Queue() for _ in range(ep_size)]
        self._result_queues = [ctx.Queue() for _ in range(ep_size)]

        # 启动worker进程
        for rank in range(ep_size):
            p = ctx.Process(
                target=_ep_worker_fn,
                args=(rank, ep_size, master_port,
                      self._cmd_queues[rank], self._result_queues[rank]),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
            self.logger.info(f"Worker {rank} started (pid={p.pid})")

        # 过滤传给worker的参数（只传模型加载参数）
        gen_params = {'max_new_tokens', 'temperature', 'top_p', 'do_sample',
                      'expert_parallel_size', 'data_parallel_size'}
        load_kwargs = {k: v for k, v in kwargs.items() if k not in gen_params}

        # 发送load_model命令
        for rank in range(ep_size):
            self._cmd_queues[rank].put({
                'type': 'load_model',
                'model_path': model_path,
                'kwargs': load_kwargs,
            })

        # 等待所有worker加载完成
        self.logger.info("Waiting for all workers to load model...")
        for rank in range(ep_size):
            result = self._result_queues[rank].get(timeout=600)
            if result.get('status') != 'ok':
                raise RuntimeError(f"Worker {rank} failed: {result}")
            if rank == 0:
                self._hidden_size = result['hidden_size']
                self._num_moe_layers = result['num_moe_layers']
                self._latency_metric = result.get('latency_metric', 'end_to_end')

                # 创建 _WorkerModuleProxy，支持在worker实际模块上注册hook
                gate_info = result.get('gate_info')
                block_info = result.get('block_info')

                gate_sd = gate_info['state_dict'] if gate_info else None
                gate_in = gate_info['in_features'] if gate_info else None
                gate_out = gate_info['out_features'] if gate_info else None
                gate_bias = gate_info['has_bias'] if gate_info else False

                # first_moe_gate proxy -> 对应worker中 ep_wrappers[0].gate
                self._first_moe_gate = _WorkerModuleProxy(
                    cmd_queues=self._cmd_queues,
                    result_queues=self._result_queues,
                    target_name='first_moe_gate',
                    adapter=self,
                    gate_state_dict=gate_sd,
                    gate_in_features=gate_in,
                    gate_out_features=gate_out,
                    gate_has_bias=gate_bias,
                )
                self.logger.info(
                    f"Created first_moe_gate proxy"
                    f"{f': Linear({gate_in}, {gate_out})' if gate_in else ''}"
                )

                # first_moe_block proxy -> 对应worker中 ep_wrappers[0]
                num_exp = block_info['num_experts'] if block_info else None
                top_k = block_info['top_k'] if block_info else None
                norm_tp = block_info['norm_topk_prob'] if block_info else None

                self._first_moe_block = _WorkerModuleProxy(
                    cmd_queues=self._cmd_queues,
                    result_queues=self._result_queues,
                    target_name='first_moe_block',
                    adapter=self,
                    gate_state_dict=gate_sd,
                    gate_in_features=gate_in,
                    gate_out_features=gate_out,
                    gate_has_bias=gate_bias,
                    num_experts=num_exp,
                    top_k=top_k,
                    norm_topk_prob=norm_tp,
                )
                self.logger.info(
                    f"Created first_moe_block proxy"
                    f"{f': num_experts={num_exp}, top_k={top_k}' if num_exp else ''}"
                )

        self._is_loaded = True

        # 更新生成配置
        self._generation_config.update({
            'max_new_tokens': kwargs.get('max_new_tokens', 64),
            'temperature': kwargs.get('temperature', 1.0),
            'top_p': kwargs.get('top_p', 1.0),
            'do_sample': kwargs.get('do_sample', False),
        })

        self.logger.info(
            f"EP model ready: {ep_size} GPUs, hidden_size={self._hidden_size}, "
            f"moe_layers={self._num_moe_layers}, latency_metric={self._latency_metric}"
        )
        return None

    def run_inference(self, inputs: Any, **kwargs) -> Any:
        if not self._is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None

        if isinstance(inputs, str):
            prompts = [inputs]
        elif isinstance(inputs, list):
            prompts = inputs
        elif isinstance(inputs, dict) and 'prompts' in inputs:
            prompts = inputs['prompts']
        else:
            self.logger.error(f"Unsupported input format: {type(inputs)}")
            return None

        if not prompts:
            self._last_outputs, self._last_output = [], None
            return []

        gen_config = self._generation_config.copy()
        gen_config.update(kwargs)

        # ========== 数据并行(DP): 将prompts均匀拆分到各EP worker ==========
        # 每个worker处理不同的数据子集，确保all-to-all交换的是不同token
        n = len(prompts)
        chunks = []
        dummy_ranks = set()
        offset = 0
        for rank in range(self._ep_size):
            chunk_size = n // self._ep_size + (1 if rank < n % self._ep_size else 0)
            if chunk_size > 0:
                chunks.append(prompts[offset:offset + chunk_size])
                offset += chunk_size
            else:
                # 该worker没有真实数据，发送dummy prompt保证其参与all-to-all通信
                chunks.append([prompts[0]])
                dummy_ranks.add(rank)

        # self.logger.info(
        #     f"DP split: {n} prompts -> {[len(c) for c in chunks]} "
        #     f"(dummy_ranks={dummy_ranks if dummy_ranks else 'none'})"
        # )

        # 发送推理命令（每个worker获得不同的数据子集）
        for rank in range(self._ep_size):
            self._cmd_queues[rank].put({
                'type': 'run_inference',
                'inputs': chunks[rank],
                'gen_config': gen_config.copy(),
            })

        # 从所有worker收集结果并按顺序合并
        all_outputs = []
        self._last_hook_states = []
        max_time = 0.0
        max_ep_comm = 0.0
        total_chars = 0
        for rank in range(self._ep_size):
            result = self._result_queues[rank].get(timeout=600)
            if result.get('status') != 'ok':
                self.logger.error(f"Worker {rank} inference failed: {result}")
                return None
            if rank not in dummy_ranks:
                all_outputs.extend(result['outputs'])
                total_chars += result.get('generated_chars', 0)
            max_time = max(max_time, result.get('inference_time', 0.0))
            max_ep_comm = max(max_ep_comm, result.get('ep_comm_delay', 0.0))
            # 收集worker中hook收集的状态
            hook_states = result.get('hook_states', [])
            if rank not in dummy_ranks:
                self._last_hook_states.extend(hook_states)

        self._last_outputs = all_outputs
        self._last_output = self._last_outputs[0] if self._last_outputs else None
        self._last_comm_delay = max_time
        self._last_ep_comm_delay = max_ep_comm
        self._last_generated_chars = total_chars

        self.logger.info(
            f"Inference done (DP+EP): {n} prompts across {self._ep_size} GPUs, "
            f"total={self._last_comm_delay*1000:.1f}ms, "
            f"ep_comm={self._last_ep_comm_delay*1000:.1f}ms"
        )
        return self._last_outputs

    def reset_comm_stats(self):
        self._last_comm_delay = 0.0
        self._last_ep_comm_delay = 0.0
        self._last_generated_chars = 0
        if self._is_loaded:
            for rank in range(self._ep_size):
                self._cmd_queues[rank].put({'type': 'reset_comm_stats'})
            for rank in range(self._ep_size):
                self._result_queues[rank].get(timeout=30)

    def get_model_output(self) -> Any:
        return self._last_output

    def get_batch_outputs(self) -> List[Any]:
        return self._last_outputs

    def get_generated_texts(self) -> List[str]:
        if not self._last_outputs:
            return []
        return [o.get('generated_text', '') if isinstance(o, dict) else ''
                for o in self._last_outputs]

    def get_comm_delay(self) -> float:
        if self._latency_metric == 'ep_comm':
            return self._last_ep_comm_delay
        return self._last_comm_delay

    def get_ep_comm_delay(self) -> float:
        """获取EP all-to-all通信时延"""
        return self._last_ep_comm_delay

    def get_comm_delay_per_token(self) -> float:
        if self._last_generated_chars > 0:
            if self._latency_metric == 'ep_comm':
                return self._last_ep_comm_delay / self._last_generated_chars
            return self._last_comm_delay / self._last_generated_chars
        if self._latency_metric == 'ep_comm':
            return self._last_ep_comm_delay
        return self._last_comm_delay

    def get_comm_delay_per_layer(self) -> Dict[int, float]:
        return {}

    def get_moe_layers(self) -> List[nn.Module]:
        """注意：MoE层在worker进程中，主进程无法直接访问"""
        return self._moe_layers

    def get_first_moe_block(self) -> Optional[nn.Module]:
        return self._first_moe_block

    def get_first_moe_gate(self) -> Optional[nn.Module]:
        return self._first_moe_gate

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_dp_rank(self) -> int:
        return 0

    def get_dp_size(self) -> int:
        return self._ep_size

    def get_ep_size(self) -> int:
        return self._ep_size

    def get_world_size(self) -> int:
        return self._ep_size

    def get_local_rank(self) -> int:
        return 0

    def tokenize(self, text):
        if self.tokenizer is None:
            return None
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        return [self.tokenizer.encode(t) for t in text]

    def decode(self, token_ids):
        if self.tokenizer is None:
            return ""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_generation_config(self, **kwargs):
        self._generation_config.update(kwargs)
        self.logger.info(f"Updated generation config: {kwargs}")

    def get_underlying_model(self) -> Optional[nn.Module]:
        """模型在worker进程中，主进程无法直接访问"""
        return None

    def sync_perturbation_weights(self, generator: nn.Module):
        """
        将主进程中更新后的扰动生成器权重同步到所有worker进程。
        在PPO更新actor后调用此方法。
        """
        if not self._is_loaded:
            return
        cpu_sd = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        for rank in range(self._ep_size):
            self._cmd_queues[rank].put({
                'type': 'sync_perturbation_weights',
                'state_dict': cpu_sd,
            })
        for rank in range(self._ep_size):
            self._result_queues[rank].get(timeout=30)

    def cleanup(self):
        """关闭所有worker进程"""
        if not self._workers:
            return
        self.logger.info("Shutting down EP workers...")
        for rank in range(self._ep_size):
            try:
                self._cmd_queues[rank].put({'type': 'shutdown'})
            except Exception:
                pass
        for rank in range(self._ep_size):
            try:
                self._result_queues[rank].get(timeout=30)
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers = []
        self._cmd_queues = []
        self._result_queues = []
        self._is_loaded = False
        self._latency_metric = 'end_to_end'
        self.logger.info("All EP workers shut down")


# ========== 工厂函数 ==========

def create_hf_accelerate_adapter(
    model_path: str = None,
    ep_size: int = 2,
    **kwargs
) -> HFAccelerateAdapter:
    """
    创建EP推理适配器

    外部只需单线程调用，内部自动spawn多进程EP：
        adapter = create_hf_accelerate_adapter("path/to/model", ep_size=2)
        outputs = adapter.run_inference("Hello")

    Args:
        model_path: 模型路径。提供则自动加载模型
        ep_size: 专家并行GPU数，默认2
        **kwargs: 其他参数传给load_model
    """
    adapter = HFAccelerateAdapter()
    if model_path is not None:
        adapter.load_model(model_path, expert_parallel_size=ep_size, **kwargs)
    return adapter


def create_hf_adapter_with_model(
    model_path: str,
    dp_size: int = 1,
    ep_size: int = 2,
    **kwargs
) -> HFAccelerateAdapter:
    """
    创建并加载模型的适配器（兼容旧接口）

    注意：DP与EP使用相同的GPU集合，dp_size自动等于ep_size
    """
    return create_hf_accelerate_adapter(model_path, ep_size=ep_size, **kwargs)


# ========== 示例 ==========

if __name__ == "__main__":
    """
    使用示例（无需torchrun，直接python运行）:
        python hf_accelerate_adapter.py --model_path path/to/model --ep_size 2
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../../models/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--ep_size', type=int, default=2)
    args = parser.parse_args()

    print(f"Creating EP adapter: ep_size={args.ep_size}")
    adapter = create_hf_accelerate_adapter(
        model_path=args.model_path,
        ep_size=args.ep_size,
    )

    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a short poem.",
    ]

    print(f"\nRunning inference on {len(prompts)} prompts...")
    outputs = adapter.run_inference(prompts)

    print(f"\n=== Results ===")
    for text in adapter.get_generated_texts():
        print(f"Generated: {text[:100]}...")

    print(f"\nInference time: {adapter.get_comm_delay()*1000:.1f}ms")
    print(f"EP comm delay: {adapter.get_ep_comm_delay()*1000:.1f}ms")
    print(f"Hidden size: {adapter.get_hidden_size()}")
    print(f"EP size: {adapter.get_ep_size()}")

    adapter.cleanup()
    print("Done!")
