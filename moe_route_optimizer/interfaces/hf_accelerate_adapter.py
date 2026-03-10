"""
Hugging Face Transformers + Accelerate 分布式推理适配器
支持数据并行(DP)和专家并行(EP)的MoE模型推理

使用方式：
    torchrun --nproc_per_node=2 your_script.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple, Union

# 确保导入路径
workspace_root = "/mnt/data/lwy/vLLM-wrok/moe_route_optimizer"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from interfaces.framework_interface import InferenceFrameworkInterface
from config import get_logger

# 设置CUDA内存分配策略（使用新的环境变量名）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CUDATimer:
    """
    精确的CUDA计时器
    
    使用CUDA events进行GPU计时，确保同步后获得准确时间
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device
        self._start_event = None
        self._end_event = None
        self._elapsed_ms = 0.0
    
    def start(self):
        """开始计时"""
        if not torch.cuda.is_available():
            return
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
    
    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        if not torch.cuda.is_available() or self._start_event is None:
            return 0.0
        
        self._end_event.record()
        # 同步等待GPU完成
        torch.cuda.synchronize()
        # 获取耗时（毫秒）
        self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        return self._elapsed_ms / 1000.0  # 转换为秒
    
    def elapsed_seconds(self) -> float:
        """获取耗时（秒）"""
        return self._elapsed_ms / 1000.0


class ExpertParallelWrapper(nn.Module):
    """
    真正的专家并行包装器
    
    实现物理专家并行：
    - 每个GPU只持有一半专家的参数（节省显存）
    - GPU 0: 专家 0 ~ N/2-1
    - GPU 1: 专家 N/2 ~ N-1
    - 当token路由到另一个GPU的专家时，通过all-to-all通信交换数据
    
    两GPU场景下的forward流程：
    1. 所有GPU计算路由决策（gate）
    2. All-to-all dispatch: 将token发送到目标GPU
    3. 本地专家计算
    4. All-to-all combine: 将结果发回原GPU
    """
    
    def __init__(
        self,
        original_moe_layer: nn.Module,
        world_size: int = 2,
        rank: int = 0,
        num_experts: int = 8,
        layer_idx: int = 0,
        logger = None,
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.experts_per_gpu = num_experts // world_size
        self.layer_idx = layer_idx
        self.logger = logger
        
        # 计算本GPU负责的专家范围
        self.expert_start = rank * self.experts_per_gpu
        self.expert_end = self.expert_start + self.experts_per_gpu
        
        # 获取gate引用（所有GPU都需要完整的gate）
        self.gate = getattr(original_moe_layer, 'gate', None)
        if self.gate is None:
            self.gate = getattr(original_moe_layer, 'router', None)
        
        # 获取原始experts
        original_experts = getattr(original_moe_layer, 'experts', None)
        
        # 只保留本GPU负责的专家（实现真正的参数分片）
        if original_experts is not None and isinstance(original_experts, nn.ModuleList):
            # 创建只包含本GPU专家的ModuleList
            self.local_experts = nn.ModuleList([
                original_experts[i] for i in range(self.expert_start, self.expert_end)
            ])
            # self._log_info(
            #     f"[EP Layer {layer_idx}] GPU {rank}: Kept experts {self.expert_start}-{self.expert_end-1}, "
            #     f"removed {num_experts - self.experts_per_gpu} experts to save memory"
            # )
        else:
            self.local_experts = None
            self._log_info(f"[EP Layer {layer_idx}] GPU {rank}: experts not found or not ModuleList")
        
        # 保存其他必要组件（如shared_expert等）
        self.shared_expert = getattr(original_moe_layer, 'shared_expert', None)
        self.shared_expert_gate = getattr(original_moe_layer, 'shared_expert_gate', None)
        
        # 通信时延统计
        self._total_comm_delay = 0.0
        self._forward_count = 0
        
        # 获取top_k配置
        self.top_k = getattr(original_moe_layer, 'top_k', 2)
        if hasattr(original_moe_layer, 'num_experts_per_tok'):
            self.top_k = original_moe_layer.num_experts_per_tok
        
        self._log_info(
            f"[EP Layer {layer_idx}] Initialized: rank={rank}, world_size={world_size}, "
            f"local_experts={self.expert_start}-{self.expert_end-1} (of {num_experts}), "
            f"top_k={self.top_k}"
        )
    
    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def _log_debug(self, msg: str):
        if self.logger:
            self.logger.debug(msg)
    
    def reset_comm_delay(self):
        self._total_comm_delay = 0.0
        self._forward_count = 0
    
    def get_comm_delay(self) -> float:
        return self._total_comm_delay
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        真正的专家并行前向传播
        
        流程：
        1. Gate计算路由决策
        2. All-to-all dispatch: 按专家分组，交换token到目标GPU
        3. 本地专家计算
        4. All-to-all combine: 将结果发回原GPU
        """
        self._forward_count += 1
        
        if not dist.is_initialized() or self.world_size <= 1:
            raise RuntimeError("Expert parallel requires distributed environment with world_size > 1")
        
        if self.local_experts is None or self.gate is None:
            raise RuntimeError(f"[EP Layer {self.layer_idx}] local_experts or gate is None")
        
        original_shape = hidden_states.shape
        batch_size, seq_len, hidden_size = original_shape
        num_tokens = batch_size * seq_len
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        hidden_flat = hidden_states.view(num_tokens, hidden_size)
        
        router_logits = self.gate(hidden_flat)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        expert_to_gpu = topk_indices // self.experts_per_gpu
        
        send_counts = []
        for gpu_id in range(self.world_size):
            count = (expert_to_gpu == gpu_id).sum().item()
            send_counts.append(count)
        
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
        
        send_hidden_list = []
        send_weights_list = []
        send_expert_ids_list = []
        send_token_ids_list = []
        
        for gpu_id in range(self.world_size):
            mask = (expert_to_gpu == gpu_id)
            
            if mask.any():
                token_ids, k_ids = torch.where(mask)
                send_hidden = hidden_flat[token_ids]
                send_weights = topk_weights[token_ids, k_ids]
                global_expert_ids = topk_indices[token_ids, k_ids]
                local_expert_ids = global_expert_ids - gpu_id * self.experts_per_gpu
                
                send_hidden_list.append(send_hidden)
                send_weights_list.append(send_weights)
                send_expert_ids_list.append(local_expert_ids)
                send_token_ids_list.append(token_ids)
            else:
                send_hidden_list.append(torch.empty(0, hidden_size, device=device, dtype=dtype))
                send_weights_list.append(torch.empty(0, device=device, dtype=dtype))
                send_expert_ids_list.append(torch.empty(0, device=device, dtype=torch.long))
                send_token_ids_list.append(torch.empty(0, device=device, dtype=torch.long))
        
        send_hidden_all = torch.cat(send_hidden_list, dim=0)
        send_weights_all = torch.cat(send_weights_list, dim=0)
        send_expert_ids_all = torch.cat(send_expert_ids_list, dim=0)
        send_token_ids_all = torch.cat(send_token_ids_list, dim=0)
        
        recv_hidden_all = torch.zeros(total_recv, hidden_size, device=device, dtype=dtype)
        recv_weights_all = torch.zeros(total_recv, device=device, dtype=dtype)
        recv_expert_ids_all = torch.zeros(total_recv, device=device, dtype=torch.long)
        recv_token_ids_all = torch.zeros(total_recv, device=device, dtype=torch.long)
        
        if total_send > 0 or total_recv > 0:
            send_hidden_split = [t.clone() for t in send_hidden_list]
            recv_hidden_split = [torch.zeros(c, hidden_size, device=device, dtype=dtype) for c in recv_counts]
            dist.all_to_all(recv_hidden_split, send_hidden_split)
            recv_hidden_all = torch.cat(recv_hidden_split, dim=0) if total_recv > 0 else recv_hidden_all
            
            send_weights_split = [t.clone() for t in send_weights_list]
            recv_weights_split = [torch.zeros(c, device=device, dtype=dtype) for c in recv_counts]
            dist.all_to_all(recv_weights_split, send_weights_split)
            recv_weights_all = torch.cat(recv_weights_split, dim=0) if total_recv > 0 else recv_weights_all
            
            send_expert_split = [t.clone() for t in send_expert_ids_list]
            recv_expert_split = [torch.zeros(c, device=device, dtype=torch.long) for c in recv_counts]
            dist.all_to_all(recv_expert_split, send_expert_split)
            recv_expert_ids_all = torch.cat(recv_expert_split, dim=0) if total_recv > 0 else recv_expert_ids_all
            
            send_token_split = [t.clone() for t in send_token_ids_list]
            recv_token_split = [torch.zeros(c, device=device, dtype=torch.long) for c in recv_counts]
            dist.all_to_all(recv_token_split, send_token_split)
            recv_token_ids_all = torch.cat(recv_token_split, dim=0) if total_recv > 0 else recv_token_ids_all
        
        comm_mid.record()
        
        recv_output_all = torch.zeros_like(recv_hidden_all)
        
        if total_recv > 0:
            for local_expert_idx in range(self.experts_per_gpu):
                expert_mask = (recv_expert_ids_all == local_expert_idx)
                
                if expert_mask.any():
                    expert_input = recv_hidden_all[expert_mask]
                    expert_output = self.local_experts[local_expert_idx](expert_input)
                    recv_output_all[expert_mask] = expert_output
        
        recv_output_weighted = recv_output_all * recv_weights_all.unsqueeze(-1)
        
        send_output_all = torch.zeros(total_send, hidden_size, device=device, dtype=dtype)
        send_back_token_ids_all = torch.zeros(total_send, device=device, dtype=torch.long)
        
        if total_send > 0 or total_recv > 0:
            recv_output_split = []
            offset = 0
            for c in recv_counts:
                if c > 0:
                    recv_output_split.append(recv_output_weighted[offset:offset+c].clone())
                    offset += c
                else:
                    recv_output_split.append(torch.empty(0, hidden_size, device=device, dtype=dtype))
            
            send_output_split = [torch.zeros(c, hidden_size, device=device, dtype=dtype) for c in send_counts]
            dist.all_to_all(send_output_split, recv_output_split)
            send_output_all = torch.cat(send_output_split, dim=0) if total_send > 0 else send_output_all
            
            recv_token_split = []
            offset = 0
            for c in recv_counts:
                if c > 0:
                    recv_token_split.append(recv_token_ids_all[offset:offset+c].clone())
                    offset += c
                else:
                    recv_token_split.append(torch.empty(0, device=device, dtype=torch.long))
            
            send_token_back_split = [torch.zeros(c, device=device, dtype=torch.long) for c in send_counts]
            dist.all_to_all(send_token_back_split, recv_token_split)
            send_back_token_ids_all = torch.cat(send_token_back_split, dim=0) if total_send > 0 else send_back_token_ids_all
        
        comm_end.record()
        torch.cuda.synchronize()
        
        dispatch_time_ms = comm_start.elapsed_time(comm_mid)
        combine_time_ms = comm_mid.elapsed_time(comm_end)
        total_comm_ms = dispatch_time_ms + combine_time_ms
        self._total_comm_delay += total_comm_ms / 1000.0
        
        output_flat = torch.zeros_like(hidden_flat)
        
        if total_send > 0:
            output_flat.scatter_add_(
                0,
                send_back_token_ids_all.unsqueeze(-1).expand(-1, hidden_size),
                send_output_all
            )
        
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_flat)
            if self.shared_expert_gate is not None:
                shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_flat))
                output_flat = output_flat + shared_gate * shared_output
            else:
                output_flat = output_flat + shared_output
        
        if self._forward_count <= 3 or self._forward_count % 50 == 0:
            self._log_info(
                f"[EP Layer {self.layer_idx}] Forward #{self._forward_count}: "
                f"tokens={num_tokens}, send={total_send}, recv={total_recv}, "
                f"comm={{dispatch={dispatch_time_ms:.2f}ms, combine={combine_time_ms:.2f}ms}}"
            )
        
        return output_flat.view(batch_size, seq_len, hidden_size)


class HFAccelerateAdapter(InferenceFrameworkInterface):
    """
    Hugging Face Transformers + Accelerate 适配器
    
    支持数据并行(DP)和专家并行(EP)的MoE模型分布式推理
    
    使用方式:
        # 使用torchrun启动
        torchrun --nproc_per_node=2 your_script.py
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.tokenizer = None
        self._last_output = None
        self._last_outputs = []
        self._last_comm_delay = 0.0  # 总推理时延（秒）
        self._last_generated_chars = 0  # 生成的字符数（用于计算per-token时延）
        self._hidden_size = 0
        self._moe_layers: List[nn.Module] = []
        self._ep_wrappers: List[ExpertParallelWrapper] = []
        self._first_moe_gate = None
        self._first_moe_block = None
        
        # 分布式相关
        self._dp_rank = 0
        self._dp_size = 1
        self._ep_size = 1
        self._world_size = 1
        self._local_rank = 0
        
        # MoE层名称映射（用于替换）
        self._moe_layer_names: List[str] = []
        
        # CUDA计时器（用于精确测量通信时延）
        self._cuda_timer = CUDATimer()
        
        # 生成配置
        self._generation_config = {
            'max_new_tokens': 128,
            'do_sample': False,
            'temperature': 1.0,
            'top_p': 1.0,
        }
        
    def _init_distributed(self, dp_size: int = 2, ep_size: int = 2):
        """
        初始化分布式环境
        
        Args:
            dp_size: 数据并行大小
            ep_size: 专家并行大小
        """
        if dist.is_initialized():
            self._world_size = dist.get_world_size()
            self._local_rank = dist.get_rank()
        elif 'RANK' in os.environ:
            # torchrun启动
            self._local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self._world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
                self._world_size = dist.get_world_size()
                self._local_rank = dist.get_rank()
        else:
            self._world_size = 1
            self._local_rank = 0
        
        # 在两卡情况下，DP和EP共享同一个world
        # DP_size=2表示数据在两卡间分割
        # EP_size=2表示专家在两卡间分割
        self._dp_size = min(dp_size, self._world_size)
        self._ep_size = min(ep_size, self._world_size)
        self._dp_rank = self._local_rank % self._dp_size
        
        self.logger.info(
            f"Distributed initialized: world_size={self._world_size}, "
            f"local_rank={self._local_rank}, dp_size={self._dp_size}, "
            f"dp_rank={self._dp_rank}, ep_size={self._ep_size}"
        )
    
    def load_model(self, model_path: str, **kwargs) -> nn.Module:
        """
        加载MoE模型并应用专家并行
        
        Args:
            model_path: 模型路径（HuggingFace模型名或本地路径）
            **kwargs: 支持的参数包括:
                - data_parallel_size (int): 数据并行大小，默认2
                - expert_parallel_size (int): 专家并行大小，默认2
                    当ep_size > 1时，自动启用物理专家并行：
                    - 每个GPU只持有一半专家参数（节省显存）
                    - Token通过all-to-all通信路由到目标GPU的专家
                - max_new_tokens (int): 最大生成token数，默认128
                - trust_remote_code (bool): 是否信任远程代码，默认True
                - torch_dtype: 模型精度，默认auto
                - device_map: 设备映射，默认None（手动放置）
        
        Returns:
            加载的模型
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        self.logger.info(f"Loading HF model from: {model_path}")
        
        # 解析参数
        dp_size = kwargs.get('data_parallel_size', 2)
        ep_size = kwargs.get('expert_parallel_size', 2)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        torch_dtype = kwargs.get('torch_dtype', 'auto')
        
        # 初始化分布式
        self._init_distributed(dp_size, ep_size)
        
        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            device = torch.device(f'cuda:{self._local_rank}')
        else:
            device = torch.device('cpu')
        self.logger.info(f"Using device: {device}")
        
        # 加载配置获取hidden_size
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        self._hidden_size = getattr(config, 'hidden_size', 2048)
        
        # 获取专家数量
        num_experts = getattr(config, 'num_experts', None)
        if num_experts is None:
            num_experts = getattr(config, 'num_local_experts', None)
        if num_experts is None:
            num_experts = getattr(config, 'n_routed_experts', 8)
        
        self.logger.info(f"Model config: hidden_size={self._hidden_size}, num_experts={num_experts}")
        
        # 加载tokenizer（设置left padding用于生成任务）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            padding_side='left',  # decoder-only模型生成时需要left padding
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        # 对于专家并行，我们需要先加载完整模型，然后手动分配专家
        self.logger.info("Loading model...")
        
        # 确定数据类型
        if torch_dtype == 'auto':
            model_dtype = torch.float16
        elif isinstance(torch_dtype, str):
            dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32,
            }
            model_dtype = dtype_map.get(torch_dtype, torch.float16)
        else:
            model_dtype = torch_dtype
        
        model_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'trust_remote_code': trust_remote_code,
            'torch_dtype': model_dtype,
            'low_cpu_mem_usage': True,
        }
        
        # 使用device_map直接加载到指定GPU，避免CPU内存爆炸
        # 在多进程环境下，每个进程直接加载到自己的GPU，不经过CPU
        model_kwargs['device_map'] = {'': device}  # 将整个模型加载到当前GPU
        
        # 如果内存仍然紧张，可以设置max_memory限制
        if torch.cuda.is_available():
            # 获取GPU显存并预留一些空间
            total_mem = torch.cuda.get_device_properties(self._local_rank).total_memory
            max_mem = int(total_mem * 0.85)  # 使用85%的显存
            model_kwargs['max_memory'] = {self._local_rank: max_mem}
        
        self.logger.info(f"Loading model with device_map to {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
        
        # 查找MoE层
        self._find_moe_layers()
        
        # 应用专家并行（真正的物理专家并行：每个GPU只持有一半专家）
        if self._ep_size > 1 and len(self._moe_layers) > 0:
            self._apply_expert_parallel(num_experts)
        
        # 更新生成配置
        self._generation_config.update({
            'max_new_tokens': kwargs.get('max_new_tokens', 64),
            'temperature': kwargs.get('temperature', 1.0),
            'top_p': kwargs.get('top_p', 1.0),
            'do_sample': kwargs.get('do_sample', False),
        })
        
        return self.model
    
    def _apply_expert_parallel(self, num_experts: int):
        """
        应用真正的专家并行
        
        将每个MoE层的专家分片到不同GPU上：
        - GPU 0: 专家 0 ~ num_experts/2 - 1
        - GPU 1: 专家 num_experts/2 ~ num_experts - 1
        
        每个GPU只持有自己负责的专家参数，节省显存。
        当token路由到另一个GPU的专家时，通过all-to-all通信交换数据。
        
        Args:
            num_experts: 专家总数
        """
        self.logger.info(f"Applying PHYSICAL expert parallel with {self._ep_size} GPUs, {num_experts} experts")
        
        experts_per_gpu = num_experts // self._ep_size
        self.logger.info(f"Each GPU holds {experts_per_gpu} experts (memory saving enabled)")
        
        # 为每个MoE层创建wrapper并替换原始层
        for idx, (moe_layer, layer_name) in enumerate(zip(self._moe_layers, self._moe_layer_names)):
            wrapper = ExpertParallelWrapper(
                original_moe_layer=moe_layer,
                world_size=self._ep_size,
                rank=self._local_rank,
                num_experts=num_experts,
                layer_idx=idx,
                logger=self.logger,
            )
            self._ep_wrappers.append(wrapper)
            
            # 用wrapper替换模型中的原始MoE层
            self._replace_module_by_name(layer_name, wrapper)
        
        # 强制垃圾回收，释放被删除专家的显存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # self.logger.info(f"Expert parallel applied: {len(self._ep_wrappers)} MoE layers wrapped")
    
    def _replace_module_by_name(self, module_name: str, new_module: nn.Module):
        """
        根据名称路径替换模型中的模块
        
        Args:
            module_name: 模块名称路径，如 'model.layers.0.mlp.moe'
            new_module: 新的模块
        """
        if self.model is None:
            self.logger.error("Model not loaded, cannot replace module")
            return
        
        parts = module_name.split('.')
        
        # 找到父模块
        parent = self.model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # 获取最后一个属性名
        last_name = parts[-1]
        
        # 替换模块
        if last_name.isdigit():
            parent[int(last_name)] = new_module
        else:
            setattr(parent, last_name, new_module)
        
        self.logger.debug(f"Replaced module at '{module_name}'")
    
    def _find_moe_layers(self):
        """
        查找模型中的MoE层和MoE块
        
        支持多种MoE架构：
        - Qwen MoE (SparseMoeBlock)
        - Mixtral (MixtralSparseMoeBlock)
        - DeepSeek MoE
        """
        if self.model is None:
            self.logger.warning("Model not loaded, cannot find MoE layers")
            return
        
        self._moe_layers = []
        self._moe_layer_names = []  # 同时记录层的名称路径
        self._first_moe_gate = None
        self._first_moe_block = None
        
        # MoE块的类名关键词
        moe_block_keywords = ['sparsemoe', 'moeblock', 'moelayer', 'moe']
        # 排除的关键词（内部组件）
        exclude_keywords = ['expert', 'fusedmoe', 'sharedfusedmoe', 'gate', 'router']
        
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__.lower()
            
            # 检查是否是完整的MoE块
            is_moe_block = any(kw in class_name for kw in moe_block_keywords)
            is_excluded = any(kw in class_name for kw in exclude_keywords)
            
            # 完整的MoE块应该同时包含gate和experts
            has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
            has_experts = hasattr(module, 'experts') or hasattr(module, 'expert')
            is_complete_moe_block = has_gate and has_experts
            
            if is_moe_block and not is_excluded and is_complete_moe_block:
                self._moe_layers.append(module)
                self._moe_layer_names.append(name)  # 记录层名称
                self.logger.debug(f"Found MoE block: {name} ({module.__class__.__name__})")
                
                if self._first_moe_block is None:
                    self._first_moe_block = module
                    self.logger.info(f"First MoE block: {name} ({module.__class__.__name__})")
                
                if self._first_moe_gate is None:
                    gate = getattr(module, 'gate', None) or getattr(module, 'router', None)
                    if gate is not None:
                        self._first_moe_gate = gate
                        self.logger.info(f"First MoE gate found at: {name}")
        
        # 如果没有找到标准MoE块，尝试更宽松的搜索
        if not self._moe_layers:
            for name, module in self.model.named_modules():
                class_name = module.__class__.__name__.lower()
                name_lower = name.lower()
                
                is_moe_related = any(kw in class_name for kw in ['moe', 'expert', 'mixture'])
                is_moe_related = is_moe_related or any(kw in name_lower for kw in ['moe', 'experts'])
                
                if is_moe_related:
                    self._moe_layers.append(module)
                    self._moe_layer_names.append(name)  # 记录层名称
                    self.logger.debug(f"Found MoE-related layer: {name}")
                    
                    if self._first_moe_block is None:
                        self._first_moe_block = module
                    
                    if self._first_moe_gate is None:
                        for attr in ['gate', 'router', 'gating_network', 'w_gate']:
                            if hasattr(module, attr):
                                self._first_moe_gate = getattr(module, attr)
                                break
        
        # self.logger.info(f"Found {len(self._moe_layers)} MoE layers: {self._moe_layer_names[:3]}...")
    
    def run_inference(self, inputs: Any, **kwargs) -> Any:
        """
        执行批量推理
        
        Args:
            inputs: 输入数据，支持:
                - str: 单个文本
                - List[str]: 多个文本
                - Dict: 包含'prompts'键的字典
            **kwargs: 生成参数
        
        Returns:
            生成的输出
        """
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None
        
        # 处理输入格式
        if isinstance(inputs, str):
            prompts = [inputs]
        elif isinstance(inputs, list):
            prompts = inputs
        elif isinstance(inputs, dict) and 'prompts' in inputs:
            prompts = inputs['prompts']
        else:
            self.logger.error(f"Unsupported input format: {type(inputs)}")
            return None
        
        # 数据并行：根据dp_rank分配prompts
        if self._dp_size > 1:
            prompts = [
                prompt for idx, prompt in enumerate(prompts)
                if idx % self._dp_size == self._dp_rank
            ]
            self.logger.debug(f"DP rank {self._dp_rank}: processing {len(prompts)} prompts")
        
        if not prompts:
            self._last_outputs = []
            self._last_output = None
            return []
        
        # 合并生成参数
        gen_kwargs = self._generation_config.copy()
        gen_kwargs.update(kwargs)
        
        # Tokenize
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        
        # 重置推理时延统计
        self._last_comm_delay = 0.0
        
        # 推理（统计总推理时延）
        import time
        inference_start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=gen_kwargs.get('max_new_tokens', 128),
                do_sample=gen_kwargs.get('do_sample', False),
                temperature=gen_kwargs.get('temperature', 1.0),
                top_p=gen_kwargs.get('top_p', 1.0),
                pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                eos_token_id=None,
            )
        
        # 统计总推理时延
        inference_end_time = time.perf_counter()
        self._last_comm_delay = inference_end_time - inference_start_time
        
        # 解码输出
        input_lengths = encoded['input_ids'].shape[1]
        generated_ids = outputs[:, input_lengths:]
        
        # 保存输出并统计生成的字符数
        self._last_outputs = []
        total_generated_chars = 0
        for i, (prompt, gen_ids) in enumerate(zip(prompts, generated_ids)):
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            total_generated_chars += len(generated_text)
            self._last_outputs.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'output_ids': gen_ids,
            })
        
        # 记录生成的字符数
        self._last_generated_chars = total_generated_chars
        
        self._last_output = self._last_outputs[0] if self._last_outputs else None
        
        self.logger.info(
            f"Inference completed for {len(prompts)} prompts, "
            f"total_inference_time={self._last_comm_delay*1000:.3f}ms, "
            # f"generated_chars={self._last_generated_chars}, "
            # f"time_per_char={self.get_comm_delay_per_token()*1000:.3f}ms"
        )
        
        return self._last_outputs
    
    def reset_comm_stats(self):
        """重置推理时延统计"""
        self._last_comm_delay = 0.0
        self._last_generated_chars = 0
        for wrapper in self._ep_wrappers:
            wrapper.reset_comm_delay()
    
    def get_model_output(self) -> Any:
        """获取最近一次推理的输出"""
        return self._last_output
    
    def get_batch_outputs(self) -> List[Any]:
        """获取最近一次批量推理的所有输出"""
        return self._last_outputs
    
    def get_generated_texts(self) -> List[str]:
        """
        获取生成的文本列表
        
        Returns:
            生成的文本列表
        """
        if not self._last_outputs:
            return []
        
        texts = []
        for output in self._last_outputs:
            if isinstance(output, dict):
                texts.append(output.get('generated_text', ''))
            else:
                texts.append('')
        return texts
    
    def get_comm_delay(self) -> float:
        """
        获取本次推理的总时延
        
        Returns:
            总推理时延（秒）
        """
        return self._last_comm_delay
    
    def get_comm_delay_per_token(self) -> float:
        """
        获取本次推理的每字符平均时延（使用字符数代替token数）
        
        Returns:
            每字符平均时延（秒），如果没有生成字符则返回总时延
        """
        if self._last_generated_chars > 0:
            return self._last_comm_delay / self._last_generated_chars
        else:
            # 如果没有生成字符，返回总时延（避免除零错误）
            return self._last_comm_delay
    
    def get_comm_delay_per_layer(self) -> Dict[int, float]:
        """获取每层MoE的通信时延（简化版本不支持，返回空字典）"""
        return {}
    
    def get_moe_layers(self) -> List[nn.Module]:
        """获取模型中的MoE层列表"""
        if not self._moe_layers and self.model is not None:
            self._find_moe_layers()
        return self._moe_layers
    
    def get_first_moe_block(self) -> Optional[nn.Module]:
        """
        获取第一个完整的MoE块（包含gate和experts）
        
        Returns:
            完整的MoE块模块，用于注册hook
        """
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        self._find_moe_layers()
        
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        moe_layers = self.get_moe_layers()
        if moe_layers:
            self.logger.warning("Could not find specific MoE block, returning first MoE layer")
            return moe_layers[0]
        
        self.logger.error("No MoE layers/blocks found")
        return None
    
    def get_first_moe_gate(self) -> Optional[nn.Module]:
        """
        获取第一个MoE层的Gate模块
        
        Returns:
            Gate模块
        """
        if self._first_moe_gate is not None:
            return self._first_moe_gate
        
        moe_layers = self.get_moe_layers()
        
        if not moe_layers:
            self.logger.error("No MoE layers found")
            return None
        
        first_moe = moe_layers[0]
        
        for attr_name in ['gate', 'router', 'gating_network', 'w_gate']:
            if hasattr(first_moe, attr_name):
                self._first_moe_gate = getattr(first_moe, attr_name)
                self.logger.info(f"Found gate attribute: {attr_name}")
                return self._first_moe_gate
        
        self.logger.warning("Could not find gate, returning MoE layer itself")
        return first_moe
    
    def get_hidden_size(self) -> int:
        """获取模型的隐藏层维度"""
        return self._hidden_size
    
    def get_dp_rank(self) -> int:
        """获取数据并行rank"""
        return self._dp_rank
    
    def get_dp_size(self) -> int:
        """获取数据并行大小"""
        return self._dp_size
    
    def get_ep_size(self) -> int:
        """获取专家并行大小"""
        return self._ep_size
    
    def get_world_size(self) -> int:
        """获取总GPU数"""
        return self._world_size
    
    def get_local_rank(self) -> int:
        """获取本地rank"""
        return self._local_rank
    
    def tokenize(self, text: Union[str, List[str]]) -> Any:
        """
        对输入文本进行tokenize
        
        Args:
            text: 单个文本或文本列表
        
        Returns:
            tokenize后的结果
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not loaded")
            return None
        
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        else:
            return [self.tokenizer.encode(t) for t in text]
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        将token ids解码为文本
        
        Args:
            token_ids: token id列表或张量
        
        Returns:
            解码后的文本
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not loaded")
            return ""
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def set_generation_config(self, **kwargs):
        """
        设置生成参数
        
        Args:
            max_new_tokens (int): 最大生成token数
            temperature (float): 温度参数
            top_p (float): nucleus sampling参数
            do_sample (bool): 是否采样
        """
        self._generation_config.update(kwargs)
        self.logger.info(f"Updated generation config: {kwargs}")
    
    def get_underlying_model(self) -> Optional[nn.Module]:
        """
        获取底层PyTorch模型
        
        Returns:
            底层模型
        """
        return self.model
    
    def cleanup(self):
        """清理资源"""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cleanup completed")


def create_hf_accelerate_adapter() -> HFAccelerateAdapter:
    """创建HF Accelerate适配器实例"""
    return HFAccelerateAdapter()


def create_hf_adapter_with_model(
    model_path: str,
    dp_size: int = 2,
    ep_size: int = 2,
    **kwargs
) -> HFAccelerateAdapter:
    """
    创建并加载模型的适配器
    
    Args:
        model_path: 模型路径
        dp_size: 数据并行大小
        ep_size: 专家并行大小
        **kwargs: 其他参数
    
    Returns:
        加载好模型的适配器
    """
    adapter = HFAccelerateAdapter()
    adapter.load_model(
        model_path,
        data_parallel_size=dp_size,
        expert_parallel_size=ep_size,
        **kwargs
    )
    return adapter


# 示例用法
if __name__ == "__main__":
    """
    使用示例:
    
    单GPU测试:
        python hf_accelerate_adapter.py
    
    分布式测试:
        torchrun --nproc_per_node=2 hf_accelerate_adapter.py
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../models/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--dp_size', type=int, default=2)
    parser.add_argument('--ep_size', type=int, default=2)
    args = parser.parse_args()
    
    # 创建适配器
    adapter = create_hf_accelerate_adapter()
    
    # 加载模型
    adapter.load_model(
        args.model_path,
        data_parallel_size=args.dp_size,
        expert_parallel_size=args.ep_size,
    )
    
    # 测试推理
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a short poem.",
    ]
    
    outputs = adapter.run_inference(prompts)
    
    # 打印结果（只在rank 0打印）
    if adapter.get_dp_rank() == 0:
        print(f"\n=== Inference Results (DP Rank {adapter.get_dp_rank()}) ===")
        for text in adapter.get_generated_texts():
            print(f"Generated: {text[:100]}...")
        
        print(f"\nComm delay: {adapter.get_comm_delay():.4f}s")
        print(f"Hidden size: {adapter.get_hidden_size()}")
        print(f"MoE layers found: {len(adapter.get_moe_layers())}")
    
    # 清理
    adapter.cleanup()
