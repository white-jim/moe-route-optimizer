"""
vLLM框架适配器
提供vLLM推理框架的具体实现
参考: vllm/examples/offline_inference/torchrun_dp_example.py
"""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Iterator, Tuple, Union
import time

import sys
# 确保使用与 vllm/all2all.py 相同的导入路径
workspace_root = "/mnt/data/lwy/vLLM-wrok/moe_route_optimizer"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from interfaces.framework_interface import (
    InferenceFrameworkInterface,
    FrameworkMetricsCollector,
)
from config import get_logger

# 导入通信时延收集器 - 必须使用与 all2all.py 完全相同的导入路径！
from hooks.comm_delay_collector import (
    get_collector as get_comm_collector,
    reset_collector as reset_comm_collector,
    get_total_comm_delay,
    get_comm_statistics,
)

# 设置CUDA内存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class VLLMAdapter(InferenceFrameworkInterface):
    """
    vLLM框架适配器
    
    支持单机和分布式推理（使用torchrun启动）
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.llm = None  # vLLM的LLM实例
        self.model = None  # 底层PyTorch模型
        self.tokenizer = None
        self.sampling_params = None
        self._last_output = None
        self._last_outputs = []  # 批量输出
        self._last_comm_delay = 0.0
        self._comm_delay_per_layer: Dict[int, float] = {}
        self._hidden_size = 0
        self._moe_layers: List[nn.Module] = []
        self._first_moe_gate = None
        self._first_moe_block = None  # 整个MoE块，用于正确注册hook
        
        # 分布式相关
        self._dp_rank = 0
        self._dp_size = 1
        
        # 指标收集器
        self.metrics_collector: Optional[VLLMMetricsCollector] = None
    
    def load_model(self, model_path: str, **kwargs) -> nn.Module:
        """
        加载vLLM模型
        
        Args:
            model_path: 模型路径
            **kwargs: 支持的参数包括:
                - tensor_parallel_size (int): 张量并行大小，默认1
                - data_parallel_size (int): 数据并行大小，默认1
                - pipeline_parallel_size (int): 流水线并行大小，默认1
                - enable_expert_parallel (bool): 是否启用专家并行，默认False
                - max_model_len (int): 最大模型长度，默认2048
                - gpu_memory_utilization (float): GPU内存使用率，默认0.9
                - seed (int): 随机种子，默认1
                - trust_remote_code (bool): 是否信任远程代码，默认True
                - distributed_executor_backend (str): 分布式后端，默认None
        
        Returns:
            加载的底层模型
        """
        from vllm import LLM, SamplingParams
        
        self.logger.info(f"Loading vLLM model from: {model_path}")
        
        # 解析参数
        tp_size = kwargs.get('tensor_parallel_size', 1)
        dp_size = kwargs.get('data_parallel_size', 1)
        pp_size = kwargs.get('pipeline_parallel_size', 1)
        enable_ep = kwargs.get('enable_expert_parallel', False)
        max_model_len = kwargs.get('max_model_len', 2048)
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        seed = kwargs.get('seed', 1)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        distributed_backend = kwargs.get('distributed_executor_backend', None)
        
        # 创建LLM实例
        llm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tp_size,
            'pipeline_parallel_size': pp_size,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'seed': seed,
            'trust_remote_code': trust_remote_code,
        }
        
        # 如果使用数据并行（torchrun启动时）
        if dp_size > 1 or distributed_backend == 'external_launcher':
            llm_kwargs['data_parallel_size'] = dp_size
            llm_kwargs['enable_expert_parallel'] = enable_ep
            llm_kwargs['distributed_executor_backend'] = 'external_launcher'
        
        self.logger.info(f"LLM config: tp={tp_size}, dp={dp_size}, pp={pp_size}, ep={enable_ep}")
        
        self.llm = LLM(**llm_kwargs)
        
        # 获取tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        
        # 获取底层模型（用于注册hook）
        # 参考: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        try:
            self.model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
            self.logger.info("Successfully accessed underlying model for hook registration")
        except AttributeError as e:
            self.logger.warning(f"Could not access underlying model directly: {e}")
            self.model = None
        
        # 获取分布式信息
        try:
            parallel_config = self.llm.llm_engine.vllm_config.parallel_config
            self._dp_rank = parallel_config.data_parallel_rank
            self._dp_size = parallel_config.data_parallel_size
            self.logger.info(f"Distributed info: dp_rank={self._dp_rank}, dp_size={self._dp_size}")
        except AttributeError:
            self._dp_rank = 0
            self._dp_size = 1
        
        # 获取hidden_size
        try:
            model_config = self.llm.llm_engine.model_config
            self._hidden_size = model_config.hf_config.hidden_size
            self.logger.info(f"Model hidden_size: {self._hidden_size}")
        except AttributeError:
            self._hidden_size = kwargs.get('hidden_size', 2048)
            self.logger.warning(f"Could not get hidden_size from config, using: {self._hidden_size}")
        
        # 创建默认的采样参数
        self.sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.0),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_tokens', 128),
        )
        
        # 查找MoE层
        self._find_moe_layers()
        
        return self.model
    
    def run_inference(self, inputs: Any, sampling_params: Any = None) -> Any:
        """
        执行批量推理
        
        Args:
            inputs: 输入数据，支持以下格式:
                - str: 单个文本
                - List[str]: 多个文本
                - Dict: 包含'prompts'键的字典
            sampling_params: 可选的采样参数，默认使用初始化时的参数
        
        Returns:
            推理输出列表
        """
        if self.llm is None:
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
        
        # 如果是分布式环境，根据dp_rank分配prompts
        if self._dp_size > 1:
            prompts = [
                prompt for idx, prompt in enumerate(prompts) 
                if idx % self._dp_size == self._dp_rank
            ]
            self.logger.debug(f"DP rank {self._dp_rank}: processing {len(prompts)} prompts")
        
        # 使用指定的或默认的采样参数
        params = sampling_params if sampling_params is not None else self.sampling_params
        
        # 开始指标收集
        if self.metrics_collector is not None:
            self.metrics_collector.start_collection()
        
        # [ADDED] 重置通信时延收集器，准备收集本次推理的时延
        reset_comm_collector()
        # get_comm_collector()._current_total_delay_ms = 0
        
        # 执行推理
        start_time = time.time()
        try:
            outputs = self.llm.generate(prompts, params)
            inference_time = time.time() - start_time
            self.logger.debug(f"Inference completed in {inference_time:.3f}s for {len(prompts)} prompts")
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            if self.metrics_collector is not None:
                self.metrics_collector.stop_collection()
            return None
        
        # 停止指标收集
        if self.metrics_collector is not None:
            self.metrics_collector.stop_collection()
        
        # 保存输出
        self._last_outputs = outputs
        self._last_output = outputs[0] if outputs else None
        
        # 更新通信时延（从全局收集器获取）
        # self._update_comm_delay()
        
        return outputs
    
    def _update_comm_delay(self):
        """
        更新通信时延统计
            
        从vLLM all2all通信层的全局收集器获取时延数据
        """
        # 从vLLM底层收集器获取时延
        self._last_comm_delay = get_total_comm_delay()  # 已经是秒
            
        # 获取每层时延
        comm_collector = get_comm_collector()
        layer_delays_ms = comm_collector.get_delay_per_layer()
        self._comm_delay_per_layer = {
            layer_idx: delay_ms / 1000.0  # 转换为秒
            for layer_idx, delay_ms in layer_delays_ms.items()
        }
            
        if self.metrics_collector is not None:
            # 同步到metrics_collector
            self.metrics_collector._total_comm_time = self._last_comm_delay
            self.metrics_collector._comm_delay_per_layer = self._comm_delay_per_layer.copy()
            
        self.logger.debug(f"Comm delay updated: {self._last_comm_delay:.4f}s")
    
    def _find_moe_layers(self):
        """
        查找模型中的MoE层和MoE块
        
        优先查找完整的MoE块（如Qwen2MoeSparseMoeBlock），
        因为在MoE块上注册hook可以确保扰动同时影响gate和experts。
        """
        if self.model is None:
            self.logger.warning("Model not accessible, cannot find MoE layers")
            return
        
        self._moe_layers = []
        self._first_moe_gate = None
        self._first_moe_block = None
        
        # MoE块的类名关键词（完整的MoE块，包含gate和experts）
        moe_block_keywords = ['sparsemoe', 'moeblock', 'moelayer', 'moe']
        # 排除的关键词（这些是MoE块内部的组件，不是完整的块）
        exclude_keywords = ['expert', 'fusedmoe', 'sharedfusedmoe', 'gate', 'router']
        
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__.lower()
            name_lower = name.lower()
            
            # 检查是否是完整的MoE块（类名匹配moe_block_keywords且不在排除列表中）
            is_moe_block = any(kw in class_name for kw in moe_block_keywords)
            is_excluded = any(kw in class_name for kw in exclude_keywords)
            
            # 完整的MoE块应该同时包含gate和experts属性
            has_gate = hasattr(module, 'gate')
            has_experts = hasattr(module, 'experts')
            is_complete_moe_block = has_gate and has_experts
            
            if is_moe_block and not is_excluded and is_complete_moe_block:
                self._moe_layers.append(module)
                self.logger.debug(f"Found MoE block: {name} ({module.__class__.__name__})")
                
                # 保存第一个完整的MoE块
                if self._first_moe_block is None:
                    self._first_moe_block = module
                    self.logger.info(f"Found first MoE block at: {name} ({module.__class__.__name__})")
                
                # 同时保存gate用于向后兼容
                if self._first_moe_gate is None and has_gate:
                    self._first_moe_gate = module.gate
                    self.logger.info(f"Found first MoE gate at: {name}.gate")
                continue
            
            # 如果还没找到MoE块，也记录其他MoE相关层
            is_moe_related = any(kw in class_name for kw in ['moe', 'expert', 'mixture', 'sparse'])
            is_moe_related = is_moe_related or any(kw in name_lower for kw in ['moe', 'expert'])
            
            if is_moe_related and module not in self._moe_layers:
                self._moe_layers.append(module)
                self.logger.debug(f"Found MoE layer: {name} ({module.__class__.__name__})")
                
                # 如果还没有找到gate，尝试查找
                if self._first_moe_gate is None:
                    for gate_attr in ['gate', 'router', 'gating_network', 'w_gate']:
                        if hasattr(module, gate_attr):
                            self._first_moe_gate = getattr(module, gate_attr)
                            self.logger.info(f"Found first MoE gate at: {name}.{gate_attr}")
                            break
        
        self.logger.info(f"Found {len(self._moe_layers)} MoE layers/blocks")
        if self._first_moe_block:
            self.logger.info(f"First MoE block type: {self._first_moe_block.__class__.__name__}")
        elif self._first_moe_gate:
            self.logger.info(f"First MoE gate type: {self._first_moe_gate.__class__.__name__}")
        else:
            self.logger.warning("No MoE block or gate found!")
    
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
            if hasattr(output, 'outputs') and output.outputs:
                texts.append(output.outputs[0].text)
            else:
                texts.append("")
        return texts
    
    def get_output_with_prompt(self) -> List[Tuple[str, str]]:
        """
        获取(prompt, generated_text)对的列表
        
        Returns:
            [(prompt, generated_text), ...]
        """
        if not self._last_outputs:
            return []
        
        results = []
        for output in self._last_outputs:
            prompt = output.prompt if hasattr(output, 'prompt') else ""
            generated = output.outputs[0].text if hasattr(output, 'outputs') and output.outputs else ""
            results.append((prompt, generated))
        return results
    
    def get_comm_delay(self) -> float:
        """
        获取本次推理的all-to-all通信时延
        
        Returns:
            总通信时延（秒）
        """
        # 直接从CommDelayCollector单例获取时延数据
        return get_total_comm_delay()
    
    def get_comm_delay_per_layer(self) -> Dict[int, float]:
        """
        获取每层MoE的通信时延
        
        Returns:
            {layer_idx: delay_seconds, ...}
        """
        if self.metrics_collector is not None:
            return self.metrics_collector.get_comm_delay_per_layer()
        
        return self._comm_delay_per_layer
    
    def get_comm_statistics(self) -> Dict[str, float]:
        """
        获取详细的通信统计信息
        
        Returns:
            包含各项统计的字典:
            - total_delay_ms: 总时延(毫秒)
            - total_delay_s: 总时延(秒)
            - dispatch_count: dispatch调用次数
            - combine_count: combine调用次数
            - dispatch_avg_ms: dispatch平均时延
            - combine_avg_ms: combine平均时延
            ...
        """
        return get_comm_statistics()
    
    def reset_comm_stats(self):
        """
        重置通信统计
        在每次新的推理开始前调用
        """
        reset_comm_collector()
    
    def get_moe_layers(self) -> List[nn.Module]:
        """
        获取模型中的MoE层列表
        
        Returns:
            MoE层模块列表
        """
        if not self._moe_layers and self.model is not None:
            self._find_moe_layers()
        return self._moe_layers
    
    def get_first_moe_block(self) -> nn.Module:
        """
        获取第一个完整的MoE块（包含gate和experts）
        
        这是推荐的hook注册位置，因为：
        1. 在MoE块上注册hook，扰动后的hidden_states会同时传给gate和experts
        2. 只在gate上注册hook，扰动只影响路由决策，不影响传给experts的hidden_states
        
        Returns:
            完整的MoE块模块，用于注册hook
        """
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        # 如果没有找到MoE块，尝试重新查找
        self._find_moe_layers()
        
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        # 如果仍然找不到MoE块，返回第一个MoE层
        moe_layers = self.get_moe_layers()
        if moe_layers:
            self.logger.warning("Could not find specific MoE block, returning first MoE layer")
            return moe_layers[0]
        
        self.logger.error("No MoE layers/blocks found")
        return None
    
    def get_first_moe_gate(self) -> nn.Module:
        """
        获取第一个MoE层的Gate模块
        
        注意：不推荐在gate上注册扰动hook，因为扰动只会影响路由决策，
        不会影响传给experts的hidden_states。推荐使用get_first_moe_block()。
        
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
        
        # 尝试常见的gate属性名
        for attr_name in ['gate', 'router', 'gating_network', 'w_gate', 'gate_proj', 'gate_up_proj']:
            if hasattr(first_moe, attr_name):
                self._first_moe_gate = getattr(first_moe, attr_name)
                self.logger.info(f"Found gate attribute: {attr_name}")
                return self._first_moe_gate
        
        # 如果找不到gate，返回整个MoE层（可以在其输入处注册hook）
        self.logger.warning("Could not find gate in first MoE layer, returning layer itself")
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
        
        return self.tokenizer.decode(token_ids)
    
    def set_sampling_params(self, **kwargs):
        """
        设置采样参数
        
        Args:
            temperature (float): 温度参数
            top_p (float): nucleus sampling参数
            top_k (int): top-k sampling参数
            max_tokens (int): 最大生成token数
            等等...
        """
        from vllm import SamplingParams
        self.sampling_params = SamplingParams(**kwargs)
        self.logger.info(f"Updated sampling params: {kwargs}")
    
    def set_metrics_collector(self, collector: 'VLLMMetricsCollector'):
        """设置指标收集器"""
        self.metrics_collector = collector
    
    def get_underlying_model(self) -> Optional[nn.Module]:
        """
        获取底层PyTorch模型
        用于注册hook等操作
        
        Returns:
            底层模型，如果无法访问则返回None
        """
        return self.model
    
    def get_llm_engine(self) -> Any:
        """
        获取vLLM的LLM引擎
        
        Returns:
            LLM引擎实例
        """
        return self.llm.llm_engine if self.llm else None


class VLLMMetricsCollector(FrameworkMetricsCollector):
    """
    vLLM指标收集器
    收集推理过程中的通信时延等指标
    
    支持注册hook来收集all-to-all通信时延
    """
    
    def __init__(self, llm_adapter: Optional[VLLMAdapter] = None):
        self.logger = get_logger()
        self.llm_adapter = llm_adapter
        self._is_collecting = False
        self._total_comm_time = 0.0
        self._total_compute_time = 0.0
        self._comm_delay_per_layer: Dict[int, float] = {}
        self._routing_distribution: Dict[int, int] = {}
        self._hook_handles = []
        self._layer_start_times: Dict[int, float] = {}
    
    def start_collection(self):
        """开始收集指标"""
        self._is_collecting = True
        self.reset()
        self.logger.debug("Started metrics collection")
    
    def stop_collection(self):
        """停止收集指标"""
        self._is_collecting = False
        self.logger.debug(f"Stopped metrics collection. Total comm time: {self._total_comm_time:.4f}s")
    
    def get_total_comm_time(self) -> float:
        """获取总通信时间"""
        return self._total_comm_time
    
    def get_total_compute_time(self) -> float:
        """获取总计算时间"""
        return self._total_compute_time
    
    def get_comm_delay_per_layer(self) -> Dict[int, float]:
        """获取每层的通信时延"""
        return self._comm_delay_per_layer.copy()
    
    def get_routing_distribution(self) -> Dict[int, int]:
        """获取路由分布统计"""
        return self._routing_distribution.copy()
    
    def reset(self):
        """重置收集器"""
        self._total_comm_time = 0.0
        self._total_compute_time = 0.0
        self._comm_delay_per_layer.clear()
        self._routing_distribution.clear()
        self._layer_start_times.clear()
    
    def record_comm_time(self, layer_idx: int, comm_time: float):
        """
        记录某层的通信时间
        
        Args:
            layer_idx: 层索引
            comm_time: 通信时间（秒）
        """
        if not self._is_collecting:
            return
        
        if layer_idx in self._comm_delay_per_layer:
            self._comm_delay_per_layer[layer_idx] += comm_time
        else:
            self._comm_delay_per_layer[layer_idx] = comm_time
        self._total_comm_time += comm_time
        
        self.logger.debug(f"Layer {layer_idx} comm time: {comm_time:.6f}s")
    
    def record_routing(self, expert_id: int, count: int = 1):
        """
        记录路由选择
        
        Args:
            expert_id: 专家ID
            count: 路由到该专家的token数量
        """
        if not self._is_collecting:
            return
        
        if expert_id not in self._routing_distribution:
            self._routing_distribution[expert_id] = 0
        self._routing_distribution[expert_id] += count
    
    def register_comm_hooks(self, model: nn.Module, moe_layer_names: Optional[List[str]] = None):
        """
        注册通信时延收集hook
        
        Args:
            model: 模型
            moe_layer_names: 可选的MoE层名称列表，用于精确定位
        
        Note: 这是一个基础实现，可能需要根据具体框架调整
        """
        if model is None:
            self.logger.warning("Model is None, cannot register hooks")
            return
        
        layer_idx = 0
        for name, module in model.named_modules():
            name_lower = name.lower()
            
            # 检查是否是MoE相关模块
            if moe_layer_names:
                if name not in moe_layer_names:
                    continue
            else:
                if not any(kw in name_lower for kw in ['moe', 'expert', 'all_to_all', 'a2a']):
                    continue
            
            # 创建闭包来捕获layer_idx
            current_layer_idx = layer_idx
            
            def pre_hook(module, inputs, layer_idx=current_layer_idx):
                if self._is_collecting:
                    self._layer_start_times[layer_idx] = time.time()
                return inputs
            
            def post_hook(module, inputs, outputs, layer_idx=current_layer_idx):
                if self._is_collecting and layer_idx in self._layer_start_times:
                    elapsed = time.time() - self._layer_start_times[layer_idx]
                    # 这里假设大部分时间是通信时延
                    # 实际实现中可能需要更精确的方式来区分计算和通信
                    self.record_comm_time(layer_idx, elapsed)
                return outputs
            
            handle_pre = module.register_forward_pre_hook(pre_hook)
            handle_post = module.register_forward_hook(post_hook)
            self._hook_handles.extend([handle_pre, handle_post])
            
            self.logger.debug(f"Registered comm hooks for layer {layer_idx}: {name}")
            layer_idx += 1
        
        self.logger.info(f"Registered comm hooks for {layer_idx} layers")
    
    def remove_hooks(self):
        """移除所有已注册的hook"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.logger.debug("Removed all comm hooks")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Returns:
            包含各项指标的字典
        """
        return {
            'total_comm_time': self._total_comm_time,
            'total_compute_time': self._total_compute_time,
            'comm_delay_per_layer': self.get_comm_delay_per_layer(),
            'routing_distribution': self.get_routing_distribution(),
            'num_layers_monitored': len(self._comm_delay_per_layer),
        }


def create_vllm_adapter() -> VLLMAdapter:
    """创建vLLM适配器实例"""
    return VLLMAdapter()


def create_vllm_adapter_with_metrics() -> Tuple[VLLMAdapter, VLLMMetricsCollector]:
    """
    创建带指标收集器的vLLM适配器
    
    Returns:
        (adapter, metrics_collector) 元组
    """
    adapter = VLLMAdapter()
    collector = VLLMMetricsCollector(adapter)
    adapter.set_metrics_collector(collector)
    return adapter, collector
