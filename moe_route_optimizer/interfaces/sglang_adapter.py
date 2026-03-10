"""
SGLang 推理适配器
支持数据并行(DP)和专家并行(EP)的MoE模型推理

SGLang特点:
- 支持真正的专家并行(EP)，使用all-to-all通信
- 支持多种all-to-all后端: deepep, flashinfer, mooncake, mori
- 高性能的MoE推理
- 框架内部自行管理多rank和各类并行，外部只需要输入数据、获取输出

使用方式：
    python your_script.py --model-path /path/to/model
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

workspace_root = "/mnt/data/lwy/vLLM-wrok/moe_route_optimizer"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from interfaces.framework_interface import InferenceFrameworkInterface
from config import get_logger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SGLangAdapter(InferenceFrameworkInterface):
    """
    SGLang推理框架适配器
    
    支持专家并行(EP)的MoE模型分布式推理
    
    专家并行配置:
    - ep_size: 专家并行大小，通常等于GPU数量
    - moe_a2a_backend: all-to-all通信后端
        - 'none': 使用all-reduce/all-gather
        - 'deepep': DeepSeek的高性能EP库
        - 'flashinfer': FlashInfer实现
        - 'mooncake': 支持弹性推理
        - 'mori': AMD ROCm优化版本
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        
        self._last_output = None
        self._last_outputs = []
        self._last_comm_delay = 0.0
        
        self._hidden_size = 0
        self._moe_layers: List[nn.Module] = []
        self._first_moe_gate = None
        self._first_moe_block = None
        
        self._dp_rank = 0
        self._dp_size = 1
        self._ep_size = 1
        self._tp_size = 1
        self._world_size = 1
        self._local_rank = 0
    
    def _init_distributed(self, tp_size: int = 1, ep_size: int = 1, dp_size: int = 1):
        """记录并行配置（框架内部自行处理分布式，外部不需要初始化torch.distributed）"""
        self._tp_size = tp_size
        self._ep_size = ep_size
        self._dp_size = dp_size
        # 从外部视角看，只有单进程，dp_rank始终为0
        self._dp_rank = 0
        self._local_rank = 0
        self._world_size = 1
        
        self.logger.info(
            f"Framework parallel config: tp_size={self._tp_size}, "
            f"ep_size={self._ep_size}, dp_size={self._dp_size}"
        )
    
    def load_model(self, model_path: str, **kwargs) -> nn.Module:
        """
        加载MoE模型
        
        Args:
            model_path: 模型路径
            **kwargs: 支持的参数:
                - tensor_parallel_size (int): 张量并行大小
                - expert_parallel_size (int): 专家并行大小
                - data_parallel_size (int): 数据并行大小
                - moe_a2a_backend (str): all-to-all后端
                - max_model_len (int): 最大模型长度
                - trust_remote_code (bool): 是否信任远程代码
                - max_tokens (int): 最大生成token数
        
        Returns:
            加载的模型
        """
        self.logger.info(f"Loading SGLang model from: {model_path}")
        
        tp_size = kwargs.get('tensor_parallel_size', 1)
        ep_size = kwargs.get('expert_parallel_size', 1)
        dp_size = kwargs.get('data_parallel_size', 1)
        moe_a2a_backend = kwargs.get('moe_a2a_backend', 'none')
        max_model_len = kwargs.get('max_model_len', 2048)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        max_tokens = kwargs.get('max_tokens', 128)
        
        self._init_distributed(tp_size=tp_size, ep_size=ep_size, dp_size=dp_size)
        
        try:
            from transformers import AutoTokenizer, AutoConfig
            from sglang import Runtime, SamplingParams
            
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            self._hidden_size = getattr(config, 'hidden_size', 2048)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                padding_side='left',
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', 0.0),
                top_p=kwargs.get('top_p', 1.0),
                max_new_tokens=max_tokens,
            )
            
            self.logger.info(
                f"SGLang config: tp={tp_size}, ep={ep_size}, dp={dp_size}, "
                f"moe_a2a_backend={moe_a2a_backend}, hidden_size={self._hidden_size}"
            )
            
            self._sglang_runtime = Runtime(
                model_path=model_path,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=dp_size,
                moe_a2a_backend=moe_a2a_backend,
                trust_remote_code=trust_remote_code,
                max_context_len=max_model_len,
            )
            
            self.model = getattr(self._sglang_runtime, 'model', None)
            
            self._find_moe_layers()
            
            self.logger.info("SGLang model loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import SGLang: {e}")
            raise RuntimeError(f"SGLang not available: {e}")
        
        return self.model
    
    def _find_moe_layers(self):
        """查找模型中的MoE层"""
        self._moe_layers = []
        self._first_moe_gate = None
        self._first_moe_block = None
        
        if self.model is None:
            return
        
        moe_keywords = [
            'moe', 'MoE', 'MOE',
            'mixtral', 'Mixtral',
            'sparsemoe', 'SparseMoe',
            'expert', 'Expert',
        ]
        
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            name_lower = name.lower()
            
            is_moe_block = any(kw.lower() in class_name.lower() for kw in moe_keywords)
            has_gate = hasattr(module, 'gate')
            has_experts = hasattr(module, 'experts')
            is_complete_moe_block = has_gate and has_experts
            
            if is_moe_block and is_complete_moe_block:
                self._moe_layers.append(module)
                
                if self._first_moe_block is None:
                    self._first_moe_block = module
                    self.logger.info(f"Found first MoE block: {name} ({class_name})")
                
                if self._first_moe_gate is None and has_gate:
                    self._first_moe_gate = module.gate
                continue
            
            if is_moe_block or any(kw in name_lower for kw in ['moe', 'expert']):
                if module not in self._moe_layers:
                    self._moe_layers.append(module)
        
        self.logger.info(f"Found {len(self._moe_layers)} MoE layers")
    
    def run_inference(self, inputs: Any, sampling_params: Any = None) -> Any:
        """
        执行批量推理
        
        Args:
            inputs: 输入数据，支持:
                - str: 单个文本
                - List[str]: 多个文本
                - Dict: 包含'prompts'键的字典
            sampling_params: 可选的采样参数
        
        Returns:
            推理输出列表（与vLLM格式兼容）
        """
        if not hasattr(self, '_sglang_runtime'):
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
            self._last_outputs = []
            self._last_output = None
            return []
        
        params = sampling_params if sampling_params is not None else self.sampling_params
        
        start_time = time.time()
        
        try:
            outputs = self._sglang_runtime.generate(
                prompts,
                params,
            )
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None
        
        self._last_comm_delay = time.time() - start_time
        
        self._last_outputs = self._convert_outputs(outputs, prompts)
        self._last_output = self._last_outputs[0] if self._last_outputs else None
        
        self.logger.debug(
            f"Inference completed for {len(prompts)} prompts in {self._last_comm_delay:.3f}s"
        )
        
        return self._last_outputs
    
    def _convert_outputs(self, outputs: Any, prompts: List[str]) -> List[Any]:
        """
        将SGLang输出转换为与vLLM兼容的格式
        
        vLLM输出格式:
        - RequestOutput对象，包含:
            - prompt: str
            - outputs: List[CompletionOutput]
                - text: str
                - token_ids: List[int]
        """
        from dataclasses import dataclass
        
        @dataclass
        class CompletionOutput:
            text: str
            token_ids: List[int] = None
            
            def __post_init__(self):
                if self.token_ids is None:
                    self.token_ids = []
        
        @dataclass
        class RequestOutput:
            prompt: str
            outputs: List[CompletionOutput]
        
        converted = []
        
        if isinstance(outputs, list):
            for i, output in enumerate(outputs):
                prompt = prompts[i] if i < len(prompts) else ""
                
                if isinstance(output, dict):
                    text = output.get('text', '')
                    token_ids = output.get('token_ids', [])
                elif isinstance(output, str):
                    text = output
                    token_ids = []
                elif hasattr(output, 'text'):
                    text = output.text
                    token_ids = getattr(output, 'token_ids', [])
                else:
                    text = str(output)
                    token_ids = []
                
                converted.append(RequestOutput(
                    prompt=prompt,
                    outputs=[CompletionOutput(text=text, token_ids=token_ids)]
                ))
        elif isinstance(outputs, dict):
            for i, prompt in enumerate(prompts):
                text = outputs.get('text', '') if 'text' in outputs else str(outputs)
                converted.append(RequestOutput(
                    prompt=prompt,
                    outputs=[CompletionOutput(text=text)]
                ))
        elif isinstance(outputs, str):
            converted.append(RequestOutput(
                prompt=prompts[0] if prompts else "",
                outputs=[CompletionOutput(text=outputs)]
            ))
        else:
            for i, prompt in enumerate(prompts):
                text = str(outputs) if i == 0 else ""
                converted.append(RequestOutput(
                    prompt=prompt,
                    outputs=[CompletionOutput(text=text)]
                ))
        
        return converted
    
    def get_model_output(self) -> Any:
        """获取最近一次推理的输出"""
        return self._last_output
    
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
    
    def get_batch_outputs(self) -> List[Any]:
        """获取最近一次批量推理的所有输出"""
        return self._last_outputs
    
    def get_dp_rank(self) -> int:
        """获取数据并行rank（从外部视角始终为0，框架内部自行处理并行）"""
        return self._dp_rank
    
    def get_dp_size(self) -> int:
        """获取数据并行大小（从外部视角始终为1，框架内部自行处理并行）"""
        return self._dp_size
    
    def get_comm_delay(self) -> float:
        """
        获取本次推理的时延
        
        Returns:
            推理时延（秒）
        """
        return self._last_comm_delay
    
    def get_hidden_size(self) -> int:
        """获取模型的隐藏层维度"""
        return self._hidden_size
    
    def get_moe_layers(self) -> List[nn.Module]:
        """获取模型中的MoE层列表"""
        if not self._moe_layers and self.model is not None:
            self._find_moe_layers()
        return self._moe_layers
    
    def get_first_moe_block(self) -> Optional[nn.Module]:
        """
        获取第一个完整的MoE块（包含gate和experts）
        
        Returns:
            完整的MoE块模块
        """
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        self._find_moe_layers()
        
        if self._first_moe_block is not None:
            return self._first_moe_block
        
        if self._moe_layers:
            return self._moe_layers[0]
        
        return None
    
    def get_first_moe_gate(self) -> Optional[nn.Module]:
        """
        获取第一个MoE层的Gate模块
        
        Returns:
            Gate模块
        """
        if self._first_moe_gate is not None:
            return self._first_moe_gate
        
        self._find_moe_layers()
        
        if self._first_moe_gate is not None:
            return self._first_moe_gate
        
        if self._moe_layers:
            first_moe = self._moe_layers[0]
            for attr in ['gate', 'router', 'gating_network', 'w_gate', 'gate_proj']:
                if hasattr(first_moe, attr):
                    return getattr(first_moe, attr)
        
        return None
    
    def tokenize(self, text: Union[str, List[str]]) -> Any:
        """对输入文本进行tokenize"""
        if self.tokenizer is None:
            self.logger.error("Tokenizer not loaded")
            return None
        
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        return [self.tokenizer.encode(t) for t in text]
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """将token ids解码为文本"""
        if self.tokenizer is None:
            self.logger.error("Tokenizer not loaded")
            return ""
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def create_sglang_adapter() -> SGLangAdapter:
    """创建SGLang适配器实例"""
    return SGLangAdapter()


def create_sglang_adapter_with_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    data_parallel_size: int = 1,
    moe_a2a_backend: str = "none",
    **kwargs
) -> SGLangAdapter:
    """
    创建并加载模型的适配器
    
    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小
        expert_parallel_size: 专家并行大小
        data_parallel_size: 数据并行大小
        moe_a2a_backend: all-to-all后端
        **kwargs: 其他参数
    
    Returns:
        加载好模型的适配器
    """
    adapter = create_sglang_adapter()
    adapter.load_model(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        data_parallel_size=data_parallel_size,
        moe_a2a_backend=moe_a2a_backend,
        **kwargs
    )
    return adapter
