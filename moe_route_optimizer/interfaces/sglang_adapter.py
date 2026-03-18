"""
SGLang 推理适配器
支持数据并行(DP)和专家并行(EP)的MoE模型推理

SGLang特点:
- 支持真正的专家并行(EP)，使用all-to-all通信
- 支持enable_ep_moe / enable_deepep_moe两种EP模式
- 高性能的MoE推理
- 框架内部自行管理多rank和各类并行，外部只需要输入数据、获取输出

使用方式：
    python your_script.py --model-path /path/to/model
"""

import dataclasses
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from interfaces.framework_interface import InferenceFrameworkInterface
from config import get_logger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SGLangAdapter(InferenceFrameworkInterface):
    """
    SGLang推理框架适配器
    
    支持专家并行(EP)的MoE模型分布式推理
    
    专家并行配置:
    - tp_size: 张量并行大小，同时决定EP大小（enable_ep_moe时ep_size=tp_size）
    - enable_ep_moe: 启用标准EP MoE模式（ep_size自动设为tp_size）
    - enable_deepep_moe: 启用DeepEP MoE模式（ep_size自动设为tp_size）
    - ep_size: 专家并行大小（通常通过enable_ep_moe/enable_deepep_moe自动设置）
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
                - data_parallel_size (int): 数据并行大小
                - enable_ep_moe (bool): 启用EP MoE模式（ep_size自动设为tp_size）
                - enable_deepep_moe (bool): 启用DeepEP MoE模式
                - context_length (int): 最大上下文长度
                - trust_remote_code (bool): 是否信任远程代码
                - max_tokens (int): 默认最大生成token数
                - random_seed (int): 随机种子
        
        Returns:
            加载的模型（SGLang Engine，框架内部自行管理实际模型参数）
        """
        self.logger.info(f"Loading SGLang model from: {model_path}")
        
        tp_size = kwargs.get('tensor_parallel_size', 1)
        dp_size = kwargs.get('data_parallel_size', 1)
        enable_ep_moe = kwargs.get('enable_ep_moe', False)
        enable_deepep_moe = kwargs.get('enable_deepep_moe', False)
        context_length = kwargs.get('context_length', None)
        trust_remote_code = kwargs.get('trust_remote_code', True)
        max_tokens = kwargs.get('max_tokens', 128)
        random_seed = kwargs.get('random_seed', 42)
        
        # ep_size 由 enable_ep_moe/enable_deepep_moe 在 ServerArgs.__post_init__ 中自动设为 tp_size
        # 这里记录的 ep_size 仅供外部查询使用
        ep_size = tp_size if (enable_ep_moe or enable_deepep_moe) else kwargs.get('ep_size', 1)
        self._init_distributed(tp_size=tp_size, ep_size=ep_size, dp_size=dp_size)
        
        try:
            import sglang as sgl
            from sglang.srt.server_args import ServerArgs
            from transformers import AutoTokenizer, AutoConfig
            
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
            
            # sampling_params 使用 dict 格式（新版 SGLang Engine.generate 接受 dict）
            self.sampling_params = {
                "temperature": kwargs.get('temperature', 0.0),
                "top_p": kwargs.get('top_p', 1.0),
                "max_new_tokens": max_tokens,
            }
            
            self.logger.info(
                f"SGLang config: tp={tp_size}, ep={ep_size}, dp={dp_size}, "
                f"enable_ep_moe={enable_ep_moe}, enable_deepep_moe={enable_deepep_moe}, "
                f"hidden_size={self._hidden_size}"
            )
            
            server_args = ServerArgs(
                model_path=model_path,
                tp_size=tp_size,
                dp_size=dp_size,
                enable_ep_moe=enable_ep_moe,
                enable_deepep_moe=enable_deepep_moe,
                trust_remote_code=trust_remote_code,
                context_length=context_length,
                random_seed=random_seed,
            )
            
            self._sglang_engine = sgl.Engine(**dataclasses.asdict(server_args))
            
            # SGLang Engine 不直接暴露 nn.Module，model 属性设为 engine 本身供外部感知
            self.model = self._sglang_engine
            
            self._find_moe_layers()
            
            self.logger.info("SGLang model loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import SGLang: {e}")
            raise RuntimeError(f"SGLang not available: {e}")
        
        return self.model
    
    def _find_moe_layers(self):
        """查找模型中的MoE层（SGLang Engine 不直接暴露 nn.Module，跳过层查找）"""
        self._moe_layers = []
        self._first_moe_gate = None
        self._first_moe_block = None
        
        # 新版 SGLang 使用 Engine 接口，不直接暴露 nn.Module 给外部
        # MoE 层由框架内部管理，无法通过 named_modules() 遍历
        self.logger.info("SGLang Engine does not expose nn.Module directly; MoE layer search skipped.")
    
    def run_inference(self, inputs: Any, sampling_params: Any = None) -> Any:
        """
        执行批量推理
        
        Args:
            inputs: 输入数据，支持:
                - str: 单个文本
                - List[str]: 多个文本
                - Dict: 包含'prompts'键的字典
            sampling_params: 可选的采样参数，支持 dict 格式
                例如: {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
        
        Returns:
            推理输出列表（与vLLM格式兼容）
        """
        if not hasattr(self, '_sglang_engine'):
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
        
        # 新版 SGLang Engine.generate 接受 dict 格式的 sampling_params
        params = sampling_params if sampling_params is not None else self.sampling_params
        
        start_time = time.time()
        
        try:
            # engine.generate 返回 list of dict，每个 dict 包含 "text" 等字段
            raw_outputs = self._sglang_engine.generate(prompts, params)
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None
        
        self._last_comm_delay = time.time() - start_time
        
        self._last_outputs = self._convert_outputs(raw_outputs, prompts)
        self._last_output = self._last_outputs[0] if self._last_outputs else None
        
        self.logger.debug(
            f"Inference completed for {len(prompts)} prompts in {self._last_comm_delay:.3f}s"
        )
        
        return self._last_outputs
    
    def _convert_outputs(self, outputs: Any, prompts: List[str]) -> List[Any]:
        """
        将SGLang输出转换为与vLLM兼容的格式
        
        新版 SGLang Engine.generate 返回 list of dict，每个 dict 包含:
            - "text": str  生成的文本
            - "meta_info": dict  元信息（包含 token 数量等）
        
        转换后格式（vLLM兼容）:
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
                    # 新版 SGLang Engine 返回 dict，包含 "text" 字段
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
    data_parallel_size: int = 1,
    enable_ep_moe: bool = False,
    enable_deepep_moe: bool = False,
    **kwargs
) -> SGLangAdapter:
    """
    创建并加载模型的适配器
    
    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小（同时决定 EP 大小）
        data_parallel_size: 数据并行大小
        enable_ep_moe: 启用标准EP MoE模式（ep_size自动设为tp_size）
        enable_deepep_moe: 启用DeepEP MoE模式（ep_size自动设为tp_size）
        **kwargs: 其他参数（context_length, random_seed, max_tokens, temperature, etc.）
    
    Returns:
        加载好模型的适配器
    """
    adapter = create_sglang_adapter()
    adapter.load_model(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_ep_moe=enable_ep_moe,
        enable_deepep_moe=enable_deepep_moe,
        **kwargs
    )
    return adapter
