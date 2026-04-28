"""
推理框架抽象接口模块
定义与推理框架交互的标准接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator, Tuple
import torch
import torch.nn as nn


class InferenceFrameworkInterface(ABC):
    """
    推理框架抽象接口
    定义与不同推理框架（如vLLM、DeepSpeed等）交互的标准方法
    """
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            **kwargs: 额外参数
        
        Returns:
            加载的模型
        """
        pass
    
    @abstractmethod
    def run_inference(self, inputs: Any) -> Any:
        """
        执行推理
        
        Args:
            inputs: 输入数据
        
        Returns:
            推理输出
        """
        pass
    
    @abstractmethod
    def get_model_output(self) -> Any:
        """
        获取模型的推理输出
        
        Returns:
            最近一次推理的输出
        """
        pass
    
    @abstractmethod
    def get_comm_delay(self) -> float:
        """
        获取本次推理的all-to-all通信时延
        
        Returns:
            通信时延（秒）
        """
        pass
    
    @abstractmethod
    def get_comm_delay_per_layer(self) -> Dict[int, float]:
        """
        获取每层MoE的通信时延
        
        Returns:
            {layer_idx: delay} 字典
        """
        pass
    
    @abstractmethod
    def get_moe_layers(self) -> List[nn.Module]:
        """
        获取模型中的MoE层列表
        
        Returns:
            MoE层模块列表
        """
        pass
    
    @abstractmethod
    def get_first_moe_gate(self) -> nn.Module:
        """
        获取第一个MoE层的Gate模块
        
        Returns:
            Gate模块
        """
        pass
    
    @abstractmethod
    def get_hidden_size(self) -> int:
        """
        获取模型的隐藏层维度
        
        Returns:
            隐藏层维度
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        对输入文本进行tokenize
        
        Args:
            text: 输入文本
        
        Returns:
            tokenize后的张量字典
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        将token ids解码为文本
        
        Args:
            token_ids: token id张量
        
        Returns:
            解码后的文本
        """
        pass


class AccuracyEvaluatorInterface(ABC):
    """
    精度评估器抽象接口
    定义不同任务数据集的评估标准方法
    """
    
    @abstractmethod
    def get_dataset_iterator(self) -> Iterator[Tuple[str, Any]]:
        """
        获取评估数据集的迭代器
        
        Returns:
            迭代器，每次返回 (input_text, ground_truth)
        """
        pass
    
    @abstractmethod
    def evaluate_single(self, model_output: Any, ground_truth: Any) -> float:
        """
        评估单个样本的精度
        
        Args:
            model_output: 模型输出
            ground_truth: 标准答案
        
        Returns:
            精度分数 (0-1)
        """
        pass
    
    @abstractmethod
    def evaluate_batch(self, model_outputs: List[Any], 
                       ground_truths: List[Any]) -> float:
        """
        评估一批样本的平均精度
        
        Args:
            model_outputs: 模型输出列表
            ground_truths: 标准答案列表
        
        Returns:
            平均精度分数 (0-1)
        """
        pass
    
    @abstractmethod
    def get_dataset_size(self) -> int:
        """
        获取数据集大小
        
        Returns:
            数据集样本数量
        """
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        获取数据集名称
        
        Returns:
            数据集名称字符串
        """
        pass


class FrameworkMetricsCollector(ABC):
    """
    框架指标收集器抽象接口
    用于收集推理过程中的各种指标
    """
    
    @abstractmethod
    def start_collection(self):
        """开始收集指标"""
        pass
    
    @abstractmethod
    def stop_collection(self):
        """停止收集指标"""
        pass
    
    @abstractmethod
    def get_total_comm_time(self) -> float:
        """获取总通信时间"""
        pass
    
    @abstractmethod
    def get_total_compute_time(self) -> float:
        """获取总计算时间"""
        pass
    
    @abstractmethod
    def get_routing_distribution(self) -> Dict[int, int]:
        """
        获取路由分布统计
        
        Returns:
            {expert_id: count} 字典
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置收集器"""
        pass
