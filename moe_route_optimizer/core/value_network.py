"""
价值网络模块 (Critic) - 优化版
用于估计状态价值，简化结构以提升推理速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ValueNetwork(nn.Module):
    """
    价值网络 (Critic) - 简化版
    使用简单平均池化 + 两层网络
    """
    
    def __init__(self, hidden_size: int, value_hidden_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 简化为两层网络
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，估计状态价值
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) 可选
        
        Returns:
            values: (batch_size,)
        """
        # 使用简单平均池化（比注意力池化更快）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        return self.value_net(pooled).squeeze(-1)


def create_value_network(hidden_size: int, 
                         value_hidden_dim: int = 64,
                         use_attention: bool = False,  # 默认不使用注意力
                         device: str = 'cuda',
                         dtype: torch.dtype = None) -> nn.Module:
    """
    创建价值网络的工厂函数
    
    Args:
        hidden_size: 模型隐藏层维度
        value_hidden_dim: 价值网络隐藏层维度
        use_attention: 已弃用，保留参数兼容性
        device: 设备
        dtype: 数据类型
    
    Returns:
        ValueNetwork实例
    """
    network = ValueNetwork(hidden_size, value_hidden_dim)
    
    if dtype is not None:
        network = network.to(dtype)
    return network.to(device)
