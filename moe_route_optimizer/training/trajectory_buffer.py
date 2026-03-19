"""
轨迹缓冲区模块
存储强化学习训练所需的经验数据（简化版，无需value）
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


class RolloutBuffer:
    """
    简化版经验缓冲区
    用于on-policy算法，每次更新后清空
    移除了value相关的数据收集
    """
    
    def __init__(self):
        self.hidden_states: List[torch.Tensor] = []
        self.selected_indices: List[torch.Tensor] = []
        self.perturb_dim_indices: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
    
    def add(self, hidden_states: torch.Tensor,
            selected_indices: torch.Tensor,
            perturb_dim_indices: torch.Tensor,
            log_prob: torch.Tensor,
            reward: float,
            done: bool = False,
            **kwargs):  # 忽略额外参数（如value）以保持兼容性
        """添加一条经验"""
        self.hidden_states.append(hidden_states.detach())
        self.selected_indices.append(selected_indices.detach())
        self.perturb_dim_indices.append(perturb_dim_indices.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get_batch(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        获取所有数据作为批次
        
        数据形状约定（与 perturbation_generator 一致）：
        - hidden_states: (batch, seq_len, hidden) - 3D
        - selected_indices: (batch, num_select) - 2D
        - perturb_dim_indices: (batch, num_select, num_perturb_dims) - 3D
        - log_prob: (batch,) - 1D
        
        简化：直接 stack，不做复杂的维度处理。
        """
        if not self.hidden_states:
            return {}
        
        # 直接 stack，假设数据形状一致
        # hidden_states: List[(batch, seq, hidden)] -> (num_exp, batch, seq, hidden)
        # 对于单条经验，取第一个元素后保持形状
        if len(self.hidden_states) == 1:
            return {
                'hidden_states': self.hidden_states[0].to(device),  # (batch, seq_len, hidden)
                'seq_lengths': torch.tensor([self.hidden_states[0].shape[1]], device=device),
                'selected_indices': self.selected_indices[0].to(device),  # (batch, num_select)
                'perturb_dim_indices': self.perturb_dim_indices[0].to(device),  # (batch, num_select)
                'log_probs': self.log_probs[0].to(device),  # (batch,)
                'rewards': torch.tensor(self.rewards, device=device),
            }
        
        # 多条经验：简单 stack
        return {
            'hidden_states': torch.stack(self.hidden_states).to(device),
            'seq_lengths': torch.tensor([h.shape[1] for h in self.hidden_states], device=device),
            'selected_indices': torch.stack(self.selected_indices).to(device),
            'perturb_dim_indices': torch.stack(self.perturb_dim_indices).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.tensor(self.rewards, device=device),
        }
    
    def clear(self):
        """清空缓冲区"""
        self.hidden_states.clear()
        self.selected_indices.clear()
        self.perturb_dim_indices.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.rewards)
