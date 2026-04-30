"""
扰动生成器模块 (优化版)
负责根据输入的hidden states生成扰动
简化网络结构和计算逻辑以提升性能

扰动逻辑:
  Step 1: TokenSelector 选择要扰动的 token 索引
  Step 2: PerturbationDimSelector 对每个选中 token，选择要被置0的维度子集
  最终扰动: 选中 token 的选中维度置0，其余维度保持原值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import PerturbationConfig


class TokenSelector(nn.Module):
    """
    Token选择器 (简化版)
    使用单层网络快速计算token重要性分数
    """
    
    def __init__(self, hidden_size: int, hidden_dim: int = 64):
        super().__init__()
        # 简化为两层网络，减少计算量
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                num_select: int,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，选择要扰动的token
        """
        batch_size, seq_len, _ = hidden_states.shape
        actual_select = min(num_select, seq_len)
        # print(f"Selecting {actual_select} tokens from {seq_len} tokens")
        
        # 计算分数
        scores = self.scorer(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        # scores = scores.float()
        # print(scores)
        
        if deterministic:
            _, selected_indices = torch.topk(scores, actual_select, dim=-1)
            selection_log_probs = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.float32)
        else:
            selected_indices, selection_log_probs = self._gumbel_topk_sample(scores, actual_select)
        
        return selected_indices, selection_log_probs
    
    def _gumbel_topk_sample(self, scores: torch.Tensor, 
                            num_select: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Gumbel-top-k技巧进行快速无放回采样
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores).clamp(min=1e-6, max=1-1e-6)))
        perturbed_scores = scores + gumbel_noise
        # perturbed_scores = scores
        
        _, selected_indices = torch.topk(perturbed_scores, num_select, dim=-1)
        
        log_probs = F.log_softmax(scores, dim=-1)
        
        batch_indices = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
        # selected_log_probs = log_probs[batch_indices, selected_indices].sum(dim=-1)
        selected_log_probs = log_probs[batch_indices, selected_indices].mean(dim=-1)
        
        return selected_indices, selected_log_probs


class PerturbationDimSelector(nn.Module):
    """
    扰动维度选择器
    对每个选中的 token，使用网络打分后通过 Gumbel-top-k 采样选出要被置0的维度子集
    """
    
    def __init__(self, hidden_size: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        # 对每个维度打分: hidden_size -> hidden_size
        self.dim_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_size)
        )
    
    def forward(self, selected_hidden_states: torch.Tensor,
                num_perturb_dims: int,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，为每个选中 token 选择要被置0的维度
        
        Args:
            selected_hidden_states: (batch_size, num_select, hidden_size)
            num_perturb_dims: 每个 token 被置0的维度数量
            deterministic: 是否确定性选择（top-k，不加噪声）
        
        Returns:
            perturb_dim_indices: (batch_size, num_select, num_perturb_dims) 被置0的维度索引
            dim_log_probs:        (batch_size,) 对数概率（各 token 取均值后再对 batch 取均值）
        """
        batch_size, num_select, _ = selected_hidden_states.shape
        actual_dims = min(num_perturb_dims, self.hidden_size)
        
        # (batch_size, num_select, hidden_size)
        dim_scores = self.dim_scorer(selected_hidden_states)
        
        if deterministic:
            _, perturb_dim_indices = torch.topk(dim_scores, actual_dims, dim=-1)
            dim_log_probs = torch.zeros(batch_size, device=selected_hidden_states.device, dtype=torch.float32)
        else:
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(dim_scores).clamp(min=1e-6, max=1 - 1e-6)
            ))
            perturbed_scores = dim_scores + gumbel_noise
            _, perturb_dim_indices = torch.topk(perturbed_scores, actual_dims, dim=-1)
            
            log_probs = F.log_softmax(dim_scores, dim=-1)  # (batch_size, num_select, hidden_size)
            # 收集选中维度的 log_prob，再对维度和 token 取均值 -> (batch_size,)
            selected_dim_log_probs = log_probs.gather(-1, perturb_dim_indices)  # (batch_size, num_select, actual_dims)
            dim_log_probs = selected_dim_log_probs.mean(dim=[-1, -2])           # (batch_size,)
        
        return perturb_dim_indices, dim_log_probs


class PerturbationGenerator(nn.Module):
    """
    扰动生成器 (Actor网络) - 优化版
    整合Token选择器和扰动维度选择器

    扰动方式: 对每个选中 token，将选中维度置0，其余维度保持原值
    """
    
    def __init__(self, config: PerturbationConfig, hidden_size: int):
        super().__init__()
        
        self.config = config
        self.hidden_size = hidden_size
        self.num_perturb_tokens = config.num_perturb_tokens
        self.num_perturb_dims = config.num_perturb_dims
        
        # 子模块
        self.token_selector = TokenSelector(
            hidden_size=hidden_size,
            hidden_dim=config.selector_hidden_dim
        )
        
        self.dim_selector = PerturbationDimSelector(
            hidden_size=hidden_size,
            hidden_dim=config.dim_selector_hidden_dim
        )
    
    def forward(self, hidden_states: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        生成扰动

        对每个选中 token 的选中维度置0，其余维度保持原值
        """
        # print(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Step 1: 选择要扰动的token
        selected_indices, selection_log_prob = self.token_selector(
            hidden_states, self.num_perturb_tokens, deterministic
        )
        
        actual_num_select = selected_indices.shape[1]
        
        # Step 2: 获取选中token的hidden states
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, actual_num_select)
        selected_indices_clamped = torch.clamp(selected_indices, min=0, max=seq_len-1)
        selected_hidden = hidden_states[batch_indices, selected_indices_clamped]
        
        # Step 3: 为每个选中token选择要被置0的维度
        perturb_dim_indices, dim_log_prob = self.dim_selector(
            selected_hidden, self.num_perturb_dims, deterministic
        )
        # perturb_dim_indices: (batch_size, actual_num_select, actual_num_perturb_dims)
        
        # Step 4: 构建扰动后的hidden states（选中维度置0，其余保持原值）
        perturbed_hidden_states = hidden_states.clone()
        # 取出选中token的切片: (batch_size, actual_num_select, hidden_size)
        selected_tokens = perturbed_hidden_states[batch_indices, selected_indices_clamped]
        # 将选中维度置0
        selected_tokens.scatter_(-1, perturb_dim_indices, 0.0)
        # 写回
        perturbed_hidden_states[batch_indices, selected_indices_clamped] = selected_tokens
        
        total_log_prob = selection_log_prob + dim_log_prob
        return {
            'perturbed_hidden_states': perturbed_hidden_states.half(),
            'selected_indices': selected_indices,
            'perturb_dim_indices': perturb_dim_indices,
            'log_prob': total_log_prob,
        }
    
    def get_log_prob(self, hidden_states: torch.Tensor,
                     selected_indices: torch.Tensor,
                     perturb_dim_indices: torch.Tensor) -> torch.Tensor:
        """
        计算给定动作的对数概率（用于PPO更新）

        Args:
            hidden_states:      (batch_size, seq_len, hidden_size)
            selected_indices:   (batch_size, num_select)  token 索引
            perturb_dim_indices:(batch_size, num_select, num_perturb_dims)  维度索引
        
        数值稳定性处理：
        1. 用 nan_to_num 处理 NaN/Inf（clamp 对 NaN 无效！）
        2. 对 scores/logits 进行裁剪，防止 softmax 前输入极端值
        3. 对 log_prob 进行裁剪，防止 -inf
        """
        hidden_states = hidden_states.to(torch.float32)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # 数值稳定性常量
        SCORE_CLAMP = 50.0      # scores 裁剪范围 [-50, 50]
        LOG_PROB_MIN = -20.0    # log_prob 最小值（防止 -inf）
        
        selected_indices = selected_indices.long()
        perturb_dim_indices = perturb_dim_indices.long()
        actual_num_select = selected_indices.shape[1]
        
        # 处理 hidden_states 中的 NaN/Inf
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 获取token选择的log_prob
        scores = self.token_selector.scorer(hidden_states).squeeze(-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=SCORE_CLAMP, neginf=-SCORE_CLAMP)
        scores = torch.clamp(scores, min=-SCORE_CLAMP, max=SCORE_CLAMP)
        
        log_probs = F.log_softmax(scores, dim=-1)
        log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN)
        
        selected_indices_clamped = torch.clamp(selected_indices, min=0, max=seq_len-1)
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        selection_log_prob = log_probs[batch_indices, selected_indices_clamped].mean(dim=-1)
        
        # 获取选中token的hidden states
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, actual_num_select)
        selected_hidden = hidden_states[batch_idx, selected_indices_clamped]
        
        # 获取维度选择的log_prob
        dim_scores = self.dim_selector.dim_scorer(selected_hidden)  # (batch_size, num_select, hidden_size)
        dim_scores = torch.nan_to_num(dim_scores, nan=0.0, posinf=SCORE_CLAMP, neginf=-SCORE_CLAMP)
        dim_scores = torch.clamp(dim_scores, min=-SCORE_CLAMP, max=SCORE_CLAMP)
        
        dim_log_probs = F.log_softmax(dim_scores, dim=-1)           # (batch_size, num_select, hidden_size)
        dim_log_probs = torch.clamp(dim_log_probs, min=LOG_PROB_MIN)
        
        actual_num_dims = perturb_dim_indices.shape[-1]
        perturb_dim_indices_clamped = torch.clamp(perturb_dim_indices, min=0, max=self.hidden_size - 1)
        # (batch_size, num_select, actual_num_dims) -> mean over dims and tokens -> (batch_size,)
        dim_log_prob = dim_log_probs.gather(-1, perturb_dim_indices_clamped).mean(dim=[-1, -2])
        
        result = selection_log_prob + dim_log_prob
        result = torch.clamp(result, min=2 * LOG_PROB_MIN)
        
        return result


def create_perturbation_generator(config: PerturbationConfig, 
                                   hidden_size: int,
                                   device: str = 'cuda',
                                   dtype: torch.dtype = None) -> PerturbationGenerator:
    """
    创建扰动生成器的工厂函数
    
    Args:
        config: 扰动配置
        hidden_size: 模型隐藏层维度
        device: 设备
        dtype: 数据类型（如 torch.float16），如果为 None 则使用默认的 float32
    
    Returns:
        PerturbationGenerator实例
    """
    generator = PerturbationGenerator(config, hidden_size)
    generator = generator.to(device)
    if dtype is not None:
        generator = generator.to(dtype)
    return generator
