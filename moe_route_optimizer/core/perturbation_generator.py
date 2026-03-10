"""
扰动生成器模块 (优化版)
负责根据输入的hidden states生成扰动
简化网络结构和计算逻辑以提升性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import sys
sys.path.append('/mnt/data/lwy/vLLM-wrok/moe_route_optimizer')

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


class PerturbationTypeDecider(nn.Module):
    """
    扰动类型决策器 (简化版)
    为每个选中的token决定扰动类型 (-1, 0, +1)
    """
    
    def __init__(self, hidden_size: int, hidden_dim: int = 32, num_types: int = 3):
        super().__init__()
        self.num_types = num_types
        # 简化为单层网络
        self.type_predictor = nn.Linear(hidden_size, num_types)
    
    def forward(self, selected_hidden_states: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，决定扰动类型
        """
        batch_size, num_select, _ = selected_hidden_states.shape
        
        logits = self.type_predictor(selected_hidden_states)
        # logits = logits.float()
        
        if deterministic:
            perturb_types = logits.argmax(dim=-1)
            type_log_probs = torch.zeros(batch_size, device=selected_hidden_states.device, dtype=torch.float32)
        else:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-6, max=1-1e-6)))
            perturb_types = (logits + gumbel_noise).argmax(dim=-1)
            # perturb_types = logits.argmax(dim=-1)
            
            log_probs = F.log_softmax(logits, dim=-1)
            # print(f"log_probs: mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f}")
            # print(log_probs)
            # type_log_probs = log_probs.gather(-1, perturb_types.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            type_log_probs = log_probs.gather(-1, perturb_types.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        
        return perturb_types, type_log_probs


class PerturbationGenerator(nn.Module):
    """
    扰动生成器 (Actor网络) - 优化版
    整合Token选择器和扰动类型决策器
    """
    
    def __init__(self, config: PerturbationConfig, hidden_size: int):
        super().__init__()
        
        self.config = config
        self.hidden_size = hidden_size
        self.num_perturb_tokens = config.num_perturb_tokens
        self.perturbation_scale = config.perturbation_scale
        
        # 扰动类型映射: 0 -> -1, 1 -> 0, 2 -> +1
        self.register_buffer('type_values', torch.tensor([-1.0, 0.0, 1.0]))
        
        # 子模块
        self.token_selector = TokenSelector(
            hidden_size=hidden_size,
            hidden_dim=config.selector_hidden_dim
        )
        
        self.type_decider = PerturbationTypeDecider(
            hidden_size=hidden_size,
            hidden_dim=config.type_decider_hidden_dim,
            num_types=config.num_perturb_types
        )
    
    def forward(self, hidden_states: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        生成扰动
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
        
        # Step 3: 决定扰动类型
        perturb_types, type_log_prob = self.type_decider(selected_hidden, deterministic)
        
        # Step 4: 向量化应用扰动
        perturbed_hidden_states = hidden_states.clone()
        perturb_values = self.type_values[perturb_types] * self.perturbation_scale
        
        expanded_values = perturb_values.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        perturbed_hidden_states.scatter_(1, selected_indices_clamped.unsqueeze(-1).expand(-1, -1, self.hidden_size), expanded_values)
        
        total_log_prob = selection_log_prob + type_log_prob
        # print(f"total_log_prob: mean={total_log_prob.mean().item():.6f}, std={total_log_prob.std().item():.6f}, shape={total_log_prob.shape}")
        # print(total_log_prob)
        return {
            'perturbed_hidden_states': perturbed_hidden_states.half(),
            'selected_indices': selected_indices,
            'perturb_types': perturb_types,
            'log_prob': total_log_prob,
        }
    
    def get_log_prob(self, hidden_states: torch.Tensor,
                     selected_indices: torch.Tensor,
                     perturb_types: torch.Tensor) -> torch.Tensor:
        """
        计算给定动作的对数概率（用于PPO更新）
        
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
        perturb_types = perturb_types.long()
        actual_num_select = selected_indices.shape[1]
        
        # 处理 hidden_states 中的 NaN/Inf
        # print(f"hidden_states: max={hidden_states.max().item():.6f}, min={hidden_states.min().item():.6f}")
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 获取token选择的log_prob
        scores = self.token_selector.scorer(hidden_states).squeeze(-1)
        # scores = scores.float()
        # print(f"scores: max={scores.max().item():.6f}, min={scores.min().item():.6f}")
        # 关键：先用 nan_to_num 处理 NaN（clamp 对 NaN 无效！），再裁剪
        scores = torch.nan_to_num(scores, nan=0.0, posinf=SCORE_CLAMP, neginf=-SCORE_CLAMP)
        scores = torch.clamp(scores, min=-SCORE_CLAMP, max=SCORE_CLAMP)
        
        log_probs = F.log_softmax(scores, dim=-1)
        # print(f"log_probs: max={log_probs.max().item():.6f}, min={log_probs.min().item():.6f}")
        log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN)
        
        selected_indices_clamped = torch.clamp(selected_indices, min=0, max=seq_len-1)
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        selection_log_prob = log_probs[batch_indices, selected_indices_clamped].mean(dim=-1)
        
        # 获取选中token的hidden states
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, actual_num_select)
        selected_hidden = hidden_states[batch_idx, selected_indices_clamped]
        
        # 获取类型选择的log_prob
        type_logits = self.type_decider.type_predictor(selected_hidden)
        # type_logits = type_logits.float()
        # print(f"type_logits: max={type_logits.max().item():.6f}, min={type_logits.min().item():.6f}")
        # 关键：先用 nan_to_num 处理 NaN，再裁剪
        type_logits = torch.nan_to_num(type_logits, nan=0.0, posinf=SCORE_CLAMP, neginf=-SCORE_CLAMP)
        type_logits = torch.clamp(type_logits, min=-SCORE_CLAMP, max=SCORE_CLAMP)
        
        type_log_probs = F.log_softmax(type_logits, dim=-1)
        # print(f"type_log_probs: max={type_log_probs.max().item():.6f}, min={type_log_probs.min().item():.6f}")
        type_log_probs = torch.clamp(type_log_probs, min=LOG_PROB_MIN)
        
        perturb_types_clamped = torch.clamp(perturb_types, min=0, max=self.type_decider.num_types-1)
        type_log_prob = type_log_probs.gather(-1, perturb_types_clamped.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        
        result = selection_log_prob + type_log_prob
        result = torch.clamp(result, min=2 * LOG_PROB_MIN)

        # 打印result的最大和最小值
        # print(f"result: max={result.max().item():.6f}, min={result.min().item():.6f}")
        
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
