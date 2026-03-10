"""
扰动生成器模块 (简化版)
只使用单个MLP进行token选择，扰动类型固定，避免梯度不稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import sys
sys.path.append('/mnt/data/lwy/vLLM-wrok/moe_route_optimizer')

from config import PerturbationConfig


class TokenSelectorNetwork(nn.Module):
    """
    Token选择网络
    使用单个MLP只输出token选择分数
    """
    
    def __init__(self, hidden_size: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Token选择网络
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            selection_logits: (batch_size, seq_len) - 每个token的选择logits
        """
        # 计算选择分数
        selection_logits = self.scorer(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        return selection_logits


class PerturbationGenerator(nn.Module):
    """
    扰动生成器 (Actor网络) - 简化版本
    只使用单个MLP进行token选择，扰动类型固定为0（对应type_values[1]=0.0）
    """
    
    def __init__(self, config: PerturbationConfig, hidden_size: int):
        super().__init__()
        
        self.config = config
        self.hidden_size = hidden_size
        self.num_perturb_tokens = config.num_perturb_tokens
        self.perturbation_scale = config.perturbation_scale
        self.num_types = config.num_perturb_types
        
        # 扰动类型映射: 0 -> -1, 1 -> 0, 2 -> +1
        # 固定使用类型1（对应值0.0）
        self.register_buffer('type_values', torch.tensor([-1.0, 0.0, 1.0]))
        self.fixed_type = 1  # 固定使用中间类型
        
        # Token选择网络（只有一个MLP）
        self.selector_network = TokenSelectorNetwork(
            hidden_size=hidden_size,
            hidden_dim=config.selector_hidden_dim
        )
    
    def forward(self, hidden_states: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        生成扰动
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            deterministic: 是否使用确定性策略
        
        Returns:
            字典包含:
            - perturbed_hidden_states: 扰动后的hidden states
            - selected_indices: 选中的token索引
            - perturb_types: 对应的扰动类型（为了兼容性，全部填充为固定值）
            - log_prob: 对数概率（只来自token选择）
        """
        # 转为全精度
        hidden_states = hidden_states.to(torch.float32)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # 获取选择logits（只有一个MLP的输出）
        selection_logits = self.selector_network(hidden_states)
        # selection_logits = selection_logits.float()
        
        actual_num_select = min(self.num_perturb_tokens, seq_len)
        
        # 选择token
        if deterministic:
            _, selected_indices = torch.topk(selection_logits, actual_num_select, dim=-1)
            log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            # Gumbel-top-k 采样
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(selection_logits)))
            perturbed_scores = selection_logits + gumbel_noise
            _, selected_indices = torch.topk(perturbed_scores, actual_num_select, dim=-1)
            
            # 计算选择的log_prob（只有这一项）
            log_probs_all = F.log_softmax(selection_logits, dim=-1)
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            log_probs = log_probs_all[batch_indices, selected_indices].mean(dim=-1)
        
        # 扰动类型固定为1（对应type_values[1]=0.0）
        perturb_types = torch.full((batch_size, actual_num_select), self.fixed_type, 
                                   dtype=torch.long, device=device)
        
        # 应用扰动
        perturbed_hidden_states = hidden_states.clone()
        selected_indices_clamped = torch.clamp(selected_indices, min=0, max=seq_len-1)
        perturb_values = self.type_values[perturb_types] * self.perturbation_scale
        
        expanded_values = perturb_values.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        perturbed_hidden_states.scatter_(1, selected_indices_clamped.unsqueeze(-1).expand(-1, -1, self.hidden_size), expanded_values)
        
        # 计算结果需要转回半精度
        # return {
        #     'perturbed_hidden_states': perturbed_hidden_states,
        #     'selected_indices': selected_indices,
        #     'perturb_types': perturb_types,
        #     'log_prob': log_probs,
        # }
        return {
            'perturbed_hidden_states': perturbed_hidden_states.half(),
            'selected_indices': selected_indices,
            'perturb_types': perturb_types,
            'log_prob': log_probs,
        }
    
    def get_log_prob(self, hidden_states: torch.Tensor,
                     selected_indices: torch.Tensor,
                     perturb_types: torch.Tensor) -> torch.Tensor:
        """
        计算给定动作的对数概率（用于策略梯度更新）
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            selected_indices: (batch_size, num_select) - 选中的token索引
            perturb_types: (batch_size, num_select) - 对应的扰动类型（为兼容性保留，实际不使用）
        
        Returns:
            log_prob: (batch_size,) - 每个样本的对数概率（只来自token选择）
        """
        hidden_states = hidden_states.to(torch.float32)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # 数值稳定性常量
        LOGITS_CLAMP = 50.0
        LOG_PROB_MIN = -20.0
        
        selected_indices = selected_indices.long()
        
        # 处理hidden_states中的NaN/Inf
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 获取选择logits（只有一个MLP的输出）
        selection_logits = self.selector_network(hidden_states)
        # selection_logits = selection_logits.float()
        
        # 数值稳定性处理
        selection_logits = torch.nan_to_num(selection_logits, nan=0.0, posinf=LOGITS_CLAMP, neginf=-LOGITS_CLAMP)
        
        # 计算选择的log_prob（只有这一项，没有类型log_prob）
        log_probs_all = F.log_softmax(selection_logits, dim=-1)
        
        selected_indices_clamped = torch.clamp(selected_indices, min=0, max=seq_len-1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        log_probs = log_probs_all[batch_indices, selected_indices_clamped].mean(dim=-1)
        
        return log_probs
        # return log_probs.half()


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
