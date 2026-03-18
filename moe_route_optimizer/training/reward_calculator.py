"""
奖励计算模块
根据通信时延和精度计算强化学习奖励
"""

import torch
from typing import Dict, Optional, Tuple

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import RewardConfig, get_train_logger


class RewardCalculator:
    """
    奖励计算器
    综合通信时延减少和精度保持来计算奖励
    """
    
    def __init__(self, config: RewardConfig):
        """
        Args:
            config: 奖励配置
        """
        self.config = config
        self.logger = get_train_logger()
        
        # 基线值（需要在训练开始前设置）
        self.baseline_latency: Optional[float] = None
        self.baseline_accuracy: Optional[float] = None
        
        # 历史记录
        self._latency_history = []
        self._accuracy_history = []
        self._reward_history = []
    
    def set_baseline(self, latency: float, accuracy: float):
        """
        设置基线值
        
        Args:
            latency: 无扰动时的基线通信时延
            accuracy: 无扰动时的基线精度
        """
        self.baseline_latency = latency
        self.baseline_accuracy = accuracy
        self.logger.info(f"Baseline set - Latency: {latency:.4f}s, Accuracy: {accuracy*100:.2f}%")
    
    def compute(self, comm_delay: float, accuracy: float) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励
        
        Args:
            comm_delay: 当前的通信时延
            accuracy: 当前的推理精度
        
        Returns:
            total_reward: 总奖励
            reward_components: 奖励各组成部分 {"latency": ..., "accuracy": ..., "penalty": ...}
        """
        if self.baseline_latency is None or self.baseline_accuracy is None:
            self.logger.warning("Baseline not set, using default values")
            self.baseline_latency = comm_delay
            self.baseline_accuracy = accuracy
        
        # 1. 时延奖励：时延减少越多，奖励越高
        latency_reduction = (self.baseline_latency - comm_delay) / (self.baseline_latency + 1e-10)
        latency_reward = latency_reduction  # 范围大约在 [-1, 1]
        
        # 2. 精度奖励：精度保持得越好，奖励越高
        accuracy_ratio = accuracy / (self.baseline_accuracy + 1e-10)
        accuracy_reward = accuracy_ratio - 1.0  # 0 表示与基线相同，负值表示下降
        
        # 3. 精度惩罚：如果精度低于阈值，额外惩罚
        penalty = 0.0
        accuracy_threshold = self.baseline_accuracy * self.config.accuracy_penalty_threshold
        if accuracy < accuracy_threshold:
            penalty = -self.config.accuracy_penalty_coef * (accuracy_threshold - accuracy)
        
        # 4. 计算总奖励
        total_reward = (
            self.config.latency_weight * latency_reward +
            self.config.accuracy_weight * accuracy_reward +
            penalty
        )
        
        # 记录历史
        self._latency_history.append(latency_reduction)
        self._accuracy_history.append(accuracy)
        self._reward_history.append(total_reward)
        
        reward_components = {
            'latency_reduction': latency_reduction,
            'latency_reward': latency_reward,
            'accuracy': accuracy,
            'accuracy_ratio': accuracy_ratio,
            'accuracy_reward': accuracy_reward,
            'penalty': penalty,
            'total_reward': total_reward,
        }
        
        return total_reward, reward_components
    
    def compute_batch(self, comm_delays: torch.Tensor, 
                      accuracies: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        批量计算奖励
        
        Args:
            comm_delays: (batch_size,) 通信时延
            accuracies: (batch_size,) 精度
        
        Returns:
            rewards: (batch_size,) 总奖励
            components: 各组成部分
        """
        batch_size = comm_delays.shape[0]
        
        if self.baseline_latency is None:
            self.baseline_latency = comm_delays.mean().item()
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracies.mean().item()
        
        # 时延奖励
        latency_reduction = (self.baseline_latency - comm_delays) / (self.baseline_latency + 1e-10)
        latency_reward = latency_reduction
        
        # 精度奖励
        accuracy_ratio = accuracies / (self.baseline_accuracy + 1e-10)
        accuracy_reward = accuracy_ratio - 1.0
        
        # 精度惩罚
        accuracy_threshold = self.baseline_accuracy * self.config.accuracy_penalty_threshold
        penalty = torch.zeros_like(accuracies)
        below_threshold = accuracies < accuracy_threshold
        penalty[below_threshold] = -self.config.accuracy_penalty_coef * (
            accuracy_threshold - accuracies[below_threshold]
        )
        
        # 总奖励
        total_reward = (
            self.config.latency_weight * latency_reward +
            self.config.accuracy_weight * accuracy_reward +
            penalty
        )
        
        components = {
            'latency_reduction': latency_reduction,
            'latency_reward': latency_reward,
            'accuracy_ratio': accuracy_ratio,
            'accuracy_reward': accuracy_reward,
            'penalty': penalty,
        }
        
        return total_reward, components
    
    def get_statistics(self) -> Dict[str, float]:
        """获取历史统计信息"""
        if not self._reward_history:
            return {}
        
        return {
            'avg_reward': sum(self._reward_history) / len(self._reward_history),
            'avg_latency_reduction': sum(self._latency_history) / len(self._latency_history),
            'avg_accuracy': sum(self._accuracy_history) / len(self._accuracy_history),
            'max_reward': max(self._reward_history),
            'min_reward': min(self._reward_history),
        }
    
    def reset_history(self):
        """重置历史记录"""
        self._latency_history.clear()
        self._accuracy_history.clear()
        self._reward_history.clear()


def create_reward_calculator(config: RewardConfig) -> RewardCalculator:
    """创建奖励计算器"""
    return RewardCalculator(config)
