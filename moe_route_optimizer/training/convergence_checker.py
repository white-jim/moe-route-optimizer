"""
收敛判断模块
判断训练是否成功收敛或需要提前终止
"""

from typing import Dict, Tuple, Optional
from collections import deque
import numpy as np

import sys
sys.path.append('/mnt/data/lwy/vLLM-wrok/moe_route_optimizer')

from config import ConvergenceConfig, get_train_logger


class ConvergenceChecker:
    """
    收敛判断器
    监控训练过程，判断是否成功收敛或需要终止
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        Args:
            config: 收敛判断配置
        """
        self.config = config
        self.logger = get_train_logger()
        
        # 基线值
        self.baseline_latency: Optional[float] = None
        self.baseline_accuracy: Optional[float] = None
        
        # 历史记录（使用deque限制长度）
        self._reward_history = deque(maxlen=config.reward_window_size)
        self._latency_reduction_history = deque(maxlen=config.reward_window_size)
        self._accuracy_history = deque(maxlen=config.reward_window_size)
        
        # 计数器
        self._episode_count = 0
        self._success_count = 0  # 连续满足成功条件的episode数
        self._accuracy_decline_count = 0  # 精度持续下降的计数
        self._stagnation_count = 0  # 时延无改善的计数
        
        # 最佳记录
        self._best_reward = float('-inf')
        self._best_latency_reduction = 0.0
        self._best_accuracy = 0.0
    
    def set_baseline(self, latency: float, accuracy: float):
        """
        设置基线值
        
        Args:
            latency: 基线通信时延
            accuracy: 基线精度
        """
        self.baseline_latency = latency
        self.baseline_accuracy = accuracy
        self.logger.info(f"Convergence checker baseline set - "
                        f"Latency: {latency:.4f}s, Accuracy: {accuracy*100:.2f}%")
    
    def update(self, reward: float, latency_reduction: float, 
               accuracy: float) -> Tuple[bool, bool, str]:
        """
        更新状态并检查是否应该终止
        
        Args:
            reward: 当前episode的奖励
            latency_reduction: 时延减少比例
            accuracy: 当前精度
        
        Returns:
            should_stop: 是否应该停止训练
            success: 是否成功收敛
            reason: 终止原因
        """
        self._episode_count += 1
        
        # 记录历史
        self._reward_history.append(reward)
        self._latency_reduction_history.append(latency_reduction)
        self._accuracy_history.append(accuracy)
        
        # 更新最佳记录
        if reward > self._best_reward:
            self._best_reward = reward
        if latency_reduction > self._best_latency_reduction:
            self._best_latency_reduction = latency_reduction
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
        
        # 检查各种终止条件
        should_stop, success, reason = self._check_termination()
        
        return should_stop, success, reason
    
    def _check_termination(self) -> Tuple[bool, bool, str]:
        """检查终止条件"""
        
        # 条件1: 达到最大episode数
        if self._episode_count >= self.config.max_episodes:
            return True, False, "Reached maximum episodes"
        
        # 如果历史数据不足，继续训练
        if len(self._reward_history) < self.config.reward_window_size:
            return False, False, ""
        
        # 计算窗口内的统计量
        avg_latency_reduction = np.mean(list(self._latency_reduction_history))
        avg_accuracy = np.mean(list(self._accuracy_history))
        reward_variance = np.var(list(self._reward_history))
        
        # 条件2: 成功收敛判断
        latency_success = avg_latency_reduction >= self.config.latency_reduction_threshold
        accuracy_success = (self.baseline_accuracy is None or 
                           avg_accuracy >= self.baseline_accuracy * self.config.accuracy_maintain_threshold)
        reward_stable = reward_variance < self.config.reward_variance_threshold
        
        if latency_success and accuracy_success and reward_stable:
            self._success_count += 1
            if self._success_count >= self.config.success_patience:
                return True, True, (
                    f"Converged: Latency↓ {avg_latency_reduction*100:.2f}% >= {self.config.latency_reduction_threshold*100:.0f}%, "
                    f"Accuracy {avg_accuracy*100:.2f}% maintained"
                )
        else:
            self._success_count = 0
        
        # 条件3: 精度持续下降 - 早停
        if self.baseline_accuracy is not None:
            if avg_accuracy < self.baseline_accuracy * 0.8:  # 精度下降超过20%
                self._accuracy_decline_count += 1
                if self._accuracy_decline_count >= self.config.accuracy_early_stop_patience:
                    return True, False, f"Early stop: Accuracy dropped to {avg_accuracy*100:.2f}%"
            else:
                self._accuracy_decline_count = 0
        
        # 条件4: 时延无改善 - 停滞
        if avg_latency_reduction <= 0:
            self._stagnation_count += 1
            if self._stagnation_count >= self.config.stagnation_patience:
                return True, False, "Stagnation: No latency improvement"
        else:
            self._stagnation_count = 0
        
        return False, False, ""
    
    def should_evaluate(self) -> bool:
        """判断是否应该进行评估"""
        return self._episode_count % self.config.eval_interval == 0
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        if not self._reward_history:
            return {}
        
        return {
            'episode': self._episode_count,
            'avg_reward': np.mean(list(self._reward_history)),
            'avg_latency_reduction': np.mean(list(self._latency_reduction_history)),
            'avg_accuracy': np.mean(list(self._accuracy_history)),
            'reward_variance': np.var(list(self._reward_history)),
            'best_reward': self._best_reward,
            'best_latency_reduction': self._best_latency_reduction,
            'best_accuracy': self._best_accuracy,
            'success_streak': self._success_count,
        }
    
    def get_final_metrics(self) -> Dict[str, float]:
        """获取最终指标（训练结束时）"""
        metrics = self.get_current_metrics()
        metrics['total_episodes'] = self._episode_count
        
        if self.baseline_latency is not None:
            metrics['baseline_latency'] = self.baseline_latency
        if self.baseline_accuracy is not None:
            metrics['baseline_accuracy'] = self.baseline_accuracy
        
        return metrics
    
    def reset(self):
        """重置检查器状态"""
        self._reward_history.clear()
        self._latency_reduction_history.clear()
        self._accuracy_history.clear()
        self._episode_count = 0
        self._success_count = 0
        self._accuracy_decline_count = 0
        self._stagnation_count = 0
        self._best_reward = float('-inf')
        self._best_latency_reduction = 0.0
        self._best_accuracy = 0.0


def create_convergence_checker(config: ConvergenceConfig) -> ConvergenceChecker:
    """创建收敛判断器"""
    return ConvergenceChecker(config)
