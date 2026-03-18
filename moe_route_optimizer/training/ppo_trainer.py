"""
策略梯度训练器模块
实现简化的REINFORCE算法（只有Actor，无Critic）
支持分布式环境（梯度同步 + 只有rank 0保存模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple
import os

import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import PPOConfig, get_train_logger, is_main_process
from core.perturbation_generator import PerturbationGenerator
from core.value_network import ValueNetwork, create_value_network
from training.trajectory_buffer import RolloutBuffer

print_num = 1
class PolicyGradientTrainer:
    """
    策略梯度训练器（简化版REINFORCE）
    只包含Actor（扰动生成器），无Critic
    支持分布式梯度同步
    
    修正：每推理一个batch收集一条经验，立即更新参数
    使用移动平均reward作为baseline来减少方差
    """
    
    def __init__(self, 
                 actor: PerturbationGenerator,
                 config: PPOConfig,
                 device: str = 'cuda',
                 value_network: Optional[nn.Module] = None):
        """
        Args:
            actor: 扰动生成器（策略网络）
            config: 训练配置
            device: 训练设备
            value_network: 价值网络（Critic），用于PPO更新。为None时只能使用REINFORCE更新。
        """
        self.actor = actor
        self.config = config
        self.device = device
        self.logger = get_train_logger()
        
        # Actor优化器
        self.actor_optimizer = Adam(actor.parameters(), lr=config.actor_lr)
        
        # Critic（价值网络）及其优化器
        self.value_network = value_network
        self.critic_optimizer = None
        if value_network is not None:
            self.critic_optimizer = Adam(value_network.parameters(), lr=config.critic_lr)
            self.logger.info(f"PPO mode enabled: value_network params={sum(p.numel() for p in value_network.parameters())}")
        
        # 经验缓冲区（简化版）
        self.rollout_buffer = RolloutBuffer()
        
        # 训练统计
        self._update_count = 0
        self._actor_loss_history = []
        self._critic_loss_history = []
        
        # 移动平均baseline（用于REINFORCE单条经验的情况）
        self._reward_baseline = 0.0
        self._baseline_momentum = 0.9  # baseline更新动量
    
    def collect_experience(self, 
                           hidden_states: torch.Tensor,
                           reward: float,
                           done: bool = False,
                           log_prob: torch.Tensor = None,
                           selected_indices: torch.Tensor = None,
                           perturb_types: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        收集一步经验（batch级别）
        
        数据形状约定（与 perturbation_generator 一致）：
        - hidden_states: (batch, seq_len, hidden) - 3D
        - selected_indices: (batch, num_select) - 2D
        - perturb_types: (batch, num_select) - 2D
        - log_prob: (batch,) - 1D
        
        直接存储原始形状，不做维度调整。
        """
        # 直接存储原始数据，保持形状一致
        self.rollout_buffer.add(
            hidden_states=hidden_states,
            selected_indices=selected_indices,
            perturb_types=perturb_types,
            log_prob=log_prob,
            reward=reward,
            done=done,
        )
        return {
            'log_prob': log_prob,
            'selected_indices': selected_indices,
            'perturb_types': perturb_types,
        }
    
    def _sync_gradients(self):
        """
        修复问题1: 在分布式环境中同步所有rank的梯度
        使用all_reduce对梯度求平均
        """
        if not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        for param in self.actor.parameters():
            if param.grad is not None:
                # 对梯度进行all_reduce求和，然后除以world_size取平均
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
    
    def _check_params_nan(self) -> bool:
        """检查模型参数是否包含NaN或Inf"""
        for param in self.actor.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False
    
    def _backup_params(self) -> dict:
        """备份当前参数"""
        return {name: p.data.clone() for name, p in self.actor.named_parameters()}
    
    def _restore_params(self, backup: dict):
        """从备份恢复参数"""
        for name, param in self.actor.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
    
    def update(self) -> Dict[str, float]:
        """
        执行策略梯度更新（简化版REINFORCE）
        
        关键：使用 get_log_prob 重新计算 log_prob，保留梯度用于反向传播
        buffer 中的 selected_indices 和 perturb_types 记录了采样时的动作
        """
        global print_num
        if len(self.rollout_buffer) == 0:
            return {'actor_loss': 0.0}
        
        batch = self.rollout_buffer.get_batch(self.device)
        if not batch:
            return {'actor_loss': 0.0}
        
        batch_size = len(batch['seq_lengths'])
        
        # 获取rewards
        rewards = batch['rewards'].float()
        
        # 获取数据（形状已统一，无需维度调整）
        # hidden_states: (batch, seq_len, hidden)
        # selected_indices: (batch, num_select)
        # perturb_types: (batch, num_select)
        hidden_states = batch['hidden_states']
        selected_indices = batch['selected_indices']
        perturb_types = batch['perturb_types']
        
        # 重新计算 log_prob（有梯度！）
        # 这样才能建立从 actor 参数到 loss 的计算图
        log_probs = self.actor.get_log_prob(hidden_states, selected_indices, perturb_types)
        # log_probs = batch['log_probs']
        
        # 使用移动平均baseline计算advantage
        if batch_size == 1:
            # 单条经验：使用移动平均baseline
            reward_value = rewards[0].item()
            advantage = reward_value - self._reward_baseline
            # 更新baseline（移动平均）
            self._reward_baseline = (self._baseline_momentum * self._reward_baseline + 
                                    (1 - self._baseline_momentum) * reward_value)
            advantages = torch.tensor([advantage], device=self.device, dtype=rewards.dtype)
        else:
            # 多条经验：使用batch内标准化
            rewards_mean = rewards.mean()
            rewards_std = rewards.std()
            
            if rewards_std < 1e-8:
                self._reward_baseline = (self._baseline_momentum * self._reward_baseline + 
                                        (1 - self._baseline_momentum) * rewards_mean.item())
                self.rollout_buffer.clear()
                return {'actor_loss': 0.0, 'skipped': True, 'reason': 'identical_rewards'}
            
            advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)
        
        # 调试信息
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(f"\n=== REINFORCE Update ===")
        #     print(f"log_probs: mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f}, shape={log_probs.shape}")
        #     print(f"advantages: mean={advantages.mean().item():.6f}")
        #     print(f"reward_baseline: {self._reward_baseline:.6f}")
        
        # REINFORCE损失：-log_prob * advantage
        actor_loss = -(log_probs * advantages).mean()
        
        # 熵正则（鼓励探索）
        entropy = -log_probs.mean()
         
        total_loss = actor_loss - self.config.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        
        # 调试信息：检查梯度
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     # total_grad_norm = 0.0
        #     # has_nan_inf = False  # 标记是否存在NaN/inf梯度
        #     # for p in self.actor.parameters():
        #     #     if p.grad is not None:
        #     #         # 检测当前参数的梯度是否有NaN/inf
        #     #         if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
        #     #             has_nan_inf = True
        #     #         grad_norm = p.grad.data.norm(2).item()
        #     #         total_grad_norm += grad_norm ** 2
        #     # total_grad_norm = total_grad_norm ** 0.5
        #     # print(f"total_loss: {total_loss.item():.6f}, actor_loss: {actor_loss.item():.6f}")
        #     # print(f"grad_norm (before clip): {total_grad_norm:.6f}, has_nan_inf: {has_nan_inf}")
        #     grads = []
        #     has_nan_inf = False
        #     for p in self.actor.parameters():
        #         if p.grad is not None:
        #             g = p.grad.detach().view(-1)
        #             if not torch.isfinite(g).all():
        #                 has_nan_inf = True
        #             grads.append(g)
        #     all_grads = torch.cat(grads)
        #     grad_stats = {
        #         "mean": all_grads.mean().item(),
        #         "var": all_grads.var(unbiased=False).item(),
        #         "max": all_grads.max().item(),
        #         "min": all_grads.min().item(),
        #         "l2_norm": all_grads.norm(2).item(),
        #     }
        #     print(
        #         f"actor_loss: {actor_loss.item():.6f} | "
        #         f"grad_mean={grad_stats['mean']:.3e}, "
        #         f"grad_var={grad_stats['var']:.3e}, "
        #         f"grad_max={grad_stats['max']:.3e}, "
        #         f"grad_min={grad_stats['min']:.3e}, "
        #         f"grad_norm={grad_stats['l2_norm']:.3e}, "
        #         f"has_nan_inf={has_nan_inf}"
        #     )
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        
        # 分布式梯度同步
        self._sync_gradients()
        
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     params = []
        #     for p in self.actor.parameters():
        #         if p is not None:
        #             params.append(p.detach().view(-1))
        #     all_params = torch.cat(params)
        #     stats = {
        #         "mean": all_params.mean().item(),
        #         "var": all_params.var(unbiased=False).item(),
        #         "max": all_params.max().item(),
        #         "min": all_params.min().item(),
        #         "nan_ratio": torch.isnan(all_params).float().mean().item(),
        #         "inf_ratio": torch.isinf(all_params).float().mean().item(),
        #     }
        #     print(
        #         "before update"
        #         f"mean={stats['mean']:.3e}, "
        #         f"var={stats['var']:.3e}, "
        #         f"max={stats['max']:.3e}, "
        #         f"min={stats['min']:.3e}, "
        #         f"nan_ratio={stats['nan_ratio']:.3e}, "
        #         f"inf_ratio={stats['inf_ratio']:.3e}"
        #     )

        # 执行更新
        self.actor_optimizer.step()

        # 参数更新后，计算参数中NaN的占比
        # total_numel = 0
        # nan_numel = 0
        # for p in self.actor.parameters():
        #     total_numel += p.numel()
        #     nan_numel += torch.isnan(p).sum().item()
        # nan_ratio = nan_numel / total_numel
        # print(f"NaN ratio: {nan_ratio:.6f}")
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     params = []
        #     for p in self.actor.parameters():
        #         if p is not None:
        #             params.append(p.detach().view(-1))
        #     all_params = torch.cat(params)
        #     stats = {
        #         "mean": all_params.mean().item(),
        #         "var": all_params.var(unbiased=False).item(),
        #         "max": all_params.max().item(),
        #         "min": all_params.min().item(),
        #         "nan_ratio": torch.isnan(all_params).float().mean().item(),
        #         "inf_ratio": torch.isinf(all_params).float().mean().item(),
        #     }
        #     print(
        #         "after update"
        #         f"mean={stats['mean']:.3e}, "
        #         f"var={stats['var']:.3e}, "
        #         f"max={stats['max']:.3e}, "
        #         f"min={stats['min']:.3e}, "
        #         f"nan_ratio={stats['nan_ratio']:.3e}, "
        #         f"inf_ratio={stats['inf_ratio']:.3e}"
        #     )
        
        self.rollout_buffer.clear()
        self._update_count += 1
        
        self._actor_loss_history.append(actor_loss.item())
        print_num -= 1
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': 0.0,
            'entropy': entropy.item(),
            'update_count': self._update_count,
            'reward_baseline': self._reward_baseline,
        }
    
    def _sync_critic_gradients(self):
        """在分布式环境中同步Critic的梯度"""
        if not dist.is_initialized() or self.value_network is None:
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        for param in self.value_network.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
    
    def _check_critic_params_nan(self) -> bool:
        """检查Critic参数是否包含NaN或Inf"""
        if self.value_network is None:
            return False
        for param in self.value_network.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False
    
    def _backup_critic_params(self) -> dict:
        """备份Critic参数"""
        if self.value_network is None:
            return {}
        return {name: p.data.clone() for name, p in self.value_network.named_parameters()}
    
    def _restore_critic_params(self, backup: dict):
        """从备份恢复Critic参数"""
        if self.value_network is None:
            return
        for name, param in self.value_network.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
    
    def update_ppo(self) -> Dict[str, float]:
        """
        执行PPO (Proximal Policy Optimization) 更新
        
        核心流程：
        1. 使用value_network估计状态价值作为baseline
        2. 计算advantage = reward - V(s)
        3. 多轮epoch中：
           - 重新计算当前策略下的log_prob
           - 计算importance sampling ratio
           - 使用clipped surrogate objective更新Actor
           - 使用MSE loss更新Critic
        4. NaN保护：更新后检查参数，异常时回滚
        """
        if self.value_network is None:
            self.logger.warning("Value network not available, falling back to REINFORCE update.")
            return self.update()
        
        if len(self.rollout_buffer) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        batch = self.rollout_buffer.get_batch(self.device)
        if not batch:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # 获取数据
        hidden_states = batch['hidden_states']         # (batch, seq_len, hidden)
        selected_indices = batch['selected_indices']    # (batch, num_select)
        perturb_types = batch['perturb_types']          # (batch, num_select)
        old_log_probs = batch['log_probs']              # (batch,)
        rewards = batch['rewards'].float()              # (num_experiences,)
        
        # 推理batch大小（一个batch forward产生的样本数）
        inference_batch_size = hidden_states.shape[0]
        
        # 将reward广播到所有样本（同一batch内所有样本共享同一个reward）
        if rewards.numel() == 1:
            returns = rewards.expand(inference_batch_size)
        else:
            returns = rewards
        
        # ------------------------------------------------------------------
        # 在更新循环开始前，用旧的value network计算advantage（固定不变）
        # ------------------------------------------------------------------
        with torch.no_grad():
            old_values = self.value_network(hidden_states.float())    # (inference_batch,)
            advantages = returns - old_values
            
            # Advantage标准化（减少方差）
            if inference_batch_size > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        
        # ------------------------------------------------------------------
        # PPO多轮更新
        # ------------------------------------------------------------------
        for epoch in range(self.config.ppo_epochs):
            # ========== Actor更新 ==========
            # 用当前策略重新计算log_prob（带梯度）
            new_log_probs = self.actor.get_log_prob(hidden_states, selected_indices, perturb_types)
            
            # Importance sampling ratio
            log_ratio = new_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # 防止极端ratio
            ratio = torch.exp(log_ratio)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config.clip_epsilon, 
                1.0 + self.config.clip_epsilon
            ) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 熵正则（鼓励探索）
            entropy = -new_log_probs.mean()
            
            actor_total_loss = actor_loss - self.config.entropy_coef * entropy
            
            # 备份参数 -> 梯度更新 -> NaN检查
            # actor_backup = self._backup_params()
            self.actor_optimizer.zero_grad()
            actor_total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self._sync_gradients()
            self.actor_optimizer.step()
            
            # if self._check_params_nan():
            #     self.logger.warning(f"NaN in actor params after PPO epoch {epoch}, rolling back")
            #     self._restore_params(actor_backup)
            
            # ========== Critic更新 ==========
            new_values = self.value_network(hidden_states.float())    # (inference_batch,)
            critic_loss = self.config.value_coef * F.mse_loss(new_values, returns)
            
            # critic_backup = self._backup_critic_params()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.max_grad_norm)
            self._sync_critic_gradients()
            self.critic_optimizer.step()
            
            # if self._check_critic_params_nan():
            #     self.logger.warning(f"NaN in critic params after PPO epoch {epoch}, rolling back")
            #     self._restore_critic_params(critic_backup)
            
            # ========== 统计信息 ==========
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_clip_frac += clip_frac
        
        # ------------------------------------------------------------------
        # 收尾
        # ------------------------------------------------------------------
        num_epochs = max(self.config.ppo_epochs, 1)
        self.rollout_buffer.clear()
        self._update_count += 1
        
        avg_actor_loss = total_actor_loss / num_epochs
        avg_critic_loss = total_critic_loss / num_epochs
        
        self._actor_loss_history.append(avg_actor_loss)
        self._critic_loss_history.append(avg_critic_loss)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': total_entropy / num_epochs,
            'clip_fraction': total_clip_frac / num_epochs,
            'update_count': self._update_count,
        }
    
    def save_checkpoint(self, path: str, episode: int):
        """
        保存检查点（只有rank 0保存）
        
        Args:
            path: 保存路径
            episode: 当前eepisode数
        """
        # 只有主进程保存
        if not is_main_process():
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'update_count': self._update_count,
        }
        if self.value_network is not None:
            checkpoint['critic_state_dict'] = self.value_network.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def save_final_model(self, save_dir: str):
        """
        保存最终训练好的模型（只有rank 0保存）
        
        Args:
            save_dir: 保存目录
        """
        if not is_main_process():
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存Actor（扰动生成器）
        actor_path = os.path.join(save_dir, "perturbation_generator.pt")
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'config': {
                'hidden_size': self.actor.hidden_size if hasattr(self.actor, 'hidden_size') else None,
            }
        }, actor_path)
        self.logger.info(f"Actor model saved to {actor_path}")
        
        # 保存完整checkpoint（包含优化器状态，以防后续需要）
        full_path = os.path.join(save_dir, "final_checkpoint.pt")
        full_ckpt = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'update_count': self._update_count,
        }
        if self.value_network is not None:
            full_ckpt['critic_state_dict'] = self.value_network.state_dict()
            full_ckpt['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        torch.save(full_ckpt, full_path)
        self.logger.info(f"Full checkpoint saved to {full_path}")
    
    def load_model(self, path: str):
        """
        加载训练好的模型（用于推理）
        
        Args:
            path: 模型路径（可以是actor或完整checkpoint）
        """
        if not os.path.exists(path):
            self.logger.error(f"Model not found: {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 支持两种格式
        if 'model_state_dict' in checkpoint:
            # 单独的actor模型
            self.actor.load_state_dict(checkpoint['model_state_dict'])
        elif 'actor_state_dict' in checkpoint:
            # 完整checkpoint
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        self.logger.info(f"Model loaded from {path}")
        return True
    
    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计信息"""
        stats = {
            'update_count': self._update_count,
        }
        
        if self._actor_loss_history:
            stats['avg_actor_loss'] = sum(self._actor_loss_history[-100:]) / min(100, len(self._actor_loss_history))
        
        return stats


# 保持向后兼容的别名
PPOTrainer = PolicyGradientTrainer


def create_ppo_trainer(actor: PerturbationGenerator,
                       config: PPOConfig,
                       device: str = 'cuda',
                       value_network: Optional[nn.Module] = None) -> PolicyGradientTrainer:
    """创建策略梯度训练器
    
    Args:
        actor: 扰动生成器（策略网络）
        config: PPO训练配置
        device: 训练设备
        value_network: 价值网络（Critic），传入后启用PPO更新
    """
    return PolicyGradientTrainer(actor, config, device, value_network=value_network)
