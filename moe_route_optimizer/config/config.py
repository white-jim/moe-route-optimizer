"""
统一配置管理模块
管理所有路径、超参数、训练参数等配置信息
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    project_root: str = "/mnt/data/lwy/moe-route-optimizer"
    
    # 日志目录
    log_dir: str = field(default="")
    
    # 模型保存目录
    checkpoint_dir: str = field(default="")
    
    # 数据集目录
    dataset_dir: str = field(default="")
    
    # 日志文件名
    train_log_file: str = "training.log"
    eval_log_file: str = "evaluation.log"
    
    def __post_init__(self):
        self.log_dir = os.path.join(self.project_root, "logs")
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        self.dataset_dir = os.path.join(self.project_root, "datasets")
        
        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    @property
    def train_log_path(self) -> str:
        return os.path.join(self.log_dir, self.train_log_file)
    
    @property
    def eval_log_path(self) -> str:
        return os.path.join(self.log_dir, self.eval_log_file)


@dataclass
class ModelConfig:
    """模型相关配置"""
    # 大模型路径
    base_model_path: str = ""
    
    # 隐藏层维度 (需要根据具体模型设置)
    hidden_size: int = 2048
    
    # MoE专家数量
    num_experts: int = 64
    
    # 目标MoE层索引 (只在第一个MoE层前插入hook)
    target_moe_layer_idx: int = 0


@dataclass
class PerturbationConfig:
    """扰动生成器配置"""
    # 每次扰动的token数量 (超参数K)
    num_perturb_tokens: int = 50
    
    # 扰动维度数量
    num_perturb_dims: int = 50
    
    # 扰动缩放因子
    perturbation_scale: float = 1.0
    
    # Token选择器隐藏层维度 (减小以加速计算)
    selector_hidden_dim: int = 64
    
    # 扰动类型决策器隐藏层维度 (减小以加速计算)
    dim_selector_hidden_dim: int = 32


@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 是否使用PPO更新（False则退回REINFORCE）
    use_ppo: bool = True
    
    # 学习率
    # actor_lr: float = 3e-4
    # 先降一点
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    
    # PPO裁剪参数
    # clip_epsilon: float = 0.2
    clip_epsilon: float = 0.1
    
    # GAE参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 训练批次大小
    batch_size: int = 32
    
    # 推理时每个mini-batch的样本数
    mini_batch_size: int = 4
    
    # 每次更新的epoch数
    # ppo_epochs: int = 4
    ppo_epochs: int = 2
    
    # 熵正则化系数
    entropy_coef: float = 0.01
    
    # 价值函数损失系数
    value_coef: float = 0.5
    
    # 最大梯度范数
    max_grad_norm: float = 0.5
    
    # 价值网络隐藏层维度
    value_hidden_dim: int = 64


@dataclass
class RewardConfig:
    """奖励计算配置"""
    # 时延奖励权重
    # latency_weight: float = 0.4
    latency_weight: float = 0.3
    
    # 精度奖励权重
    # accuracy_weight: float = 0.6
    accuracy_weight: float = 0.7
    
    # 精度惩罚阈值 (低于此值时额外惩罚)
    accuracy_penalty_threshold: float = 0.9
    
    # 精度惩罚系数
    # accuracy_penalty_coef: float = 2.0
    accuracy_penalty_coef: float = 3.0


@dataclass
class ConvergenceConfig:
    """收敛判断配置"""
    # 最大训练episode数
    max_episodes: int = 100
    
    # 成功判定的时延减少阈值 (如20%表示0.2)
    latency_reduction_threshold: float = 0.20
    
    # 成功判定的精度保持阈值 (相对基线的比例)
    accuracy_maintain_threshold: float = 1.0
    
    # 奖励稳定判定的窗口大小
    reward_window_size: int = 10
    
    # 奖励稳定判定的方差阈值
    reward_variance_threshold: float = 0.01
    
    # 成功收敛需要持续的episode数
    success_patience: int = 3
    
    # 精度持续下降的早停耐心值
    accuracy_early_stop_patience: int = 5
    
    # 时延无改善的停滞耐心值
    stagnation_patience: int = 5
    
    # 评估间隔 (每多少episode评估一次)
    eval_interval: int = 5


@dataclass
class TrainingConfig:
    """整体训练配置"""
    # 随机种子
    seed: int = 42
    
    # 设备
    device: str = "cuda"
    
    # 是否使用混合精度训练
    use_amp: bool = False
    
    # 日志输出间隔 (每多少step输出一次)
    log_interval: int = 5
    
    # 模型保存间隔 (每多少episode保存一次)
    save_interval: int = 100
    
    # 是否开启debug模式
    debug: bool = False


@dataclass
class Config:
    """总配置类"""
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, path: str):
        """保存配置到文件"""
        import json
        config_dict = {
            'path': self.path.__dict__,
            'model': self.model.__dict__,
            'perturbation': self.perturbation.__dict__,
            'ppo': self.ppo.__dict__,
            'reward': self.reward.__dict__,
            'convergence': self.convergence.__dict__,
            'training': self.training.__dict__,
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.get('path', {}).items():
            if hasattr(config.path, key):
                setattr(config.path, key, value)
        for key, value in config_dict.get('model', {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        for key, value in config_dict.get('perturbation', {}).items():
            if hasattr(config.perturbation, key):
                setattr(config.perturbation, key, value)
        for key, value in config_dict.get('ppo', {}).items():
            if hasattr(config.ppo, key):
                setattr(config.ppo, key, value)
        for key, value in config_dict.get('reward', {}).items():
            if hasattr(config.reward, key):
                setattr(config.reward, key, value)
        for key, value in config_dict.get('convergence', {}).items():
            if hasattr(config.convergence, key):
                setattr(config.convergence, key, value)
        for key, value in config_dict.get('training', {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
        
        return config


# 全局默认配置实例
default_config = Config()
