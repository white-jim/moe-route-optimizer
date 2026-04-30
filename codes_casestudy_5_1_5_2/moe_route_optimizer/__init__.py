"""
MoE Route Optimizer
==================

基于强化学习的MoE模型推理加速框架

通过训练扰动生成器来优化MoE模型的路由决策，
减少分布式部署时的all-to-all通信时延。

主要模块:
- config: 配置管理和日志系统
- core: 扰动生成器和价值网络
- hooks: Hook管理器，用于注入扰动
- interfaces: 推理框架和评估器抽象接口
- training: PPO训练器和相关组件
- evaluation: 数据集评估器

使用示例:
---------
训练:
    python main.py --model-path /path/to/model --dataset boolq

推理:
    python inference.py --checkpoint /path/to/checkpoint --model-path /path/to/model

编程接口:
    from moe_route_optimizer import MoERouteOptimizer
    
    optimizer = MoERouteOptimizer("checkpoint.pt")
    optimizer.attach_to_model(your_model)
    
    # 运行推理...
    
    optimizer.detach()
"""

__version__ = "0.1.0"
__author__ = "MoE Route Optimizer Team"

from .config import Config, LoggerManager, get_logger
from .inference import MoERouteOptimizer

__all__ = [
    'Config',
    'LoggerManager', 
    'get_logger',
    'MoERouteOptimizer',
]
