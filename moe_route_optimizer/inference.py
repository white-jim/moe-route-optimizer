#!/usr/bin/env python
"""
MoE路由优化器 - 推理入口
加载训练好的扰动生成器并应用于推理
"""

import os
import sys
import argparse
import torch
from typing import Optional

# 添加项目路径（使用相对路径，兼容任何工作环境）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    Config,
    LoggerManager,
    get_logger,
)
from core import create_perturbation_generator
from hooks import create_hook_manager
from interfaces import (
    InferenceFrameworkInterface,
    create_vllm_adapter,
)


class MoERouteOptimizer:
    """
    MoE路由优化器封装类
    用于在推理时方便地加载和使用训练好的扰动生成器
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config: Optional[Config] = None,
                 device: str = 'cuda'):
        """
        Args:
            checkpoint_path: 训练好的模型检查点路径
            config: 配置对象（如果为None则使用默认配置）
            device: 推理设备
        """
        self.config = config or Config()
        self.device = device
        self.logger = get_logger()
        
        self.generator = None
        self.hook_manager = None
        self._is_attached = False
        
        # 加载模型
        self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 从检查点中获取hidden_size（如果有保存的话）
        hidden_size = self.config.model.hidden_size
        
        # 创建扰动生成器
        self.generator = create_perturbation_generator(
            self.config.perturbation,
            hidden_size,
            self.device
        )
        
        # 加载权重
        self.generator.load_state_dict(checkpoint['actor_state_dict'])
        self.generator.eval()
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def attach_to_model(self, model: torch.nn.Module, 
                        gate_module_name: Optional[str] = None):
        """
        将优化器附加到模型上
        
        Args:
            model: 目标模型
            gate_module_name: Gate模块的名称（如果为None则自动查找）
        """
        if self._is_attached:
            self.logger.warning("Already attached to a model. Detaching first...")
            self.detach()
        
        # 创建Hook管理器
        self.hook_manager = create_hook_manager(self.generator, for_moe=True)
        self.hook_manager.set_training_mode(False)  # 设置为推理模式
        
        if gate_module_name:
            # 使用指定的模块名称
            self.hook_manager.register_hook_by_name(model, gate_module_name)
        else:
            # 自动查找第一个MoE Gate
            success = self.hook_manager.find_and_register_first_moe_gate(model)
            if not success:
                raise RuntimeError("Could not find MoE gate in model")
        
        self._is_attached = True
        self.logger.info("Optimizer attached to model")
    
    def attach_to_framework(self, framework: InferenceFrameworkInterface):
        """
        将优化器附加到推理框架
        
        Args:
            framework: 推理框架接口
        """
        if self._is_attached:
            self.detach()
        
        self.hook_manager = create_hook_manager(self.generator, for_moe=True)
        self.hook_manager.set_training_mode(False)
        
        first_gate = framework.get_first_moe_gate()
        if first_gate is not None:
            self.hook_manager.register_hook(first_gate, "first_moe_gate")
            self._is_attached = True
            self.logger.info("Optimizer attached to framework")
        else:
            raise RuntimeError("Could not find MoE gate in framework")
    
    def detach(self):
        """从模型上分离"""
        if self.hook_manager is not None:
            self.hook_manager.remove_hooks()
            self.hook_manager = None
        self._is_attached = False
        self.logger.info("Optimizer detached")
    
    def enable(self):
        """启用扰动"""
        if self.hook_manager:
            self.hook_manager.enable()
    
    def disable(self):
        """禁用扰动"""
        if self.hook_manager:
            self.hook_manager.disable()
    
    @property
    def is_attached(self) -> bool:
        return self._is_attached
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()


def run_inference_demo(checkpoint_path: str, model_path: str, prompt: str):
    """
    运行推理演示
    
    Args:
        checkpoint_path: 模型检查点路径
        model_path: 基础模型路径
        prompt: 输入提示
    """
    logger = get_logger()
    
    # 创建配置
    config = Config()
    
    # 创建推理框架
    framework = create_vllm_adapter()
    framework.load_model(model_path, hidden_size=config.model.hidden_size)
    
    # 创建优化器并附加
    optimizer = MoERouteOptimizer(checkpoint_path, config)
    optimizer.attach_to_framework(framework)
    
    logger.info(f"Running inference with prompt: {prompt[:50]}...")
    
    # 运行优化后的推理
    optimizer.enable()
    framework.run_inference(prompt)
    optimized_output = framework.get_model_output()
    optimized_latency = framework.get_comm_delay()
    
    logger.info(f"Optimized - Latency: {optimized_latency:.4f}s")
    logger.info(f"Output: {optimized_output}")
    
    # 对比：运行原始推理
    optimizer.disable()
    framework.run_inference(prompt)
    original_output = framework.get_model_output()
    original_latency = framework.get_comm_delay()
    
    logger.info(f"Original - Latency: {original_latency:.4f}s")
    logger.info(f"Output: {original_output}")
    
    # 计算加速比
    if original_latency > 0:
        speedup = (original_latency - optimized_latency) / original_latency * 100
        logger.info(f"Latency Reduction: {speedup:.2f}%")
    
    # 清理
    optimizer.detach()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MoE Route Optimizer Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained model checkpoint')
    parser.add_argument('--model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--prompt', type=str, default="Hello, how are you?", help='Input prompt')
    parser.add_argument('--config', type=str, default=None, help='Configuration file')
    
    args = parser.parse_args()
    
    # 初始化日志
    LoggerManager.setup(
        log_dir="/mnt/data/lwy/vLLM-wrok/moe_route_optimizer/logs",
        debug=False
    )
    
    run_inference_demo(args.checkpoint, args.model_path, args.prompt)


if __name__ == "__main__":
    main()
