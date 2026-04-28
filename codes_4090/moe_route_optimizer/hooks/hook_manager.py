"""
Hook管理器模块
负责在推理框架的MoE层注入扰动逻辑
只在第一个MoE层的路由前插入hook
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
import weakref

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import get_train_logger


@dataclass
class CollectedState:
    """收集的单次状态信息"""
    hidden_states: torch.Tensor  # 原始hidden states
    selected_indices: torch.Tensor  # 选中的token索引
    perturb_dim_indices: torch.Tensor  # 每个选中token被置0的维度索引
    log_prob: torch.Tensor  # 动作对数概率
    perturbation: torch.Tensor  # 应用的扰动


@dataclass
class StateBuffer:
    """状态缓冲区，用于收集训练数据"""
    states: List[CollectedState] = field(default_factory=list)
    
    def add(self, state: CollectedState):
        self.states.append(state)
    
    def clear(self):
        self.states.clear()
    
    def get_batch(self) -> Optional[Dict[str, Any]]:
        """
        获取批量数据，按序列长度分组处理变长序列
        
        Returns:
            包含以下键的字典：
            - 'grouped_data': Dict[int, Dict[str, torch.Tensor]] 
              按序列长度分组的数据，键为序列长度
            - 'all_states': List[CollectedState] 
              所有原始状态的列表（用于需要逐个处理的场景）
            - 'seq_lengths': List[int]
              每个状态对应的序列长度列表
        """
        if not self.states:
            return None
        
        # 按序列长度分组
        from collections import defaultdict
        grouped: Dict[int, List[CollectedState]] = defaultdict(list)
        seq_lengths: List[int] = []
        
        for state in self.states:
            # hidden_states 形状: (batch, seq_len, hidden_dim)
            seq_len = state.hidden_states.shape[1]
            grouped[seq_len].append(state)
            seq_lengths.append(seq_len)
        
        # 对每组进行堆叠
        grouped_data: Dict[int, Dict[str, torch.Tensor]] = {}
        for seq_len, states_group in grouped.items():
            grouped_data[seq_len] = {
                'hidden_states': torch.stack([s.hidden_states for s in states_group]),
                'selected_indices': torch.stack([s.selected_indices for s in states_group]),
                'perturb_dim_indices': torch.stack([s.perturb_dim_indices for s in states_group]),
                'log_probs': torch.stack([s.log_prob for s in states_group]),
                'perturbations': torch.stack([s.perturbation for s in states_group]),
                'count': len(states_group),
            }
        
        return {
            'grouped_data': grouped_data,
            'all_states': self.states,
            'seq_lengths': seq_lengths,
        }
    
    def get_batch_as_list(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        获取批量数据，返回列表形式（每个元素是一个状态的字典）
        适用于需要逐个处理的场景
        """
        if not self.states:
            return None
        
        return [
            {
                'hidden_states': s.hidden_states,
                'selected_indices': s.selected_indices,
                'perturb_dim_indices': s.perturb_dim_indices,
                'log_prob': s.log_prob,
                'perturbation': s.perturbation,
            }
            for s in self.states
        ]
    
    def __len__(self):
        return len(self.states)


class HookManager:
    """
    Hook管理器
    负责在推理框架的MoE层注册和管理hook
    只在第一个MoE层的路由前插入扰动
    """
    
    def __init__(self, perturbation_generator: nn.Module):
        """
        Args:
            perturbation_generator: 扰动生成器模块
        """
        self.generator = perturbation_generator
        self.logger = get_train_logger()
        
        # 模式控制
        self._is_training = True
        self._is_enabled = True
        
        # 状态收集
        self.state_buffer = StateBuffer()
        
        # Hook句柄存储
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # 用于存储当前batch的信息
        self._current_batch_info: Dict[str, Any] = {}
        
        # 跨进程模式（用于 EP adapter 的 _WorkerModuleProxy）
        self._cross_process_mode = False
        self._cross_process_proxy = None
    
    @property
    def is_training(self) -> bool:
        return self._is_training
    
    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value
        if value:
            self.generator.train()
        else:
            self.generator.eval()
        if self._cross_process_mode and self._cross_process_proxy is not None:
            self._cross_process_proxy._set_hook_training(value)
    
    @property
    def is_enabled(self) -> bool:
        return self._is_enabled
    
    @is_enabled.setter
    def is_enabled(self, value: bool):
        self._is_enabled = value
        if self._cross_process_mode and self._cross_process_proxy is not None:
            self._cross_process_proxy._set_hook_enabled(value)
    
    def register_hook(self, target_module: nn.Module, module_name: str = "moe_block"):
        """
        注册forward pre hook到目标模块
        
        如果target_module是跨进程代理（_WorkerModuleProxy），
        则自动切换到跨进程模式：将generator发送到worker进程注册。
        
        Args:
            target_module: 目标模块（推荐使用整个MoE块，而不是只用Gate）
            module_name: 模块名称（用于日志）
        """
        # 检测是否是跨进程代理（通过duck typing避免循环导入）
        if hasattr(target_module, '_register_perturbation_generator'):
            self._cross_process_mode = True
            self._cross_process_proxy = target_module
            target_module._register_perturbation_generator(self.generator)
            # 同步当前状态到worker
            target_module._set_hook_enabled(self._is_enabled)
            target_module._set_hook_training(self._is_training)
            self.logger.info(
                f"Registered perturbation hook (cross-process) on module: {module_name}"
            )
            return
        def perturbation_hook(module, args):
            """
            Forward pre hook，在MoE块/Gate计算前修改hidden states
            
            支持的hidden_states形状：
            - 3D: (batch, seq, hidden) 标准Transformer格式
            - 2D: (num_tokens, hidden) vLLM的批处理格式
            
            Args:
                module: 被hook的模块
                args: 输入参数元组
            
            Returns:
                修改后的输入参数元组
            """
            if not self._is_enabled:
                return args
            
            # 获取hidden states（假设是第一个参数）
            if len(args) == 0:
                return args
            
            hidden_states = args[0]
            # 这里的输入是3D张量 (batch, seq, hidden)
            # 本hook只在prefill阶段执行，所以如果seq为1则直接跳过
            if hidden_states.shape[1] == 1:
                return args
            print("hidden_states shape:", hidden_states.shape)
            # 检查输入类型
            if not isinstance(hidden_states, torch.Tensor):
                return args
            
            # 支持2D和3D的hidden_states
            orig_shape = hidden_states.shape
            orig_dim = hidden_states.dim()
            
            if orig_dim == 2:
                # 2D: (num_tokens, hidden) -> 扩展为 (1, num_tokens, hidden) 以便处理
                hidden_states_3d = hidden_states.unsqueeze(0)
            elif orig_dim == 3:
                # 3D: (batch, seq, hidden) 已经是标准格式
                hidden_states_3d = hidden_states
            else:
                # 不支持的维度，跳过
                return args
            
            # 生成扰动
            with torch.set_grad_enabled(self._is_training):
                result = self.generator(
                    hidden_states_3d, 
                    deterministic=not self._is_training
                )
            
            # 获取扰动后的hidden states
            perturbed_hidden_3d = result['perturbed_hidden_states']
            
            # 恢复原始形状
            if orig_dim == 2:
                perturbed_hidden = perturbed_hidden_3d.squeeze(0)
            else:
                perturbed_hidden = perturbed_hidden_3d
            
            # 确保输出形状与输入一致
            if perturbed_hidden.shape != orig_shape:
                self.logger.warning(
                    f"Shape mismatch: input {orig_shape}, output {perturbed_hidden.shape}. "
                    f"Reshaping to original shape."
                )
                perturbed_hidden = perturbed_hidden.view(orig_shape)
            
            # 如果在训练模式，收集状态
            if self._is_training:
                # 存储为detach的副本以节省内存
                state = CollectedState(
                    hidden_states=hidden_states_3d.detach().clone(),
                    selected_indices=result['selected_indices'].detach().clone(),
                    perturb_dim_indices=result['perturb_dim_indices'].detach().clone(),
                    log_prob=result['log_prob'].detach().clone(),
                    perturbation=perturbed_hidden_3d.detach().clone(),
                )
                # state = CollectedState(
                #     hidden_states=hidden_states_3d,
                #     selected_indices=result['selected_indices'],
                #     perturb_dim_indices=result['perturb_dim_indices'],
                #     log_prob=result['log_prob'],
                #     perturbation=perturbed_hidden_3d,
                # )
                self.state_buffer.add(state)
            
            # 返回修改后的输入
            return (perturbed_hidden,) + args[1:]
        
        # 注册hook
        handle = target_module.register_forward_pre_hook(perturbation_hook)
        self._hooks.append(handle)
        
        self.logger.info(f"Registered perturbation hook on module: {module_name}")
    
    def register_hook_by_name(self, model: nn.Module, target_layer_name: str):
        """
        通过层名称注册hook
        
        Args:
            model: 完整模型
            target_layer_name: 目标层的名称（支持点分隔的嵌套名称）
        """
        # 解析层名称
        parts = target_layer_name.split('.')
        module = model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit():
                module = module[int(part)]
            else:
                raise ValueError(f"Cannot find layer: {target_layer_name}")
        
        self.register_hook(module, target_layer_name)
    
    def remove_hooks(self):
        """移除所有注册的hook"""
        if self._cross_process_mode and self._cross_process_proxy is not None:
            self._cross_process_proxy._remove_perturbation_hook()
            self._cross_process_mode = False
            self._cross_process_proxy = None
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.logger.info("All hooks removed")
    
    def clear_buffer(self):
        """清空状态缓冲区"""
        self.state_buffer.clear()
        if self._cross_process_mode and self._cross_process_proxy is not None:
            self._cross_process_proxy._clear_hook_buffer()
    
    def get_collected_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取收集的训练数据"""
        if self._cross_process_mode and self._cross_process_proxy is not None:
            raw_states = self._cross_process_proxy._get_collected_states()
            if not raw_states:
                return None
            # 将worker传回的原始字典转换为CollectedState格式
            states = []
            for s in raw_states:
                states.append(CollectedState(
                    hidden_states=self._restore_tensor_from_queue_payload(s['hidden_states']),
                    selected_indices=self._restore_tensor_from_queue_payload(s['selected_indices']),
                    perturb_dim_indices=self._restore_tensor_from_queue_payload(s['perturb_dim_indices']),
                    log_prob=self._restore_tensor_from_queue_payload(s['log_prob']),
                    perturbation=self._restore_tensor_from_queue_payload(s['perturbation']),
                ))
            # 返回与StateBuffer.get_batch()相同的格式
            from collections import defaultdict
            grouped: Dict[int, list] = defaultdict(list)
            seq_lengths = []
            for state in states:
                seq_len = state.hidden_states.shape[1]
                grouped[seq_len].append(state)
                seq_lengths.append(seq_len)
            grouped_data = {}
            for seq_len, sg in grouped.items():
                grouped_data[seq_len] = {
                    'hidden_states': torch.stack([s.hidden_states for s in sg]),
                    'selected_indices': torch.stack([s.selected_indices for s in sg]),
                    'perturb_dim_indices': torch.stack([s.perturb_dim_indices for s in sg]),
                    'log_probs': torch.stack([s.log_prob for s in sg]),
                    'perturbations': torch.stack([s.perturbation for s in sg]),
                    'count': len(sg),
                }
            return {
                'grouped_data': grouped_data,
                'all_states': states,
                'seq_lengths': seq_lengths,
            }
        return self.state_buffer.get_batch()

    @staticmethod
    def _restore_tensor_from_queue_payload(payload: Any) -> torch.Tensor:
        """Restore queue-safe hook payloads back into CPU tensors."""
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, dict) and 'array' in payload:
            tensor = torch.from_numpy(payload['array'])
            dtype_name = payload.get('dtype')
            if dtype_name == 'torch.bfloat16':
                return tensor.to(torch.bfloat16)
            if dtype_name == 'torch.float16':
                return tensor.to(torch.float16)
            if dtype_name == 'torch.float32':
                return tensor.to(torch.float32)
            if dtype_name == 'torch.float64':
                return tensor.to(torch.float64)
            if dtype_name == 'torch.int64':
                return tensor.to(torch.int64)
            if dtype_name == 'torch.int32':
                return tensor.to(torch.int32)
            if dtype_name == 'torch.int16':
                return tensor.to(torch.int16)
            if dtype_name == 'torch.int8':
                return tensor.to(torch.int8)
            if dtype_name == 'torch.uint8':
                return tensor.to(torch.uint8)
            if dtype_name == 'torch.bool':
                return tensor.to(torch.bool)
            return tensor
        raise TypeError(f"Unsupported hook payload type: {type(payload)!r}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/推理模式"""
        self.is_training = training
    
    def enable(self):
        """启用hook"""
        self.is_enabled = True
    
    def disable(self):
        """禁用hook（用于收集baseline）"""
        self.is_enabled = False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，清理hook"""
        self.remove_hooks()

    def sync_weights(self):
        """
        同步扰动生成器权重到worker进程。
        在PPO更新actor后调用此方法，使worker中的generator副本与主进程保持一致。
        在非跨进程模式下为no-op（因为hook直接引用同一个generator对象）。
        """
        if self._cross_process_mode and self._cross_process_proxy is not None:
            self._cross_process_proxy._sync_perturbation_weights(
                self.generator.state_dict()
            )


class HookManagerForMoE(HookManager):
    """
    专门用于MoE模型的Hook管理器
    提供更便捷的MoE层hook注册方法
    """
    
    def __init__(self, perturbation_generator: nn.Module):
        super().__init__(perturbation_generator)
        self._moe_layer_hooked = False
    
    def find_and_register_first_moe_gate(self, model: nn.Module,
                                          gate_pattern: str = "gate"):
        """
        自动查找并在第一个MoE层的Gate前注册hook
        
        Args:
            model: 完整模型
            gate_pattern: Gate模块名称的匹配模式
        """
        def find_first_gate(module: nn.Module, prefix: str = "") -> Optional[Tuple[nn.Module, str]]:
            """递归查找第一个包含gate_pattern的模块"""
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if gate_pattern.lower() in name.lower():
                    return child, full_name
                
                result = find_first_gate(child, full_name)
                if result is not None:
                    return result
            
            return None
        
        result = find_first_gate(model)
        
        if result is None:
            self.logger.warning(f"Could not find any module matching pattern: {gate_pattern}")
            return False
        
        gate_module, gate_name = result
        self.register_hook(gate_module, gate_name)
        self._moe_layer_hooked = True
        
        return True
    
    def register_hook_on_moe_layer(self, moe_layer: nn.Module, 
                                    gate_attr_name: str = "gate"):
        """
        在指定的MoE层上注册hook
        
        Args:
            moe_layer: MoE层模块
            gate_attr_name: Gate模块在MoE层中的属性名
        """
        if hasattr(moe_layer, gate_attr_name):
            gate = getattr(moe_layer, gate_attr_name)
            self.register_hook(gate, f"moe.{gate_attr_name}")
            self._moe_layer_hooked = True
        else:
            # 如果没有gate属性，直接在MoE层注册
            self.register_hook(moe_layer, "moe_layer")
            self._moe_layer_hooked = True


def create_hook_manager(perturbation_generator: nn.Module,
                        for_moe: bool = True) -> HookManager:
    """
    创建Hook管理器的工厂函数
    
    Args:
        perturbation_generator: 扰动生成器
        for_moe: 是否用于MoE模型
    
    Returns:
        HookManager实例
    """
    if for_moe:
        return HookManagerForMoE(perturbation_generator)
    return HookManager(perturbation_generator)
