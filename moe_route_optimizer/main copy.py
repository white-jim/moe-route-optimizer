#!/usr/bin/env python
"""
MoE路由优化器 - 训练主入口
通过强化学习训练扰动生成器，优化MoE模型的推理时延

支持使用torchrun进行分布式训练：
    torchrun --nproc-per-node=2 main.py --model-path /path/to/model
    torchrun --nproc-per-node=2 main.py --model-path ../models/Qwen1.5-MoE-A2.7B
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import random
import numpy as np
from typing import Optional, List, Tuple

# 添加项目路径
sys.path.insert(0, '/mnt/data/lwy/vLLM-wrok/moe_route_optimizer')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from vllm.distributed.device_communicators.all2all import _total_all2all_time
_total_all2all_time = 0.0

from config import (
    Config, 
    LoggerManager, 
    get_train_logger, 
    get_eval_logger,
    TrainingMetricsLogger,
    EvaluationLogger,
    is_main_process,
    get_rank,
)
from core import (
    create_perturbation_generator,
    create_value_network,
)
from hooks import create_hook_manager
from hooks.comm_delay_collector import init_shared_memory
init_shared_memory()

from interfaces import (
    InferenceFrameworkInterface,
    AccuracyEvaluatorInterface,
    create_vllm_adapter,
    create_hf_accelerate_adapter,
    create_evaluator,
)
from training import (
    create_reward_calculator,
    create_convergence_checker,
    create_ppo_trainer,
)


def set_seed(seed: int, set_cuda: bool = False):
    """设置随机种子
    
    Args:
        seed: 随机种子
        set_cuda: 是否设置CUDA种子。注意：在vLLM加载模型之前不要设置，
                  否则会导致 'Cannot re-initialize CUDA in forked subprocess' 错误
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 分布式工具函数
# ==============================================================================

def init_distributed():
    """
    初始化分布式环境（如果尚未初始化）
    """
    if not dist.is_initialized():
        # 尝试从环境变量获取分布式配置
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            
            # 设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            # 初始化进程组
            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size
            )
            return True
    return dist.is_initialized()


def get_world_size() -> int:
    """
    获取分布式环境的世界大小
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    对所有rank的tensor求平均
    
    Args:
        tensor: 要聚合的张量
    
    Returns:
        聚合后的平均值张量
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # 确保tensor在正确的设备上
    if not tensor.is_cuda:
        tensor = tensor.cuda()
    
    # 执行all_reduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 计算平均值
    tensor = tensor / get_world_size()
    
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    对所有rank的tensor求和
    
    Args:
        tensor: 要聚合的张量
    
    Returns:
        聚合后的总和张量
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # 确保tensor在正确的设备上
    if not tensor.is_cuda:
        tensor = tensor.cuda()
    
    # 执行all_reduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor


def aggregate_metrics(local_correct: int, local_total: int, local_comm_delay: float) -> Tuple[float, float]:
    """
    聚合分布式环境中各rank的评估指标
    
    Args:
        local_correct: 本rank的正确数量
        local_total: 本rank的总样本数
        local_comm_delay: 本rank的通信时延
    
    Returns:
        (global_accuracy, global_comm_delay): 全局精确度和全局通信时延
    """
    if not dist.is_initialized() or get_world_size() == 1:
        accuracy = local_correct / local_total if local_total > 0 else 0.0
        return accuracy, local_comm_delay
    
    # 创建用于聚合的张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 聚合正确数量和总数量
    correct_tensor = torch.tensor([local_correct], dtype=torch.float32, device=device)
    total_tensor = torch.tensor([local_total], dtype=torch.float32, device=device)
    delay_tensor = torch.tensor([local_comm_delay], dtype=torch.float32, device=device)
    
    # All-reduce
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(delay_tensor, op=dist.ReduceOp.SUM)
    
    # 计算全局精确度
    global_total = total_tensor.item()
    global_accuracy = correct_tensor.item() / global_total if global_total > 0 else 0.0
    
    # 通信时延取平均（因为每个rank都有各自的时延）
    global_comm_delay = delay_tensor.item() / get_world_size()
    
    return global_accuracy, global_comm_delay


def split_data_for_rank(data_list: List, ground_truths: List, dp_rank: int, dp_size: int) -> Tuple[List, List]:
    """
    根据当前rank分割数据，与vLLM的数据分配方式保持一致
    
    Args:
        data_list: 输入数据列表
        ground_truths: 对应的ground truth列表
        dp_rank: 当前数据并行rank
        dp_size: 数据并行大小
    
    Returns:
        (rank_data, rank_ground_truths): 当前rank负责的数据和对应的ground truth
    """
    if dp_size <= 1:
        return data_list, ground_truths
    
    # 与vLLM的分配方式保持一致: idx % dp_size == dp_rank
    rank_data = []
    rank_ground_truths = []
    
    for idx, (data, gt) in enumerate(zip(data_list, ground_truths)):
        if idx % dp_size == dp_rank:
            rank_data.append(data)
            rank_ground_truths.append(gt)
    
    return rank_data, rank_ground_truths


def extract_generated_text(framework: InferenceFrameworkInterface) -> str:
    """
    从vLLM输出中提取生成的文本
    
    Args:
        framework: 推理框架接口
    
    Returns:
        生成的文本字符串
    """
    output = framework.get_model_output()
    
    # vLLM 输出格式: output.outputs[0].text
    if hasattr(output, 'outputs') and output.outputs:
        return output.outputs[0].text
    
    # 备选：使用 get_generated_texts 方法
    if hasattr(framework, 'get_generated_texts'):
        texts = framework.get_generated_texts()
        return texts[0] if texts else ""
    
    # 最后备选：转换为字符串
    return str(output) if output else ""

def extract_generated_texts_batch(framework: InferenceFrameworkInterface) -> List[str]:
    """
    从vLLM批量输出中提取所有生成的文本
    
    Args:
        framework: 推理框架接口
    
    Returns:
        生成的文本字符串列表
    """
    # 优先使用批量获取方法
    if hasattr(framework, 'get_generated_texts'):
        texts = framework.get_generated_texts()
        if texts:
            return texts
    
    # 备选：从 batch_outputs 获取
    if hasattr(framework, 'get_batch_outputs'):
        outputs = framework.get_batch_outputs()
        if outputs:
            texts = []
            for output in outputs:
                if hasattr(output, 'outputs') and output.outputs:
                    texts.append(output.outputs[0].text)
                else:
                    texts.append(str(output) if output else "")
            return texts
    
    # 最后备选：获取单个输出
    single_text = extract_generated_text(framework)
    return [single_text] if single_text else []


def prepare_batched_dataset(evaluator: AccuracyEvaluatorInterface,
                            batch_size: int,
                            max_samples: Optional[int] = None) -> List[Tuple[List[str], List[str]]]:
    """
    预处理数据集，按批次分组，舍弃不足一个批次的余数
    
    Args:
        evaluator: 精度评估器
        batch_size: 每批次的样本数
        max_samples: 最大样本数限制（用于调试），None表示使用全部数据
    
    Returns:
        batches: 批次列表，每个元素是 (inputs, ground_truths) 元组
    """
    logger = get_train_logger()
    
    # 收集所有数据
    all_inputs = []
    all_ground_truths = []
    
    for input_text, ground_truth in evaluator.get_dataset_iterator():
        all_inputs.append(input_text)
        all_ground_truths.append(ground_truth)
        
        # 如果设置了最大样本数限制
        if max_samples is not None and len(all_inputs) >= max_samples:
            break
    
    total_samples = len(all_inputs)
    num_complete_batches = total_samples // batch_size
    used_samples = num_complete_batches * batch_size
    discarded_samples = total_samples - used_samples
    
    logger.info(f"Dataset preparation: total={total_samples}, batch_size={batch_size}, "
                f"num_batches={num_complete_batches}, used={used_samples}, discarded={discarded_samples}")
    
    # 按批次分组
    batches = []
    for batch_idx in range(num_complete_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_inputs = all_inputs[start_idx:end_idx]
        batch_ground_truths = all_ground_truths[start_idx:end_idx]
        batches.append((batch_inputs, batch_ground_truths))
    
    return batches


def collect_batch_baseline(framework: InferenceFrameworkInterface,
                           evaluator: AccuracyEvaluatorInterface,
                           batch_inputs: List[str],
                           batch_ground_truths: List[str],
                           batch_idx: int) -> Tuple[float, float]:
    """
    收集单个批次的基线指标（无扰动时的性能）
    
    Args:
        framework: 推理框架接口
        evaluator: 精度评估器
        batch_inputs: 批次输入
        batch_ground_truths: 批次标签
        batch_idx: 批次索引
    
    Returns:
        (latency, accuracy): 该批次的时延和精确度
    """
    logger = get_eval_logger()
    
    # 获取分布式信息
    dp_rank = framework.get_dp_rank()
    dp_size = framework.get_dp_size()
    
    # 根据当前rank分配ground_truths
    _, rank_ground_truths = split_data_for_rank(
        batch_inputs, batch_ground_truths, dp_rank, dp_size
    )
    
    # 执行批量推理
    framework.run_inference(batch_inputs)
    
    # 收集时延
    latency = framework.get_comm_delay()
    # latency = framework.get_comm_delay_per_token()
    # 提取生成的文本
    generated_texts = extract_generated_texts_batch(framework)
    # 打印第一个输出文本的后150个字符
    # print(f"Generated text for batch {batch_idx+1}: {generated_texts[0][-150:]}")
    
    # 评估精度
    local_correct = 0
    local_total = len(rank_ground_truths)
    
    for gen_text, gt in zip(generated_texts, rank_ground_truths):
        acc = evaluator.evaluate_single(gen_text, gt)
        local_correct += int(acc)
    
    # 计算本rank的平均时延
    rank_batch_size = len(rank_ground_truths)
    local_avg_latency = latency / rank_batch_size if rank_batch_size > 0 else 0.0
    
    # 聚合所有rank的结果
    global_accuracy, global_avg_latency = aggregate_metrics(
        local_correct, local_total, local_avg_latency
    )
    
    logger.debug(f"Batch {batch_idx}: latency={global_avg_latency:.4f}s, accuracy={global_accuracy*100:.2f}%")
    
    return global_avg_latency, global_accuracy


def collect_all_baselines(framework: InferenceFrameworkInterface,
                          evaluator: AccuracyEvaluatorInterface,
                          batches: List[Tuple[List[str], List[str]]]) -> List[Tuple[float, float]]:
    """
    收集所有批次的基线指标（无扰动），作为历史最优值的初始化
    
    Args:
        framework: 推理框架接口
        evaluator: 精度评估器
        batches: 批次列表
    
    Returns:
        baselines: 每个批次的基线 (latency, accuracy) 列表
    """
    logger = get_eval_logger()
    logger.info(f"Collecting baseline for all {len(batches)} batches...")
    
    baselines = []
    total_latency = 0.0
    total_accuracy = 0.0
    
    for batch_idx, (batch_inputs, batch_ground_truths) in enumerate(batches):
        latency, accuracy = collect_batch_baseline(
            framework, evaluator, batch_inputs, batch_ground_truths, batch_idx
        )
        baselines.append((latency, accuracy))
        total_latency += latency
        total_accuracy += accuracy
    
    avg_latency = total_latency / len(batches) if batches else 0.0
    avg_accuracy = total_accuracy / len(batches) if batches else 0.0
    
    logger.info(f"Baseline collection complete: avg_latency={avg_latency:.4f}s, "
                f"avg_accuracy={avg_accuracy*100:.2f}%")
    
    return baselines


def collect_baseline(framework: InferenceFrameworkInterface,
                     evaluator: AccuracyEvaluatorInterface,
                     num_samples: int = 30,
                     mini_batch_size: int = 30) -> tuple:
    """
    收集基线指标（无扰动时的性能）- 使用小批量推理
    支持分布式环境下各rank的数据分配和结果合并
    
    注意：此函数保留用于定期评估，新的训练流程使用 collect_all_baselines
    
    Args:
        framework: 推理框架接口
        evaluator: 精度评估器
        num_samples: 采样数量
        mini_batch_size: 每次批量推理的样本数
    
    Returns:
        (baseline_latency, baseline_accuracy)
    """
    logger = get_eval_logger()
    logger.info(f"Collecting baseline metrics with mini_batch_size={mini_batch_size}...")
    
    # 获取分布式信息
    dp_rank = framework.get_dp_rank()
    dp_size = framework.get_dp_size()
    
    total_latency = 0.0
    local_correct = 0  # 本rank的正确数量
    local_total = 0    # 本rank的总样本数
    num_batches = 0
    count = 0
    
    # 收集一个小批量的数据
    batch_inputs = []
    batch_ground_truths = []
    
    for input_text, ground_truth in evaluator.get_dataset_iterator():
        if count >= num_samples:
            break
        
        batch_inputs.append(input_text)
        batch_ground_truths.append(ground_truth)
        count += 1
        
        # 当收集够一个mini_batch或者达到样本限制时，执行批量推理
        if len(batch_inputs) >= mini_batch_size or count >= num_samples:
            
            # 根据当前rank分配ground_truths（与vLLM的数据分配方式保持一致）
            _, rank_ground_truths = split_data_for_rank(
                batch_inputs, batch_ground_truths, dp_rank, dp_size
            )
            
            # 执行批量推理（vLLM内部会根据dp_rank自动分配数据）
            logger.debug(f"Rank {dp_rank}: batch_inputs count={len(batch_inputs)}, rank_ground_truths count={len(rank_ground_truths)}")
            framework.run_inference(batch_inputs)
                        
            # 收集指标（通信时延是本 rank的）
            latency = framework.get_comm_delay()
            # latency = framework.get_comm_delay_per_token()
            logger.debug(f"Rank {dp_rank} Batch {num_batches}: Latency={latency:.4f}s, rank_samples={len(rank_ground_truths)}")
                        
            # 提取本 rank生成的文本
            generated_texts = extract_generated_texts_batch(framework)
            logger.debug(f"Rank {dp_rank}: generated_texts count={len(generated_texts)}")
            
            # 评估本rank的样本精度（使用对应的ground_truths）
            for gen_text, gt in zip(generated_texts, rank_ground_truths):
                acc = evaluator.evaluate_single(gen_text, gt)
                local_correct += int(acc)
                local_total += 1
            
            # 累加本rank的时延
            total_latency += latency
            num_batches += 1
            
            # 清空批次缓存
            batch_inputs = []
            batch_ground_truths = []
    
    # 计算本rank的平均时延
    rank_batch_size = mini_batch_size / 2
    local_avg_latency = total_latency / rank_batch_size if rank_batch_size > 0 else 0.0
    
    # 聚合所有rank的结果
    global_accuracy, global_avg_latency = aggregate_metrics(
        local_correct, local_total, local_avg_latency
    )
    
    logger.info(f"Rank {dp_rank}: local_correct={local_correct}, local_total={local_total}, local_avg_latency={local_avg_latency:.4f}s")
    logger.info(f"Baseline collected: Latency={global_avg_latency:.4f}s, Accuracy={global_accuracy*100:.2f}% (total {count} samples across all ranks)")
    
    return global_avg_latency, global_accuracy


def train(config: Config,
          framework: InferenceFrameworkInterface,
          evaluator: AccuracyEvaluatorInterface,
          max_samples: Optional[int] = None):
    """
    主训练循环 - 批次对齐的迭代训练
    
    改进的训练流程：
    1. 预处理数据集，按批次分组（舍弃余数）
    2. 用原始模型推理所有批次，收集每个批次的基线指标（作为历史最优值初始化）
    3. 多次迭代：用带hook的模型逐批次推理，每个批次与该批次的历史最优值比较更新
    4. 每完成一次完整数据集推理后检查收敛性
    
    Args:
        config: 配置对象
        framework: 推理框架接口
        evaluator: 精度评估器
        max_samples: 最大样本数限制（用于调试），None表示使用全部数据
    """
    logger = get_train_logger()
    train_metrics_logger = TrainingMetricsLogger(config.training.log_interval)
    eval_logger_instance = EvaluationLogger()
    
    # 获取分布式信息
    dp_rank = framework.get_dp_rank()
    dp_size = framework.get_dp_size()
    
    logger.info("=" * 60)
    logger.info("Starting MoE Route Optimizer Training (Batch-Aligned Iteration)")
    logger.info(f"Rank: {get_rank()}, DP Rank: {dp_rank}, DP Size: {dp_size}, Is Main Process: {is_main_process()}")
    logger.info(f"Max samples limit: {max_samples if max_samples is not None else 'unlimited'}")
    logger.info("=" * 60)
    
    # 获取模型隐藏层维度
    hidden_size = framework.get_hidden_size()
    logger.info(f"Model hidden size: {hidden_size}")
    
    # 创建扰动生成器（Actor）
    actor = create_perturbation_generator(
        config.perturbation, 
        hidden_size, 
        config.training.device,
        dtype=torch.float32
    )
    
    logger.info(f"Actor parameters: {sum(p.numel() for p in actor.parameters())}")
    
    # 创建价值网络（Critic），用于PPO更新
    value_network = None
    if getattr(config.ppo, 'use_ppo', True):
        value_network = create_value_network(
            hidden_size=hidden_size,
            value_hidden_dim=getattr(config.ppo, 'value_hidden_dim', 64),
            device=config.training.device,
            dtype=torch.float32,
        )
        logger.info(f"Critic parameters: {sum(p.numel() for p in value_network.parameters())}")
        logger.info("PPO update mode enabled")
    else:
        logger.info("REINFORCE update mode (no Critic)")
    
    # 创建Hook管理器
    hook_manager = create_hook_manager(actor, for_moe=True)
    
    # 在第一个MoE块上注册hook
    first_moe_block = framework.get_first_moe_block()
    if first_moe_block is not None:
        hook_manager.register_hook(first_moe_block, "first_moe_block")
        logger.info(f"Registered hook on MoE block: {first_moe_block.__class__.__name__}")
    else:
        first_gate = framework.get_first_moe_gate()
        if first_gate is not None:
            hook_manager.register_hook(first_gate, "first_moe_gate")
            logger.warning("Using gate for hook registration (not recommended)")
        else:
            logger.error("Could not find MoE block or gate for hook registration!")
    
    # 创建训练组件
    reward_calculator = create_reward_calculator(config.reward)
    convergence_checker = create_convergence_checker(config.convergence)
    ppo_trainer = create_ppo_trainer(actor, config.ppo, config.training.device, value_network=value_network)
    
    # ============================================================
    # Step 1: 预处理数据集，按批次分组
    # ============================================================
    mini_batch_size = getattr(config.ppo, 'mini_batch_size', 20)
    logger.info(f"Preparing batched dataset with batch_size={mini_batch_size}, max_samples={max_samples}")
    
    batches = prepare_batched_dataset(evaluator, mini_batch_size, max_samples)
    num_batches = len(batches)
    
    if num_batches == 0:
        logger.error("No complete batches available! Please reduce batch_size or increase dataset size.")
        return
    
    logger.info(f"Dataset prepared: {num_batches} batches")
    
    # ============================================================
    # Step 2: 用原始模型收集所有批次的基线（禁用hook）
    # ============================================================
    hook_manager.disable()
    logger.info("Collecting baseline for all batches (without perturbation)...")
    
    batch_baselines = collect_all_baselines(framework, evaluator, batches)
    
    # 初始化历史最优值（精确度越大越好，时延越小越好）
    # best_metrics[batch_idx] = (best_latency, best_accuracy)
    import copy
    best_metrics = copy.deepcopy(batch_baselines)  # 复制基线作为初始历史最优值
    base_metrics = copy.deepcopy(batch_baselines)
    
    # 计算全局平均基线（用于收敛检查和日志）
    avg_baseline_latency = sum(bl[0] for bl in batch_baselines) / num_batches
    avg_baseline_accuracy = sum(bl[1] for bl in batch_baselines) / num_batches
    
    # 设置基线（用于reward计算和收敛检查）
    reward_calculator.set_baseline(avg_baseline_latency, avg_baseline_accuracy)
    convergence_checker.set_baseline(avg_baseline_latency, avg_baseline_accuracy)
    eval_logger_instance.log_baseline(avg_baseline_latency, avg_baseline_accuracy)
    
    logger.info(f"Global baseline: latency={avg_baseline_latency:.4f}s, accuracy={avg_baseline_accuracy*100:.2f}%")
    
    # 启用hook并设置训练模式
    hook_manager.enable()
    hook_manager.set_training_mode(True)
    
    # ============================================================
    # Step 3: 主训练循环 - 多次迭代数据集
    # ============================================================
    logger.info("Starting iterative training loop...")
    
    for iteration in range(config.convergence.max_episodes):
        iteration_rewards = []
        iteration_latency_reductions = []
        iteration_accuracies = []
        iteration_improved_batches = 0  # 记录本次迭代中改进的批次数
        
        logger.info(f"\n{'='*40} Iteration {iteration + 1} {'='*40}")
        
        # 遍历所有批次
        for batch_idx, (batch_inputs, batch_ground_truths) in enumerate(batches):
            # 清空之前收集的状态
            hook_manager.clear_buffer()
            
            # 获取该批次的历史最优值
            best_latency, best_accuracy = best_metrics[batch_idx]
            base_latency, base_accuracy = base_metrics[batch_idx]
            
            # 根据当前rank分配ground_truths
            _, rank_ground_truths = split_data_for_rank(
                batch_inputs, batch_ground_truths, dp_rank, dp_size
            )
            
            # 执行批量推理（hook会自动注入扰动）
            framework.run_inference(batch_inputs)
            
            # 获取指标
            comm_delay = framework.get_comm_delay()
            # comm_delay = framework.get_comm_delay_per_token()
            model_outputs = extract_generated_texts_batch(framework)
            # 打印第一个输出文本的后150个字符
            # print(f"Generated text for batch {batch_idx+1}: {model_outputs[0][-150:]}")
            collected_data = hook_manager.get_collected_data()
            # print(f"Collected {len(collected_data['all_states'])} states")
            
            # ============================================================
            # 修正：每个batch只产生一条CollectedState，按batch级别处理
            # ============================================================
            
            # 计算本rank的样本精度
            rank_batch_size = len(rank_ground_truths)
            local_avg_delay = comm_delay / rank_batch_size if rank_batch_size > 0 else 0.0
            
            local_correct = 0
            local_total = 0
            
            for model_output, gt in zip(model_outputs, rank_ground_truths):
                accuracy = evaluator.evaluate_single(model_output, gt)
                local_correct += int(accuracy)
                local_total += 1
            
            # 聚合各rank的精确度和时延
            global_accuracy, global_avg_delay = aggregate_metrics(
                local_correct, local_total, local_avg_delay
            )
            
            # ============================================================
            # 关键逻辑：与该批次的历史最优值比较，计算奖励
            # 精确度越大越好，时延越小越好
            # ============================================================
            
            # 判断是否有改进
            is_better_latency = global_avg_delay < best_latency
            is_better_accuracy = global_accuracy > best_accuracy
            # 综合判断：精度不下降的前提下时延有改进，或者精度有提升
            is_improved = (is_better_latency and global_accuracy >= best_accuracy * 0.95) or \
                         (is_better_accuracy and global_avg_delay <= best_latency * 1.05)
            
            # 计算相对于历史最优值的改进（用于reward计算）
            latency_reduction = (best_latency - global_avg_delay) / (best_latency + 1e-10)
            # accuracy_ratio = global_accuracy / (best_accuracy + 1e-10)

            # 计算相对于基线的改进
            latency_reduction_base = (base_latency - global_avg_delay) / (base_latency + 1e-10)
            # accuracy_ratio_base = global_accuracy / (base_accuracy + 1e-10)
            
            # 计算reward（使用当前批次的历史最优值作为基准）
            # 临时设置该批次的基准值用于reward计算
            original_baseline_latency = reward_calculator.baseline_latency
            original_baseline_accuracy = reward_calculator.baseline_accuracy
            reward_calculator.baseline_latency = best_latency
            reward_calculator.baseline_accuracy = best_accuracy
            
            reward, reward_components = reward_calculator.compute(global_avg_delay, global_accuracy)
            
            # 恢复原始基准值
            reward_calculator.baseline_latency = original_baseline_latency
            reward_calculator.baseline_accuracy = original_baseline_accuracy
            
            # 如果有改进，更新历史最优值
            if is_improved:
                # 更新最优值：取更小的时延和更大的精度
                new_best_latency = min(best_latency, global_avg_delay)
                new_best_accuracy = max(best_accuracy, global_accuracy)
                best_metrics[batch_idx] = (new_best_latency, new_best_accuracy)
                iteration_improved_batches += 1
                logger.info(f"Batch {batch_idx}: IMPROVED! latency: {best_latency:.4f}->{new_best_latency:.4f}, "
                           f"accuracy: {best_accuracy*100:.2f}%->{new_best_accuracy*100:.2f}%")
            
            # ============================================================
            # 简化：直接使用hook收集的数据，不重复计算
            # hook只会产生一条CollectedState，包含log_prob、selected_indices等
            # ============================================================
            if collected_data is not None and len(collected_data['all_states']) > 0:
                state = collected_data['all_states'][0]
                is_done = (batch_idx == num_batches - 1)
                
                # 直接传递hook收集的所有数据，避免重复计算
                ppo_trainer.collect_experience(
                    hidden_states=state.hidden_states,
                    reward=reward,
                    done=is_done,
                    log_prob=state.log_prob,
                    selected_indices=state.selected_indices,
                    perturb_types=state.perturb_types,
                )
                
                # 立即执行PPO或REINFORCE更新（每个batch更新一次）
                use_ppo = getattr(config.ppo, 'use_ppo', True) and value_network is not None
                if use_ppo:
                    update_stats = ppo_trainer.update_ppo()
                else:
                    update_stats = ppo_trainer.update()
                
                # 记录step日志
                train_metrics_logger.log_step(
                    reward=reward,
                    latency_reduction=latency_reduction,
                    accuracy=global_accuracy,
                    actor_loss=update_stats['actor_loss'],
                    critic_loss=update_stats.get('critic_loss', 0.0),
                )
            
            # 记录本batch的指标用于迭代统计
            iteration_rewards.append(reward)
            # iteration_latency_reductions.append(latency_reduction)
            iteration_accuracies.append(global_accuracy)
            iteration_latency_reductions.append(latency_reduction_base)
            # iteration_accuracies.append(accuracy_ratio_base)
        # ============================================================
        # 完成一次数据集迭代后：收敛检查（不再进行大规模PPO更新）
        # ============================================================
        
        # 计算本次迭代的平均指标
        avg_reward = np.mean(iteration_rewards)
        avg_latency_reduction = np.mean(iteration_latency_reductions)
        avg_accuracy = np.mean(iteration_accuracies)
        improvement_rate = iteration_improved_batches / num_batches
        
        # 获取最新的训练统计
        train_stats = ppo_trainer.get_training_stats()
        
        train_metrics_logger.log_episode(
            episode=iteration,
            total_reward=avg_reward,
            avg_latency_reduction=avg_latency_reduction,
            avg_accuracy=avg_accuracy,
            extra_info=f"Update Count: {train_stats.get('update_count', 0)}, "
                      f"Improved batches: {iteration_improved_batches}/{num_batches} ({improvement_rate*100:.1f}%)"
        )
            
        logger.info(f"Iteration {iteration + 1} summary: "
                    f"avg_reward={avg_reward:.4f}, "
                    f"avg_latency_reduction={avg_latency_reduction*100:.2f}%, "
                    f"avg_accuracy={avg_accuracy*100:.2f}%, "
                    f"improved_batches={iteration_improved_batches}/{num_batches}, "
                    f"total_updates={train_stats.get('update_count', 0)}")
        
        # 检查收敛
        should_stop, success, reason = convergence_checker.update(
            avg_reward, avg_latency_reduction, avg_accuracy
        )
        
        # 定期评估
        if convergence_checker.should_evaluate():
            eval_log = get_eval_logger()
            eval_log.info(f"[PERIODIC EVAL] ===== Starting Periodic Evaluation (Iteration {iteration + 1}) =====")
            hook_manager.set_training_mode(False)
            
            # 随机选择一些批次进行评估
            eval_batch_indices = random.sample(range(num_batches), min(10, num_batches))
            eval_latencies = []
            eval_accuracies = []
            
            for eval_batch_idx in eval_batch_indices:
                batch_inputs, batch_ground_truths = batches[eval_batch_idx]
                latency, accuracy = collect_batch_baseline(
                    framework, evaluator, batch_inputs, batch_ground_truths, eval_batch_idx
                )
                eval_latencies.append(latency)
                eval_accuracies.append(accuracy)
            
            hook_manager.set_training_mode(True)
            
            eval_latency = np.mean(eval_latencies)
            eval_accuracy = np.mean(eval_accuracies)
            eval_latency_reduction = (avg_baseline_latency - eval_latency) / avg_baseline_latency
            accuracy_ratio = eval_accuracy / avg_baseline_accuracy
            
            eval_log.info(f"[PERIODIC EVAL] Results: latency={eval_latency:.4f}s, accuracy={eval_accuracy*100:.2f}%")
            eval_log.info(f"[PERIODIC EVAL] latency_reduction={eval_latency_reduction*100:.2f}%, accuracy_ratio={accuracy_ratio:.4f}")
            
            eval_logger_instance.log_evaluation(
                episode=iteration,
                latency=eval_latency,
                accuracy=eval_accuracy,
                latency_reduction=eval_latency_reduction,
                accuracy_ratio=accuracy_ratio
            )
        
        # 检查是否应该停止
        if should_stop:
            final_metrics = convergence_checker.get_final_metrics()
            train_metrics_logger.log_convergence(success, reason, final_metrics)
            ppo_trainer.save_final_model(config.path.checkpoint_dir)
            break
    
    else:
        # 达到最大迭代次数
        logger.info("Reached max iterations, saving final model...")
        ppo_trainer.save_final_model(config.path.checkpoint_dir)
    
    # 输出最终的历史最优值统计
    final_best_latencies = [m[0] for m in best_metrics]
    final_best_accuracies = [m[1] for m in best_metrics]
    logger.info(f"\nFinal best metrics summary:")
    logger.info(f"  Avg best latency: {np.mean(final_best_latencies):.4f}s (baseline: {avg_baseline_latency:.4f}s)")
    logger.info(f"  Avg best accuracy: {np.mean(final_best_accuracies)*100:.2f}% (baseline: {avg_baseline_accuracy*100:.2f}%)")
    logger.info(f"  Latency improvement: {(avg_baseline_latency - np.mean(final_best_latencies))/avg_baseline_latency*100:.2f}%")
    
    # 清理
    hook_manager.remove_hooks()
    logger.info("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MoE Route Optimizer Training")
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--dataset', type=str, default='boolq', choices=['boolq', 'hellaswag', 'dummy'])
    parser.add_argument('--dataset-path', type=str, default='../datasets/datasets/boolq/default/0.0.0/35b264d03638db9f4ce671b711558bf7ff0f80d5', help='Dataset file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # 数据集大小控制参数（用于调试）
    parser.add_argument('--max-samples', type=int, default=None, 
                        help='Maximum number of samples to use from dataset (for debugging). '
                             'If not specified, use all data.')
    # 分布式相关参数
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--dp-size', type=int, default=2, help='Data parallel size')
    parser.add_argument('--enable-ep', default=True, help='Enable expert parallel')
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
    
    # 更新配置
    config.model.base_model_path = args.model_path
    config.training.seed = args.seed
    config.training.debug = args.debug
    
    # 初始化日志（会自动检测 rank）
    LoggerManager.setup(
        config.path.log_dir,
        config.path.train_log_file,
        config.path.eval_log_file,
        config.training.debug
    )
    
    logger = get_train_logger()
    logger.info(f"Configuration: {config}")
    if args.max_samples is None:
        args.max_samples = 600
    logger.info(f"Dataset size limit: {args.max_samples} samples (for debugging)")
    
    # 设置随机种子（注意：CUDA种子需要在vLLM加载后设置）
    set_seed(config.training.seed, set_cuda=False)
    
    # 创建推理框架适配器
    framework = create_vllm_adapter()
    framework.load_model(
        config.model.base_model_path,
        tensor_parallel_size=args.tp_size,
        data_parallel_size=args.dp_size,
        enable_expert_parallel=args.enable_ep,
        trust_remote_code=True,
    )
    # framework = create_hf_accelerate_adapter()
    # framework.load_model(
    #     config.model.base_model_path,
    #     data_parallel_size=2,
    #     expert_parallel_size=2,
    # )
    
    # vLLM加载完成后，设置CUDA种子
    set_seed(config.training.seed, set_cuda=True)
    
    # 创建评估器
    # dataset_path = args.dataset_path or os.path.join(config.path.dataset_dir, f"{args.dataset}.jsonl")
    dataset_path = "../datasets/datasets/boolq"
    evaluator = create_evaluator(args.dataset, dataset_path)
    
    # 开始训练（传入max_samples参数）
    train(config, framework, evaluator, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
