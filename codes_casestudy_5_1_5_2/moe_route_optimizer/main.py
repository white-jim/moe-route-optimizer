#!/usr/bin/env python
"""
MoE路由优化器 - 训练主入口
通过强化学习训练扰动生成器，优化MoE模型的推理时延

模型推理由框架（vLLM/SGLang）负责，框架内部自行处理专家并行等分布式部署，
训练脚本将框架视为黑盒：输入数据 -> 框架推理 -> 获取输出 -> 评估 -> 更新PPO模块。

使用方式：
    python main.py --model-path /path/to/model
    python main.py --model-path ../models/Qwen1.5-MoE-A2.7B
"""

import os
import sys
import argparse
import json
import torch
import random
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoConfig

# 添加项目路径（使用相对路径，兼容任何工作环境）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
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


def resolve_dataset_path(dataset_name: str, dataset_path: Optional[str] = None) -> str:
    """Resolve dataset path from CLI input or built-in defaults."""
    if dataset_path:
        return dataset_path

    dataset_defaults = {
        'boolq': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'boolq'),
        'hellaswag': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'hellaswag'),
        'mmlu': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'mmlu'),
        'arc': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'arc'),
        'logiqa': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'logiqa'),
        'gsm8k': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'gsm8k'),
        'lambada_openai': os.path.join(_REPO_ROOT, 'datasets', 'datasets', 'lambada_openai'),
        'dummy': '',
    }
    return dataset_defaults.get(dataset_name.lower(), '')


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


def _json_safe(value: Any) -> Any:
    """Convert tensors/numpy scalars to JSON-serializable objects."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: str, payload: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + '\n')


def _collect_samples(evaluator: AccuracyEvaluatorInterface,
                     max_samples: int) -> Tuple[List[str], List[Any]]:
    inputs, ground_truths = [], []
    for input_text, ground_truth in evaluator.get_dataset_iterator():
        inputs.append(input_text)
        ground_truths.append(ground_truth)
        if len(inputs) >= max_samples:
            break
    return inputs, ground_truths


def _evaluate_outputs(evaluator: AccuracyEvaluatorInterface,
                      prompts: List[str],
                      ground_truths: List[Any],
                      outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [
        output.get('generated_text', '') if isinstance(output, dict) else str(output)
        for output in outputs
    ]
    records = []
    for idx, (prompt, ground_truth, generated_text) in enumerate(
        zip(prompts, ground_truths, texts)
    ):
        records.append({
            'sample_idx': idx,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated_text': generated_text,
            'accuracy': evaluator.evaluate_single(generated_text, ground_truth),
        })
    return records


def _concat_worker_tensors(tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _build_sample_meta(framework: InferenceFrameworkInterface,
                       prompts: List[str],
                       ground_truths: List[Any]) -> List[Dict[str, Any]]:
    input_ids = _concat_worker_tensors(framework.get_last_input_ids())
    attention_masks = None
    if hasattr(framework, 'get_last_attention_masks'):
        attention_masks = _concat_worker_tensors(framework.get_last_attention_masks())

    meta = []
    for idx, (prompt, ground_truth) in enumerate(zip(prompts, ground_truths)):
        token_ids = []
        tokens = []
        if input_ids is not None and idx < input_ids.shape[0]:
            row_ids = input_ids[idx]
            if attention_masks is not None and idx < attention_masks.shape[0]:
                row_ids = row_ids[attention_masks[idx].bool()]
            token_ids = row_ids.detach().cpu().tolist()
            if getattr(framework, 'tokenizer', None) is not None:
                tokens = [
                    framework.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                    for token_id in token_ids
                ]
        meta.append({
            'sample_idx': idx,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'token_ids': token_ids,
            'tokens': tokens,
        })
    return meta


def _summarize_latency(baseline: List[Dict[str, Any]],
                       perturbed: List[Dict[str, Any]]) -> Dict[str, Any]:
    def aggregate(rows):
        totals = defaultdict(float)
        by_layer = defaultdict(lambda: defaultdict(float))
        for row in rows:
            key = int(row.get('layer_idx', -1))
            for metric in [
                'router_time',
                'dispatch_a2a_time',
                'local_expert_time',
                'combine_a2a_time',
            ]:
                value = float(row.get(metric, 0.0))
                totals[metric] += value
                by_layer[key][metric] += value
        totals['total_instrumented_time'] = sum(
            totals[m] for m in [
                'router_time',
                'dispatch_a2a_time',
                'local_expert_time',
                'combine_a2a_time',
            ]
        )
        return {'totals': dict(totals), 'by_layer': {k: dict(v) for k, v in by_layer.items()}}

    base_agg = aggregate(baseline)
    pert_agg = aggregate(perturbed)
    delta = {}
    for metric, base_value in base_agg['totals'].items():
        pert_value = pert_agg['totals'].get(metric, 0.0)
        delta[metric] = {
            'baseline': base_value,
            'perturbed': pert_value,
            'absolute_delta': pert_value - base_value,
            'relative_delta': (pert_value - base_value) / (base_value + 1e-12),
        }
    return {'baseline': base_agg, 'perturbed': pert_agg, 'delta': delta}


def _summarize_expert_load(route_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_layer = defaultdict(lambda: {
        'expert_token_counts': None,
        'num_records': 0,
        'num_tokens': 0,
    })
    for trace in route_traces:
        layer_idx = int(trace.get('layer_idx', -1))
        counts = trace.get('expert_token_counts')
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts if counts is not None else [], dtype=torch.float32)
        counts = counts.detach().cpu().to(torch.float32)
        row = by_layer[layer_idx]
        if row['expert_token_counts'] is None:
            row['expert_token_counts'] = counts.clone()
        else:
            max_len = max(row['expert_token_counts'].numel(), counts.numel())
            padded_old = torch.zeros(max_len)
            padded_new = torch.zeros(max_len)
            padded_old[:row['expert_token_counts'].numel()] = row['expert_token_counts']
            padded_new[:counts.numel()] = counts
            row['expert_token_counts'] = padded_old + padded_new
        row['num_records'] += 1
        row['num_tokens'] += int(trace.get('num_tokens', 0))

    summary = {}
    for layer_idx, row in by_layer.items():
        counts = row['expert_token_counts']
        if counts is None:
            counts = torch.zeros(0)
        total = float(counts.sum().item())
        mean = float(counts.mean().item()) if counts.numel() else 0.0
        std = float(counts.std(unbiased=False).item()) if counts.numel() else 0.0
        summary[layer_idx] = {
            'num_records': row['num_records'],
            'num_tokens': row['num_tokens'],
            'expert_token_counts': counts.to(torch.int64).tolist(),
            'total_assignments': total,
            'mean_load': mean,
            'std_load': std,
            'coefficient_of_variation': std / (mean + 1e-12),
            'max_load': float(counts.max().item()) if counts.numel() else 0.0,
            'min_load': float(counts.min().item()) if counts.numel() else 0.0,
        }
    return summary


def _compare_route_traces(baseline: List[Dict[str, Any]],
                          perturbed: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = {}
    for trace in baseline:
        key = (int(trace.get('rank', -1)), int(trace.get('layer_idx', -1)), int(trace.get('num_tokens', -1)))
        pairs.setdefault(key, {})['baseline'] = trace
    for trace in perturbed:
        key = (int(trace.get('rank', -1)), int(trace.get('layer_idx', -1)), int(trace.get('num_tokens', -1)))
        pairs.setdefault(key, {})['perturbed'] = trace

    layer_changes = defaultdict(list)
    for key, pair in pairs.items():
        if 'baseline' not in pair or 'perturbed' not in pair:
            continue
        base_idx = pair['baseline'].get('topk_indices')
        pert_idx = pair['perturbed'].get('topk_indices')
        if not isinstance(base_idx, torch.Tensor) or not isinstance(pert_idx, torch.Tensor):
            continue
        rows = min(base_idx.shape[0], pert_idx.shape[0])
        cols = min(base_idx.shape[1], pert_idx.shape[1])
        if rows == 0 or cols == 0:
            continue
        changed = (base_idx[:rows, :cols] != pert_idx[:rows, :cols]).any(dim=1)
        layer_changes[key[1]].append(float(changed.float().mean().item()))

    return {
        'route_change_rate_by_layer': {
            layer: float(np.mean(values)) for layer, values in layer_changes.items()
        },
        'num_matched_trace_pairs': sum(len(v) for v in layer_changes.values()),
    }


def _summarize_token_selection(hook_data: Optional[Dict[str, Any]],
                               sample_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not hook_data or not hook_data.get('all_states'):
        return {'num_hook_states': 0, 'selected_positions': [], 'position_histogram': {}}

    selected_positions = []
    token_counter = Counter()
    num_samples_seen = 0
    for state in hook_data['all_states']:
        selected = state.selected_indices.detach().cpu()
        if selected.dim() == 1:
            selected = selected.unsqueeze(0)
        for batch_row in range(selected.shape[0]):
            meta = sample_meta[num_samples_seen] if num_samples_seen < len(sample_meta) else {}
            tokens = meta.get('tokens', [])
            for pos in selected[batch_row].tolist():
                pos = int(pos)
                token_text = tokens[pos] if 0 <= pos < len(tokens) else ''
                selected_positions.append({
                    'sample_idx': meta.get('sample_idx', num_samples_seen),
                    'position': pos,
                    'relative_position': pos / max(len(tokens), 1),
                    'token': token_text,
                })
                if token_text:
                    token_counter[token_text] += 1
            num_samples_seen += 1

    position_hist = Counter(item['position'] for item in selected_positions)
    return {
        'num_hook_states': len(hook_data.get('all_states', [])),
        'num_selected_tokens': len(selected_positions),
        'selected_positions': selected_positions,
        'position_histogram': dict(position_hist),
        'top_selected_tokens': token_counter.most_common(50),
    }


def _plain_hook_data(hook_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not hook_data:
        return {'all_states': [], 'seq_lengths': []}
    states = []
    for state in hook_data.get('all_states', []):
        states.append({
            'hidden_states': state.hidden_states.detach().cpu(),
            'selected_indices': state.selected_indices.detach().cpu(),
            'perturb_dim_indices': state.perturb_dim_indices.detach().cpu(),
            'log_prob': state.log_prob.detach().cpu(),
            'perturbation': state.perturbation.detach().cpu(),
        })
    return {
        'all_states': states,
        'seq_lengths': list(hook_data.get('seq_lengths', [])),
    }


def _load_actor_checkpoint(actor: torch.nn.Module, checkpoint_path: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_path:
        return {'loaded': False, 'checkpoint_path': None}
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get('actor_state_dict')
            or checkpoint.get('model_state_dict')
            or checkpoint.get('state_dict')
            or checkpoint
        )
    else:
        state_dict = checkpoint
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    return {
        'loaded': True,
        'checkpoint_path': checkpoint_path,
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    }


def run_analysis(config: Config,
                 framework: InferenceFrameworkInterface,
                 evaluator: AccuracyEvaluatorInterface,
                 args):
    """Run one baseline and one perturbed pass for supplementary analysis."""
    logger = get_train_logger()
    output_root = args.analysis_output_dir or os.path.join(_REPO_ROOT, 'analysis_outputs')
    run_name = args.analysis_run_name or (
        f"{os.path.basename(args.model_path.rstrip(os.sep))}_{args.dataset}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    prompts, ground_truths = _collect_samples(evaluator, args.analysis_max_samples)
    if not prompts:
        raise RuntimeError("No samples available for analysis")

    framework.set_generation_config(max_new_tokens=args.analysis_max_new_tokens)
    framework.set_analysis_enabled(True)

    hidden_size = framework.get_hidden_size()
    actor = create_perturbation_generator(
        config.perturbation,
        hidden_size,
        config.training.device,
        dtype=torch.float32,
    )
    checkpoint_info = _load_actor_checkpoint(actor, args.checkpoint)
    hook_manager = create_hook_manager(actor, for_moe=True)
    first_moe_block = framework.get_first_moe_block()
    first_moe_gate = framework.get_first_moe_gate()
    if first_moe_block is not None:
        hook_manager.register_hook(first_moe_block, "first_moe_block")
    elif first_moe_gate is not None:
        hook_manager.register_hook(first_moe_gate, "first_moe_gate")
    else:
        raise RuntimeError("Could not find MoE block/gate for perturbation hook")

    run_config = {
        'mode': 'analysis',
        'model_path': args.model_path,
        'dataset': args.dataset,
        'dataset_path': args.dataset_path,
        'checkpoint': checkpoint_info,
        'seed': args.seed,
        'ep_size': args.ep_size,
        'max_samples': args.analysis_max_samples,
        'max_new_tokens': args.analysis_max_new_tokens,
        'output_dir': output_dir,
    }
    _write_json(os.path.join(output_dir, 'run_config.json'), run_config)

    hook_manager.disable()
    framework.clear_analysis_buffers()
    baseline_outputs = framework.run_inference(prompts, seed=args.seed)
    baseline_records = _evaluate_outputs(evaluator, prompts, ground_truths, baseline_outputs)
    sample_meta = _build_sample_meta(framework, prompts, ground_truths)
    baseline_routes = framework.get_route_traces()
    baseline_latency = framework.get_latency_breakdown()

    hook_manager.enable()
    hook_manager.clear_buffer()
    hook_manager.set_analysis_mode(deterministic=True, collect_states=True)
    framework.clear_analysis_buffers()
    perturbed_outputs = framework.run_inference(prompts, seed=args.seed)
    perturbed_records = _evaluate_outputs(evaluator, prompts, ground_truths, perturbed_outputs)
    perturbed_routes = framework.get_route_traces()
    perturbed_latency = framework.get_latency_breakdown()
    hook_data = hook_manager.get_collected_data()
    hook_data_plain = _plain_hook_data(hook_data)

    _write_json(os.path.join(output_dir, 'sample_meta.json'), sample_meta)
    _write_jsonl(os.path.join(output_dir, 'baseline_outputs.jsonl'), baseline_records)
    _write_jsonl(os.path.join(output_dir, 'perturbed_outputs.jsonl'), perturbed_records)
    torch.save(baseline_routes, os.path.join(output_dir, 'baseline_routes.pt'))
    torch.save(perturbed_routes, os.path.join(output_dir, 'perturbed_routes.pt'))
    torch.save(hook_data_plain, os.path.join(output_dir, 'hook_states.pt'))

    latency_summary = _summarize_latency(baseline_latency, perturbed_latency)
    expert_load_summary = {
        'baseline': _summarize_expert_load(baseline_routes),
        'perturbed': _summarize_expert_load(perturbed_routes),
    }
    token_selection_summary = _summarize_token_selection(hook_data, sample_meta)
    delta_route_correlation = _compare_route_traces(baseline_routes, perturbed_routes)

    _write_json(os.path.join(output_dir, 'latency_breakdown.json'), {
        'baseline_records': baseline_latency,
        'perturbed_records': perturbed_latency,
        'summary': latency_summary,
    })
    _write_json(os.path.join(output_dir, 'expert_load_summary.json'), expert_load_summary)
    _write_json(os.path.join(output_dir, 'token_selection_summary.json'), token_selection_summary)
    _write_json(os.path.join(output_dir, 'delta_route_correlation.json'), delta_route_correlation)

    hook_manager.remove_hooks()
    framework.set_analysis_enabled(False)
    logger.info(f"Analysis outputs written to: {output_dir}")
    return output_dir


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
    
    # 执行批量推理（框架内部自行处理分布式）
    framework.run_inference(batch_inputs)
    
    # 收集时延
    latency = framework.get_comm_delay()
    # 提取生成的文本
    generated_texts = extract_generated_texts_batch(framework)
    
    # 评估精度
    correct = 0
    total = len(batch_ground_truths)
    
    for gen_text, gt in zip(generated_texts, batch_ground_truths):
        acc = evaluator.evaluate_single(gen_text, gt)
        correct += int(acc)
    
    # 计算平均时延和精确度
    avg_latency = latency / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    logger.debug(f"Batch {batch_idx}: latency={avg_latency:.4f}s, accuracy={accuracy*100:.2f}%")
    
    return avg_latency, accuracy


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
    
    total_latency = 0.0
    correct = 0
    total = 0
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
            
            # 执行批量推理（框架内部自行处理分布式）
            framework.run_inference(batch_inputs)
                        
            # 收集指标
            latency = framework.get_comm_delay()
                        
            # 提取生成的文本
            generated_texts = extract_generated_texts_batch(framework)
            
            # 评估精度
            for gen_text, gt in zip(generated_texts, batch_ground_truths):
                acc = evaluator.evaluate_single(gen_text, gt)
                correct += int(acc)
                total += 1
            
            # 累加时延
            total_latency += latency
            num_batches += 1
            
            # 清空批次缓存
            batch_inputs = []
            batch_ground_truths = []
    
    # 计算平均时延
    avg_latency = total_latency / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Baseline collected: Latency={avg_latency:.4f}s, Accuracy={accuracy*100:.2f}% (total {count} samples)")
    
    return avg_latency, accuracy


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
    
    logger.info("=" * 60)
    logger.info("Starting MoE Route Optimizer Training (Batch-Aligned Iteration)")
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
            
            # 执行批量推理（hook会自动注入扰动，框架内部自行处理分布式）
            framework.run_inference(batch_inputs)
            
            # 获取指标
            comm_delay = framework.get_comm_delay()
            model_outputs = extract_generated_texts_batch(framework)
            collected_data = hook_manager.get_collected_data()
            
            # ============================================================
            # 计算精度和时延
            # ============================================================
            batch_size = len(batch_ground_truths)
            avg_delay = comm_delay / batch_size if batch_size > 0 else 0.0
            
            correct = 0
            for model_output, gt in zip(model_outputs, batch_ground_truths):
                accuracy = evaluator.evaluate_single(model_output, gt)
                correct += int(accuracy)
            
            batch_accuracy = correct / batch_size if batch_size > 0 else 0.0
            
            # ============================================================
            # 关键逻辑：与该批次的历史最优值比较，计算奖励
            # 精确度越大越好，时延越小越好
            # ============================================================
            
            # 判断是否有改进
            is_better_latency = avg_delay < best_latency
            is_better_accuracy = batch_accuracy > best_accuracy
            # 综合判断：精度不下降的前提下时延有改进，或者精度有提升
            is_improved = (is_better_latency and batch_accuracy >= best_accuracy * 0.95) or \
                         (is_better_accuracy and avg_delay <= best_latency * 1.05)
            
            # 计算相对于历史最优值的改进（用于reward计算）
            latency_reduction = (best_latency - avg_delay) / (best_latency + 1e-10)

            # 计算相对于基线的改进
            latency_reduction_base = (base_latency - avg_delay) / (base_latency + 1e-10)
            
            # 计算reward（使用当前批次的历史最优值作为基准）
            # 临时设置该批次的基准值用于reward计算
            original_baseline_latency = reward_calculator.baseline_latency
            original_baseline_accuracy = reward_calculator.baseline_accuracy
            reward_calculator.baseline_latency = best_latency
            reward_calculator.baseline_accuracy = best_accuracy
            
            reward, reward_components = reward_calculator.compute(avg_delay, batch_accuracy)
            
            # 恢复原始基准值
            reward_calculator.baseline_latency = original_baseline_latency
            reward_calculator.baseline_accuracy = original_baseline_accuracy
            
            # 如果有改进，更新历史最优值
            if is_improved:
                # 更新最优值：取更小的时延和更大的精度
                new_best_latency = min(best_latency, avg_delay)
                new_best_accuracy = max(best_accuracy, batch_accuracy)
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
                    perturb_dim_indices=state.perturb_dim_indices,
                )
                
                # 立即执行PPO或REINFORCE更新（每个batch更新一次）
                use_ppo = getattr(config.ppo, 'use_ppo', True) and value_network is not None
                if use_ppo:
                    update_stats = ppo_trainer.update_ppo()
                else:
                    update_stats = ppo_trainer.update()
                
                # 同步更新后的generator权重到worker进程
                hook_manager.sync_weights()
                
                # 记录step日志
                train_metrics_logger.log_step(
                    reward=reward,
                    latency_reduction=latency_reduction,
                    accuracy=batch_accuracy,
                    actor_loss=update_stats['actor_loss'],
                    critic_loss=update_stats.get('critic_loss', 0.0),
                )
            
            # 记录本batch的指标用于迭代统计
            iteration_rewards.append(reward)
            iteration_accuracies.append(batch_accuracy)
            iteration_latency_reductions.append(latency_reduction_base)
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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'analysis'],
                        help='Run training or supplementary analysis')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument(
        '--model-path',
        type=str,
        # default=os.path.join(_REPO_ROOT, 'models', 'Mixtral-8x7B')
        default=os.path.join(_REPO_ROOT, 'models', 'MoEPP-7B')
    )
    parser.add_argument('--dataset', type=str, default='lambada_openai', choices=['boolq', 'hellaswag', 'mmlu', 'arc', 'logiqa', 'gsm8k', 'lambada_openai', 'dummy'])
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # 数据集大小控制参数（用于调试）
    parser.add_argument('--max-samples', type=int, default=None, 
                        help='Maximum number of samples to use from dataset (for debugging). '
                             'If not specified, use all data.')
    # 框架并行配置参数（传递给推理框架，框架内部自行处理分布式）
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size (framework internal)')
    parser.add_argument('--dp-size', type=int, default=2, help='Data parallel size (framework internal)')
    parser.add_argument('--ep_size', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Perturbation actor checkpoint for --mode analysis')
    parser.add_argument('--analysis-output-dir', type=str, default=None,
                        help='Directory for supplementary analysis outputs')
    parser.add_argument('--analysis-run-name', type=str, default=None,
                        help='Subdirectory name for this analysis run')
    parser.add_argument('--analysis-max-samples', type=int, default=8,
                        help='Number of representative samples for analysis')
    parser.add_argument('--analysis-max-new-tokens', type=int, default=64,
                        help='Generation length used by analysis mode')
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
    config.path.project_root = _REPO_ROOT
    config.path.refresh_derived_paths()
    
    # 更新配置
    config.model.base_model_path = args.model_path
    config.training.seed = args.seed
    config.training.debug = args.debug
    try:
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.model.hidden_size = getattr(model_config, 'hidden_size', config.model.hidden_size)
        config.model.num_experts = (
            getattr(model_config, 'num_experts', None)
            or getattr(model_config, 'moe_num_experts', None)
            or getattr(model_config, 'num_local_experts', None)
            or getattr(model_config, 'n_routed_experts', None)
            or config.model.num_experts
        )
    except Exception:
        pass
    
    # 初始化日志
    LoggerManager.setup(
        config.path.log_dir,
        config.path.train_log_file,
        config.path.eval_log_file,
        config.training.debug
    )
    
    logger = get_train_logger()
    logger.info(f"Configuration: {config}")
    if args.max_samples is None:
        args.max_samples = 120
    logger.info(f"Dataset size limit: {args.max_samples} samples (for debugging)")
    
    # 设置随机种子（注意：CUDA种子需要在vLLM加载后设置）
    set_seed(config.training.seed, set_cuda=False)
    
    # 创建推理框架适配器
    framework = create_hf_accelerate_adapter(
        model_path=args.model_path,
        ep_size=args.ep_size,
    )
    
    # 加载完成后，设置CUDA种子
    set_seed(config.training.seed, set_cuda=True)
    
    args.dataset_path = resolve_dataset_path(args.dataset, args.dataset_path)

    # 创建评估器
    evaluator = create_evaluator(args.dataset, args.dataset_path)

    if args.mode == 'analysis':
        run_analysis(config, framework, evaluator, args)
        if hasattr(framework, 'cleanup'):
            framework.cleanup()
        return
    
    # 开始训练（传入max_samples参数）
    train(config, framework, evaluator, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
