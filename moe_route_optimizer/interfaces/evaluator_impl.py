"""
评估器实现模块
基于 lm-eval (lm-evaluation-harness) 框架实现模型评估

主要功能:
1. LMEvalEvaluator: 基于 lm-eval 的评估器，支持多种标准评测任务
2. ModelAdapter: 将自定义模型适配到 lm-eval 的 LM 接口
3. 保留原有的简单评估器作为备选

使用示例:
    from interfaces.evaluator_impl import LMEvalEvaluator, create_evaluator
    
    # 基本使用 - 使用默认缓存目录 (~/.cache/huggingface/datasets)
    evaluator = LMEvalEvaluator(
        tasks=["boolq", "hellaswag"],
        batch_size=32,
        limit=100  # 可选，限制样本数用于快速测试
    )
    
    # 指定缓存目录 - 避免重复下载数据集
    evaluator = LMEvalEvaluator(
        tasks=["boolq"],
        batch_size=32,
        cache_dir="/path/to/your/cache",  # 指定数据集缓存目录
        download_mode="reuse_dataset_if_exists"  # 复用已下载的数据集(默认)
    )
    
    # 全局设置缓存目录(通过环境变量)
    import os
    os.environ['HF_DATASETS_CACHE'] = '/path/to/your/cache'
    evaluator = LMEvalEvaluator(tasks=["boolq"])
    
    # 强制重新下载数据集
    evaluator = LMEvalEvaluator(
        tasks=["boolq"],
        download_mode="force_redownload"
    )
    
    results = evaluator.evaluate_model(model, tokenizer)
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Iterator, Tuple, Union
import json
import os
import re
import warnings
import logging

import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/mnt/data/lwy/vLLM-wrok/lm-evaluation-harness')
sys.path.append('/mnt/data/lwy/vLLM-wrok/moe_route_optimizer')

from interfaces.framework_interface import AccuracyEvaluatorInterface, InferenceFrameworkInterface
from config import get_eval_logger

# lm-eval imports
try:
    from lm_eval import evaluator as lm_evaluator
    from lm_eval.api.model import LM, TemplateLM
    from lm_eval.api.registry import register_model
    from lm_eval import tasks as lm_tasks
    from lm_eval.evaluator_utils import get_task_list
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    # 提供一个占位LM基类，这样PrecomputedLM可以定义但会抛出错误
    class LM:
        """Placeholder LM class when lm-eval is not available"""
        def __init__(self):
            raise ImportError(
                "lm-eval not available. Please install: pip install lm-eval"
            )
    TemplateLM = LM
    warnings.warn(
        "lm-eval not available. Please install: pip install lm-eval. "
        "Falling back to simple evaluators."
    )

logger = get_eval_logger()


# ==============================================================================
# lm-eval 模型适配器
# ==============================================================================

class FrameworkModelAdapter(nn.Module):
    """
    将 InferenceFrameworkInterface 适配为 HuggingFace 风格的模型
    用于与 lm-eval 的 HFLM 集成
    """
    
    def __init__(
        self,
        framework: InferenceFrameworkInterface,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[Any] = None,
        device: str = "cuda"
    ):
        """
        Args:
            framework: 推理框架接口实例
            model: 底层模型
            tokenizer: tokenizer 实例
            config: 模型配置 (可选)
            device: 设备
        """
        super().__init__()
        self.framework = framework
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # HuggingFace 兼容方法
        self.tie_weights = lambda: self
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        前向传播，返回带有 logits 属性的输出
        """
        # 通过框架执行推理
        if hasattr(self.model, 'forward'):
            output = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        else:
            # 使用框架的推理方法
            output = self.framework.run_inference({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                **kwargs
            })
        
        # 确保输出有 logits 属性
        if not hasattr(output, 'logits'):
            if isinstance(output, torch.Tensor):
                # 创建一个简单的输出对象
                class ModelOutput:
                    def __init__(self, logits):
                        self.logits = logits
                output = ModelOutput(output)
            elif isinstance(output, tuple):
                class ModelOutput:
                    def __init__(self, logits):
                        self.logits = logits
                output = ModelOutput(output[0])
        
        return output
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()


class PrecomputedLM(LM):
    """
    使用预计算结果的 LM 适配器
    
    这个类实现了 lm-eval 的 LM 接口，但不会重新运行推理，
    而是使用预先计算并缓存的结果。
    
    使用流程:
    1. 用户通过 get_all_requests() 获取所有需要处理的请求
    2. 用户使用自己的框架执行推理并缓存结果
    3. lm-eval 调用 loglikelihood/generate_until 时从缓存读取
    
    示例:
        precomputed_lm = PrecomputedLM(tokenizer)
        
        # 缓存 loglikelihood 结果
        precomputed_lm.cache_loglikelihood("context", "continuation", -1.5, True)
        
        # 缓存生成结果  
        precomputed_lm.cache_generation("prompt", "generated text")
    """
    
    def __init__(
        self,
        tokenizer: Any = None,
        device: str = "cuda"
    ):
        """
        Args:
            tokenizer: tokenizer 实例（用于编码/解码）
            device: 设备
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        
        # 缓存推理结果
        self._loglikelihood_cache: Dict[Tuple[str, str], Tuple[float, bool]] = {}
        self._generate_cache: Dict[str, str] = {}
        self._rolling_cache: Dict[str, float] = {}
    
    @property
    def device(self):
        return self._device
    
    @property
    def batch_size(self):
        return 1
    
    def cache_loglikelihood(
        self,
        context: str,
        continuation: str,
        log_prob: float,
        is_greedy: bool
    ):
        """
        缓存 loglikelihood 结果
        
        Args:
            context: 上下文文本
            continuation: 续写文本
            log_prob: 对数概率
            is_greedy: 是否是贪婪解码结果
        """
        self._loglikelihood_cache[(context, continuation)] = (log_prob, is_greedy)
    
    def cache_generation(self, context: str, generated_text: str):
        """
        缓存生成结果
        
        Args:
            context: 输入上下文
            generated_text: 生成的文本
        """
        self._generate_cache[context] = generated_text
    
    def cache_rolling(self, text: str, log_prob: float):
        """
        缓存 rolling loglikelihood 结果
        
        Args:
            text: 完整文本
            log_prob: 对数概率
        """
        self._rolling_cache[text] = log_prob
    
    def clear_cache(self):
        """清空所有缓存"""
        self._loglikelihood_cache.clear()
        self._generate_cache.clear()
        self._rolling_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'loglikelihood': len(self._loglikelihood_cache),
            'generate': len(self._generate_cache),
            'rolling': len(self._rolling_cache)
        }
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        从缓存中返回 loglikelihood 结果
        
        Args:
            requests: Instance 对象列表
        
        Returns:
            (log_prob, is_greedy) 元组列表
        """
        results = []
        
        for request in requests:
            context, continuation = request.args
            key = (context, continuation)
            
            if key in self._loglikelihood_cache:
                results.append(self._loglikelihood_cache[key])
            else:
                # 如果缓存中没有，返回一个默认的低分值
                logger.warning(f"Cache miss for loglikelihood: {context[:50]}...")
                results.append((-float('inf'), False))
        
        return results
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        从缓存中返回 rolling loglikelihood 结果
        """
        results = []
        
        for request in requests:
            text = request.args[0]
            
            if text in self._rolling_cache:
                results.append(self._rolling_cache[text])
            else:
                logger.warning(f"Cache miss for rolling loglikelihood: {text[:50]}...")
                results.append(-float('inf'))
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        """
        从缓存中返回生成结果
        """
        results = []
        
        for request in requests:
            context = request.args[0]
            
            if context in self._generate_cache:
                results.append(self._generate_cache[context])
            else:
                logger.warning(f"Cache miss for generate_until: {context[:50]}...")
                results.append("")
        
        return results


class DirectLMAdapter(LM):
    """
    直接实现 lm-eval 的 LM 接口的适配器
    适用于需要精细控制推理过程的场景
    """
    
    def __init__(
        self,
        framework: InferenceFrameworkInterface,
        batch_size: int = 1,
        max_length: int = 2048,
        device: str = "cuda"
    ):
        """
        Args:
            framework: 推理框架接口实例
            batch_size: 批处理大小
            max_length: 最大序列长度
            device: 设备
        """
        super().__init__()
        self.framework = framework
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device(device)
    
    @property
    def device(self):
        return self._device
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_length(self):
        return self._max_length
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        计算生成 continuation 的对数似然
        
        Args:
            requests: Instance 对象列表，每个包含 (context, continuation)
        
        Returns:
            (log_prob, is_greedy) 元组列表
        """
        results = []
        
        for request in requests:
            context, continuation = request.args
            
            # tokenize
            context_tokens = self.framework.tokenize(context)
            full_tokens = self.framework.tokenize(context + continuation)
            
            # 获取 continuation 部分的 token ids
            context_len = context_tokens['input_ids'].shape[-1]
            
            # 执行推理获取 logits
            with torch.no_grad():
                output = self.framework.run_inference(full_tokens)
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output[0] if isinstance(output, tuple) else output
            
            # 计算 continuation 部分的对数似然
            # logits shape: [batch, seq_len, vocab_size]
            continuation_logits = logits[:, context_len-1:-1, :]  # 取 continuation 对应位置的 logits
            continuation_ids = full_tokens['input_ids'][:, context_len:]  # continuation 的 token ids
            
            # 计算 log softmax
            log_probs = torch.log_softmax(continuation_logits, dim=-1)
            
            # 获取目标 token 的对数概率
            target_log_probs = log_probs.gather(
                dim=-1, 
                index=continuation_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # 求和得到总对数似然
            total_log_prob = target_log_probs.sum().item()
            
            # 检查是否是 greedy decoding 的结果
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == continuation_ids).all().item()
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        计算字符串的完整对数似然（无截断）
        用于计算 perplexity
        """
        results = []
        
        for request in requests:
            text = request.args[0]
            
            # tokenize
            tokens = self.framework.tokenize(text)
            
            with torch.no_grad():
                output = self.framework.run_inference(tokens)
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output[0]
            
            # 计算所有 token 的对数似然
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = tokens['input_ids'][:, 1:]
            
            target_log_probs = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            total_log_prob = target_log_probs.sum().item()
            results.append(total_log_prob)
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        """
        生成文本直到遇到停止序列
        
        Args:
            requests: Instance 对象列表，每个包含 (context, gen_kwargs)
        
        Returns:
            生成的文本列表
        """
        results = []
        
        for request in requests:
            context, gen_kwargs = request.args
            
            # 获取生成参数
            until = gen_kwargs.get('until', None)
            max_gen_tokens = gen_kwargs.get('max_gen_toks', 128)
            
            # tokenize context
            context_tokens = self.framework.tokenize(context)
            input_ids = context_tokens['input_ids']
            
            # 简单的自回归生成
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(max_gen_tokens):
                    output = self.framework.run_inference({
                        'input_ids': generated_ids,
                        'attention_mask': torch.ones_like(generated_ids)
                    })
                    
                    if hasattr(output, 'logits'):
                        next_token_logits = output.logits[:, -1, :]
                    else:
                        next_token_logits = output[:, -1, :]
                    
                    # greedy decoding
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # 解码检查停止条件
                    generated_text = self.framework.decode(generated_ids[0, input_ids.shape[-1]:])
                    
                    if until is not None:
                        should_stop = False
                        for stop_seq in until:
                            if stop_seq in generated_text:
                                should_stop = True
                                # 截断到停止序列
                                generated_text = generated_text.split(stop_seq)[0]
                                break
                        if should_stop:
                            break
            
            results.append(generated_text)
        
        return results


# ==============================================================================
# lm-eval 评估器
# ==============================================================================

class LMEvalEvaluator(AccuracyEvaluatorInterface):
    """
    基于 lm-eval 框架的评估器
    支持多种标准评测任务（boolq, hellaswag, mmlu, arc 等）
    
    兼容 AccuracyEvaluatorInterface 接口：
    - get_dataset_iterator() 返回 (input_text, ground_truth) 迭代器
    - evaluate_single() 评估单个样本
    """
    
    # 支持的任务列表
    SUPPORTED_TASKS = [
        "boolq", "hellaswag", "arc_easy", "arc_challenge",
        "winogrande", "piqa", "siqa", "openbookqa",
        "mmlu", "truthfulqa_mc", "lambada_openai"
    ]
    
    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        batch_size: int = 32,
        limit: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        device: str = "cuda",
        verbosity: str = "WARNING",
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None
    ):
        """
        Args:
            tasks: 要评估的任务列表,默认为 ["boolq"]
            batch_size: 批处理大小
            limit: 每个任务的样本数限制 (None 表示全部)
            num_fewshot: few-shot 样本数
            device: 设备
            verbosity: 日志级别
            cache_dir: 数据集缓存目录,默认为 ~/.cache/huggingface/datasets
                      可以通过设置环境变量 HF_DATASETS_CACHE 全局指定
            download_mode: 下载模式,可选值:
                - None 或 'reuse_dataset_if_exists': 复用已下载的数据集(默认)
                - 'reuse_cache_if_exists': 复用下载但重新处理数据集
                - 'force_redownload': 强制重新下载
        """
        if not LM_EVAL_AVAILABLE:
            raise ImportError(
                "lm-eval is not available. Please install: pip install lm-eval"
            )
            
        self.task_names = tasks or ["boolq"]
        self.batch_size = batch_size
        self.limit = limit
        self.num_fewshot = num_fewshot
        self.device = device
        self.verbosity = verbosity
        self.cache_dir = cache_dir
        self.download_mode = download_mode
            
        # 设置下载模式
        import datasets as hf_datasets
        if download_mode is None or download_mode == 'reuse_dataset_if_exists':
            self._download_mode = hf_datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
        elif download_mode == 'reuse_cache_if_exists':
            self._download_mode = hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
        elif download_mode == 'force_redownload':
            self._download_mode = hf_datasets.DownloadMode.FORCE_REDOWNLOAD
        else:
            logger.warning(f"Unknown download_mode '{download_mode}', using default REUSE_DATASET_IF_EXISTS")
            self._download_mode = hf_datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
            
        self._last_results: Optional[Dict] = None
        self._dataset_name = f"LMEval[{','.join(self.task_names)}]"
            
        # 加载 lm-eval 任务以获取数据集
        # 注意: TaskManager 本身不支持 cache_dir,需要在加载任务时通过配置传递
        self._task_manager = lm_tasks.TaskManager()
            
        # 为每个任务创建带缓存配置的 task config
        task_configs = []
        for task_name in self.task_names:
            task_config = {
                'task': task_name,
                'dataset_kwargs': {
                    'cache_dir': self.cache_dir,
                    'download_mode': self._download_mode
                }
            }
            task_configs.append(task_config)
            
        # 使用配置加载任务
        self._task_dict = lm_tasks.get_task_dict(task_configs, self._task_manager)
        self._task_objects = [x.task for x in get_task_list(self._task_dict)]
        
        # 预加载数据集样本
        self._samples: List[Tuple[str, Any, Any]] = []  # (input_text, ground_truth, task)
        self._load_samples()
        
        # 迭代器状态：记录当前迭代位置，支持断点续传
        self._current_index: int = 0
        
        logger.info(f"LMEvalEvaluator initialized with tasks: {self.task_names}")
        logger.info(f"Total samples loaded: {len(self._samples)}")
    
    def _load_samples(self):
        """
        从 lm-eval 任务中加载样本数据
        针对不同任务类型构建规范的prompt格式
        """
        self._samples = []
        
        for task in self._task_objects:
            task_name = task.config.task.lower() if hasattr(task.config, 'task') else ''
            
            # 获取文档迭代器
            if task.has_validation_docs():
                docs = task.validation_docs()
            elif task.has_test_docs():
                docs = task.test_docs()
            else:
                logger.warning(f"Task {task.config.task} has no validation or test docs")
                continue
            
            # 迭代文档
            count = 0
            for doc in docs:
                if self.limit is not None and count >= self.limit:
                    break
                
                # 针对不同任务类型构建规范的prompt
                input_text = self._build_prompt(task, task_name, doc)
                target = task.doc_to_target(doc)
                
                # 存储 (input_text, target, task_reference) 用于后续评估
                self._samples.append((input_text, target, task))
                count += 1
        
        logger.info(f"Loaded {len(self._samples)} samples from {len(self._task_objects)} tasks")
    
    def _build_prompt(self, task, task_name: str, doc: Dict) -> str:
        """
        针对不同任务类型构建规范的prompt格式
        
        Args:
            task: lm-eval任务对象
            task_name: 任务名称
            doc: 文档数据
        
        Returns:
            构建好的prompt文本
        """
        if 'boolq' in task_name:
            # BoolQ: 是非问答任务
            # 文档格式: {"question": "...", "passage": "...", "answer": True/False}
            passage = doc.get('passage', '')
            question = doc.get('question', '')
            
            # 构建引导模型输出Yes/No的prompt
            prompt = (
                f"Read the following passage and answer the question with only 'Yes' or 'No'.\n\n"
                f"Passage: {passage}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
            return prompt
        
        elif 'hellaswag' in task_name:
            # HellaSwag: 常识推理多选题
            context = doc.get('ctx', doc.get('context', ''))
            endings = doc.get('endings', doc.get('choices', []))
            
            options_text = "\n".join([f"{chr(65+i)}. {e}" for i, e in enumerate(endings)])
            prompt = (
                f"Complete the following sentence by choosing the most appropriate option.\n\n"
                f"Context: {context}\n\n"
                f"Options:\n{options_text}\n\n"
                f"The correct option is:"
            )
            return prompt
        
        elif 'arc' in task_name:
            # ARC: 科学问答多选题
            question = doc.get('question', '')
            choices = doc.get('choices', {})
            choice_labels = choices.get('label', [])
            choice_texts = choices.get('text', [])
            
            options_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
            prompt = (
                f"Answer the following science question by choosing the correct option.\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"The correct answer is:"
            )
            return prompt
        
        else:
            # 默认使用lm-eval原生的doc_to_text
            return task.doc_to_text(doc)
    
    @property
    def tasks(self):
        """兼容性属性，返回任务名称列表"""
        return self.task_names
    
    def get_dataset_name(self) -> str:
        return self._dataset_name
    
    def get_dataset_size(self) -> int:
        """返回数据集的实际样本数"""
        return len(self._samples)
    
    def get_dataset_iterator(self, batch_size: Optional[int] = None) -> Iterator[Tuple[str, Any]]:
        """
        返回数据集迭代器（支持断点续传）
        
        每次调用会从上次停止的位置继续迭代，而不是从头开始。
        迭代完所有样本后，会自动回到开头。
        
        Args:
            batch_size: 每次迭代返回的样本数量，None 表示逐个返回
                       如果指定 batch_size，迭代器将返回最多 batch_size 个样本后停止
        
        Returns:
            迭代器，每次返回 (input_text, ground_truth)
        
        示例:
            # 逐个获取样本（断点续传）
            for input_text, ground_truth in evaluator.get_dataset_iterator():
                # 处理样本...
                break  # 可以中途停止，下次调用会继续
            
            # 批量获取样本
            for input_text, ground_truth in evaluator.get_dataset_iterator(batch_size=32):
                # 最多获取 32 个样本
                pass
            
            # 重置迭代器到开头
            evaluator.reset_iterator()
        """
        if not self._samples:
            return
        
        yielded_count = 0
        total_samples = len(self._samples)
        
        while self._current_index < total_samples:
            input_text, target, task = self._samples[self._current_index]
            self._current_index += 1
            yielded_count += 1
            
            yield input_text, target
            
            # 如果指定了 batch_size 且已经返回了足够的样本，停止迭代
            if batch_size is not None and yielded_count >= batch_size:
                return
        
        # 如果迭代完了所有样本，重置索引到开头（支持循环迭代）
        if self._current_index >= total_samples:
            self._current_index = 0
    
    def reset_iterator(self) -> None:
        """
        重置迭代器到数据集开头
        
        调用此方法后，下次 get_dataset_iterator() 将从第一个样本开始
        """
        self._current_index = 0
        logger.debug("Iterator reset to the beginning")
    
    def get_iterator_position(self) -> Tuple[int, int]:
        """
        获取当前迭代器位置
        
        Returns:
            (current_index, total_samples) 元组
        """
        return self._current_index, len(self._samples)
    
    def set_iterator_position(self, position: int) -> None:
        """
        设置迭代器位置
        
        Args:
            position: 要设置的位置索引 (0-based)
        
        Raises:
            ValueError: 如果位置超出范围
        """
        if position < 0 or position > len(self._samples):
            raise ValueError(f"Position {position} out of range [0, {len(self._samples)}]")
        self._current_index = position
        logger.debug(f"Iterator position set to {position}")
    
    def evaluate_single(self, model_output: Any, ground_truth: Any) -> float:
        """
        评估单个样本的精度
        
        Args:
            model_output: 模型输出文本
            ground_truth: 标准答案（来自 doc_to_target）
        
        Returns:
            精度分数 (0-1)
        """
        if model_output is None:
            return 0.0

        import re
        def extract_yes_no(output_str: str):
            """
            从模型输出中提取明确的 yes / no 判断
            返回 'yes' / 'no' / None
            """
            # 只看前 10 个 token，避免后面解释干扰
            tokens = re.findall(r'\b(yes|no|true|false)\b', output_str.lower())
            if not tokens:
                return None
            
            first = tokens[0]
            if first in ['yes', 'true']:
                return 'yes'
            if first in ['no', 'false']:
                return 'no'
            return None
        
        # 将输出转换为字符串进行比较
        output_str = extract_yes_no(str(model_output).strip().lower())
        target_str = 'yes' if str(ground_truth).strip().lower() == '1' else 'no'
        
        # 尝试不同的匹配策略
        # 策略1: 完全匹配
        if output_str == target_str:
            return 1.0
        
        # 策略2: 目标包含在输出中
        if target_str in output_str:
            return 1.0
        
        # 策略3: 针对 yes/no 类型问题的特殊处理
        if target_str in ['yes', 'no', 'true', 'false']:
            # BoolQ 风格
            if target_str in ['yes', 'true']:
                if 'yes' in output_str or 'true' in output_str:
                    return 1.0
            else:  # no, false
                if 'no' in output_str or 'false' in output_str:
                    # 确保不是 "not" 而是明确的 "no"
                    if 'no' in output_str.split() or 'false' in output_str:
                        return 1.0
        
        # 策略4: 针对多选题的处理 (A, B, C, D 或 0, 1, 2, 3)
        # 尝试从输出中提取选项
        
        # 检查目标是否是选项索引
        if target_str.isdigit():
            target_idx = int(target_str)
            # 查找输出中的数字
            numbers = re.findall(r'\b(\d+)\b', output_str)
            if numbers and int(numbers[0]) == target_idx:
                return 1.0
            # 查找字母选项 (A=0, B=1, C=2, D=3)
            letters = re.findall(r'\b([A-Da-d])\b', output_str)
            if letters:
                letter_idx = ord(letters[0].upper()) - ord('A')
                if letter_idx == target_idx:
                    return 1.0
        
        # 检查目标是否是字母选项
        if len(target_str) == 1 and target_str.upper() in 'ABCD':
            target_letter = target_str.upper()
            # 查找输出中的字母
            letters = re.findall(r'\b([A-Da-d])\b', output_str.upper())
            if letters and letters[0].upper() == target_letter:
                return 1.0
        
        return 0.0
    
    def evaluate_batch(self, model_outputs: List[Any], 
                       ground_truths: List[Any]) -> float:
        """
        评估批量样本的平均精度
        """
        if not model_outputs or not ground_truths:
            return 0.0
        
        scores = [
            self.evaluate_single(output, truth)
            for output, truth in zip(model_outputs, ground_truths)
        ]
        return sum(scores) / len(scores)
    
    # def evaluate_model(
    #     self,
    #     model: Union[nn.Module, str, LM],
    #     tokenizer: Optional[Any] = None,
    #     framework: Optional[InferenceFrameworkInterface] = None,
    #     **kwargs
    # ) -> Dict[str, Any]:
    #     """
    #     使用 lm-eval 评估模型
        
    #     Args:
    #         model: 模型，可以是:
    #             - nn.Module: PyTorch 模型
    #             - str: 模型名称/路径（将使用 HFLM 加载）
    #             - LM: 已经是 lm-eval 的 LM 实例
    #         tokenizer: tokenizer 实例（如果 model 是 nn.Module）
    #         framework: 推理框架接口（可选，用于获取额外指标）
    #         **kwargs: 传递给 simple_evaluate 的额外参数
        
    #     Returns:
    #         评估结果字典，包含每个任务的指标
    #     """
    #     warnings.filterwarnings("ignore", message="Failed to get model SHA for")
        
    #     # 准备 lm-eval 兼容的模型
    #     if isinstance(model, str):
    #         # 字符串路径，直接传给 simple_evaluate
    #         lm_model = model
    #         model_args = kwargs.pop('model_args', '')
    #     elif isinstance(model, LM):
    #         # 已经是 LM 实例
    #         lm_model = model
    #         model_args = None
    #     elif isinstance(model, nn.Module):
    #         # PyTorch 模型，需要包装
    #         if tokenizer is None:
    #             raise ValueError("tokenizer is required when model is nn.Module")
            
    #         if framework is not None:
    #             # 使用框架适配器
    #             adapted_model = FrameworkModelAdapter(
    #                 framework=framework,
    #                 model=model,
    #                 tokenizer=tokenizer,
    #                 device=self.device
    #             )
    #         else:
    #             adapted_model = model
            
    #         # 包装为 HFLM
    #         lm_model = HFLM(
    #             pretrained=adapted_model,
    #             tokenizer=tokenizer,
    #             batch_size=self.batch_size,
    #             device=self.device
    #         )
    #         model_args = None
    #     else:
    #         raise TypeError(
    #             f"Unsupported model type: {type(model)}. "
    #             "Expected str, nn.Module, or lm_eval.api.model.LM"
    #         )
        
    #     # 执行评估
    #     logger.info(f"Starting lm-eval evaluation on tasks: {self.tasks}")
        
    #     results = lm_evaluator.simple_evaluate(
    #         model=lm_model,
    #         model_args=model_args,
    #         tasks=self.tasks,
    #         batch_size=self.batch_size,
    #         limit=self.limit,
    #         num_fewshot=self.num_fewshot,
    #         device=self.device,
    #         verbosity=self.verbosity,
    #         **kwargs
    #     )
        
    #     self._last_results = results
        
    #     # 提取关键指标
    #     if results is not None:
    #         self._log_results(results)
        
    #     return results
    
    # def evaluate_with_framework(
    #     self,
    #     framework: InferenceFrameworkInterface,
    #     model_path: str,
    #     **kwargs
    # ) -> Dict[str, Any]:
    #     """
    #     使用推理框架评估模型
        
    #     Args:
    #         framework: 推理框架接口实例
    #         model_path: 模型路径
    #         **kwargs: 额外参数
        
    #     Returns:
    #         评估结果
    #     """
    #     # 加载模型
    #     model = framework.load_model(model_path)
        
    #     # 获取 tokenizer（假设框架支持 tokenize/decode）
    #     class FrameworkTokenizer:
    #         def __init__(self, framework):
    #             self.framework = framework
            
    #         def __call__(self, text, **kwargs):
    #             return self.framework.tokenize(text)
            
    #         def decode(self, token_ids, **kwargs):
    #             return self.framework.decode(token_ids)
        
    #     tokenizer = FrameworkTokenizer(framework)
        
    #     return self.evaluate_model(
    #         model=model,
    #         tokenizer=tokenizer,
    #         framework=framework,
    #         **kwargs
    #     )
    
    def _log_results(self, results: Dict[str, Any]):
        """记录评估结果"""
        if 'results' not in results:
            return
        
        logger.info("=" * 60)
        logger.info("Evaluation Results:")
        logger.info("=" * 60)
        
        for task_name, metrics in results['results'].items():
            logger.info(f"\nTask: {task_name}")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {value}")
        
        logger.info("=" * 60)
    
    def get_accuracy(self, task_name: Optional[str] = None) -> float:
        """
        获取指定任务的准确率
        
        Args:
            task_name: 任务名称，None 表示所有任务的平均
        
        Returns:
            准确率 (0-1)
        """
        if self._last_results is None or 'results' not in self._last_results:
            return 0.0
        
        results = self._last_results['results']
        
        if task_name is not None:
            if task_name in results:
                metrics = results[task_name]
                return metrics.get('acc', metrics.get('acc,none', 0.0))
            return 0.0
        
        # 计算所有任务的平均准确率
        accuracies = []
        for metrics in results.values():
            if 'acc' in metrics:
                accuracies.append(metrics['acc'])
            elif 'acc,none' in metrics:
                accuracies.append(metrics['acc,none'])
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """获取完整的评估结果"""
        return self._last_results
    
    # ==========================================================================
    # 预计算模式支持 - 允许用户使用自己的框架运行推理，然后使用 lm-eval 评估
    # ==========================================================================
    
    def get_full_sample_iterator(self) -> Iterator[Tuple[str, Any, Any, Any]]:
        """
        返回包含完整信息的数据集迭代器
        
        Returns:
            迭代器，每次返回 (input_text, ground_truth, doc, task)
            - input_text: 输入文本
            - ground_truth: 标准答案
            - doc: 原始文档对象
            - task: 任务对象（用于后续 process_results）
        """
        for task in self._task_objects:
            if task.has_validation_docs():
                docs = task.validation_docs()
            elif task.has_test_docs():
                docs = task.test_docs()
            else:
                continue
            
            count = 0
            for doc in docs:
                if self.limit is not None and count >= self.limit:
                    break
                
                input_text = task.doc_to_text(doc)
                target = task.doc_to_target(doc)
                
                yield input_text, target, doc, task
                count += 1
    
    def get_all_requests(self) -> Dict[str, List]:
        """
        获取所有需要处理的 lm-eval 请求
        
        这个方法构建所有任务的请求，用户可以用自己的框架处理这些请求，
        然后将结果缓存到 PrecomputedLM 中。
        
        Returns:
            Dict 包含:
            - 'loglikelihood': List[(context, continuation, doc_id, task_name)]
            - 'generate_until': List[(context, gen_kwargs, doc_id, task_name)]
            - 'loglikelihood_rolling': List[(text, doc_id, task_name)]
        """
        requests = {
            'loglikelihood': [],
            'generate_until': [],
            'loglikelihood_rolling': []
        }
        
        for task in self._task_objects:
            task_name = task.config.task
            output_type = task.OUTPUT_TYPE
            
            if task.has_validation_docs():
                docs = task.validation_docs()
            elif task.has_test_docs():
                docs = task.test_docs()
            else:
                continue
            
            for doc_id, doc in enumerate(docs):
                if self.limit is not None and doc_id >= self.limit:
                    break
                
                # 构建请求 (简化版，实际的请求构建更复杂)
                ctx = task.fewshot_context(doc, 0)  # 0-shot context
                
                if output_type == 'loglikelihood':
                    # 对于 loglikelihood 类型，需要获取所有可能的 continuation
                    if hasattr(task, 'doc_to_choice'):
                        choices = task.doc_to_choice(doc)
                        for choice in choices:
                            requests['loglikelihood'].append(
                                (ctx, f" {choice}", doc_id, task_name)
                            )
                    else:
                        target = task.doc_to_target(doc)
                        requests['loglikelihood'].append(
                            (ctx, f" {target}", doc_id, task_name)
                        )
                
                elif output_type == 'multiple_choice':
                    # 多选题：为每个选项创建 loglikelihood 请求
                    if hasattr(task, 'doc_to_choice'):
                        choices = task.doc_to_choice(doc)
                        for choice in choices:
                            requests['loglikelihood'].append(
                                (ctx, f" {choice}", doc_id, task_name)
                            )
                
                elif output_type == 'generate_until':
                    gen_kwargs = task.config.generation_kwargs or {}
                    requests['generate_until'].append(
                        (ctx, gen_kwargs, doc_id, task_name)
                    )
                
                elif output_type == 'loglikelihood_rolling':
                    text = task.doc_to_text(doc) + task.doc_to_target(doc)
                    requests['loglikelihood_rolling'].append(
                        (text, doc_id, task_name)
                    )
        
        logger.info(f"Built requests: loglikelihood={len(requests['loglikelihood'])}, "
                   f"generate_until={len(requests['generate_until'])}, "
                   f"rolling={len(requests['loglikelihood_rolling'])}")
        
        return requests
    
    def evaluate_with_precomputed(
        self,
        precomputed_lm: 'PrecomputedLM',
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用预计算的结果进行评估
        
        这个方法使用已经缓存好推理结果的 PrecomputedLM 来进行评估，
        lm-eval 会从缓存中读取结果而不是重新运行推理。
        
        使用流程:
        1. 调用 get_all_requests() 获取所有需要处理的请求
        2. 用自己的框架处理这些请求并将结果缓存到 precomputed_lm
        3. 调用这个方法进行评估
        
        Args:
            precomputed_lm: 已经缓存好结果的 PrecomputedLM 实例
            **kwargs: 传递给 simple_evaluate 的额外参数
        
        Returns:
            评估结果字典
        
        示例:
            # 创建预计算 LM
            precomputed_lm = PrecomputedLM(tokenizer)
            
            # 获取所有请求
            requests = evaluator.get_all_requests()
            
            # 用自己的框架处理 loglikelihood 请求
            for ctx, cont, doc_id, task_name in requests['loglikelihood']:
                # 运行推理得到 log_prob
                log_prob = my_framework.compute_loglikelihood(ctx, cont)
                precomputed_lm.cache_loglikelihood(ctx, cont, log_prob, True)
            
            # 评估
            results = evaluator.evaluate_with_precomputed(precomputed_lm)
        """
        warnings.filterwarnings("ignore", message="Failed to get model SHA for")
        
        cache_stats = precomputed_lm.get_cache_stats()
        logger.info(f"Evaluating with precomputed results: {cache_stats}")
        
        # 使用 PrecomputedLM 进行评估
        results = lm_evaluator.simple_evaluate(
            model=precomputed_lm,
            tasks=self.tasks,
            batch_size=self.batch_size,
            limit=self.limit,
            num_fewshot=self.num_fewshot,
            device=self.device,
            verbosity=self.verbosity,
            **kwargs
        )
        
        self._last_results = results
        
        if results is not None:
            self._log_results(results)
        
        return results
    
    def evaluate_single_with_logits(
        self,
        sample_idx: int,
        logits: torch.Tensor,
        tokenizer: Any
    ) -> float:
        """
        使用 logits 评估单个样本（用于多选题）
        
        对于多选题任务（如 BoolQ, HellaSwag），评估是基于比较不同选项的
        log probability，而不是生成的文本。
        
        Args:
            sample_idx: 样本索引
            logits: 模型输出的 logits [seq_len, vocab_size]
            tokenizer: tokenizer 实例
        
        Returns:
            精度分数 (0-1)
        """
        if sample_idx >= len(self._samples):
            return 0.0
        
        input_text, target, task = self._samples[sample_idx]
        
        # 获取原始文档
        if task.has_validation_docs():
            docs = list(task.validation_docs())
        elif task.has_test_docs():
            docs = list(task.test_docs())
        else:
            return 0.0
        
        # 找到对应的文档（简化实现）
        doc_idx = sample_idx % len(docs) if docs else 0
        if doc_idx >= len(docs):
            return 0.0
        doc = docs[doc_idx]
        
        # 对于多选题，计算每个选项的 log probability
        if hasattr(task, 'doc_to_choice'):
            choices = task.doc_to_choice(doc)
            correct_idx = int(target) if isinstance(target, (int, str)) and str(target).isdigit() else 0
            
            # 计算每个选项的对数概率
            log_probs = []
            for choice in choices:
                choice_text = f" {choice}"
                choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
                
                # 简化：使用最后几个 token 的 logits
                if len(choice_ids) > 0 and logits.dim() >= 2:
                    # 取最后 len(choice_ids) 个位置的 logits
                    choice_logits = logits[-len(choice_ids):, :]
                    choice_ids_tensor = torch.tensor(choice_ids, device=logits.device)
                    
                    # 计算 log softmax
                    log_softmax = torch.log_softmax(choice_logits, dim=-1)
                    
                    # 获取目标 token 的对数概率
                    target_log_probs = log_softmax.gather(
                        dim=-1,
                        index=choice_ids_tensor.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    total_log_prob = target_log_probs.sum().item()
                    log_probs.append(total_log_prob)
                else:
                    log_probs.append(-float('inf'))
            
            # 选择概率最高的选项
            if log_probs:
                predicted_idx = log_probs.index(max(log_probs))
                return 1.0 if predicted_idx == correct_idx else 0.0
        
        return 0.0
    
    def create_precomputed_lm(self, tokenizer: Any = None) -> 'PrecomputedLM':
        """
        创建一个新的 PrecomputedLM 实例
        
        Args:
            tokenizer: 可选的 tokenizer
        
        Returns:
            PrecomputedLM 实例
        """
        return PrecomputedLM(tokenizer=tokenizer, device=self.device)


def create_lm_eval_evaluator(
    tasks: Optional[List[str]] = None,
    **kwargs
) -> LMEvalEvaluator:
    """
    创建 lm-eval 评估器的便捷函数
    
    Args:
        tasks: 任务列表
        **kwargs: 传递给 LMEvalEvaluator 的参数
    
    Returns:
        LMEvalEvaluator 实例
    """
    return LMEvalEvaluator(tasks=tasks, **kwargs)


# ==============================================================================
# 保留原有的简单评估器（作为备选）
# ==============================================================================

class BaseEvaluator(AccuracyEvaluatorInterface):
    """
    评估器基类（简单实现，不依赖 lm-eval）
    
    支持以下功能：
    1. 从多种格式加载数据集（JSON, JSONL, Arrow, HuggingFace cache）
    2. 断点续传：迭代器支持从上次停止的位置继续
    3. 批量获取：支持指定 batch_size 参数
    """
    
    def __init__(self, dataset_path: str, split: str = "validation"):
        """
        Args:
            dataset_path: 数据集文件路径或 HuggingFace datasets 缓存目录
            split: 数据集划分 (train/validation/test)
        """
        self.logger = get_eval_logger()
        self.dataset_path = dataset_path
        self.split = split
        self._data: List[Dict] = []
        self._loaded = False
        
        # 断点续传支持：记录当前迭代位置
        self._current_index: int = 0
    
    def _load_data(self):
        """加载数据集，支持多种格式"""
        if self._loaded:
            return
        
        if not os.path.exists(self.dataset_path):
            self.logger.warning(f"Dataset not found: {self.dataset_path}")
            return
        
        # 判断是文件还是目录
        if os.path.isdir(self.dataset_path):
            # HuggingFace datasets 缓存目录
            self._load_from_hf_cache()
        elif self.dataset_path.endswith('.arrow'):
            # Arrow 文件
            self._load_from_arrow(self.dataset_path)
        elif self.dataset_path.endswith('.jsonl'):
            # JSONL 文件
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self._data = [json.loads(line) for line in f]
        else:
            # JSON 文件
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        
        self._loaded = True
        self.logger.info(f"Loaded {len(self._data)} samples from {self.dataset_path}")
    
    def _load_from_hf_cache(self):
        """从 HuggingFace datasets 缓存目录加载数据"""
        # 尝试使用 datasets 库加载
        try:
            from datasets import load_from_disk, Dataset, DatasetDict
            import pyarrow as pa
            
            # 如果没找到 arrow 文件，尝试直接加载目录
            dataset = load_from_disk(self.dataset_path)
            
            # 处理 DatasetDict（包含多个 split 的数据集）
            if isinstance(dataset, DatasetDict):
                if self.split in dataset:
                    self._data = list(dataset[self.split])
                else:
                    # 如果指定的 split 不存在，使用第一个可用的 split
                    available_splits = list(dataset.keys())
                    if available_splits:
                        self.logger.warning(
                            f"Split '{self.split}' not found, using '{available_splits[0]}' instead. "
                            f"Available splits: {available_splits}"
                        )
                        self._data = list(dataset[available_splits[0]])
                    else:
                        raise ValueError(f"No splits found in dataset: {self.dataset_path}")
            else:
                # 单个 Dataset 对象
                self._data = list(dataset)
                
        except ImportError:
            self.logger.warning("datasets library not installed, trying to load arrow file directly")
            arrow_file = self._find_arrow_file()
            if arrow_file:
                self._load_from_arrow(arrow_file)
            else:
                raise ValueError(f"Cannot load HuggingFace cache without datasets library: {self.dataset_path}")
    
    def _find_arrow_file(self) -> Optional[str]:
        """在缓存目录中查找对应 split 的 arrow 文件"""
        # 递归搜索目录
        for root, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('.arrow'):
                    # 优先匹配指定的 split
                    if self.split in f.lower():
                        return os.path.join(root, f)
        
        # 如果没找到匹配的 split，返回第一个 arrow 文件
        for root, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('.arrow'):
                    return os.path.join(root, f)
        return None
    
    def _load_from_arrow(self, arrow_path: str):
        """从 Arrow 文件加载数据"""
        try:
            import pyarrow as pa
            
            # 读取 arrow 文件
            with pa.memory_map(arrow_path, 'r') as source:
                table = pa.ipc.open_file(source).read_all()
            
            # 转换为字典列表
            self._data = table.to_pydict()
            # 转换格式: {col1: [v1,v2], col2: [v1,v2]} -> [{col1:v1, col2:v1}, {col1:v2, col2:v2}]
            if self._data:
                keys = list(self._data.keys())
                num_rows = len(self._data[keys[0]])
                self._data = [
                    {k: self._data[k][i] for k in keys}
                    for i in range(num_rows)
                ]
        except ImportError:
            raise ImportError("pyarrow is required to load arrow files. Install with: pip install pyarrow")
    
    def get_dataset_size(self) -> int:
        """获取数据集大小"""
        self._load_data()
        return len(self._data)
    
    def reset_iterator(self) -> None:
        """
        重置迭代器到数据集开头
        
        调用此方法后，下次 get_dataset_iterator() 将从第一个样本开始
        """
        self._current_index = 0
        self.logger.debug("Iterator reset to the beginning")
    
    def get_iterator_position(self) -> Tuple[int, int]:
        """
        获取当前迭代器位置
        
        Returns:
            (current_index, total_samples) 元组
        """
        self._load_data()
        return self._current_index, len(self._data)
    
    def set_iterator_position(self, position: int) -> None:
        """
        设置迭代器位置
        
        Args:
            position: 要设置的位置索引 (0-based)
        
        Raises:
            ValueError: 如果位置超出范围
        """
        self._load_data()
        if position < 0 or position > len(self._data):
            raise ValueError(f"Position {position} out of range [0, {len(self._data)}]")
        self._current_index = position
        self.logger.debug(f"Iterator position set to {position}")


class BoolQEvaluator(BaseEvaluator):
    """
    BoolQ数据集评估器
    
    BoolQ是一个是非问答数据集，支持以下功能：
    1. 断点续传：迭代器支持从上次停止的位置继续
    2. 自动构建提示词：根据 passage 和 question 构建规范的 prompt
    3. 完善的评估逻辑：支持多种输出格式的解析
    """
    
    def __init__(self, dataset_path: str, split: str = "validation"):
        super().__init__(dataset_path, split)
        self._dataset_name = "BoolQ"
    
    def get_dataset_name(self) -> str:
        return self._dataset_name
    
    def _build_prompt(self, doc: Dict) -> str:
        """
        针对 BoolQ 任务构建规范的 prompt 格式
        
        Args:
            doc: 文档数据，包含 passage 和 question 字段
        
        Returns:
            构建好的 prompt 文本
        """
        passage = doc.get('passage', '')
        question = doc.get('question', '')
        
        # 构建引导模型输出 Yes/No 的 prompt
        prompt = (
            f"Read the following passage and answer the question with only 'Yes' or 'No'.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt
    
    def get_dataset_iterator(self, batch_size: Optional[int] = None) -> Iterator[Tuple[str, Any]]:
        """
        返回数据集迭代器（支持断点续传）
        
        每次调用会从上次停止的位置继续迭代，而不是从头开始。
        迭代完所有样本后，会自动回到开头。
        
        Args:
            batch_size: 每次迭代返回的样本数量，None 表示逐个返回
                       如果指定 batch_size，迭代器将返回最多 batch_size 个样本后停止
        
        Returns:
            迭代器，每次返回 (input_text, ground_truth)
        
        示例:
            # 逐个获取样本（断点续传）
            for input_text, ground_truth in evaluator.get_dataset_iterator():
                # 处理样本...
                break  # 可以中途停止，下次调用会继续
            
            # 批量获取样本
            for input_text, ground_truth in evaluator.get_dataset_iterator(batch_size=32):
                # 最多获取 32 个样本
                pass
            
            # 重置迭代器到开头
            evaluator.reset_iterator()
        """
        self._load_data()
        
        if not self._data:
            return
        
        yielded_count = 0
        total_samples = len(self._data)
        
        while self._current_index < total_samples:
            item = self._data[self._current_index]
            self._current_index += 1
            yielded_count += 1
            
            # BoolQ格式: {"question": "...", "passage": "...", "answer": true/false}
            answer = item.get('answer', item.get('label', False))
            
            # 使用 _build_prompt 构造输入文本
            input_text = self._build_prompt(item)
            
            yield input_text, answer
            
            # 如果指定了 batch_size 且已经返回了足够的样本，停止迭代
            if batch_size is not None and yielded_count >= batch_size:
                return
        
        # 如果迭代完了所有样本，重置索引到开头（支持循环迭代）
        if self._current_index >= total_samples:
            self._current_index = 0
    
    def evaluate_single(self, model_output: Any, ground_truth: Any) -> float:
        """
        评估单个样本的精度
        
        支持多种输出格式的解析：
        1. 完全匹配 yes/no
        2. 包含 yes/no/true/false 关键词
        3. 数字格式 (1/0)
        
        Args:
            model_output: 模型输出文本
            ground_truth: 标准答案（来自数据集，可以是 bool/str/'1'/'0'）
        
        Returns:
            精度分数 (0-1)
        """
        if model_output is None:
            return 0.0
        
        def extract_yes_no(output_str: str) -> Optional[str]:
            """
            从模型输出中提取明确的 yes / no 判断
            返回 'yes' / 'no' / None
            """
            # 提取输出中的关键词
            tokens = re.findall(r'\b(yes|no|true|false)\b', output_str.lower())
            if not tokens:
                return None
            
            first = tokens[0]
            if first in ['yes', 'true']:
                return 'yes'
            if first in ['no', 'false']:
                return 'no'
            return None
        
        # 提取模型输出的答案
        output_str = str(model_output).strip().lower()
        predicted_answer = extract_yes_no(output_str)
        
        # 标准化 ground_truth
        # ground_truth 可能是: bool(True/False), str('yes'/'no'/'1'/'0'), int(1/0)
        if isinstance(ground_truth, bool):
            expected_answer = 'yes' if ground_truth else 'no'
        elif isinstance(ground_truth, str):
            gt_lower = ground_truth.strip().lower()
            if gt_lower in ['yes', 'true', '1']:
                expected_answer = 'yes'
            elif gt_lower in ['no', 'false', '0']:
                expected_answer = 'no'
            else:
                expected_answer = 'yes' if gt_lower else 'no'
        elif isinstance(ground_truth, (int, float)):
            expected_answer = 'yes' if ground_truth == 1 else 'no'
        else:
            expected_answer = 'yes' if bool(ground_truth) else 'no'
        
        # 策略1: 完全匹配
        if predicted_answer == expected_answer:
            return 1.0
        
        # 策略2: 如果没有提取到明确的 yes/no，尝试其他匹配方式
        if predicted_answer is None:
            # 检查输出中是否包含期望的答案
            if expected_answer in output_str:
                return 1.0
            # 检查是否包含对应的 true/false
            if expected_answer == 'yes' and 'true' in output_str:
                return 1.0
            if expected_answer == 'no' and 'false' in output_str:
                return 1.0
        
        return 0.0
    
    def evaluate_batch(self, model_outputs: List[Any], 
                       ground_truths: List[Any]) -> float:
        """评估批量样本"""
        if not model_outputs or not ground_truths:
            return 0.0
        
        scores = [
            self.evaluate_single(output, truth) 
            for output, truth in zip(model_outputs, ground_truths)
        ]
        return sum(scores) / len(scores)


class HellaSwagEvaluator(BaseEvaluator):
    """
    HellaSwag数据集评估器（简单实现）
    HellaSwag是一个常识推理数据集，需要从多个选项中选择正确答案
    """
    
    def __init__(self, dataset_path: str, split: str = "validation"):
        super().__init__(dataset_path, split)
        self._dataset_name = "HellaSwag"
    
    def get_dataset_name(self) -> str:
        return self._dataset_name
    
    def get_dataset_iterator(self) -> Iterator[Tuple[str, Any]]:
        """
        获取数据集迭代器
        
        Returns:
            迭代器，每次返回 (上下文+选项文本, 正确选项索引)
        """
        self._load_data()
        
        for item in self._data:
            # HellaSwag格式: {"ctx": "...", "endings": ["...", "...", ...], "label": 0-3}
            context = item.get('ctx', item.get('context', ''))
            endings = item.get('endings', item.get('choices', []))
            label = item.get('label', 0)
            
            # 构造输入文本
            options_text = "\n".join([f"{i}. {e}" for i, e in enumerate(endings)])
            input_text = f"Context: {context}\n\nOptions:\n{options_text}\n\nThe correct option is:"
            
            yield input_text, label
    
    def evaluate_single(self, model_output: Any, ground_truth: Any) -> float:
        """
        评估单个样本
        
        Args:
            model_output: 模型输出（应该是选项索引）
            ground_truth: 正确选项索引
        
        Returns:
            1.0 如果正确，0.0 如果错误
        """
        if model_output is None:
            return 0.0
        
        # 尝试从输出中提取数字
        output_text = str(model_output).strip()
        
        # 查找第一个数字
        numbers = re.findall(r'\d+', output_text)
        if numbers:
            predicted = int(numbers[0])
        else:
            # 尝试解析字母选项 (A, B, C, D -> 0, 1, 2, 3)
            output_upper = output_text.upper()
            if 'A' in output_upper:
                predicted = 0
            elif 'B' in output_upper:
                predicted = 1
            elif 'C' in output_upper:
                predicted = 2
            elif 'D' in output_upper:
                predicted = 3
            else:
                return 0.0
        
        expected = int(ground_truth) if not isinstance(ground_truth, int) else ground_truth
        
        return 1.0 if predicted == expected else 0.0
    
    def evaluate_batch(self, model_outputs: List[Any], 
                       ground_truths: List[Any]) -> float:
        """评估批量样本"""
        if not model_outputs or not ground_truths:
            return 0.0
        
        scores = [
            self.evaluate_single(output, truth) 
            for output, truth in zip(model_outputs, ground_truths)
        ]
        return sum(scores) / len(scores)


class DummyEvaluator(AccuracyEvaluatorInterface):
    """
    虚拟评估器
    用于测试或当实际数据集不可用时
    """
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self._dataset_name = "Dummy"
    
    def get_dataset_name(self) -> str:
        return self._dataset_name
    
    def get_dataset_size(self) -> int:
        return self.num_samples
    
    def get_dataset_iterator(self) -> Iterator[Tuple[str, Any]]:
        """生成虚拟数据"""
        for i in range(self.num_samples):
            yield f"Sample {i}: What is the answer?", i % 2 == 0
    
    def evaluate_single(self, model_output: Any, ground_truth: Any) -> float:
        """随机返回0或1（用于测试）"""
        import random
        return random.random()
    
    def evaluate_batch(self, model_outputs: List[Any], 
                       ground_truths: List[Any]) -> float:
        """返回平均得分"""
        scores = [self.evaluate_single(o, g) for o, g in zip(model_outputs, ground_truths)]
        return sum(scores) / len(scores) if scores else 0.0


# ==============================================================================
# 工厂函数
# ==============================================================================

def create_evaluator(
    dataset_name: str, 
    dataset_path: str = "",
    split: str = "validation",
    use_lm_eval: bool = False,
    **kwargs
) -> AccuracyEvaluatorInterface:
    """
    创建评估器的工厂函数
    
    Args:
        dataset_name: 数据集名称 ("boolq", "hellaswag", "dummy", 或 lm-eval 支持的任务)
        dataset_path: 数据集文件路径或 HuggingFace datasets 缓存目录（仅用于简单评估器）
        split: 数据集划分 (train/validation/test)
        use_lm_eval: 是否使用 lm-eval 框架
        **kwargs: 传递给评估器的额外参数
                  - cache_dir: 数据集缓存目录（默认: /mnt/data/lwy/vLLM-wrok/datasets）
                  - download_mode: 下载模式（默认: reuse_dataset_if_exists）
    
    Returns:
        评估器实例
    """
    dataset_name_lower = dataset_name.lower()
    
    # 如果启用 lm-eval 且可用
    if use_lm_eval and LM_EVAL_AVAILABLE:
        # 检查是否是 lm-eval 支持的任务
        if dataset_name_lower in LMEvalEvaluator.SUPPORTED_TASKS or dataset_name_lower == "lm_eval":
            tasks = kwargs.pop('tasks', [dataset_name_lower])
            if dataset_name_lower == "lm_eval":
                tasks = kwargs.pop('tasks', ["boolq", "hellaswag"])
            
            # 设置默认的缓存目录和下载模式
            if 'cache_dir' not in kwargs:
                kwargs['cache_dir'] = '/mnt/data/lwy/vLLM-wrok/datasets'
            if 'download_mode' not in kwargs:
                kwargs['download_mode'] = 'reuse_dataset_if_exists'
            
            return LMEvalEvaluator(tasks=tasks, **kwargs)
    
    # 回退到简单评估器
    if dataset_name_lower == "boolq":
        return BoolQEvaluator(dataset_path, split=split)
    elif dataset_name_lower == "hellaswag":
        return HellaSwagEvaluator(dataset_path, split=split)
    elif dataset_name_lower == "dummy":
        return DummyEvaluator()
    else:
        # 尝试创建 lm-eval 评估器
        if LM_EVAL_AVAILABLE:
            # 设置默认的缓存目录和下载模式
            if 'cache_dir' not in kwargs:
                kwargs['cache_dir'] = '/mnt/data/lwy/vLLM-wrok/datasets'
            if 'download_mode' not in kwargs:
                kwargs['download_mode'] = 'reuse_dataset_if_exists'
            
            return LMEvalEvaluator(tasks=[dataset_name_lower], **kwargs)
        raise ValueError(f"Unknown dataset: {dataset_name}. lm-eval not available for extended tasks.")
