"""
评估器实现模块

主要功能:
1. BoolQEvaluator: BoolQ 数据集评估器
2. HellaSwagEvaluator: HellaSwag 数据集评估器
3. DummyEvaluator: 虚拟评估器（用于测试）
"""

from typing import Any, Dict, List, Optional, Iterator, Tuple
import json
import os
import re
import logging

import torch
import torch.nn as nn

import sys
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from interfaces.framework_interface import AccuracyEvaluatorInterface
from config import get_eval_logger

logger = get_eval_logger()


# ==============================================================================
# 简单评估器
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
    **kwargs
) -> AccuracyEvaluatorInterface:
    """
    创建评估器的工厂函数
    
    Args:
        dataset_name: 数据集名称 ("boolq", "hellaswag", "dummy")
        dataset_path: 数据集文件路径或 HuggingFace datasets 缓存目录
        split: 数据集划分 (train/validation/test)
        **kwargs: 传递给评估器的额外参数
    
    Returns:
        评估器实例
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == "boolq":
        return BoolQEvaluator(dataset_path, split=split)
    elif dataset_name_lower == "hellaswag":
        return HellaSwagEvaluator(dataset_path, split=split)
    elif dataset_name_lower == "dummy":
        return DummyEvaluator()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: boolq, hellaswag, dummy.")
