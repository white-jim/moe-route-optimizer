"""
evaluation 模块

基于 lm-eval (lm-evaluation-harness) 框架的模型评估工具

主要组件:
- LMEvalEvaluator: 基于 lm-eval 的评估器，支持 boolq, hellaswag, mmlu 等任务
- DirectLMAdapter: 直接适配 lm-eval LM 接口的适配器
- FrameworkModelAdapter: 将推理框架适配为 HuggingFace 风格模型

使用示例:
    from evaluation import LMEvalEvaluator, create_evaluator
    
    # 方式1: 使用 LMEvalEvaluator
    evaluator = LMEvalEvaluator(
        tasks=["boolq", "hellaswag"],
        batch_size=32,
        limit=100
    )
    results = evaluator.evaluate_model(model, tokenizer)
    accuracy = evaluator.get_accuracy()
    
    # 方式2: 使用工厂函数
    evaluator = create_evaluator("boolq", use_lm_eval=True)
"""

from interfaces.evaluator_impl import (
    # lm-eval 相关
    LMEvalEvaluator,
    DirectLMAdapter,
    FrameworkModelAdapter,
    create_lm_eval_evaluator,
    LM_EVAL_AVAILABLE,
    
    # 简单评估器（备选）
    BaseEvaluator,
    BoolQEvaluator,
    HellaSwagEvaluator,
    DummyEvaluator,
    
    # 工厂函数
    create_evaluator,
)

__all__ = [
    # lm-eval 相关
    'LMEvalEvaluator',
    'DirectLMAdapter',
    'FrameworkModelAdapter',
    'create_lm_eval_evaluator',
    'LM_EVAL_AVAILABLE',
    
    # 简单评估器
    'BaseEvaluator',
    'BoolQEvaluator',
    'HellaSwagEvaluator',
    'DummyEvaluator',
    
    # 工厂函数
    'create_evaluator',
]
