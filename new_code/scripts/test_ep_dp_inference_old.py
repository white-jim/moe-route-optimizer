"""
SGLang 专家并行(EP) + 数据并行(DP) 测试脚本

测试环境: 单机双卡
测试配置:
- EP=2, DP=1: 专家并行模式（每个GPU持有一半专家）
- 使用 moe_a2a_backend='none'（不使用deepep）

使用方式:
    python test_ep_dp_inference.py
"""

import sglang as sgl
from sglang.srt.server_args import ServerArgs
import dataclasses

import sys
sys.path.insert(0, '/sgl-workspace/sglang/DeepEP')

def main():
    # model_path = "../../models/Qwen1.5-MoE-A2.7B"
    model_path = "../../models/Qwen1.5-MoE-A2.7B"
    
    ep_size = 2
    dp_size = 1
    tp_size = 2
    enable_deepep_moe = True
    enable_ep_moe = True
    
    server_args = ServerArgs(
        model_path=model_path,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        # enable_deepep_moe=enable_deepep_moe,
        # deepep_mode="normal",
        disable_cuda_graph=True,
        enable_ep_moe=enable_ep_moe,
        trust_remote_code=True,
        random_seed=42,
    )
    
    engine = sgl.Engine(**dataclasses.asdict(server_args))
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is machine learning?",
        "Explain the concept of expert parallelism in MoE models.",
        "Write a short poem about AI.",
        "What are the benefits of distributed computing?",
        "How does a transformer model work?",
        "What is the difference between CPU and GPU?",
        "Explain the concept of gradient descent.",
        "What is the meaning of life?",
    ]
    
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
    }
    
    print(f"\nRunning batch inference with {len(prompts)} prompts...")
    
    outputs = engine.generate(prompts, sampling_params)
    
    print("\n" + "=" * 60)
    print("Inference Results:")
    print("=" * 60)
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        generated_text = output["text"]
        print(f"\n[{i+1}] Prompt: {prompt}")
        print(f"    Generated: {generated_text}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    engine.shutdown()


if __name__ == "__main__":
    main()
