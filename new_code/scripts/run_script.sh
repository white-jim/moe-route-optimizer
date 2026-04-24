# 启用 MoE 扰动 hook
export MOE_PERTURBATION_ENABLED=1

export HF_ENDPOINT=https://hf-mirror.com

# 设置检查点路径（可选，如果有训练好的模型）
# export MOE_PERTURBATION_CHECKPOINT=/path/to/your/checkpoint.pt

# 设置 hidden size（根据你的模型配置）
export MOE_PERTURBATION_HIDDEN_SIZE=2048

cd ../moe_route_optimizer

# export CUDA_VISIBLE_DEVICES=1

# torchrun --nproc-per-node=2 main.py --model-path ../models/Qwen1.5-MoE-A2.7B
python main.py --model-path ../../models/Qwen1.5-MoE-A2.7B