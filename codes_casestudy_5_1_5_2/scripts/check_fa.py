import torch

def check_fa1(device_id):
    """检查是否支持 FlashAttention 1（计算能力 ≥7.0）"""
    c = torch.cuda.get_device_capability(device_id)
    return c[0] >= 7

def check_fa2(device_id):
    """检查是否支持 FlashAttention 2（计算能力 8.x 或 9.0）"""
    c = torch.cuda.get_device_capability(device_id)
    return (c[0] == 8 and c[1] >= 0) or (c[0] == 9 and c[1] == 0)

# 检查 GPU 0 和 GPU 1
for device_id in [0, 1]:
    if device_id >= torch.cuda.device_count():
        print(f"GPU {device_id}: 不存在")
        continue
    fa1_support = check_fa1(device_id)
    fa2_support = check_fa2(device_id)
    cc = torch.cuda.get_device_capability(device_id)
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"GPU {device_id} ({gpu_name}):")
    print(f"  计算能力：{cc[0]}.{cc[1]}")
    print(f"  支持 FlashAttention 1：{fa1_support}")
    print(f"  支持 FlashAttention 2：{fa2_support}")
    print("-" * 60)