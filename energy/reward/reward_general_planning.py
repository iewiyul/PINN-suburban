import torch

def reward_coverage(layout: torch.Tensor) -> torch.Tensor:
    """
    将每张图划分成四个象限，计算每个建筑中心点所在的象限，每个象限的期望是7.5
    """
    x, y = layout[:, :, 0], layout[:, :, 1]

    # 数学方法划分象限
    x_int = (x / 0.5).long().clamp(0,1)
    y_int = (y / 0.5).long().clamp(0,1)
    idx = 2 * y_int + x_int # [batch,30]

    # 利用独热编码处理
    counts = torch.nn.functional.one_hot(idx, num_classes=4).float()
    all_counts = counts.sum(dim=1) # [batch,4]

    # 建筑的数量
    building_avg = layout.shape[1] / 4

    # 使用arctan惩罚偏离平均值的程度（梯度稳定）
    deviation = torch.abs(all_counts - building_avg)
    penalty = torch.atan(deviation * 2.0) * (2/3.14159)
    penalty = penalty.sum(dim=1)

    return penalty

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("覆盖范围奖励测试")
    print("=" * 60)

    # 测试1: 完美均匀分布（每个象限7-8个建筑）
    print("\n测试1: 完美均匀分布")
    layout1 = torch.zeros(1, 30, 3)
    # Q1: 8个 (x>0.5, y>0.5)
    layout1[0, 0:8, 0] = 0.75
    layout1[0, 0:8, 1] = 0.75
    # Q2: 7个 (x<0.5, y>0.5)
    layout1[0, 8:15, 0] = 0.25
    layout1[0, 8:15, 1] = 0.75
    # Q3: 8个 (x<0.5, y<0.5)
    layout1[0, 15:23, 0] = 0.25
    layout1[0, 15:23, 1] = 0.25
    # Q4: 7个 (x>0.5, y<0.5)
    layout1[0, 23:30, 0] = 0.75
    layout1[0, 23:30, 1] = 0.25

    penalty1 = reward_coverage(layout1)
    print(f"Q1:8, Q2:7, Q3:8, Q4:7 -> 惩罚: {penalty1.item():.4f} (应该接近0)")

    # 测试2: 极度不均匀（全在一个象限）
    print("\n测试2: 极度不均匀（全在Q1）")
    layout2 = torch.zeros(1, 30, 3)
    layout2[0, :, 0] = 0.75
    layout2[0, :, 1] = 0.75

    penalty2 = reward_coverage(layout2)
    expected = (30 - 7.5)**2 + (0 - 7.5)**2 + (0 - 7.5)**2 + (0 - 7.5)**2
    print(f"Q1:30, Q2:0, Q3:0, Q4:0 -> 惩罚: {penalty2.item():.4f}")
    print(f"预期: (30-7.5)^2 + (0-7.5)^2*3 = {expected:.4f}")

    # 测试3: 批处理
    print("\n测试3: 批处理")
    layout3 = torch.rand(4, 30, 3)
    penalty3 = reward_coverage(layout3)
    print(f"输入形状: {layout3.shape}")
    print(f"输出形状: {penalty3.shape}")
    print(f"惩罚值: {penalty3}")

    # 测试4: GPU兼容性
    print("\n测试4: GPU兼容性")
    if torch.cuda.is_available():
        penalty_gpu = reward_coverage(layout3.cuda())
        print(f"GPU惩罚形状: {penalty_gpu.shape}")
        print("[OK] GPU兼容!")
    else:
        print("[跳过] CUDA不可用")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
