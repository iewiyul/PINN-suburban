"""
建筑类型半径约束
确保每种建筑类型的半径在合理范围内
"""

import torch


# 建筑类型索引范围 (building_num=30)
BUILDING_TYPES = {
    '广场': (0, 2, [0.09, 0.12]),      # 0-2: 直径 25-42米
    '餐厅': (3, 8, [0.05, 0.08]),      # 3-8: 直径 14-22米
    '商店': (9, 23, [0.03, 0.05]),     # 9-23: 直径 8-14米
    '厕所': (24, 25, [0.025, 0.04]),   # 24-25: 直径 7-11米
    '酒店': (26, 29, [0.07, 0.11]),    # 26-29: 直径 19-38米
}


def constraint_radius(layout: torch.Tensor, tolerance: float = 0.005) -> torch.Tensor:
    """
    约束每种建筑类型的半径在合理范围内

    参数:
        layout: [batch, building_num, 3] 布局参数 (x, y, r)
        tolerance: 容差参数，允许小偏差不产生惩罚（默认 0.005，即 0.5%）

    返回:
        torch.Tensor: [batch] 每个样本的半径约束惩罚
    """

    r = layout[:, :, 2]  # [batch, building_num]
    building_num = layout.shape[1]

    # 对每个建筑计算半径约束违反
    penalties = []

    for start_idx, end_idx, (r_min, r_max) in BUILDING_TYPES.values():
        # 提取该类型的半径
        type_radii = r[:, start_idx:end_idx+1]  # [batch, count]

        # 计算违反约束的部分（减去容差）
        # 只有当偏差 > tolerance 时才惩罚
        below_min = torch.relu(r_min - type_radii - tolerance)
        above_max = torch.relu(type_radii - r_max - tolerance)

        # 使用 arctan 惩罚（梯度稳定，有上界）
        below_min = torch.atan(below_min * 10.0) * (2 / 3.14159)
        above_max = torch.atan(above_max * 10.0) * (2 / 3.14159)
        # below_min = torch.pow(below_min , 2.0)
        # above_max = torch.pow(above_max , 2.0)

        # 求和
        type_penalty = (below_min + above_max).sum(dim=1)  # [batch]
        penalties.append(type_penalty)

    # 总惩罚 = 所有类型惩罚之和
    total_penalty = torch.stack(penalties, dim=0).sum(dim=0) # [batch]

    return total_penalty


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("半径约束测试")
    print("=" * 60)

    # 测试1: 所有半径都在范围内
    print("\n测试1: 所有半径合规")
    layout1 = torch.zeros(1, 30, 3)
    layout1[0, 0:3, 2] = 0.10    # 广场: [0.09, 0.12] ✓
    layout1[0, 3:9, 2] = 0.06    # 餐厅: [0.05, 0.08] ✓
    layout1[0, 9:24, 2] = 0.04   # 商店: [0.03, 0.05] ✓
    layout1[0, 24:26, 2] = 0.03  # 厕所: [0.025, 0.04] ✓
    layout1[0, 26:30, 2] = 0.09  # 酒店: [0.07, 0.11] ✓

    penalty1 = constraint_radius(layout1)
    print(f"惩罚值: {penalty1.item():.6f} (应该接近0)")

    # 测试2: 有违规
    print("\n测试2: 有违规")
    layout2 = torch.zeros(1, 30, 3)
    layout2[0, 0, 2] = 0.15   # 广场太大: 0.15 > 0.12
    layout2[0, 3, 2] = 0.03   # 餐厅太小: 0.03 < 0.05

    penalty2 = constraint_radius(layout2)
    print(f"惩罚值: {penalty2.item():.6f} (应该>0)")
    expected = (0.15 - 0.12)**2 + (0.05 - 0.03)**2
    print(f"  预期广场: (0.15-0.12)^2 = {0.03**2:.6f}")
    print(f"  预期餐厅: (0.05-0.03)^2 = {0.02**2:.6f}")

    # 测试3: 批处理
    print("\n测试3: 批处理 + GPU兼容性")
    layout3 = torch.randn(4, 30, 3) * 0.1  # 随机半径
    penalty3 = constraint_radius(layout3)
    print(f"输入形状: {layout3.shape}")
    print(f"输出形状: {penalty3.shape}")
    print(f"惩罚值: {penalty3}")

    if torch.cuda.is_available():
        penalty3_gpu = constraint_radius(layout3.cuda())
        print(f"GPU惩罚形状: {penalty3_gpu.shape}")
        print("[OK] GPU兼容!")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
