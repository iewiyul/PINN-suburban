"""
重叠约束 - 计算圆与圆之间的重叠惩罚
"""

import torch


def circle_to_circle_edge_distance(circles: torch.Tensor) -> torch.Tensor:
    batch_size, n, _ = circles.shape

    # 提取坐标和半径
    x = circles[:, :, 0]  # [batch, n]
    y = circles[:, :, 1]  # [batch, n]
    r = circles[:, :, 2]  # [batch, n]

    # 使用广播计算所有圆对之间的圆心距离
    # [batch, n, 1] - [batch, 1, n] → [batch, n, n]
    dx = x.unsqueeze(2) - x.unsqueeze(1)
    dy = y.unsqueeze(2) - y.unsqueeze(1)
    # 添加epsilon防止sqrt梯度爆炸
    center_dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    # 半径和
    # [batch, n, 1] + [batch, 1, n] → [batch, n, n]
    r_sum = r.unsqueeze(2) + r.unsqueeze(1)

    # 边缘距离 = 圆心距离 - 半径和
    # < 0 表示有重叠
    edge_dist = center_dist - r_sum

    return edge_dist


def constraint_overlap(layout: torch.Tensor, overlap_threshold: float = 0.0) -> torch.Tensor:
    """
    计算重叠惩罚

    参数:
        layout: [batch, 30, 3] 布局参数 (x, y, r)
        overlap_threshold: 允许的负距离阈值（负值表示允许轻微重叠）

    返回:
        torch.Tensor: [batch] 每个样本的重叠惩罚
    """
    building_num = layout.shape[1]

    # 计算所有圆对之间的边缘距离
    edge_dist = circle_to_circle_edge_distance(layout)  # [batch, 30, 30]

    # 只考虑上三角矩阵（避免重复计算和自身比较）
    # 取 i < j 的部分
    mask = torch.triu(torch.ones(building_num, building_num, device=layout.device), diagonal=1).bool()

    # 提取上三角的距离
    violations = edge_dist[:, mask]

    # 计算惩罚：只对负距离（重叠）部分施加惩罚
    # 使用arctan惩罚（梯度稳定，有上界）
    overlap_violations = torch.relu(overlap_threshold - violations)
    penalties = torch.atan(overlap_violations * 2.0) * (2/3.14159)  # 归一化到(0,1)

    # 求和得到每个样本的总惩罚
    total_penalty = penalties.sum(dim=1)  # [batch]

    return total_penalty

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("重叠约束测试")
    print("=" * 60)

    # 测试1: 无重叠（30个圆）
    print("\n测试1: 无重叠")
    layout1 = torch.zeros(1, 30, 3)
    # 在网格上放置30个圆，确保不重叠
    for i in range(30):
        row = i // 6
        col = i % 6
        layout1[0, i, 0] = 0.1 + col * 0.15  # x
        layout1[0, i, 1] = 0.1 + row * 0.15  # y
        layout1[0, i, 2] = 0.05  # r

    penalty1 = constraint_overlap(layout1)
    print(f"惩罚值: {penalty1.item():.4f} (应该接近0)")

    # 测试2: 有重叠
    print("\n测试2: 有重叠")
    layout2 = torch.zeros(1, 30, 3)
    layout2[0, 0] = torch.tensor([0.5, 0.5, 0.1])
    layout2[0, 1] = torch.tensor([0.55, 0.5, 0.1])  # 与圆0重叠
    penalty2 = constraint_overlap(layout2)
    print(f"惩罚值: {penalty2.item():.4f} (应该>0)")

    # 测试3: 批处理
    print("\n测试3: 批处理")
    batch_layout = torch.randn(4, 30, 3)
    batch_penalty = constraint_overlap(batch_layout)
    print(f"输入形状: {batch_layout.shape}")
    print(f"输出形状: {batch_penalty.shape}")
    print(f"惩罚值: {batch_penalty}")
    print("测试成功 " + "=" * 60)
