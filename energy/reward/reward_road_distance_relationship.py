"""
奖励：建筑与道路的距离关系
使用道路圆方法 - 计算建筑到最近道路圆的距离，与目标距离进行比较
"""

import torch
import numpy as np

# 不同类型建筑与道路的目标距离
TARGET_DISTANCES = [0.10, 0.02, 0.02, 0.20, 0.20]
# 建筑类型索引范围
BUILDING_INDICES = [[0, 2], [3, 8], [9, 23], [24, 25], [26, 29]]


def create_target_distances(layout: torch.Tensor) -> torch.Tensor:
    """
    为每个建筑创建目标距离

    参数:
        layout: [batch, 30, 3] 布局参数

    返回:
        torch.Tensor: [30] 每个建筑的目标距离
    """
    device = layout.device
    target_dist = torch.zeros(30, device=device)
    for i, idx_range in enumerate(BUILDING_INDICES):
        target_dist[idx_range[0]:idx_range[1] + 1] = TARGET_DISTANCES[i]
    return target_dist


def compute_road_distance_reward(layout: torch.Tensor, road_circles: torch.Tensor, tolerance: float = 0.01) -> torch.Tensor:
    """
    计算建筑与道路的距离奖励（惩罚偏离目标距离的程度）

    方法：
        1. 对每个建筑，计算到所有道路圆的最小距离（建筑边缘到道路圆边缘）
        2. 比较实际距离与目标距离，使用 arctan 平滑惩罚偏离程度
        3. 添加容差参数，允许小偏差不产生惩罚

    参数:
        layout: [batch, 30, 3] 建筑参数 (x, y, r)，归一化坐标 [0, 1]
        road_circles: [batch, N, 3] 预生成的道路圆张量 (x, y, r)
        tolerance: 容差参数，允许小偏差不产生惩罚（默认 0.01）

    返回:
        torch.Tensor: [batch] 距离奖励（值越小越好）
    """
    device = layout.device
    road_circles = road_circles.to(device)
    batch_size = layout.shape[0]
    building_num = layout.shape[1]

    # 过滤掉填充的零值圆
    valid_mask = road_circles[:, :, 2] > 0  # [batch, N]

    # 如果没有有效道路圆，返回最大惩罚（假设离道路无限远）
    # if not valid_mask.any():
    #     return torch.ones(batch_size, device=device) * 100.0

    # 创建每个建筑的目标距离 [30]
    target_distances = create_target_distances(layout)  # [30]

    # 向量化计算：扩展维度进行广播
    # layout: [batch, 30, 3] -> [batch, 30, 1, 3]
    # road_circles: [batch, N, 3] -> [batch, 1, N, 3]
    layout_expanded = layout.unsqueeze(2)  # [batch, 30, 1, 3]
    road_expanded = road_circles.unsqueeze(1)  # [batch, 1, N, 3]

    # 扩展 valid_mask 以匹配广播维度
    valid_mask_expanded = valid_mask.unsqueeze(1)  # [batch, 1, N]

    # 广播后：都是 [batch, 30, N, 3]
    # 提取坐标和半径
    x1 = layout_expanded[:, :, :, 0]  # [batch, 30, N]
    y1 = layout_expanded[:, :, :, 1]  # [batch, 30, N]
    r1 = layout_expanded[:, :, :, 2]  # [batch, 30, N]

    x2 = road_expanded[:, :, :, 0]  # [batch, 30, N]
    y2 = road_expanded[:, :, :, 1]  # [batch, 30, N]
    r2 = road_expanded[:, :, :, 2]  # [batch, 30, N]

    # 计算圆心距离
    center_dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + 1e-8)  # [batch, 30, N]

    # 计算边缘距离（建筑边缘到道路圆边缘）
    edge_dist = center_dist - (r1 + r2)  # [batch, 30, N]
    # 负值表示重叠，正值表示分离

    # 应用有效掩码：对无效道路圆，使用一个大值（不是inf，保持梯度）
    # 使用乘法掩码：有效道路圆保持原值，无效道路圆设为大值
    large_value = 10.0  # 足够大，会被 min 忽略
    valid_mask_float = valid_mask_expanded.float()  # 转换为 float
    edge_dist_masked = edge_dist * valid_mask_float + large_value * (1 - valid_mask_float)

    # 找到最近的道路圆距离
    min_dist, _ = edge_dist_masked.min(dim=2)  # [batch, 30]

    # 对于重叠的建筑（min_dist < 0），使用 0 作为距离
    min_dist = torch.clamp(min_dist, min=0)  # [batch, 30]

    # 计算与目标距离的偏离程度
    # 目标距离是建筑边缘到道路边缘的距离
    # deviation = |实际距离 - 目标距离|
    target_dist_expanded = target_distances.unsqueeze(0).expand(batch_size, -1)  # [batch, 30]
    deviation = torch.abs(min_dist - target_dist_expanded)  # [batch, 30]

    # 应用容差：只有当偏差 > tolerance 时才惩罚
    deviation = torch.relu(deviation - tolerance)  # [batch, 30]

    # 使用 arctan 惩罚偏离（梯度稳定）
    # deviation=0 时 penalty=0，deviation 很大时 penalty 趋近于 1
    penalties = torch.atan(deviation * 10.0) * (2 / 3.14159)  # [batch, 30]
    # penalties = torch.pow(deviation,0.5)

    # 对所有建筑求和 -> [batch]
    total_penalty = penalties.sum(dim=1)  # [batch]

    # 归一化
    total_penalty = total_penalty / building_num

    return total_penalty


# 向后兼容的别名
constraint_space_to_road = compute_road_distance_reward


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("道路距离奖励测试（使用道路圆方法）")
    print("=" * 60)

    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from dataloader import RoadDataLoader
    from net.models import RoadFeatureExtractor

    # 测试1: 验证目标距离分配
    print("\n测试1: 目标距离分配")
    test_layout = torch.randn(1, 30, 3)
    target_dist = create_target_distances(test_layout)
    print(f"目标距离向量: {target_dist}")
    print(f"  广场(0-2): {target_dist[0:3]} (预期: 0.06)")
    print(f"  餐厅(3-8): {target_dist[3:9]} (预期: 0.04)")
    print(f"  商店(9-23): {target_dist[9:24]} (预期: 0.04)")
    print(f"  厕所(24-25): {target_dist[24:26]} (预期: 0.08)")
    print(f"  酒店(26-29): {target_dist[26:30]} (预期: 0.12)")

    # 测试2: 使用真实数据
    print("\n测试2: 使用真实路网数据")
    dataloader = RoadDataLoader(batch_size=2)
    train_loader, _, _ = dataloader.get_dataloaders()

    model = RoadFeatureExtractor(in_channels=5, building_num=30)
    model.eval()

    for batch in train_loader:
        features, road_circles = batch
        layout = model(features, building_num=30)

        print(f"  特征形状: {features.shape}")
        print(f"  道路圆形状: {road_circles.shape}")
        print(f"  布局形状: {layout.shape}")

        # 计算奖励
        reward = compute_road_distance_reward(layout, road_circles)
        print(f"  奖励形状: {reward.shape}")
        print(f"  奖励值: {reward}")

        # 测试梯度
        layout.retain_grad()
        reward.sum().backward()

        if layout.grad is not None:
            grad_norm = layout.grad.norm().item()
            print(f"  梯度范数: {grad_norm:.6f}")
            if grad_norm > 0:
                print("  [OK] 梯度流动正常!")
            else:
                print("  [FAIL] 梯度为零!")
        else:
            print("  [FAIL] 没有梯度!")

        break

    # 测试3: GPU兼容性
    print("\n测试3: GPU兼容性")
    if torch.cuda.is_available():
        reward_gpu = compute_road_distance_reward(layout.cuda(), road_circles.cuda())
        print(f"  GPU奖励形状: {reward_gpu.shape}")
        print("  [OK] GPU兼容!")
    else:
        print("  [跳过] CUDA不可用")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
