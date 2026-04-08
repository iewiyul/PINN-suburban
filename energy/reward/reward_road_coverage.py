"""
奖励：道路覆盖均匀性
确保建筑均匀分布在道路周围

完全向量化、可微版本：
    - 使用道路圆信息（road_circles）
    - 只统计索引3-23的建筑（共21个，餐厅+商店）
    - 计算每个道路圆到建筑的边缘距离
    - 使用ReLU软阈值，保持可微性
"""

import torch


def reward_road_coverage(
    layout: torch.Tensor,
    road_circles: torch.Tensor,
    coverage_threshold: float = 0.1
) -> torch.Tensor:
    """
    计算道路覆盖奖励（可微版本）

    方法：
        1. 对每个道路圆，计算到所有建筑的边缘距离
        2. 如果距离 < threshold，无惩罚；否则线性惩罚超出部分
        3. 使用atan归一化到有限范围

    参数:
        layout: [batch, 30, 3] 建筑参数 (x, y, r)
        road_circles: [batch, N, 3] 道路圆参数 (x, y, r)
        coverage_threshold: 距离阈值（默认0.1）

    返回:
        torch.Tensor: [batch] 惩罚值（越小越好）
    """
    # 确保所有张量在同一设备上
    device = layout.device
    road_circles = road_circles.to(device)

    # 只提取索引3-23的建筑（餐厅+商店，共21个）
    NEAR_ROAD_BUILDINGS = slice(3, 24)
    building_centers = layout[:, NEAR_ROAD_BUILDINGS, :2]  # [batch, 21, 2]
    building_radii = layout[:, NEAR_ROAD_BUILDINGS, 2]     # [batch, 21]

    # 道路圆中心点和半径
    circle_centers = road_circles[:, :, :2]  # [batch, N, 2]
    circle_radii = road_circles[:, :, 2]     # [batch, N]

    # 计算所有建筑到所有道路圆的中心距离
    # building_centers: [batch, 21, 2] -> [batch, 1, 21, 2]
    # circle_centers: [batch, N, 2] -> [batch, N, 1, 2]
    # 广播计算: [batch, N, 21]
    center_distances = torch.norm(
        building_centers.unsqueeze(1) - circle_centers.unsqueeze(2),
        dim=3
    )  # [batch, N, 21]

    # 边缘距离 = 中心距离 - 道路圆半径 - 建筑半径
    # circle_radii: [batch, N, 1], building_radii: [batch, 1, 21]
    edge_distances = (
        center_distances -
        circle_radii.unsqueeze(2) -
        building_radii.unsqueeze(1)
    )  # [batch, N, 21]

    # 每个道路圆到最近建筑的边缘距离
    min_edge_distances = edge_distances.min(dim=2)[0]  # [batch, N]

    coverage_probs = torch.relu(min_edge_distances - coverage_threshold)  # [batch, N]

    # 未覆盖的比例（使用可微的1 - coverage_prob）
    uncovered_ratio = coverage_probs.mean(dim=1)  # [batch]

    penalties = torch.atan(uncovered_ratio) * (2 / 3.14159)

    # 使用平方函数放大惩罚
    return penalties
