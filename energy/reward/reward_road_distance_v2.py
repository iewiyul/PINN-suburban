"""
奖励：建筑与道路的距离关系（V2版本）
基于距离场的方法，支持不同建筑类型有不同目标距离
"""

import torch
import torch.nn.functional as F


# =============================================================================
# 不同建筑类型的目标距离配置
# =============================================================================

# 目标距离：建筑边缘到道路边缘的期望距离
# 距离越小，建筑越靠近道路
TARGET_DISTANCES = [0.20, 0.05, 0.05, 0.30, 0.30]  # 归一化距离 [0, 1]

# 建筑类型索引范围
# 0-2: 广场（远离道路，保持开阔）
# 3-8: 餐厅（靠近道路，方便吸引人流）
# 9-23: 商店（靠近道路，方便展示和吸引人流）
# 24-25: 厕所（远离道路，保持私密性）
# 26-29: 酒店（远离道路，保持安静）
BUILDING_INDICES = [[0, 2], [3, 8], [9, 23], [24, 25], [26, 29]]

# 建筑类型名称（用于调试）
BUILDING_NAMES = ['广场', '餐厅', '商店', '厕所', '酒店']


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


def compute_road_distance_reward_v2(
    layout: torch.Tensor,
    road_features: torch.Tensor,
    tolerance: float = 0.01
) -> torch.Tensor:
    """
    计算建筑与道路的距离奖励（基于距离场的方法）

    新方法优势：
        1. 使用距离场（通道1）直接采样，更精确
        2. 支持不同建筑类型有不同目标距离
        3. 连续可微，梯度信号更强

    参数:
        layout: [batch, 30, 3] 建筑参数 (x, y, r)，归一化坐标 [0, 1]
        road_features: [batch, 5, 256, 256] 道路特征
                       通道1: 距离场（到最近道路的归一化距离）
        tolerance: 容差参数，允许小偏差不产生惩罚（默认 0.01）

    返回:
        torch.Tensor: [batch] 距离奖励（值越小越好）
    """
    device = layout.device
    batch_size = layout.shape[0]
    building_num = layout.shape[1]
    H, W = 256, 256

    # ==================== 步骤1：提取距离场 ====================
    # 距离场（通道1）：每个像素到最近道路的距离，归一化到 [0, 1]
    distance_field = road_features[:, 1, :, :]  # [batch, 256, 256]

    # ==================== 步骤2：对每个建筑采样其位置的距离值 ====================
    # 建筑坐标 [batch, 30, 2]
    building_positions = layout[:, :, :2]  # [batch, 30, 2]

    # 将归一化坐标 [0, 1] 映射到 grid_sample 需要的 [-1, 1] 范围
    grid = building_positions * 2 - 1  # [batch, 30, 2]
    grid = grid.unsqueeze(1)  # [batch, 1, 30, 2]

    # grid_sample 需要 [batch, C, H, W] 格式的输入
    distance_field_expanded = distance_field.unsqueeze(1)  # [batch, 1, 256, 256]

    # 使用双线性插值采样每个建筑位置的距离值
    # 这是建筑中心到道路中心的距离
    center_to_road_distance = F.grid_sample(
        distance_field_expanded,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1).squeeze(1)  # [batch, 30]

    # ==================== 步骤3：计算建筑边缘到道路边缘的距离 ====================
    # 建筑半径
    building_radii = layout[:, :, 2]  # [batch, 30]

    # 说明：
    # - distance_field 存储的是"背景像素到最近道路像素的距离"
    # - 道路内部的距离值为 0
    # - 所以采样得到的 center_to_road_distance 实际上是：
    #   - 如果建筑中心在道路外：到道路边缘的距离
    #   - 如果建筑中心在道路内：0（因为道路内部距离场为0）

    # 计算建筑边缘到道路边缘的距离
    # 理想情况：edge_distance = center_to_road_distance - building_radius
    # 但当建筑中心在道路内时，center_to_road_distance = 0，这会导致计算错误

    # 更准确的方法：
    # 1. 如果建筑中心在道路外（distance > 0）：
    #    edge_distance = center_to_road_distance - building_radius
    #    如果 edge_distance < 0，说明建筑与道路重叠
    # 2. 如果建筑中心在道路内（distance = 0）：
    #    edge_distance = -building_radius（表示建筑深入道路内部的距离）

    edge_distance = center_to_road_distance - building_radii  # [batch, 30]

    # 对于负值（表示重叠或建筑在道路内），我们保留负值
    # 这样可以区分"轻微重叠"和"严重重叠"

    # ==================== 步骤4：计算与目标距离的偏离程度 ====================
    # 创建每个建筑的目标距离 [30]
    target_distances = create_target_distances(layout)  # [30]
    target_dist_expanded = target_distances.unsqueeze(0).expand(batch_size, -1)  # [batch, 30]

    # 计算偏离程度
    # 修正：只有当建筑边缘距离超过目标距离时才惩罚
    deviation = torch.abs(edge_distance - target_dist_expanded)  # [batch, 30]

    # 应用容差：只有当偏差 > tolerance 时才惩罚
    deviation = torch.relu(deviation - tolerance)  # [batch, 30]

    # ==================== 步骤5：使用平滑惩罚函数 ====================
    # 使用 arctan 惩罚偏离（梯度稳定）
    # deviation=0 时 penalty=0，deviation 很大时 penalty 趋近于 1
    penalties = torch.atan(deviation * 20.0) * (2 / 3.14159)  # [batch, 30]

    # ==================== 步骤6：对所有建筑求平均 ====================
    total_penalty = penalties.mean(dim=1)  # [batch]

    return total_penalty


# 向后兼容的别名（如果需要替换原来的函数）
# constraint_space_to_road = compute_road_distance_reward_v2


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("道路距离奖励测试（V2版本：基于距离场）")
    print("=" * 60)

    import sys
    from pathlib import Path
    import numpy as np
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from dataloader import RoadDataLoader
    from net.models import RoadFeatureExtractor

    # 测试1: 验证目标距离分配
    print("\n测试1: 目标距离分配")
    test_layout = torch.randn(1, 30, 3)
    target_dist = create_target_distances(test_layout)
    print(f"目标距离向量（前30个值）:")
    for i, (name, idx_range) in enumerate(zip(BUILDING_NAMES, BUILDING_INDICES)):
        dist = target_dist[idx_range[0]:idx_range[1] + 1]
        print(f"  {name}({idx_range[0]}-{idx_range[1]}): {dist[0].item():.2f}")

    # 测试2: 创建测试场景
    print("\n测试2: 不同距离场景测试")

    # 创建道路特征和距离场
    batch_size = 1
    road_features = torch.zeros(batch_size, 5, 256, 256)

    # 创建横向道路
    road_features[0, 0, 125:131, :] = 1.0  # 横向道路（6像素宽）

    # 计算距离场
    for i in range(256):
        for j in range(256):
            # 到横向道路的垂直距离
            if 125 <= i < 131:
                dist = 0
            else:
                dist = min(abs(i - 125), abs(i - 131))
            # 归一化到 [0, 1]
            road_features[0, 1, i, j] = dist / 256.0

    # 场景1：所有建筑都放在理想距离上
    print("\n场景1: 建筑在理想距离上")
    layout1 = torch.zeros(batch_size, 30, 3)
    for i in range(30):
        # 沿着道路，y坐标在道路附近
        # 根据建筑类型调整距离
        target_d = target_dist[i].item()
        layout1[0, i] = torch.tensor([0.1 + i * 0.03, 0.5 + target_d, 0.04])

    reward1 = compute_road_distance_reward_v2(layout1, road_features)
    print(f"  惩罚值: {reward1.item():.4f}")
    print(f"  加权后 (×30): {reward1.item() * 30:.2f}")

    # 场景2：建筑远离道路
    print("\n场景2: 建筑远离道路")
    layout2 = torch.zeros(batch_size, 30, 3)
    for i in range(30):
        layout2[0, i] = torch.tensor([0.1 + i * 0.03, 0.8, 0.04])  # y=0.8，远离道路

    reward2 = compute_road_distance_reward_v2(layout2, road_features)
    print(f"  惩罚值: {reward2.item():.4f}")
    print(f"  加权后 (×30): {reward2.item() * 30:.2f}")

    # 场景3：建筑在道路上（重叠）
    print("\n场景3: 建筑与道路重叠")
    layout3 = torch.zeros(batch_size, 30, 3)
    for i in range(30):
        layout3[0, i] = torch.tensor([0.1 + i * 0.03, 0.5, 0.04])  # 直接在道路上

    reward3 = compute_road_distance_reward_v2(layout3, road_features)
    print(f"  惩罚值: {reward3.item():.4f}")
    print(f"  加权后 (×30): {reward3.item() * 30:.2f}")

    print("\n" + "=" * 60)
    print("总结（期望：理想距离 < 道路重叠 < 远离道路）:")
    print(f"  理想距离:   {reward1.item() * 30:.2f}")
    print(f"  道路重叠:   {reward3.item() * 30:.2f}")
    print(f"  远离道路:   {reward2.item() * 30:.2f}")
    print("=" * 60)

    # 测试3: 使用真实数据
    print("\n测试3: 使用真实路网数据")
    dataloader = RoadDataLoader(batch_size=2)
    train_loader, _, _ = dataloader.get_dataloaders()

    model = RoadFeatureExtractor(in_channels=5, building_num=30)
    model.eval()

    for batch in train_loader:
        features, road_circles = batch
        layout = model(features, building_num=30)

        print(f"  特征形状: {features.shape}")
        print(f"  布局形状: {layout.shape}")

        # 计算奖励
        reward = compute_road_distance_reward_v2(layout, features)
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

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
