"""
能量函数 - 整合所有约束和奖励
计算总能量 E = Σ(weight × component)
"""
import torch
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入约束函数
from energy.constraint.constraint_boundary import constraint_boundary
from energy.constraint.constraint_overlap import constraint_overlap
from energy.constraint.constraint_space_to_road import constraint_space_to_road
from energy.constraint.constraint_radius import constraint_radius

# 导入奖励函数
# from energy.reward.reward_cluster import reward_cluster  # 已禁用

# 旧版本（基于道路圆方法）- 已注释
# from energy.reward.reward_road_distance_relationship import constraint_space_to_road as reward_road_distance

# 新版本（基于距离场方法）- 当前使用
from energy.reward.reward_road_distance_v2 import compute_road_distance_reward_v2 as reward_road_distance
from energy.reward.reward_road_coverage import reward_road_coverage

# ==================== 能量权重配置 ====================

# 约束权重（违反约束的惩罚）
# CONSTRAINT_WEIGHTS = {
#     'boundary': 120.0,      # 边界约束（高权重，必须满足）
#     'overlap': 60.0,        # 建筑之间重叠（降低权重，原50.0过高）
#     'space_to_road': 80,  # 建筑与道路重叠（中高权重）
#     'radius': 20.0,         # 半径约束（中权重）
# }

# # 奖励权重（鼓励好的布局）
# REWARD_WEIGHTS = {
#     # 'cluster': 10.0,      # 聚集+关系奖励（已禁用）
#     # 'coverage': 5.0,       # 覆盖范围奖励（已禁用）
#     'road_distance': 200.0,   # 道路距离奖励（防止太远）
# }

# 约束权重（违反约束的惩罚）
CONSTRAINT_WEIGHTS = {
    'boundary': 8.0,      # 边界约束（高权重，必须满足）
    'overlap': 10.0,        # 建筑之间重叠（降低权重，允许轻微重叠，原10.0）
    'space_to_road': 15.0,  # 建筑与道路重叠（中高权重）
    'radius': 5.0,         # 半径约束（降低权重，原10.0）
}

# 奖励权重（鼓励好的布局）
REWARD_WEIGHTS = {
    'road_distance': 30.0,   # 道路距离奖励（防止太远，提高权重）
    'road_coverage': 35.0,  # 道路覆盖均匀性（大幅提高权重，重点引导沿道路分布）
}

def compute_energy(
    layout: torch.Tensor,
    road_features: torch.Tensor,
    road_circles: torch.Tensor = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    计算总能量

    参数:
        layout: [batch, 30, 3] 布局参数 (x, y, r)
        road_features: [batch, 5, 256, 256] 道路特征 (通道0=二值, 通道1=距离场)
        road_circles: [batch, N, 3] 预生成的道路圆张量（可选）
        verbose: 是否打印详细信息

    返回:
        torch.Tensor: [batch] 总能量 E
    """
    device = layout.device
    batch_size = layout.shape[0]

    # 提取道路二值图（通道0）
    road_binary = road_features[:, 0:1, :, :]  # [batch, 1, 256, 256]

    # ==================== 约束能量 ====================
    constraint_energies = {}

    # # 1. 边界约束
    e_boundary = constraint_boundary(layout)
    constraint_energies['boundary'] = e_boundary

    # # 2. 建筑之间重叠
    e_overlap = constraint_overlap(layout)
    constraint_energies['overlap'] = e_overlap

    # 3. 建筑与道路重叠
    # 使用预生成的道路圆
    e_space_to_road = constraint_space_to_road(layout, road_circles)
    constraint_energies['space_to_road'] = e_space_to_road

    # # 4. 半径约束
    e_radius = constraint_radius(layout)
    constraint_energies['radius'] = e_radius

    # ==================== 奖励能量 ====================
    reward_energies = {}

    # 1. 聚集+关系奖励
    # e_cluster = reward_cluster(layout)
    # reward_energies['cluster'] = e_cluster

    # 2. 覆盖范围奖励
    # from energy.reward.reward_general_planning import reward_coverage
    # e_coverage = reward_coverage(layout)
    # reward_energies['coverage'] = e_coverage

    # # 3. 道路距离奖励（使用距离场方法V2）
    e_road_distance = reward_road_distance(layout, road_features)
    reward_energies['road_distance'] = e_road_distance

    # # 4. 道路覆盖均匀性奖励
    e_road_coverage = reward_road_coverage(layout, road_circles)
    reward_energies['road_coverage'] = e_road_coverage

    # ==================== 计算总能量 ====================
    total_energy = torch.zeros(batch_size, device=device)

    # 累加约束能量
    for name, energy in constraint_energies.items():
        weight = CONSTRAINT_WEIGHTS[name]
        total_energy += weight * energy

    # 累加奖励能量
    for name, energy in reward_energies.items():
        weight = REWARD_WEIGHTS[name]
        total_energy += weight * energy

    # ==================== 打印详细信息 ====================
    if verbose:
        print("\n" + "=" * 60)
        print("能量计算详情")
        print("=" * 60)

        print("\n【约束能量】")
        for name, energy in constraint_energies.items():
            weight = CONSTRAINT_WEIGHTS[name]
            weighted = weight * energy
            if batch_size == 1:
                print(f"  {name:20s}: {energy.item():10.4f} × {weight:5.1f} = {weighted.item():10.4f}")
            else:
                print(f"  {name:20s}: {energy.tolist()} × {weight:5.1f} = {weighted.tolist()}")

        print("\n【奖励能量】")
        for name, energy in reward_energies.items():
            weight = REWARD_WEIGHTS[name]
            weighted = weight * energy
            if batch_size == 1:
                print(f"  {name:20s}: {energy.item():10.4f} × {weight:5.1f} = {weighted.item():10.4f}")
            else:
                print(f"  {name:20s}: {energy.tolist()} × {weight:5.1f} = {weighted.tolist()}")

        print("\n" + "=" * 60)
        if batch_size == 1:
            print(f"总能量: {total_energy.item():.4f}")
        else:
            print(f"总能量: {total_energy.tolist()}")
        print("=" * 60)

    return total_energy

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("能量函数测试")
    print("=" * 60)

    # 创建测试数据
    batch_size = 2
    layout = torch.rand(batch_size, 30, 3)
    road_distance = torch.rand(batch_size, 1, 256, 256) * 0.3

    print(f"\n输入形状:")
    print(f"  layout: {layout.shape}")
    print(f"  road_distance: {road_distance.shape}")

    # 计算能量
    energy = compute_energy(layout, road_distance, verbose=False)

    print(f"\n最终能量形状: {energy.shape}")
    print(f"能量值: {energy}")

    # 测试GPU兼容性
    print("\n" + "=" * 60)
    print("GPU兼容性测试")
    print("=" * 60)
    if torch.cuda.is_available():
        energy_gpu = compute_energy(layout.cuda(), road_distance.cuda())
        print(f"GPU能量形状: {energy_gpu.shape}")
        print(f"GPU能量值: {energy_gpu}")
        print("[OK] GPU兼容!")
    else:
        print("[跳过] CUDA不可用")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
