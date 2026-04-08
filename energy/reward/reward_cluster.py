"""
聚集 + 关系奖励
整合同类聚集和异类关系到一个函数中
"""
import sys
from pathlib import Path
import torch

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from energy.constraint.constraint_overlap import circle_to_circle_edge_distance

# 建筑类型索引 (building_num=30)，非常暴力
BUILDING_TYPE_MAPPING = {
    0: '广场', 1: '广场', 2: '广场',
    3: '餐厅', 4: '餐厅', 5: '餐厅', 6: '餐厅', 7: '餐厅', 8: '餐厅',
    9: '商店', 10: '商店', 11: '商店', 12: '商店', 13: '商店', 14: '商店',
    15: '商店', 16: '商店', 17: '商店', 18: '商店', 19: '商店', 20: '商店',
    21: '商店', 22: '商店', 23: '商店',
    24: '厕所', 25: '厕所',
    26: '酒店', 27: '酒店', 28: '酒店', 29: '酒店',
}

# 类型名称到索引的映射
TYPE_NAME_TO_INDEX = {
    '广场': 0,
    '餐厅': 1,
    '商店': 2,
    '厕所': 3,
    '酒店': 4,
}

# 关系矩阵：目标边缘距离 [5, 5]
RELATION_MATRIX = torch.tensor([
    #       广场    餐厅    商店   厕所    酒店
    [ 0.20,  0.10,  0.15,  0.15,  0.25],  # 广场 (半径~0.10)
    [ 0.10,  0.08,  0.15,  0.08,  0.20],  # 餐厅 (半径~0.06)
    [ 0.15,  0.15,  0.10,  0.15,  0.30],  # 商店 (半径~0.04)
    [ 0.15,  0.08,  0.15,  0.10,  0.18],  # 厕所 (半径~0.03)
    [ 0.25,  0.20,  0.30,  0.18,  0.15],  # 酒店 (半径~0.09)
])


def reward_cluster(layout: torch.Tensor) -> torch.Tensor:
    """
    聚集 + 关系奖励

    计算所有建筑对之间的距离，根据关系矩阵获取目标距离，
    计算 (实际距离 - 目标距离)²，累加所有惩罚

    参数:
        layout: [batch, 30, 3] 布局参数 (x, y, r)

    返回:
        torch.Tensor: [batch] 总惩罚
    """
    batch_size, building_num, _ = layout.shape
    device = layout.device

    # 1. 计算所有建筑对的边缘距离 [batch, 30, 30]
    edge_dist = circle_to_circle_edge_distance(layout)  # [batch, 30, 30]

    # 2. 创建类型索引数组 [30]
    type_indices = torch.tensor([
        TYPE_NAME_TO_INDEX[BUILDING_TYPE_MAPPING[i]] for i in range(building_num)
    ], device=device)

    # 3. 创建类型对矩阵 [30, 30]
    # type_pair_matrix[i, j] = (type_indices[i], type_indices[j]) 的展平索引
    type_pair_matrix = type_indices.unsqueeze(1) * 5 + type_indices.unsqueeze(0)  # [30, 30]

    # 4. 从关系矩阵获取目标距离 [30, 30]
    target_distances = RELATION_MATRIX.flatten().to(device)[type_pair_matrix]  # [30, 30]

    # 5. 只考虑上三角（避免重复和自身比较）
    mask = torch.triu(torch.ones(building_num, building_num, device=device), diagonal=1).bool()

    # 6. 提取上三角的实际距离和目标距离
    actual_dists = edge_dist[:, mask]  # [batch, num_pairs]
    target_dists = target_distances[mask]  # [num_pairs]

    # 7. 计算惩罚：偏离目标距离的程度（使用arctan保持梯度稳定）
    deviation = torch.abs(actual_dists - target_dists)  # [batch, num_pairs]
    penalties = torch.atan(deviation * 2.0) * (2/3.14159)  # 归一化到(0,1)

    # 8. 求和得到每个样本的总惩罚
    total_penalty = penalties.sum(dim=1)  # [batch]

    return total_penalty


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("聚集 + 关系奖励测试")
    print("=" * 60)

    # 测试1: 所有建筑在同一位置（极端聚集）
    print("\n测试1: 所有建筑在同一位置")
    layout1 = torch.zeros(1, 30, 3)
    layout1[0, :, 0] = 0.5
    layout1[0, :, 1] = 0.5
    layout1[0, :, 2] = 0.05

    penalty1 = reward_cluster(layout1)
    print(f"惩罚值: {penalty1.item():.4f} (所有建筑重叠，应该很大)")

    # 测试2: 随机布局
    print("\n测试2: 随机布局")
    layout2 = torch.rand(1, 30, 3)
    penalty2 = reward_cluster(layout2)
    print(f"惩罚值: {penalty2.item():.4f}")

    # 测试3: 批处理
    print("\n测试3: 批处理")
    layout3 = torch.rand(4, 30, 3)
    penalty3 = reward_cluster(layout3)
    print(f"输入形状: {layout3.shape}")
    print(f"输出形状: {penalty3.shape}")
    print(f"惩罚值: {penalty3}")

    # 测试4: GPU兼容性
    print("\n测试4: GPU兼容性")
    if torch.cuda.is_available():
        penalty_gpu = reward_cluster(layout3.cuda())
        print(f"GPU惩罚形状: {penalty_gpu.shape}")
        print("[OK] GPU兼容!")
    else:
        print("[跳过] CUDA不可用")

    # 测试5: 验证关系矩阵
    print("\n测试5: 关系矩阵验证")
    print("关系矩阵:")
    print(RELATION_MATRIX)
    print("\n类型映射:")
    for i in range(30):
        print(f"  建筑{i}: {BUILDING_TYPE_MAPPING[i]}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
