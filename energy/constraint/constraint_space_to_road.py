"""
约束：建筑与道路的重叠惩罚
使用道路圆方法 - 将道路像素点转换为圆，利用圆-圆重叠检测计算惩罚
"""

import torch
import numpy as np

def constraint_space_to_road(layout: torch.Tensor, road_circles: torch.Tensor) -> torch.Tensor:
    """
    计算建筑与道路的重叠惩罚

    方法：使用预生成的道路圆张量，计算建筑与所有道路圆的重叠

    参数:
        layout: [batch, 30, 3] 建筑参数 (x, y, r)，归一化坐标 [0, 1]
        road_circles: [batch, N, 3] 预生成的道路圆张量

    返回:
        torch.Tensor: [batch] 重叠惩罚
    """
    device = layout.device

    # 将道路圆移到正确设备
    road_circles = road_circles.to(device)

    building_num = layout.shape[1]

    # 向量化计算：扩展维度进行广播
    # layout: [batch, 30, 3] -> [batch, 30, 1, 3]
    # road_circles: [batch, N, 3] -> [batch, 1, N, 3]
    layout_expanded = layout.unsqueeze(2)  # [batch, 30, 1, 3]
    road_expanded = road_circles.unsqueeze(1)  # [batch, 1, N, 3]

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

    # 计算边缘距离
    edge_dist = center_dist - (r1 + r2)  # [batch, 30, N]

    # 计算重叠惩罚（负距离表示重叠）
    overlap_violations = torch.relu(-edge_dist)  # [batch, 30, N]

    # 使用 arctan 惩罚（梯度稳定）
    penalties = torch.atan(overlap_violations * 2.0) * (2 / 3.14159)  # [batch, 30, N]

    # 对每个建筑，对所有道路圆求和 -> [batch, 30]
    building_penalties = penalties.sum(dim=2)  # [batch, 30]

    # 对所有建筑求和 -> [batch]
    total_penalty = building_penalties.sum(dim=1)  # [batch]

    # 归一化惩罚（除以建筑数量和缩放因子）
    total_penalty = total_penalty / building_num

    return total_penalty


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("道路圆约束测试（使用预生成道路圆）")
    print("=" * 60)

    # 快速测试
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from dataloader import RoadDataLoader
    from net.models import RoadFeatureExtractor

    dataloader = RoadDataLoader(batch_size=2)
    train_loader, _, _ = dataloader.get_dataloaders()

    model = RoadFeatureExtractor(in_channels=5, building_num=30)
    model.train()

    for batch in train_loader:
        features, road_circles = batch[0], batch[1]
        layout = model(features, building_num=30)
        layout.retain_grad()

        print(f'layout requires_grad: {layout.requires_grad}')
        print(f'road_circles shape: {road_circles.shape}')

        # 测试约束函数
        energy = constraint_space_to_road(layout, road_circles)

        print(f'energy: {energy}')
        print(f'energy requires_grad: {energy.requires_grad}')

        try:
            energy.sum().backward()

            # 检查 layout 的梯度
            x_grad = layout.grad[:, :, 0].abs().sum().item()
            y_grad = layout.grad[:, :, 1].abs().sum().item()
            r_grad = layout.grad[:, :, 2].abs().sum().item()

            print(f'\n坐标 x 梯度: {x_grad:.6f}')
            print(f'坐标 y 梯度: {y_grad:.6f}')
            print(f'半径 r 梯度: {r_grad:.6f}')

            if x_grad > 0.001 and y_grad > 0.001 and r_grad > 0.001:
                print('[OK] 坐标和半径都有梯度!')
            else:
                print('[FAIL] 梯度不完整')

            break
        except Exception as e:
            print(f'[FAIL] 反向传播失败: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 60)
