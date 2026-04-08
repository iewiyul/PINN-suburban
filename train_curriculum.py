# 课程学习训练策略
# 分阶段逐步引入约束和奖励

"""
阶段划分：

阶段0：道路距离奖励（首先建立建筑与道路的位置关系）
  - road_distance: 建筑到道路的目标距离

阶段1：基础几何约束（不依赖道路信息）
  - boundary: 边界约束
  - overlap: 建筑之间重叠
  - radius: 半径约束

阶段2：道路约束（依赖道路信息，但不鼓励特定分布）
  - space_to_road: 建筑与道路重叠

阶段3：道路覆盖奖励（引导建筑均匀分布在道路周围）
  - road_coverage: 建筑均匀分布在道路周围
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from net.models import RoadFeatureExtractor
from dataloader import RoadDataLoader
from energy.energy_function import compute_energy, CONSTRAINT_WEIGHTS, REWARD_WEIGHTS
from energy.constraint.constraint_boundary import constraint_boundary
from energy.constraint.constraint_overlap import constraint_overlap
from energy.constraint.constraint_radius import constraint_radius
from energy.constraint.constraint_space_to_road import constraint_space_to_road
from energy.reward.reward_road_distance_v2 import compute_road_distance_reward_v2
from energy.reward.reward_road_coverage import reward_road_coverage


class CurriculumLearning:
    """课程学习训练器"""

    def __init__(self, save_dir='outputs/checkpoints'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 阶段配置
        self.stages = [
            {
                'name': 'stage0_road_coverage',
                'description': '道路覆盖奖励（单独训练）',
                'epochs': 200,
                'constraints': [],
                'rewards': ['road_coverage'],
                'weights': {
                    'road_coverage': 10.0,
                }
            },
            {
                'name': 'stage1_road_distance',
                'description': '添加道路距离奖励',
                'epochs': 200,
                'constraints': [],
                'rewards': ['road_coverage', 'road_distance'],
                'weights': {
                    'road_coverage': 10.0,
                    'road_distance': 30.0,
                }
            },
            {
                'name': 'stage2_basic',
                'description': '基础几何约束',
                'epochs': 200,
                'constraints': ['boundary', 'overlap', 'radius'],
                'rewards': ['road_coverage', 'road_distance'],
                'weights': {
                    'boundary': 10.0,
                    'overlap': 10.0,
                    'radius': 10.0,
                    'road_coverage': 10.0,
                    'road_distance': 30.0,
                }
            },
            {
                'name': 'stage3_road_constraint',
                'description': '添加道路约束',
                'epochs': 200,
                'constraints': ['boundary', 'overlap', 'radius', 'space_to_road'],
                'rewards': ['road_coverage', 'road_distance'],
                'weights': {
                    'boundary': 8.0,
                    'overlap': 10.0,
                    'radius': 5.0,
                    'space_to_road': 10.0,
                    'road_coverage': 35.0,
                    'road_distance': 30.0,
                }
            },
        ]

    def compute_stage_energy(self, layout, road_features, road_circles, stage_config):
        """计算当前阶段的能量"""
        device = layout.device
        batch_size = layout.shape[0]
        total_energy = torch.zeros(batch_size, device=device)

        # 计算约束
        for name in stage_config['constraints']:
            if name == 'boundary':
                energy = constraint_boundary(layout)
            elif name == 'overlap':
                energy = constraint_overlap(layout)
            elif name == 'radius':
                energy = constraint_radius(layout)
            elif name == 'space_to_road':
                energy = constraint_space_to_road(layout, road_circles)
            else:
                continue

            weight = stage_config['weights'][name]
            total_energy += weight * energy

        # 计算奖励
        for name in stage_config['rewards']:
            if name == 'road_distance':
                energy = compute_road_distance_reward_v2(layout, road_features)
            elif name == 'road_coverage':
                energy = reward_road_coverage(layout, road_circles)
            else:
                continue

            weight = stage_config['weights'][name]
            total_energy += weight * energy

        return total_energy

    def train_stage(self, stage_idx, load_from=None, learning_rate=1e-3):
        """训练单个阶段"""
        stage_config = self.stages[stage_idx]
        print(f"\n{'='*60}")
        print(f"阶段 {stage_idx + 1}: {stage_config['description']}")
        print(f"{'='*60}")
        print(f"约束: {stage_config['constraints']}")
        print(f"奖励: {stage_config['rewards']}")
        print(f"权重: {stage_config['weights']}")

        # 创建模型
        model = RoadFeatureExtractor(in_channels=5, building_num=30)

        # 加载预训练权重
        if load_from is not None:
            checkpoint = torch.load(load_from, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载预训练权重: {load_from}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 数据加载器
        dataloader = RoadDataLoader(batch_size=16)
        train_loader, val_loader, _ = dataloader.get_dataloaders()

        # 训练循环
        best_val_energy = float('inf')
        patience_counter = 0
        patience = 50
        batch_size = 16  # 定义batch_size

        for epoch in range(stage_config['epochs']):
            # 训练
            model.train()
            train_energy_sum = 0
            train_batches = 0

            for batch in train_loader:
                features, road_circles = batch
                features = features.to(device)
                road_circles = road_circles.to(device)

                optimizer.zero_grad()
                layout = model(features, building_num=30)

                energy = self.compute_stage_energy(layout, features, road_circles, stage_config)
                energy.mean().backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_energy_sum += energy.sum().item()
                train_batches += batch_size

            train_energy = train_energy_sum / train_batches

            # 验证
            model.eval()
            val_energy_sum = 0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    features, road_circles = batch
                    features = features.to(device)
                    road_circles = road_circles.to(device)

                    layout = model(features, building_num=30)
                    energy = self.compute_stage_energy(layout, features, road_circles, stage_config)

                    val_energy_sum += energy.sum().item()
                    val_batches += batch_size

            val_energy = val_energy_sum / val_batches

            # 打印进度
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{stage_config['epochs']}: "
                      f"Train={train_energy:.4f}, Val={val_energy:.4f}")

            # 早停
            if val_energy < best_val_energy:
                best_val_energy = val_energy
                patience_counter = 0

                # 保存最佳模型
                save_path = self.save_dir / f"stage{stage_idx+1}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'energy': val_energy,
                    'stage': stage_config['name'],
                    'stage_idx': stage_idx,
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发于epoch {epoch+1}")
                    break

        print(f"阶段 {stage_idx + 1} 完成！最佳验证能量: {best_val_energy:.4f}")

        return self.save_dir / f"stage{stage_idx+1}_best.pth"

    def train_all_stages(self):
        """按顺序训练所有阶段"""
        print("="*60)
        print("课程学习训练开始")
        print("="*60)

        checkpoint_paths = []

        # 阶段0：道路覆盖奖励（单独训练，从头开始）
        path0 = self.train_stage(0, learning_rate=1e-3)
        checkpoint_paths.append(path0)

        # 阶段1：添加道路距离奖励（加载阶段0的权重）
        path1 = self.train_stage(1, load_from=path0, learning_rate=1e-3)
        checkpoint_paths.append(path1)

        # 阶段2：基础几何约束（加载阶段1的权重）
        path2 = self.train_stage(2, load_from=path1, learning_rate=5e-4)
        checkpoint_paths.append(path2)

        # 阶段3：道路约束（加载阶段2的权重）
        path3 = self.train_stage(3, load_from=path2, learning_rate=2e-4)
        checkpoint_paths.append(path3)

        print("\n" + "="*60)
        print("所有阶段训练完成！")
        print(f"最终模型: {checkpoint_paths[-1]}")
        print("="*60)

        return checkpoint_paths[-1]


if __name__ == "__main__":
    trainer = CurriculumLearning()
    final_model = trainer.train_all_stages()
