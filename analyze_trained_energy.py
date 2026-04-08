"""
使用训练好的模型分析能量组件
"""
import torch
import numpy as np
import sys
from pathlib import Path

# 临时切换到ResNet版本
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RoadFeatureExtractor_ResNet(nn.Module):
    def __init__(self, in_channels: int = 5, building_num: int = 30):
        super(RoadFeatureExtractor_ResNet, self).__init__()
        self.building_num = building_num
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, building_num * 3)
        )

    def forward(self, x: torch.Tensor, building_num: int = None) -> torch.Tensor:
        if building_num is None:
            building_num = self.building_num
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(batch_size, 256)
        raw_layout = self.fc(x)
        raw_layout = raw_layout[:, :building_num * 3]
        raw_layout = raw_layout.view(batch_size, building_num, 3)
        xy = torch.sigmoid(raw_layout[:, :, :2])
        r = 0.025 + torch.sigmoid(raw_layout[:, :, 2:]) * 0.095
        return torch.cat([xy, r], dim=2)


# 导入其他模块
from dataloader import RoadDataLoader
from energy.constraint.constraint_boundary import constraint_boundary
from energy.constraint.constraint_overlap import constraint_overlap
from energy.constraint.constraint_space_to_road import constraint_space_to_road
from energy.constraint.constraint_radius import constraint_radius
from energy.reward.reward_road_distance_relationship import compute_road_distance_reward
from energy.reward.reward_road_coverage import reward_road_coverage

print('='*70)
print('使用训练好的模型分析能量组件')
print('='*70)

# 加载checkpoint
checkpoint_path = 'outputs/checkpoints/best_model_e374_20260316_233611.pth'
print(f'加载checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f'  训练轮数: {checkpoint["epoch"]}')
print(f'  最佳能量: {checkpoint["energy"]:.4f}')
print(f'  权重配置: {checkpoint["config"]}')

# 创建模型并加载权重
model = RoadFeatureExtractor_ResNet(in_channels=5, building_num=30)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载数据
dataloader = RoadDataLoader(batch_size=16)
_, val_loader, _ = dataloader.get_dataloaders()

# 收集能量值
all_energies = {
    'boundary': [],
    'overlap': [],
    'space_to_road': [],
    'radius': [],
    'road_distance': [],
    'road_coverage': [],
}

num_batches = 20
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        features, road_circles = batch
        layout = model(features, building_num=30)

        all_energies['boundary'].append(constraint_boundary(layout))
        all_energies['overlap'].append(constraint_overlap(layout))
        all_energies['space_to_road'].append(constraint_space_to_road(layout, road_circles))
        all_energies['radius'].append(constraint_radius(layout))
        all_energies['road_distance'].append(compute_road_distance_reward(layout, road_circles))
        all_energies['road_coverage'].append(reward_road_coverage(layout, features))

for key in all_energies:
    all_energies[key] = torch.cat(all_energies[key])

print()
print('='*70)
print('各组件能量统计（训练后模型，未加权）')
print('='*70)
print('组件                最小值        最大值        平均值        中位数      标准差')
print('-'*70)

for key in all_energies:
    values = all_energies[key]
    std_val = values.std()
    print(f'{key:<20} {values.min():<12.4f} {values.max():<12.4f} {values.mean():<12.4f} {values.median():<12.4f} {std_val:<12.4f}')

# 训练时使用的权重
train_config = checkpoint['config']
print()
print('='*70)
print('训练时使用的权重配置')
print('='*70)
print(f'约束权重: boundary={train_config.get("boundary", "N/A")}, overlap={train_config.get("overlap", "N/A")}, space_to_road={train_config.get("space_to_road", "N/A")}, radius={train_config.get("radius", "N/A")}')
print(f'奖励权重: road_distance={train_config.get("road_distance", "N/A")}, road_coverage={train_config.get("road_coverage", "N/A")}')

print()
print('='*70)
print('建议：基于当前训练结果分析权重调整方向')
print('='*70)

# 分析瓶颈
avg_energies = {k: v.mean().item() for k, v in all_energies.items()}
print('当前各组件平均能量：')
for key, avg in sorted(avg_energies.items(), key=lambda x: x[1], reverse=True):
    print(f'  {key:<20} {avg:>8.4f}')

print()
print('优化建议：')
print('  1. 能量越高的组件，说明模型在这方面做得越差，需要重点关注')
print('  2. 需要降低的组件：增加权重')
print('  3. 已经做得很好的组件：可以降低权重，让其他组件有更多优化空间')
