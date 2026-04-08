"""
能量组件详细分析脚本
"""
import torch
import numpy as np
from net.models import RoadFeatureExtractor
from dataloader import RoadDataLoader
from energy.energy_function import compute_energy
from energy.constraint.constraint_boundary import constraint_boundary
from energy.constraint.constraint_overlap import constraint_overlap
from energy.constraint.constraint_space_to_road import constraint_space_to_road
from energy.constraint.constraint_radius import constraint_radius
from energy.reward.reward_road_distance_relationship import compute_road_distance_reward
from energy.reward.reward_road_coverage import reward_road_coverage

print('='*60)
print('能量组件详细分析')
print('='*60)

dataloader = RoadDataLoader(batch_size=16)
_, val_loader, _ = dataloader.get_dataloaders()

model = RoadFeatureExtractor(in_channels=5, building_num=30)
model.eval()

all_energies = {
    'boundary': [],
    'overlap': [],
    'space_to_road': [],
    'radius': [],
    'road_distance': [],
    'road_coverage': [],
}

num_batches = 10
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
print('='*60)
print('各组件能量统计（未加权）')
print('='*60)
print('组件                最小值        最大值        平均值        中位数      标准差')
print('-'*70)

for key in all_energies:
    values = all_energies[key]
    std_val = values.std()
    print(f'{key:<20} {values.min():<12.4f} {values.max():<12.4f} {values.mean():<12.4f} {values.median():<12.4f} {std_val:<12.4f}')

CONSTRAINT_WEIGHTS = {'boundary': 10.0, 'overlap': 3.0, 'space_to_road': 10.0, 'radius': 5.0}
REWARD_WEIGHTS = {'road_distance': 30.0, 'road_coverage': 150.0}

print()
print('='*60)
print('加权后能量贡献分析')
print('='*60)
print('组件                平均能量      权重       加权贡献      占比')
print('-'*70)

total_weighted = 0
contributions = {}

for key, weight in {**CONSTRAINT_WEIGHTS, **REWARD_WEIGHTS}.items():
    avg_energy = all_energies[key].mean()
    weighted = avg_energy * weight
    contributions[key] = weighted
    total_weighted += weighted

for key, weighted in contributions.items():
    weight = {**CONSTRAINT_WEIGHTS, **REWARD_WEIGHTS}[key]
    pct = weighted / total_weighted * 100
    print(f'{key:<20} {all_energies[key].mean():<12.4f} {weight:<10.1f} {weighted:<12.2f} {pct:<10.1f}%')

print('-'*70)
print(f'总计                                        {total_weighted:<12.2f} 100.0%')

print()
print('='*60)
print('分析结论与建议')
print('='*60)
print('根据占比从高到低：')
sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
for i, (key, weighted) in enumerate(sorted_contrib, 1):
    pct = weighted / total_weighted * 100
    weight = {**CONSTRAINT_WEIGHTS, **REWARD_WEIGHTS}[key]
    avg_energy = all_energies[key].mean()
    print(f'{i}. {key:<20} 占比{pct:>5.1f}% | 权重{weight:>5.1f} | 平均能量{avg_energy:>6.2f}')

print()
print('='*60)
print('权重调整建议')
print('='*60)

# 分析问题
print('当前问题分析：')
for key, avg_energy in all_energies.items():
    weight = {**CONSTRAINT_WEIGHTS, **REWARD_WEIGHTS}[key]
    weighted = avg_energy * weight
    pct = weighted / total_weighted * 100
    print(f'  {key}: 平均能量{avg_energy:.2f} × 权重{weight:.1f} = {weighted:.2f} ({pct:.1f}%)')

print()
print('瓶颈识别：')
max_contrib = max(contributions.items(), key=lambda x: x[1])
print(f'  当前最大瓶颈: {max_contrib[0]} (贡献{max_contrib[1]:.2f}, {max_contrib[1]/total_weighted*100:.1f}%)')

# 分析各组件的优化空间
print()
print('优化潜力分析：')
for key in all_energies:
    values = all_energies[key]
    avg = values.mean()
    min_val = values.min()
    print(f'  {key}: 平均{avg:.4f}, 最小可达{min_val:.4f}, 优化潜力{(avg-min_val)/avg*100:.1f}%')
