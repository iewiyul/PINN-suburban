"""
综合处理模块：生成路网图像 → 提取5通道特征 → 保存
使用 RoadChannelProcessor 类进行流程编排
"""

import torch
import os
import sys
import numpy as np
from pathlib import Path

# 导入特征提取器
from channel_process import RoadChannelProcessor
import random

from data_augment import augment_road_images

# =============================================================================
# 原始方法（已注释 - 改用真实路网数据）
# =============================================================================

# def generate_road_images(num_samples: int, img_size: int = 256) -> torch.Tensor:
#     """
#     生成路网图像
#
#     参数:
#         num_samples: 生成数量
#         img_size: 图像尺寸
#
#     返回:
#         torch.Tensor: (num_samples, 3, img_size, img_size)
#     """
#     # 添加 data 目录到路径
#     data_dir = Path(__file__).parent
#     sys.path.insert(0, str(data_dir))
#     from generate_synthetic_roads import generate_synthetic_road_network
#
#     # network_types = ['tree', 'grid']
#     network_types = ['grid']
#     samples_per_type = num_samples // len(network_types)
#     all_images = []
#
#     for idx, network_type in enumerate(network_types):
#         num_this_type = samples_per_type + (1 if idx < num_samples % len(network_types) else 0)
#         for _ in range(num_this_type):
#             img = generate_synthetic_road_network(img_size, network_type)
#             img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
#             all_images.append(img_tensor)
#
#     return torch.stack(all_images, dim=0)


def generate_and_process_roads(
    num_samples: int = 30,
    img_size: int = 256,
    output_dir: str = 'data/processed_features'
) -> torch.Tensor:
    """
    综合处理：加载真实路网图像 → 数据增强 → 提取5通道特征 → 保存

    参数:
        num_samples: 加载图像数量
        img_size: 图像尺寸
        output_dir: 输出目录

    返回:
        torch.Tensor: (num_samples * 8, 5, img_size, img_size) 5通道特征
                  每张输入图像生成 8 张增强图像
    """
    print("=" * 60)
    print(f"从真实路网数据加载 {num_samples} 张图像并处理为5通道特征")
    print("=" * 60)

    # 1. 加载真实路网图像
    print("\n加载真实路网图像...")
    from load_original_roads import load_original_roads
    all_images = load_original_roads(
        data_dir=r"C:\Users\ANASON\Desktop\PINN-suburban\data\original mask",
        img_size=img_size,
        num_samples=num_samples,
        grayscale=True
    )
    print(f"加载完成！共 {len(all_images)} 张图像")

    # 1.5 数据增强（每张图像生成 8 张增强图像）
    print(f"\n进行数据增强（每张图像生成 8 张增强图像）...")
    all_images = augment_road_images(all_images)
    print(f"增强完成！共 {len(all_images)} 张图像")

    # 2. 提取特征（使用 RoadChannelProcessor）
    print("\n提取5通道特征...")
    processor = RoadChannelProcessor(img_size=img_size)
    features = processor.extract_all_channels(all_images)
    print(f"特征提取完成！形状: {features.shape}")

    # 3. 保存道路圆特征
    print(f"\n保存特征到 {output_dir}...")
    processor.save_features(features, output_dir)

    print("\n" + "=" * 60)
    print("综合处理完成!")
    print("=" * 60)

    return features


if __name__ == "__main__":
    # 测试：生成 30 张样本
    features = generate_and_process_roads(num_samples=380)
    print(f"\n最终特征形状: {features.shape}")
