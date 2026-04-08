"""
在验证集上生成布局图
使用训练好的最佳模型在验证集上生成建筑布局并保存
"""
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from net.models import RoadFeatureExtractor
from dataloader import RoadDataLoader
from train import TrainingConfig

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def save_layout(model, features, road_circles, save_path, idx=0, val_energy=None):
    """保存生成的布局"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        features = features.to(device)
        road_circles = road_circles.to(device)
        layout = model(features, building_num=30)

    layout = layout.cpu().numpy()[0]  # [30, 3]
    road = features[0, 0].cpu().numpy()  # 道路二值图

    # 建筑类型和颜色
    building_types = {
        0: 'Plaza', 1: 'Plaza', 2: 'Plaza',
        3: 'Restaurant', 4: 'Restaurant', 5: 'Restaurant', 6: 'Restaurant', 7: 'Restaurant', 8: 'Restaurant',
        9: 'Shop', 10: 'Shop', 11: 'Shop', 12: 'Shop', 13: 'Shop', 14: 'Shop',
        15: 'Shop', 16: 'Shop', 17: 'Shop', 18: 'Shop', 19: 'Shop', 20: 'Shop',
        21: 'Shop', 22: 'Shop', 23: 'Shop',
        24: 'Restroom', 25: 'Restroom',
        26: 'Hotel', 27: 'Hotel', 28: 'Hotel', 29: 'Hotel',
    }

    type_colors = {
        'Plaza': '#FFD700', 'Restaurant': '#FF6B6B', 'Shop': '#4ECDC4',
        'Restroom': '#95E1D3', 'Hotel': '#A8E6CF',
    }

    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制道路
    ax.imshow(road, cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.3)

    # 绘制建筑
    for i in range(30):
        x, y, r = layout[i]
        b_type = building_types[i]
        color = type_colors[b_type]
        circle = plt.Circle((x, y), r, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    # 标题包含索引和能量信息
    title = f'Test Layout #{idx}'
    if val_energy is not None:
        title += f' (Energy: {val_energy:.2f})'
    ax.set_title(title, fontsize=14)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=name) for name, color in type_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_test_layouts(test_iter, config=None, num_samples=None, save_dir=None, best_model_path=None):
    """
    在测试集上生成布局图

    参数:
        test_iter: 测试集数据迭代器，返回 (features, road_circles)
        config: TrainingConfig对象，如果为None则使用默认配置
        num_samples: 生成的样本数量，如果为None则生成整个测试集
        save_dir: 保存目录，如果为None则自动创建
        best_model_path: 最佳模型文件路径（含时间戳的文件名）
    """
    if config is None:
        config = TrainingConfig()

    # 创建输出目录（使用日期+时间作为子目录，精确到秒）
    if save_dir is None:
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path('outputs/layouts') / f'test_{datetime_str}'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("在测试集上生成布局")
    print("=" * 60)
    print(f"\n设备: {config.device}")
    print(f"保存目录: {save_dir}")

    # 创建模型
    print("\n创建模型...")
    model = RoadFeatureExtractor(
        in_channels=config.in_channels,
        building_num=config.building_num
    ).to(config.device)

    # 加载最佳模型
    print("\n加载最佳模型...")
    if best_model_path is None:
        # 如果没有传入路径，尝试查找最新的
        model_files = list(config.checkpoint_dir.glob('best_model_*.pth'))
        if model_files:
            model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            best_model_path = model_files[0]
            print(f"找到最佳模型: {best_model_path.name}")
        else:
            print(f"[错误] 找不到模型文件: {config.checkpoint_dir}/best_model_*.pth")
            print("请先运行训练: python train_viz.py")
            return

    print(f"加载模型: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载 epoch {checkpoint['epoch']} 的模型 (验证能量: {checkpoint['energy']:.4f})")

    # 计算能量函数
    from energy.energy_function import compute_energy

    # 生成布局
    print("\n开始生成...")
    sample_count = 0

    for batch in test_iter:
        if num_samples is not None and sample_count >= num_samples:
            break

        # 解包批次数据
        features, road_circles = batch  # features: [batch, 5, 256, 256], road_circles: [batch, N, 3]

        # 取第一个样本
        if features.shape[0] > 1:
            features = features[0:1]
            road_circles = road_circles[0:1]

        # 生成布局
        model.eval()
        with torch.no_grad():
            features = features.to(config.device)
            layout = model(features, building_num=30)

        # 计算能量
        energy = compute_energy(layout, features, road_circles).item()

        # 保存图像
        save_path = save_dir / f'test_layout_{sample_count+1:03d}.png'
        save_layout(model, features.cpu(), road_circles.cpu(), save_path, idx=sample_count+1, val_energy=energy)

        sample_count += 1
        print(f"  [{sample_count:3d}] {save_path.name} (Energy: {energy:.2f})")

    print(f"\n完成! 共生成 {sample_count} 张布局图像")
    print(f"保存位置: {save_dir}")
    print("=" * 60)


# ==================== 主程序 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='在测试集上生成布局图')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='生成的样本数量（默认：全部测试集）')
    args = parser.parse_args()

    # 创建数据加载器
    from dataloader import RoadDataLoader
    from train import TrainingConfig

    config = TrainingConfig()
    dataloader = RoadDataLoader(batch_size=1)
    _, _, test_loader = dataloader.get_dataloaders()

    generate_test_layouts(test_loader, config=config, num_samples=100, 
                          best_model_path='D:\study\PINN-suburban\outputs\checkpoints\\stage4_best.pth')
