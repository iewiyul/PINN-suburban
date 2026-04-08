"""
训练可视化脚本 - 绘制训练曲线和生成布局
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from net.models import RoadFeatureExtractor
from energy.energy_function import compute_energy


def plot_training_history(history_path='outputs/training_history.npy', save_path='outputs/training_curves.png'):
    """
    绘制训练曲线

    参数:
        history_path: 训练历史文件路径
        save_path: 保存图片路径
    """
    # 加载训练历史
    history = np.load(history_path, allow_pickle=True).item()

    train_energies = history['train_energies']
    val_energies = history['val_energies']
    epochs = range(1, len(train_energies) + 1)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子图1: 能量曲线
    axes[0].plot(epochs, train_energies, label='Train Energy', linewidth=2)
    axes[0].plot(epochs, val_energies, label='Val Energy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Training and Validation Energy', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 子图2: 学习率曲线
    if 'learning_rates' in history:
        learning_rates = history['learning_rates']
        axes[1].plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")

    # 打印统计信息
    print(f"\n训练统计:")
    print(f"  最终训练能量: {train_energies[-1]:.4f}")
    print(f"  最终验证能量: {val_energies[-1]:.4f}")
    print(f"  最佳验证能量: {min(val_energies):.4f}")
    print(f"  总训练轮数: {len(train_energies)}")


def visualize_layout(model, features, save_path='outputs/layout_visualization.png'):
    """
    可视化生成的布局

    参数:
        model: 训练好的模型
        features: 输入特征 [1, 5, 256, 256]
        save_path: 保存图片路径
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        features = features.to(device)
        layout = model(features, building_num=30)

    # 转为numpy
    layout = layout.cpu().numpy()[0]  # [30, 3]
    road = features[0, 0].cpu().numpy()  # 道路二值图 [256, 256]

    # 建筑类型映射
    building_types = {
        0: '广场', 1: '广场', 2: '广场',
        3: '餐厅', 4: '餐厅', 5: '餐厅', 6: '餐厅', 7: '餐厅', 8: '餐厅',
        9: '商店', 10: '商店', 11: '商店', 12: '商店', 13: '商店', 14: '商店',
        15: '商店', 16: '商店', 17: '商店', 18: '商店', 19: '商店', 20: '商店',
        21: '商店', 22: '商店', 23: '商店',
        24: '厕所', 25: '厕所',
        26: '酒店', 27: '酒店', 28: '酒店', 29: '酒店',
    }

    # 建筑类型颜色
    type_colors = {
        '广场': '#FFD700',      # 金色
        '餐厅': '#FF6B6B',      # 红色
        '商店': '#4ECDC4',      # 青色
        '厕所': '#95E1D3',      # 浅青色
        '酒店': '#A8E6CF',      # 浅绿色
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
    ax.set_title('Generated Layout', fontsize=14)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=name) for name, color in type_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"布局可视化已保存: {save_path}")


def visualize_multiple_layouts(model, data_loader, num_samples=4, save_path='outputs/layout_samples.png'):
    """
    可视化多个样本布局

    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        num_samples: 样本数量
        save_path: 保存图片路径
    """
    model.eval()
    device = next(model.parameters()).device

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    count = 0
    for batch in data_loader:
        if count >= num_samples:
            break

        features = batch[0].to(device)

        with torch.no_grad():
            layout = model(features, building_num=30)

        # 获取第一个样本
        layout = layout[0].cpu().numpy()  # [30, 3]
        road = features[0, 0].cpu().numpy()

        # 建筑类型
        building_types = {
            0: '广场', 1: '广场', 2: '广场',
            3: '餐厅', 4: '餐厅', 5: '餐厅', 6: '餐厅', 7: '餐厅', 8: '餐厅',
            9: '商店', 10: '商店', 11: '商店', 12: '商店', 13: '商店', 14: '商店',
            15: '商店', 16: '商店', 17: '商店', 18: '商店', 19: '商店', 20: '商店',
            21: '商店', 22: '商店', 23: '商店',
            24: '厕所', 25: '厕所',
            26: '酒店', 27: '酒店', 28: '酒店', 29: '酒店',
        }

        type_colors = {
            '广场': '#FFD700', '餐厅': '#FF6B6B', '商店': '#4ECDC4',
            '厕所': '#95E1D3', '酒店': '#A8E6CF',
        }

        # 绘制
        ax = axes[count]
        ax.imshow(road, cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.3)

        for i in range(30):
            x, y, r = layout[i]
            b_type = building_types[i]
            color = type_colors[b_type]
            circle = plt.Circle((x, y), r, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Sample {count + 1}', fontsize=12)

        count += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"多样本布局已保存: {save_path}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("训练可视化")
    print("=" * 60)

    # 创建输出目录
    Path('outputs').mkdir(exist_ok=True)

    # 1. 绘制训练曲线
    print("\n1. 绘制训练曲线...")
    history_path = 'outputs/training_history.npy'
    if Path(history_path).exists():
        plot_training_history(history_path)
    else:
        print(f"  [跳过] 训练历史文件不存在: {history_path}")

    # 2. 可视化生成的布局
    print("\n2. 可视化生成的布局...")

    checkpoint_path = 'outputs/checkpoints/best_model.pth'
    if Path(checkpoint_path).exists():
        # 加载模型
        print(f"  加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model = RoadFeatureExtractor(in_channels=5, building_num=30)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 加载一个样本
        from dataloader import RoadDataLoader
        dataloader = RoadDataLoader(batch_size=4)
        _, val_loader, _ = dataloader.get_dataloaders()

        # 获取一个batch
        for batch in val_loader:
            visualize_multiple_layouts(model, [batch], num_samples=4,
                                        save_path='outputs/layout_samples.png')
            break

        print(f"\n最佳模型能量: {checkpoint['energy']:.4f}")
    else:
        print(f"  [跳过] 模型文件不存在: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
