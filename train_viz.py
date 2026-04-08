"""
训练 + 实时可视化工作流
调用现有训练方法，添加实时可视化和布局生成
"""
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train import TrainingConfig, train_epoch, validate, save_checkpoint
from net.models import RoadFeatureExtractor
from dataloader import RoadDataLoader
from energy.energy_function import compute_energy
from generate_test_layouts import generate_test_layouts

# 设置字体（避免中文显示问题）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_training_curve(train_energies, val_energies, learning_rates):
    """绘制实时训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_energies) + 1)

    # 能量曲线
    axes[0].plot(epochs, train_energies, label='Train Energy', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, val_energies, label='Val Energy', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 学习率曲线
    axes[1].plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    return fig


def train_with_visualization(config):
    """带实时可视化的训练"""
    print("=" * 60)
    print("训练 + 实时可视化")
    print("=" * 60)
    print(f"\n设备: {config.device}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")

    # 创建输出目录
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    layouts_dir = Path('outputs/layouts')
    layouts_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    print("\n加载数据...")
    dataloader = RoadDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size
    )
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()

    # 创建模型
    print("\n创建模型...")
    model = RoadFeatureExtractor(
        in_channels=config.in_channels,
        building_num=config.building_num
    ).to(config.device)

    # 创建优化器（从train.py导入）
    import torch.optim as optim
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器（已禁用，保持固定学习率）
    # from torch.optim import lr_scheduler
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=20
    # )

    # 训练历史
    history = {
        'train_energies': [],
        'val_energies': [],
        'learning_rates': []
    }

    best_energy = float('inf')
    patience_counter = 0

    # 记录训练开始时已存在的最佳模型（避免删除之前训练的模型）
    existing_models = set(config.checkpoint_dir.glob('best_model_*.pth'))
    training_created_models = []  # 记录本次训练创建的模型

    # 启用matplotlib交互模式（实时更新）
    plt.ion()

    # 初始化图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    line_train, = axes[0].plot([], [], label='Train Energy', linewidth=2, alpha=0.8)
    line_val, = axes[0].plot([], [], label='Val Energy', linewidth=2, alpha=0.8)
    line_lr, = axes[1].plot([], [], label='Learning Rate', linewidth=2, color='orange')

    # 设置坐标轴
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    print("\n" + "=" * 60)
    print("训练开始（实时可视化）")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        # 调用现有的训练和验证方法
        #train_loader每次都应该打乱一下吧
        train_energy = train_epoch(model, train_loader, optimizer, config.device, config)
        val_energy = validate(model, val_loader, config.device, config)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_energies'].append(train_energy)
        history['val_energies'].append(val_energy)
        history['learning_rates'].append(current_lr)

        # 学习率调度（已禁用，保持固定学习率）
        # scheduler.step(val_energy)

        # 打印结果
        print(f"Epoch {epoch + 1}/{config.num_epochs} | "
              f"Train: {train_energy:.4f} | Val: {val_energy:.4f} | LR: {current_lr:.6f}")

        # 保存最佳模型（带时间戳，保留训练历史）
        if val_energy < best_energy - config.min_delta:
            best_energy = val_energy
            patience_counter = 0

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_name = f'best_model_e{epoch+11}_{timestamp}.pth'
            new_model_path = config.checkpoint_dir / new_model_name

            # 保存新模型
            save_checkpoint(model, optimizer, epoch, val_energy, config, new_model_name)

            # 记录本次训练创建的模型
            training_created_models.append(new_model_path)

            if len(training_created_models) > 1:
                # 删除本次训练的上一个模型（保留历史和其他训练的模型）
                previous_model = training_created_models[-2]
                previous_model.unlink()
                print(f"  ✓ 新的最佳模型! 能量: {val_energy:.4f}")
                print(f"     文件: {new_model_name}")
                print(f"     已删除本次训练的上一个模型")
            else:
                # 本次训练的第一个最佳模型
                print(f"  ✓ 新的最佳模型! 能量: {val_energy:.4f}")
                print(f"     文件: {new_model_name}")
        else:
            patience_counter += 1

        # 实时更新图表（每个epoch都更新）
        epochs = list(range(1, len(history['train_energies']) + 1))

        line_train.set_data(epochs, history['train_energies'])
        line_val.set_data(epochs, history['val_energies'])
        line_lr.set_data(epochs, history['learning_rates'])

        axes[0].relim()
        axes[0].autoscale_view()
        axes[1].relim()
        axes[1].autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

        # 保存历史
        np.save(config.log_dir / 'training_history.npy', history)

        # 早停
        if patience_counter >= config.patience:
            print(f"\n早停触发! {config.patience} 轮没有改善")
            break

    # 关闭交互模式
    plt.ioff()
    plt.show()

    # 保存最终历史
    np.save(config.log_dir / 'training_history.npy', history)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证能量: {best_energy:.4f}")
    print("=" * 60)

    # 查找最新的最佳模型文件
    # print("\n查找最佳模型...")
    # model_files = list(config.checkpoint_dir.glob('best_model_*.pth'))
    # if model_files:
    #     model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    #     best_model_path = model_files[0]
    #     print(f"找到最佳模型: {best_model_path.name}")
    # else:
    #     print("[错误] 找不到最佳模型文件")
    #     return model, best_energy

    # # 在测试集上生成布局（传入最佳模型路径）
    # print("\n在测试集上生成布局...")
    # generate_test_layouts(test_loader, config=config, num_samples=10, best_model_path=best_model_path)

    return model, best_energy


# ==================== 主程序 ====================
if __name__ == "__main__":
    config = TrainingConfig()

    model, best_energy = train_with_visualization(config)

    print("\n训练结束!")
