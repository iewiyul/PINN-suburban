"""
训练脚本 - 能量最小化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from net.models import RoadFeatureExtractor
from dataloader import RoadDataLoader
from energy.energy_function import compute_energy


# ==================== 配置参数 ====================

class TrainingConfig:
    # 数据
    data_dir = 'data/processed_features'
    batch_size = 16
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 模型
    in_channels = 5
    building_num = 30

    # 训练
    num_epochs = 500
    learning_rate = 0.001
    weight_decay = 1e-5

    # 学习率调度
    scheduler_type = 'reduce_on_plateau'  # 'reduce_on_plateau' 或 'cosine' 或 'none'
    scheduler_patience = 15               # ReduceLROnPlateau 的 patience
    scheduler_factor = 0.5                # ReduceLROnPlateau 的 factor
    scheduler_tmax = 100                  # CosineAnnealingLR 的周期

    # 早停
    patience = 50
    min_delta = 0.001

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存
    checkpoint_dir = Path('outputs/checkpoints')
    log_dir = Path('outputs/logs')


def train_epoch(model, train_loader, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_energy = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # 获取特征数据和道路圆
        features = batch[0].to(device)  # [batch, 5, 256, 256]
        road_circles = batch[1].to(device)  # [batch, N, 3]

        # 前向传播
        layout = model(features, building_num=config.building_num)

        # 计算能量（使用预生成的道路圆）
        energy = compute_energy(layout, features, road_circles, verbose=False)

        # 能量裁剪（防止异常值）
        energy = torch.clamp(energy, max=10000)

        # 反向传播
        optimizer.zero_grad()
        energy.mean().backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)

        optimizer.step()

        # 统计
        total_energy += energy.sum().item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({'Energy': f'{energy.mean().item():.2f}'})

    avg_energy = total_energy / (num_batches * config.batch_size)
    return avg_energy


def validate(model, val_loader, device, config):
    """验证"""
    model.eval()
    total_energy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            features = batch[0].to(device)
            road_circles = batch[1].to(device)
            layout = model(features, building_num=config.building_num)
            energy = compute_energy(layout, features, road_circles, verbose=False)

            total_energy += energy.sum().item()
            num_batches += 1

    avg_energy = total_energy / (num_batches * config.batch_size)
    return avg_energy


def save_checkpoint(model, optimizer, epoch, energy, config, filename):
    """保存检查点"""
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'energy': energy,
        'config': config.__dict__
    }

    torch.save(checkpoint, checkpoint_dir / filename)
    print(f"检查点已保存: {checkpoint_dir / filename}")


def train(config):
    """主训练函数"""
    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"\n设备: {config.device}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"训练轮数: {config.num_epochs}")

    # 创建输出目录
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 创建学习率调度器
    if config.scheduler_type == 'reduce_on_plateau':
        # 当验证能量停止下降时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True,
            min_lr=1e-6
        )
        use_scheduler_step = True
        print(f"学习率调度器: ReduceLROnPlateau (patience={config.scheduler_patience}, factor={config.scheduler_factor})")
    elif config.scheduler_type == 'cosine':
        # 余弦退火：周期性调整学习率
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_tmax,
            eta_min=1e-6
        )
        use_scheduler_step = True
        print(f"学习率调度器: CosineAnnealingLR (T_max={config.scheduler_tmax})")
    else:
        scheduler = None
        use_scheduler_step = False
        print(f"学习率调度器: 禁用（固定学习率）")

    # 训练历史记录
    history = {
        'train_energies': [],
        'val_energies': [],
        'learning_rates': []
    }

    # 训练循环
    best_energy = float('inf')
    patience_counter = 0

    print("\n" + "=" * 60)
    print("训练开始")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # 训练
        train_energy = train_epoch(model, train_loader, optimizer, config.device, config)

        # 验证
        val_energy = validate(model, val_loader, config.device, config)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_energies'].append(train_energy)
        history['val_energies'].append(val_energy)
        history['learning_rates'].append(current_lr)

        # 学习率调度
        if use_scheduler_step:
            if config.scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_energy)
            elif config.scheduler_type == 'cosine':
                scheduler.step()

        # 打印结果
        print(f"训练能量: {train_energy:.4f} | 验证能量: {val_energy:.4f} | LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_energy < best_energy - config.min_delta:
            best_energy = val_energy
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_energy, config, 'best_model.pth')
            print(f"✓ 新的最佳模型! 能量: {val_energy:.4f}")
        else:
            patience_counter += 1

        # 定期保存
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch, val_energy, config, f'checkpoint_epoch_{epoch+1}.pth')
            # 保存训练历史
            np.save(config.log_dir / 'training_history.npy', history)

        # 早停
        if patience_counter >= config.patience:
            print(f"\n早停触发! {config.patience} 轮没有改善")
            break

    # 保存最终训练历史
    np.save(config.log_dir / 'training_history.npy', history)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证能量: {best_energy:.4f}")
    print("=" * 60)

    # 测试
    print("\n在测试集上评估...")
    test_energy = validate(model, test_loader, config.device, config)
    print(f"测试能量: {test_energy:.4f}")

    return model, best_energy


# ==================== 主程序 ====================
if __name__ == "__main__":
    config = TrainingConfig()

    model, best_energy = train(config)

    print("\n训练结束!")
