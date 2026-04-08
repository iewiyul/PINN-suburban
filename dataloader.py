"""
数据迭代器创建模块
加载特征数据，划分数据集，创建 PyTorch DataLoader
无监督学习 - 不需要标签
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


class RoadDataLoader:
    """
    路网特征数据加载器（无监督学习）

    功能:
        - 加载特征数据
        - 划分数据集 (训练/验证/测试)
        - 创建 DataLoader
    """

    def __init__(
        self,
        data_dir: str = 'data/processed_features',
        batch_size: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        num_workers: int = 0
    ):
        """
        初始化数据加载器

        参数:
            data_dir: 特征数据目录
            batch_size: 批次大小
            train_ratio, val_ratio, test_ratio: 数据集划分比例
            random_seed: 随机种子
            num_workers: DataLoader 工作进程数
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.num_workers = num_workers

        # 验证比例
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"比例之和必须为1，当前为 {train_ratio + val_ratio + test_ratio}")

        # 数据
        self.features = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self) -> tuple:
        """
        加载特征数据和道路圆

        返回:
            tuple: (features, road_circles)
                - features: (N, 5, H, W) 特征
                - road_circles: (N, M, 3) 道路圆，M 是最大道路像素数
        """
        # 加载完整batch
        batch_path = self.data_dir / 'all_features.npy'
        if not batch_path.exists():
            raise FileNotFoundError(f"找不到特征文件: {batch_path}")

        data = np.load(batch_path)
        self.features = torch.from_numpy(data).float()

        # 尝试加载道路圆
        circles_path = self.data_dir / 'all_circles.npy'
        if circles_path.exists():
            circles_data = np.load(circles_path)
            self.road_circles = torch.from_numpy(circles_data).float()
            print(f"加载 {len(self.features)} 个样本")
            print(f"特征形状: {self.features.shape}")
            print(f"道路圆形状: {self.road_circles.shape}")
        else:
            # 如果没有预生成的道路圆，创建空张量
            self.road_circles = torch.zeros(len(self.features), 0, 3)
            print(f"加载 {len(self.features)} 个样本")
            print(f"特征形状: {self.features.shape}")
            print(f"警告: 未找到道路圆文件，将在运行时生成")

        return self.features, self.road_circles

    def split_dataset(self, features: torch.Tensor = None, road_circles: torch.Tensor = None):
        """
        划分数据集

        参数:
            features: 特征数据，为 None 则使用 self.features
            road_circles: 道路圆数据，为 None 则使用 self.road_circles

        返回:
            tuple: (train_set, val_set, test_set)
        """
        if features is None:
            features = self.features
        if road_circles is None:
            road_circles = self.road_circles

        if features is None:
            raise ValueError("请先调用 load_data() 或传入数据")

        # 创建完整数据集（包含特征和道路圆）
        full_dataset = TensorDataset(features, road_circles)

        # 计算各集合大小
        total_size = len(full_dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        print(f"\n数据集划分:")
        print(f"  训练集: {train_size} ({train_size/total_size*100:.1f}%)")
        print(f"  验证集: {val_size} ({val_size/total_size*100:.1f}%)")
        print(f"  测试集: {test_size} ({test_size/total_size*100:.1f}%)")

        # 划分数据集
        train_set, val_set, test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

        return train_set, val_set, test_set

    def create_dataloaders(
        self,
        features = None,
        road_circles = None,
        train_set = None,
        val_set = None,
        test_set = None
    ) -> tuple:
        """
        创建 DataLoader

        参数:
            features: 特征数据
            road_circles: 道路圆数据
            train_set, val_set, test_set: 数据集，为 None 则自动划分

        返回:
            tuple: (train_loader, val_loader, test_loader)
        """
        # 如果没有传入数据集，自动划分
        if train_set is None or val_set is None or test_set is None:
            train_set, val_set, test_set = self.split_dataset(features, road_circles)

        # 创建 DataLoader
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False
        )

        print(f"\nDataLoader 创建完成:")
        print(f"  训练批次数: {len(self.train_loader)}")
        print(f"  验证批次数: {len(self.val_loader)}")
        print(f"  测试批次数: {len(self.test_loader)}")

        return self.train_loader, self.val_loader, self.test_loader

    def get_dataloaders(self) -> tuple:
        """
        一站式获取 DataLoader: 加载 → 划分 → 创建

        返回:
            tuple: (train_loader, val_loader, test_loader)
        """
        features, road_circles = self.load_data()
        return self.create_dataloaders(features, road_circles)


# ==================== 快捷函数 ====================

def create_dataloaders(
    data_dir: str = 'data/processed_features',
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> tuple:
    """
    快捷创建 DataLoader

    参数:
        data_dir: 特征数据目录
        batch_size: 批次大小
        train_ratio, val_ratio, test_ratio: 数据集划分比例
        random_seed: 随机种子
        num_workers: DataLoader 工作进程数

    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    loader = RoadDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        num_workers=num_workers
    )

    return loader.get_dataloaders()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("数据迭代器测试（无监督学习）")
    print("=" * 60)

    # # 方式1: 使用类
    # print("\n方式1: 使用 RoadDataLoader 类")
    # loader = RoadDataLoader(
    #     data_dir='data/processed_features',
    #     batch_size=10,
    #     num_workers=0
    # )

    # train_loader, val_loader, test_loader = loader.get_dataloaders()

    # # 测试迭代
    # print("\n训练集迭代测试:")
    # for i, batch in enumerate(train_loader):
    #     # batch 是一个 tuple，只包含 features
    #     features = batch[0]  # TensorDataset 返回 (features,)
    #     print(f"  Batch {i}: features={features.shape}")
    #     if i >= 2:
    #         break

    # 方式2: 使用快捷函数
    print("\n方式2: 使用快捷函数")
    train_loader2, val_loader2, test_loader2 = create_dataloaders(batch_size=10, num_workers=0)
    print(f"训练集批次数: {len(train_loader2)}") 
    print(f"验证集批次数: {len(val_loader2)}")
    print(f"测试集批次数: {len(test_loader2)}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
