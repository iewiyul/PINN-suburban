"""
路网图像多通道特征生成
从路网图像提取 5 通道特征图 [5, 256, 256]

重构为类形式，便于配置管理和流程编排
"""

import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


class RoadChannelProcessor:
    """
    路网特征提取器 - 一站式处理 5 通道特征

    5 通道说明:
        通道1: 原始道路 (二值图)
        通道2: 距离场 (到最近道路的距离)
        通道3: 道路密度 (周围道路像素占比)
        通道4: 道路类型 (主干道=1.0, 分支道=0.5)
        通道5: 中心性 (到道路中心的距离)
    """

    def __init__(
        self,
        img_size: int = 256,
        road_features_dir: str = None,
        output_dir: str = 'data/processed_features'
    ):
        """
        初始化特征提取器

        参数:
            img_size: 图像尺寸
            road_features_dir: 路网图像目录
            output_dir: 特征输出目录
        """
        self.img_size = img_size
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = Path(__file__).parent.parent
        if road_features_dir is None:
            self.road_features_dir = str(current_dir / 'data' / 'road_features')
        else:
            self.road_features_dir = road_features_dir
        self.output_dir = output_dir

        # 缓存
        self._cached_images = None
        self._cached_features = None

    # ==================== 图像加载 ====================

    def load_image_as_tensor(self, image_path: str) -> torch.Tensor:
        """加载单张图像为 (3, H, W) tensor"""
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        return torch.from_numpy(img_array).permute(2, 0, 1).float()

    def load_all_road_images(self, road_dir: str = None) -> torch.Tensor:
        """
        加载目录下所有路网图像

        参数:
            road_dir: 图像目录，默认使用初始化时的 road_features_dir

        返回:
            torch.Tensor: (batch, 3, H, W)
        """
        if road_dir is None:
            road_dir = self.road_features_dir

        road_path = Path(road_dir)
        image_files = sorted(road_path.glob('road_*.png'))

        print(f"找到 {len(image_files)} 张路网图像...")

        tensor_list = []
        for img_file in image_files:
            try:
                tensor = self.load_image_as_tensor(str(img_file))
                tensor_list.append(tensor)
            except Exception as e:
                print(f"[Error] 加载 {img_file.name} 失败: {e}")

        if tensor_list:
            road_images = torch.stack(tensor_list, dim=0)
        else:
            road_images = torch.tensor([])

        print(f"成功加载 {len(road_images)} 张图像")
        return road_images

    # ==================== 通道计算 ====================

    def grayscale_to_binary(self, tensor: torch.Tensor, threshold: int = 128) -> torch.Tensor:
        """
        通道1: RGB 转二值图

        参数:
            tensor: (batch, 3, H, W) RGB 图像

        返回:
            torch.Tensor: (batch, 1, H, W) 二值图
        """
        gray = tensor[:, 0:1, :, :]
        return (gray < threshold).float()

    def compute_distance_field(self, binary: torch.Tensor) -> torch.Tensor:
        """
        通道2: 计算距离场（背景到道路的距离）

        距离定义：
        - 道路内部: 0（建筑不应该在道路内）
        - 道路外: 到最近道路像素的像素距离（归一化到[0,1]）

        参数:
            binary: (batch, 1, H, W) 二值图 (1=道路, 0=背景)

        返回:
            torch.Tensor: (batch, 1, H, W) 距离场
        """
        batch_size = binary.shape[0]
        distance_fields = []

        for i in range(batch_size): # 这个其实可以优化，懒了
            mask = binary[i, 0].cpu().numpy().astype(np.uint8)

            # 计算背景到道路的距离
            # distanceTransform 计算非零像素到最近零像素的距离
            # 使用 1-mask 反转：道路=0，背景=1，计算背景到道路的距离
            dist = cv2.distanceTransform(1 - mask, distanceType=cv2.DIST_L2, maskSize=0)

            # 归一化到 [0, 1]
            max_dist = dist.max()
            if max_dist > 0:
                dist = dist / max_dist

            dist_tensor = torch.from_numpy(dist).float().unsqueeze(0)
            distance_fields.append(dist_tensor)

        return torch.stack(distance_fields, dim=0).to(binary.device)

    def compute_road_density(self, binary: torch.Tensor, kernel_size: int = 10) -> torch.Tensor:
        """
        通道3: 计算道路密度

        参数:
            binary: (batch, 1, H, W) 二值图
            kernel_size: 滑动窗口大小

        返回:
            torch.Tensor: (batch, 1, H, W) 道路密度
        """
        batch_size = binary.shape[0]
        densities = []
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size) # 事先归一化了

        for i in range(batch_size):
            mask = binary[i, 0].cpu().numpy().astype(np.float32) # 简单来说就是提取了一个(256,256)
            density = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)# 做一个二维卷积
            density_tensor = torch.from_numpy(density).float().unsqueeze(0)
            densities.append(density_tensor)

        return torch.stack(densities, dim=0).to(binary.device)

    def compute_road_type(self, binary: torch.Tensor) -> torch.Tensor:
        """
        通道4: 计算道路类型（主干道 vs 分支道）

        参数:
            binary: (batch, 1, H, W) 二值图

        返回:
            torch.Tensor: (batch, 1, H, W) 道路类型
        """
        batch_size = binary.shape[0]
        road_types = []
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)

        for i in range(batch_size):
            mask = binary[i, 0].cpu().numpy().astype(np.uint8)

            all_roads = cv2.dilate(mask, kernel_small, iterations=1)# padding是隐式处理的
            main_roads = cv2.dilate(mask, kernel_large, iterations=1)
            main_roads = cv2.erode(main_roads, kernel_large, iterations=1)

            type_map = np.zeros_like(mask, dtype=np.float32)
            type_map[main_roads > 0] = 1.0  # 主干道
            type_map[(all_roads > 0) & (main_roads == 0)] = 0.5  # 分支道,python的高级索引真tm抽象

            type_tensor = torch.from_numpy(type_map).float().unsqueeze(0)
            road_types.append(type_tensor)

        return torch.stack(road_types, dim=0).to(binary.device)

    def compute_road_center(self, road_density: torch.Tensor) -> torch.Tensor:
        """
        通道5: 计算到道路中心的距离

        参数:
            road_density: (batch, 1, H, W) 道路密度

        返回:
            torch.Tensor: (batch, 1, H, W) 中心性
        """
        batch_size = road_density.shape[0]
        image_size = road_density.shape[2]

        # 找到密度最大的位置（中心点）
        flattened = road_density.squeeze(1).reshape(batch_size, -1)
        max_idx = torch.max(flattened, dim=1).indices
        center_y = max_idx // image_size
        center_x = max_idx % image_size

        # 生成坐标网格
        sample_x = torch.arange(image_size).view(image_size, 1).repeat(1, image_size)
        sample_x = sample_x.unsqueeze(0).expand(batch_size, -1, -1).float()
        sample_y = torch.arange(image_size).view(1, image_size).repeat(image_size, 1)
        sample_y = sample_y.unsqueeze(0).expand(batch_size, -1, -1).float()

        center_x = center_x.view(batch_size, 1, 1)
        center_y = center_y.view(batch_size, 1, 1)

        # 计算距离并归一化
        distance_map = torch.sqrt((sample_x - center_x) ** 2 + (sample_y - center_y) ** 2)
        max_dist = distance_map.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        distance_map = distance_map / (max_dist + 1e-8)

        return distance_map.unsqueeze(1)

    def compute_road_circles(self, binary: torch.Tensor, road_radius: float = 0.01) -> list:
        """
        从道路二值图生成道路圆张量列表

        对每个值为1的像素点（道路），生成一个 (x, y, r) 张量
        x, y: 归一化坐标 [0, 1]
        r: 道路半径（归一化）

        参数:
            binary: (batch, 1, H, W) 二值图 (1=道路, 0=背景)
            road_radius: 道路圆的半径（归一化到 [0, 1]）

        返回:
            list: 每个样本的道路圆张量，形状为 (N, 3)，N 是道路像素点数量
        """
        batch_size = binary.shape[0]
        image_size = binary.shape[2]
        road_circles_list = []

        for i in range(batch_size):
            # 获取道路像素点的坐标
            mask = binary[i, 0].cpu().numpy()
            road_indices = np.argwhere(mask > 0.5)  # (N, 2) 数组，每行是

            if len(road_indices) == 0:
                # 如果没有道路，创建一个空张量
                road_circles = torch.zeros(0, 3)
            else:
                # 转换为归一化坐标
                road_y = road_indices[:, 0].astype(np.float32) / (image_size - 1)
                road_x = road_indices[:, 1].astype(np.float32) / (image_size - 1)

                # 创建道路圆张量 (N, 3): (x, y, r)
                N = len(road_indices)
                road_circles = torch.zeros(N, 3)
                road_circles[:, 0] = torch.from_numpy(road_x)
                road_circles[:, 1] = torch.from_numpy(road_y)
                road_circles[:, 2] = road_radius  # 所有道路圆使用相同半径

            road_circles_list.append(road_circles)

        return road_circles_list

    # ==================== 高层功能 ====================

    def extract_all_channels(self, road_images: torch.Tensor) -> torch.Tensor:
        """
        提取所有 5 通道特征

        参数:
            road_images: (batch, 3, H, W) 原始图像

        返回:
            torch.Tensor: (batch, 5, H, W) 5 通道特征
        """
        binary = self.grayscale_to_binary(road_images)
        road_density = self.compute_road_density(binary)

        return torch.cat([
            binary,                               # 通道1
            self.compute_distance_field(binary),  # 通道2
            road_density,                         # 通道3
            self.compute_road_type(binary),       # 通道4
            self.compute_road_center(road_density)  # 通道5
        ], dim=1)

    def merge_from_directory(self, road_dir: str = None) -> torch.Tensor:
        """
        从目录加载图像并合并为 5 通道特征

        参数:
            road_dir: 图像目录，默认使用初始化时的 road_features_dir

        返回:
            torch.Tensor: (batch, 5, H, W) 5 通道特征
        """
        road_images = self.load_all_road_images(road_dir)
        if len(road_images) == 0:
            raise ValueError(f"没有找到图像文件，请检查目录: {road_dir}")
        return self.extract_all_channels(road_images)

    def save_features(self, features: torch.Tensor, output_dir: str = None, road_radius: float = 0.01) -> None:
        """
        保存特征到文件（包括道路圆）

        参数:
            features: (batch, 5, H, W) 特征
            output_dir: 输出目录，默认使用初始化时的 output_dir
            road_radius: 道路圆半径
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # 生成道路圆
        road_binary = features[:, 0:1, :, :]  # (batch, 1, H, W)
        road_circles = self.compute_road_circles(road_binary, road_radius)

        # 保存单个样本
        for i in range(features.shape[0]):
            filepath = os.path.join(output_dir, f"features_{i:04d}.npy")
            np.save(filepath, features[i].cpu().numpy())

            # 保存道路圆
            circles_filepath = os.path.join(output_dir, f"circles_{i:04d}.npy")
            np.save(circles_filepath, road_circles[i].cpu().numpy())

        # 保存完整 batch
        batch_filepath = os.path.join(output_dir, 'all_features.npy')
        np.save(batch_filepath, features.cpu().numpy())

        # 保存完整 batch 的道路圆（填充对齐）
        max_road_pixels = max([c.shape[0] for c in road_circles])
        if max_road_pixels > 0:
            padded_circles = torch.zeros(features.shape[0], max_road_pixels, 3)
            for i, circles in enumerate(road_circles):
                if circles.shape[0] > 0:
                    padded_circles[i, :circles.shape[0], :] = circles
            circles_batch_filepath = os.path.join(output_dir, 'all_circles.npy')
            np.save(circles_batch_filepath, padded_circles.cpu().numpy())
            print(f"已保存 {features.shape[0]} 个文件（含道路圆）到: {output_dir}")
            print(f"  道路圆数量: {max_road_pixels}")
        else:
            print(f"已保存 {features.shape[0]} 个文件到: {output_dir}")
            print(f"  警告: 没有检测到道路像素")

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("路网特征提取测试")
    print("=" * 60)

    # 使用类的方式
    processor = RoadChannelProcessor()
    features = processor.merge_from_directory()
    print(f"\n特征形状: {features.shape}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
