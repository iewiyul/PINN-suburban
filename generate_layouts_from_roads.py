"""
从原始道路图像生成布局可视化
加载最新训练的模型，对 data/road_features 路径下的道路图像生成布局，保存结果
"""
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False
from pathlib import Path
import sys
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from net.models import RoadFeatureExtractor


def compute_distance_field(binary: np.ndarray) -> np.ndarray:
    """计算距离场（到最近道路像素的距离）"""
    # 使用scipy的距离变换
    distance = distance_transform_edt(1 - binary)
    return distance / (distance.max() + 1e-8)


def compute_road_density(binary: np.ndarray, kernel_size: int = 10) -> np.ndarray:
    """计算道路密度（周围道路像素占比）"""
    # 创建卷积核
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    density = convolve(binary.astype(float), kernel, mode='reflect')
    return density


def compute_road_type(distance: np.ndarray) -> np.ndarray:
    """计算道路类型（基于距离场）"""
    # 距离场值大的区域视为主干道
    return distance / (distance.max() + 1e-8)


def compute_road_center(density: np.ndarray) -> np.ndarray:
    """计算道路中心性"""
    # 使用反向距离场
    center = distance_transform_edt(1 - density)
    return center / (center.max() + 1e-8)


def process_single_road_image(image_path: Path) -> torch.Tensor:
    """
    处理单张道路图像，生成5通道特征

    参数:
        image_path: 道路图像路径

    返回:
        torch.Tensor: [1, 5, 256, 256] 特征张量
    """
    # 加载图像
    img = Image.open(image_path).convert('L')  # 转灰度
    img_array = np.array(img)

    # 调整大小到256x256
    if img_array.shape[0] != 256 or img_array.shape[1] != 256:
        img = Image.fromarray(img_array).resize((256, 256), Image.BILINEAR)
        img_array = np.array(img)

    # 归一化并二值化（道路是黑色，所以用 < 0.5）
    binary = (img_array.astype(float) / 255.0 < 0.5).astype(float)

    # 计算5个通道
    channel0 = binary  # 道路二值图
    channel1 = compute_distance_field(binary)  # 距离场
    channel2 = compute_road_density(binary)  # 道路密度
    channel3 = compute_road_type(channel1)  # 道路类型
    channel4 = compute_road_center(channel2)  # 中心性

    # 转为tensor并堆叠为5通道 [1, 5, 256, 256]
    features = torch.stack([
        torch.from_numpy(channel0).float(),
        torch.from_numpy(channel1).float(),
        torch.from_numpy(channel2).float(),
        torch.from_numpy(channel3).float(),
        torch.from_numpy(channel4).float(),
    ], dim=0).unsqueeze(0)

    return features


def visualize_layout(road_image: np.ndarray, layout: np.ndarray, save_path: Path):
    """
    可视化布局（道路+建筑）

    参数:
        road_image: 道路图像 [256, 256]
        layout: 布局参数 [30, 3]
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制道路（白色背景，黑色道路）
    ax.imshow(road_image, cmap='gray_r', extent=[0, 1, 0, 1], alpha=0.5)

    # 建筑类型颜色
    building_colors = [
        '#FFD700',  # 0-2: 广场（金色）
        '#FFD700',
        '#FFD700',
        '#FF6B6B',  # 3-8: 餐厅（红色）
        '#FF6B6B',
        '#FF6B6B',
        '#FF6B6B',
        '#FF6B6B',
        '#FF6B6B',
        '#4ECDC4',  # 9-23: 商店（青色）
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#4ECDC4',
        '#95E1D3',  # 24-25: 厕所（浅青色）
        '#95E1D3',
        '#A8E6CF',  # 26-29: 酒店（浅绿色）
        '#A8E6CF',
        '#A8E6CF',
        '#A8E6CF',
    ]

    # 绘制建筑
    for i in range(30):
        x, y, r = layout[i]
        color = building_colors[i]
        circle = plt.Circle((x, y), r, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # 翻转Y轴匹配图像坐标
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    print("=" * 60)
    print("从道路图像生成布局可视化")
    print("=" * 60)

    # 配置
    road_features_dir = Path('data/road_features')
    output_dir = Path('data/road_features')
    checkpoint_path = Path('outputs/checkpoints/best_model_e353_20260316_205033.pth')

    # 检查路径
    if not road_features_dir.exists():
        print(f"错误: 道路图像目录不存在: {road_features_dir}")
        return

    if not checkpoint_path.exists():
        print(f"错误: 模型checkpoint不存在: {checkpoint_path}")
        return

    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  训练轮数: {checkpoint['epoch']}")
    print(f"  最佳能量: {checkpoint['energy']:.4f}")

    model = RoadFeatureExtractor(in_channels=5, building_num=30)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取所有道路图像
    road_images = sorted(road_features_dir.glob('road_*.png'))
    print(f"\n找到 {len(road_images)} 张道路图像")

    if len(road_images) == 0:
        print("错误: 没有找到道路图像")
        return

    # 处理每张图像
    print("\n开始生成布局...")
    success_count = 0
    skip_count = 0

    for img_path in road_images:
        try:
            # 生成输出文件名
            output_filename = f"layout_{img_path.stem}.png"
            output_path = output_dir / output_filename

            # 如果输出文件已存在且较新，跳过
            if output_path.exists():
                img_mtime = img_path.stat().st_mtime
                out_mtime = output_path.stat().st_mtime
                if out_mtime > img_mtime:
                    skip_count += 1
                    continue

            # 读取原始道路图像
            road_img = Image.open(img_path).convert('L')
            road_img_resized = road_img.resize((256, 256), Image.BILINEAR)
            road_array = np.array(road_img_resized)

            # 处理为5通道特征
            features = process_single_road_image(img_path)

            # 生成布局
            with torch.no_grad():
                layout = model(features, building_num=30)
            layout = layout.cpu().numpy()[0]  # [30, 3]

            # 可视化并保存
            visualize_layout(road_array, layout, output_path)

            success_count += 1
            if success_count % 10 == 0:
                print(f"  已处理: {success_count}/{len(road_images)}")

        except Exception as e:
            print(f"  错误: 处理 {img_path.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n完成!")
    print(f"  成功生成: {success_count} 张")
    print(f"  跳过已存在: {skip_count} 张")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
