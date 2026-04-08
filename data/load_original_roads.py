r"""
真实路网数据加载模块
从 C:\Users\ANASON\Desktop\PINN-suburban\data\original mask 加载路网图像
在保持比例的情况下调整为 (3, 256, 256)
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List
import random
from scipy import ndimage


def remove_small_black_regions(binary_image: np.ndarray, min_size: int) -> np.ndarray:
    """
    移除二值图像中小的黑色连通区域（离散点/噪声）

    参数:
        binary_image: 二值图像数组 (0=黑色道路, 1=白色背景)
        min_size: 最小连通区域大小（像素数），小于此值的区域将被移除

    返回:
        去噪后的二值图像
    """
    # 复制图像避免修改原图
    result = binary_image.copy()

    # 标记连通区域（找黑色区域，即值为0的区域）
    # 反转图像以便标记黑色区域（label函数标记的是非零区域）
    inverted = 1 - binary_image
    labeled, num_features = ndimage.label(inverted)

    # 计算每个连通区域的大小
    sizes = ndimage.sum(inverted, labeled, range(num_features + 1))

    # 找到小于阈值的区域
    small_regions = [i for i, size in enumerate(sizes) if size < min_size]

    # 移除小区域（设为白色背景1）
    for region_id in small_regions:
        result[labeled == region_id] = 1.0

    return result


def load_original_roads(
    data_dir: str = r"C:\Users\ANASON\Desktop\PINN-suburban\data\original mask",
    img_size: int = 256,
    num_samples: Optional[int] = None,    # 期望返回的图像数量
    grayscale: bool = True,
    min_content_ratio: float = 0.01,     # 最小道路像素比例（低于此值的图像将被剔除）
    remove_noise: bool = True,            # 是否去除黑色离散点（噪声）
    min_region_size: int = 50             # 最小连通区域大小（小于此值的黑色区域将被移除）
) -> torch.Tensor:
    """
    加载真实路网图像，在保持比例的情况下调整为 (3, img_size, img_size)

    处理流程：
        1. 处理文件夹中的所有图像
        2. 读取原始图像（黑底灰路）
        3. 反转颜色（黑→白，灰→黑）
        4. 二值化处理（>0.5为背景1，<=0.5为道路0）
        5. 去除离散点（移除小的黑色连通区域）
        6. 检查内容比例（剔除低质量图像）
        7. 从所有有效图像中随机选取 num_samples 张返回

    参数:
        data_dir: 原始数据目录
        img_size: 目标图像尺寸
        num_samples: 期望返回的图像数量（None表示返回所有有效图像）
        grayscale: 是否转为灰度图（路网通常是单通道）
        min_content_ratio: 最小道路像素比例（低于此值的图像将被剔除）
        remove_noise: 是否去除黑色离散点（噪声）
        min_region_size: 最小连通区域大小（像素数，小于此值的黑色区域将被移除）

    返回:
        torch.Tensor: (num_samples, 3, img_size, img_size) uint8 图像张量（背景=255，道路=0）
        如果有效图像数量少于 num_samples，返回所有有效图像
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 获取所有 .tif 文件（处理所有图像）
    image_files = sorted(data_path.glob("*.tif"))
    total_files = len(image_files)

    print(f"找到 {total_files} 张图像，开始处理...")

    all_images = []
    skipped_count = 0  # 被剔除的图像计数

    for idx, img_file in enumerate(image_files):
        try:
            # 使用 PIL 读取图像
            img = Image.open(img_file)

            # 如果是灰度图，转换为 RGB（复制3次）
            if grayscale:
                img = img.convert('L')  # 确保是灰度
                # 转换为 numpy
                img_array = np.array(img, dtype=np.float32) / 255.0

                # 保持比例调整大小
                h, w = img_array.shape
                target_size = img_size

                # 计算缩放比例（保持宽高比）
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)

                # 使用 PIL 调整大小
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0

                # 创建目标大小的画布，填充黑色背景
                canvas = np.zeros((target_size, target_size), dtype=np.float32)

                # 计算粘贴位置（居中）
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2

                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_array

                # 反转颜色（黑→白，白→黑）
                canvas = 1.0 - canvas

                # 二值化处理（>0.5为背景1，<=0.5为道路0）
                canvas = (canvas > 0.5).astype(np.float32)

                # 去除离散点（移除小的黑色连通区域）
                if remove_noise:
                    canvas = remove_small_black_regions(canvas, min_region_size)

                # 检查内容比例（在去噪后检查黑色道路像素）
                # 黑色像素(道路)应该至少占一定比例
                black_pixel_ratio = (canvas < 0.5).sum() / (target_size * target_size)
                if black_pixel_ratio < min_content_ratio:
                    skipped_count += 1
                    continue

                # 转换为 uint8 格式（道路=0, 背景=255）
                canvas_uint8 = (canvas * 255).astype(np.uint8)
                # 转换为 3 通道（复制3次）
                img_tensor = torch.from_numpy(canvas_uint8).unsqueeze(0)  # [1, H, W] uint8
                img_tensor = img_tensor.repeat(3, 1, 1)  # [3, H, W] uint8 格式
            else:
                # RGB 处理
                img_array = np.array(img, dtype=np.float32) / 255.0

                if len(img_array.shape) == 2:  # 灰度图
                    img_array = np.stack([img_array] * 3, axis=2)

                h, w = img_array.shape[:2]
                target_size = img_size

                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)

                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0

                canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2

                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_array

                # 反转颜色（黑→白，灰→黑）
                canvas = 1.0 - canvas

                # 二值化处理（>0.5为背景1，<=0.5为道路0）
                canvas = (canvas > 0.5).astype(np.float32)

                # 去除离散点（对每个通道分别处理）
                if remove_noise:
                    for c in range(3):
                        canvas[:, :, c] = remove_small_black_regions(canvas[:, :, c], min_region_size)

                # 检查内容比例（在去噪后检查黑色道路像素）
                black_pixel_ratio = (canvas < 0.5).sum() / (target_size * target_size * 3)
                if black_pixel_ratio < min_content_ratio:
                    skipped_count += 1
                    continue

                # 转换为 uint8 格式（道路=0, 背景=255）
                canvas_uint8 = (canvas * 255).astype(np.uint8)
                img_tensor = torch.from_numpy(canvas_uint8).permute(2, 0, 1)  # [3, H, W] uint8 格式

            all_images.append(img_tensor)

            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(image_files)} 张图像，已跳过 {skipped_count} 张")

        except Exception as e:
            print(f"处理 {img_file.name} 时出错: {e}")
            continue

    # 堆叠为张量
    if len(all_images) > 0:
        result = torch.stack(all_images, dim=0)
        num_valid = result.shape[0]
        print(f"有效图像: {num_valid} 张（跳过了 {skipped_count} 张低质量图像）")

        # 如果指定了 num_samples，从有效图像中随机选取
        if num_samples is not None and num_samples < num_valid:
            indices = random.sample(range(num_valid), num_samples)
            indices = torch.tensor(indices, dtype=torch.long)
            result = result[indices]
            print(f"从 {num_valid} 张有效图像中随机选取了 {num_samples} 张")
    else:
        raise ValueError("没有有效的图像被加载，请检查数据或降低 min_content_ratio 阈值")

    print(f"最终返回形状: {result.shape}")
    return result


def save_selected_images(
    num_samples: int,
    data_dir: str = r"C:\Users\ANASON\Desktop\PINN-suburban\data\original mask",
    output_dir: str = r"C:\Users\ANASON\Desktop\PINN-suburban\data\selected_roads",
    img_size: int = 256,
    min_content_ratio: float = 0.02,     # 最小道路像素比例
    remove_noise: bool = True,            # 是否去除黑色离散点
    min_region_size: int = 50             # 最小连通区域大小
) -> None:
    """
    选取指定数量的图像，处理后保存为 tif 格式

    处理流程（通过 load_original_roads）：
        1. 处理所有原始图像
        2. 反转颜色（黑→白，灰→黑）
        3. 二值化处理（>0.5为背景，<=0.5为道路）
        4. 去除离散点（噪声）
        5. 从有效图像中随机选取 num_samples 张
        6. 保存为单通道 uint8 tif 格式（背景=255，道路=0）

    参数:
        num_samples: 选取的图像数量
        data_dir: 原始数据目录
        output_dir: 保存目录（会自动创建）
        img_size: 调整后的图像尺寸
        min_content_ratio: 最小道路像素比例，低于此值的图像将被跳过
        remove_noise: 是否去除黑色离散点（噪声）
        min_region_size: 最小连通区域大小（像素数）
    """
    print("=" * 60)
    print(f"选取并保存 {num_samples} 张路网图像为 tif 格式")
    print("=" * 60)

    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 调用 load_original_roads 获取处理后的图像
    print(f"\n加载并处理图像...")
    images_tensor = load_original_roads(
        data_dir=data_dir,
        img_size=img_size,
        num_samples=num_samples,
        grayscale=True,
        min_content_ratio=min_content_ratio,
        remove_noise=remove_noise,
        min_region_size=min_region_size
    )

    num_loaded = images_tensor.shape[0]
    print(f"成功处理 {num_loaded} 张图像")

    # 保存为 tif 格式（单通道）
    saved_count = 0
    for idx in range(num_loaded):
        try:
            # 从张量中提取单张图像 [3, H, W] -> [H, W]
            img_tensor = images_tensor[idx]  # [3, H, W]

            # 取第一个通道（3个通道都相同）
            img_array = img_tensor[0].numpy()  # [H, W]

            # 数据已经是 uint8 格式，无需再次转换
            img_uint8 = img_array.astype(np.uint8)

            # 转换为 PIL Image
            img_pil = Image.fromarray(img_uint8, mode='L')

            # 保存为 tif 格式
            output_name = f"road_{idx:04d}.tif"
            output_file = output_path / output_name
            img_pil.save(output_file, format='TIFF')
            saved_count += 1

            if (idx + 1) % 50 == 0:
                print(f"已保存 {idx + 1}/{num_loaded} 张图像")

        except Exception as e:
            print(f"保存第 {idx} 张图像时出错: {e}")
            continue

    print(f"\n保存完成！共保存 {saved_count} 张图像")
    print(f"保存位置: {output_path}")
    print("=" * 60)


def generate_and_process_original_roads(
    num_samples: Optional[int] = 500,
    img_size: int = 256,
    output_dir: str = 'data/processed_features'
) -> torch.Tensor:
    """
    综合处理：加载真实路网图像 → 提取5通道特征 → 保存

    参数:
        num_samples: 期望返回的图像数量（None表示全部）
        img_size: 图像尺寸
        output_dir: 输出目录

    返回:
        torch.Tensor: (num_samples, 5, img_size, img_size) 5通道特征
    """
    print("=" * 60)
    print(f"从真实路网数据加载并处理为5通道特征")
    print("=" * 60)

    # 1. 加载图像
    print("\n加载真实路网图像...")
    all_images = load_original_roads(
        data_dir=r"C:\Users\ANASON\Desktop\PINN-suburban\data\original mask",
        img_size=img_size,
        num_samples=num_samples,
        grayscale=True
    )
    print(f"加载完成！共 {len(all_images)} 张图像")

    # 2. 提取特征（使用 RoadChannelProcessor）
    print("\n提取5通道特征...")
    from channel_process import RoadChannelProcessor
    processor = RoadChannelProcessor(img_size=img_size)
    features = processor.extract_all_channels(all_images)
    print(f"特征提取完成！形状: {features.shape}")

    # 3. 保存特征
    print(f"\n保存特征到 {output_dir}...")
    processor.save_features(features, output_dir)

    print("\n" + "=" * 60)
    print("综合处理完成!")
    print("=" * 60)

    return features


if __name__ == "__main__":
    # 测试：加载 500 张样本
    # features = generate_and_process_original_roads(max_samples=500)
    # print(f"\n最终特征形状: {features.shape}")
    save_selected_images(num_samples=390,min_content_ratio=0.01)
