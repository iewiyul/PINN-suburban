"""
数据增强模块 - 对路网图像进行增强
新版本：每张图像生成 1 张增强图像（原图保留）
"""

import torch
import random


# =============================================================================
# 原始版本（每张图像生成8张）- 已注释
# =============================================================================

# def augment_road_images(images: torch.Tensor) -> torch.Tensor:
#     """
#     对路网图像进行数据增强

#     每张输入图像生成 8 张增强图像（不包括原图）

#     增强方式：
#         - 2 张：随机翻转（水平翻转、垂直翻转、或两者）
#         - 2 张：随机旋转（90°、180°、270° 中选择）
#         - 4 张：非等比例缩放

#     参数:
#         images: [N, 3, H, W] 图像张量（uint8 格式，范围 [0, 255]）

#     返回:
#         [N * 8, 3, H, W] 增强后的图像张量
#     """
#     augmented_list = []
#     _, _, H, W = images.shape

#     for img in images:
#         # ==================== 2 张：随机翻转 ====================
#         for _ in range(2):
#             aug_img = img.clone()
#             aug_type = random.choice([
#                 'horizontal_flip',   # 水平翻转
#                 'vertical_flip',     # 垂直翻转
#                 'flip_both'          # 水平+垂直翻转（等价于180°旋转）
#             ])

#             if aug_type == 'horizontal_flip':
#                 aug_img = torch.flip(aug_img, dims=[2])
#             elif aug_type == 'vertical_flip':
#                 aug_img = torch.flip(aug_img, dims=[1])
#             elif aug_type == 'flip_both':
#                 aug_img = torch.flip(aug_img, dims=[1, 2])

#             augmented_list.append(aug_img)

#         # ==================== 2 张：随机旋转 ====================
#         for _ in range(2):
#             aug_img = img.clone()
#             # 随机选择旋转角度：90°、180°、270°
#             k = random.choice([1, 2, 3])  # rot90 的 k 参数
#             aug_img = torch.rot90(aug_img, k=k, dims=[1, 2])
#             augmented_list.append(aug_img)

#         # ==================== 4 张：非等比例缩放 ====================
#         for _ in range(4):
#             aug_img = img.clone().float() / 255.0  # 转换为 [0, 1] 范围

#             # 随机生成 x 和 y 方向的缩放因子
#             scale_x = random.uniform(0.85, 1.15)
#             scale_y = random.uniform(0.85, 1.15)

#             # 计算缩放后的新尺寸
#             new_h = int(H * scale_y)
#             new_w = int(W * scale_x)

#             # 使用 interpolate 进行缩放
#             aug_img_batch = aug_img.unsqueeze(0)
#             resized = torch.nn.functional.interpolate(
#                 aug_img_batch,
#                 size=(new_h, new_w),
#                 mode='nearest'
#             ).squeeze(0)

#             # 创建填充后的图像（用白色背景 1.0 填充）
#             aug_img_padded = torch.ones(3, H, W, dtype=torch.float32)

#             # 计算粘贴位置（确保居中）
#             # 如果 new_h > H，从缩放图像裁剪；否则使用 y_offset
#             if new_h >= H:
#                 y_start_src = (new_h - H) // 2
#                 y_start_dst = 0
#                 h_copy = H
#             else:
#                 y_start_src = 0
#                 y_start_dst = (H - new_h) // 2
#                 h_copy = new_h

#             # 同样处理 x 方向
#             if new_w >= W:
#                 x_start_src = (new_w - W) // 2
#                 x_start_dst = 0
#                 w_copy = W
#             else:
#                 x_start_src = 0
#                 x_start_dst = (W - new_w) // 2
#                 w_copy = new_w

#             # 从缩放后的图像复制到填充后的图像
#             aug_img_padded[:, y_start_dst:y_start_dst+h_copy, x_start_dst:x_start_dst+w_copy] = \
#                 resized[:, y_start_src:y_start_src+h_copy, x_start_src:x_start_src+w_copy]

#             aug_img = (aug_img_padded * 255).to(torch.uint8)
#             augmented_list.append(aug_img)

#     # 堆叠为张量
#     result = torch.stack(augmented_list, dim=0)
#     return result


# =============================================================================
# 新版本 - 每张图像生成 1 张增强图像（保留原图）
# =============================================================================

def augment_road_images(images: torch.Tensor) -> torch.Tensor:
    """
    对路网图像进行轻量级数据增强

    每张输入图像保留，并额外生成 1 张增强图像
    总输出量 = 输入图像数 × 2（原图 + 增强图）

    增强方式（随机选择一种）：
        - 水平翻转
        - 垂直翻转
        - 水平+垂直翻转（等价于180°旋转）
        - 旋转90°
        - 旋转180°
        - 旋转270°

    参数:
        images: [N, 3, H, W] 图像张量（uint8 格式，范围 [0, 255]）

    返回:
        [N * 2, 3, H, W] 增强后的图像张量（包含原图和增强图）

    示例:
        输入: 380张图像
        输出: 760张图像 (380张原图 + 380张增强图)
    """
    result_list = []

    for img in images:
        # 1. 添加原图
        result_list.append(img)

        # 2. 生成1张增强图（随机选择增强方式）
        aug_img = img.clone()

        aug_type = random.choice([
            'horizontal_flip',   # 水平翻转
            'vertical_flip',     # 垂直翻转
            'flip_both',         # 水平+垂直翻转（等价于180°旋转）
            'rotate_90',         # 旋转90°
            'rotate_180',        # 旋转180°
            'rotate_270',        # 旋转270°
        ])

        if aug_type == 'horizontal_flip':
            aug_img = torch.flip(aug_img, dims=[2])
        elif aug_type == 'vertical_flip':
            aug_img = torch.flip(aug_img, dims=[1])
        elif aug_type == 'flip_both':
            aug_img = torch.flip(aug_img, dims=[1, 2])
        elif aug_type == 'rotate_90':
            aug_img = torch.rot90(aug_img, k=1, dims=[1, 2])
        elif aug_type == 'rotate_180':
            aug_img = torch.rot90(aug_img, k=2, dims=[1, 2])
        elif aug_type == 'rotate_270':
            aug_img = torch.rot90(aug_img, k=3, dims=[1, 2])

        result_list.append(aug_img)

    # 堆叠为张量
    result = torch.stack(result_list, dim=0)
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("数据增强模块测试（新版本：1原图+1增强）")
    print("=" * 60)

    # 创建测试图像
    print("\n创建测试图像...")
    test_images = torch.zeros(5, 3, 256, 256, dtype=torch.uint8)

    # 第一张：左上角白色方块
    test_images[0, :, :50, :50] = 255

    # 第二张：横向条纹
    test_images[1, :, 100:150, :] = 255

    # 第三张：纵向条纹
    test_images[2, :, :, 100:150] = 255

    # 第四张：中心方块
    test_images[3, :, 100:150, 100:150] = 255

    # 第五张：四个角的方块
    test_images[4, :, :30, :30] = 255
    test_images[4, :, :30, -30:] = 255
    test_images[4, :, -30:, :30] = 255
    test_images[4, :, -30:, -30:] = 255

    print(f"输入形状: {test_images.shape}")
    print(f"输入图像数: {test_images.shape[0]}")

    # 进行增强
    print("\n执行数据增强...")
    augmented = augment_road_images(test_images)

    print(f"输出形状: {augmented.shape}")
    print(f"输出图像数: {augmented.shape[0]}")
    print(f"预期输出数: {test_images.shape[0] * 2}")

    # 验证
    if augmented.shape[0] == test_images.shape[0] * 2:
        print("[OK] 增强数量正确!")
    else:
        print("[FAIL] 增强数量不正确!")

    if augmented.dtype == torch.uint8:
        print("[OK] 输出格式正确 (uint8)!")
    else:
        print(f"[FAIL] 输出格式错误 ({augmented.dtype})!")

    # 验证原图被保留
    print("\n验证原图保留...")
    for i in range(test_images.shape[0]):
        original = test_images[i]
        augmented_original = augmented[i * 2]  # 每对的第一个是原图
        if torch.equal(original, augmented_original):
            print(f"  图像{i}: 原图已保留 [OK]")
        else:
            print(f"  图像{i}: 原图未保留 [FAIL]")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
