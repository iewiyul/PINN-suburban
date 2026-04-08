"""
生成模拟路网数据 - 三种形态
树状、放射状、网状
"""

import cv2
import numpy as np
import os
import random

# ============ 配置区 ============
# 选择模式：True=测试单张，False=生成30张
TEST_MODE = False

# 生成数量（当TEST_MODE=False时生效）
NUM_SAMPLES = 120

# 图像尺寸（所有生成图像统一使用）
IMAGE_SIZE = 256

 # 主干线（水平或垂直）- 主路加粗
MAIN_ROAD_THICKNESS = 4  # 主路粗度
BRANCH_THICKNESS = 2     # 分支粗度

# 输出目录
OUTPUT_DIR = 'data/road_features'
# =============================


def generate_tree_network(img_size):
    """
    树状路网：一条主干道 + 多条分支
    分支数量：2-3条，分支从主干延伸，分支不能有交叉
    分支间有足够间距，尽可能延伸至边界
    主路（主干）加粗显示
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # 分支数量 2-3条
    num_branches = random.randint(2, 3)

    if random.choice([True, False]):
        # 水平主干道
        main_y = img_size // 2
        cv2.line(img, (0, main_y), (img_size, main_y), (0, 0, 0), MAIN_ROAD_THICKNESS)

        # 分支在主干道上的位置，确保有足够间距
        # 主干道中间区域长度为 img_size // 2
        # 最小间距设为该区域的 1/4
        min_spacing = (img_size // 2) // 4

        branch_positions = []
        if num_branches == 2:
            # 2条分支：分布在1/3和2/3位置
            pos1 = img_size // 4 + (img_size // 2) // 3
            pos2 = img_size // 4 + 2 * (img_size // 2) // 3
            branch_positions = [pos1, pos2]
        else:  # 3条分支
            # 3条分支：均匀分布在中间区域
            for i in range(3):
                progress = (i + 1) / 4
                branch_x = int(img_size // 4 + progress * (img_size // 2))
                branch_positions.append(branch_x)

        # 选择分支方向策略
        use_same_direction = random.choice([True, False])

        # 生成并绘制分支
        for i in range(num_branches):
            branch_x = branch_positions[i]

            if use_same_direction:
                # 所有分支同向
                direction = random.choice([-1, 1])  # -1=向上, 1=向下
            else:
                # 交替方向（从左到右交替）
                direction = 1 if i % 2 == 0 else -1

            # 基础角度（垂直于主干道）
            if direction == 1:
                base_angle = np.pi / 2  # 向下 90度
            else:
                base_angle = -np.pi / 2  # 向上 -90度

            # 添加角度偏移（±15度内，进一步减少以避免交叉）
            angle = base_angle + random.uniform(-np.pi / 12, np.pi / 12)

            # 长度：98%概率延伸到边界，2%概率不延伸
            if random.random() < 0.98:
                # 延伸到边界
                if direction == 1:
                    max_length = img_size - main_y - 5
                else:
                    max_length = main_y - 5
                length = int(max_length / abs(np.sin(angle))) + random.randint(0, 10)
            else:
                # 不延伸到边界（少数情况）
                length = random.randint(img_size // 6, img_size // 4)

            # 绘制分支（较细）
            end_x = int(branch_x + length * np.cos(angle))
            end_y = int(main_y + length * np.sin(angle))
            end_x = max(0, min(img_size - 1, end_x))
            end_y = max(0, min(img_size - 1, end_y))
            cv2.line(img, (branch_x, main_y), (end_x, end_y), (0, 0, 0), BRANCH_THICKNESS)

    else:
        # 垂直主干道
        main_x = img_size // 2
        cv2.line(img, (main_x, 0), (main_x, img_size), (0, 0, 0), MAIN_ROAD_THICKNESS)

        # 分支在主干道上的位置，确保有足够间距
        min_spacing = (img_size // 2) // 4

        branch_positions = []
        if num_branches == 2:
            # 2条分支：分布在1/3和2/3位置
            pos1 = img_size // 4 + (img_size // 2) // 3
            pos2 = img_size // 4 + 2 * (img_size // 2) // 3
            branch_positions = [pos1, pos2]
        else:  # 3条分支
            # 3条分支：均匀分布在中间区域
            for i in range(3):
                progress = (i + 1) / 4
                branch_y = int(img_size // 4 + progress * (img_size // 2))
                branch_positions.append(branch_y)

        # 选择分支方向策略
        use_same_direction = random.choice([True, False])

        # 生成并绘制分支
        for i in range(num_branches):
            branch_y = branch_positions[i]

            if use_same_direction:
                # 所有分支同向
                direction = random.choice([-1, 1])  # -1=向左, 1=向右
            else:
                # 交替方向
                direction = 1 if i % 2 == 0 else -1

            # 基础角度（垂直于主干道）
            if direction == 1:
                base_angle = 0  # 向右 0度
            else:
                base_angle = np.pi  # 向左 180度

            # 添加角度偏移（±15度内）
            angle = base_angle + random.uniform(-np.pi / 12, np.pi / 12)

            # 长度：98%概率延伸到边界，2%概率不延伸
            if random.random() < 0.7:
                # 延伸到边界
                if direction == 1:
                    max_length = img_size - main_x - 5
                else:
                    max_length = main_x - 5
                length = int(max_length / abs(np.cos(angle))) + random.randint(0, 10)
            else:
                # 不延伸到边界（少数情况）
                length = random.randint(img_size // 6, img_size // 4)

            # 绘制分支（较细）
            end_x = int(main_x + length * np.cos(angle))
            end_y = int(branch_y + length * np.sin(angle))
            end_x = max(0, min(img_size - 1, end_x))
            end_y = max(0, min(img_size - 1, end_y))
            cv2.line(img, (main_x, branch_y), (end_x, end_y), (0, 0, 0), BRANCH_THICKNESS)

    return img


def generate_radial_network(img_size):
    """
    放射状路网：从中心圆向外辐射
    主路（中心圆）加粗显示
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # 中心点
    center_x = img_size // 2
    center_y = img_size // 2

    # 中心圆半径（圆内没有道路）
    inner_radius = img_size // 10

    # 绘制中心圆边界（主路，加粗）- 使用全局配置
    cv2.circle(img, (center_x, center_y), inner_radius, (0, 0, 0), MAIN_ROAD_THICKNESS)

    # 辐射道路数量 3-5条
    num_rays = random.randint(3, 5)

    # 生成角度：均匀分布，覆盖整张图
    angles = []

    # 基础角度间隔（360度除以道路数量）
    base_angle = 2 * np.pi / num_rays

    # 生成均匀分布的角度
    for i in range(num_rays):
        # 均匀角度 + 小幅随机偏移
        angle = base_angle * i + random.uniform(-0.1, 0.1)
        angles.append(angle)

    # 生成辐射道路
    for angle in angles:
        # 起点：圆边缘
        start_x = int(center_x + inner_radius * np.cos(angle))
        start_y = int(center_y + inner_radius * np.sin(angle))

        # 终点：向外延伸，长度随机但偏向较长
        # 使用加权随机，使长道路概率更大
        rand_val = random.random()
        if rand_val < 0.1:
            # 10%概率：较短（不超过图像1/3）
            length = random.randint(img_size // 5, img_size // 3)
        elif rand_val < 0.3:
            # 20%概率：中等（1/3到1/2）
            length = random.randint(img_size // 3, img_size // 2)
        else:
            # 70%概率：较长，更可能延伸到边界
            length = random.randint(img_size // 2, img_size - inner_radius - 10)

        # 添加一些角度随机偏移，使道路不完全笔直
        angle_offset = random.uniform(-0.05, 0.05)

        end_x = int(start_x + length * np.cos(angle + angle_offset))
        end_y = int(start_y + length * np.sin(angle + angle_offset))
        end_x = max(0, min(img_size - 1, end_x))
        end_y = max(0, min(img_size - 1, end_y))

        # 绘制分支（较细）
        cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 0, 0), BRANCH_THICKNESS)

    return img


def generate_anchor_network(img_size):
    """
    锚点辐射路网：从靠近中心的锚点向外辐射
    锚点位置有小幅波动，辐射道路最大数量4条
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # 锚点位置：靠近中心，有波动
    # 波动范围：中心点的 ±img_size//8
    center_x = img_size // 2 + random.randint(-img_size // 8, img_size // 8)
    center_y = img_size // 2 + random.randint(-img_size // 8, img_size // 8)
    # 确保在图像范围内
    center_x = max(img_size // 4, min(3 * img_size // 4, center_x))
    center_y = max(img_size // 4, min(3 * img_size // 4, center_y))

    # 辐射道路数量 3-4条
    num_rays = random.randint(3, 4)

    # 生成角度：均匀分布，覆盖整张图
    angles = []

    # 基础角度间隔（360度除以道路数量）
    base_angle = 2 * np.pi / num_rays

    # 生成均匀分布的角度
    for i in range(num_rays):
        # 均匀角度 + 小幅随机偏移
        angle = base_angle * i + random.uniform(-0.1, 0.1)
        angles.append(angle)

    # 绘制锚点（主路，加粗）- 锚点是一个点
    cv2.line(img, (center_x, center_y), (center_x, center_y), (0, 0, 0), MAIN_ROAD_THICKNESS)

    # 生成辐射道路
    for angle in angles:
        # 起点：锚点中心
        start_x = center_x
        start_y = center_y

        # 终点：向外延伸，长度随机但偏向较长
        # 使用加权随机，使长道路概率更大
        rand_val = random.random()

        if rand_val < 0.1:
            # 5%概率：较短（不超过图像1/3）
            length = random.randint(img_size // 5, img_size // 3)
        elif rand_val < 0.3:
            # 5%概率：中等（1/3到1/2）
            length = random.randint(img_size // 3, img_size // 2)
        else:
            # 95%概率：延伸到图像边界
            # 添加一些角度随机偏移
            angle_offset = random.uniform(-0.05, 0.05)
            final_angle = angle + angle_offset

            # 根据角度方向计算到边界的距离
            cos_a = np.cos(final_angle)
            sin_a = np.sin(final_angle)

            # 初始化所有距离为无穷大
            dist_to_right = float('inf')
            dist_to_left = float('inf')
            dist_to_bottom = float('inf')
            dist_to_top = float('inf')

            # 计算沿当前方向到边界的距离
            if abs(cos_a) > 0.01:  # 不是完全垂直
                if cos_a > 0:
                    dist_to_right = (img_size - 5 - start_x) / cos_a
                else:
                    dist_to_left = (start_x - 5) / abs(cos_a)

            if abs(sin_a) > 0.01:  # 不是完全水平
                if sin_a > 0:
                    dist_to_bottom = (img_size - 5 - start_y) / sin_a
                else:
                    dist_to_top = (start_y - 5) / abs(sin_a)

            # 选择最短的有效距离（确保到达某个边界）
            length = int(min(dist_to_right, dist_to_left, dist_to_bottom, dist_to_top))
            length = max(length, 1) + random.randint(0, 10)

        # 添加一些角度随机偏移，使道路不完全笔直
        angle_offset = random.uniform(-0.05, 0.05)

        end_x = int(start_x + length * np.cos(angle + angle_offset))
        end_y = int(start_y + length * np.sin(angle + angle_offset))
        end_x = max(0, min(img_size - 1, end_x))
        end_y = max(0, min(img_size - 1, end_y))

        # 绘制分支（较细）
        cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 0, 0), BRANCH_THICKNESS)

    return img


def generate_grid_network(img_size):
    """
    网状路网：网格状交叉道路，可旋转
    道路数量：2-4条（横向+纵向之和）
    主路（最长道路）加粗显示，确保主路贯穿整张图
    所有道路真正延伸到边界
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # 随机选择道路数量组合（横向+纵向 = 总共2-4条）
    road_config = random.choice([
        (1, 1),  # 1横 + 1纵 = 2条
        (1, 2),  # 1横 + 2纵 = 3条
        (2, 1),  # 2横 + 1纵 = 3条
        (2, 2),  # 2横 + 2纵 = 4条
    ])
    num_h, num_v = road_config

    # 是否旋转网格（实现类对角线效果）
    # 暂时注释掉旋转功能，先训练水平/垂直道路的模型
    # rotate_grid = random.random() < 0.7
    rotate_grid = False  # 强制不旋转，只生成水平/垂直道路
    rotation_angle = 0

    if rotate_grid:
        # 随机旋转角度（15度到75度，模拟对角线效果）
        rotation_angle = random.uniform(np.pi / 12, np.pi / 3)  # 15-60度

    # 存储所有道路信息：(x1, y1, x2, y2, length, is_through)
    # is_through: 是否贯穿整张图
    roads = []

    if rotate_grid:
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)

        # 横向线（旋转后）- 确保延伸到边界
        for i in range(1, num_h + 1):
            y_orig = i * img_size // (num_h + 2)

            # 计算旋转后的中心点
            center_x = img_size / 2
            center_y = y_orig

            # 旋转后的中心点
            rot_center_x = (center_x - img_size/2) * cos_a - (center_y - img_size/2) * sin_a + img_size/2
            rot_center_y = (center_x - img_size/2) * sin_a + (center_y - img_size/2) * cos_a + img_size/2

            # 道路方向：横向线旋转后的角度
            road_angle = rotation_angle  # 横向线旋转后的方向

            # 计算从中心点沿道路方向到边界的距离
            cos_r = np.cos(road_angle)
            sin_r = np.sin(road_angle)

            # 计算到四个边界的距离
            dist_to_right = (img_size - 5 - rot_center_x) / cos_r if cos_r > 0.01 else float('inf')
            dist_to_left = (rot_center_x - 5) / (-cos_r) if cos_r < -0.01 else float('inf')
            dist_to_bottom = (img_size - 5 - rot_center_y) / sin_r if sin_r > 0.01 else float('inf')
            dist_to_top = (rot_center_y - 5) / (-sin_r) if sin_r < -0.01 else float('inf')

            # 正方向距离（沿道路方向）
            dist_forward = min(dist_to_right, dist_to_bottom, dist_to_left, dist_to_top)
            # 负方向距离（沿道路反方向）
            dist_backward = min(
                (rot_center_x - 5) / cos_r if cos_r > 0.01 else float('inf'),
                (rot_center_y - 5) / sin_r if sin_r > 0.01 else float('inf'),
                (img_size - 5 - rot_center_x) / (-cos_r) if cos_r < -0.01 else float('inf'),
                (img_size - 5 - rot_center_y) / (-sin_r) if sin_r < -0.01 else float('inf')
            )

            # 计算起点和终点（沿道路方向延伸到边界）
            x_rot_start = int(rot_center_x - dist_backward * cos_r)
            y_rot_start = int(rot_center_y - dist_backward * sin_r)
            x_rot_end = int(rot_center_x + dist_forward * cos_r)
            y_rot_end = int(rot_center_y + dist_forward * sin_r)

            # 确保在图像范围内
            x_rot_start = max(0, min(img_size - 1, x_rot_start))
            y_rot_start = max(0, min(img_size - 1, y_rot_start))
            x_rot_end = max(0, min(img_size - 1, x_rot_end))
            y_rot_end = max(0, min(img_size - 1, y_rot_end))

            # 计算实际长度
            length = np.sqrt((x_rot_end - x_rot_start)**2 + (y_rot_end - y_rot_start)**2)
            roads.append((x_rot_start, y_rot_start, x_rot_end, y_rot_end, length, True))

        # 纵向线（旋转后）- 确保延伸到边界
        for i in range(1, num_v + 1):
            x_orig = i * img_size // (num_v + 2)

            # 计算旋转后的中心点
            center_x = x_orig
            center_y = img_size / 2

            # 旋转后的中心点
            rot_center_x = (center_x - img_size/2) * cos_a - (center_y - img_size/2) * sin_a + img_size/2
            rot_center_y = (center_x - img_size/2) * sin_a + (center_y - img_size/2) * cos_a + img_size/2

            # 道路方向：纵向线旋转后的角度（与横向垂直）
            road_angle = rotation_angle + np.pi / 2

            # 计算从中心点沿道路方向到边界的距离
            cos_r = np.cos(road_angle)
            sin_r = np.sin(road_angle)

            # 计算到四个边界的距离
            dist_to_right = (img_size - 5 - rot_center_x) / cos_r if cos_r > 0.01 else float('inf')
            dist_to_left = (rot_center_x - 5) / (-cos_r) if cos_r < -0.01 else float('inf')
            dist_to_bottom = (img_size - 5 - rot_center_y) / sin_r if sin_r > 0.01 else float('inf')
            dist_to_top = (rot_center_y - 5) / (-sin_r) if sin_r < -0.01 else float('inf')

            # 正方向距离（沿道路方向）
            dist_forward = min(dist_to_right, dist_to_bottom, dist_to_left, dist_to_top)
            # 负方向距离（沿道路反方向）
            dist_backward = min(
                (rot_center_x - 5) / cos_r if cos_r > 0.01 else float('inf'),
                (rot_center_y - 5) / sin_r if sin_r > 0.01 else float('inf'),
                (img_size - 5 - rot_center_x) / (-cos_r) if cos_r < -0.01 else float('inf'),
                (img_size - 5 - rot_center_y) / (-sin_r) if sin_r < -0.01 else float('inf')
            )

            # 计算起点和终点（沿道路方向延伸到边界）
            x_rot_start = int(rot_center_x - dist_backward * cos_r)
            y_rot_start = int(rot_center_y - dist_backward * sin_r)
            x_rot_end = int(rot_center_x + dist_forward * cos_r)
            y_rot_end = int(rot_center_y + dist_forward * sin_r)

            # 确保在图像范围内
            x_rot_start = max(0, min(img_size - 1, x_rot_start))
            y_rot_start = max(0, min(img_size - 1, y_rot_start))
            x_rot_end = max(0, min(img_size - 1, x_rot_end))
            y_rot_end = max(0, min(img_size - 1, y_rot_end))

            # 计算实际长度
            length = np.sqrt((x_rot_end - x_rot_start)**2 + (y_rot_end - y_rot_start)**2)
            roads.append((x_rot_start, y_rot_start, x_rot_end, y_rot_end, length, True))

    else:
        # 标准正交网格（不旋转）
        # 横向道路 - 贯穿整个图像
        for i in range(1, num_h + 1):
            y = i * img_size // (num_h + 2)
            length = img_size - 1
            roads.append((0, y, img_size - 1, y, length, True))

        # 纵向道路 - 贯穿整个图像
        for i in range(1, num_v + 1):
            x = i * img_size // (num_v + 2)
            length = img_size - 1
            roads.append((x, 0, x, img_size - 1, length, True))

    # 找出最长的道路作为主路（只选一条）
    # 所有道路都贯穿整张图，选择最长的一条作为主路
    if roads:
        # 按长度排序，找出最长的
        roads.sort(key=lambda r: r[4], reverse=True)

        # 找出所有最长的道路（可能有长度相同的多条）
        longest_length = roads[0][4]
        longest_roads = [r for r in roads if abs(r[4] - longest_length) < 1]

        # 从最长的道路中随机选择一条作为主路
        main_road = random.choice(longest_roads) if longest_roads else roads[0]

        # 绘制道路
        for x1, y1, x2, y2, length, is_through in roads:
            # 检查是否是选中的主路
            if (x1, y1, x2, y2) == (main_road[0], main_road[1], main_road[2], main_road[3]):
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), MAIN_ROAD_THICKNESS)
            else:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), BRANCH_THICKNESS)

    return img


def generate_synthetic_road_network(img_size, network_type=None):
    """
    生成模拟路网图像

    参数:
        img_size: 图像尺寸
        network_type: 路网类型 ('tree', 'radial', 'anchor', 'grid', None=随机)

    返回:
        [img_size, img_size, 3] BGR图像（白底黑路）
    """
    # 随机选择类型
    if network_type is None:
        network_type = random.choice(['tree', 'radial', 'anchor', 'grid'])

    # 根据类型生成
    if network_type == 'tree':
        return generate_tree_network(img_size)
    elif network_type == 'radial':
        return generate_radial_network(img_size)
    elif network_type == 'anchor':
        return generate_anchor_network(img_size)
    else:  # grid
        return generate_grid_network(img_size)


def generate_multiple_synthetic_roads(num_samples=30, output_dir='data/road_features', img_size=512):
    """
    生成多个模拟路网图像

    参数:
        num_samples: 生成数量
        output_dir: 输出目录
        img_size: 图像尺寸
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"生成 {num_samples} 张模拟路网图像 (尺寸: {img_size}x{img_size})...")
    count = 0

    # 确保每种类型数量大致相等
    # types = ['tree', 'radial', 'anchor', 'grid']
    # samples_per_type = num_samples // len(types)
    types = ['tree','grid']
    samples_per_type = num_samples // len(types)

    for network_type in types:
        for i in range(samples_per_type):
            # 生成图像
            img = generate_synthetic_road_network(img_size, network_type)

            # 保存
            filename = f"road_{network_type}_{count:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

            print(f"[{count+1}/{num_samples}] {filename}")
            count += 1

    # 如果有余数，随机添加
    while count < num_samples:
        network_type = random.choice(types)
        img = generate_synthetic_road_network(img_size, network_type)
        filename = f"road_{network_type}_{count:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"[{count+1}/{num_samples}] {filename}")
        count += 1

    print(f"\n完成! 共生成 {count} 个模拟路网图像")
    print(f"保存位置: {output_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("模拟路网数据生成器 - 四种形态")
    print("=" * 50)

    if TEST_MODE:
        # 只生成单张测试
        print("\n模式: 测试单张")
        print("\n生成单张测试图像...")
        for network_type in ['tree', 'radial', 'anchor', 'grid']:
            test_img = generate_synthetic_road_network(img_size=IMAGE_SIZE, network_type=network_type)
            filename = f'test_{network_type}.png'
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), test_img)
            print(f"[OK] 已保存: {OUTPUT_DIR}/{filename}")
    else:
        # 生成多张
        print(f"\n模式: 批量生成 {NUM_SAMPLES} 张")
        generate_multiple_synthetic_roads(NUM_SAMPLES, OUTPUT_DIR, img_size=IMAGE_SIZE)

    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)
