"""
交互式建筑布局编辑器
功能：
1. 加载训练集中的路网图像作为背景
2. 鼠标点击放置/选择建筑
3. 拖动调整位置
4. 滚轮/按键调整半径
5. 实时计算能量
6. 保存布局数据
7. 左侧历史记录窗口，可滚动查看并加载
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataloader import RoadDataLoader
from energy.energy_function import compute_energy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class Building:
    """建筑类"""
    def __init__(self, x, y, r, building_type, b_id):
        self.x = x
        self.y = y
        self.r = r
        self.type = building_type
        self.id = b_id
        self.selected = False

    def to_tensor(self):
        """转换为tensor格式 (x, y, r)"""
        return torch.tensor([self.x, self.y, self.r])

    @property
    def color(self):
        """根据类型返回颜色"""
        colors = {
            'Plaza': '#FFD700',
            'Restaurant': '#FF6B6B',
            'Shop': '#4ECDC4',
            'Restroom': '#95E1D3',
            'Hotel': '#A8E6CF',
        }
        return colors.get(self.type, '#888888')


# 建筑类型定义
BUILDING_TYPES = {
    0: ('Plaza', (0.09, 0.12)),
    1: ('Plaza', (0.09, 0.12)),
    2: ('Plaza', (0.09, 0.12)),
    3: ('Restaurant', (0.05, 0.08)),
    4: ('Restaurant', (0.05, 0.08)),
    5: ('Restaurant', (0.05, 0.08)),
    6: ('Restaurant', (0.05, 0.08)),
    7: ('Restaurant', (0.05, 0.08)),
    8: ('Restaurant', (0.05, 0.08)),
    9: ('Shop', (0.03, 0.05)),
    10: ('Shop', (0.03, 0.05)),
    11: ('Shop', (0.03, 0.05)),
    12: ('Shop', (0.03, 0.05)),
    13: ('Shop', (0.03, 0.05)),
    14: ('Shop', (0.03, 0.05)),
    15: ('Shop', (0.03, 0.05)),
    16: ('Shop', (0.03, 0.05)),
    17: ('Shop', (0.03, 0.05)),
    18: ('Shop', (0.03, 0.05)),
    19: ('Shop', (0.03, 0.05)),
    20: ('Shop', (0.03, 0.05)),
    21: ('Shop', (0.03, 0.05)),
    22: ('Shop', (0.03, 0.05)),
    23: ('Shop', (0.03, 0.05)),
    24: ('Restroom', (0.025, 0.04)),
    25: ('Restroom', (0.025, 0.04)),
    26: ('Hotel', (0.07, 0.11)),
    27: ('Hotel', (0.07, 0.11)),
    28: ('Hotel', (0.07, 0.11)),
    29: ('Hotel', (0.07, 0.11)),
}


class InteractiveLayoutEditor:
    """交互式布局编辑器 - 左侧历史记录窗口"""

    def __init__(self, road_features, road_distance, road_circles):
        """
        初始化编辑器

        参数:
            road_features: [1, 5, 256, 256] 路网特征
            road_distance: [1, 1, 256, 256] 道路距离场
            road_circles: [1, N, 3] 预生成的道路圆
        """
        self.road_features = road_features
        self.road_distance = road_distance
        self.road_image = road_features[0, 0].numpy()

        # 使用预生成的道路圆
        if road_circles is not None:
            self.road_circles = road_circles
            print(f"使用预生成道路圆: {road_circles.shape[1]} 个")
        else:
            raise ValueError("road_circles 参数必须提供，请从 dataloader 加载预生成的数据")

        # 初始化建筑列表
        self.buildings = []
        self.current_building_id = 0
        self.selected_building = None
        self.dragging = False
        self.drag_update_counter = 0  # 用于降低更新频率

        # 历史记录相关
        self.history_files = []
        self.history_display_offset = 0  # 滚动偏移
        self.items_per_page = 10
        self.selected_history_index = None

        # 创建图形 - 使用GridSpec分栏布局
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('交互式建筑布局编辑器')

        # 创建分栏布局：左侧历史记录，右侧编辑区
        gs = GridSpec(1, 2, width_ratios=[1, 2], figure=self.fig)

        # 左侧：历史记录窗口
        self.ax_history = self.fig.add_subplot(gs[0])
        self.ax_history.set_title('历史记录', fontsize=12, fontweight='bold')
        self.ax_history.axis('off')

        # 右侧：编辑区域
        self.ax_edit = self.fig.add_subplot(gs[1])
        self.setup_edit_area()

        # 设置控制按钮
        self.setup_controls()

        # 加载历史记录
        self.load_history_list()

        # 绑定事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # 显示历史记录
        self.refresh_history_display()

        # 初始能量计算
        self.update_energy()

        plt.show()

    def setup_edit_area(self):
        """设置编辑区域"""
        # 显示背景
        # 使用 origin='lower' 确保坐标系一致：y=0 在底部，y=1 在顶部
        self.ax_edit.imshow(self.road_image, cmap='gray_r', extent=[0, 1, 0, 1],
                           alpha=0.5, origin='lower')
        self.ax_edit.set_xlim(0, 1)
        self.ax_edit.set_ylim(0, 1)
        self.ax_edit.set_aspect('equal')
        self.ax_edit.set_title('建筑布局编辑区', fontsize=12, fontweight='bold')
        self.ax_edit.set_xlabel('X', fontsize=10)
        self.ax_edit.set_ylabel('Y', fontsize=10)

        # 能量显示文本
        self.energy_text = self.ax_edit.text(0.02, 0.98, '', transform=self.ax_edit.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def setup_controls(self):
        """设置控制按钮"""
        # 底部按钮区域
        ax_save = plt.axes([0.55, 0.02, 0.08, 0.04])
        self.btn_save = Button(ax_save, '保存布局')
        self.btn_save.on_clicked(self.save_layout)

        ax_clear = plt.axes([0.64, 0.02, 0.08, 0.04])
        self.btn_clear = Button(ax_clear, '清空')
        self.btn_clear.on_clicked(self.clear_all)

        ax_random = plt.axes([0.73, 0.02, 0.08, 0.04])
        self.btn_random = Button(ax_random, '随机布局')
        self.btn_random.on_clicked(self.random_layout)

        # 历史记录控制按钮（左下角）
        ax_prev = plt.axes([0.02, 0.02, 0.06, 0.04])
        self.btn_prev = Button(ax_prev, '↑ 上一页')
        self.btn_prev.on_clicked(self.history_page_up)

        ax_next = plt.axes([0.09, 0.02, 0.06, 0.04])
        self.btn_next = Button(ax_next, '↓ 下一页')
        self.btn_next.on_clicked(self.history_page_down)

        ax_load_hist = plt.axes([0.16, 0.02, 0.08, 0.04])
        self.btn_load_hist = Button(ax_load_hist, '加载选中')
        self.btn_load_hist.on_clicked(self.load_selected_history)

        ax_refresh = plt.axes([0.25, 0.02, 0.06, 0.04])
        self.btn_refresh = Button(ax_refresh, '刷新列表')
        self.btn_refresh.on_clicked(self.refresh_and_reload)

        # 操作提示
        self.fig.text(0.5, 0.001,
                     '操作说明: 左键放置/选择 | 拖动移动 | 滚轮调半径 | Delete删除 | 左侧选择历史',
                     fontsize=9, style='italic', ha='center')

    def load_history_list(self):
        """加载历史记录列表"""
        save_dir = Path('outputs/manual_layouts')

        self.history_files = []

        if save_dir.exists():
            # 查找所有布局文件
            files = sorted(save_dir.glob('layout_*.npy'),
                         key=lambda x: x.stat().st_mtime,
                         reverse=True)

            for f in files:
                # 读取信息
                info = {'path': f, 'name': f.name}

                # 读取能量信息
                info_file = f.with_suffix('.txt')
                if info_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                if '总能量' in line:
                                    info['energy'] = line.split(':')[1].strip()
                                    break
                    except:
                        info['energy'] = 'N/A'
                else:
                    info['energy'] = 'N/A'

                # 读取时间
                mtime = f.stat().st_mtime
                info['time'] = datetime.datetime.fromtimestamp(mtime).strftime('%m-%d %H:%M')

                self.history_files.append(info)

        print(f"[OK] 已加载 {len(self.history_files)} 个历史记录")

    def refresh_and_reload(self, event):
        """刷新历史记录列表（用于刷新按钮）"""
        self.load_history_list()
        self.refresh_history_display()
        print("[OK] 历史记录已刷新")

    def refresh_history_display(self):
        """刷新历史记录显示"""
        self.ax_history.clear()
        self.ax_history.axis('off')
        self.ax_history.set_title(f'历史记录 ({len(self.history_files)}个)', fontsize=12, fontweight='bold')

        if not self.history_files:
            self.ax_history.text(0.5, 0.5, '暂无历史记录\n点击"保存布局"保存第一个布局',
                                ha='center', va='center', fontsize=12, style='italic',
                                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            return

        # 显示当前页的历史记录
        start_idx = self.history_display_offset
        end_idx = min(start_idx + self.items_per_page, len(self.history_files))

        for i in range(start_idx, end_idx):
            info = self.history_files[i]
            y_pos = 0.95 - (i - start_idx) * 0.08

            # 绘制选择框
            is_selected = (self.selected_history_index == i)
            face_color = 'lightblue' if is_selected else 'white'
            edge_color = 'blue' if is_selected else 'gray'
            line_width = 2 if is_selected else 1

            rect = Rectangle((0.05, y_pos - 0.03), 0.9, 0.06,
                            facecolor=face_color, edgecolor=edge_color,
                            linewidth=line_width, alpha=0.8,
                            transform=self.ax_history.transAxes)
            self.ax_history.add_patch(rect)

            # 文件信息（分两行显示）
            line1 = f"{i+1}. {info['name'][:20]}"
            line2 = f"   {info['time']} | E={info['energy']}"
            text = line1 + '\n' + line2
            self.ax_history.text(0.07, y_pos, text, fontsize=8,
                                transform=self.ax_history.transAxes,
                                verticalalignment='center')

        # 页码信息
        if len(self.history_files) > self.items_per_page:
            current_page = self.history_display_offset // self.items_per_page + 1
            total_page = (len(self.history_files) - 1) // self.items_per_page + 1
            self.ax_history.text(0.5, 0.08, f'第 {current_page}/{total_page} 页',
                                ha='center', fontsize=10, style='italic',
                                transform=self.ax_history.transAxes)

        self.fig.canvas.draw_idle()

    def history_page_up(self, event):
        """历史记录上一页"""
        if self.history_display_offset > 0:
            self.history_display_offset -= self.items_per_page
            self.refresh_history_display()

    def history_page_down(self, event):
        """历史记录下一页"""
        if self.history_display_offset + self.items_per_page < len(self.history_files):
            self.history_display_offset += self.items_per_page
            self.refresh_history_display()

    def load_selected_history(self, event):
        """加载选中的历史记录"""
        if self.selected_history_index is None:
            print("[!] 请先选择一个历史记录！")
            return

        info = self.history_files[self.selected_history_index]
        print(f"\n正在加载: {info['name']}")

        try:
            # 加载布局
            layout_data = np.load(info['path'])

            # 检查路网文件
            road_path = info['path'].parent / (info['path'].stem + '_road.npy')

            if road_path.exists():
                road_data = np.load(road_path)
                self.road_features = torch.from_numpy(road_data).float()
                self.road_image = self.road_features[0, 0].numpy()
                self.road_distance = self.road_features[:, 1:2, :, :]

                # TODO: 道路圆需要预生成，暂时不支持加载历史路网
                # # 重新生成道路圆
                # road_binary = self.road_features[:, 0, :, :]
                # self.road_circles = generate_road_circles(road_binary, self.road_radius)
                # print(f"[OK] 已生成 {self.road_circles.shape[1]} 个道路圆")

                # 更新背景（保持当前路网和道路圆）
                # self.ax_edit.images[0].set_data(self.road_image)
                print(f"[!] 历史记录的路网加载暂不支持，保持当前路网")
            else:
                print(f"[!] 未找到对应路网文件")

            # 清空并重建建筑
            self.buildings = []
            self.current_building_id = 0
            self.selected_building = None

            for i in range(30):
                x, y, r = layout_data[i]
                type_name, _ = BUILDING_TYPES[i]
                building = Building(x, y, r, type_name, i)
                self.buildings.append(building)

            self.current_building_id = 30

            self.redraw()
            self.update_energy()

            print(f"[OK] 布局加载完成！能量: {info['energy']}")

        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            import traceback
            traceback.print_exc()

    def get_building_at_position(self, x, y):
        """获取指定位置的建筑"""
        for building in self.buildings:
            dist = np.sqrt((building.x - x)**2 + (building.y - y)**2)
            if dist <= building.r:
                return building
        return None

    def on_click(self, event):
        """鼠标点击事件"""
        # 检查是否点击在历史记录区域
        if event.inaxes == self.ax_history:
            self.handle_history_click(event)
            return

        if event.inaxes != self.ax_edit:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:
            clicked_building = self.get_building_at_position(x, y)

            if clicked_building:
                if self.selected_building:
                    self.selected_building.selected = False
                self.selected_building = clicked_building
                clicked_building.selected = True
                self.dragging = True
            else:
                if self.current_building_id < 30:
                    type_name, _ = BUILDING_TYPES[self.current_building_id]
                    r = 0.05  # 默认半径，可自由调整

                    new_building = Building(x, y, r, type_name, self.current_building_id)
                    self.buildings.append(new_building)
                    self.selected_building = new_building
                    self.current_building_id += 1
                    self.dragging = True

            self.redraw()
            self.update_energy()

    def handle_history_click(self, event):
        """处理历史记录区域的点击"""
        if event.inaxes != self.ax_history:
            return

        # 计算点击了哪个历史记录
        y_pos = event.ydata
        if y_pos is None:
            return

        start_idx = self.history_display_offset
        end_idx = min(start_idx + self.items_per_page, len(self.history_files))

        for i in range(start_idx, end_idx):
            item_y = 0.95 - (i - start_idx) * 0.08
            if abs(y_pos - item_y) < 0.04:
                self.selected_history_index = i
                self.refresh_history_display()
                return

    def on_release(self, event):
        """鼠标释放事件"""
        if self.dragging:
            # 拖拽结束时，完整更新一次显示和能量
            self.dragging = False
            self.redraw()
            self.update_energy()
            self.drag_update_counter = 0

    def on_motion(self, event):
        """鼠标移动事件（优化帧率）"""
        if not self.dragging or self.selected_building is None:
            return

        if event.inaxes == self.ax_edit:
            x = np.clip(event.xdata, 0, 1)
            y = np.clip(event.ydata, 0, 1)

            self.selected_building.x = x
            self.selected_building.y = y

            # 降低更新频率，每3帧更新一次
            self.drag_update_counter += 1
            if self.drag_update_counter % 3 == 0:
                self.fast_redraw()

            # 能量计算更慢，每10帧更新一次
            if self.drag_update_counter % 10 == 0:
                self.update_energy()

    def fast_redraw(self):
        """快速重绘（只更新建筑）"""
        # 清除旧的圆形
        for patch in self.ax_edit.patches:
            patch.remove()

        # 清除旧的文本
        for text in self.ax_edit.texts:
            if text != self.energy_text:
                text.remove()

        # 重新绘制建筑
        for building in self.buildings:
            linewidth = 2.0 if building.selected else 0.5
            circle = Circle((building.x, building.y), building.r,
                           facecolor=building.color, alpha=0.7,
                           edgecolor='black', linewidth=linewidth)
            self.ax_edit.add_patch(circle)

            self.ax_edit.text(building.x, building.y, str(building.id),
                        ha='center', va='center', fontsize=8, fontweight='bold')

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """滚轮事件"""
        # 如果在历史记录区域滚动
        if event.inaxes == self.ax_history:
            if event.button == 'up':
                self.history_page_up(None)
            elif event.button == 'down':
                self.history_page_down(None)
            return

        # 如果在编辑区域，调整建筑半径
        if event.inaxes != self.ax_edit or self.selected_building is None:
            return

        # 自由调整半径范围：0.01 ~ 0.15
        r_min, r_max = 0.01, 0.15
        delta = 0.002 * (-event.step)
        new_r = np.clip(self.selected_building.r + delta, r_min, r_max)

        self.selected_building.r = new_r
        self.redraw()
        self.update_energy()

    def on_key_press(self, event):
        """键盘事件"""
        if event.key == 'delete' and self.selected_building is not None:
            self.buildings.remove(self.selected_building)
            self.selected_building = None
            self.redraw()
            self.update_energy()
        elif event.key == 'up':
            self.history_page_up(None)
        elif event.key == 'down':
            self.history_page_down(None)

    def redraw(self):
        """重绘所有建筑"""
        for patch in self.ax_edit.patches:
            patch.remove()

        for text in self.ax_edit.texts:
            if text != self.energy_text:
                text.remove()

        # 绘制建筑
        for building in self.buildings:
            linewidth = 2.0 if building.selected else 0.5
            circle = Circle((building.x, building.y), building.r,
                           facecolor=building.color, alpha=0.7,
                           edgecolor='black', linewidth=linewidth)
            self.ax_edit.add_patch(circle)

            self.ax_edit.text(building.x, building.y, str(building.id),
                        ha='center', va='center', fontsize=8, fontweight='bold')

        self.fig.canvas.draw_idle()

    def update_energy(self):
        """更新能量显示"""
        if len(self.buildings) == 0:
            self.energy_text.set_text(f"建筑数量: 0/30\n总能量: N/A")
            return

        layout_tensor = torch.zeros(1, 30, 3)

        for building in self.buildings:
            layout_tensor[0, building.id, 0] = building.x
            layout_tensor[0, building.id, 1] = building.y
            layout_tensor[0, building.id, 2] = building.r

        try:
            # 使用预生成的道路圆计算能量
            energy = compute_energy(layout_tensor, self.road_features, self.road_circles)
            total_energy = energy.item()
            text = f"建筑数量: {len(self.buildings)}/30\n总能量: {total_energy:.2f}"
            self.energy_text.set_text(text)
        except Exception as e:
            self.energy_text.set_text(f"能量计算错误: {str(e)}")

        self.fig.canvas.draw_idle()

    def random_layout(self, event):
        """随机生成布局"""
        self.clear_all(None)

        for i in range(30):
            type_name, _ = BUILDING_TYPES[i]
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.02, 0.12)  # 自由半径范围

            building = Building(x, y, r, type_name, i)
            self.buildings.append(building)

        self.current_building_id = 30
        self.redraw()
        self.update_energy()

    def clear_all(self, event):
        """清空所有建筑"""
        self.buildings = []
        self.current_building_id = 0
        self.selected_building = None
        self.redraw()
        self.update_energy()

    def save_layout(self, event):
        """保存布局到文件"""
        if len(self.buildings) == 0:
            print("没有建筑可保存！")
            return

        layout = np.zeros((30, 3))

        for building in self.buildings:
            layout[building.id, 0] = building.x
            layout[building.id, 1] = building.y
            layout[building.id, 2] = building.r

        save_dir = Path('outputs/manual_layouts')
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'layout_{timestamp}'

        # 保存文件
        layout_path = save_dir / f'{base_name}.npy'
        np.save(layout_path, layout)

        road_path = save_dir / f'{base_name}_road.npy'
        np.save(road_path, self.road_features.numpy())

        energy = compute_energy(torch.from_numpy(layout).unsqueeze(0).float(),
                               self.road_features, self.road_circles)
        info_path = save_dir / f'{base_name}.txt'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"总能量: {energy.item():.4f}\n")
            f.write(f"建筑数量: {len(self.buildings)}\n")
            f.write(f"保存时间: {timestamp}\n")

        print(f"\n[OK] 布局已保存: {layout_path.name}")
        print(f"  总能量: {energy.item():.4f}")

        # 刷新历史记录列表
        self.load_history_list()
        self.refresh_history_display()


def load_random_road():
    """加载随机路网（包含道路圆）"""
    print("加载数据...")
    dataloader = RoadDataLoader(batch_size=1)
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()

    import random
    all_data = list(train_loader)
    batch = random.choice(all_data)

    features = batch[0]
    road_circles = batch[1]
    road_distance = features[:, 1:2, :, :]

    print(f"道路圆数量: {road_circles.shape[1]}")
    return features, road_distance, road_circles


if __name__ == "__main__":
    print("=" * 60)
    print("交互式建筑布局编辑器")
    print("=" * 60)

    features, road_distance, road_circles = load_random_road()

    print("\n启动编辑器...")
    print("操作说明:")
    print("  - 左侧窗口: 历史记录列表（可滚动选择）")
    print("  - 右侧窗口: 建筑布局编辑区")
    print("  - 鼠标操作: 左键放置/选择 | 拖动移动 | 滚轮调半径")
    print("  - 键盘操作: Delete删除 | ↑↓翻页历史记录")

    # 使用预生成的道路圆
    editor = InteractiveLayoutEditor(features, road_distance, road_circles)

    print("\n编辑器已关闭")
