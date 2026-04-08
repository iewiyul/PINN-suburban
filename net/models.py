"""
路网布局生成模型
输入: 路网特征 [batch, 5, 256, 256]
输出: 布局参数 [batch, 30, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# =============================================================================
# 原始版本（已注释）
# =============================================================================

# def xavier_init(module):
#     """Xavier权重初始化"""
#     if isinstance(module, nn.Conv2d):
#         nn.init.xavier_uniform_(module.weight, gain=1.0)
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0)
#     elif isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight, gain=1.0)
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0)

# def make_vgg_block(in_channels, out_channels, num_convs=3):
#     """创建VGG块"""
#     layers = []
#     for _ in range(num_convs):
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1))#1*1的卷积层
#         in_channels = out_channels
#     layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#     return nn.Sequential(*layers)

# class RoadFeatureExtractor(nn.Module):

#     def __init__(self, in_channels: int = 5, building_num: int = 30):
#         """
#         初始化 CNN 特征提取器

#         参数:
#             in_channels: 输入通道数 (默认5，对应5个特征通道)
#         """
#         super(RoadFeatureExtractor, self).__init__()

#         self.construct = nn.Sequential(
#             make_vgg_block(in_channels, 32),
#             make_vgg_block(32, 64),
#             # make_vgg_block(64, 128),
#             # make_vgg_block(128, 256),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64,128),
#             nn.ReLU(),
#             nn.Linear(128,1024),
#             nn.ReLU(),
#             nn.Linear(1024,building_num * 3)
#         )

#         # 应用Xavier初始化
#         self.apply(xavier_init)


#     def forward(self, x: torch.Tensor, building_num) -> torch.Tensor:
#         """
#         前向传播 + 硬约束

#         参数:
#             x: [batch, 5, 256, 256] 路网特征

#         返回:
#             torch.Tensor: [batch, 30, 3] 布局参数
#         """
#         x = self.construct(x)  # [batch, 90]
#         x = x.view(x.shape[0], building_num, 3)  # [batch, 30, 3]

#         # 硬约束(归一化)
#         xy = torch.sigmoid(x[:, :, :2])  # x, y ∈ (0, 1)
#         r = 0.025 + torch.sigmoid(x[:, :, 2:]) * 0.095  # r ∈ [0.025, 0.12]

#         return torch.cat([xy, r], dim=2)  # [batch, 30, 3]

# =============================================================================
# 空间注意力版本 - 当前使用
# =============================================================================

# class RoadFeatureExtractor(nn.Module):
#     """
#     道路感知布局生成网络（带空间注意力机制）

#     核心改进：添加空间注意力分支，让模型知道"哪里有道路"
#     """

#     def __init__(self, in_channels: int = 5, building_num: int = 30):
#         super(RoadFeatureExtractor, self).__init__()

#         self.building_num = building_num

#         # ==================== 主干网络 ====================
#         self.backbone = nn.Sequential(
#             # 第1层：256x256 → 128x128
#             nn.Conv2d(in_channels, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # → 128x128

#             # 第2层：128x128 → 64x64
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # → 64x64

#             # 第3层：64x64 → 32x32
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # → 32x32
#         )

#         self.global_branch = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(128, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.ReLU(inplace=True),
#         )

#         self.attention_branch = nn.Sequential(
#             nn.Conv2d(128, 64, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 32, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, 1),
#             nn.Sigmoid(),
#         )

#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(128, 64, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64, 256),
#             nn.ReLU(inplace=True),
#         )

#         self.fusion = nn.Sequential(
#             nn.Linear(512 + 256, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(256, building_num * 3)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight, gain=1.0)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=1.0)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x: torch.Tensor, building_num: int = None) -> torch.Tensor:
#         if building_num is None:
#             building_num = self.building_num

#         batch_size = x.shape[0]

#         spatial_features = self.backbone(x)
#         global_feat = self.global_branch(spatial_features)
#         attention_map = self.attention_branch(spatial_features) #[batch,1,32,32]
#         weighted_features = spatial_features * attention_map
#         position_hint = self.position_encoder(weighted_features)

#         fused = torch.cat([global_feat, position_hint], dim=1)
#         raw_layout = self.fusion(fused)
#         raw_layout = raw_layout[:, :building_num * 3]
#         raw_layout = raw_layout.view(batch_size, building_num, 3)

#         xy = torch.sigmoid(raw_layout[:, :, :2])
#         r = 0.025 + torch.sigmoid(raw_layout[:, :, 2:]) * 0.095

#         return torch.cat([xy, r], dim=2)


# =============================================================================
# ResNet 版本 
# =============================================================================

class ResBlock(nn.Module):
    """
    ResNet 基础残差块
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接（如果维度不匹配，需要用 1x1 卷积调整），通常情况下我不用
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接，这步是关键，其实非常简单，就是加上卷积前的信息
        out = F.relu(out)
        return out


class RoadFeatureExtractor(nn.Module):
    """
    基于 ResNet 的路网布局生成网络

    结构：
        - ResNet 特征提取器（简化版 ResNet-18）
        - 全局池化 + MLP
        - 输出 30 个建筑的 (x, y, r) 参数
    """

    def __init__(self, in_channels: int = 5, building_num: int = 30):
        super(RoadFeatureExtractor, self).__init__()

        self.building_num = building_num

        # ==================== 初始卷积 ====================
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ==================== ResNet 残差块 ====================
        # Layer 1: 256x256 -> 64x64
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        # Layer 2: 64x64 -> 32x32
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )

        # Layer 3: 32x32 -> 16x16
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )

        # ==================== 输出头 ====================
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [batch, 256, 1, 1]

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, building_num * 3)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, building_num: int = None) -> torch.Tensor:
        """
        前向传播

        参数:
            x: [batch, 5, 256, 256] 道路特征
            building_num: 建筑数量（兼容旧接口）

        返回:
            torch.Tensor: [batch, 30, 3] 建筑位置 (x, y, r)
        """
        if building_num is None:
            building_num = self.building_num

        batch_size = x.shape[0]

        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))  # [batch, 64, 128, 128]
        x = self.maxpool(x)  # [batch, 64, 64, 64]

        # ResNet 殮差层
        x = self.layer1(x)  # [batch, 64, 64, 64]
        x = self.layer2(x)  # [batch, 128, 32, 32]
        x = self.layer3(x)  # [batch, 256, 16, 16]

        # 全局池化和输出
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(batch_size, 256)  # [batch, 256]

        raw_layout = self.fc(x)  # [batch, 90]
        raw_layout = raw_layout[:, :building_num * 3]
        raw_layout = raw_layout.view(batch_size, building_num, 3)

        # 硬约束
        xy = torch.sigmoid(raw_layout[:, :, :2])  # x, y ∈ (0, 1)
        r = 0.025 + torch.sigmoid(raw_layout[:, :, 2:]) * 0.095  # r ∈ [0.025, 0.12]

        return torch.cat([xy, r], dim=2)  # [batch, 30, 3]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("ResNet 模型测试")
    print("=" * 60)

    # 创建模型
    model = RoadFeatureExtractor(in_channels=5, building_num=30)
    print("\n[OK] ResNet 模型已创建")

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 4
    test_input = torch.randn(batch_size, 5, 256, 256)

    with torch.no_grad():
        output = model(test_input, building_num=30)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: x,y∈(0,1), r∈[0.025, 0.12]")

    # 测试不同输入产生不同输出
    print("\n测试输入敏感性:")
    input1 = torch.randn(1, 5, 256, 256)
    input2 = torch.randn(1, 5, 256, 256)

    with torch.no_grad():
        output1 = model(input1)
        output2 = model(input2)

    diff = (output1 - output2).abs().mean().item()
    print(f"两个随机输入的输出差异: {diff:.6f}")
    print(f"期望: 差异应该较大（说明模型对输入敏感）")

    print("\n[OK] 测试通过!")
    print("=" * 60)
