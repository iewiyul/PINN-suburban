import torch

def constraint_boundary(output: torch.Tensor) -> torch.Tensor:
    """
    圆不能超出边界
    输入：（batchsize, 30, 3）
    """

    r = output[:, :, 2]
    building_num = output.shape[1]
    
    # 安全的边界约束计算
    r_clamped = torch.clamp(r, 0, 0.5)  # 防止r过大

    x1 = output[:, :, 0] - r_clamped
    x2 = 1 - output[:, :, 0] - r_clamped
    y1 = output[:, :, 1] - r_clamped
    y2 = 1 - output[:, :, 1] - r_clamped

    all = torch.stack([x1,x2,y1,y2],dim=0)  # [4, batch, 30]

    # 使用arctan惩罚（梯度稳定，有上界）
    violation = torch.relu(-all)  # 只惩罚违反约束的
    penalty = torch.atan(violation * 2.0) * (2/3.14159)  # 归一化到(0,1)
    penalty = penalty.sum(dim=(0, 2)) # [batch]
    return penalty

if __name__ == '__main__':

    output = torch.randn((10,30,3))
    penalty = constraint_boundary(output)
    print(penalty.shape)
    print(penalty)
    
    print('\n'+'='*60)
    print("测试成功")
    print('='*60)
    pass