#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : t1.py
#   author   : ***
#   date     : 2025-11-25
#   desc     : 
# ======================================================== 
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 GPU：
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 一个最简单的一层网络
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.1)

x = torch.tensor([[1.0]])
adv = torch.tensor([1.0])   # 测试正样本
# adv = torch.tensor([-1.0])  # 换成负样本也可以测试

for step in range(5):

    optimizer.zero_grad()

    pi_new = model(x)            # 分子
    pi_new_sg = pi_new.detach()  # 分母（stop_gradient）

    ratio = pi_new / pi_new_sg   # 关键：π_new / sg(π_new)
    loss = -(ratio * adv).mean() # 加上 advantage
    #loss = -torch.log(pi_new).mean()


    print(f"\nStep {step}")
    print("pi_new:", pi_new.item())
    #print("ratio:", ratio.item())

    loss.backward()

    # 打印梯度
    print("fc.weight.grad:", model.fc.weight.grad.item())

    optimizer.step()

