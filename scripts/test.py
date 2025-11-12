# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : test.py
#   author   : ***
#   date     : 2025-08-22
#   desc     : 
# ======================================================== 


import torch
import json

# 加载 TorchScript 模型

loaded_model = torch.jit.load("../data/serving/recmodel_new.pt")
loaded_model.eval()  # 切换到 eval 模式

with open("sample", "r", encoding="utf-8") as f:
    batch_json = json.load(f)[:1]

inputs = dict()
device = "cuda:0"

for k in batch_json[0]:
    inputs[k] = torch.tensor([batch_json[i][k] for i in range(1)]).to(device)
    
print(loaded_model(inputs))
print(loaded_model.generate(inputs, group_num=100))
