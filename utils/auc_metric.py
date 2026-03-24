# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : auc_metric.py
#   author   : ***
#   date     : 2025-08-27
#   desc     : 
# ======================================================== 

import torch
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Optional
import numpy as np

class AUCMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态，清空所有缓存的预测和标签"""
        self.preds: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets):
        """
        更新当前 batch 的预测概率和标签

        Args:
            preds (torch.Tensor): 模型输出的概率，shape [b]
            targets: 真实标签，可以是 list、numpy array 或 torch.Tensor，shape [b]，值为 0 或 1
        """
        # 处理 targets
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        elif isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        elif isinstance(targets, list):
            targets = torch.tensor(targets)
        else:
            targets = torch.tensor(targets)

        # 处理 preds
        preds = preds.detach().cpu()

        # 存储
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> Optional[float]:
        """
        计算当前所有累积样本的 AUC

        Returns:
            float or None: AUC 值，若无法计算则返回 None
        """
        if len(self.preds) == 0:
            return None

        # 拼接所有 batch
        all_preds = torch.cat(self.preds).numpy()
        all_targets = torch.cat(self.targets).numpy()

        # 检查标签是否只有一类
        if len(set(all_targets)) < 2:
            print("Warning: Only one class present in labels. AUC is not defined.")
            return 0.5  # 或返回 float('nan')

        try:
            auc = roc_auc_score(all_targets, all_preds)
            return float(auc)
        except Exception as e:
            print(f"AUC 计算失败: {e}")
            return None

    def __call__(self, preds: torch.Tensor, targets):
        """方便调用 update"""
        self.update(preds, targets)


def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [ values2 for values1, values2 in sorted(f, key = lambda x: x[0]) ]
    rankList = [i+1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1

    #all pair num
    sum_num = posNum*negNum
    
    #pair num of posi gt neg
    right_num = sum(rankList) - (posNum*(posNum+1))/2
    
    #false pair num
    false_num = sum_num - right_num
    
    if sum_num == 0:
        auc = 0.0
    else:
        auc = float(right_num) / float(sum_num)
    return posNum, negNum, int(right_num), int(false_num), auc
