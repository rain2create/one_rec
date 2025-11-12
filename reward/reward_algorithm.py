# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : reward_algorithm.py
#   author   : ***
#   date     : 2025-08-12
#   desc     : 
# ======================================================== 
import torch

def compute_advantages(rewards, num_samples_per_input, sample_mask, eps=1e-8):
    """
    计算组内优势（intra-group advantage）：每组独立计算均值和标准差

    Args:
        rewards: shape [B * num_samples_per_input]，展平的 reward 向量
        num_samples_per_input: 每个输入生成的样本数 G
        sample_mask: mask解码为空的数据
        eps: 数值稳定性小量

    Returns:
        advantages: shape [B * num_samples_per_input]，组内标准化后的优势
    """
    G = num_samples_per_input
    B = rewards.shape[0] // G
    rewards = rewards.view(B, G)
    mask = sample_mask.view(B, G)

    # 计算每组内的均值和标准差
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # 避免除零
    group_mean = (rewards * mask).sum(dim=1, keepdim=True) / valid_counts
    diff_sq = (rewards - group_mean).pow(2) * mask
    group_std = (diff_sq.sum(dim=1, keepdim=True) / valid_counts).sqrt().clamp(min=eps)

    # 组内标准化：使用每组自己的均值和标准差
    advantages = (rewards - group_mean) / (group_std + eps)  # [B, G]

    # 展平回 [B * G]
    return advantages.flatten()

def compute_decoding_advantage(sample_mask):
    """
    计算基于解码成功与否的二元优势：
        - 解码成功（非空）→ advantage = 1
        - 解码失败（空）   → advantage = 0

    Args:
        sample_mask: [B * G], bool tensor, True 表示解码非空
        device: 输出所在设备

    Returns:
        advantages: [B * G], float tensor, 值为 0 或 1
    """
    return sample_mask.float()  # True -> 1.0, False -> 0.0

def compute_supervised_advantage(algfinish_arr, mask, add_corpus_url):
    """
    algfinish_arr [batch]
    mask [batch]
    """
    algfinish_arr = algfinish_arr.to(mask.device)
    if add_corpus_url:
        supervised_advantages = torch.where(mask, algfinish_arr, torch.tensor(0.0, device=mask.device))
    else:
        supervised_advantages = torch.zeros_like(algfinish_arr)
    return supervised_advantages
