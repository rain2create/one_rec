# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : reward_utils.py
#   author   : ***
#   date     : 2025-08-12
#   desc     : 
# ======================================================== 
import torch
import numpy as np

def compute_ndcg(rewards, log_probs, G, sample_mask):
    """
    计算每个输入的 NDCG@G，并返回在 batch 上的平均值。

    Args:
        rewards: [B, G] 或 [B * G] - 真实相关性得分（reward）
        log_probs: [B * G, codebook_layers] - 模型输出的 log 概率
        G: G - 每个输入生成的样本数
        sample_mask: [B, G] 或 [B * G] - 有效样本 mask

    Returns:
        avg_ndcg: Python float, 所有 batch 的平均 NDCG（只考虑有效样本）
    """
    # Step 1: reshape 所有输入为 [B, G]
    B = rewards.shape[0] // G
    rewards = rewards.view(B, G)
    scores = log_probs.sum(dim=-1).view(B, G)  # 使用 log_prob 总和作为预测得分
    mask = sample_mask.view(B, G).bool()

    # Step 2: 预计算折扣因子 1 / log2(i+1), i 从 1 开始
    discounts = 1.0 / torch.log2(torch.arange(2, G + 2, device=rewards.device).float())  # [G]

    total_ndcg = 0.0
    valid_batches = 0

    for b in range(B):
        # 提取当前 batch 的有效样本
        valid_idx = mask[b]  # [G]
        n_valid = valid_idx.sum().item()
        if n_valid == 0:
            continue  # 跳过无有效样本的 batch

        # 获取有效的真实得分和预测得分
        r_valid = rewards[b][valid_idx]     # [n_valid]
        s_valid = scores[b][valid_idx]      # [n_valid]

        # 根据预测得分排序（降序），得到排序后的相关性得分
        _, pred_order = torch.sort(s_valid, descending=True)
        dcg_relevance = r_valid[pred_order]  # 按预测排序后的真实得分

        # 计算 DCG: sum( r_i / log2(rank+1) )
        dcg = ((10**dcg_relevance-1) * discounts[:n_valid]).sum()

        # 计算 IDCG: 将真实得分按理想排序
        _, ideal_order = torch.sort(r_valid, descending=True)
        idcg_relevance = r_valid[ideal_order]
        idcg = ((10**idcg_relevance-1) * discounts[:n_valid]).sum()

        # 避免除零
        if idcg > 1e-8:
            ndcg = dcg / idcg
            total_ndcg += ndcg.item()
            valid_batches += 1
        # 否则：IDCG=0（所有相关性为0），可跳过或视为 NDCG=1（见下文说明）

    # 返回平均 NDCG
    return total_ndcg / valid_batches if valid_batches > 0 else 0.0

def compute_avgreward(rewards, log_probs, G, sample_mask, topk):
    """
    计算每个输入的 AvgReward@k，并返回在 batch 上的平均值。

    Args:
        rewards: [B * G] - 真实相关性得分（reward）
        log_probs: [B * G, 3] - 模型生成的概率得分
        G: G - 每个输入生成的样本数
        sample_mask: [B, G] 或 [B * G] - 有效样本 mask
        topk: int - 每行取top k的reward

    Returns:
        avg_reward: Python float, 所有 batch 的top k的平均 reward
    """
    B = rewards.shape[0] // G
    probs = torch.exp(log_probs.sum(dim=1))  # [B * G]
    rewards = rewards.view(B, G)
    probs = probs.view(B, G)  # 将概率也转换为 [B, G] 形状
    mask = sample_mask.view(B, G).bool()
    row_means = []
    row_top3_means = []

    for i in range(B):
        # 取出当前行中 mask 为 True 的 rewards 和 probs
        valid_mask = mask[i]
        valid_rewards = rewards[i][valid_mask]  # [n_valid]
        valid_probs = probs[i][valid_mask]      # [n_valid]
        
        n_valid = valid_rewards.numel()
        
        if n_valid == 0:
            row_mean = torch.tensor(0.0, device=rewards.device)
            row_top3_mean = torch.tensor(0.0, device=rewards.device)
        else:
            # 根据概率排序，获取索引
            _, sorted_indices = torch.topk(valid_probs, k=n_valid, largest=True)
            # 取 top-k，如果不足 k 个就取全部
            k = min(topk, n_valid)
            topk_values = valid_rewards[sorted_indices[:k]]
            k2 = min(3, n_valid)
            top3_values = valid_rewards[sorted_indices[:k2]]
            row_mean = topk_values.mean()
            row_top3_mean = top3_values.mean()

        row_means.append(row_mean)
        row_top3_means.append(row_top3_mean)

    # 拼成 [B] 的张量
    row_means = torch.stack(row_means)  # shape: [B]
    row_top3_means = torch.stack(row_top3_means)  # shape: [B]

    # 最终对 B 维度求均值
    final_mean = row_means.mean()
    final_mean_top3 = row_top3_means.mean()
    return final_mean, final_mean_top3

def compute_unique_counts(output_urls, G):
    # 计算去重后答案数
    reshaped_urls = output_urls.reshape(-1, G)
    unique_counts = []
    unique_counts_all = 0
    for row in reshaped_urls:
        unique_count = len(np.unique(row))  # np.unique 自动去重
        unique_counts.append(unique_count)
        unique_counts_all += unique_count
    unique_count = unique_counts_all/len(unique_counts)
    return unique_count

def compute_hitrate(output_ids, url_ids, G, sample_mask):
    # output_ids [B*G, 3]
    # url_ids [B, 3]
    B = output_ids.shape[0] // G
    output_urls_3d = output_ids.reshape(B, G, 3)
    cand_urls_3d = url_ids.reshape(B, 1, 3)
    mask = sample_mask.view(B, G).bool().to(output_ids.device)
    batch_matches = []
    for b in range(B):
        # 提取当前 batch 的有效样本
        valid_idx = mask[b]  # [G]
        n_valid = valid_idx.sum().item()
        if n_valid == 0:
            continue
        matches = (output_urls_3d[b] == cand_urls_3d[b])
        matches = matches.all(dim=-1)[valid_idx]
        hit = matches.any().item()
        batch_matches.append(hit)
    result = np.array(batch_matches).mean()
    return result
