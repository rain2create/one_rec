 # coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : train_reward2.py
#   author   : ***
#   date     : 2025-07-28
#   desc     : 
# ======================================================== 
import os, sys
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from torch.optim import AdamW
from torch.amp import GradScaler
import numpy as np

from utils.train_utils import unwrap_module
from utils.logger import logger
from utils.auc_metric import AUCMetric

from arguments.data_args import DataArguments
from arguments.model_args import ModelArguments
from arguments.train_args import TrainingArguments
from arguments.reward_args import RewardArguments

from train_pt import build_optimizers, load_checkpoint, save_checkpoint
from parser.tokenizer import UrlTokenizer
from parser.rl_parser import RLParser
from dataset.pt_dataset import PtDataset
from model.rec_model import RecModel

from reward.reward_scorer_online import RewardScorerOnline
from reward.reward_algorithm import compute_advantages, compute_decoding_advantage, compute_supervised_advantage
from reward.reward_evaluate import compute_ndcg, compute_avgreward, compute_unique_counts, compute_hitrate

torch.set_printoptions(
    sci_mode=False,  # 关闭科学计数法
    precision=4      # 保留 4 位小数
)

def weighted_bce(probs, targets, pos_weight=1.0, neg_weight=1.0, reduction="mean"):
    """
    Binary Cross-Entropy with positive/negative sample weighting,
    and returns both positive and negative losses for monitoring.

    Args:
        probs (Tensor): 已经是概率 (0~1)，shape = (N, ...)
        targets (Tensor): 标签 (0 或 1)，shape 与 probs 相同
        pos_weight (float): 正样本权重
        neg_weight (float): 负样本权重
        reduction (str): "mean" | "sum" | "none"

    Returns:
        total_loss: 总 loss
        pos_loss: 正样本 loss（未乘正样本权重前，已包含log）
        neg_loss: 负样本 loss（未乘负样本权重前，已包含log）
    """
    pos_loss = - targets * torch.log(probs + 1e-8)
    neg_loss = - (1 - targets) * torch.log(1 - probs + 1e-8)

    # 加权后的总 loss
    loss = pos_weight * pos_loss + neg_weight * neg_loss

    num_pos = targets.sum()
    num_neg = (targets == 0).sum()

    if reduction == "mean":
        total_loss = loss.mean()
        pos_loss_val = (pos_loss.sum() / num_pos) if num_pos > 0 else torch.tensor(0.0, device=probs.device)
        neg_loss_val = (neg_loss.sum() / num_neg) if num_neg > 0 else torch.tensor(0.0, device=probs.device)
    elif reduction == "sum":
        total_loss = loss.sum()
        pos_loss_val = pos_loss.sum()
        neg_loss_val = neg_loss.sum()
    else:  # "none"
        total_loss = loss
        pos_loss_val = pos_loss
        neg_loss_val = neg_loss
    
    pos_prob = (probs[targets == 1].mean() if num_pos > 0 else torch.tensor(0.0, device=probs.device))
    neg_prob = (probs[targets == 0].mean() if num_neg > 0 else torch.tensor(0.0, device=probs.device))
    return total_loss, pos_loss_val, neg_loss_val, num_pos, num_neg, pos_prob, neg_prob



def gbpo_update(model, optimizer_dense, optimizer_sparse, scaler, probs, rl_label, clip_epsilon=0.1):
    print(probs)
    probs = torch.sigmoid(probs)
    print(probs)
    loss, pos_loss_val, neg_loss_val, num_pos, num_neg, pos_prob, neg_prob = weighted_bce(probs, rl_label, pos_weight=100.)

    # 参数更新
    optimizer_dense.zero_grad()
    if optimizer_sparse is not None:
        optimizer_sparse.zero_grad()

    # 使用GradScaler进行反向传播
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_dense)
    else:
        loss.backward()
    
    """
    # 手动梯度裁剪
    parameters = [p for p in model.parameters() if p.grad is not None and not p.grad.is_sparse]
    if len(parameters) > 0:
        total_norm = torch.sqrt(sum([torch.sum(p.grad.data ** 2) for p in parameters]))
        max_norm = 1.0
        clip_coef = max_norm / (total_norm + 1e-6)  # 防止除零
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    """

    # 优化器更新
    if scaler is not None:
        if optimizer_sparse is not None:
            optimizer_sparse.step()
        scaler.step(optimizer_dense)
        scaler.update()
    else:
        optimizer_dense.step()
        if optimizer_sparse is not None:
            optimizer_sparse.step()

    return {
            'loss': loss.item(),
            'pos_loss': pos_loss_val.item(),
            'neg_loss': neg_loss_val.item(),
            'pos_num': num_pos.item(),
            'neg_num': num_neg.item(),
            'pos_prob': pos_prob.item(),
            'neg_prob': neg_prob.item()
        }


if __name__ == "__main__":
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RewardArguments))
    model_args, data_args, train_args, reward_args = parser.parse_args_into_dataclasses()
    #data_args.file_shuffle = False
    #data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 100
    # train_args.learning_rate /= 10
    # train_args.lr_sparse /= 10
    # model_args.encoder_dropout = 0.
    # model_args.decoder_dropout = 0.

    like_auc_metric = AUCMetric()
    reward_auc_metric = AUCMetric()

    # 加载预训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt_model = RecModel.from_args(model_args)
    pt_model = pt_model.to(device)
    optimizer_dense, optimizer_sparse = build_optimizers(pt_model, train_args)
    scaler = None
    # scaler = GradScaler('cuda', enabled=train_args.use_amp)
    ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-272030.pt")
    load_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, ckpt_path, logger)
    pt_model.train()

    # 初始化其他组件
    url_tokenizer = UrlTokenizer.from_args(data_args)
    rl_parser = RLParser(data_args.columns_def, data_args.reward_serving_column_idx, tokenizer=url_tokenizer)
    dataset = PtDataset.from_args(data_args, parser_func = rl_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # reward model
    scorer = RewardScorerOnline.from_args(reward_args)
    # 训练过程进行固定权重的多目标拟合
    #right_click_pred, dt_pred, finish_pred, like_pred, zmtlink_pred, follow_pred, dislike_pred, skip_pred, last_pred
    reward_weights = torch.tensor([1., 0., 0., 1., 1., 1., 0., 0., 0.], device=device)
    
    # 强化学习训练循环
    for epoch in range(reward_args.reward_train_epochs):
        total_start_time = time.time()
        batch_start_time = time.time()
        total_loss = 0
        batch_count = 0
        total_pos_num = 0
        total_neg_num = 0
        total_pos_loss = 0
        total_neg_loss = 0
        total_pos_prob = 0
        total_neg_prob = 0
        for x in dataloader:
            if not x:
                continue
            pt_input, reward_input, rl_label = x
            rl_label = rl_label.to(device)
            # rl_label = rl_label.to(device).unsqueeze(dim=1)
            pt_input = {k: v.to(device) for k, v in pt_input.items()}
            real_batch = reward_input.shape[0]
            url_ids = pt_input["url"][:, 1:]
            log_probs, target_logits = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
            
            # url_probs = log_probs.sum(-1).exp()
            # [batch_size, 3]
            url_logits = target_logits.sum(-1)
            # url_probs = log_probs.exp().reshape(-1)
            # rl_label = rl_label.repeat_interleave(3, dim=-1).reshape(-1) 
            
            # reward 模型获取得分
            rewards = scorer.predict(reward_input)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            rewards = (rewards * reward_weights.unsqueeze(0)).sum(dim=-1)# .unsqueeze(1)
            # rewards = rewards.repeat_interleave(3, dim=-1).reshape(-1)
            
            stats = gbpo_update(pt_model, optimizer_dense, optimizer_sparse, scaler, url_logits, rl_label, clip_epsilon=0.1)
            total_loss += stats["loss"]
            total_pos_num += stats["pos_num"]
            total_neg_num += stats["neg_num"]
            total_pos_loss += stats["pos_loss"]
            total_neg_loss += stats["neg_loss"]
            total_pos_prob += stats["pos_prob"]
            total_neg_prob += stats["neg_prob"]
            
            # 计算like auc
            like_auc_metric(url_probs, rl_label)
            reward_auc_metric(rewards, rl_label)
            batch_count += 1

            # 定期保存检查点
            if batch_count % reward_args.reward_save_steps == 0:
                save_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, batch_count, train_args, logger)

            # 打印批次信息（每10个batch打印一次）
            if batch_count % 10 == 0:
                batch_duration = time.time() - batch_start_time
                batch_start_time = time.time()  # 重置为下一个 batch 开始时间
                avg_loss = total_loss / batch_count
                avg_pos_num = total_pos_num / batch_count
                avg_neg_num = total_neg_num / batch_count
                avg_pos_loss = total_pos_loss / batch_count
                avg_neg_loss = total_neg_loss / batch_count
                avg_pos_prob = total_pos_prob / batch_count
                avg_neg_prob = total_neg_prob / batch_count


                print(f"Time={batch_duration:.2f}s "
                      f"Batch {batch_count}: "
                      f"pos_num={avg_pos_num:.4f}, "
                      f"neg_num={avg_neg_num:.4f}, "
                      f"pos_loss={avg_pos_loss:.4f}, "
                      f"neg_loss={avg_neg_loss:.4f}, "
                      f"pos_prob={avg_pos_prob:.4f}, "
                      f"neg_prob={avg_neg_prob:.4f}, "
                      f"Loss={avg_loss:.4f}, "
                      f"LikeAUC={like_auc_metric.compute():.4f}, "
                      f"RewardLikeAUC={reward_auc_metric.compute():.4f}")
            
            if batch_count % 1000 == 0:
                pt_model.eval()
                scripted_model = torch.jit.script(pt_model)
                scripted_model.save(os.path.join(train_args.serving_output_dir, "recmodel.pt"))
                pt_model.train()

        # Epoch结束时打印汇总
        if batch_count > 0:
            total_duration = time.time() - total_start_time
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch} Summary: "
                  f"Time={total_duration:.2f}s "
                  f"Loss={avg_loss:.4f}, "
                  f"TotalBatches={batch_count}")
            pt_model.eval()
            scripted_model = torch.jit.script(pt_model)
            scripted_model.save(os.path.join(train_args.serving_output_dir, "recmodel.pt"))
            pt_model.train()

