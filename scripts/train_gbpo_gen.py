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
from utils.auc_metric import AUCMetric, calAUC

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

def compute_gbpo_loss(old_log_prob, new_log_prob, advantage, pos_weight=1.0, neg_weight=1.0, token_wise=True):
    if token_wise:
        # [batch_size, 3]
        advantage = advantage.unsqueeze(dim=1).repeat_interleave(3, dim=-1)
        # token-wise
        pi_theta = torch.exp(new_log_prob) # [batch_size, 3]
        pi_theta_old = torch.exp(old_log_prob) # [batch_size, 3]
    else:
        pi_theta = torch.exp(new_log_prob.sum(dim=-1)) # [batch_size]
        pi_theta_old = torch.exp(old_log_prob.sum(dim=-1)) # [batch_size]
    pi_theta_old = torch.where(advantage >= 0, torch.max(pi_theta_old, pi_theta.detach()), torch.max(pi_theta_old, 1-pi_theta.detach()))
    ratio = pi_theta / pi_theta_old
    loss = -(ratio*advantage)
    weight = torch.where(advantage >= 0, pos_weight, neg_weight)
    final_loss = (loss * weight).sum() / (weight.sum()+1e-8)
    return final_loss 



def gbpo_update(model, optimizer_dense, optimizer_sparse, scaler, loss):
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
    
    # 手动梯度裁剪
    parameters = [p for p in model.parameters() if p.grad is not None and not p.grad.is_sparse]
    if len(parameters) > 0:
        total_norm = torch.sqrt(sum([torch.sum(p.grad.data ** 2) for p in parameters]))
        max_norm = 1.0
        clip_coef = max_norm / (total_norm + 1e-6)  # 防止除零
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

    
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
            'loss': loss.item()
        }


if __name__ == "__main__":
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RewardArguments))
    model_args, data_args, train_args, reward_args = parser.parse_args_into_dataclasses()
    data_args.file_shuffle = False
    data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 200
    train_args.learning_rate /= 100
    train_args.lr_sparse /= 10000
    model_args.encoder_dropout = 0.
    model_args.decoder_dropout = 0.
    G = reward_args.num_samples_per_input

    task_auc_metric = AUCMetric()
    task_after_auc_metric = AUCMetric()
    reward_auc_metric = AUCMetric()

    # 加载预训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt_model = RecModel.from_args(model_args)
    pt_model = pt_model.to(device)
    optimizer_dense, optimizer_sparse = build_optimizers(pt_model, train_args)
    scaler = GradScaler('cuda', enabled=train_args.use_amp)
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
        total_pos_num = 0
        total_neg_num = 0
        total_pos_prob = 0
        total_neg_prob = 0
        total_batch_auc = 0
        batch_count = 0
        for x in dataloader:
            if not x:
                continue
            pt_input, reward_input, rl_label = x
            rl_label = rl_label.to(device)
            pt_input = {k: v.to(device) for k, v in pt_input.items()}
            real_batch = reward_input.shape[0]
            url_ids = pt_input["url"][:, 1:]
            
            # beam-search采样
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    # beam search
                    output_rl = pt_model.bs_rl_generate(pt_input, G)
            output_ids = output_rl['output_ids'].clone()
            
            bs_old_log_probs, _ = pt_model.compute_sequence_log_probs(pt_input, output_ids, G)
            # 解码生成url
            output_urls, sample_mask, empty_ratio, per_batch_non_empty_counts = url_tokenizer.batch_decode_processed(output_ids, device)
            if not sample_mask.any():
                logger.warning("All samples decoded to empty URLs. Skipping batch.")
                continue


            # old prob
            old_log_probs, _ = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
            old_log_probs = old_log_probs.detach()
            url_probs = old_log_probs.sum(-1).exp()
            
            pos_num = rl_label.sum()
            neg_num = (rl_label == 0).sum()

            pos_prob = (url_probs[rl_label == 1].sum() if pos_num > 0 else torch.tensor(0.0, device=url_probs.device))
            neg_prob = (url_probs[rl_label == 0].sum() if neg_num > 0 else torch.tensor(0.0, device=url_probs.device))
            
            total_pos_prob += pos_prob
            total_neg_prob += neg_prob
            
            total_pos_num += pos_num
            total_neg_num += neg_num

            # reward 模型获取得分
            rewards = scorer.predict(reward_input)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            rewards = (rewards * reward_weights.unsqueeze(0)).sum(dim=-1)
            
            update_count = 0
            for _ in range(reward_args.grpo_epochs_per_batch):
                # update 
                new_log_probs, _ = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
                # 计算GBPO
                advantage = torch.where(rl_label == 0, -1., 1.)
                loss = compute_gbpo_loss(old_log_probs, new_log_probs, advantage, pos_weight=1.0, neg_weight=1.0)
                stats = gbpo_update(pt_model, optimizer_dense, optimizer_sparse, scaler, loss)
                update_count += 1
                if update_count == reward_args.grpo_epochs_per_batch:
                    # 最后step的loss
                    total_loss += stats['loss']


            # 计算like auc
            new_url_probs = new_log_probs.detach().sum(-1).exp()
            task_auc_metric(url_probs, rl_label)
            task_after_auc_metric(new_url_probs, rl_label)
            reward_auc_metric(rewards, rl_label)
            batch_count += 1
            _, _, _, _, batch_auc = calAUC(url_probs.tolist(), rl_label.tolist())
            total_batch_auc += batch_auc

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
                avg_pos_prob = total_pos_prob / total_pos_num
                avg_neg_prob = total_neg_prob / total_neg_num
                avg_batch_auc = total_batch_auc / batch_count 

                print(f"Time={batch_duration:.2f}s "
                      f"Batch {batch_count}: "
                      f"pos_num={avg_pos_num:.4f}, "
                      f"neg_num={avg_neg_num:.4f}, "
                      f"pos_prob={avg_pos_prob:.5f}, "
                      f"neg_prob={avg_neg_prob:.5f}, "
                      f"Loss={avg_loss:.4f}, "
                      f"LikeAUC={task_auc_metric.compute():.4f}, "
                      f"LikeAfterAUC={task_after_auc_metric.compute():.4f}, "
                      f"RewardLikeAUC={reward_auc_metric.compute():.4f}")
            
        # Epoch结束时打印汇总
        if batch_count > 0:
            total_duration = time.time() - total_start_time
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch} Summary: "
                  f"Time={total_duration:.2f}s "
                  f"Loss={avg_loss:.4f}, "
                  f"TotalBatches={batch_count}")

