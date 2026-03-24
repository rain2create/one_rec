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

def compute_pt_loss(old_log_probs):
    # old_log_prob: [batch_size, 3]
    pt_loss = -old_log_probs.mean()
    return pt_loss

def compute_gbpo_loss(old_log_prob, new_log_prob, advantage, pos_weight=1.0, neg_weight=1.0, token_wise=True):
    # token-wise
    if token_wise:
        advantage = advantage.unsqueeze(dim=1).repeat_interleave(3, dim=-1)
        pi_theta = torch.exp(new_log_prob) # [batch_size, 3]
        pi_theta_old = torch.exp(old_log_prob) # [batch_size, 3]
    else:
        pi_theta = torch.exp(new_log_prob.sum(-1)) # [batch_size]
        pi_theta_old = torch.exp(old_log_prob.sum(-1)) # [batch_size]
    pi_theta_old = torch.where(advantage >= 0, torch.max(pi_theta_old, pi_theta.detach()), torch.max(pi_theta_old, 1-pi_theta.detach()))
    ratio = pi_theta / pi_theta_old
    loss = -(ratio*advantage)
    weight = torch.where(advantage >= 0, pos_weight, neg_weight)
    final_loss = (loss * weight).sum() / (weight.sum())
    return final_loss 

def compute_sample_loss(old_log_prob, new_log_prob, advantage, token_wise=False, clip_epsilon=0.1, weight=0.3):
    if token_wise:
        advantage = advantage.unsqueeze(dim=-1).repeat_interleave(3, dim=-1)
        pi_theta = torch.exp(new_log_prob) # [batch_size*G, 3]
        pi_theta_old = torch.exp(old_log_prob) # [batch_size*G, 3]
    else:
        pi_theta = torch.exp(new_log_prob.sum(dim=-1)) # [batch_size*G]
        pi_theta_old = torch.exp(old_log_prob.sum(dim=-1)) # [batch_size*G]

    if clip_epsilon > 0:
        ratio = pi_theta / pi_theta_old
        legal_surr1 = ratio * advantage
        legal_surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        legal_loss = -weight * torch.min(legal_surr1, legal_surr2).mean()
    else:
        pi_theta_old = torch.where(advantage >= 0, torch.max(pi_theta_old, pi_theta.detach()), torch.max(pi_theta_old, 1-pi_theta.detach()))
        ratio = pi_theta / pi_theta_old
        legal_loss = -weight * (ratio * advantage).mean()
    return legal_loss

def model_update(model, optimizer_dense, optimizer_sparse, scaler, loss):
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
    data_args.reward_batch_size = 50
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
        total_grpo_loss = 0
        total_gbpo_loss = 0
        total_legal_loss = 0
        total_pos_num = 0
        total_neg_num = 0
        total_pos_prob = 0
        total_neg_prob = 0
        total_ndcg = 0
        total_reward = 0
        total_top3_reward = 0
        total_hitrate = 0
        total_pt_loss = 0
        batch_count = 0
        for x in dataloader:
            if not x:
                continue
            pt_input, reward_input, rl_label = x
            rl_label = rl_label.to(device)
            pt_input = {k: v.to(device) for k, v in pt_input.items()}
            real_batch = reward_input.shape[0]
            url_ids = pt_input["url"][:, 1:]
            
            # beam search采样生成
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    # beam search
                    output_rl = pt_model.bs_rl_generate(pt_input, G)
            output_ids = output_rl['output_ids'].clone()
            output_ids = output_ids.reshape(real_batch, G, 3)
            # [batch_size, G+1, 3]
            all_output_ids = torch.cat([output_ids, url_ids.unsqueeze(dim=1)], dim=1).reshape(-1, 3)
            final_old_log_probs = pt_model.compute_sequence_log_probs(pt_input, all_output_ids, G+1).detach() # [batch_size*(G+1), 3]
            final_old_log_probs = final_old_log_probs.reshape(real_batch, G+1, 3)
            all_old_log_probs = final_old_log_probs[:, :G, :].reshape(-1, 3)
            old_log_probs = final_old_log_probs[:, -1, :]

            # 曝光item的old_log_probs [batch_size, 3]
            # old_log_probs = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
            # old_log_probs = old_log_probs.detach()
            output_ids = output_ids.reshape(-1, 3) 

            # 生成序列解码成url
            output_urls, sample_mask, empty_ratio, _ = url_tokenizer.batch_decode_processed(output_ids, device)
            if not sample_mask.any():
                logger.warning("All samples decoded to empty URLs. Skipping batch.")
                continue
            
            # print(output_urls)
            # print(sample_mask)
            # print(empty_ratio)

            # 扩展reward_input
            expanded_reward_input = np.repeat(reward_input, repeats=G, axis=0)
            expanded_reward_input[:, 2] = output_urls
            reward_input_combined = np.concatenate([expanded_reward_input, reward_input])

            # reward模型获取得分
            rewards = scorer.predict(reward_input_combined)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            rewards = (rewards * reward_weights.unsqueeze(0)).sum(dim=-1)
            corpus_url_rewards = rewards[-real_batch:]
            rewards = rewards[:-real_batch]

            # 计算优势
            advantages = compute_advantages(rewards, G, sample_mask)
            decoding_advantages = compute_decoding_advantage(sample_mask)
            
            # print(advantages)
            # print(decoding_advantages)

            # 预计算pi_theta_old pi_theta_old_token
            url_probs = torch.exp(old_log_probs.sum(dim=1)) # [b]
            url_probs_token = torch.exp(old_log_probs) # [b*3]
            
            pos_num = rl_label.sum()
            neg_num = (rl_label == 0).sum()

            pos_prob = (url_probs[rl_label == 1].sum() if pos_num > 0 else torch.tensor(0.0, device=url_probs.device))
            neg_prob = (url_probs[rl_label == 0].sum() if neg_num > 0 else torch.tensor(0.0, device=url_probs.device))
            
            total_pos_prob += pos_prob
            total_neg_prob += neg_prob
            
            total_pos_num += pos_num
            total_neg_num += neg_num

            unique_count = compute_unique_counts(output_urls, G)
            # 计算NDCG监控指标
            batch_ndcg = compute_ndcg(rewards, all_old_log_probs, G, sample_mask)
            total_ndcg += batch_ndcg
            # 计算AvgReward监控指标
            batch_avgreward, batch_top3_avgreward = compute_avgreward(rewards, all_old_log_probs, G, sample_mask, reward_args.avgreward_topk)
            total_reward += batch_avgreward
            total_top3_reward += batch_top3_avgreward
            # 计算生成的hitrate
            url_ids = pt_input["url"][:,1:]
            batch_hitrate = compute_hitrate(output_ids, url_ids, G, sample_mask)
            total_hitrate += batch_hitrate
            # 计算语料中url资源的在policy模型的auc
            task_auc_metric(url_probs, rl_label)
            # 计算语料中url资源的在reward模型的auc 
            reward_auc_metric(corpus_url_rewards, rl_label)
            update_count = 0
            # 决定当前数据是否进行rl的学习
            rl_stage = True if random.random() < 0.05 else False
            steps = reward_args.grpo_epochs_per_batch if rl_stage else 1
            for _ in range(steps):
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    # all_new_log_probs = pt_model.compute_sequence_log_probs(pt_input, output_ids, G)
                    # new_log_probs = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
                    final_new_log_probs = pt_model.compute_sequence_log_probs(pt_input, all_output_ids, G+1) # [batch_size*G, 3]
                    final_new_log_probs = final_new_log_probs.reshape(real_batch, G+1, 3)
                    all_new_log_probs = final_new_log_probs[:, :G, :].reshape(-1, 3)
                    new_log_probs = final_new_log_probs[:, -1, :]

                # 计算GBPO loss
                gbpo_advantage = torch.where(rl_label == 0, -1., 1.)
                gbpo_loss = compute_sample_loss(old_log_probs, new_log_probs, gbpo_advantage, token_wise=True, clip_epsilon=0.2, weight=1.0)
                # legal_loss = compute_sample_loss(all_old_log_probs, all_new_log_probs, decoding_advantages, token_wise=True, clip_epsilon=0.0, weight=1.0)
                legal_loss = 0.
                grpo_loss = compute_sample_loss(all_old_log_probs, all_new_log_probs, advantages, token_wise=True, clip_epsilon=0.2, weight=1.0)
                # loss = grpo_loss + 0.1 * legal_loss + 1.0 * gbpo_loss if rl_stage else 0
                loss = grpo_loss + 0.1 * gbpo_loss if rl_stage else 0
                if update_count == 0:
                    pt_loss = compute_pt_loss(new_log_probs)
                    loss += pt_loss
                stats = model_update(pt_model, optimizer_dense, optimizer_sparse, scaler, loss)
                update_count += 1
                if update_count == steps:
                    # 最后step的loss
                    total_grpo_loss += grpo_loss
                    total_gbpo_loss += gbpo_loss
                    total_legal_loss += legal_loss
                    total_pt_loss += pt_loss

            # 训练后再推理一次概率, 计算afer auc
            with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                new_log_probs = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1)
                new_log_probs = new_log_probs.detach()
            after_gbpo_pi_theta = torch.exp(new_log_probs.sum(dim=1))
            new_url_probs = new_log_probs.detach().sum(-1).exp()
            task_after_auc_metric(new_url_probs, rl_label)
            batch_count += 1

            # 定期保存检查点
            if batch_count % reward_args.reward_save_steps == 0:
                save_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, batch_count, train_args, logger, rl=True)

            # 打印批次信息（每10个batch打印一次）
            if batch_count % 10 == 0:
                batch_duration = time.time() - batch_start_time
                batch_start_time = time.time()  # 重置为下一个 batch 开始时间
                avg_grpo_loss = total_grpo_loss / batch_count
                avg_gbpo_loss = total_gbpo_loss / batch_count
                avg_legal_loss = total_legal_loss / batch_count
                avg_pos_num = total_pos_num / batch_count
                avg_neg_num = total_neg_num / batch_count
                avg_pos_prob = total_pos_prob / total_pos_num
                avg_neg_prob = total_neg_prob / total_neg_num
                avg_ndcg = total_ndcg / batch_count
                avg_reward = total_reward / batch_count
                avg_top3_reward = total_top3_reward / batch_count
                avg_hitrate = total_hitrate / batch_count
                avg_pt_loss = total_pt_loss / batch_count
                print(f"Time={batch_duration:.2f}s "
                      f"Batch {batch_count}: "
                      f"grpo_Loss={avg_grpo_loss:.4f}, "
                      f"gbpo_Loss={avg_gbpo_loss:.4f}, "
                      f"pt_Loss={avg_pt_loss:.4f}, "
                      f"legal_Loss={avg_legal_loss:.4f}, " 
                      f"pos_num={avg_pos_num:.4f}, "
                      f"neg_num={avg_neg_num:.4f}, "
                      f"AvgReward@{reward_args.avgreward_topk}={avg_reward:.4f}, "
                      f"AvgReward@3={avg_top3_reward:.4f}, "
                      # f"NDCG={avg_ndcg:.4f}, "
                      f"Hitrate@{G}={avg_hitrate:.4f}, "
                      f"EmptyDecodingRatio={empty_ratio:.4f}, "
                      f"Unique={unique_count:.2f}, "
                      f"pos_prob={avg_pos_prob:.5f}, "
                      f"neg_prob={avg_neg_prob:.5f}, "
                      f"LikeAUC={task_auc_metric.compute():.4f}, "
                      f"LikeAfterAUC={task_after_auc_metric.compute():.4f}, "
                      f"RewardLikeAUC={reward_auc_metric.compute():.4f}")
        break

        # Epoch结束时打印汇总
        if batch_count > 0:
            total_duration = time.time() - total_start_time
            print(f"Epoch {epoch} Summary: "
                  f"Time={total_duration:.2f}s "
                  f"TotalBatches={batch_count}")

