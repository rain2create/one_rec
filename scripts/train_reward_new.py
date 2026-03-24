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
    precision=8      # 保留 4 位小数
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

def compute_gbpo_loss(pi_theta_old_token, new_log_probs, supervised_advantages, valid_idx, G, clip_epsilon=0.1):
    """
    pi_theta_old_token [B*3]
    new_log_probs [B*G, 3]
    supervised_advantages, valid_idx [B]
    """
    '''
    # seqence
    pi_theta_old = torch.exp(old_log_probs.sum(dim=1)).view(-1, G)[:, -1]
    pi_theta = torch.exp(new_log_probs.sum(dim=1)).view(-1, G)[:, -1] # [B]
    '''
    # token 
    pi_theta = torch.exp(new_log_probs.view(-1, G, 3)[:, -1, :]).view(-1)
    supervised_advantages = supervised_advantages.unsqueeze(1).repeat(1, 3).view(-1)
    valid_idx = valid_idx.unsqueeze(1).repeat(1, 3).view(-1)

    pi_theta_old_ = torch.max(torch.where(supervised_advantages >= 0, pi_theta.detach(), 1 - pi_theta.detach()), pi_theta_old_token) 
    #pi_theta_old_ = torch.max(pi_theta.detach(), pi_theta_old_token) / torch.where(supervised_advantages >= 0, 10., 1.)
    ratio = pi_theta/pi_theta_old_

    #print("======gbpo_update========")
    #print("pi_theta:", pi_theta[valid_idx])
    #print("pi_theta_old_:", pi_theta_old_[valid_idx])
    #print("ratio:", ratio[valid_idx])
    supervised_loss = -(ratio * supervised_advantages)[valid_idx].mean()
    return supervised_loss

def grpo_update(model, optimizer_dense, optimizer_sparse, scaler, old_log_probs, new_log_probs, advantages, decoding_advantages, supervised_loss, sample_mask, clip_epsilon=0.1):
    """
    GRPO算法核心更新逻辑
    old_log_probs, new_log_probs [B*G, 3]
    advantages [B*G]
    decoding_advantages [B*G]
    """
    # 只对有效样本计算 ratio 和 loss
    valid_idx = sample_mask.nonzero(as_tuple=True)[0]

    # GRPO loss with clipping
    #ratio = torch.exp(new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))
    #surr1 = ratio * advantages
    #surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    #policy_loss = -torch.min(surr1, surr2)[valid_idx].mean()

    #print("======grpo_update========")
    #print("new_log_probs:", torch.exp(new_log_probs.sum(dim=1))[valid_idx])
    #print("ratio:", ratio[valid_idx])

    # GBPO loss
    pi_theta_old = torch.exp(old_log_probs.sum(dim=1))
    pi_theta = torch.exp(new_log_probs.sum(dim=1)) # [B*G]
    pi_theta_old_ = torch.max(torch.where(advantages>= 0, pi_theta.detach(), 1 - pi_theta.detach()), pi_theta_old) 
    ratio = pi_theta/pi_theta_old_
    policy_loss = -(ratio * advantages)[valid_idx].mean()
 
    # 熵奖励 loss
    entropy = - (new_log_probs.exp() * new_log_probs).sum(dim=1)[valid_idx].mean()
    entropy_loss = -0.0 * entropy

    # 合法非法loss
    legal_surr1 = ratio * decoding_advantages
    legal_surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * decoding_advantages
    legal_loss = -0.1 * torch.min(legal_surr1, legal_surr2).mean()

    # 计算KL散度: KL(new||old) = Σ p_new * log(p_new / p_old)
    kl_div = 0. * (new_log_probs.exp() * (new_log_probs - old_log_probs)).sum(dim=1)[valid_idx].mean()
    #loss = policy_loss + entropy_loss + legal_loss + kl_div + supervised_loss
    loss = supervised_loss + legal_loss

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

    valid_ratio = ratio[valid_idx]
    return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'ratio_mean': valid_ratio.mean().item(),
            'clip_fraction': ((valid_ratio- 1.0).abs() > clip_epsilon).float().mean().item(),
        }

if __name__ == "__main__":
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RewardArguments))
    model_args, data_args, train_args, reward_args = parser.parse_args_into_dataclasses()
    data_args.file_shuffle = False
    data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 200
    train_args.learning_rate /= 1000
    train_args.lr_sparse /= 100000
    model_args.encoder_dropout = 0.
    model_args.decoder_dropout = 0.
    G = reward_args.num_samples_per_input

    task_auc_metric = AUCMetric()
    task_after_auc_metric = AUCMetric()
    reward_auc_metric = AUCMetric()

    # 加载预训练模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pt_model = RecModel.from_args(model_args)
    pt_model = pt_model.to(device)
    optimizer_dense, optimizer_sparse = build_optimizers(pt_model, train_args)
    scaler = GradScaler('cuda', enabled=train_args.use_amp)
    #ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-237757.pt") # pretrain
    #ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-250117.pt") # sft
    ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-280000.pt") # pretrain dim=256
    load_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, ckpt_path, logger)
    pt_model.train()

    # 初始化其他组件
    url_tokenizer = UrlTokenizer.from_args(data_args)
    reward_parser = RewardParser(data_args.columns_def, data_args.reward_serving_column_idx, tokenizer=url_tokenizer, logreallike_len=reward_args.logreallike_len)
    dataset = PtDataset.from_args(data_args, parser_func = reward_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # reward model
    scorer = RewardScorerOnline.from_args(reward_args)
    # 训练过程进行固定权重的多目标拟合
    #right_click_pred, dt_pred, finish_pred, like_pred, zmtlink_pred, follow_pred, dislike_pred, skip_pred, last_pred
    reward_weights = torch.tensor([0., 1., 1.1, 20., 1., 2., 1., 0., 1.], device=device)
    #reward_weights = torch.tensor([0., 0., 0., 100., 0., 0., 0., 0., 0.], device=device)
    os.makedirs(train_args.serving_output_dir, exist_ok=True)    

    # 强化学习训练循环
    for epoch in range(reward_args.reward_train_epochs):
        total_start_time = time.time()
        batch_start_time = time.time()
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl = 0
        total_ndcg = 0
        total_reward = 0
        total_top3_reward = 0
        total_hitrate = 0
        total_gbpo_t_right = 0
        total_gbpo_f_right = 0
        batch_count = 0

        for x in dataloader:
            if not x:
                continue
            pt_input, reward_input, url_label, supervised_advantages = x
            pt_input = {k: v.to(device) for k, v in pt_input.items()}
            url_label = url_label.to(device)
            supervised_advantages = supervised_advantages.to(device)
            real_batch = reward_input.shape[0]

            # 生成多个输出并获取log概率
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    # beam search
                    output_rl = pt_model.bs_rl_generate(pt_input, G)
                    # other sample methods
                    #output_rl = pt_model.rl_generate(pt_input, G, top_p = reward_args.top_p, strategy = reward_args.strategy)
            output_ids = output_rl['output_ids'].clone()

            # 将最后一个生成的item替代为语料曝光item
            output_ids = output_ids.view(real_batch, G, 3)
            url = pt_input["url"][:,1:]
            add_corpus_url_mask = (url == data_args.unk_token_id) # [b ,3]
            valid_idx = ~add_corpus_url_mask[:,0] # [b] 语料曝光item能够反解码出的mask标识
            output_ids[:, -1, :] = torch.where(add_corpus_url_mask, output_ids[:, -1, :], url)
            output_ids = output_ids.view(-1, 3)
            old_log_probs = pt_model.compute_sequence_log_probs(pt_input, output_ids, G).detach()

            # 生成序列解码成url
            output_urls, sample_mask, empty_ratio, per_batch_non_empty_counts = url_tokenizer.batch_decode_processed(output_ids, device)
            if not sample_mask.any():
                logger.warning("All samples decoded to empty URLs. Skipping batch.")
                continue

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

            # 预计算pi_theta_old pi_theta_old_token
            pi_theta_old = torch.exp(old_log_probs.sum(dim=1)).view(-1, G)[:, -1] # [b] 
            pi_theta_old_token = torch.exp(old_log_probs.view(-1, G, 3)[:, -1, :]).view(-1) # [b*3]
            
            # 监控指标
            # 计算去重后答案数
            unique_count = compute_unique_counts(output_urls, G)
            # 计算NDCG监控指标
            batch_ndcg = compute_ndcg(rewards, old_log_probs, G, sample_mask)
            total_ndcg += batch_ndcg
            # 计算AvgReward监控指标
            batch_avgreward, batch_top3_avgreward = compute_avgreward(rewards, old_log_probs, G, sample_mask, reward_args.avgreward_topk)
            total_reward += batch_avgreward
            total_top3_reward += batch_top3_avgreward
            # 计算生成的hitrate
            url_ids = pt_input["url"][:,1:]
            batch_hitrate = compute_hitrate(output_ids, url_ids, G, sample_mask)
            total_hitrate += batch_hitrate
            # 计算语料中url资源的在policy模型的auc
            url_log_probs = pt_model.compute_sequence_log_probs(pt_input, url_ids, 1).detach()
            url_probs = url_log_probs.sum(-1).exp()
            task_auc_metric(url_probs[valid_idx], url_label[valid_idx])
            # 计算语料中url资源的在reward模型的auc
            reward_auc_metric(corpus_url_rewards, url_label)

            #grpo_valid_idx = sample_mask.nonzero(as_tuple=True)[0]
            #print("output_ids: ", output_ids[grpo_valid_idx])
            #print("advantages:", advantages[grpo_valid_idx])
            ##print("decoding_advantages:", decoding_advantages)
            #print("old_log_probs:", torch.exp(old_log_probs.sum(dim=1))[grpo_valid_idx])
            #print("advantages/old_log_probs:", torch.exp(-old_log_probs.sum(dim=1))[grpo_valid_idx] * advantages[grpo_valid_idx])

            # GBPO debug
            #print("url: ", url[valid_idx])
            #print("supervised_advantages:", supervised_advantages[valid_idx])
            #print("before gbpo:", pi_theta_old[valid_idx])

            # 数据多轮更新
            update_count = 0
            #reward_args.grpo_epochs_per_batch = 0
            for _ in range(reward_args.grpo_epochs_per_batch):
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    new_log_probs = pt_model.compute_sequence_log_probs(
                        pt_input,
                        output_ids,
                        G
                    )
                # 计算GBPO 
                supervised_loss = compute_gbpo_loss(
                    pi_theta_old_token,
                    new_log_probs,
                    supervised_advantages, # 监督信息优势
                    valid_idx, #标识最后一位
                    G
                )
                # 执行GRPO更新
                stats = grpo_update(
                    pt_model,
                    optimizer_dense,
                    optimizer_sparse,
                    scaler,
                    old_log_probs,
                    new_log_probs,           # 新策略的log概率
                    advantages,
                    decoding_advantages,     # 合法非法优势
                    supervised_loss,
                    sample_mask # 标识解码为空的数据
                )

                # 累计统计
                if update_count == reward_args.grpo_epochs_per_batch-1:
                    total_policy_loss += stats['policy_loss']
                    total_entropy_loss += stats['entropy_loss']
                    total_kl += stats['kl_div']
                update_count += 1

            batch_count += 1
            # 训练后再推理一次概率
            with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    new_log_probs = pt_model.compute_sequence_log_probs(
                        pt_input,
                        output_ids,
                        G
                    )
            # 计算计算gbpo后生成概率
            after_gbpo_pi_theta = torch.exp(new_log_probs.sum(dim=1)).view(-1, G)[:, -1]
            #print("after gbpo:", after_gbpo_pi_theta[valid_idx])
            #print("ratio:", after_gbpo_pi_theta[valid_idx]/pi_theta_old[valid_idx])

            # 计算正负样本其gbpo后概率提升降低比例
            pi_theta_acc = torch.where(after_gbpo_pi_theta[valid_idx] > pi_theta_old[valid_idx], 1., -1)
            pi_theta_acc_t = pi_theta_acc[supervised_advantages[valid_idx] > 0]
            term = torch.where(pi_theta_acc_t < 0, torch.tensor(0.0, device=pi_theta_acc_t.device),
                    pi_theta_acc_t).nanmean()
            total_gbpo_t_right += torch.nan_to_num(term, nan=0.0)
            pi_theta_acc_f = pi_theta_acc[supervised_advantages[valid_idx] < 0]
            term = torch.where(pi_theta_acc_f > 0, torch.tensor(0.0, device=pi_theta_acc_f.device),
                   -pi_theta_acc_f).nanmean()
            total_gbpo_f_right += torch.nan_to_num(term, nan=0.0)
            #print("rate:", torch.where(pi_theta_acc == supervised_advantages[valid_idx], 1., 0.).mean())
            #print("task_after_auc_metric:", new_log_probs.sum(-1).exp().view(-1, G)[:, -1][valid_idx], url_label[valid_idx])

            # 计算语料中url资源的在gbpo训练后auc
            task_after_auc_metric(new_log_probs.sum(-1).exp().view(-1, G)[:, -1][valid_idx], url_label[valid_idx])

            # 定期保存检查点
            if batch_count % reward_args.reward_save_steps == 0:
                save_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, batch_count, train_args, logger)

            # 打印批次信息（每10个batch打印一次）
            if batch_count % 10 == 0:
                batch_duration = time.time() - batch_start_time
                batch_start_time = time.time()  # 重置为下一个 batch 开始时间
                avg_policy_loss = total_policy_loss / batch_count
                avg_entropy_loss = total_entropy_loss / batch_count
                avg_kl = total_kl / batch_count
                avg_ndcg = total_ndcg / batch_count
                avg_reward = total_reward / batch_count
                avg_top3_reward = total_top3_reward / batch_count
                avg_hitrate = total_hitrate / batch_count
                avg_gbpo_t_right_rate = total_gbpo_t_right / batch_count
                avg_gbpo_f_right_rate = total_gbpo_f_right / batch_count

                print(f"Time={batch_duration:.2f}s "
                      f"Batch {batch_count}: "
                      #f"PolicyLoss={avg_policy_loss:.4f}, "
                      #f"EntropyLoss={avg_entropy_loss:.4f}, "
                      #f"KL={avg_kl:.4f}, "
                      #f"AvgReward@{reward_args.avgreward_topk}={avg_reward:.4f}, "
                      #f"AvgReward@3={avg_top3_reward:.4f}, "
                      #f"NDCG={avg_ndcg:.4f}, "
                      #f"Hitrate@{G}={avg_hitrate:.4f}, "
                      f"EmptyDecodingRatio={empty_ratio:.4f}, "
                      f"Unique={unique_count:.2f}, "
                      #f"RewardLikeAUC={reward_auc_metric.compute():.4f}, "
                      f"TaskAUC={task_auc_metric.compute():.4f}, "
                      f"TaskAfterAUC={task_after_auc_metric.compute():.4f}, "
                      f"GBPO_T_rightRate={avg_gbpo_t_right_rate:.4f}, "
                      f"GBPO_F_rightRate={avg_gbpo_f_right_rate:.4f}, "
                      )
            #break

        # Epoch结束时打印汇总
        if batch_count > 0:
            total_duration = time.time() - total_start_time
            avg_policy_loss = total_policy_loss / batch_count
            avg_entropy_loss = total_entropy_loss / batch_count
            avg_kl = total_kl / batch_count
            avg_ndcg = total_ndcg / batch_count
            print(f"Epoch {epoch} Summary: "
                  f"Time={total_duration:.2f}s "
                  #f"PolicyLoss={avg_policy_loss:.4f}, "
                  #f"EntropyLoss={avg_entropy_loss:.4f}, "
                  #f"KL={avg_kl:.4f}, "
                  f"NDCG={avg_ndcg:.4f}, "
                  f"TotalBatches={batch_count}")
            #save_checkpoint(pt_model, optimizer_dense, optimizer_sparse, scaler, batch_count, train_args, logger)
            #pt_model.eval()
            #scripted_model = torch.jit.script(pt_model)
            #scripted_model.save(os.path.join(train_args.serving_output_dir, "recmodel.pt"))
            
