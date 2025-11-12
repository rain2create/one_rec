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
from parser.reward_parser import RewardParser
from dataset.pt_dataset import PtDataset
from model.rec_model import RecModel

from reward.reward_scorer_online import RewardScorerOnline
from reward.reward_algorithm import compute_advantages, compute_decoding_advantage, compute_supervised_advantage
from reward.reward_evaluate import compute_ndcg, compute_avgreward, compute_unique_counts, compute_hitrate

torch.set_printoptions(
    sci_mode=False,  # 关闭科学计数法
    precision=4      # 保留 4 位小数
)

def compute_gbpo_loss(old_log_probs, new_log_probs, supervised_advantages, add_url_mask, G, clip_epsilon=0.1):
    """
    old_log_probs, new_log_probs [B*G, 3]
    supervised_advantages, add_url_mask [B]
    """
    pi_theta = torch.exp(new_log_probs.sum(dim=1)).view(-1, G)[:, -1] # [B]
    pi_theta_old = torch.exp(old_log_probs.sum(dim=1)).view(-1, G)[:, -1] # [B]
    pi_theta_old_ = torch.max(torch.where(supervised_advantages >= 0, pi_theta.detach(), 1 - pi_theta.detach()), pi_theta_old) 
    ratio = pi_theta/pi_theta_old_
    supervised_loss = -0.1*ratio[~add_url_mask].mean()
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
    ratio = torch.exp(new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)[valid_idx].mean()
    #print("======grpo_update========")
    #print("new_log_probs:", torch.exp(new_log_probs.sum(dim=1))[valid_idx])
    #print("ratio:", ratio[valid_idx])

    # 熵奖励 loss
    entropy = - (new_log_probs.exp() * new_log_probs).sum(dim=1)[valid_idx].mean()
    entropy_loss = -0.0 * entropy

    # 合法非法loss
    legal_surr1 = ratio * decoding_advantages
    legal_surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * decoding_advantages
    legal_loss = -0.4 * torch.min(legal_surr1, legal_surr2).mean()

    # 计算KL散度: KL(new||old) = Σ p_new * log(p_new / p_old)
    kl_div = 0. * (new_log_probs.exp() * (new_log_probs - old_log_probs)).sum(dim=1)[valid_idx].mean()

    loss = policy_loss + entropy_loss + legal_loss + kl_div + supervised_loss

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
    #data_args.file_shuffle = False
    #data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 15
    train_args.learning_rate /= 1000
    train_args.lr_sparse /= 100000
    model_args.encoder_dropout = 0.
    model_args.decoder_dropout = 0.
    G = reward_args.num_samples_per_input

    like_auc_metric = AUCMetric()
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
    reward_parser = RewardParser(data_args.columns_def, data_args.reward_serving_column_idx, tokenizer=url_tokenizer, logreallike_len=reward_args.logreallike_len)
    dataset = PtDataset.from_args(data_args, parser_func = reward_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # reward model
    scorer = RewardScorerOnline.from_args(reward_args)
    # 训练过程进行固定权重的多目标拟合
    #right_click_pred, dt_pred, finish_pred, like_pred, zmtlink_pred, follow_pred, dislike_pred, skip_pred, last_pred
    reward_weights = torch.tensor([0., 1., 1.1, 2., 1., 2., 1., 0., 1.], device=device)
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
        batch_count = 0

        for x in dataloader:
            if not x:
                continue
            pt_input, reward_input, like_label = x
            pt_input = {k: v.to(device) for k, v in pt_input.items()}
            real_batch = reward_input.shape[0]

            # 生成多个输出并获取log概率
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    # beam search
                    output_rl = pt_model.bs_rl_generate(pt_input, G)
                    # other sample methods
                    #output_rl = pt_model.rl_generate(pt_input, G, top_p = reward_args.top_p, strategy = reward_args.strategy)
            output_ids = output_rl['output_ids'].clone()
            if reward_args.add_corpus_url:
                output_ids = output_ids.view(real_batch, G, 3)
                url = pt_input["url"][:,1:]
                add_corpus_url_mask = (url == data_args.unk_token_id) # [b ,3]
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
            cand_urls = expanded_reward_input[:, 2].copy()
            expanded_reward_input[:, 2] = output_urls
            reward_input_combined = np.concatenate([expanded_reward_input, reward_input])

            # reward模型获取得分
            rewards = scorer.predict(reward_input_combined)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            rewards = (rewards * reward_weights.unsqueeze(0)).sum(dim=-1)
            like_rewards = rewards[-real_batch:]
            rewards = rewards[:-real_batch]

            # 计算优势
            advantages = compute_advantages(rewards, G, sample_mask)
            decoding_advantages = compute_decoding_advantage(sample_mask)
            if reward_args.add_corpus_url:
                supervised_advantages = compute_supervised_advantage(algfinish_arr, add_corpus_url_mask[:,0], reward_args.add_corpus_url)
            
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
            # 计算like auc
            url_log_probs = pt_model.compute_sequence_log_probs(
                    pt_input, url_ids, 1).detach()
            url_probs = url_log_probs.sum(-1).exp()
            like_auc_metric(url_probs, like_label.astype(int))
            reward_auc_metric(like_rewards, like_label.astype(int))

            #valid_idx = sample_mask.nonzero(as_tuple=True)[0]
            #print("output_ids: ", output_ids[valid_idx])
            #print("advantages:", advantages[valid_idx])
            #print("decoding_advantages:", decoding_advantages)
            #print("old_log_probs:", torch.exp(old_log_probs.sum(dim=1))[valid_idx])
            #print("advantages/old_log_probs:", torch.exp(-old_log_probs.sum(dim=1))[valid_idx] * advantages[valid_idx])
            # 数据多轮更新
            update_count = 0
            for _ in range(reward_args.grpo_epochs_per_batch):
                with torch.amp.autocast('cuda', enabled=train_args.use_amp):
                    new_log_probs = pt_model.compute_sequence_log_probs(
                        pt_input,
                        output_ids,
                        G
                    )
                # 计算GBPO 
                supervised_loss = 0
                if reward_args.add_corpus_url and reward_args.add_supervised_loss:
                    supervised_loss = compute_gbpo_loss(
                        old_log_probs,
                        new_log_probs,
                        supervised_advantages, # 监督信息优势
                        add_corpus_url_mask[:,0], #标识最后一位
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

                print(f"Time={batch_duration:.2f}s "
                      f"Batch {batch_count}: "
                      #f"PolicyLoss={avg_policy_loss:.4f}, "
                      #f"EntropyLoss={avg_entropy_loss:.4f}, "
                      #f"KL={avg_kl:.4f}, "
                      f"AvgReward@{reward_args.avgreward_topk}={avg_reward:.4f}, "
                      f"AvgReward@3={avg_top3_reward:.4f}, "
                      f"NDCG={avg_ndcg:.4f}, "
                      f"Hitrate@{G}={avg_hitrate:.4f}, "
                      f"EmptyDecodingRatio={empty_ratio:.4f}, "
                      f"Unique={unique_count:.2f}, "
                      f"LikeAUC={like_auc_metric.compute():.4f}, "
                      f"RewardLikeAUC={reward_auc_metric.compute():.4f}")
            #break
            if batch_count % 100 == 0:
                pt_model.eval()
                scripted_model = torch.jit.script(pt_model)
                scripted_model.save(os.path.join(train_args.serving_output_dir, "recmodel.pt"))
                pt_model.train()

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
            pt_model.eval()
            scripted_model = torch.jit.script(pt_model)
            scripted_model.save(os.path.join(train_args.serving_output_dir, "recmodel.pt"))
            pt_model.train()
