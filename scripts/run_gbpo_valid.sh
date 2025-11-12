mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES=5 nohup /data03/search/tanxiaofeng/miniconda3/envs/reward/bin/python -u train_gbpo_token_valid.py --corpus_day 5 --with_today False > ../log/log_reward_gbpo_1_valid_0.1_grpo_1_group_100_bs_50_ptratio_01 2>&1 &
