mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES=4 nohup /data03/search/tanxiaofeng/miniconda3/envs/reward/bin/python -u train_gbpo_token_valid.py --corpus_day 4 --with_today False > ../log/log_reward 2>&1 &
