mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES=4 nohup /data03/search/tanxiaofeng/miniconda3/envs/reward/bin/python -u train_grpo_neg_sample.py --corpus_day 1 --with_today True> ../log/log_rl 2>&1 &
