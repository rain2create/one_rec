mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES=6 nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/python train_pt.py --corpus_day 14 --with_today True > ../log/log 2>&1 &
