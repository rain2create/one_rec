mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="1" nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/python train_pt.py --corpus_day 14 --with_today False --resume_from_checkpoint "184398" --train_sft True --batch_size 20480 --num_workers 5 --val_steps 10000 --save_steps 1000  > ../log/log_sft_like_follow 2>&1 &
