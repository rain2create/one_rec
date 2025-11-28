mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="5," nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 train_pt.py --corpus_day 30 --with_today False --done_spec "pretrain_moe_gbpo_ce_mask_book" --resume_from_checkpoint "xxxx" > ../log/log_pt 2>&1 &
