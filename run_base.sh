mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="2," nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 train_pt.py --corpus_day 30 --with_today False --done_spec "pretrain_t2kmix1_incr_epsxx" --resume_from_checkpoint "xxxx" > ../log/log_pt 2>&1 &
