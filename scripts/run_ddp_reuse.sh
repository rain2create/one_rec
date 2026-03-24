mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="0,2,3,4" nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 4 train_pt.py --corpus_day 7 --with_today True --resume_from_checkpoint "237757" > ../log/log 2>&1 &
