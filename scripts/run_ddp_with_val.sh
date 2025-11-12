mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="1,2,3,4" nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 4 train_pt.py --test_root_dirs "/mnt/video.rank.s101/20250717" --val_root_dirs "/mnt/video.rank.s101/20250718" > ../log/log 2>&1 &
