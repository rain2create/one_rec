mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="0,1,3,4" nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 4 train_pt.py --corpus_day 7 --with_today False --resume_from_checkpoint "220970" --train_sft True --batch_size 1024 --done_spec "sft_update_desample_n3" --num_workers 5 > ../log/log 2>&1 &
#CUDA_VISIBLE_DEVICES="0,1,2,4" nohup /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 4 train_pt.py --corpus_day 7 --with_today False --train_sft True --batch_size 10240 --done_spec "sft_update_desample_n1" --num_workers 5 > ../log/log 2>&1 &
