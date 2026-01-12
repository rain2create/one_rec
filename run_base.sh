mkdir -p ../data
mkdir -p ../log
CUDA_VISIBLE_DEVICES="4," nohup /mnt_onerec_model/tools/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 train_pt.py --corpus_day 30 --with_today False --done_spec "base.finish.expl.64.dur" --resume_from_checkpoint "xxxx" > ../log/log_pt 2>&1 &
