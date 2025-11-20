mkdir -p ../data
mkdir -p ../log
max_ckpt=$(ls ../data/pt_model/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
CUDA_VISIBLE_DEVICES="1," /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 train_pt.py --corpus_day 7 --with_today False --done_spec "gbpo_base" --resume_from_checkpoint "${max_ckpt}" > ../log/log_pt 2>&1
mv ../data/serving/recmodel_new.pt ../data/serving/recmodel_new.pt.bak
CUDA_VISIBLE_DEVICES="4" /data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/python train_reload_trace.py --checkpoint_step ${max_ckpt} > ../log/reload_trace.log 2>&1
/data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/python ../scripts/concat_dupid.py >> ../log/concat_dupid.log 2>&1
/data03/search/tanxiaofeng/miniconda3/envs/cuda12/bin/python ../scripts/upload.py >> ../log/upload.log 2>&1

