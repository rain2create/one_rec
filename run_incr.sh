mkdir -p ../data
mkdir -p ../log

torch_path=/mnt_onerec_model/tools/cuda12/bin/torchrun
python_path=/mnt_onerec_model/tools/cuda12/bin/python
done_spec=seq_enc.finish.expl
use_corpus_days=30
cuda_idx=1

max_ckpt=$(ls ../data/pt_model/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
CUDA_VISIBLE_DEVICES="${cuda_idx}" ${torch_path} --standalone --nnodes 1 --nproc-per-node 1 train_pt.py --corpus_day ${use_corpus_days} --with_today False --done_spec "${done_spec}" --resume_from_checkpoint "${max_ckpt}" > ../log/log_pt 2>&1
mv ../data/serving/recmodel_new.pt ../data/serving/recmodel_new.pt.bak
max_ckpt=$(ls ../data/pt_model/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
CUDA_VISIBLE_DEVICES="${cuda_idx}" ${python_path} train_reload_trace.py --checkpoint_step ${max_ckpt} > ../log/reload_trace.log 2>&1
${python_path} ../scripts/concat_dupid.py >> ../log/concat_dupid.log 2>&1
${python_path} ../scripts/upload.py >> ../log/upload.log 2>&1

