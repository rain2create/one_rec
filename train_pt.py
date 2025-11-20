# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : train_pt.py
#   author   : ***
#   date     : 2025-07-18
#   desc     : 
# ======================================================== 
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from torch import autocast
from transformers import get_scheduler
from torch.optim import AdamW, SparseAdam
from itertools import islice
from dataset.pt_dataset import PtDataset
from model.rec_model import RecModel
from utils.logger import logger
from parser.pt_parser import PtParser
from parser.sft_parser import SFTParser
from parser.tokenizer import UrlTokenizer

from arguments.model_args import ModelArguments
from arguments.data_args import DataArguments
from arguments.train_args import TrainingArguments

from utils.train_utils import check_ddp_sync, setup_distributed, cleanup, set_random_seed, log_model_ddp, log_model_info, log_model_parm_info, reduce_sparse_parameters, unwrap_module
from reward.reward_evaluate import compute_ndcg, compute_avgreward, compute_unique_counts, compute_hitrate

def evaluate(model, dataloader, device, url_tokenizer, G, eval_batches=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_hitrate = 0.
    total_batches = 0
    with torch.no_grad():
        # 只迭代eval_batches个batch
        for batch in islice(dataloader, eval_batches):
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            output_logits, loss, _, _, _ = model(batch)
            batch_size = next(iter(batch.values())).size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            bs_res = model.decoder.module.bs_rl_generate(batch, G)
            output_ids = bs_res["output_ids"]
            cand_ids = batch["url"][:, 1:]
            output_urls, sample_mask, empty_ratio, _ = url_tokenizer.batch_decode_processed(output_ids, device)
            batch_hitrate = compute_hitrate(output_ids, cand_ids, G, sample_mask)
            total_hitrate += batch_hitrate
            total_batches += 1
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_hitrate = total_hitrate / total_batches
    return avg_loss, avg_hitrate

def save_checkpoint(model, optimizer_dense, optimizer_sparse, scaler, step, train_args, logger, mode=""):
    if mode == 'rl':
        ckpt_dir = train_args.rl_output_dir
    elif mode == 'sft':
        ckpt_dir = train_args.sft_output_dir
    else:
        ckpt_dir = train_args.output_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{step}.pt")
    max_ckpt_keep = train_args.max_ckpt_keep
    state = {
         'model': {
                     'encoder': unwrap_module(model.encoder).state_dict(),
                     'decoder': unwrap_module(model.decoder).state_dict(),
                     'sparse_model': unwrap_module(model.sparse_model).state_dict()
                  },
         'optimizer_dense': optimizer_dense.state_dict(),
         'optimizer_sparse': optimizer_sparse.state_dict(),
         'scaler': scaler.state_dict(),
         'step': step,
         'hyperparameters': vars(train_args),  # 假设 train_args 是 argparse.Namespace
    }
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint at step {step} to {ckpt_path}")
    # 最多保留的ckpt数量
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
    if len(ckpt_files) > max_ckpt_keep:
        # 按step排序，删除最旧的
        ckpt_files_sorted = sorted(ckpt_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
        to_delete = ckpt_files_sorted[:-max_ckpt_keep]
        for f in to_delete:
            try:
                os.remove(os.path.join(ckpt_dir, f))
            except Exception as e:
                pass

def load_checkpoint(model, optimizer_dense, optimizer_sparse, scaler, ckpt_path, logger, map_location='cpu'):
    # 加载检查点并恢复模型和优化器状态
    # return: 加载后的状态字典（含 step, hyperparameters, env_config）
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if 'model' in checkpoint:
            sub_model = checkpoint['model']
            unwrap_module(model.encoder).load_state_dict(sub_model['encoder'])
            unwrap_module(model.decoder).load_state_dict(sub_model['decoder'])
            unwrap_module(model.sparse_model).load_state_dict(sub_model['sparse_model'])
        if optimizer_dense and 'optimizer_dense' in checkpoint:
            optimizer_dense.load_state_dict(checkpoint['optimizer_dense'])
        if optimizer_sparse and 'optimizer_sparse' in checkpoint:
            optimizer_sparse.load_state_dict(checkpoint['optimizer_sparse'])
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        step = checkpoint['step']
        logger.info(f"Loaded checkpoint from {ckpt_path} at step {step}")
        return {'step': checkpoint['step'], 'hyperparameters': checkpoint['hyperparameters'] }
    else:
        logger.warning(f"Checkpoint file {ckpt_path} not found. Starting from scratch.")
        return {'step': 0, 'hyperparameters': None }
   

def build_optimizers(model, train_args):
    lr_dense = train_args.learning_rate
    lr_sparse = train_args.lr_sparse if hasattr(train_args, "lr_sparse") else 1e-2
    weight_decay = train_args.weight_decay if hasattr(train_args, "weight_decay") else 0.01

    # 收集稀疏和非稀疏参数
    sparse_params = []
    dense_decay_params = []
    dense_no_decay_params = []
    
    sparse_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and module.sparse:
            sparse_param_names.add(name + '.weight')

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in sparse_param_names:
            sparse_params.append(param)
        else:
            if name.endswith(".bias") or "norm" in name.lower():
                dense_no_decay_params.append(param)
            else:
                dense_decay_params.append(param)

    # AdamW 优化 dense 参数，支持分组
    optimizer_dense = AdamW(
        [
            {"params": dense_decay_params, "weight_decay": weight_decay},
            {"params": dense_no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr_dense,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 稀疏参数优化器（可选 Adagrad or SparseAdam）
    optimizer_sparse = SparseAdam(sparse_params, lr=lr_sparse)
    logger.info(f"Dense with decay: {len(dense_decay_params)}, "
                f"Dense no decay: {len(dense_no_decay_params)}, "
                f"Sparse: {len(sparse_params)}")
    return optimizer_dense, optimizer_sparse

def has_grad(optimizer):
    return any(p.grad is not None for group in optimizer.param_groups for p in group['params'])

def train(model_args, data_args, train_args):
    # 判断是否分布式
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ 
    local_rank = int(os.environ["LOCAL_RANK"]) if use_distributed else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    G = 10

    if use_distributed:
        setup_distributed()

    set_random_seed(train_args.seed + local_rank)
    is_main_process = (not use_distributed) or (dist.get_rank() == 0)

    # dataloader
    url_tokenizer = UrlTokenizer.from_args(data_args)
    mode = ""
    if not train_args.train_sft:
        pt_parser = PtParser(data_args.columns_def, tokenizer=url_tokenizer)
    else:
        pt_parser = SFTParser(data_args.columns_def, tokenizer=url_tokenizer)
        mode = "sft"

    logger.info(f"Define train dataset")
    pt_dataset = PtDataset.from_args(data_args, parser_func=pt_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(pt_dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # val dataloader
    val_dataloader = None
    if data_args.test_root_dirs and data_args.val_root_dirs:
        # 测试模式
        logger.info(f"Define val dataset")
        val_dataset = PtDataset.from_args(data_args, parser_func=pt_parser.parser, test_root_dirs=data_args.val_root_dirs)
        val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=max(1, data_args.num_workers // 2), pin_memory=True)

    # 构造val dataloader
    val_dataset = PtDataset("/mnt/video.rank.s101", pt_parser.parser, 0, with_today=True, batch_size=10000, shuffle_buffer_size=1000)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=max(1, data_args.num_workers // 2), pin_memory=True)

    ### model
    model = RecModel.from_args(model_args).cuda(device)
    #model = RecModel.from_args(model_args)
    
    #model.decoder.compile()
    #model.encoder.compile()

    #model.sparse_model = model.sparse_model.cpu()
    #model.encoder = model.encoder.cuda(device)
    #model.decoder = model.decoder.cuda(device)


    if use_distributed and torch.cuda.device_count() > 1:
        # 只封装dense部分
        model.encoder = DDP(model.encoder, device_ids=[local_rank], find_unused_parameters=True)
        model.decoder = DDP(model.decoder, device_ids=[local_rank], find_unused_parameters=True)
    else:
        use_distributed = False

    # 打印模型结构和参数统计（只在主进程）
    if is_main_process:
        log_model_ddp(model, logger)
        log_model_info(model, logger)
        log_model_parm_info(model, logger)    

    # 构造 optimizer 
    optimizer_dense, optimizer_sparse = build_optimizers(model, train_args)

    scaler = torch.amp.GradScaler('cuda', enabled=False)

    lr_scheduler = get_scheduler( 
                "constant_with_warmup", 
                optimizer=optimizer_dense,
                num_warmup_steps=train_args.warmup_steps
            )

    model.train()
    global_step = 0

    if train_args.resume_from_checkpoint:
        ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-{train_args.resume_from_checkpoint}.pt")
        global_step = load_checkpoint(model, optimizer_dense, optimizer_sparse, scaler, ckpt_path, logger, )['step']
        if use_distributed:
            dist.barrier()  # 所有 rank 同步

    try:
        for epoch in range(train_args.epochs):
            data_iter = iter(dataloader)
            step = 0
            dummy_batch = None  # 用于存第一个 batch 作为 dummy
            total_mask_valid_num = 0
            total_final_valid_num = 0
            total_mask_valid_cnt = 0
            total_final_valid_cnt = 0
            while True:
                # 本地是否结束
                try:
                    batch = next(data_iter)
                    local_finished = torch.tensor(0, device=device)
                    if dummy_batch is None:
                        dummy_batch = {k: v.clone().detach().to(device) for k, v in batch.items()}
                except StopIteration:
                    batch = None
                    local_finished = torch.tensor(1, device=device)

                # 所有的rank同步是否结束
                global_finished = local_finished.clone()
                if use_distributed:
                    dist.all_reduce(global_finished, op=dist.ReduceOp.SUM)

                # 如果所有rank都结束，跳出epoch
                if global_finished.item() == (dist.get_world_size() if use_distributed else 1):
                    break 

                # 如果当前rank空了，构造dummy batch
                if batch is None:
                    is_dummy = True
                    batch = {k: v.cuda(device) for k, v in dummy_batch.items()}
                    #batch = {k: v for k, v in dummy_batch.items()}
                else:
                    batch = {k: v.cuda(device) for k, v in batch.items()}
                    #batch = {k: v for k, v in batch.items()}
                    is_dummy = False

                # with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=train_args.use_amp):
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                    output_logits, loss, mask_valid, final_valid, probs = model(batch)
                    loss = loss / train_args.gradient_accumulation_steps
                    if is_main_process:
                        total_mask_valid_num += mask_valid
                        total_final_valid_num += final_valid
                        total_mask_valid_cnt += 1 
                        total_final_valid_cnt += 1 
                
                if is_dummy:
                    loss = loss * 0.0

                scaler.scale(loss).backward()

                if (step + 1) % train_args.gradient_accumulation_steps == 0:
                    # 每隔gradient_accumulation_steps步，更新一次模型参数
                    # dense参数更新
                    scaler.unscale_(optimizer_dense)
                    # 只对dense参数做clip_norm
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad and p.grad is not None and not p.grad.is_sparse],
                        train_args.max_grad_norm
                    )
                    
                    scaler.step(optimizer_dense)
                    optimizer_dense.zero_grad()

                    optimizer_sparse.step()
                    optimizer_sparse.zero_grad()

                    scaler.update()
                    lr_scheduler.step()

                    if use_distributed and global_step % train_args.reduce_steps == 0:
                        reduce_sparse_parameters(model.sparse_model)
                        # if is_main_process:
                        #     logger.info(f"[SYNC] Sparse parameters averaged across {dist.get_world_size()} ranks at step {step}")
                        # check_ddp_sync(model, prefix=f"step{step}", verbose=True, num_params_to_check=5)

                    if is_main_process:
                        if is_dummy:
                            if global_step % train_args.logging_steps == 0:
                                logger.info(f"[{epoch}] step {global_step}, rank 0 data empty")
                        else:
                            if global_step % train_args.logging_steps == 0:
                                logger.info(
                                        f"[{epoch}] step {global_step}, actual loss: {loss.item() * train_args.gradient_accumulation_steps:.4f}, target prob: {probs:.9f}, "
                                        f"avg mask valid rate {total_mask_valid_num / (data_args.batch_size * total_mask_valid_cnt * train_args.gradient_accumulation_steps):.4f}, "
                                        f"avg final valid rate {total_final_valid_num / (data_args.batch_size * total_final_valid_cnt * train_args.gradient_accumulation_steps):.4f}, "
                                        f"pos_pi_new: {model.decoder.pos_pi_new.result():.6f}, neg_pi_new: {model.decoder.neg_pi_new.result():.6f}, cts_pi_new: {model.decoder.cts_pi_new.result():.6f}, "
                                        f"ct_loss: {model.decoder.loss_ct.result():.6f}, rl_loss: {model.decoder.loss_rl.result():.6f}"
                                    )
                                total_mask_valid_num = 0 
                                total_final_valid_num = 0
                                total_mask_valid_cnt = 0
                                total_final_valid_cnt = 0

                            if train_args.output_dir and train_args.save_steps > 0 and global_step % train_args.save_steps == 0 and global_step > 0:
                                save_checkpoint(model, optimizer_dense, optimizer_sparse, scaler, global_step, train_args, logger, mode)
                        
                            '''
                            if val_dataloader is not None and global_step % train_args.val_steps == 0:
                                # 在evaluate时计算beam search hitrate, 如果是pt，计算曝光样本的hitrate，如果是sft，计算正样本的hitrate
                                val_loss, val_hitrate = evaluate(model, val_dataloader, device, url_tokenizer, G, eval_batches=train_args.val_batches)
                                logger.info(f"[{epoch}] step {global_step}, val loss: {val_loss:.4f}, hitrate@{G} : {val_hitrate}")
                            '''
                    global_step += 1
            step += 1

        if is_main_process:
            save_checkpoint(model, optimizer_dense, optimizer_sparse, scaler, global_step, train_args, logger, mode)
        if use_distributed:
            dist.barrier()  # 等待所有 rank 到达末尾

        if not data_args.test_root_dirs and is_main_process:
            # 使用真实语料, 打印pretrain done标记
            pt_dataset.after_train_process()

        if use_distributed:
            cleanup()

    except Exception as e:
        logger.error(f"Error in training: {e}")
        if use_distributed:
            cleanup()
        raise e

    
if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, train_args)

