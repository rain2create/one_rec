# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : train_utils.py
#   author   : ***
#   date     : 2025-07-28
#   desc     : 
# ======================================================== 
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def check_ddp_sync(model, prefix="", max_diff=1e-6, max_std=1e-6, verbose=True, num_params_to_check=3):
    """
    检查模型中 的 DDP 参数是否同步。
    Args:
        model: 主模型，包含 encoder / decoder / sparse_model 模块。
        prefix: 日志前缀，用于标识当前步骤。
        max_diff: 检查各 rank 参数 mean 的最大容忍差值。
        max_std: 检查参数 mean 的标准差阈值，超过则报警。
        verbose: 是否打印所有均值信息。
        num_params_to_check: 每个模块最多检查多少个参数。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    modules = [("encoder", model.encoder), ("decoder", model.decoder), ("sparse_model", model.sparse_model)]

    for name, module in modules:
        checked = 0
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue

            with torch.no_grad():
                local_mean = p.mean()
            gathered_means = [torch.zeros_like(local_mean) for _ in range(world_size)]
            dist.all_gather(gathered_means, local_mean)

            gathered_vals = torch.tensor([t.item() for t in gathered_means])
            mean_val = gathered_vals.mean()
            std_val = gathered_vals.std()
            max_val = gathered_vals.max()
            min_val = gathered_vals.min()
            diff = (max_val - min_val).item()

            if rank == 0:
                status = "✅ OK"
                if diff > max_diff or std_val > max_std:
                    status = "⚠️ MISMATCH"
                if verbose or status != "✅ OK":
                    print(f"[DDP-CHECK] {prefix} {name} param {n} {checked}: mean={mean_val:.6f}, std={std_val:.2e}, min={min_val:.6f}, max={max_val:.6f}, diff={diff:.2e} -> {status}")
            checked += 1
            if checked >= num_params_to_check:
                break

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """销毁分布式进程组"""
    dist.destroy_process_group()

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_model_ddp(model, logger):
    for name, param in model.named_children():
        logger.info(f"{name}: {type(param)}")

def log_model_info(model, logger):
    if isinstance(model, DDP):
        model = model.module
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sparse_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and module.sparse:
            sparse_param_names.add(name + '.weight')
    logger.info(f"sparse_param_names: {sparse_param_names}")
    sparse_params = sum(p.numel() for name, p in model.named_parameters() if name in sparse_param_names)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Sparse Parameters: {sparse_params:,}")
    logger.info(f"Model Structure:\n{model}")

def log_model_parm_info(model, logger):
    if isinstance(model, DDP):
        model = model.module
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.shape} {param.numel()} {param.device} (requires_grad)")
        else:
            logger.info(f"{name}: {param.shape} {param.numel()} {param.device} (not requires_grad)")

def reduce_sparse_parameters(model_sparse):
    # 对 sparse_model 的所有参数做 all_reduce 求和，并除以 world_size 实现平均。
    # 周期性调用，缓解多卡稀疏参数漂移。
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    for param in model_sparse.parameters():
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= world_size

def unwrap_module(m):
    return m.module if hasattr(m, 'module') else m


def compute_hitrate(output_urls, cand_urls, G):
    # output_urls: [batch_size*G, 3]
    # cand_urls: [batch_size, 3]
    B = output_urls.shape[0] // G
    output_urls_2d = output_urls.reshape(B, G, -1)
    cand_urls_2d = cand_urls.reshape(B, G)
    matches = (output_urls_2d == cand_urls_2d)
    any_match_per_row = matches.any(axis=1)  # shape: (B,), dtype=bool
    result = any_match_per_row.mean()
    return result
