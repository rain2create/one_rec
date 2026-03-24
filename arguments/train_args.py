# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : train_args.py
#   author   : ***
#   date     : 2025-07-18
#   desc     : 
# ======================================================== 
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingArguments:
    seed: int = 42
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 100
    train_sft: bool = False
    output_dir: Optional[str] = "../data/pt_model"
    rl_output_dir: Optional[str] = "../data/rl_model"
    sft_output_dir: Optional[str] = "../data/sft_model"
    serving_output_dir: Optional[str] = "../data/serving"
    save_steps: int = 10000
    val_steps: int = 10000
    reduce_steps: int = 500
    warmup_steps: int = 10000
    learning_rate: float = 2e-4
    lr_sparse: float = 6e-2
    max_grad_norm: float = 1.0
    use_amp: bool = True
    weight_decay: float = 0.01
    val_batches: int = 100  # 验证集跑多少个batch
    resume_from_checkpoint: Optional[str] = None
    max_ckpt_keep: int = 3
    checkpoint_step: int = 0
