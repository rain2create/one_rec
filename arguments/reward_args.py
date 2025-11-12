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
class RewardArguments:
    # parser
    reward_service_endpoint: str = "10.177.75.200:9503"
    reward_service_model_name: str = "model"
    reward_service_signature: str = "serving_default"
    reward_service_input_name: str = "inputs"
    reward_service_output_name: str = "scores_list"
    reward_service_timeout: float = 30.0
    reward_max_batch_size: int = 20  # 服务端限制

    # train
    reward_train_epochs: int = 1
    reward_save_steps: int = 1000
    num_samples_per_input: int = 100
    grpo_epochs_per_batch: int = 4
    strategy: str = "topp"
    top_p: float = 1.0
    temperature: float = 1.5
    logreallike_len: int = 30
    add_corpus_url: bool = False
    add_supervised_loss: bool = False

    # evaluate
    avgreward_topk: int = 10
