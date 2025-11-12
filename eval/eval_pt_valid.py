# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : eval_pt_valid.py
#   author   : ***
#   date     : 2025-08-01
#   desc     : 
# ========================================================

# 验证pretrain model生成结果的合法性

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataset.pt_dataset import PtDataset
from model.rec_model import RecModel
from utils.logger import logger
from parser.pt_parser import PtParser
from parser.tokenizer import UrlTokenizer

from arguments.model_args import ModelArguments
from arguments.data_args import DataArguments
from arguments.train_args import TrainingArguments
from train_pt import load_checkpoint
from utils.train_utils import set_random_seed


def eval_valid(model_args, data_args, train_args):
    device = torch.device("cuda:6")
    set_random_seed(train_args.seed)

    # dataloader
    url_tokenizer = UrlTokenizer.from_args(data_args)
    pt_parser = PtParser(data_args.columns_def, tokenizer=url_tokenizer)

    logger.info(f"Define val dataset")
    pt_dataset = PtDataset.from_args(data_args, parser_func=pt_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(pt_dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    # model
    model = RecModel.from_args(model_args).cuda(device)
    model.eval()
    

    if train_args.resume_from_checkpoint:
        ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-{train_args.resume_from_checkpoint}.pt")
        global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, None)
    
    valid_num = 0
    total_num = 0

    idx = 0
    for step, batch in enumerate(dataloader):
        batch = {k: v.cuda(device) for k, v in batch.items()}
        output_token = model.generate(batch)
        # output_token = model.rl_generate(batch, top_p=1.0)['output_ids']
        output_urls =  url_tokenizer.batch_decode(output_token.tolist())
        for x in output_urls:
            if x:
                valid_num += 1
            total_num += 1
        idx += 1
        if idx == 100:
            break
    print("total_num: %d, valid num: %d, valid_ratio: %.4f" %(total_num, valid_num, valid_num / total_num))


    
if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    eval_valid(model_args, data_args, train_args)
