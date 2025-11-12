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
from functools import lru_cache
import redis

ops = redis.Redis(host='10.175.162.34', port=8648, password='3fe885fb4cec4dcd')

@lru_cache(maxsize=10000)
def get_md5_info(md5):
    try:
        gnid = "https://360kuai.com/%s" %(ops.hget(md5, "gnid").decode())
        return "%s\t%s\t%s\t%s\t%s" %(md5, ops.hget(md5, "title").decode(), ops.hget(md5, "store_videotag").decode(), ops.hget(md5, "from").decode(), gnid)
    except Exception as e:
        return md5

def eval_valid(model_args, data_args, train_args):
    device = torch.device("cuda:0")
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
    
    # ckpt_path = "/data04/search/tanxiaofeng/dev/onerec_v12/data/pt_model/checkpoint-272030.pt"
    # ckpt_path = "/data04/search/tanxiaofeng/dev/onerec_v12_new/data/rl_model/checkpoint-302596.pt"
    # ckpt_path = "/data04/search/tanxiaofeng/dev/onerec_v12_new/data/sft_model/checkpoint-273871.pt"
    ckpt_path = "/data04/search/tanxiaofeng/dev/onerec_new_id/data/rl_model/checkpoint-4000.pt"
    global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, None)
    
    valid_num = 0
    total_num = 0

    idx = 0
    G = 10
    for step, batch in enumerate(dataloader):
        logreallike_md5_list = batch["logreallike_md5_list"]
        logque_md5_list = batch["logque_md5_list"]
        batch = {k: v.cuda(device) for k, v in batch.items() if k != "logreallike_md5_list" and k != "logque_md5_list"} 
        res = model.bs_rl_generate(batch, group_num=G)
        output_token = res['output_ids'] # [B*G, 3]
        output_urls, sample_mask, empty_ratio, per_batch_non_empty_counts = url_tokenizer.batch_decode_processed(output_token, device)
        output_urls = output_urls.reshape(-1, G)
        
        for i, (x, y) in enumerate(zip(logreallike_md5_list, output_urls)):
            for tmp in x.split("|"):
                print(get_md5_info(tmp))
            print("****************")
            for idx in range(G):
                try:
                    print(get_md5_info(y[idx]))
                except:
                    print("error")
            print("\n\n")
        if step > 10:
            break
        

    
if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    eval_valid(model_args, data_args, train_args)
