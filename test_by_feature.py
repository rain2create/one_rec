# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : test.py
#   author   : ***
#   date     : 2025-08-22
#   desc     : 
# ======================================================== 

import os
import torch
import json
from parser.tokenizer import UrlTokenizer
from arguments.data_args import DataArguments 
from parser.pt_parser import PtParser
from dataset.pt_dataset import PtDataset
from torch.utils.data import DataLoader
from arguments.model_args import ModelArguments
from arguments.train_args import TrainingArguments
from model.rec_model import RecModel
from train_pt import load_checkpoint
from utils.logger import logger
import redis
from functools import lru_cache

ops = redis.Redis(host='10.175.162.34', port=8648, password='3fe885fb4cec4dcd')

@lru_cache(maxsize=10000)
def get_md5_info(md5):
    try:
        gnid = "https://360kuai.com/%s" %(ops.hget(md5, "gnid").decode())
        return "%s\t%s\t%s\t%s\t%s" %(md5, ops.hget(md5, "title").decode(), ops.hget(md5, "store_videotag").decode(), ops.hget(md5, "from").decode(), gnid)
    except Exception as e:
        return md5


data_args = DataArguments()
model_args = ModelArguments()
train_args = TrainingArguments()

url_tokenizer = UrlTokenizer.from_args(data_args)

# load_data
pt_parser = PtParser(data_args.columns_def, tokenizer=url_tokenizer) 
pt_dataset = PtDataset(".", pt_parser.parser, 0, batch_size = 1, 
        # test_root_dirs = "/mnt/video.rank.s101.pos_neg_test/20251113/2025111316/")
        test_root_dirs = "./tmp")
dataloader = DataLoader(pt_dataset, batch_size=None, num_workers=1, pin_memory=True)

# 加载 TorchScript 模型

# loaded_model = torch.jit.load("../data/serving/recmodel.pt")
# loaded_model.eval()  # 切换到 eval 模式


device = "cuda:0"
model = RecModel.from_args(model_args)

ckpt_path = "../data/pt_model/checkpoint-53168.pt"
global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, None)
model = model.to(device)


for x in dataloader:
    inputs = {k:v.to(device) for k, v in x.items()}
    print(inputs)
    with torch.no_grad():
        res = model.generate(inputs, group_num=300)
    output_ids = res["output_ids"]
    log_probs = res["output_logprobs"]

    urls = url_tokenizer.batch_decode(output_ids)
    for x, p, url in zip(output_ids.tolist(), log_probs.tolist(), urls):
        print(x, p, url)
        print(get_md5_info(url))
    print("*" * 120)

