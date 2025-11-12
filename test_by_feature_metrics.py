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
import numpy as np
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
    test_root_dirs = "/mnt/video.rank.s101.pos_neg/20251104/2025110409/")
dataloader = DataLoader(pt_dataset, batch_size=None, num_workers=1, pin_memory=True)

# 加载 TorchScript 模型

# loaded_model = torch.jit.load("../data/serving/recmodel.pt")
# loaded_model.eval()  # 切换到 eval 模式


device = "cuda:0"
model = RecModel.from_args(model_args)

#ckpt_path = os.path.join(train_args.output_dir, f"checkpoint-936217.pt")
#ckpt_path = "/data04/search/tanxiaofeng/dev/onerec_new_id/data/pt_model/checkpoint-315068.pt"
ckpt_path = "../data/pt_model/checkpoint-112847.pt"
global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, None)

model = model.to(device)


def cal_hits(his_ids, r_ids):
    his_set = set(his_ids.tolist())
    return len(his_set), len(his_set & set(r_ids))

metrics = [0] * 7
urls_recall = set()

for i, x in enumerate(dataloader):
    if i > 50000:
        break

    inputs = {k:v.to(device) for k, v in x.items()}

    #print(inputs)
    loglik_ids = inputs['logreallike_md5'][0]
    logque_ids = inputs['logque_md5'][0]
    logses_ids = inputs['logsess_md5'][0]

    res = model.generate(inputs, group_num=100)
    # res = model.rl_generate(inputs, group_num=100, top_k=20, top_p=0.1, strategy='topp')
    output_ids = res["output_ids"]
    # log_probs = res["output_logprobs"].sum(dim=-1)

    urls = url_tokenizer.batch_decode(output_ids)
    urls_hashid = pt_parser._hash_to_tensor(np.asarray(urls), 20000019).tolist()
    urls_recall.update(urls_hashid)

    ######## cal recall@k #######
    a, b = cal_hits(loglik_ids, urls_hashid)
    metrics[0] += a
    metrics[1] += b
    a, b = cal_hits(logque_ids, urls_hashid)
    metrics[2] += a
    metrics[3] += b
    a, b = cal_hits(logses_ids, urls_hashid)
    metrics[4] += a
    metrics[5] += b
    metrics[6] = len(urls_recall)
    if i % 1000 == 0:
        print(metrics)

    #for x, p, url, hshid in zip(output_ids.tolist(), log_probs.tolist(), urls, urls_hashid):
    #    print(x, p, url, hshid)
    #    print(get_md5_info(url))

print(metrics)

