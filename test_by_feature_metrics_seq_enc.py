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
import re
import glob

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
    test_root_dirs = "/mnt_onerec_model/corpus/video.rank.s101.pos_neg_test/20260118/2026011816/")
dataloader = DataLoader(pt_dataset, batch_size=None, num_workers=1, pin_memory=True)

# 加载 TorchScript 模型
# loaded_model = torch.jit.load("../data/serving/recmodel.pt")
# loaded_model.eval()  # 切换到 eval 模式


device = "cuda:0"
model = RecModel.from_args(model_args)

latest_version = max(int(re.search(r'(\d+)\.pt$', f).group(1)) for f in glob.glob('../data/pt_model/checkpoint-*.pt'))
ckpt_path = "../data/pt_model/checkpoint-{}.pt".format(latest_version)
global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, None)

model = model.to(device)
model.eval()


def cal_hits(his_ids, r_ids):
    his_set = set(his_ids)
    return len(his_set), len(his_set & set(r_ids))

def parse_his(his_seq):
    his_seq = his_seq.tolist() if type(his_seq) is not list else his_seq
    [x for x in his_seq if x != '1501_1501_1501']
    return ['_'.join([str(y) for y in x]) for x in his_seq]

metrics = [0] * 11
urls_recall = set()
urls_recall_cnt = dict()


# load pool
def load_pool():
    res = set()
    with open('../data/url_mapping_concat') as ip:
        for line in ip:
            secs = line.strip().split('\t')
            res.add(secs[0])
    return res
#pool_urls = load_pool()
pool_urls = set()


def gini_from_counts(counts):
    """
    counts: 任意长度的一维数组/列表，元素为出现次数
    内部先转成概率再算 Gini
    """
    # ---------- 概率版 Gini ----------
    def gini_prob(p):
        """
        p : array-like，已归一化的概率分布（和为1）
        返回 0~1
        """
        p = np.asarray(p, dtype=float)
        p = np.sort(p)
        n = p.size
        if n <= 1:
            return 0.0
        # 公式：G = (2 * sum_{i=1}^n i * p_i) / n  - (n+1)/n
        return (2 * np.sum(np.arange(1, n + 1) * p)) / n - (n + 1) / n
    p = np.asarray(list(counts), dtype=float)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    return gini_prob(p[p > 0])          # 去掉 0 次项加速


for i, x in enumerate(dataloader):
    if i > 20000:
        break

    inputs = {k:v.to(device) for k, v in x.items()}
    #print(inputs)

    loglik_ids = inputs['logreallike_lmcode'][0]
    logque_ids = inputs['logque_lmcode'][0]
    logses_ids = inputs['logsess_lmcode'][0]

    with torch.no_grad():
        res = model.generate(inputs, group_num=100)
        # res = model.rl_generate(inputs, group_num=100, top_k=10, top_p=0.1, strategy='topp')
    output_ids = res["output_ids"].detach().cpu()
    # log_probs = res["output_logprobs"].sum(dim=-1)

    urls = url_tokenizer.batch_decode(output_ids)
    urls = [x for x in urls if x != '']
    urls_hashid = pt_parser._hash_to_tensor(np.asarray(urls), 20000019).tolist()
    urls_recall.update(urls_hashid)
    for _hid in urls_hashid:
        if _hid not in urls_recall_cnt:
            urls_recall_cnt[_hid] = 0
        urls_recall_cnt[_hid] += 1

    ######## cal recall@k #######
    a, b = cal_hits(parse_his(loglik_ids), parse_his(output_ids))
    metrics[0] += a
    metrics[1] += b
    a, b = cal_hits(parse_his(logque_ids), parse_his(output_ids))
    metrics[2] += a
    metrics[3] += b
    a, b = cal_hits(parse_his(logses_ids), parse_his(output_ids))
    metrics[4] += a
    metrics[5] += b
    metrics[6] = len(urls_recall)

    ## 反解率
    metrics[7] += len(urls)
    metrics[8] += len([url for url in urls if url != ''])
    metrics[9] += len([url for url in urls if url in pool_urls])
    if i % 500 == 0:
        metrics[10] = round(gini_from_counts(urls_recall_cnt.values()), 4)
        print(i, metrics)

    #for x, p, url, hshid in zip(output_ids.tolist(), log_probs.tolist(), urls, urls_hashid):
    #    print(x, p, url, hshid)
    #    print(get_md5_info(url))

print(metrics)

