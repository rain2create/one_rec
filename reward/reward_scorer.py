# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : test_reward.py
#   author   : ***
#   date     : 2025-07-24
#   desc     : 
# ======================================================== 
import tensorflow as tf
import numpy as np
import os
from utils.logger import logger

class RewardScorer:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.score_fn = self.model.signatures['serving_default']
        logger.info(f"模型加载成功: {model_path}")
        
    def predict(self, input_data):
        return self.score_fn(inputs=inputs)

if __name__ == "__main__":
    from transformers import HfArgumentParser
    from parser.tokenizer import UrlTokenizer
    from parser.reward_parser import RewardParser
    from dataset.pt_dataset import PtDataset
    from torch.utils.data import DataLoader
    from arguments.data_args import DataArguments
    from arguments.model_args import ModelArguments
    from arguments.train_args import TrainingArguments
    from arguments.reward_args import RewardArguments
    # 模型路径
    model_path = "/data01/search/lusitong/dev/onerec_reward/data/serving_dir/1753346154"
    
    # 创建scorer实例
    scorer = RewardScorer(model_path)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RewardArguments))
    model_args, data_args, train_args, reward_args = parser.parse_args_into_dataclasses()
    url_tokenizer = UrlTokenizer.from_args(data_args)
    reward_parser = RewardParser(data_args.columns_def, tokenizer=url_tokenizer)
    data_args.file_shuffle, data_args.inter_file_shuffle = False, False # 双pipeline不能shuffle
    dataset = PtDataset.from_args(data_args, parser_func=reward_parser.parser, test_root_dirs=data_args.test_root_dirs)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=data_args.num_workers, pin_memory=True)

    for x in dataloader:
        reward_input = x[1]
        scores = scorer.predict(reward_input)
        print("test:",scores)
        break
    
