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
import sys
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

def reload_model(model_args, data_args, train_args):
    device = torch.device("cuda:0")
    model = RecModel.from_args(model_args)
    ckpt_path = "../data/pt_model/checkpoint-{}.pt".format(train_args.checkpoint_step)
    print(ckpt_path)
    global_step = load_checkpoint(model, None, None, None, ckpt_path, logger, device)
    scripted_model = torch.jit.script(model)
    scripted_model.save("../data/serving/recmodel_new.pt")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    reload_model(model_args, data_args, train_args)

