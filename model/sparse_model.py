# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : sparse_model.py
#   author   : ***
#   date     : 2025-07-25
#   desc     : 
# ======================================================== 


## 主要是在做embedding！！！ 主要是输入encoder-decoder结构前的embedding层
## 这里的行为序列特征码本已经训练好，但输入主模型需要单独的embedding层,即需要重新做embedding！！！
## 需要搞懂码本训练过程！！！
## 注意这里行为序列特征是存储了30个历史行为序列

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from utils.logger import logger

class SparseModel(nn.Module):
    """
    将 sparse=True 的 embedding 独立出来, 方便后续使用ddp判断是否需要同步，以及定义单独的optimizer
    """
    def __init__(
        self,
        emb_dim: int,
        feature_vocabs: Dict[str, int]
    ):
        super(SparseModel, self).__init__()
        self.emb_dim = emb_dim
        self.feature_vocabs = feature_vocabs

        self.feature_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, self.emb_dim, sparse=True)
                for name, vocab_size in feature_vocabs.items()
            }
        )  ## 拿对应特征来做embedding  不是共享的emb！！！！

        ## 每一层码本都要emb，每一层都是codebook大小都是1502，一共有三层
        self.code_books_embeddings = nn.ModuleList(
            [nn.Embedding(1502, self.emb_dim, sparse=True) for i in range(3)]
        )  ## 码本单独做embedding  这里也是每层都是单独做


        ## 行为序列用码本编译后 再过embedding encode，其他特征直接embedding
        self.user_features = ['sign', 'autok', 'autoplay', 'reqtimes']
        self.logsess_features = ['lmcode', 'qid', 'videotag', 'timegap', 'duration', 'dwell', 'label']
        self.logqueue_features = self.logsess_features
        self.logreallike_features = ['lmcode', 'qid', 'videotag', 'duration']

    def _get_embedding(self, name: str, x: torch.Tensor) -> torch.Tensor:
        # 用 if-elif 静态分支，TorchScript 能编译
        if name == "sign":
            return self.feature_embeddings["sign"](x)
        elif name == "autok":
            return self.feature_embeddings["autok"](x)
        elif name == "autoplay":
            return self.feature_embeddings["autoplay"](x)
        elif name == "reqtimes":
            return self.feature_embeddings["reqtimes"](x)
        elif name == "lmcode":
            a = self.code_books_embeddings[0](x[..., 0])  ## 拿第一个码本x，取第一层emb
            b = self.code_books_embeddings[1](x[..., 1])  ## 拿第二个码本x，取第二层emb
            c = self.code_books_embeddings[2](x[..., 2])  ## 拿第三个码本x，取第三层emb
            return torch.concat([a, b, c], -1)  ## 横着拼一起
        elif name == "qid":
            return self.feature_embeddings["qid"](x)
        elif name == "videotag":
            return self.feature_embeddings["videotag"](x)
        elif name == "timegap":
            return self.feature_embeddings["timegap"](x)
        elif name == "duration":
            return self.feature_embeddings["duration"](x)
        elif name == "dwell":
            return self.feature_embeddings["dwell"](x)
        elif name == "label":
            return self.feature_embeddings["label"](x)
        else:
            raise ValueError(f"Unsupported feature: {name}")
 
    def forward(self, features: Dict[str, torch.Tensor]):
        """
        Args:
            features: Dict[str, torch.Tensor]
        Returns:
            embedding: shape (batch_size, seq_len, embed_dim)
        """

        #### 最后产出的大维度特征都是拼接后的特征 ！!

        # [batch_size, len(user_features) * emb_dim]  
        user_features = torch.cat([self._get_embedding(name, features[name]).to_dense() for name in self.user_features], dim=1)


        ## 以下为行为序列特征 长度30 即一个对应loss类的特征 有30个过去的历史行为 !!!!!

        # [batch_size, 30, len(logsess_features) * emb_dim+2]  因为lmcode是3个码本拼接 所以+2
        logsess_features = torch.cat([self._get_embedding(name, features["logsess_" + name]).to_dense() for name in self.logsess_features], dim=2)

        # [batch_size, 30, len(logqueue_features) * emb_dim+2]  因为lmcode是3个码本拼接 所以+2
        logqueue_features = torch.cat([self._get_embedding(name, features["logque_" + name]).to_dense() for name in self.logqueue_features], dim=2)

        # [batch_size, 30, len(logreallike_features) * emb_dim+2]  因为lmcode是3个码本拼接 所以+2
        logreallike_features = torch.cat([self._get_embedding(name, features["logreallike_" + name]).to_dense() for name in self.logreallike_features], dim=2)
        
        return [user_features, logsess_features, logqueue_features, logreallike_features]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.pt_dataset import PtDataset
    from collections import defaultdict
    from parser.pt_parser import PtParser
    from parser.tokenizer import UrlTokenizer
    from transformers import HfArgumentParser
    from pathlib import Path
    from arguments.data_args import DataArguments
    from arguments.model_args import ModelArguments

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # 创建PtParser实例
    url_tokenizer = UrlTokenizer.from_args(data_args)
    pt_parser: PtParser = PtParser(data_args.columns_def, tokenizer=url_tokenizer)

    # 创建数据集
    dataset = PtDataset.from_args(data_args, parser_func=pt_parser.parser)

    # 创建DataLoader
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=None,   # 因为我们在dataset中已经分batch
        num_workers=10      # 因为在dataset内部以及定义了子进程
    )
    
    device = "cuda:1"
    # 创建model
    model = SparseModel(
        emb_dim=model_args.emb_dim,
        feature_vocabs=model_args.feature_vocabs,
    )

    model.to(device)
    model.eval()
    print(model)

    # 训练循环
    for x in dataloader:
        x = {k: v.to(device) for k,v in x.items()}
        # att_mask: True表示mask，False表示有效
        feature_embeddings = model(x)
        print(feature_embeddings)
        break
