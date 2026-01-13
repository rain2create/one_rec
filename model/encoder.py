# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : encoder.py
#   author   : ***
#   date     : 2025-07-12
#   desc     : 
# ======================================================== 
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from utils.logger import logger

class Encoder(nn.Module):
    """
    使用Transformer对用户序列特征进行编码，输出embedding
    """
    def __init__(
        self,
        emb_dim: int,
        model_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pre_norm: bool = True,

    ):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.dropout = dropout

        # self.user_features = ['uid', 'sign', 'autok', 'autoplay', 'reqtimes']
        self.user_features = ['sign', 'autok', 'autoplay', 'reqtimes']
        self.logsess_features = ['md5', 'qid', 'videotag', 'timegap', 'duration', 'dwell', 'label']
        self.logqueue_features = self.logsess_features
        self.logreallike_features = ['md5', 'qid', 'videotag', 'duration']

        # dense(leakyrelu(dense))
        self.user_proj = nn.Sequential(
            nn.Linear(len(self.user_features) * self.emb_dim, self.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.model_dim, self.model_dim)
        )

        # dense(leakyrelu(dense))
        self.logsess_proj = nn.Sequential(
            nn.Linear(len(self.logsess_features) * self.emb_dim + self.emb_dim * 2, self.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.model_dim, self.model_dim)
        )

        # dense(leakyrelu(dense))
        self.logqueue_proj = nn.Sequential(
            nn.Linear(len(self.logqueue_features) * self.emb_dim + self.emb_dim * 2, self.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.model_dim, self.model_dim)
        )

        # dense(leakyrelu(dense))
        self.logreallike_proj = nn.Sequential(
            nn.Linear(len(self.logreallike_features) * self.emb_dim + self.emb_dim * 2, self.model_dim),
            nn.LeakyReLU(),
            nn.Linear(self.model_dim, self.model_dim)
        )

        # 位置编码 
        # self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, max_seq_len, model_dim).uniform_(-0.02, 0.02))

        # 输出归一化
        self.norm = nn.LayerNorm(self.model_dim)

    def _get_mask(self, features: Dict[str, torch.Tensor]):
        """
        Args:
            features: Dict[str, torch.Tensor]
        Returns:
            mask: shape (batch_size, seq_len)
        """
        # 获取logsess_features的mask
        logsess_mask = features['logsess_mask']
        # 获取logqueue_features的mask
        logqueue_mask = features['logque_mask']
        # 获取logreallike_features的mask
        logreallike_mask = features['logreallike_mask']
        masks = torch.cat([torch.ones(logsess_mask.size(0), 1, device=logsess_mask.device), logsess_mask, logqueue_mask, logreallike_mask], dim=1)
        return masks

    def forward(self, features: Dict[str, torch.Tensor], feature_embeddings: List[torch.Tensor]):
        """
        Args:
            features: Dict[str, torch.Tensor]
        Returns:
            embedding: shape (batch_size, seq_len, embed_dim)
        """
        user_features, logsess_features, logqueue_features, logreallike_features = feature_embeddings

        # [batch_size, len(user_features) * model_dim]
        # [batch_size, model_dim] -> [batch_size, 1, model_dim]
        user_features = self.user_proj(user_features).unsqueeze(1)

        # [batch_size, 30, len(logsess_features) * model_dim]
        logsess_features = self.logsess_proj(logsess_features)

        # [batch_size, 30, len(logqueue_features) * model_dim]
        logqueue_features = self.logqueue_proj(logqueue_features)

        # [batch_size, 30, len(logreallike_features) * model_dim]
        logreallike_features = self.logreallike_proj(logreallike_features)

        # [batch_size, 1 + 30 + 30 + 30, model_dim]        
        inputs = torch.cat([user_features, logsess_features, logqueue_features, logreallike_features], dim=1)
        seq_len = inputs.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(f"输入序列长度 {seq_len} 超过最大支持长度 {self.max_seq_len}")

        # 加位置编码
        pos_emb = self.pos_embedding[:, :seq_len, :]
        inputs = inputs + pos_emb

        attn_mask = self._get_mask(features)
        # TransformerEncoder期望mask为True表示被mask（即pad），False为有效
        # [batch_size, 1 + 30 + 30 + 30]
        attn_mask = ~attn_mask.to(torch.bool)

        # Transformer编码
        x = self.norm(inputs)
        return x, attn_mask

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
    from model.sparse_model import SparseModel
    
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
    
    device = "cuda:0"

    # 创建sparse model
    sparse_model = SparseModel(
        emb_dim=model_args.emb_dim,
        feature_vocabs=model_args.feature_vocabs
    )
    sparse_model.to(device)
    sparse_model.eval()
    print(sparse_model)

    # 创建model
    model = Encoder(
        emb_dim=model_args.emb_dim,
        model_dim=model_args.model_dim,
        num_layers=model_args.encoder_num_layers,
        num_heads=model_args.encoder_num_heads,
        dropout=model_args.encoder_dropout,
        max_seq_len=model_args.encoder_max_seq_len,
        pre_norm=model_args.encoder_pre_norm
    )

    model.to(device)
    model.eval()
    print(model)

    # 训练循环
    for x in dataloader:
        x = {k: v.to(device) for k,v in x.items()}
        # att_mask: True表示mask，False表示有效
        feature_embeddings = sparse_model(x)
        print(feature_embeddings)
        att_emb, att_mask = model(x, feature_embeddings)
        print(att_mask)
        print(feature_embeddings)
        break
