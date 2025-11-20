# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : rec_model.py
#   author   : ***
#   date     : 2025-07-15
#   desc     : 
# ======================================================== 
import torch
import torch.nn as nn
from typing import Dict
from utils.logger import logger
from model.sparse_model import SparseModel
from model.encoder import Encoder
from model.decoder import Decoder
from arguments.model_args import ModelArguments

# # 允许 cuBLAS 在 FP32 矩阵乘法中使用 TF32
# Torch.backends.cuda.matmul.allow_tf32 = True
# # 允许 cuDNN 在 FP32 卷积中使用 TF32
# Torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision('high')   # 或 'medium'，效果相同

class RecModel(nn.Module):
    def __init__(self, emb_dim : int,
                        model_dim : int, 
                        feature_vocabs : Dict[str, int], 
                        encoder_num_layers : int, 
                        encoder_num_heads : int, 
                        encoder_dropout : float, 
                        encoder_max_seq_len : int, 
                        encoder_pre_norm : bool,
                        decoder_num_layers : int, 
                        decoder_num_heads : int, 
                        decoder_dropout : float, 
                        decoder_codebook_layers : int,
                        decoder_codebook_size : int, 
                        decoder_pre_norm : bool,
                        decoder_ignore_idx : int,
                        decoder_bos_token_id : int,
                        decoder_temperature : float,
                        decoder_weight_tying : bool
                    ):

        super(RecModel, self).__init__()
        self.sparse_model = SparseModel(emb_dim = emb_dim, 
                                        feature_vocabs=feature_vocabs)

        self.encoder = Encoder(emb_dim = emb_dim, 
                                model_dim = model_dim,
                                num_layers=encoder_num_layers, 
                                num_heads=encoder_num_heads, 
                                dropout=encoder_dropout, 
                                max_seq_len=encoder_max_seq_len, 
                                pre_norm=encoder_pre_norm)
 
        self.decoder = Decoder(model_dim = model_dim, 
                                layer_num=decoder_num_layers, 
                                head_num=decoder_num_heads, 
                                codebook_size=decoder_codebook_size, 
                                codebook_layers=decoder_codebook_layers, 
                                dropout=decoder_dropout,
                                pre_norm=decoder_pre_norm,
                                ignore_idx=decoder_ignore_idx,
                                bos_token_id=decoder_bos_token_id,
                                temperature=decoder_temperature,
                                weight_tying=decoder_weight_tying)

    def forward(self, features: Dict[str, torch.Tensor]):
        feature_embeddings = self.sparse_model(features)
        att_emb, att_mask = self.encoder(features, feature_embeddings)
        output_logits, loss, mask_valid, final_valid, probs = self.decoder(features, att_emb, att_mask)
        return output_logits, loss, mask_valid, final_valid, probs

    @torch.jit.export
    def rl_generate(self, features: Dict[str, torch.Tensor], group_num: int = 1, top_p: float = 0.9, top_k: int = 10, strategy: str = "topk"):
        feature_embeddings = self.sparse_model(features)
        att_emb, att_mask = self.encoder(features, feature_embeddings)
        output_rl = self.decoder.rl_generate(att_emb, att_mask, group_num, top_p, top_k, strategy)
        return output_rl

    @torch.jit.export
    def bs_rl_generate(self, features: Dict[str, torch.Tensor], group_num: int = 1):
        feature_embeddings = self.sparse_model(features)
        att_emb, att_mask = self.encoder(features, feature_embeddings)
        output_rl = self.decoder.bs_rl_generate(att_emb, att_mask, group_num)
        return output_rl
    
    @torch.jit.export
    def generate(self, features: Dict[str, torch.Tensor], group_num: int = 1):
        feature_embeddings = self.sparse_model(features)
        att_emb, att_mask = self.encoder(features, feature_embeddings)
        output_rl = self.decoder.bs_rl_generate(att_emb, att_mask, group_num)
        # output_rl = self.decoder.rl_generate(att_emb, att_mask, group_num=group_num)
        return output_rl

    @torch.jit.export
    def compute_sequence_log_probs(self,
                                  features: Dict[str, torch.Tensor],
                                  target_ids: torch.Tensor,
                                  group_num: int) -> torch.Tensor: 
        feature_embeddings = self.sparse_model(features)
        att_emb, att_mask = self.encoder(features, feature_embeddings)
        att_emb = att_emb.repeat_interleave(group_num, dim=0)
        att_mask = att_mask.repeat_interleave(group_num, dim=0)
        return self.decoder.compute_log_probs(att_emb, att_mask, target_ids)

    @classmethod
    def from_args(cls, args: ModelArguments):
        return cls(
            emb_dim=args.emb_dim,
            model_dim=args.model_dim,
            feature_vocabs=args.feature_vocabs,
            encoder_num_layers=args.encoder_num_layers,
            encoder_num_heads=args.encoder_num_heads,
            encoder_dropout=args.encoder_dropout,
            encoder_max_seq_len=args.encoder_max_seq_len,
            encoder_pre_norm=args.encoder_pre_norm,
            decoder_num_layers=args.decoder_num_layers,
            decoder_num_heads=args.decoder_num_heads,
            decoder_dropout=args.decoder_dropout,
            decoder_codebook_layers=args.decoder_codebook_layers,
            decoder_codebook_size=args.decoder_codebook_size,
            decoder_pre_norm=args.decoder_pre_norm,
            decoder_ignore_idx=args.decoder_ignore_idx,
            decoder_bos_token_id=args.decoder_bos_token_id,
            decoder_temperature=args.decoder_temperature,
            decoder_weight_tying=args.decoder_weight_tying
        )

if __name__ == "__main__":
    from arguments.model_args import ModelArguments
    from arguments.data_args import DataArguments
    from transformers import HfArgumentParser
    from parser.tokenizer import UrlTokenizer
    from parser.pt_parser import PtParser
    from dataset.pt_dataset import PtDataset
    from torch.utils.data import DataLoader

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model = RecModel.from_args(model_args)
    print(model)

    url_tokenizer = UrlTokenizer.from_args(data_args)
    pt_parser: PtParser = PtParser(data_args.columns_def, tokenizer=url_tokenizer)

    # 创建数据集
    dataset: PtDataset = PtDataset.from_args(data_args, parser_func=pt_parser.parser)

    # 创建DataLoader
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=None,   # 因为我们在dataset中已经分batch
        num_workers=10      # 因为在dataset内部以及定义了子进程
    )

    for x in dataloader:
        output_logits, loss = model(x)
        print("output_logits", output_logits.shape)
        print(output_logits)
        print("loss", loss)
        break
    
