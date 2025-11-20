# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : decoder.py
#   author   : ***
#   date     : 2025-07-15
#   desc     : 
# ======================================================== 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from model.decoder_moe import DecoderMOE, DecoderLayerMOE
from model.moe_auxloss import MANAGER
from utils.logger import logger
from utils.metrics import MeanMetric


class Decoder(nn.Module):
    """
    生成target item 的 semantic id
    """
    def __init__(
        self,
        model_dim: int,
        layer_num: int,
        head_num: int,
        codebook_size: int,
        codebook_layers: int = 3, 
        bos_token_id: int = 0,
        dropout: float = 0.1,
        ignore_idx: int = 1501,
        temperature: float = 1.0,
        weight_tying: bool = True,
        pre_norm: bool = True
    ):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.layer_num = layer_num
        self.bos_token_id = bos_token_id
        self.num_heads = head_num
        self.codebook_size = codebook_size
        self.codebook_layers = codebook_layers
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.temperature = temperature
        self.weight_tying = weight_tying
        self.seqs_len = self.codebook_layers + 1
        self.ignore_idx = ignore_idx

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, reduction='none')
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

        # embeddings为nn.ModuleList，序列中每个位置使用不同的embedding layer
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=self.codebook_size+2, embedding_dim=self.model_dim)
            for i in range(self.codebook_layers+1)
        ])

        """
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=self.pre_norm
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.layer_num)
        """

        decoder_layer = DecoderLayerMOE(
            num_experts=24,
            top_k=2,
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=self.pre_norm,
            enable_moe=True,
        )
        self.transformer_decoder = DecoderMOE(decoder_layer, num_layers=self.layer_num)

        # output_layer为nn.ModuleList，每个位置一个Linear，权重与对应的embeddings共享
        self.output_layers = nn.ModuleList([
            nn.Linear(self.model_dim, self.codebook_size+2, bias=False) for _ in range(self.codebook_layers)
        ])
        if self.weight_tying:
            for i in range(self.codebook_layers):
                self.output_layers[i].weight = self.embeddings[i+1].weight

        if not torch.jit.is_scripting():
            self.loss_rl = MeanMetric()
            self.loss_ct = MeanMetric()
            self.pos_pi_new = MeanMetric()
            self.neg_pi_new = MeanMetric()
            self.cts_pi_new = MeanMetric()

    def _generate_square_subsequent_mask(self, sz: int):
        """Generate a square subsequent mask (upper triangular -inf)."""
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def _compute_output_logits(self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        TorchScript 兼容版本，不使用 ModuleList 动态索引
        Args:
            encoder_output: [B, seq_enc, D]
            encoder_mask: [B, seq_enc]
            input_ids: [B, seq_dec] -> token ids for decoder input
        Returns:
            output_logits: [B, seq_dec, V]
        """
        B, slen = input_ids.size()
        D = self.model_dim
        V = self.output_layers[0].weight.size(0)

        # 1. tgt mask
        tgt_mask_ = self._generate_square_subsequent_mask(slen).to(input_ids.device)

        # 2. 获取 embeddings 权重堆叠成 Tensor: [L, V, D]
        emb_weights = torch.stack([emb.weight for emb in self.embeddings])  # [max_len, V_emb, D]
        input_embs = torch.empty(B, slen, D, device=input_ids.device)
        for i in range(slen):
            input_embs[:, i, :] = F.embedding(input_ids[:, i], emb_weights[i])

        # 3. transformer decoder forward
        decoder_out = self.transformer_decoder(
            input_embs, encoder_output, tgt_mask=tgt_mask_, memory_key_padding_mask=encoder_mask
        )  # [B, slen, D]

        # 4. 输出层权重堆叠成 Tensor: [L, V, D]
        out_weights = torch.stack([layer.weight for layer in self.output_layers])  # [max_len, V, D]

        # 5. 逐步计算 logits
        output_list = []
        for i in range(slen):
            emb = F.normalize(decoder_out[:, i, :], dim=-1)          # [B, D]
            weight = F.normalize(out_weights[i], dim=-1).transpose(0, 1)  # [D, V]
            output_list.append(torch.matmul(emb, weight) * self.temperature)
            """            
            emb = decoder_out[:, i, :]
            weight = out_weights[i].transpose(0, 1)
            output_list.append(torch.matmul(emb, weight))
            """
        output_logits = torch.stack(output_list, dim=1)  # [B, slen, V]
        return output_logits

    @torch.jit.export
    def _sample_from_logits(self, 
                          logits: torch.Tensor,
                          strategy: str = "topp",
                          top_p: float = 0.9,
                          top_k: int = 50) -> torch.Tensor:
        """
        返回:
            next_token: 采样结果 [B]
        """
        
        if strategy == "greedy":
            next_token = torch.argmax(logits, dim=-1)
        elif strategy == "topk":
            topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            next_token = topk_indices.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        elif strategy == "topp":
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # 创建top_p掩码
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            # 过滤低概率token
            filtered_probs = probs.masked_fill(mask, 0.0)
            next_token = torch.multinomial(filtered_probs, 1).squeeze(-1)
            next_token = sorted_indices.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return next_token

    @torch.jit.export
    def compute_log_probs(self,
                         encoder_output: torch.Tensor,
                         encoder_mask: torch.Tensor,
                         target_ids: torch.Tensor) -> torch.Tensor:
        """
        计算给定序列在当前decoder下的对数概率
        
        Args:
            encoder_output: 编码器输出 [batch_size, seq_len, model_dim]
            encoder_mask: 编码器mask [batch_size, seq_len]
            target_ids: 目标序列 [batch_size, codebook_layers]
        
        Returns:
            log_probs: 序列的对数概率 [batch_size, codebook_layers]
        """

        B = target_ids.size()[0]
        input_ids = torch.cat([
            torch.full([B, 1], self.bos_token_id, dtype = torch.long, device = target_ids.device), target_ids[:, 0 : -1]], dim = 1)

        output_logits = self._compute_output_logits(encoder_output, encoder_mask, input_ids)     #[batch, 3, V]
        log_probs_all = F.log_softmax(output_logits, dim=-1)                                     #[batch, 3, V]
        target_probs  = torch.gather(log_probs_all, 2, target_ids.unsqueeze(2)).squeeze(-1)      #[batch, 3]
        return target_probs

    def rl_loss(self, features: Dict[str, torch.Tensor], pi_new: torch.Tensor):
        pi_safe = torch.where(
                features['advantages'] >= 0.0, 
                torch.maximum(features['pi_old'], pi_new.detach()),
                torch.maximum(features['pi_old'], 1.0 - pi_new.detach())
        )
        ratio = pi_new / pi_safe
        # print(
        #         'adv', features['advantages'][:3], 
        #         'pin', pi_new[:3],
        #         'pis', pi_safe[:3],
        #         'rat', ratio[:3],
        #         'adv', features['advantages'][-3:],
        #         'pin', pi_new[-3:],
        #         'pis', pi_safe[-3:],
        #         'rat', ratio[-3:]
        # )
        loss = -(ratio * features['advantages']).mean()
        pos_pi_new = torch.where(features['advantages'] > 0, pi_new, 0.0).sum() / (torch.sum(features['advantages'] > 0) + 1e-6)
        neg_pi_new = torch.where(features['advantages'] < 0, pi_new, 0.0).sum() / (torch.sum(features['advantages'] < 0) + 1e-6)
        cts_pi_new = torch.where(features['advantages'] == 0, pi_new, 0.0).sum() / (torch.sum(features['advantages'] == 0) + 1e-6)
        self.pos_pi_new(pos_pi_new)
        self.neg_pi_new(neg_pi_new)
        self.cts_pi_new(cts_pi_new)
        return loss

    def forward(self, features: Dict[str, torch.Tensor], encoder_output: torch.Tensor, encoder_mask: torch.Tensor):
        """
        features: Dict[str, torch.Tensor]
        encoder_output: torch.Tensor [batch_size, en_seqs_len, model_dim]
        encoder_mask: torch.Tensor [batch_size, en_seqs_len] True表示被mask（即pad），False为有效

        return:
            output_logits: torch.Tensor [batch_size, seqs_len, codebook_size+2]
            loss: torch.Tensor 
        """

        # shape: [batch_size, 4]
        url_ids = features["url"]
        seqs_len = url_ids.shape[1]
        output_logits = self._compute_output_logits(encoder_output, encoder_mask, url_ids[:, 0 : -1])  #[batch, codebook_layers, V]

        labels = url_ids[:, 1:]
        batch_size, seq_len_minus1, vocab_size = output_logits.shape

        output_logits = output_logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)

        # 注意这里的mask: 1表示有效，0表示mask
        mask = features["mask"]
        mask_expanded = mask.unsqueeze(1).expand(-1, seqs_len-1).reshape(-1).to(torch.bool)
        masked_labels = labels.clone()
        masked_labels[~mask_expanded] = self.ignore_idx # 设置无效位置忽略

        if (masked_labels != self.ignore_idx).any():
            loss = self.loss_fn(output_logits, masked_labels)

            # v1 对比 + 权重融合
            # # handle neg loss
            # loss_neg = torch.log(1e-6 + 1.0 - torch.exp(loss * -1.0))
            # w = features['sample_weights'].repeat_interleave(3, dim=0)
            # loss = torch.where(w > 0.0, loss, loss_neg) * torch.sign(w)
            # # 跟进 th 做截断, hard negative
            # t = features['loss_th'].repeat_interleave(3, dim=0)
            # loss = torch.where(loss > t, loss, 0.0)
            # loss = (loss * torch.abs(w)).mean()

            if not torch.jit.is_scripting():
                # v2, 只有对比的部分
                loss_neg = - torch.log(1e-6 + 1.0 - torch.exp(loss * -1.0))
                s = features['sample_weights'].repeat_interleave(3, dim=0)
                loss = (torch.where(s > 0.0, loss, loss_neg)).mean()
                self.loss_ct(loss)

            # 专家负载均衡loss
            if not torch.jit.is_scripting():
                aux_loss = 0.01 * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()
                loss = loss + aux_loss
        else:
            # 返回0损失，保持梯度流通
            loss = (output_logits * 0.0).mean()

        output_logits = output_logits.view(batch_size, seq_len_minus1, vocab_size)
        labels = labels.view(batch_size, seq_len_minus1)
        log_probs = F.log_softmax(output_logits, dim=-1)
        # [batch_size]
        probs = torch.exp(torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1).sum(dim=-1))

        if not torch.jit.is_scripting():
            # v2, 强化部分，通过advantage 影响目标
            rl_loss = self.rl_loss(features, probs)
            if not torch.jit.is_scripting():
                if loss > 0.0:
                    loss = loss + rl_loss
                    self.loss_rl(rl_loss)

        # valid prob 
        mask_valid = mask.sum()
        final_valid = torch.where(masked_labels == self.ignore_idx, 0., 1.).sum() / (seqs_len-1)
        batch_valid = torch.where(masked_labels.view(batch_size, seq_len_minus1) == self.ignore_idx, 0., 1.).sum(dim=1) / (seqs_len - 1)
        probs = torch.sum(probs * batch_valid) / (torch.sum(batch_valid) + 1e-6)
        return output_logits, loss, mask_valid, final_valid, probs

    @torch.jit.export
    def rl_generate(self,
                   encoder_output: torch.Tensor,
                   encoder_mask: torch.Tensor,
                   group_num: int,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   strategy: str = "topp") -> Dict[str, torch.Tensor]:
        B = encoder_output.size(0)
        device = encoder_output.device

        # 使每个原始输入对应 group_num 个独立采样序列
        encoder_output = encoder_output.repeat_interleave(group_num, dim=0)  # [B*group_num, seq_len, hidden]
        encoder_mask   = encoder_mask.repeat_interleave(group_num, dim=0)    # [B*group_num, seq_len]
        new_B = B * group_num  # 新的batch size

        # 初始化bos_token
        input_ids = torch.full((new_B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for step in range(self.codebook_layers):
            # 1. 计算对应位置的预测logits
            output_logits = self._compute_output_logits(encoder_output, encoder_mask, input_ids)     #[batch, curlen, V]

            # 2. 获取最后一个位置的logits
            last_logits = output_logits[:, -1, :]       # [new_b, V]
            
            # 3. 根据logits采样得到token_id
            next_token = self._sample_from_logits(last_logits, strategy=strategy, top_p=top_p, top_k=top_k + step * 20)    #[new_b]
    
            # 4. 将新采样得到的tokenid放入input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)  

        # 组装结果
        output_ids = input_ids[:, 1:]

        return {
            "output_ids": output_ids,
        }

    @torch.jit.export
    def bs_rl_generate(self,
                       encoder_output: torch.Tensor,
                       encoder_mask: torch.Tensor,
                       group_num: int = 5) -> Dict[str, torch.Tensor]:
        B = encoder_output.size(0)
        device = encoder_output.device
        new_B = B * group_num
     
        # 只缓存 token ids，不缓存 embeddings
        input_ids = torch.full((new_B, 1), self.bos_token_id, dtype=torch.long, device=device)
        logprobs = torch.full((new_B, 1), 0., dtype=torch.float32, device=device)
        scores = torch.zeros(new_B, device=device)

        # repeat encoder_output & mask by group_size
        encoder_output = encoder_output.repeat_interleave(group_num, dim=0)
        encoder_mask = encoder_mask.repeat_interleave(group_num, dim=0)
     
        for step in range(self.codebook_layers):
            # 1. 计算对应位置的预测logits
            output_logits = self._compute_output_logits(encoder_output, encoder_mask, input_ids)     #[batch, curlen, V]
            last_logits = output_logits[:, -1, :]       # [new_b, V]

            # 温度缩放，解决前两级别锁死的问题
            last_logits = last_logits / last_logits.std()

            # 2. 扩展候选
            log_probs_step = F.log_softmax(last_logits, dim=-1)
            topk_log_probs, topk_tokens = torch.topk(log_probs_step, group_num, dim=-1)  # [new_B, group_num]
     
            # 3. 计算新分数,累计log_probs
            current_scores = scores.view(B, group_num)
            new_scores = current_scores.unsqueeze(2) + topk_log_probs.view(B, group_num, group_num)
            new_scores = new_scores.view(B, -1)  # [B, group_num*group_num]
            
            if step != 0:
                top_scores, top_indices = torch.topk(new_scores, group_num, dim=1)  # [B, group_num]
            else:
                top_scores = new_scores[:, :group_num].clone()  # 或 .contiguous()
                top_indices = torch.arange(group_num, device=device).expand(B, group_num)

            # 4. 计算重排索引
            old_beam_indices = top_indices // group_num  # [B, group_num]
            global_old_beam_indices = (torch.arange(B, device=device) * group_num).unsqueeze(1) + old_beam_indices
            global_old_beam_indices = global_old_beam_indices.view(-1)  # [B*group_num]
     
            # 5. 重排状态
            input_ids = input_ids[global_old_beam_indices]  # [B*group_num, seq_len]
            logprobs = logprobs[global_old_beam_indices]
            scores = top_scores.view(-1)  # [B*group_num]
     
            # 6. 获取选中的 token
            topk_tokens_flat = topk_tokens.view(B, -1)  # [B, group_num*group_num]
            selected_tokens = topk_tokens_flat.gather(1, top_indices).view(-1)  # [B*group_num]
     
            topk_log_probs_flat = topk_log_probs.view(B, -1)
            selected_log_probs = topk_log_probs_flat.gather(1, top_indices).view(-1)  # [B*group_num]
            # 7. 更新 input_ids
            input_ids = torch.cat([input_ids, selected_tokens.unsqueeze(1)], dim=1)  # [B*group_num, seq_len+1]
            logprobs = torch.cat([logprobs, selected_log_probs.unsqueeze(1)], dim=1)  # [B*group_num, seq_len+1]

        # 12. 输出
        output_ids = input_ids[:, 1:]  # 去掉 bos
        log_probs = logprobs[:, 1:]
     
        return {
            "output_ids": output_ids,
            "output_logprobs": log_probs
        }


   

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.pt_dataset import PtDataset
    from parser.pt_parser import PtParser
    from parser.tokenizer import UrlTokenizer
    from model.encoder import Encoder
    from model.sparse_model import SparseModel

    # 创建PtParser实例
    from arguments.data_args import DataArguments
    from arguments.model_args import ModelArguments
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    data_args.file_shuffle = False
    data_args.inter_file_shuffle = False
    data_args.reward_stage = True
    data_args.reward_batch_size = 1

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
    
    device = "cuda:1"

    # 创建sparse model
    sparse_model = SparseModel(
        emb_dim=model_args.emb_dim,
        feature_vocabs=model_args.feature_vocabs
    )
    sparse_model.to(device)
    sparse_model.eval()

    # 创建model
    encode_model = Encoder(
        emb_dim=model_args.emb_dim,
        model_dim=model_args.model_dim,
        num_layers=model_args.encoder_num_layers,
        num_heads=model_args.encoder_num_heads,
        dropout=model_args.encoder_dropout,
        max_seq_len=model_args.encoder_max_seq_len,
        pre_norm=model_args.encoder_pre_norm
    )

    encode_model.to(device)
    encode_model.train()

    decode_model = Decoder(
        model_dim=model_args.model_dim, 
        layer_num=model_args.decoder_num_layers, 
        head_num=model_args.decoder_num_heads, 
        codebook_size=model_args.decoder_codebook_size,
        codebook_layers=model_args.decoder_codebook_layers,
        bos_token_id=model_args.decoder_bos_token_id,
        dropout=model_args.decoder_dropout,
        ignore_idx=model_args.decoder_ignore_idx,
        temperature=model_args.decoder_temperature,
        pre_norm=model_args.decoder_pre_norm,
        weight_tying=model_args.decoder_weight_tying
    )

    decode_model.to(device)
    decode_model.train()

    for x in dataloader:
        x = {k: v.to(device) for k, v in x.items()}
        feature_embeddings = sparse_model(x)
        att_emb, att_mask = encode_model(x, feature_embeddings)
        print("att_emb", att_emb.shape)
        print("att_mask", att_mask.shape)

        print("\n" + "="*50)
        print("Starting consistency test between rl_generate and compute_log_probs")
        print("="*50)

        # 设置测试参数
        group_num = 3  # 每个样本生成3个序列
        codebook_layers = model_args.decoder_codebook_layers

        # 步骤1: 使用rl_generate生成序列
        print("\nStep 1: Generating sequences with rl_generate...")
        rl_result = decode_model.rl_generate(
            encoder_output=att_emb,
            encoder_mask=att_mask,
            group_num=group_num,
            top_p=0.9,
            temperature=1.0,
            top_k=50,
            strategy="topp"
        )
        output_ids_rl = rl_result["output_ids"]  # [B*group_num, codebook_layers]
        log_probs_rl = rl_result["log_probs"]    # [B*group_num, codebook_layers]

        # 打印生成结果示例
        print(f"\nGenerated sequences (first sample, first {min(5, group_num)} beams):")
        print(output_ids_rl[:group_num, :])
        print(f"Shape of generated sequences: {output_ids_rl.shape}")
        print(f"Shape of log_probs from rl_generate: {log_probs_rl.shape}")

        # 步骤2: 重复encoder输出以匹配生成序列的batch size
        print("\nStep 2: Preparing repeated encoder outputs for recomputation...")
        B = att_emb.size(0)
        encoder_output_repeated = att_emb.repeat_interleave(group_num, dim=0)
        encoder_mask_repeated = att_mask.repeat_interleave(group_num, dim=0)

        print(f"Repeated encoder output shape: {encoder_output_repeated.shape}")
        print(f"Repeated mask shape: {encoder_mask_repeated.shape}")

        # 步骤3: 使用compute_log_probs重新计算log概率
        print("\nStep 3: Recomputing log probabilities with compute_log_probs...")
        with torch.no_grad():  # 避免不必要的梯度计算
            log_probs_recomputed = decode_model.compute_log_probs(
                encoder_output=encoder_output_repeated,
                encoder_mask=encoder_mask_repeated,
                target_ids=output_ids_rl
            )
        print(log_probs_rl, log_probs_recomputed)
        ratio = torch.exp(log_probs_recomputed.sum(dim=1) - log_probs_rl.detach().sum(dim=1))
        print(f"ratio: {ratio}")
        print(f"Shape of recomputed log_probs: {log_probs_recomputed.shape}")

        # 步骤4: 比较两个log概率
        print("\nStep 4: Comparing log probabilities...")

        # 计算差异
        abs_diff = (log_probs_rl - log_probs_recomputed).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # 检查是否在浮点误差范围内一致
        is_consistent = torch.allclose(log_probs_rl, log_probs_recomputed, atol=1e-5, rtol=1e-5)

        # 打印结果
        print(f"\nConsistency check result: {'PASSED' if is_consistent else 'FAILED'}")
        print(f"Max absolute difference: {max_diff:.8f}")
        print(f"Mean absolute difference: {mean_diff:.8f}")
        break
