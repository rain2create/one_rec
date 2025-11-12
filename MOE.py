#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#
#   filename : test.py
#   author   : ***
#   date     : 2025-11-11
#   desc     :
# ========================================================
import math
import torch
from torch import Tensor, nn
from einops import rearrange
from typing import Optional


class TopKGate(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, aux_loss_alpha: float = 1e-2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.alpha = aux_loss_alpha
        self.gate_proj = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: Tensor):
        B, T, D = x.shape
        flat_x = x.view(-1, D)                                    # [B*T, D]
        logits = self.gate_proj(flat_x)                           # [B*T, N]

        # ---- 1. 路由 ----
        top_k_weights, top_k_idx = logits.topk(self.top_k, dim=-1)  # both [B*T, k]
        top_k_weights = top_k_weights.softmax(dim=-1)             # 归一化

        # ---- 2. 计算 aux_loss ----
        router_prob = logits.softmax(dim=-1)                      # [B*T, N]
        # f_i: 实际分到专家的 token 比例
        # 先构造 one-hot 掩码，再平均
        expert_mask = torch.zeros_like(router_prob).scatter_(
            -1, top_k_idx, 1.0)                                   # [B*T, N]
        f = expert_mask.mean(dim=0)                               # [N]
        P = router_prob.mean(dim=0)                               # [N]
        aux_loss = self.alpha * (f * P).sum() * self.num_experts  # 标量

        return top_k_weights, top_k_idx, aux_loss


class MoeFFN(nn.Module):
    """
    完全对齐你原来的 _ff_block 接口：
    input  :  x  [B, T, d_model]
    output :  y  [B, T, d_model]
    """
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int = 2048,
                 activation=nn.GELU,
                 dropout: float = 0.1,
                 num_experts: int = 8,
                 top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. router
        self.gate = TopKGate(d_model, num_experts, top_k)

        # 2. 专家网络：每个专家独立拥有  linear1 + act + drop + linear2
        self.w1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward, bias=True),
                activation(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.w2 = nn.ModuleList([
            nn.Linear(dim_feedforward, d_model, bias=True)
            for _ in range(num_experts)
        ])
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        device = x.device
        flat_x = x.view(-1, D)

        # 1. router 打分
        combine_weights, expert_indices, aux_loss = self.gate(x) # [B*T, k], [B*T, k]

        # 2. 构造一个大的零张量，后面把各专家结果 scatter 进去
        out = torch.zeros(B * T, D, device=device, dtype=x.dtype)

        # 3. 逐专家计算：把属于该专家的 token 挑出来，算完再乘权重加回
        for expert_id in range(self.num_experts):
            mask = expert_indices == expert_id              # [B*T, k]  bool
            selected = mask.any(dim=-1)                     # [B*T]     bool
            if selected.sum() == 0:                         # 没有 token 选这个专家
                continue
            selected_tokens = flat_x[selected]              # [S, D]   S = 选中数
            h = self.w1[expert_id](selected_tokens)         # [S, dim_feedforward]
            h = self.w2[expert_id](h)                       # [S, D]
            # 权重加和：一个 token 可能在 k 个专家里都出现，要累加
            # 先 expand 权重到 [B*T, k, 1]，再 scatter
            weight = combine_weights.masked_fill(~mask, 0.)  # [B*T, k]
            weight = weight.sum(dim=1, keepdim=True)         # [B*T, 1]
            out[selected] += weight[selected] * h

        out = self.dropout3(out)
        return out.view(B, T, D), aux_loss


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    d_model = 768
    seq_len, batch = 128, 4
    x = torch.randn(batch, seq_len, d_model)

    # 原来的稠密 FFN
    dense_ffn = nn.Sequential(
        nn.Linear(d_model, 2048),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(2048, d_model),
        nn.Dropout(0.1)
    )

    # 等价 MoE-FFN：8 专家，Top-2 激活
    moe_ffn = MoeFFN(d_model, dim_feedforward=2048,
                     activation=nn.GELU, dropout=0.1,
                     num_experts=8, top_k=2)

    # 接口完全一致
    y_dense = dense_ffn(x)
    y_moe, aux_loss = moe_ffn(x)
    print("dense out:", y_dense.shape)
    print("moe out  :", y_moe.shape)

    # 参数量对比
    def param_count(m): return sum(p.numel() for p in m.parameters())
    print("dense params:", param_count(dense_ffn))
    print("moe params  :", param_count(moe_ffn))

