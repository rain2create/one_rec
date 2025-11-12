#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : a.py
#   author   : ***
#   date     : 2025-11-11
#   desc     : 
# ======================================================== 
import torch
import math
import torch.nn as nn
from typing import Tuple


class TopKGate(nn.Module):

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate_proj = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        flat_x = x.view(-1, D)                                    # [B*T, D]
        logits = self.gate_proj(flat_x)                           # [B*T, N]

        # ---- 1. 路由 ----
        top_k_weights, top_k_idx = logits.topk(self.top_k, dim=-1)  # both [B*T, k]
        top_k_weights = top_k_weights.softmax(dim=-1)             # 归一化

        # ---- 2. 计算 aux_loss ----
        router_prob = logits.softmax(dim=-1)                      # [B*T, N]
        print(router_prob)
        # f_i: 实际分到专家的 token 比例
        # 先构造 one-hot 掩码，再平均
        expert_mask = torch.zeros_like(router_prob).scatter_(-1, top_k_idx, 1.0)  # [B*T, N]
        f = expert_mask.mean(dim=0)                               # [N], 每个专家这个batch分到topk中的数量 / batch_size 不可微
        P = router_prob.mean(dim=0)                               # [N], 每个专家这个batch中的打分均值. 可微
        aux_loss = (f * P).sum()  # 标量

        return top_k_weights, top_k_idx, aux_loss


class MoeFFN(nn.Module):
    """
    无 for 循环的稀疏 MoE 层
    """
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 top_k: int,
                 dim_feedforward: int,
                 activation=nn.SiLU,
                 dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = TopKGate(d_model, num_experts, top_k)

        # 专家权重：一次性注册成 ParameterList，方便后续 index_select
        device = torch.device('cpu')
        self.w1 = nn.ParameterList([
            nn.Parameter(torch.empty(d_model, dim_feedforwardi, device=device))
            for _ in range(num_experts)
        ])
        self.w2 = nn.ParameterList([
            nn.Parameter(torch.empty(dim_feedforward, d_model, device=device))
            for _ in range(num_experts)
        ])
        for p in self.parameters():
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.act = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x:               [B, T, D]
        combine_weights: [B*T, top_k]  归一化权重（gate 输出）
        expert_indices:  [B*T, top_k]  int64
        return:          [B, T, D]
        """
        B, T, D = x.shape
        device = x.device
        flat_x = x.reshape(B * T, D)          # [B*T, D]

        combine_weights, expert_indices, aux_loss = self.gate(x) # [B*T, k], [B*T, k]

        # 1. 拉平成 [B*T*top_k]
        idx = expert_indices.reshape(-1)      # [B*T*top_k]
        weight = combine_weights.reshape(-1)  # [B*T*top_k]

        # 2. 构造输入：每个 token 复制 top_k 份
        inp = flat_x[:, None, :].expand(-1, self.top_k, -1).reshape(-1, D)  # [B*T*top_k, D]

        # 3. 一次性取出对应专家的权重
        # 先把所有专家权重拼成 [num_experts, ...] 的大张量
        w1_stack = torch.stack([p for p in self.w1], dim=0)  # [E, D, d_ff]
        w2_stack = torch.stack([p for p in self.w2], dim=0)  # [E, d_ff, D]

        w1_sel = w1_stack[idx]  # [B*T*top_k, d_ff, D]
        w2_sel = w2_stack[idx]  # [B*T*top_k, D,   d_ff]

        # 4. 批量矩阵乘：inp @ w1_sel
        h = torch.einsum('bd,bdi->bi', inp, w1_sel)  # [B*T*top_k, d_ff]
        h = self.act(h)
        h = torch.einsum('bi,bid->bd', h, w2_sel)    # [B*T*top_k, D]

        # 5. 加权并 scatter_add 回 [B*T, D]
        h = h * weight.unsqueeze(1)  # 先乘权重
        out = torch.zeros(B * T, D, device=device, dtype=x.dtype)
        idx_exp = idx.unsqueeze(1).expand(-1, D)
        out.scatter_add_(0, idx_exp, h)

        out = self.dropout(out)
        return out.view(B, T, D), aux_loss

