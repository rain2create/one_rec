#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : moe_v2.py
#   author   : ***
#   date     : 2025-11-13
#   desc     : 
# ======================================================== 
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
from model.manager import MOEManager

import torch
import torch.nn as nn
from torch.nn import functional as F

MANAGER = MOEManager()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)

    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32

        B, T, _ = x.size()
        num_tokens = B * T

        # eq (4) in (https://arxiv.org/abs/1701.06538)
        logits = self.w_g(x)  # [B, T, n_exp]

        # find top k experts for each token
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

        # normalize expert probabilities
        # Question: should we normalize over all experts or just top-k?
        # we choose to normalize over top-k, other option is commented out below

        # Shazeer et al (https://arxiv.org/abs/1701.06538) does only topk
        # see page 4 eq (3)-(5), the code for this is commented out below
        router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
        router_probs.scatter_(-1, top_k_indices, top_k_logits)
        router_probs = F.softmax(router_probs, dim=-1)

        # # normalize all router logits (not just top-k) via softmax      
        # router_probs = F.softmax(logits, dim=-1)

        # compute auxiliary load balancing loss
        # this loss encourages equal probability assigned to each expert
        # and equal load balancing of tokens assigned to each expert

        if not torch.jit.is_scripting():
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

        # compute expert capacity
        exp_capacity = self.get_capacity(num_tokens)

        # make a multi-hot mask of chosen experts, size [B, T, n_exp]
        # entries are 0 if expert not chosen and 1 if expert chosen
        exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
        exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
        exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

        # compute cumulative sum of each token over experts, this stores
        # the index of each token within the batch of each expert
        # NOTE: cumsum should count all top-1 first, top-2 second, etc.
        # so that we prioritize top experts when dropping tokens (this is
        # done by putting k dimension first for the reshape operation)
        exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
        exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
        exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

        # mask out (set to zero) entries that go beyond expert capacity
        # compute amount of used capacity by taking a sum over mask
        exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
        used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

        # mask rank to only include tokens that are selected
        # perform a sum so each row only contains index of token
        # for the expert that is selected in that row
        # result is a matrix that contains the position of each token
        # in the batch of its corresponding expert
        exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

        # mask probabilities to only include selected experts
        router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
        exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

        # convert rank into one-hot vectors over the available capacity
        # stores the position of each token within the capacity of the selected expert
        exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

        # create a vector that stores, for each token, the weight of selected
        # experts at token's position in the capacity of that expert
        # size of tensor is [B * T, n_exp, exp_capacity]
        cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
        # sec_mask = cb_weight.bool() # binary mask of selected experts for each token
        sec_mask = cb_weight.to(torch.bool)
        return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch: int):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            # perform standard (normal) initialization of weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            # perform standard (normal) initialization of weights
            torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]
        
        # resize output before return
        return output.view(B, T, n_embd)

class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MOEConfig:
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # MoE-related configs 
    n_exp: int = 8 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router

