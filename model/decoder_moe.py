#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#
#   filename : trf_decoder.py
#   author   : ***
#   date     : 2025-11-11
#   desc     :
# ========================================================
import torch
import torch.nn as nn
from model.moe_auxloss import MOELayer, MOEConfig
from typing import Any, Callable, Optional, Union
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


### 这里是decoder_moe模块的实现！！！ 包含decoder_moe layer和decoder_moe
### decoder_moe layer 包含x过 layernorm + causal attention + layernorm + cross attention + layernorm + moe/mlp
### decoder_moe 就是多层decoder_moe layer组成的decoder



def _generate_square_subsequent_mask(  ## 上三角矩阵的mask ！！！
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(  ## 只保留上三角区域，其他位置为0
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),  ## 这里full是生成shape 为（sz,sz)的矩阵，元素都一样为 -inf
        diagonal=1, ## 代表主对角线以上
    )

def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:  ## 取数据集里的序列长度！！！
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask( ## 检查万一没传is_causal还有mask的情况，检查这个mask是不是上三角矩阵 最后看情况返回true或者false
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)  ## 拿seq长度
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        ) ## 生成上三角的mask

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all()) ## 确认是不是casual mask
        else:
            make_causal = False

    return make_causal  ##最终返回的是True或者False


class DecoderLayerMOE(TransformerDecoderLayer):  ## 这个layer就是 x进来之后 先过self-attention, 再过cross-attention,再过MOE 再过ffn ！！！！ 正常的decoder_moe结构结构

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        num_experts=8,
        top_k=2,
        enable_moe=True,
        **kw
    ) -> None:
        super().__init__(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,**kw)

        self.moe_config = MOEConfig(
            n_embd = d_model,
            dropout = 0.1,
            n_exp = num_experts,
            top_k = top_k,
            use_aux_loss = True,
        )
        self.enable_moe = enable_moe
        if self.enable_moe:
            self.moe = MOELayer(self.moe_config)

        # 注意：cross-attn 不再有自己的 K/V 投影
        self.multihead_attn = None
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _mha_block(self, x: Tensor, shared_k: Tensor, shared_v: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool=False):
        q = self.q_proj(x)
        out = F.scaled_dot_product_attention(
            q, shared_k, shared_v,
            attn_mask=attn_mask,
            dropout_p=0.1,
            is_causal=is_causal
        )
        return self.dropout2(self.out_proj(out))   ## 这里是做flash_attention 但用的是别人的kv 相当于cross attention!!! 用之前的kv cache

    '''
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False):
        """
        Implements Lazy KV-Sharing:
        - Compute K/V only once
        - Each layer only computes Q
        """

        # MultiheadAttention weights
        mha = self.multihead_attn
        W = mha.in_proj_weight     # shape [3*d_model, d_model]
        b = mha.in_proj_bias
        E = mha.embed_dim

        # ------------------------------
        # 1. Build memory K/V once
        # ------------------------------
        if self._memory_k is None or self._memory_v is None:
            # K = W_k * mem
            self._memory_k = F.linear(
                mem,
                W[E:2*E],    # W_k
                b[E:2*E]
            )
            # V = W_v * mem
            self._memory_v = F.linear(
                mem,
                W[2*E:3*E],  # W_v
                b[2*E:3*E]
            )

        # ------------------------------
        # 2. Compute Q every layer
        # ------------------------------
        q = F.linear(x,
                     W[0:E],     # W_q
                     b[0:E])

        # ------------------------------
        # 3. SDPA Attention: Q × (KV cache)
        # ------------------------------
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, self._memory_k, self._memory_v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal
        )
        # Output projection
        out = mha.out_proj(attn_out)
        return self.dropout2(out)
    '''

    def forward(
        self,
        tgt: Tensor,
        shared_k: Tensor,
        shared_v: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(  ## sa_block是标准attention 有mask
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(  ## 这里是做cross_attention
                self.norm2(x),
                shared_k,
                shared_v,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            if self.enable_moe:  ## 如果enable_moe为True，则过moe 否则过mlp
                x_ = self.moe(self.norm3(x))
                x = x + x_
            else:
                x = x + self._ff_block(self.norm3(x))  ## ff block就是普通的mlp
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    shared_k,
                    shared_v,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )
            if self.enable_moe:
                x_ = self.moe(x)
                x = self.norm3(x + x_)
            else:
                x = self.norm3(x + self._ff_block(x))

        return x


class SharedMemoryKV(nn.Module): ## kv cache
    """
    Project memory to shared K / V (used by all decoder layers)
    """
    def __init__(self, d_model):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        # self.v_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, memory):
        """
        memory: [B, M, D]
        """
        k = self.k_proj(memory)
        #v = self.v_proj(memory)
        v = k
        return k, v


class DecoderMOE(TransformerDecoder):## 有多层decoder_moe layer组成的decoder

    def __init__(
        self,
        decoder_layer: "DecoderLayerMOE",
        num_layers: int,
        d_model: int,
        norm: Optional[TransformerDecoder] = None,
    ) -> None:
        super().__init__(decoder_layer=decoder_layer,num_layers=num_layers,norm=norm)
        self.shared_kv_layer = SharedMemoryKV(d_model=d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:

        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)  ## 取数据集序列长度
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)  ## 检查是否为上三角mask

        shared_k, shared_v = self.shared_kv_layer(memory)  ## encoder里的kv

        for mod in self.layers:  ## 这里的 mod 就是单个 DecoderLayerMOE 实例（即一个 Decoder 层）
            output = mod(
                output,
                shared_k,
                shared_v,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


if __name__ == "__main__":
    d_model = 32
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

    """
    # 等价 MoE-FFN：8 专家，Top-2 激活
    moe_ffn = MoeFFN(d_model, dim_feedforward=2048,
                     activation=nn.GELU, dropout=0.1,
                     num_experts=8, top_k=2)
    """
    decoder_layer = DecoderLayerMOE(d_model=d_model, nhead=4, dim_feedforward=4*d_model,dropout=0.1,activation='gelu',batch_first=True,norm_first=True)
    decoder_moe = DecoderMOE(decoder_layer, num_layers=3)

    # 接口完全一致
    y_dense = dense_ffn(x)
    y_moe, aux_losses = decoder_moe(x, x)
    print("dense out:", y_dense.shape)
    print("moe out  :", y_moe.shape)

    # 参数量对比
    def param_count(m): return sum(p.numel() for p in m.parameters())
    print("dense params:", param_count(dense_ffn))
    print("moe params  :", param_count(decoder_moe))

