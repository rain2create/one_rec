#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#
#   filename : trf_decoder.py
#   author   : ***
#   date     : 2025-11-11
#   desc     :
# ========================================================
import torch, torch.nn as nn
#from model.moe_auxloss import MoeFFN
from model.moe_auxloss import MOELayer, MOEConfig
from typing import Any, Callable, Optional, Union
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
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


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class DecoderMOE(TransformerDecoder):

    def __init__(
        self,
        decoder_layer: "DecoderLayerMOE",
        num_layers: int,
        norm: Optional[TransformerDecoder] = None,
    ) -> None:
        super().__init__(decoder_layer=decoder_layer,num_layers=num_layers,norm=norm)

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

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
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


class DecoderLayerMOE(TransformerDecoderLayer):

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
        '''
        self.moe = MoeFFN(
                d_model, 
                num_experts=num_experts,
                top_k=top_k,
                dim_feedforward=dim_feedforward
            )
        '''

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            if self.enable_moe:
                x_ = self.moe(self.norm3(x))
                x = x + x_
            else:
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            if self.enable_moe:
                x_ = self.moe(x)
                x = self.norm3(x + x_)
            else:
                x = self.norm3(x + self._ff_block(x))

        return x


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

