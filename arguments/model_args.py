# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : model_args.py
#   author   : ***
#   date     : 2025-07-18
#   desc     : 
# ======================================================== 
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ModelArguments:
    # encoder
    emb_dim: int = 128
    model_dim: int = 256
    encoder_num_layers: int = 6
    encoder_num_heads: int = 4
    encoder_dropout: float = 0.1
    encoder_max_seq_len: int = 128
    encoder_pre_norm: bool = True
    feature_vocabs : Dict[str, int] = field(default_factory=lambda: {
            # "uid": 20000019, 
            "sign": 4, 
            "autok": 11, 
            "autoplay": 3, 
            "reqtimes": 30, 
            #"md5": 20000019,
            "qid": 1000003,
            "videotag": 10007,
            "timegap": 30,
            "duration": 30,
            "dwell": 30,
            "label": 100019
    })

    # decoder
    decoder_num_layers : int = 6
    decoder_num_heads : int = 4
    decoder_dropout : float = 0.1
    decoder_codebook_layers : int = 3
    decoder_codebook_size : int = 1500
    decoder_pre_norm : bool = True
    decoder_bos_token_id : int = 0
    decoder_ignore_idx: int = 1501
    decoder_temperature : float = 10.0
    decoder_weight_tying : bool = True
