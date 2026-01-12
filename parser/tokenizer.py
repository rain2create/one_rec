# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : tokenizer.py
#   author   : ***
#   date     : 2025-07-12
#   desc     : 
# ======================================================== 

import os
import torch
import random
import numpy as np
from typing import List, Optional, Union
from utils.logger import logger
from collections import defaultdict
from arguments.data_args import DataArguments
import torch.distributed as dist

class UrlTokenizer:
    # 实现一个url(str) 到l层codebook的tokenizer
    # 从文件中读取url 到 l 个int的映射    
    def __init__(self, layers: int, codebook_size: int, remote_mapping_file, mapping_file: str, bos_token_id: Optional[int] = None, unk_token_id: Optional[int] = None):
        # 不需要eos token, 因为layers固定
        self.layers = layers
        self.codebook_size = codebook_size
        self.remote_mapping_file = remote_mapping_file
        self.mapping_file = mapping_file
        self.bos_token_id = 0 if bos_token_id is None else bos_token_id
        self.unk_token_id = self.codebook_size + 1 if unk_token_id is None else unk_token_id
        self.url2tokens = {}  # url -> [token1, token2, token3]
        self.tokens2url = defaultdict(list)  # tuple(token1, token2, token3) -> url list
        self.sample_dict = {} # int -> [[int, int, int], ...]
        if self.get_mapping_file() != 0:
            logger.error(f"Failed to copy mapping file from {self.remote_mapping_file} to {self.mapping_file}")
        if self.init_mapping() != 0:
            logger.error(f"Failed to initialize mapping from {self.mapping_file}")
        else:
            logger.info(f"Tokenizer initialized with {len(self.url2tokens)} URLs in {self.layers} layers and {self.codebook_size} codebooks.")

    def get_mapping_file(self) -> int:
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                ret = 0
                if not os.path.exists(self.remote_mapping_file):
                    logger.error(f"Remote mapping file {self.remote_mapping_file} does not exist.")
                    ret = -1
                else:
                    ret = os.system(f"cp {self.remote_mapping_file} {self.mapping_file}")
                    logger.info(f"get_mapping_file (rank0): cp {self.remote_mapping_file} {self.mapping_file}")
            else:
                ret = 0  # 其他 rank 无需拷贝
            dist.barrier()  # 等待文件准备好
            return ret
        else:
            # 单机单卡
            if not os.path.exists(self.remote_mapping_file):
                logger.error(f"Remote mapping file {self.remote_mapping_file} does not exist.")
                return -1
            return os.system(f"cp {self.remote_mapping_file} {self.mapping_file}")	

    def init_mapping(self) -> int:
        ensure_matrix = np.zeros([2, 1502, 1502]) * -1e3
        # 假设是三层
        def handle_ensure_matrix(ids):
            if len(ids) < 3:
                return
            ensure_matrix[0][ids[0]][ids[1]] = 1.0
            ensure_matrix[1][ids[1]][ids[2]] = 1.0
        # 从文件中读取 URL 到 token 的映射
        with open(self.mapping_file, 'r') as f:
            for line in f:
                url, ids = line.strip().split("\t")
                ids = list(map(int, ids.split(" ")))
                handle_ensure_matrix(ids)
                self.url2tokens[url] = ids
                self.tokens2url[tuple(ids)].append(url)
                # sample dict init
                if ids[0] not in self.sample_dict:
                    self.sample_dict[ids[0]] = []
                self.sample_dict[ids[0]].append(ids)
        np.save(DataArguments.ensure_matrix_path, ensure_matrix)
        return 0

    def encode(self, url: str, add_bos) -> List[int]:
        """
        Args:
            url: 单个字符串 URL
        Returns:
            token ids: List[int], 长度 = 3 (+1 如果 add_bos)
        """
        tokens = self.url2tokens.get(url, [self.unk_token_id] * 3)
        if add_bos:
            return [self.bos_token_id] + tokens
        return tokens

    def seq_encode(self, seq_urls: np.ndarray) -> torch.Tensor:
        try:
            B, L = seq_urls.shape
            res = np.zeros([B, L, 3], dtype=np.int32)
            for i in range(seq_urls.shape[0]):
                for j in range(seq_urls.shape[1]):
                    res[i][j] = self.encode(seq_urls[i][j], add_bos=False)
        except Exception as err:
            print(err)
        return torch.from_numpy(res)

    def sample_from_cls(self, tokens: List[int], add_bos) -> List[int]:
        """
        input url
        ouput one url code in url class
        """
        try:
            added = []
            if add_bos:
                added.append(self.bos_token_id)
            if tokens[1] == self.unk_token_id:
                k = random.sample(tuple(self.sample_dict), 1)[0]
                v_i = random.sample(self.sample_dict[k], 1)[0]
            else:
                k = tokens[1]
                v_i = random.sample(self.sample_dict[k], 1)[0]
            return added + v_i
        except Exception as err:
            print('xxxxxxxxx', err)
        return [0] * 4

    def decode(self, ids: Union[List[int], torch.LongTensor]) -> List[str]:
        """
        Args:
            ids: List[int] 或 torch.LongTensor, 长度 = 3 (+1 如果 add_bos)
        Returns:
            url: List[str], 所有满足tokenize后为 ids 的 URL 列表
        """
        if isinstance(ids, torch.Tensor):
            if ids.numel() == 0 or ids.dim() == 0:  # 空 tensor or scalar tensor, e.g., tensor(123)
                return []
            ids = ids.tolist()
        if not isinstance(ids, (list, tuple)) or len(ids) == 0:
            return []
        if ids[0] == self.bos_token_id:
            ids = ids[1:]
        if len(ids) < 3:
            logger.error(f"Invalid ids length: {len(ids)}, expected at least 3")
            return []
        tokens = tuple(ids[:3]) # 只取前3个token
        return self.tokens2url.get(tokens, [])  

    def batch_encode(self, urls: Union[List, np.ndarray], add_bos) -> torch.LongTensor:
        """
        Args:
            urls: List[str], 每个字符串 URL
            add_bos: bool, 是否在每个 URL 前添加 bos token
        Returns:
            token ids: torch.LongTensor, shape = (batch_size, layers)
        """
        if isinstance(urls, np.ndarray):
            urls = urls.tolist()
        batch_size = len(urls)
        layers = self.layers + 1 if add_bos else self.layers
        ids = torch.zeros((batch_size, layers), dtype=torch.long)
        for i, url in enumerate(urls):
            tokens = self.encode(url, add_bos)
            ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)   
        return ids

    def batch_decode(self, ids_list: Union[List[List[int]], torch.LongTensor]) -> List[List[str]]:
        """
        Args:
            ids_list: 可以是:
                      - List[List[int]] (CPU)
                      - torch.LongTensor [B, seq_len] (GPU/CPU)
        Returns:
            List[List[str]]: 每个元素是解码出的 URL 列表
        """
        # 如果输入是 tensor，转为 list of list
        if isinstance(ids_list, torch.Tensor):
            if ids_list.dim() != 2:
                raise ValueError(f"Expected 2D tensor, got {ids_list.dim()}D")
            ids_list = ids_list.tolist()  # 自动从 GPU 搬到 CPU 并转 list
    
        urls = []
        cashe = {}
        for ids in ids_list:
            ids_k = ",".join([str(x) for x in ids])
            if ids_k in cashe:
                decode_md5 = cashe[ids_k]
            else:
                decode_md5s = self.decode(ids)
                if len(decode_md5s) == 0:
                    decode_md5 = ""
                elif len(decode_md5s) == 1:
                    decode_md5 = decode_md5s[0]
                else:
                    decode_md5 = random.choice(decode_md5s)
                cashe[ids_k] = decode_md5
            urls.append(decode_md5)  # decode 支持 list
        return urls

    def batch_decode_processed(self, output_ids, device):
        decoded_results = self.batch_decode(output_ids)  # List[str]

        # 随机选择一个解码结果（即使为空也占位）
        #selected_urls = []
        #for urls in decoded_results:
        #    if len(urls) == 0:
        #        selected_urls.append("")
        #    elif len(urls) == 1:
        #        selected_urls.append(urls[0])
        #    else:
        #        selected_urls.append(random.choice(urls))
        selected_urls = np.array(decoded_results)
        
        # 构造 sample_mask：非空才有意义
        sample_mask = torch.tensor([len(urls) > 0 for urls in decoded_results], 
                                  device=device, dtype=torch.bool)  # [B*G]
        
        empty_ratio = 1.0 - sample_mask.float().mean()

        sample_mask_2d = sample_mask.view(-1, len(decoded_results))
        per_batch_non_empty_counts = sample_mask_2d.sum(dim=1)
        return selected_urls, sample_mask, empty_ratio, per_batch_non_empty_counts

    def __len__(self):
        return self.vocab_size
            
    @classmethod
    def from_args(cls, args: DataArguments):
        return cls(
            layers=args.codebook_layers,
            codebook_size=args.codebook_size,
            remote_mapping_file=args.remote_mapping_file,
            mapping_file=args.mapping_file,
            bos_token_id=args.bos_token_id,
            unk_token_id=args.unk_token_id,
        )

if __name__ == "__main__":
    # Example usage
    tokenizer = UrlTokenizer(layers=3, codebook_size=1500, remote_mapping_file="/data01/search/lusitong/dev/onerec_tokenizor/data/clustering_results", mapping_file="../data/url_mapping", bos_token_id=0, unk_token_id=1501)
    
    url = "29da24498b96ca9ca2ab55b7ddcfb93b"
    tokens = tokenizer.encode(url, add_bos=True)
    print(f"Encoded tokens for URL '{url}': {tokens}")

    urls = ["5cc0d043667e79af49fd8ca43a7a4171", "6a46124c72b09694d76447a8780430fa", "2581f40bc9b9c49f9dbe2dcc013b7c05"]
    tokens = tokenizer.batch_encode(urls, add_bos=True)
    print(f"Batch encoded tokens: {tokens}")

    ids = [1213, 663, 1018]
    decoded_url = tokenizer.decode(ids)
    print(f"Decoded URL for tokens {ids}: {decoded_url}")   

    batch_ids = torch.tensor([[662, 474, 862], [830, 901, 1188], [1, 2, 3]], dtype=torch.long)
    decoded_urls = tokenizer.batch_decode(batch_ids)
    print(f"Decoded URLs: {decoded_urls}")
