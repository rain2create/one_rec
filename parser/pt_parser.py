# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : pt_parser.py
#   author   : ***
#   date     : 2025-07-12
#   desc     : 
# ======================================================== 

import numpy as np
from typing import List, Dict, Optional, Union
from functools import lru_cache
import torch
import xxhash
from .tokenizer import UrlTokenizer 

class PtParser:
    def __init__(self, column_def: List[tuple], tokenizer: UrlTokenizer):
        self.column_def = column_def 
        self.col_num = len(column_def)
        self.col_name = [x for x, y, z in self.column_def]
        self.col_default = [y for x, y, z in self.column_def]
        self.valid_sign = set(["360_97286759", "360_6453942f", "360_26aafc3a"])
        self.sign_lookup_table = {"360_97286759" : 1, "360_6453942f" : 2, "360_26aafc3a": 3}
        self.reqtimes_boundaries = torch.tensor([0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 200, 500], dtype=torch.int32)
        self.duration_boundaries = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0], dtype=torch.float32)
        self.dwell_boundaries = torch.tensor([2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0], dtype=torch.float32)
        self.timestamp_gap_boundaries_logsess = torch.tensor([2, 4, 15, 30, 50, 70, 120, 300, 600, 800, 1000, 1600, 3600], dtype=torch.int32)
        self.url_tokenizer = tokenizer  # tokenizer is an instance of UrlTokenizer
        self.url_cnts = dict()

    def fetch_cnts(self, urls):
        res = []
        urls = urls.tolist()
        for url in urls:
            hid = self._cached_hash(url, 0) % 99999999
            if hid not in self.url_cnts:
                self.url_cnts[hid] = 1
            else:
                self.url_cnts[hid] += 1
            res.append(self.url_cnts[hid])
        return np.asarray(res)

    @lru_cache(maxsize=5000000)
    def _cached_hash(self, key, seed):
        return xxhash.xxh64(key, seed=seed).intdigest()

    def _hash_to_tensor_with_table(
        self,
        array: np.ndarray,
        map_table: dict,
        default_value: int = 0,
        dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        """
        对任意 shape 的字符串数组查找map_table, 返回 torch.LongTensor。
        Args:
            array (np.ndarray): 输入 NumPy 字符串数组, dtype=str
            map_table : 待查找的dict
            default_value : 如果查不到时的default value
        Returns:
            torch.LongTensor: 与输入形状相同
        """
        flat = array.ravel()  # 展平成一维列表
        hashed = np.empty(flat.shape, dtype=np.uint64)

        for i, val in enumerate(flat):
            h = map_table.get(val, default_value)
            hashed[i] = h
        # reshape 并转为 torch.LongTensor
        hashed = hashed.reshape(array.shape)
        return torch.tensor(hashed, dtype=dtype)

    def _hash_to_tensor(
        self,
        array: np.ndarray,
        vocab_size: Optional[int] = None,
        seed: int = 0,
        dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        """
        对任意 shape 的字符串数组做哈希，返回 torch.LongTensor。
        Args:
            array (np.ndarray): 输入 NumPy 字符串数组, dtype=str
            vocab_size (int or None): 若设置，则哈希值将取模限制范围
            seed (int): 哈希函数的种子，用于控制一致性
        Returns:
            torch.LongTensor: 与输入形状相同
        """
        flat = array.ravel()  # 展平成一维列表
        hashed = np.empty(flat.shape, dtype=np.uint64)

        for i, val in enumerate(flat):
            h = self._cached_hash(val, seed)
            if vocab_size:
                h %= vocab_size
            hashed[i] = h

        # reshape 并转为 torch.LongTensor
        hashed = hashed.reshape(array.shape)
        return torch.tensor(hashed, dtype=dtype)

    def _to_tensor(
        self,
        array: np.ndarray,
        pad_val: Union[int, float, str] = 0,
        limit_val: Optional[float] = None,
        np_dtype: type = np.int64
    ) -> torch.Tensor:
        """
        将类似np.array([['1', '2'], ['2', '0']])的数组转成tensor。
        如果存在''等无法转换的情况, 自动用pad_val(默认0)填充。
        """
        # 将空字符串替换为 pad_val
        array[array == ''] = str(pad_val)
        numeric_arr = array.astype(np_dtype)
        if limit_val is not None:
            numeric_arr = np.clip(numeric_arr, 0, limit_val)
        tensor = torch.from_numpy(numeric_arr)
        return tensor

    def _str_not_eq(self, arr: np.ndarray, target: str) -> torch.Tensor:
        """
        判断arr中元素是否等于target, 1表示不等于，0表示等于
        arr是numpy str数组
        """
        return torch.tensor(arr != target, dtype=torch.bool).int()

    def _bucketize(self, arr: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """
        将一维数组 arr 中的值根据 boundaries 进行分桶，返回每个值对应的桶索引。boundaries 是一个一维 tensor，表示每个桶的边界。
        """
        return torch.bucketize(arr, boundaries)

    def _split_and_pad_fast(
        self,
        arr: np.ndarray,
        sep: str = '|',
        pad_val: str = '',
        max_len: int = 30
    ) -> np.ndarray:
        # 对任意 shape的字符串数组进行split和padding
        # 向量化 split
        split_arr = np.char.split(arr, sep)
        # pad/truncate
        result = np.full((arr.shape[0], max_len), pad_val, dtype=object)
        for i, tokens in enumerate(split_arr):
            tokens = tokens[:max_len]
            tokens = tokens[::-1] # 自然语言序
            result[i, :len(tokens)] = tokens
        return result

    def _preprocess_dwell(
        self,
        dwell_list: np.ndarray,
        dur_list: np.ndarray
    ) -> np.ndarray:
        """
        对 logsess_dwell_list 进行预处理：
        1. 空字符串（''）替换为 '0'
        2. 'x' 替换为对应位置的 logsess_dur_list
        输入和输出均为 shape=(batch, seq_len) 的 np.ndarray，dtype=str
        """
        dwell = dwell_list.copy()
        # 1. 空字符串替换为 '0'
        dwell[dwell == ''] = '0'
        # 2. 'x' 替换为对应位置的 dur_list
        dwell[dwell == 'x'] = dur_list[dwell == 'x']
        return dwell

    def preprocess_user_features(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        处理用户静态特征和上下文特征。
        返回一个字典，包含处理后的特征张量。
        """
        res = {}
        res["uid"] = self._hash_to_tensor(features["f003.uid"], vocab_size=20000019, seed=0, dtype=torch.int64)
        res["sign"] = self._hash_to_tensor_with_table(features["f001.sign"], self.sign_lookup_table)
        res["autok"] = self._to_tensor(features["f119.autok"], limit_val=10)
        res["autoplay"] = self._to_tensor(features["f016.auto_play"], limit_val=1)
        res["reqtimes"] = self._bucketize(self._to_tensor(features["f011.reqtimes"]), self.reqtimes_boundaries)
        res["logtime"] = self._to_tensor(features["f015.logtime"], pad_val=0, np_dtype=np.float32)
        return res

    def preprocess_logsess_features(
        self,
        features: Dict[str, np.ndarray],
        logtime: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        处理用户短时行为特征 logsess
        返回一个字典，包含处理后的特征张量。
        """
        res = {}
        logtime_expanded = logtime.unsqueeze(1) if logtime.dim() == 1 else logtime
        logsess_md5_list = self._split_and_pad_fast(features["f044.logsess_item_seqs"], sep='|', pad_val='url_pad', max_len=3)
        logsess_qid_list = self._split_and_pad_fast(features["f072.logsess_seqsQid"], sep='|', pad_val='qid_pad', max_len=3)
        logsess_videotag_list = self._split_and_pad_fast(features["f074.logsess_seqsVideotag"], sep='|', pad_val='tag_pad', max_len=3)
        logsess_timestamp_list = self._split_and_pad_fast(features["f046.logsess_time_seqs"], sep='|', pad_val='0', max_len=3)
        logsess_dur_list = self._split_and_pad_fast(features["f073.logsess_seqsDuration"], sep='|', pad_val='0', max_len=3)
        logsess_dwell_list = self._split_and_pad_fast(features["f071.logsess_seqsDwell"], sep='|', pad_val='0', max_len=3)
        logsess_dwell_list = self._preprocess_dwell(logsess_dwell_list, logsess_dur_list)  # 处理用户观看时长
        logsess_label_list = self._split_and_pad_fast(features["f047.logsess_att_seqs"], sep='|', pad_val='0', max_len=3)

        res["logsess_md5"] = self._hash_to_tensor(logsess_md5_list, vocab_size=20000019, seed=0, dtype=torch.int64)
        res["logsess_qid"] = self._hash_to_tensor(logsess_qid_list, vocab_size=1000003, seed=0, dtype=torch.int64)
        res["logsess_videotag"] = self._hash_to_tensor(logsess_videotag_list, vocab_size=10007, seed=0, dtype=torch.int64)
        res["logsess_timegap"] = self._bucketize(logtime_expanded - self._to_tensor(logsess_timestamp_list), self.timestamp_gap_boundaries_logsess)
        res["logsess_duration"] = self._bucketize(self._to_tensor(logsess_dur_list, pad_val=0, np_dtype=np.float32), self.duration_boundaries)
        res["logsess_dwell"] = self._bucketize(self._to_tensor(logsess_dwell_list, pad_val=0, np_dtype=np.float32), self.dwell_boundaries)
        res["logsess_label"] = self._hash_to_tensor(logsess_label_list, vocab_size=100019, seed=0, dtype=torch.int64)
        res["logsess_mask"] = self._str_not_eq(logsess_md5_list, 'url_pad')  # mask: pad token为0, 其他为1
        return res

    def preprocess_logqueue_features(
        self,
        features: Dict[str, np.ndarray],
        logtime: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        处理用户 logqueue 行为特征，特征列为 logQue_ 开头的相关特征。
        返回一个字典，包含处理后的特征张量。
        """
        res = {}
        logtime_expanded = logtime.unsqueeze(1) if logtime.dim() == 1 else logtime
        # logQue_相关特征名见 columns.py
        logque_md5_list = self._split_and_pad_fast(features["f080.logQue_seqsItem"], sep='|', pad_val='url_pad', max_len=30)
        logque_qid_list = self._split_and_pad_fast(features["f084.logQue_seqsQid"], sep='|', pad_val='qid_pad', max_len=30)
        logque_videotag_list = self._split_and_pad_fast(features["f086.logQue_seqsVideotag"], sep='|', pad_val='tag_pad', max_len=30)
        logque_timestamp_list = self._split_and_pad_fast(features["f081.logQue_seqsTime"], sep='|', pad_val='0', max_len=30)
        logque_dur_list = self._split_and_pad_fast(features["f085.logQue_seqsDuration"], sep='|', pad_val='0', max_len=30)
        logque_dwell_list = self._split_and_pad_fast(features["f083.logQue_seqsDwell"], sep='|', pad_val='0', max_len=30)
        logque_dwell_list = self._preprocess_dwell(logque_dwell_list, logque_dur_list)  # 处理用户观看时长
        logque_label_list = self._split_and_pad_fast(features["f082.logQue_seqsAtt"], sep='|', pad_val='0', max_len=30)

        # res["logque_md5_list"] = features["f080.logQue_seqsItem"]
        res["logque_md5"] = self._hash_to_tensor(logque_md5_list, vocab_size=20000019, seed=0, dtype=torch.int64)
        res["logque_qid"] = self._hash_to_tensor(logque_qid_list, vocab_size=1000003, seed=0, dtype=torch.int64)
        res["logque_videotag"] = self._hash_to_tensor(logque_videotag_list, vocab_size=10007, seed=0, dtype=torch.int64)
        res["logque_timegap"] = self._bucketize(logtime_expanded - self._to_tensor(logque_timestamp_list), self.timestamp_gap_boundaries_logsess)
        res["logque_duration"] = self._bucketize(self._to_tensor(logque_dur_list, pad_val=0, np_dtype=np.float32), self.duration_boundaries)
        res["logque_dwell"] = self._bucketize(self._to_tensor(logque_dwell_list, pad_val=0, np_dtype=np.float32), self.dwell_boundaries)
        res["logque_label"] = self._hash_to_tensor(logque_label_list, vocab_size=100019, seed=0, dtype=torch.int64)
        res["logque_mask"] = self._str_not_eq(logque_md5_list, 'url_pad')  # mask: pad token为0, 其他为1
        return res

    def preprocess_logreallike_features(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        处理用户正反馈序列特征，只处理视频id、作者id、tag id、视频时长。
        返回一个字典，包含处理后的特征张量。
        """
        res = {}
        # 只处理四个特征
        logreallike_md5_list = self._split_and_pad_fast(features["f091.logreallike_seqsItem"], sep='|', pad_val='url_pad', max_len=30)
        logreallike_authorid_list = self._split_and_pad_fast(features["f093.logreallike_seqsFunc"], sep='|', pad_val='author_pad', max_len=30)
        logreallike_videotag_list = self._split_and_pad_fast(features["f097.logreallike_seqsVideotag"], sep='|', pad_val='tag_pad', max_len=30)
        logreallike_dur_list = self._split_and_pad_fast(features["f096.logreallike_seqsDuration"], sep='|', pad_val='0', max_len=30)
        
        # res["logreallike_md5_list"] = features["f091.logreallike_seqsItem"]
        res["logreallike_md5"] = self._hash_to_tensor(logreallike_md5_list, vocab_size=20000019, seed=0, dtype=torch.int64)
        res["logreallike_qid"] = self._hash_to_tensor(logreallike_authorid_list, vocab_size=1000003, seed=0, dtype=torch.int64)
        res["logreallike_videotag"] = self._hash_to_tensor(logreallike_videotag_list, vocab_size=10007, seed=0, dtype=torch.int64)
        res["logreallike_duration"] = self._bucketize(self._to_tensor(logreallike_dur_list, pad_val=0, np_dtype=np.float32), self.duration_boundaries)
        res["logreallike_mask"] = self._str_not_eq(logreallike_md5_list, 'url_pad') # mask: pad token为0, 其他为1
        return res

    def sample_mask(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        样本mask, 根据用户label, mask所有非alive的样本
        """
        # zmtlink_label
        zmtlink_label = (
            (features['f033.authorpanel'] == "1") |
            (features['f034.videocollectpanel'] == "1") |
            (features['f060.collectionpanel'] == "1") |
            (features['f051.playletpanel'] == "1") |
            (features['f035.manu_right_click'] == "1")
        ).astype(np.float32)

        # dislike_label
        dislike_label = ((features['f031.dislike'] == "1") | (features['f030.like_c'] == "1")).astype(np.float32)

        # like_label
        like_label = ((features['f029.like'] == "1") | (features['f089.favorite'] == "1")).astype(np.float32)
        like_label = np.where(dislike_label == 1.0, 0.0, like_label)

        # follow_label
        follow_label = (features['f032.follow'] == "1").astype(np.float32)

        # down/up
        down = (features["f023.is_manu_down"] == "1").astype(np.float32)
        manu = np.where(features["f025.is_manu_up"] == "1", 1.0, down)

        # play_type
        play_type = np.char.split(features["f036.play_type"], '|')
        from_down_m = np.array(['manu_down_m' in pt for pt in play_type])
        from_down_k = np.array(['manu_down_kb' in pt for pt in play_type])
        from_down_h = np.array(['manu_down_wh' in pt for pt in play_type])
        from_down = (from_down_m | from_down_k | from_down_h).astype(np.float32)
        from_up = (features["f050.last_video"] == "1").astype(np.float32)

        from_valid_up = ((features["f050.last_video"] == "1") & (features["f021.is_finish_video"] == "1")).astype(np.float32)
        from_valid_up = np.where((features["f050.last_video"] == "1") & (zmtlink_label == 1.0), 1.0, from_valid_up)
        from_valid_up = np.where((features["f050.last_video"] == "1") & (follow_label == 1.0), 1.0, from_valid_up)
        from_valid_up = np.where((features["f050.last_video"] == "1") & (like_label == 1.0), 1.0, from_valid_up)
        features["last_video_label"] = from_valid_up

        # finish_label
        finish = (features["f021.is_finish_video"] == "1").astype(np.float32)
        features['finish_label'] = finish

        # all positive interactions
        positive = np.maximum.reduce([like_label, follow_label, zmtlink_label])
        actions = np.maximum(dislike_label, positive)

        # total alive
        alive = np.maximum.reduce([from_up, from_down, manu, actions])

        raw_dur = features['f005.duration'].astype(np.float32)
        alive = np.where(raw_dur < 9.0, 0.0, alive)

        # skip
        skip_bound = np.where(raw_dur < 10.0, 2.0, 2.0)
        down_time = features['f024.manu_down_time'].astype(np.float32)
        skip = np.where(down_time < skip_bound, 1.0, 0.0)
        skip = np.where(down == 0.0, 0.0, skip)
        skip = np.where(finish == 1.0, 0.0, skip)
        skip = np.where(positive == 1.0, 0.0, skip)
        # alive = np.where(skip == 1.0, 0.0, alive)
        features['skip'] = skip

        # sign mask
        if not self.valid_sign:
            sign_mask = np.ones(features["f001.sign"].shape, dtype=bool)
        else:
            sign_mask = np.zeros(features["f001.sign"].shape, dtype=bool)
            for sign in self.valid_sign:
                sign_mask |= (features["f001.sign"] == sign)

        alive = np.where(sign_mask, alive, 0.0)
        features['weight'] = alive

        # all in
        features['weight'] = np.ones_like(alive, dtype=np.float32)

        # 1 表示alive有效样本, 0表示dead
        return {"mask": torch.from_numpy(features['weight'].astype(np.int64)), "duration": torch.from_numpy(raw_dur)}

    def extract_features(self, features: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        res = {}
        # user static and context 
        # uid, sign, 过去几个自动播, 是否自动播, 刷屏深度, 当前时间
        res.update(self.preprocess_user_features(features))

        # user short time behavior: logsess
        # 视频id, 作者id, tag id, 时间戳，视频时长, 观看时长，用户label
        res.update(self.preprocess_logsess_features(features, res["logtime"]))

        # user long time behavior: logqueue
        # 视频id, 作者id, tag id, 时间戳，视频时长, 观看时长，用户label
        res.update(self.preprocess_logqueue_features(features, res["logtime"]))

        # user positive feedback sequence
        # 视频id, 作者id, tag id, 视频时长
        res.update(self.preprocess_logreallike_features(features))

        # 是否需要做样本 mask 
        res.update(self.sample_mask(features))

        # target url tokenizer
        res["url"] = self.url_tokenizer.batch_encode(features["f004.url"], add_bos=True)

        # sample weights
        w = np.ones_like(res['mask'], dtype=np.float32)
        w = np.where(features['f029.like'] == '1', 8.0, w)
        w = np.where(features['f030.like_c'] == '1', -1.0, w)
        w = np.where(features['f090.favorite_c'] == '1', -1.0, w)
        w = np.where(features['f031.dislike'] == '1', -1.0, w)
        w = np.where(features['skip'] == 1.0, -1.0, w)
        duration_w = np.log(res['duration'].numpy() + 1.0) - 1.0
        duration_w = duration_w / np.mean(duration_w)
        res['sample_weights'] = torch.from_numpy(w * duration_w)

        # loss th
        th = np.zeros_like(res['mask'], dtype=np.float32)
        th = np.where(features['skip'], 5.0, th)
        th = np.where(features['f030.like_c'], 5.0, th)
        th = np.where(features['f031.dislike'], 5.0, th)
        th = np.where(features['f090.favorite_c'], 5.0, th)
        res['loss_th'] = torch.from_numpy(th)

        return res

    # def expand_old(self, features, alpha = -1.0):
    #     for k, v in features.items():
    #         if type(v) == torch.Tensor:
    #             v = v.numpy()
    #         if k == 'url':
    #             v_raw = v.copy()
    #             np.random.shuffle(v)
    #             features[k] = np.concatenate([v_raw, v], 0)
    #             features['sample_weights'] = np.concatenate([features['sample_weights'], np.ones_like(features['sample_weights']) * alpha * np.mean(features['sample_weights'])], 0)
    #         elif k == 'sample_weights':
    #             continue
    #         else:
    #             v_raw = v.copy()
    #             features[k] = np.concatenate([v_raw, v], 0)
    #     for k, v in features.items():
    #         features[k] = torch.from_numpy(v)

    def sample_index(self, features, alpha):
        """
        对 sample_weights > 0的数据采样, 占比alpha, 返回下标。
        """
        idx_valid = (features['sample_weights'] > 0).nonzero(as_tuple=False).squeeze(-1)
        L         = int((alpha * idx_valid.numel()))
        perm      = torch.randperm(idx_valid.numel())
        indexes   = idx_valid[perm[:L]]
        return indexes

    def expand(self, features, alpha=-1.0):
        url_indexes = self.sample_index(features, 1.0)
        neg_indexes = self.sample_index(features, 1.0)

        for k, v in features.items():
            if k == 'url':
                v_shuffle = v[url_indexes]
                features[k] = torch.concat([v, v_shuffle], 0)
            elif k == 'sample_weights':
                sampled_weights = v[url_indexes]
                features[k] = torch.concat([v, torch.ones_like(sampled_weights) * torch.mean(sampled_weights) * alpha], 0)
            else:
                v_shuffle = v[neg_indexes]
                features[k] = torch.concat([v, v_shuffle], 0)

    def expand_pos(self, features, alpha = 0.1):
        ## v1
        # indexes = torch.randperm(features['url'].size(0))[:int(alpha * features['url'].size(0))]

        ## v2
        indexes = self.sample_index(features, alpha)

        for k, v in features.items():
            if k == 'url':
                urls_target_cls = v[indexes].tolist()
                res = []
                for url_token in urls_target_cls:
                    res.append(self.url_tokenizer.sample_from_cls(url_token, add_bos=True))
                features[k] = torch.concat([v, torch.tensor(res, dtype=torch.long)], 0)
            else:
                features[k] = torch.concat([v, v[indexes]], 0)

    def parser(self, batch_lines: List[str]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        # 1. 字符串切割成二维列表
        splitted = [line.split(",") if line else [""] * self.col_num for line in batch_lines]  # List[List[str]]

        # 2. 转为 numpy 字符串数组 (batch_size, col_num)，直接用 dtype=str，方便后续字符串处理
        arr = np.array(splitted, dtype=str)

        # 3. 用默认值替换空字符串
        for i, default in enumerate(self.col_default):
            mask = (arr[:, i] == '')
            arr[mask, i] = default

        # 4. 转为字典，key: 列名，value: 字符串数组
        features = {self.col_name[i]: arr[:, i] for i in range(self.col_num)}

        # 5. 处理特征
        features = self.extract_features(features)

        ## 扩展样本
        #self.expand_pos(features, 0.1)

        ## 负样本
        self.expand(features, -1.0)

        return features


if __name__ == "__main__":
    import gzip
    from utils.column import columns_def

    url_tokenizer = UrlTokenizer(layers=3, codebook_size=1500, \
                                remote_mapping_file="/data01/search/lusitong/dev/onerec_tokenizor/data/clustering_results", \
                                mapping_file="../data/url_mapping", bos_token_id=0, unk_token_id=1501)
    pt_parser = PtParser(columns_def, tokenizer=url_tokenizer)

    file_path = "/mnt/video.rank.s101/20250710/2025071011/20250710-11-38506.gz"
    batch_lines = []
    batch_size = 32
    with gzip.open(file_path, "rt", encoding="utf-8") as f, open("./raw_lines", "w") as g:
        for line in f:
            line = line.strip()
            batch_lines.append(line)
            for x in batch_lines:
                g.write(x + "\n")
            if len(batch_lines) == batch_size:
                # 处理一批数据
                features = pt_parser.parser(batch_lines)
                for k in features:
                    print(k, features[k])
                break
