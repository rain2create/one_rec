# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : dual_parser.py
#   author   : ***
#   date     : 2025-07-24
#   desc     : 
# ======================================================== 
from typing import List, Dict, Union, Tuple
import numpy as np
import torch
import tensorflow as tf
from parser.pt_parser import PtParser
from parser.tokenizer import UrlTokenizer

class RewardParser(PtParser):
    def __init__(self, column_def: List[tuple], serving_column_idx: List[int], tokenizer: UrlTokenizer, logreallike_len: int):
        super().__init__(column_def, tokenizer)
        self.serving_column_idx = serving_column_idx
        self.logreallike_len = logreallike_len
        # 辅助函数：从特征名解析列索引 (f001 -> 0, f033 -> 32)
        def get_col_idx(feature_name: str) -> int:
            base = feature_name.split('.')[0]  # 提取 'f001' 部分
            return int(base[1:]) - 1  # 转换为列索引 (001 -> 0)
        # 预先计算所有需要的列索引
        self.COLS = {
            'sign': get_col_idx('f001.sign'),
            'duration': get_col_idx('f005.duration'),
            'is_finish': get_col_idx('f021.is_finish_video'),
            'manu_down_time': get_col_idx('f024.manu_down_time'),
            'authorpanel': get_col_idx('f033.authorpanel'),
            'videocollectpanel': get_col_idx('f034.videocollectpanel'),
            'collectionpanel': get_col_idx('f060.collectionpanel'),
            'playletpanel': get_col_idx('f051.playletpanel'),
            'manu_right_click': get_col_idx('f035.manu_right_click'),
            'dislike': get_col_idx('f031.dislike'),
            'like_c': get_col_idx('f030.like_c'),
            'like': get_col_idx('f029.like'),
            'favorite': get_col_idx('f089.favorite'),
            'follow': get_col_idx('f032.follow'),
            'is_manu_down': get_col_idx('f023.is_manu_down'),
            'is_manu_up': get_col_idx('f025.is_manu_up'),
            'play_type': get_col_idx('f036.play_type'),
            'last_video': get_col_idx('f050.last_video'),
            'is_finish_video': get_col_idx('f021.is_finish_video'),
            'logreallike': get_col_idx('f091.logreallike_seqsItem'),
        }


    def extract_features(self, features: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        # 保留原逻辑 删除mask构造过程
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

        # target url tokenizer
        res["url"] = self.url_tokenizer.batch_encode(features["f004.url"], add_bos=True)
        return res

    def parser(self, batch_lines: List[str]) -> Tuple[Dict[str, Union[torch.Tensor, np.ndarray]], tf.Tensor]:
        # 1. 字符串切割成二维列表
        splitted = [line.split(",") if line else [""] * self.col_num for line in batch_lines]  # List[List[str]]

        # 2. 转为 numpy 字符串数组 (batch_size, col_num)，直接用 dtype=str，方便后续字符串处理
        arr = np.array(splitted, dtype=str)

        # 保留alive有效数据
        arr, like_arr = self.sample_alive_rows(arr)
        if arr.size == 0:
            return None

        # 3. 用默认值替换空字符串
        for i, default in enumerate(self.col_default):
            mask = (arr[:, i] == '')
            arr[mask, i] = default

        # 4. 转为字典，key: 列名，value: 字符串数组
        features = {self.col_name[i]: arr[:, i] for i in range(self.col_num)}

        # 5. 处理特征（调用父类方法）
        processed_features = self.extract_features(features)

        # 6. 返回扩展结果（字典 + TensorFlow张量）
        return processed_features, arr[:, self.serving_column_idx], like_arr#, algfinish_arr

    def sample_alive_rows(self, arr: np.ndarray) -> np.ndarray:
       """
       过滤样本，仅保留alive的行
       输入: arr (batch_size, col_num) 字符串数组
       输出: 过滤后的 (new_batch_size, col_num) 字符串数组
       """

       batch_size = arr.shape[0]

       # 1. 计算各类标签
       # zmtlink_label
       zmtlink_label = (
           (arr[:, self.COLS['authorpanel']] == "1") |
           (arr[:, self.COLS['videocollectpanel']] == "1") |
           (arr[:, self.COLS['collectionpanel']] == "1") |
           (arr[:, self.COLS['playletpanel']] == "1") |
           (arr[:, self.COLS['manu_right_click']] == "1")
       ).astype(np.float32)

       # dislike_label
       dislike_label = (
           (arr[:, self.COLS['dislike']] == "1") |
           (arr[:, self.COLS['like_c']] == "1")
       ).astype(np.float32)

       # like_label (需排除dislike)
       like_label = (
           (arr[:, self.COLS['like']] == "1") |
           (arr[:, self.COLS['favorite']] == "1")
       ).astype(np.float32)
       like_label = np.where(dislike_label == 1.0, 0.0, like_label)

       # follow_label
       follow_label = (arr[:, self.COLS['follow']] == "1").astype(np.float32)

       # down/up 标签
       down = (arr[:, self.COLS['is_manu_down']] == "1").astype(np.float32)
       manu = np.where(arr[:, self.COLS['is_manu_up']] == "1", 1.0, down)

       # play_type 处理
       play_type = np.char.split(arr[:, self.COLS['play_type']], '|')
       from_down_m = np.array(['manu_down_m' in pt for pt in play_type])
       from_down_k = np.array(['manu_down_kb' in pt for pt in play_type])
       from_down_h = np.array(['manu_down_wh' in pt for pt in play_type])
       from_down = (from_down_m | from_down_k | from_down_h).astype(np.float32)
       from_up = (arr[:, self.COLS['last_video']] == "1").astype(np.float32)

       # 有效上滑标签 (考虑finish/zmtlink/follow/like)
       from_valid_up = (
           (arr[:, self.COLS['last_video']] == "1") &
           (arr[:, self.COLS['is_finish_video']] == "1")
       ).astype(np.float32)
       from_valid_up = np.where(
           (arr[:, self.COLS['last_video']] == "1") & (zmtlink_label == 1.0),
           1.0, from_valid_up
       )
       from_valid_up = np.where(
           (arr[:, self.COLS['last_video']] == "1") & (follow_label == 1.0),
           1.0, from_valid_up
       )
       from_valid_up = np.where(
           (arr[:, self.COLS['last_video']] == "1") & (like_label == 1.0),
           1.0, from_valid_up
       )

       # 完播标签
       finish = (arr[:, self.COLS['is_finish_video']] == "1").astype(np.float32)

       # 2. 计算alive基础值
       positive = np.maximum.reduce([like_label, follow_label, zmtlink_label])
       actions = np.maximum(dislike_label, positive)
       alive = np.maximum.reduce([from_up, from_down, manu, actions])

       # 3. 时长过滤 (<6秒无效)
       raw_dur = arr[:, self.COLS['duration']].astype(np.float32)
       alive = np.where(raw_dur < 6.0, 0.0, alive)

       # 4. sign过滤
       if not self.valid_sign:
           sign_mask = np.ones(batch_size, dtype=bool)
       else:
           sign_mask = np.zeros(batch_size, dtype=bool)
           for sign in self.valid_sign:
               sign_mask |= (arr[:, self.COLS['sign']] == sign)

       alive = np.where(sign_mask, alive, 0.0)

       # 5. reward过滤
       reallike = arr[:, self.COLS['logreallike']]
       split_lengths = np.array([len(s.split('|')) if s.strip() else 0 for s in reallike])
       reward_mask = split_lengths >= self.logreallike_len
       random_mask = np.random.rand(*reward_mask.shape) < 1.0
       reward_mask = reward_mask & random_mask
       alive = np.where(reward_mask | (positive == 1.), alive, 0.0)
       #alive = reward_mask

       # 6. reward supervised loss 逻辑
       #safe_input = np.where(raw_dur >= 6, raw_dur - 6, 0.0)
       #algtime = np.where(raw_dur >= 6, np.maximum(7.5, 6 * np.power(safe_input, 0.25)), float('inf'))
       #md_time = arr[:, self.COLS['manu_down_time']].astype(np.float32)
       #skip = np.where(md_time < 2., True, False)
       #algfinish = np.where(down == 1., md_time >= algtime, False).astype(np.float32)
       #algfinish = np.where(skip, -1., algfinish)

       # 7. 提取alive行 (alive > 0.5)
       alive_mask = alive > 0.5
       return arr[alive_mask, :], arr[alive_mask, self.COLS['like']]#, algfinish[alive_mask]

if __name__ == "__main__":
    import gzip, os, sys
    from parser.pt_parser import PtParser

    from arguments.data_args import DataArguments
    from arguments.model_args import ModelArguments
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    url_tokenizer = UrlTokenizer.from_args(data_args)
    reward_parser = RewardParser(data_args.columns_def, data_args.reward_serving_column_idx, tokenizer=url_tokenizer, logreallike_len=30)

    file_path = "/mnt/video.rank.s101/20250810/2025081011/20250810-11-38506.gz"

    batch_lines = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if len(batch_lines) < 1000:
                batch_lines.append(line.strip())
            else:
                break

    print(reward_parser.parser(batch_lines))

