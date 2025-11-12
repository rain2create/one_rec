# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : pt_dataset_v2.py
#   author   : ***
#   date     : 2025-07-14
#   desc     : 
# ======================================================== 

import os
import datetime
import random
from turtle import mode
from typing import List, Callable, Any, Iterator, Optional
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import isal.igzip as gzip
from utils.logger import logger
from arguments.data_args import DataArguments
import torch.distributed as dist
from pathlib import Path
from itertools import islice

# pretrain时一般无法将数据全部读入内存, 因此使用IterableDataset
class PtDataset(IterableDataset):
    def __init__(
        self,
        root_dir: str,
        parser_func: Callable[[List[str]], Any],
        corpus_days: int,
        with_today: bool = True,
        batch_size: int = 1024,
        shuffle_buffer_size: int = 10000,
        compress_type: str = 'gzip',
        done_spec: str = '.done',
        file_shuffle: bool = False,
        inter_file_shuffle: bool = True,
        reward_stage: bool = False,
        reward_batch_size: int = 20,
        test_root_dirs: str = '',
    ):
        super().__init__()
        self.root_dir = root_dir
        self.parser_func = parser_func
        self.corpus_days = corpus_days
        self.with_today = with_today
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.compress_type = compress_type
        self.done_spec = done_spec
        self.done_spec = "done.{}days.{}".format(self.corpus_days, self.done_spec)
        self.file_shuffle = file_shuffle
        self.inter_file_shuffle = inter_file_shuffle
        if reward_stage:
            self.batch_size = reward_batch_size
        if not test_root_dirs:
            self.all_files = self._get_train_file_names(self.with_today)
        else:
            # 测试模式
            self.all_files = []
            for root_dir in test_root_dirs.split(","):
                self.all_files.extend(list(Path(root_dir).rglob("*.gz")))
            if self.file_shuffle:
                random.shuffle(self.all_files)
            else:
                self.all_files = sorted(self.all_files)
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        logger.info(f"start load dataset of {len(self.all_files)} files in rank {self.rank} of world size {self.world_size}")

    def _get_train_file_names(
        self,
        with_today: bool = True
    ) -> List[str]:
        # 获取待训练的语料文件
        begin_day    = datetime.date.today() - datetime.timedelta(self.corpus_days)
        begin_day    = begin_day.strftime("%Y%m%d")
        today = datetime.date.today().strftime("%Y%m%d")
        corpus_dirs = []  # 天级文件夹
        corpus_dirs_  = os.listdir(self.root_dir)
        corpus_dirs_  = sorted(filter(lambda x : x >= begin_day,  corpus_dirs_))
        if not with_today:
            corpus_dirs_  = sorted(filter(lambda x : x < today,  corpus_dirs_))
        corpus_dirs  += [os.path.join(self.root_dir, x) for x in corpus_dirs_]

        done_dirs    = []
        for day_corpus in corpus_dirs:
            hour_corpus = [os.path.join(day_corpus, hour) for hour in os.listdir(day_corpus)]
            hour_corpus = map(lambda x: x.rstrip(".done"), filter(lambda x : x.endswith(".done"), hour_corpus))
            done_dirs.extend(hour_corpus)

        if self.done_spec:
            done_dirs = filter(lambda x : os.path.exists(x) and (not os.path.exists("{}.{}".format(x, self.done_spec))), done_dirs)
        else:
            done_dirs = filter(lambda x : os.path.exists(x), done_dirs)

        need_train_files = []
        for hour_corpus in done_dirs:
            gzip_files = [os.path.join(hour_corpus, x) for x in os.listdir(hour_corpus)]
            gzip_files = filter(lambda x : x.endswith('.gz'), gzip_files)
            need_train_files.extend(gzip_files)

        # 是否需要随机打乱文件顺序
        if self.file_shuffle:
            random.shuffle(need_train_files)
            return need_train_files
        else:
            return sorted(need_train_files)

    def _shuffle_buffer_iterator(self, lines: Iterator[str]) -> Iterator[str]:
        """
        Args:
            lines: 一个文件每一行文本的迭代器
        """
        # 使用一个固定大小的buffer来shuffle (类似蓄水池采样)
        buffer = []
        it = iter(lines)
        try:
            # 先取出buffer_size个放入buffer
            for _ in range(self.shuffle_buffer_size):
                buffer.append(next(it))
        except StopIteration:
            random.shuffle(buffer)
            for x in buffer:
                yield x
            return

        # 对buffer做一次shuffle
        random.shuffle(buffer)

        for line in it:
            idx = random.randint(0, self.shuffle_buffer_size - 1)
            yield buffer[idx]
            buffer[idx] = line

        for x in buffer:
            yield x

    def check_line(self, line):
        secs = line.strip().split(',')
        flag = False
        if len(secs) < 130:
            return flag
        if secs[129] in ('dj', 'xsv') :
            if random.random() > 0.1:
                return flag
        try:
            duration = float(secs[4])
            if duration < 9.0:
                return flag
        except Excpetion as err:
            return flag
        # remove likc_c
        if secs[29] == '1' or secs[89] == '1':
            return flag
        # from menu up
        if secs[49] == '1' and secs[20] == '1':
            flag = True
        # follow
        if secs[31] == '1':
            flag = True
        # like
        if secs[28] == '1' or secs[88] == '1':
            flag = True
        # panel
        if secs[32] == '1':
            flag = True
        return flag

    def _parse_file(self, file_path: str) -> Iterator[Any]:
        open_fn = gzip.open if self.compress_type == 'gzip' else open
        with open_fn(file_path, "rt", encoding="utf-8") as f:
            # lines = (line.strip() for line in f if line.strip())
            try:
                lines = (line.strip() for line in f if self.check_line(line))
                if self.inter_file_shuffle and self.shuffle_buffer_size > 0:
                    lines = self._shuffle_buffer_iterator(lines)
                while batch := list(islice(lines, self.batch_size)):
                    yield self.parser_func(batch)
            except Exception as err:
                print(err)
                return

            '''
            batch = []
            for line in lines:
                batch.append(line)
                if len(batch) == self.batch_size:
                    yield self.parser_func(batch)
                    batch = []
            if batch:
                yield self.parser_func(batch)
            '''

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # 分布式：先按照 rank 分文件
        files_per_rank = self.all_files[self.rank::self.world_size]

        # 每个 worker 处理其中的一部分
        local_chunks = files_per_rank[worker_id::num_workers]
        # logger.info(f"[RANK {self.rank} WORKER {worker_id}] processing {len(local_chunks)} of total {len(self.all_files)} files")

        for file_path in local_chunks:
            for sample in self._parse_file(file_path):
                yield sample

    def after_train_process(self):
        file_names = self.all_files
        done_dirs = [os.path.dirname(filename) for filename in file_names]
        done_dirs = set([done_dir for done_dir in done_dirs if os.path.isdir(done_dir)])
        for done in done_dirs:
            done_spec = "{}.{}".format(done, self.done_spec)
            if os.path.exists(done_spec):
                os.system("rm -rf %s" %(done_spec))
            os.system("touch %s" %(done_spec))
        return 0

    @classmethod
    def from_args(cls, args: DataArguments, parser_func: Callable[[List[str]], Any], test_root_dirs: Optional[List[str]] = None):
        return cls(
            root_dir=args.root_dir,
            parser_func=parser_func,
            corpus_days=args.corpus_days,
            with_today=args.with_today,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            compress_type=args.compress_type,
            done_spec=args.done_spec,
            file_shuffle=args.file_shuffle,
            inter_file_shuffle=args.inter_file_shuffle,
            reward_stage=args.reward_stage,
            reward_batch_size=args.reward_batch_size,
            test_root_dirs=test_root_dirs
        )


if __name__ == "__main__":
    from collections import defaultdict
    from utils.column import columns_def
    from parser.pt_parser import PtParser
    from parser.tokenizer import UrlTokenizer
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from pathlib import Path
    
    AutoTokenizer.register("md5_tokenizer", UrlTokenizer)
    tokenizer_data_dir = Path("../data/tokenizer_data")
    url_tokenizer = AutoTokenizer.from_pretrained(tokenizer_data_dir)

    # 创建PtParser实例
    pt_parser: PtParser = PtParser(columns_def, tokenizer=url_tokenizer)

    dataset = PtDataset(
        root_dir="/mnt/video.rank.s101",
        parser_func=pt_parser.parser,
        corpus_days=1,
        with_today=False,
        batch_size=1024,
        shuffle_buffer_size=10000,
        compress_type="gzip",
        done_spec="onerec_pt",
        file_shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=10,
        pin_memory=True
    )

    for x in tqdm(dataloader):
        pass
