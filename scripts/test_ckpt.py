# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : test_ckpt.py
#   author   : ***
#   date     : 2025-08-29
#   desc     : 
# ======================================================== 

import torch
import hashlib
import os

def file_md5(fname):
    """计算文件的MD5值"""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compare_ckpts(ckpt_paths, model_class=None):
    """
    对比多个ckpt文件
    ckpt_paths: list of str, 文件路径
    model_class: 可选，如果传入模型类，会加载并打印参数均值
    """
    print(">>> 检查 checkpoint 文件一致性")
    md5s = {}
    for path in ckpt_paths:
        md5 = file_md5(path)
        size = os.path.getsize(path)
        md5s[path] = (md5, size)
        print(f"{path} | size={size/1024/1024:.2f} MB | md5={md5}")

    unique = set([m[0] for m in md5s.values()])
    if len(unique) == 1:
        print("⚠️  所有 ckpt 文件完全一致（字节级别相同）")
    else:
        print("✅ ckpt 文件之间存在差异")

    if model_class is not None:
        print("\n>>> 检查模型参数统计信息")
        for path in ckpt_paths:
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model = model_class()
            model.load_state_dict(state_dict, strict=False)

            params = torch.cat([p.flatten() for p in model.parameters()])
            print(f"{path}: mean={params.mean().item():.6f}, std={params.std().item():.6f}")

ckpt_paths = ["../data/pt_model/checkpoint-280000.pt", "../data/pt_model/checkpoint-290000.pt"]
compare_ckpts(ckpt_paths, model_class=None)
