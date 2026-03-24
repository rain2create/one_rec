# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : upload_new.py
#   author   : ***
#   date     : 2025-08-28
#   desc     : 
# ======================================================== 
import numpy as np
import os
import h5py
from tqdm import tqdm
import time
import shutil

def copy_if_different_size(input_file, remote_file):
    """
    检查remote_file与input_file的大小是否相同，不同则复制并返回True，否则返回False

    参数:
        input_file: 目标文件路径
        remote_file: 源文件路径

    返回:
        bool: 如果执行了复制操作则返回True，否则返回False
    """
    try:
        # 获取两个文件的大小
        input_size = os.path.getsize(input_file)
        remote_size = os.path.getsize(remote_file)
        # 比较大小
        if input_size != remote_size:
            # 复制文件，覆盖目标文件
            shutil.copy2(remote_file, input_file)
            return True
        return False

    except FileNotFoundError as e:
        print(f"错误：文件不存在 - {e.filename}")
        raise  # 可以根据需要选择是否抛出异常或返回False
    except Exception as e:
        print(f"操作出错：{str(e)}")
        raise  # 可以根据需要选择是否抛出异常或返回False

def load_existing_results(clustering_results_file):
    """加载已有聚类结果，返回MD5集合和聚类中心信息"""
    existing_md5s = set()
    existing_md5s_l3cnt = dict()

    if os.path.exists(clustering_results_file):
        print(f"加载已有聚类结果: {clustering_results_file}")
        with open(clustering_results_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取已有结果"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    existing_md5s.add(parts[0])
                    codes = parts[1].split(' ')
                    if len(codes) >= 3:
                        key = '{}_{}'.format(codes[0], codes[1])
                        cur_cnt = existing_md5s_l3cnt.get(key, 0)
                        existing_md5s_l3cnt[key] = cur_cnt + 1
    return existing_md5s, existing_md5s_l3cnt

def load_cluster_centers(centers_dir, N):
    """加载所有轮的聚类中心"""
    centers = []
    for round_idx in range(1, N+1):
        center_file = os.path.join(centers_dir, f"round_{round_idx}_centers.txt")
        if not os.path.exists(center_file):
            raise FileNotFoundError(f"聚类中心文件不存在: {center_file}")

        round_centers = []
        with open(center_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    # 解析中心向量
                    center_vector = np.array([float(x) for x in parts[2].split()])
                    round_centers.append(center_vector)

        centers.append(np.array(round_centers))
        print(f"加载第 {round_idx} 轮聚类中心，共 {len(round_centers)} 个中心")

    return centers

def predict_new_embeddings(input_file, existing_md5s, existing_md5s_l3cnt, centers, output_file):
    """预测新嵌入向量的聚类结果"""
    # 读取新嵌入向量
    new_md5s = []
    new_embeddings = []

    print(f"读取新嵌入向量文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理嵌入向量"):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            md5 = parts[0]
            # 跳过已有结果的MD5
            if md5 in existing_md5s:
                continue

            emb_str = parts[1].split()
            if len(emb_str) != 64:
                continue

            try:
                emb = np.array([float(x) for x in emb_str])
                new_md5s.append(md5)
                new_embeddings.append(emb)
            except ValueError:
                continue

    if not new_md5s:
        print("没有发现新的嵌入向量")
        return

    print(f"发现 {len(new_md5s)} 个新嵌入向量")
    new_embeddings = np.array(new_embeddings)

    # 预测聚类标签
    all_labels = []
    start_time = time.time()

    with open(output_file, 'a', encoding='utf-8') as op:
        for i, emb in enumerate(tqdm(new_embeddings, desc="预测聚类标签")):
            labels = []
            current_emb = emb.copy()

            # 逐轮预测
            for j, round_centers in enumerate(centers):
                # 计算到所有聚类中心的距离
                distances = np.linalg.norm(round_centers - current_emb, axis=1)
                if j < 2:
                    # 找到最近的聚类中心
                    closest_idx = np.argmin(distances)
                else:
                    key = '{}_{}'.format(labels[0], labels[1])
                    cur_cnt = existing_md5s_l3cnt.get(key, 0)
                    closest_idx = cur_cnt % 1500
                    existing_md5s_l3cnt[key] = cur_cnt + 1
                labels.append(str(closest_idx + 1))  # 索引从1开始

                # 计算残差用于下一轮
                current_emb = current_emb - round_centers[closest_idx]

            op.write('{}\t{}\n'.format(new_md5s[i], ' '.join(labels)))

    print(f"预测完成，耗时: {time.time()-start_time:.2f}秒")
    print(f"结果已追加到: {output_file}")

def main():
    # 配置参数
    remote_file = "/home/s/apps/verttagembcf_large_train/data/url_emb_mask" # tagemb文件路径
    input_file = "../data/video_emb"  # 新输入文件路径
    output_file = "../data/clustering_results"  # 聚类结果输出文件
    centers_dir = "../data/cluster_centers"  # 聚类中心目录
    N = 3  # 迭代次数

    has_new_file = copy_if_different_size(input_file, remote_file)

    if has_new_file:

        # 加载已有结果
        existing_md5s, existing_md5s_l3cnt = load_existing_results(output_file)

        # 加载聚类中心
        centers = load_cluster_centers(centers_dir, N)

        # 预测新嵌入向量
        predict_new_embeddings(input_file, existing_md5s, existing_md5s_l3cnt, centers, output_file)

if __name__ == "__main__":
    main()

