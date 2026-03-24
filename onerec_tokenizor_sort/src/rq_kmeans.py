# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : rq_kmeans2.py
#   author   : ***
#   date     : 2025-07-07
#   desc     : 
# ======================================================== 
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
import json
from tqdm import tqdm
import time
import h5py

def read_embeddings_hdf5(input_file, hdf5_path="../data/embeddings.hdf5"):
    """
    高效读取大规模嵌入向量文件，使用HDF5格式存储
    
    参数:
    input_file (str): 输入文件路径
    hdf5_path (str): HDF5缓存文件路径
    
    返回:
    tuple: (md5列表, 嵌入向量数组路径)
    """
    # 如果HDF5文件已存在，直接使用
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            md5s = list(hf['md5s'][:])
        return md5s, hdf5_path
    
    # 否则创建HDF5文件
    md5_list = []
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    with h5py.File(hdf5_path, 'w') as hf, \
         open(input_file, 'r', encoding='utf-8') as f, \
         tqdm(total=total_lines, desc="转换数据到HDF5") as pbar:
        
        # 先收集所有MD5
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                pbar.update(1)
                continue
            md5_list.append(parts[0].encode('utf-8'))
            pbar.update(1)
        
        # 创建数据集
        hf.create_dataset('md5s', data=np.array(md5_list, dtype='S32'))
        
        # 重置文件指针
        f.seek(0)
        pbar.reset()
        
        # 创建嵌入向量数据集（使用float32节省空间）
        embeddings_dset = hf.create_dataset('embeddings', 
                                           shape=(len(md5_list), 64), 
                                           dtype=np.float32)
        
        # 填充嵌入向量
        idx = 0
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2 or len(parts[1].split()) != 64:
                pbar.update(1)
                continue
            
            emb_str = parts[1].split()
            try:
                emb = [float(x) for x in emb_str]
                embeddings_dset[idx] = emb
                idx += 1
            except ValueError:
                continue
            pbar.update(1)
    
    return md5_list, hdf5_path

def hierarchical_clustering_large(input_file, K, N, output_centers_dir, batch_size=100000):
    """
    高效执行大规模多层次聚类
    
    参数:
    input_file (str): 输入文件路径
    K (int): 每次聚类的簇数
    N (int): 迭代次数
    output_centers_dir (str): 聚类中心输出目录
    batch_size (int): 批量处理大小
    
    返回:
    dict: 包含每个视频MD5及其聚类索引序列的字典
    """
    # 确保输出目录存在
    os.makedirs(output_centers_dir, exist_ok=True)
    
    # 读取数据到HDF5
    print("开始读取数据...")
    md5_list, hdf5_path = read_embeddings_hdf5(input_file)
    
    with h5py.File(hdf5_path, 'r') as hf:
        total_samples = len(hf['md5s'])
        print(f"总样本数: {total_samples:,}")
        
        # 初始化结果存储
        results_file = os.path.join(output_centers_dir, "clustering_results.hdf5")
        with h5py.File(results_file, 'w') as results_hf:
            # 创建存储聚类结果的dataset
            cluster_labels_dset = results_hf.create_dataset('cluster_labels', 
                                                          shape=(total_samples, N), 
                                                          dtype=np.uint16)
            
            # 存储MD5
            results_hf.create_dataset('md5s', data=hf['md5s'][:])
            
            # 初始化嵌入向量
            X = hf['embeddings'][:]
            
            # 迭代聚类
            for round_idx in range(1, N+1):
                start_time = time.time()
                print(f"\n开始第 {round_idx}/{N} 轮聚类...")
                
                # 使用MiniBatchKMeans处理大规模数据
                kmeans = MiniBatchKMeans(n_clusters=K, 
                                        batch_size=batch_size, 
                                        n_init=3, 
                                        random_state=0,
                                        compute_labels=False)
                
                # 分批拟合模型
                for i in tqdm(range(0, total_samples, batch_size), desc="聚类处理"):
                    batch = X[i:i+batch_size]
                    kmeans.partial_fit(batch)

                # shuffle后拟合模型
                shuffled_indices = np.random.permutation(total_samples) 
                for start_idx in tqdm(range(0, total_samples, batch_size), desc="shuffle后聚类处理"):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch_indices = shuffled_indices[start_idx:end_idx]
                    batch = X[batch_indices]
                    kmeans.partial_fit(batch)
                
                # 预测聚类标签（分批进行）
                cluster_labels = np.zeros(total_samples, dtype=np.uint16)
                for i in tqdm(range(0, total_samples, batch_size), desc="标签预测"):
                    batch = X[i:i+batch_size]
                    labels = kmeans.predict(batch)
                    cluster_labels[i:i+batch_size] = labels
                    # 存储当前轮次的聚类标签
                    cluster_labels_dset[i:i+batch_size, round_idx-1] = labels
                
                # 获取聚类中心和计数
                cluster_centers = kmeans.cluster_centers_
                unique, counts = np.unique(cluster_labels, return_counts=True)
                cluster_counts = np.zeros(K, dtype=int)
                for u, c in zip(unique, counts):
                    if u < K:  # 确保索引在范围内
                        cluster_counts[u] = c
                
                # 保存聚类中心信息
                output_center_file = os.path.join(output_centers_dir, f"round_{round_idx}_centers.txt")
                with open(output_center_file, 'w') as f:
                    f.write(f"cluster_index\tcluster_size\tcenter_vector\n")
                    for center_idx in range(K):
                        center_vector = " ".join([f"{x:.6f}" for x in cluster_centers[center_idx]])
                        f.write(f"{center_idx+1}\t{cluster_counts[center_idx]}\t{center_vector}\n")
                
                # 保存聚类覆盖资源统计
                coverage_file = os.path.join(output_centers_dir, f"round_{round_idx}_coverage.json")
                with open(coverage_file, 'w') as f:
                    json.dump({
                        "round": round_idx,
                        "total_samples": total_samples,
                        "cluster_counts": {str(k+1): int(v) for k, v in enumerate(cluster_counts)}
                    }, f, indent=2)
                
                # 打印统计信息
                print(f"  - 第 {round_idx} 轮聚类完成, 耗时: {time.time()-start_time:.2f}秒")
                print(f"  - 聚类覆盖统计: 最小={cluster_counts.min()}, 最大={cluster_counts.max()}, 平均={cluster_counts.mean():.1f}")
                
                # 计算残差（分批进行）
                for i in tqdm(range(0, total_samples, batch_size), desc="计算残差"):
                    batch = X[i:i+batch_size]
                    labels_batch = cluster_labels[i:i+batch_size]
                    # 计算残差：当前向量减去聚类中心
                    X[i:i+batch_size] = batch - cluster_centers[labels_batch]
        
        print(f"\n聚类完成！结果已保存至 {results_file}")
        return results_file

def convert_results_to_text(hdf5_results, output_file):
    """将HDF5聚类结果转换为文本格式"""
    with h5py.File(hdf5_results, 'r') as hf, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        md5s = hf['md5s'][:]
        cluster_labels = hf['cluster_labels'][:]
        
        for i in tqdm(range(len(md5s)), desc="转换结果"):
            md5 = md5s[i].decode('utf-8')
            labels = " ".join([str(x+1) for x in cluster_labels[i]])
            out_f.write(f"{md5}\t{labels}\n")

if __name__ == "__main__":
    # 配置参数
    input_file = "../data/video_emb"  # 输入文件路径
    output_file = "../data/clustering_results"  # 聚类结果输出文件
    centers_dir = "../data/cluster_centers"  # 聚类中心输出目录
    K = 1500  # 每次聚类的簇数
    N = 3  # 迭代次数
    batch_size = 100000  # 批量大小（根据内存调整）
    
    # 执行聚类
    results_hdf5 = hierarchical_clustering_large(input_file, K, N, centers_dir, batch_size)
    
    # 转换为文本格式
    output_text_file = "../data/clustering_results"
    convert_results_to_text(results_hdf5, output_text_file)
    print(f"文本格式结果已保存至 {output_text_file}") 
