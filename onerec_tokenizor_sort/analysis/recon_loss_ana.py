import numpy as np 
import os 
from sklearn.metrics  import mean_squared_error 
from tqdm import tqdm 
from datetime import datetime 
 
# ============================================================================= 
# 配置文件路径 
# ============================================================================= 
DATA_DIR = '../data'
 
# 输入文件路径配置 
FILE_PATHS = {
    "emb": os.path.join(DATA_DIR,  'video_emb'), 
    "clusters": os.path.join(DATA_DIR,  'clustering_results'),
    "center_rounds": {
        1: os.path.join(DATA_DIR,  'cluster_centers/round_1_centers.txt'), 
        2: os.path.join(DATA_DIR,  'cluster_centers/round_2_centers.txt'), 
        3: os.path.join(DATA_DIR,  'cluster_centers/round_3_centers.txt') 
    }
}
 
# 输出文件路径 
OUTPUT_PATH = os.path.join(DATA_DIR,  'resource_mse_results.tsv') 
 
# ============================================================================= 
# 数据加载函数 
# ============================================================================= 
def load_md5_embeddings():
    """
    从单个文本文件加载MD5和对应的嵌入向量 
    
    文件格式：每行包含MD5值 + '\\t' + 空格分隔的嵌入向量值
    示例：abc123\t0.1 0.2 0.3 ... 0.768
    
    返回:
        dict: {md5: numpy数组(嵌入向量)}
    """
    file_path = FILE_PATHS["emb"]
    print(f"[INFO] 开始加载嵌入向量文件: {file_path}")
    
    if not os.path.exists(file_path): 
        print(f"[ERROR] 文件不存在: {file_path}")
        return {}
    
    md5_emb_dict = {}
    line_count = 0
    success_count = 0 
    error_count = 0 
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析嵌入向量"):
                line_count += 1 
                line = line.strip() 
                
                # 跳过空行 
                if not line:
                    continue
                
                # 分割MD5和嵌入向量部分 
                parts = line.split('\t') 
                if len(parts) < 2:
                    error_count += 1
                    continue 
                
                md5 = parts[0].strip()
                vector_str = parts[1].strip()
                
                try:
                    # 将空格分隔的字符串转换为浮点数数组
                    vector = np.array([float(x)  for x in vector_str.split()]) 
                    md5_emb_dict[md5] = vector 
                    success_count += 1 
                except (ValueError, TypeError):
                    error_count += 1 
    
    except Exception as e:
        print(f"[ERROR] 文件加载失败: {str(e)}")
        return {}
    
    # 统计报告 
    print(f"[INFO] 加载完成: 总行数={line_count}, 成功解析={success_count}, 错误行数={error_count}")
    
    # 检查向量维度一致性 
    if md5_emb_dict:
        first_vector = next(iter(md5_emb_dict.values())) 
        dim = len(first_vector)
        inconsistent = 0 
        
        for md5, vector in md5_emb_dict.items(): 
            if len(vector) != dim:
                inconsistent += 1
        
        if inconsistent > 0:
            print(f"[WARN] 发现{inconsistent}个向量的维度不一致(应为{dim})")
    
    return md5_emb_dict

def load_cluster_results():
    """加载聚类结果（MD5 -> 三层聚类ID）"""
    print(f"[{get_time()}] 加载聚类结果...")
    results = {}
    
    try:
        with open(FILE_PATHS["clusters"], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t') 
                if len(parts) < 2:
                    continue 
                
                md5 = parts[0]
                cluster_ids = tuple(map(int, parts[1].split()))
                results[md5] = cluster_ids 
        
        return results 
    except Exception as e:
        print(f"加载失败: {str(e)}")
        raise 
 
def load_cluster_centers(round_num):
    """加载指定层级的聚类中心"""
    print(f"[{get_time()}] 加载第{round_num}层聚类中心...")
    centers = {}
    
    try:
        file_path = FILE_PATHS["center_rounds"][round_num]
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行 
            
            for line in f:
                parts = line.strip().split('\t') 
                if len(parts) < 3:
                    continue 
                
                cluster_id = int(parts[0])
                center_vector = np.fromstring(parts[2],  sep=' ')
                centers[cluster_id] = center_vector 
        
        return centers 
    except Exception as e:
        print(f"层级{round_num}加载失败: {str(e)}")
        raise 
 
# ============================================================================= 
# 核心计算函数 
# ============================================================================= 
def calculate_centroid_sum(cluster_ids, center_dicts):
    """计算三层聚类中心的向量和"""
    try:
        v1 = center_dicts[1].get(cluster_ids[0])
        v2 = center_dicts[2].get(cluster_ids[1])
        v3 = center_dicts[3].get(cluster_ids[2])
        
        return np.add(np.add(v1,  v2), v3)
    except TypeError:
        print(f"缺失聚类中心: IDs={cluster_ids}")
        return None 
 
def compute_mse(resource_emb, centroid_sum):
    """计算向量间的MSE"""
    if centroid_sum is None or resource_emb is None:
        return float('nan')
    
    return mean_squared_error(
        resource_emb.astype(np.float32), 
        centroid_sum.astype(np.float32) 
    )
 
# ============================================================================= 
# 实用函数 
# ============================================================================= 
def get_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d  %H:%M")
 
def save_results(results, output_path):
    """保存结果到TSV文件"""
    print(f"[{get_time()}] 保存计算结果...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MD5\tRound1_ID\tRound2_ID\tRound3_ID\tMSE_Score\n") 
            
            for md5, cluster_ids, mse in results:
                f.write(f"{md5}\t{cluster_ids[0]}\t{cluster_ids[1]}\t{cluster_ids[2]}\t{mse:.8f}\n") 
        
        print(f"结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存失败: {str(e)}")
 
# ============================================================================= 
# 主流程控制 
# ============================================================================= 
def main():
    print("\n" + "="*60)
    print("多层聚类资源向量分析")
    print("="*60 + "\n")
    
    # 阶段1：数据加载 
    print(f"[{get_time()}] 开始数据加载...")
    md5_to_emb = load_md5_embeddings()
    md5_to_clusters = load_cluster_results()
    
    center_dicts = {
        1: load_cluster_centers(1),
        2: load_cluster_centers(2),
        3: load_cluster_centers(3)
    }
    
    # 阶段2：计算处理 
    print(f"\n[{get_time()}] 开始多层聚类分析...")
    results = []
    stats = {"success": 0, "missing_md5": 0, "missing_center": 0}
    
    for md5, cluster_ids in tqdm(md5_to_clusters.items(),  desc="处理资源"):
        if md5 not in md5_to_emb:
            stats["missing_md5"] += 1 
            continue 
        
        if (cluster_ids[0] not in center_dicts[1] or 
            cluster_ids[1] not in center_dicts[2] or 
            cluster_ids[2] not in center_dicts[3]):
            stats["missing_center"] += 1 
            continue 
        
        resource_emb = md5_to_emb[md5]
        centroid_sum = calculate_centroid_sum(cluster_ids, center_dicts)
        mse = compute_mse(resource_emb, centroid_sum)
        
        results.append((md5,  cluster_ids, mse))
        stats["success"] += 1 
    
    # 阶段3：结果输出 
    print(f"\n[{get_time()}] 输出分析结果...")
    save_results(results, OUTPUT_PATH)
    
    # 阶段4：生成统计报告 
    print("\n分析统计报告:")
    print(f"  成功处理: {stats['success']} 个资源 ({stats['success']/len(md5_to_clusters)*100:.1f}%)")
    print(f"  缺失向量: {stats['missing_md5']} 个资源")
    print(f"  缺失聚类中心: {stats['missing_center']} 个资源")
    
    if results:
        mse_values = [mse for _, _, mse in results]
        print("\nMSE统计指标:")
        print(f"  平均值: {np.nanmean(mse_values):.6f}") 
        print(f"  中位数: {np.nanmedian(mse_values):.6f}") 
        print(f"  最小值: {np.nanmin(mse_values):.6f}") 
        print(f"  最大值: {np.nanmax(mse_values):.6f}") 
        print(f"  标准差: {np.nanstd(mse_values):.6f}") 
 
# ============================================================================= 
# 执行入口 
# ============================================================================= 
if __name__ == "__main__":
    main()
