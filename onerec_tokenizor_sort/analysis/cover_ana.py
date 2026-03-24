import os 
from collections import defaultdict
from datetime import datetime 
 
def analyze_cluster_coverage():
    """分析三层聚类ID组合的覆盖率"""
    # 文件路径配置 
    file_path = "../data/clustering_results"
    
    if not os.path.exists(file_path): 
        print(f"[ERROR] 文件不存在: {file_path}")
        return
 
    # 初始化数据结构 
    cluster_combinations = defaultdict(int)  # 记录每种组合的出现次数
    total_resources = 0                     # 总资源数
    unique_combinations = set()             # 唯一组合集合
    
    print(f"[{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}] 开始分析文件: {file_path}")
 
    # 读取文件并统计组合 
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip() 
            if not line:
                continue 
                
            parts = line.split('\t') 
            if len(parts) < 2:
                continue
                
            md5 = parts[0]
            ids = parts[1].split()
            if len(ids) != 3:
                continue
                
            try:
                # 构造三层聚类ID组合（元组可哈希）
                combo = (int(ids[0]), int(ids[1]), int(ids[2]))
                cluster_combinations[combo] += 1 
                unique_combinations.add(combo) 
                total_resources += 1 
            except ValueError:
                continue
    
    # 计算结果指标 
    total_possible_combinations = 1500 ** 3  # 全部可能的组合数 
    coverage_ratio = len(unique_combinations) / total_possible_combinations * 100 
    
    # 统计组合出现频率分布 
    combo_freq_dist = defaultdict(int)
    for count in cluster_combinations.values(): 
        combo_freq_dist[count] += 1
 
    # 打印分析报告
    print("\n" + "="*60)
    print("🔍 三层聚类ID组合覆盖率分析报告")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("="*60)
    
    print(f"\n📊 基础统计:")
    print(f"  总资源数量: {total_resources:,}")
    print(f"  唯一组合数量: {len(unique_combinations):,}")
    print(f"  理论可能组合: {total_possible_combinations:,} (1000^3)")
    
    print(f"\n📈 覆盖率分析:")
    print(f"  实际覆盖率: {coverage_ratio:.6f}%")
    print(f"  平均每个组合的资源数: {total_resources/len(unique_combinations):.2f}")
    
    #print("\n🔢 组合频率分布 (出现次数 → 组合数量):")
    #for freq, count in sorted(combo_freq_dist.items(),  key=lambda x: x[0]):
    #    print(f"  {freq:>3}次 → {count:>6,}种 ({count/len(unique_combinations):.2%})")
    
    # 返回关键数据
    return {
        "total_resources": total_resources,
        "unique_combinations": len(unique_combinations),
        "coverage_ratio": coverage_ratio,
        "combo_freq_dist": dict(combo_freq_dist)
    }
 
# 执行分析
results = analyze_cluster_coverage()
