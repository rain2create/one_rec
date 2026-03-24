import math
from collections import defaultdict 
import textwrap
from datetime import datetime
 
def calculate_and_print_tde(file_path):
    """计算并打印三层聚类的Token Distribution Entropy"""
    # 初始化数据结构
    layers = {
        "第一层聚类": defaultdict(int),
        "第二层聚类": defaultdict(int),
        "第三层聚类": defaultdict(int)
    }
    total_samples = 0
    valid_lines = 0 
    
    # 读取文件并统计频率 
    try:
        with open(file_path, 'r') as f:
            print(f"📂 开始分析聚类结果文件: {file_path}")
            
            for line_num, line in enumerate(f, 1):
                line = line.strip() 
                if not line:
                    continue
                    
                parts = line.split('\t') 
                if len(parts) < 2:
                    print(f"⚠️ 第 {line_num} 行格式错误: 缺少聚类ID部分")
                    continue
                    
                cluster_ids = parts[1].split()
                if len(cluster_ids) < 3:
                    print(f"⚠️ 第 {line_num} 行格式错误: 聚类ID不足3个")
                    continue
                    
                try:
                    id1 = int(cluster_ids[0])
                    id2 = int(cluster_ids[1])
                    id3 = int(cluster_ids[2])
                    
                    layers["第一层聚类"][id1] += 1
                    layers["第二层聚类"][id2] += 1 
                    layers["第三层聚类"][id3] += 1
                    valid_lines += 1 
                    total_samples += 1
                except ValueError as e:
                    print(f"⚠️ 第 {line_num} 行转换错误: {e}")
                    continue 
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {file_path} 不存在")
        return
    except Exception as e:
        print(f"❌ 文件读取错误: {str(e)}")
        return 
    
    # 文件读取报告 
    print("\n🔍 文件分析报告:")
    print(f"  总行数: {line_num}")
    print(f"  有效数据行: {valid_lines} ({valid_lines/line_num:.1%})")
    print(f"  总样本量: {total_samples}")
    
    # 计算各层TDE 
    tde_results = {}
    for layer_name, counts in layers.items(): 
        if not counts:
            tde_results[layer_name] = {"TDE": 0.0}
            continue 
            
        # 计算频率分布
        unique_ids = len(counts)
        probabilities = [count / total_samples for count in counts.values()] 
        
        # 计算信息熵
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p) 
        
        # 计算最大可能熵
        max_entropy = math.log2(unique_ids)  if unique_ids > 0 else 0 
        
        # 计算熵占比 
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0 
        
        tde_results[layer_name] = {
            'TDE': entropy,
            'Max_Entropy': max_entropy,
            'Entropy_Ratio': entropy_ratio,
            'Unique_Clusters': unique_ids,
            'Min_Count': min(counts.values()), 
            'Max_Count': max(counts.values()) 
        }
    
    # 美化打印结果 
    print("\n" + "="*60)
    print("🔬 三层聚类 Token Distribution Entropy (TDE) 分析报告")
    print(f"📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("="*60)
    
    # 表头
    print("\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "聚类层", "TDE(bits)", "最大熵", "熵占比", "聚类中心数", "最小簇大小", "最大簇大小"
    ))
    print("-"*80)
    
    # 数据行
    for layer, data in tde_results.items(): 
        print("{:<14} {:<12.4f} {:<12.4f} {:<12.2%} {:<12} {:<12} {:<12}".format(
            layer.split()[0],  
            data['TDE'],
            data['Max_Entropy'],
            data['Entropy_Ratio'],
            data['Unique_Clusters'],
            data['Min_Count'],
            data['Max_Count']
        ))
    
    # 优化建议
    print("\n" + "="*60)
    print("💡 聚类系统优化建议")
    print("="*60)
    
    suggestions = []
    layer1 = tde_results["第一层聚类"]
    layer3 = tde_results["第三层聚类"]
    
    # 第一层优化建议
    if layer1['Entropy_Ratio'] < 0.85:
        new_count = int(layer1['Unique_Clusters'] * 1.15)
        suggestions.append( 
            f"• 第一层熵占比 {layer1['Entropy_Ratio']:.2%} 略低，建议增加聚类中心数量至约 {new_count} 个"
        )
    
    # 第三层优化建议
    if layer3['Max_Count'] > total_samples * 0.05:
        imbalance = layer3['Max_Count'] / total_samples
        suggestions.append( 
            f"• 第三层检测到最大簇占比 {imbalance:.2%}，建议细分过大的聚类簇"
        )
    
    # 层间平衡建议
    ratio_diff = layer3['Unique_Clusters'] / layer1['Unique_Clusters']
    if ratio_diff < 4 or ratio_diff > 8:
        suggestions.append( 
            f"• 聚类中心数量比例 (三层/一层={ratio_diff:.1f}) 超出理想范围(5-7)，应调整层级粒度"
        )
    
    # 输出建议或成功消息 
    if suggestions:
        print("\n".join(suggestions))
    else:
        print("✅ 当前聚类层级分布均衡，无需重大调整")
    
    # 返回计算结果 
    return tde_results 
 
# 执行计算并打印结果 
file_path = "../data/clustering_results"
tde_results = calculate_and_print_tde(file_path)
