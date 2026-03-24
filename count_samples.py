# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : count_samples.py
#   author   : ***
#   date     : 2025-07-18
#   desc     : 统计训练数据的样本数量和比例
# ======================================================== 

import os
import gzip
from collections import defaultdict
import numpy as np
from typing import List, Dict

def count_samples_in_file(file_path: str) -> Dict:
    """
    统计单个文件中的样本数量和比例
    """
    pos_count = 0
    neg_count = 0
    neutral_count = 0
    total_count = 0
    
    like_count = 0
    follow_count = 0
    dislike_count = 0
    favorite_count = 0
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 32:
                    continue
                
                total_count += 1
                
                # 解析行为标签
                like = parts[28] if len(parts) > 28 else '0'  # f029.like
                follow = parts[31] if len(parts) > 31 else '0'  # f032.follow
                dislike = parts[30] if len(parts) > 30 else '0'  # f031.dislike
                favorite = parts[88] if len(parts) > 88 else '0'  # f089.favorite
                
                # 统计行为
                if like == '1':
                    like_count += 1
                if follow == '1':
                    follow_count += 1
                if dislike == '1':
                    dislike_count += 1
                if favorite == '1':
                    favorite_count += 1
                
                # 计算advantages
                w = 0
                if like == '1':
                    w += 4
                if dislike == '1':
                    w -= 4
                if follow == '1':
                    w += 4
                if favorite == '1':
                    w += 4
                
                # 判断样本类型
                if w > 0:
                    pos_count += 1
                elif w < 0:
                    neg_count += 1
                else:
                    neutral_count += 1
                
                # 每10000行打印一次进度
                if total_count % 10000 == 0:
                    print(f"  已处理 {total_count} 行...")
                    
    except Exception as e:
        print(f"  处理文件 {file_path} 时出错: {e}")
    
    return {
        'total': total_count,
        'pos': pos_count,
        'neg': neg_count,
        'neutral': neutral_count,
        'like': like_count,
        'follow': follow_count,
        'dislike': dislike_count,
        'favorite': favorite_count
    }

def count_samples_in_directory(root_dir: str, corpus_days: int = 2, with_today: bool = True) -> Dict:
    """
    统计目录下所有文件的样本数量和比例
    """
    print(f"开始统计目录: {root_dir}")
    print(f"使用最近 {corpus_days} 天的数据")
    print(f"包含今天: {with_today}")
    print("-" * 60)
    
    # 获取天级目录
    corpus_dirs = sorted(os.listdir(root_dir))[-corpus_days:]
    if not with_today:
        corpus_dirs = corpus_dirs[:-1]
    
    total_stats = {
        'total': 0,
        'pos': 0,
        'neg': 0,
        'neutral': 0,
        'like': 0,
        'follow': 0,
        'dislike': 0,
        'favorite': 0
    }
    
    file_count = 0
    for day_dir in corpus_dirs:
        day_path = os.path.join(root_dir, day_dir)
        if not os.path.isdir(day_path):
            continue
        
        print(f"\n处理日期: {day_dir}")
        
        # 获取小时级文件
        hour_files = []
        for hour_file in os.listdir(day_path):
            if hour_file.endswith('.done'):
                hour_files.append(os.path.join(day_path, hour_file))
        
        hour_files.sort()
        
        for hour_file in hour_files:
            print(f"  处理文件: {os.path.basename(hour_file)}")
            file_stats = count_samples_in_file(hour_file)
            
            for key in total_stats:
                total_stats[key] += file_stats[key]
            
            file_count += 1
    
    print("\n" + "=" * 60)
    print("统计结果汇总")
    print("=" * 60)
    
    # 打印统计结果
    print(f"\n文件总数: {file_count}")
    print(f"样本总数: {total_stats['total']:,}")
    print(f"\n样本类型分布:")
    print(f"  正样本 (advantages > 0): {total_stats['pos']:,} ({total_stats['pos']/total_stats['total']*100:.2f}%)")
    print(f"  负样本 (advantages < 0): {total_stats['neg']:,} ({total_stats['neg']/total_stats['total']*100:.2f}%)")
    print(f"  中性样本 (advantages = 0): {total_stats['neutral']:,} ({total_stats['neutral']/total_stats['total']*100:.2f}%)")
    
    print(f"\n用户行为统计:")
    print(f"  点赞 (like): {total_stats['like']:,} ({total_stats['like']/total_stats['total']*100:.2f}%)")
    print(f"  关注 (follow): {total_stats['follow']:,} ({total_stats['follow']/total_stats['total']*100:.2f}%)")
    print(f"  不喜欢 (dislike): {total_stats['dislike']:,} ({total_stats['dislike']/total_stats['total']*100:.2f}%)")
    print(f"  收藏 (favorite): {total_stats['favorite']:,} ({total_stats['favorite']/total_stats['total']*100:.2f}%)")
    
    # 计算扩展后的样本数量
    print(f"\n样本扩展后的数量（假设原始batch）:")
    original_total = total_stats['total']
    pos_samples = total_stats['pos']
    
    # 正样本扩展: alpha=0.5
    expanded_pos = int(pos_samples * 0.5)
    
    # 负样本扩展: alpha=1.0
    expanded_neg = int(pos_samples * 1.0)
    
    expanded_total = original_total + expanded_pos + expanded_neg
    
    print(f"  原始样本: {original_total:,}")
    print(f"  正样本扩展: +{expanded_pos:,}")
    print(f"  负样本扩展: +{expanded_neg:,}")
    print(f"  扩展后总计: {expanded_total:,}")
    
    # 扩展后的比例
    final_pos = pos_samples + expanded_pos
    final_neg = total_stats['neg']
    final_neutral = total_stats['neutral'] + expanded_neg
    
    print(f"\n扩展后的样本比例:")
    print(f"  正样本: {final_pos:,} ({final_pos/expanded_total*100:.2f}%)")
    print(f"  负样本: {final_neg:,} ({final_neg/expanded_total*100:.2f}%)")
    print(f"  中性样本: {final_neutral:,} ({final_neutral/expanded_total*100:.2f}%)")
    
    return total_stats

if __name__ == "__main__":
    import sys
    
    # 默认参数
    root_dir = "/mnt_onerec_model/corpus/video.rank.s101.pos_neg"
    corpus_days = 2
    with_today = True
    
    # 如果有命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    if len(sys.argv) > 2:
        corpus_days = int(sys.argv[2])
    if len(sys.argv) > 3:
        with_today = sys.argv[3].lower() == 'true'
    
    # 统计样本
    count_samples_in_directory(root_dir, corpus_days, with_today)
