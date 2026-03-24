#!/usr/bin/env python3
"""
读取 experiments/实验结果/ 目录下的所有 docx 文件，生成 experiments.md
"""

import os
from docx import Document

def read_docx(filepath):
    """读取docx文件内容"""
    try:
        doc = Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def extract_key_info(text, filename):
    """从文本中提取关键信息"""
    lines = text.strip().split('\n')
    
    # 尝试提取标题（第一个非空行）
    title = filename.replace('.docx', '')
    for line in lines[:10]:
        line = line.strip()
        if line and len(line) > 3:
            title = line
            break
    
    # 提取关键内容（非空行）
    content_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]
    
    return {
        'title': title,
        'content': content_lines[:30]  # 取前30行关键内容
    }

def main():
    base_dir = '/mnt_onerec_model/jobs/onerec_gen_pos_v0_seq_encoding_256_s64_lazy_decoder_finish_expl/src/docs/experiments/实验结果'
    
    # 列出所有docx文件
    docx_files = [f for f in os.listdir(base_dir) if f.endswith('.docx')]
    docx_files.sort()
    
    print(f"找到 {len(docx_files)} 个实验文档:\n")
    
    # 读取每个文件
    experiments = []
    for filename in docx_files:
        filepath = os.path.join(base_dir, filename)
        text = read_docx(filepath)
        info = extract_key_info(text, filename)
        experiments.append({
            'filename': filename,
            'title': info['title'],
            'content': info['content']
        })
        
        print(f"\n{'='*60}")
        print(f"文件: {filename}")
        print(f"标题: {info['title']}")
        print(f"内容预览:")
        for line in info['content'][:10]:
            print(f"  {line[:80]}")
    
    # 生成 experiments.md
    output_path = '/mnt_onerec_model/jobs/onerec_gen_pos_v0_seq_encoding_256_s64_lazy_decoder_finish_expl/src/docs/experiments.md'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# One-Rec 实验总结\n\n")
        f.write("本文档汇总了所有专项实验的结果。\n\n")
        f.write("## 实验列表\n\n")
        
        for i, exp in enumerate(experiments, 1):
            f.write(f"### {i}. {exp['title']}\n\n")
            f.write(f"**文件**: `{exp['filename']}`\n\n")
            f.write("**实验内容**: \n")
            for line in exp['content']:
                if line.strip():
                    f.write(f"- {line}\n")
            f.write("\n---\n\n")
    
    print(f"\n\n已生成: {output_path}")

if __name__ == '__main__':
    main()
