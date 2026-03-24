#!/usr/bin/env python3
"""
读取 experiments/实验结果/ 目录下的所有 docx 文件，提取完整内容（包括表格）
"""

import os
from docx import Document

def read_docx_full(filepath):
    """完整读取docx文件，包括段落和表格"""
    try:
        doc = Document(filepath)
        
        # 读取所有段落
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
        
        # 读取所有表格
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)
        
        return {
            'paragraphs': paragraphs,
            'tables': tables
        }
    except Exception as e:
        return {'paragraphs': [f"Error: {e}"], 'tables': []}

def format_markdown_table(table_data):
    """将表格数据格式化为Markdown表格"""
    if not table_data or len(table_data) < 1:
        return ""
    
    lines = []
    # 表头
    header = "| " + " | ".join(table_data[0]) + " |"
    lines.append(header)
    
    # 分隔线
    separator = "|" + "|".join([" --- " for _ in table_data[0]]) + "|"
    lines.append(separator)
    
    # 数据行
    for row in table_data[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)

def main():
    base_dir = '/mnt_onerec_model/jobs/onerec_gen_pos_v0_seq_encoding_256_s64_lazy_decoder_finish_expl/src/docs/experiments/实验结果'
    
    # 列出所有docx文件
    docx_files = [f for f in os.listdir(base_dir) if f.endswith('.docx')]
    docx_files.sort()
    
    output_path = '/mnt_onerec_model/jobs/onerec_gen_pos_v0_seq_encoding_256_s64_lazy_decoder_finish_expl/src/docs/experiments_summary_raw.md'
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("# One-Rec 实验详细内容\n\n")
        out.write("本文档包含所有实验的原始详细内容。\n\n")
        
        for i, filename in enumerate(docx_files, 1):
            filepath = os.path.join(base_dir, filename)
            data = read_docx_full(filepath)
            
            out.write(f"## {i}. {filename}\n\n")
            
            # 写入段落内容
            out.write("### 文本内容\n\n")
            for para in data['paragraphs']:
                out.write(f"{para}\n\n")
            
            # 写入表格内容
            if data['tables']:
                out.write("### 表格内容\n\n")
                for j, table in enumerate(data['tables'], 1):
                    out.write(f"**表格 {j}**:\n\n")
                    out.write(format_markdown_table(table))
                    out.write("\n\n")
            
            out.write("---\n\n")
            
            # 同时打印到控制台
            print(f"\n{'='*60}")
            print(f"{i}. {filename}")
            print('='*60)
            print("\n文本内容:")
            for para in data['paragraphs'][:15]:  # 只打印前15行
                print(f"  {para[:100]}")
            if data['tables']:
                print(f"\n表格数量: {len(data['tables'])}")
                for t in data['tables'][:2]:  # 只打印前2个表格
                    print(f"  表格行数: {len(t)}")
    
    print(f"\n\n完整内容已保存到: {output_path}")

if __name__ == '__main__':
    main()
