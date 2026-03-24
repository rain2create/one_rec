#!/usr/bin/env python3
"""
将 experiments.md 转换为格式良好的 docx 文件
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

def set_cell_border(cell, **kwargs):
    """设置单元格边框"""
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        if edge in kwargs:
            edge_element = tcPr.find(qn(f'w:{edge}'))
            if edge_element is None:
                edge_element = docx.oxml.OxmlElement(f'w:{edge}')
                tcPr.append(edge_element)
            edge_element.set(qn('w:val'), kwargs[edge])

def create_formatted_docx():
    doc = Document()
    
    # 设置中文字体
    doc.styles['Normal'].font.name = 'Microsoft YaHei'
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    
    # 标题
    title = doc.add_heading('One-Rec 实验总结报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 副标题
    subtitle = doc.add_paragraph('专项实验结果汇总')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].font.color.rgb = RGBColor(128, 128, 128)
    
    doc.add_paragraph()
    
    # 概述
    doc.add_heading('概述', 1)
    doc.add_paragraph(
        '本文档汇总了One-Rec项目中的8项专项实验结果，包括MoE专家策略、Scaling Law、'
        '码本优化、生成策略、半精度训练、消重优化、超参数调优等。所有实验均基于实际训练数据，'
        '提供了详细的对比表格和结论建议。'
    )
    
    # 实验概览表格
    doc.add_heading('实验概览', 1)
    
    overview_table = doc.add_table(rows=9, cols=4)
    overview_table.style = 'Light Grid Accent 1'
    
    # 表头
    headers = ['序号', '实验名称', '状态', '关键结论']
    for i, header in enumerate(headers):
        cell = overview_table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(11)
    
    # 数据行
    data = [
        ['1', 'MoE专家策略', '已完成', '24专家优于FFN'],
        ['2', 'Scaling Law', '已完成', '2048dim(5.3B)最佳'],
        ['3', '码本优化', '已完成', 'sort@3_ce重复率最低'],
        ['4', '生成策略', '已完成', 'BeamSearch指标最好'],
        ['5', '消重优化', '已完成', 'hash-based logit bias'],
        ['6', '半精度训练', '已完成', 'mixed精度加速4.5倍'],
        ['7', '超参数调优', '已完成', '多轮训练无收益'],
        ['8', '其他优化', '已完成', 'seq_enc正向明显'],
    ]
    
    for i, row_data in enumerate(data, 1):
        for j, text in enumerate(row_data):
            overview_table.rows[i].cells[j].text = text
    
    doc.add_paragraph()
    
    # 详细实验
    # 1. MoE专家策略
    doc.add_heading('1. MoE专家策略实验', 1)
    doc.add_paragraph('实验目的：探索不同专家数量对模型性能的影响')
    
    doc.add_heading('实验结果', 2)
    moe_table = doc.add_table(rows=6, cols=6)
    moe_table.style = 'Light List Accent 1'
    
    moe_headers = ['版本(dim 256)', 'like_hits', 'pos_hits', 'gen_urls', 'like_hitrate', 'pos_hitrate']
    for i, h in enumerate(moe_headers):
        cell = moe_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    moe_data = [
        ['256_base_len3(FFN)', '9,095', '23,679', '48,774', '5.52%', '9.03%'],
        ['moe_03_exp', '8,586', '24,371', '44,887', '5.22%', '9.30%'],
        ['moe_06_exp', '9,324', '24,741', '50,738', '5.66%', '9.44%'],
        ['moe_12_exp', '11,037', '27,994', '59,054', '6.72%', '10.69%'],
        ['moe_24_exp', '11,256', '28,326', '60,450', '6.82%', '10.81%'],
    ]
    
    for i, row in enumerate(moe_data, 1):
        for j, text in enumerate(row):
            moe_table.rows[i].cells[j].text = text
    
    p = doc.add_paragraph()
    p.add_run('结论：').bold = True
    p.add_run('专家数从3增加到24，性能持续提升。24专家配置下 like_hitrate 达到6.82%，pos_hitrate达到10.81%。相比FFN基线提升显著，推荐用于生产。')
    
    # 2. Scaling Law
    doc.add_heading('2. Scaling Law实验', 1)
    doc.add_paragraph('实验目的：研究模型维度与性能的关系')
    
    scaling_table = doc.add_table(rows=8, cols=6)
    scaling_table.style = 'Light List Accent 1'
    
    scaling_headers = ['dim_size', '参数量', 'like_hits', 'pos_hits', 'like_hitrate', 'pos_hitrate']
    for i, h in enumerate(scaling_headers):
        cell = scaling_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    scaling_data = [
        ['64', '-', '4,219', '18,203', '2.56%', '6.31%'],
        ['128', '-', '4,929', '23,793', '2.98%', '8.25%'],
        ['256', '0.2B', '5,438', '28,856', '3.29%', '10.01%'],
        ['512', '0.5B', '5,907', '32,228', '3.58%', '11.18%'],
        ['1024', '1.4B', '6,091', '34,045', '3.69%', '11.81%'],
        ['2048', '5.3B', '6,202', '34,688', '3.76%', '12.03%'],
    ]
    
    for i, row in enumerate(scaling_data, 1):
        for j, text in enumerate(row):
            scaling_table.rows[i].cells[j].text = text
    
    p = doc.add_paragraph()
    p.add_run('结论：').bold = True
    p.add_run('模型维度从64增加到2048，性能持续提升。2048 dim (5.3B参数)达到最佳性能，但需权衡计算成本。')
    
    # 3. 码本优化
    doc.add_heading('3. 码本优化实验', 1)
    doc.add_paragraph('实验目的：优化三层码本的编码策略')
    
    code_table = doc.add_table(rows=4, cols=7)
    code_table.style = 'Light List Accent 1'
    
    code_headers = ['方案', '重复率', 'like_hits', 'que_hits', '生成url数', 'like_hitrate', 'pos_hitrate']
    for i, h in enumerate(code_headers):
        cell = code_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    code_data = [
        ['BASE', '34.97%', '11,080', '26,967', '59,229', '6.72%', '10.28%'],
        ['hash@3_ce', '3.46%', '15,515', '29,661', '19,266', '9.20%', '11.35%'],
        ['sort@3_ce', '0.31%', '15,459', '27,221', '36,928', '9.07%', '10.41%'],
    ]
    
    for i, row in enumerate(code_data, 1):
        for j, text in enumerate(row):
            cell = code_table.rows[i].cells[j]
            cell.text = text
            # 高亮最优值
            if i == 3 and j in [1]:  # sort@3_ce的重复率
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0, 128, 0)
    
    p = doc.add_paragraph()
    p.add_run('结论：').bold = True
    p.add_run('sort@3_ce 重复率最低(仅0.31%)，大幅优于BASE(34.97%)。总体有优势，推荐用于生产。')
    
    # 4. 生成策略
    doc.add_heading('4. 生成策略实验', 1)
    doc.add_paragraph('实验目的：对比不同生成策略的效果')
    
    gen_table = doc.add_table(rows=4, cols=6)
    gen_table.style = 'Light List Accent 1'
    
    gen_headers = ['生成策略', 'like_hits', 'que_hits', 'recall_urls', '链接命中率', '特点']
    for i, h in enumerate(gen_headers):
        cell = gen_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    gen_data = [
        ['BeamSearch', '5,450', '22,287', '38,159', '9.28%', '指标最好'],
        ['topk(30)', '4,004', '13,601', '52,088', '4.72%', '多样性最好'],
        ['topp(0.5)', '3,494', '11,962', '50,428', '4.15%', '平衡方案'],
    ]
    
    for i, row in enumerate(gen_data, 1):
        for j, text in enumerate(row):
            cell = gen_table.rows[i].cells[j]
            cell.text = text
            if j == 4 and i == 1:  # BeamSearch的命中率
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0, 128, 0)
    
    p = doc.add_paragraph()
    p.add_run('结论：').bold = True
    p.add_run('BeamSearch 牺牲多样性但大幅提升离线指标。要指标用BeamSearch，要多样性用TopK=30。')
    
    # 5. 半精度训练
    doc.add_heading('5. 半精度训练实验', 1)
    doc.add_paragraph('实验目的：测试半精度(FP16/BF16)的训练效果和耗时')
    
    time_table = doc.add_table(rows=9, cols=2)
    time_table.style = 'Light List Accent 1'
    
    time_headers = ['dim', '训练时间']
    for i, h in enumerate(time_headers):
        cell = time_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    time_data = [
        ['64', '0.18 s/batch'],
        ['128', '0.18 s/batch'],
        ['256', '0.24 s/batch'],
        ['512', '0.51 s/batch'],
        ['1024', '1.2 s/batch'],
        ['2048', '4.81 s/batch'],
        ['2048_mixed', '1.08 s/batch ⚡'],
        ['256_mixed', '0.15 s/batch ⚡'],
    ]
    
    for i, row in enumerate(time_data, 1):
        for j, text in enumerate(row):
            cell = time_table.rows[i].cells[j]
            cell.text = text
            if 'mixed' in text:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0, 128, 0)
    
    p = doc.add_paragraph()
    p.add_run('结论：').bold = True
    p.add_run('半精度训练(mixed)可大幅降低训练时间。2048_mixed (1.08s)比2048 (4.81s)快4.5倍。推荐混合精度训练。')
    
    # 6. 超参数调优
    doc.add_heading('6. 超参数调优实验', 1)
    doc.add_paragraph('实验目的：寻找最优超参数配置')
    
    epoch_table = doc.add_table(rows=3, cols=6)
    epoch_table.style = 'Light List Accent 1'
    
    epoch_headers = ['epoch', 'like_hits', 'pos_hits', 'gen_urls', 'like_hitrate', 'pos_hitrate']
    for i, h in enumerate(epoch_headers):
        cell = epoch_table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    epoch_data = [
        ['1', '11,256', '28,326', '60,450', '6.82%', '10.81%'],
        ['3', '10,925', '26,814', '77,023', '6.62%', '10.23%'],
    ]
    
    for i, row in enumerate(epoch_data, 1):
        for j, text in enumerate(row):
            epoch_table.rows[i].cells[j].text = text
    
    p = doc.add_paragraph()
    p.add_run('关键发现：').bold = True
    p.add_run('\n1. 多轮训练不会更优 - epoch 3性能反而下降\n2. 学习率、batch_size等有影响，实验对比需要控制变量\n')
    
    # 最终推荐配置
    doc.add_page_break()
    doc.add_heading('推荐配置', 1)
    
    rec_table = doc.add_table(rows=8, cols=2)
    rec_table.style = 'Medium Grid 1 Accent 1'
    
    rec_data = [
        ['配置项', '推荐值'],
        ['MoE专家数', '24'],
        ['模型维度', '256（生产推荐）/ 512-2048（追求性能）'],
        ['码本编码策略', 'sort@3_ce'],
        ['生成策略', 'BeamSearch（要指标）/ TopK=30（要多样性）'],
        ['精度', 'mixed（混合精度）'],
        ['训练轮数', '1 epoch'],
        ['batch_size', '1024'],
    ]
    
    for i, row in enumerate(rec_data):
        for j, text in enumerate(row):
            cell = rec_table.rows[i].cells[j]
            cell.text = text
            if i == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.size = Pt(12)
    
    # 保存
    output_path = '/mnt_onerec_model/jobs/onerec_gen_pos_v0_seq_encoding_256_s64_lazy_decoder_finish_expl/src/docs/One-Rec_实验总结报告.docx'
    doc.save(output_path)
    print(f"已生成: {output_path}")

if __name__ == '__main__':
    create_formatted_docx()
