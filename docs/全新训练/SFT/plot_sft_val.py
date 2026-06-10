#!/usr/bin/env python3
"""
SFT 阶段 val 曲线
数据源：log_2025091618 的 3 个 val 点（step 280k/290k/300k）
val_steps=10000, val_batches=100, G=10
val 集硬编码: PtDataset('/mnt/video.rank.s101', day=0, with_today=True, batch_size=10000)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = '/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/SFT'

# log_2025091618 提取的 3 个 val 点
steps    = np.array([280_000, 290_000, 300_000])
val_loss = np.array([4.1680, 3.9758, 3.8772])
val_hr10 = np.array([0.0335, 0.0378, 0.0400])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
fig.text(0.5, 0.98,
         'SFT Stage - Validation Metrics (log_2025091618, every 10k step, G=10)',
         ha='center', fontsize=13, fontweight='bold')
fig.text(0.5, 0.94,
         "Val set: PtDataset('/mnt/video.rank.s101', corpus_day=0, with_today=True) - today's exposure, 100 batches × 10k = 1M samples / eval",
         ha='center', fontsize=9, style='italic', color='#555555')

C_LOSS = '#E63946'
C_HR   = '#264653'

# Left: val_loss
ax = axes[0]
ax.plot(steps, val_loss, color=C_LOSS, marker='o', markersize=10, linewidth=2.5,
        label=f'val_loss (NTP CE)')
for x, y in zip(steps, val_loss):
    ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Global Step', fontsize=12)
ax.set_ylabel('Val NTP CE Loss', fontsize=12)
delta_loss_pct = (val_loss[-1] / val_loss[0] - 1) * 100
ax.set_title(f'val_loss: {val_loss[0]:.4f} → {val_loss[-1]:.4f}  ({delta_loss_pct:+.1f}%)',
             fontsize=12, fontweight='bold', pad=10, color=C_LOSS)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(3.7, 4.3)

# Right: val_hitrate@10
ax = axes[1]
ax.plot(steps, val_hr10, color=C_HR, marker='s', markersize=10, linewidth=2.5,
        label='val_hitrate@10 (beam-search Top-10 命中真实 URL 的样本率)')
for x, y in zip(steps, val_hr10):
    ax.annotate(f'{y*100:.2f}%', xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Global Step', fontsize=12)
ax.set_ylabel('Hitrate@10', fontsize=12)
delta_hr_pct = (val_hr10[-1] / val_hr10[0] - 1) * 100
ax.set_title(f'val_hitrate@10: {val_hr10[0]*100:.2f}% → {val_hr10[-1]*100:.2f}%  ({delta_hr_pct:+.1f}%)',
             fontsize=12, fontweight='bold', pad=10, color=C_HR)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.030, 0.045)

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(f'{OUT}/sft_val.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/sft_val.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}/sft_val.png / .pdf")

print("\n" + "=" * 60)
print("SFT Validation Summary:")
print("=" * 60)
print(f"Number of val points:  {len(steps)}")
print(f"Step range:            {steps[0]:,} → {steps[-1]:,}")
print(f"val_loss:              {val_loss[0]:.4f} → {val_loss[-1]:.4f}  ({delta_loss_pct:+.2f}%)")
print(f"val_hitrate@10:        {val_hr10[0]*100:.2f}% → {val_hr10[-1]*100:.2f}%  ({delta_hr_pct:+.2f}%)")
