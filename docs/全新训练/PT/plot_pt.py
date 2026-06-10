#!/usr/bin/env python3
"""
v12_new PT 阶段训练曲线 - 参考 docs/plot_training_curves.py 风格
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

data = []
with open('/tmp/pt_data.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            step = int(parts[0])
            loss = float(parts[1])
            tp = float(parts[2])
            mvr = float(parts[3])
            fvr = float(parts[4])
            data.append((step, loss, tp, mvr, fvr))

steps = np.array([d[0] for d in data])
losses = np.array([d[1] for d in data])
tprobs = np.array([d[2] for d in data])
mvrs = np.array([d[3] for d in data])
fvrs = np.array([d[4] for d in data])

idx = np.argsort(steps)
steps, losses, tprobs, mvrs, fvrs = steps[idx], losses[idx], tprobs[idx], mvrs[idx], fvrs[idx]

def smooth(y, box_pts=20):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

color_loss = '#2E86AB'   # 蓝
color_tp   = '#A23B72'   # 紫
color_fvr  = '#F18F01'   # 橙
color_mvr  = '#3B7A57'   # 绿

# ==================== 图1: Loss + Target Prob 主图 ====================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.text(0.5, 0.98,
          f'v12_new PT Stage Training (Step {steps[0]}-{steps[-1]}, ~10.5 hours, 30-day corpus)',
          ha='center', fontsize=14, fontweight='bold')

# Actual Loss
ax1 = axes1[0]
ax1.plot(steps/1000, losses, alpha=0.3, color=color_loss)
ax1.plot(steps/1000, smooth(losses), color=color_loss, linewidth=2.5, label='Actual Loss (NTP CE)')
ax1.fill_between(steps/1000, losses, alpha=0.2, color=color_loss)

ax1.axhline(y=losses[-1], color=color_loss, linestyle='--', alpha=0.7)
ax1.text(steps[-1]/1000, losses[-1], f'  Final: {losses[-1]:.2f}',
         fontsize=10, va='center', color=color_loss, fontweight='bold')

min_idx = np.argmin(losses)
ax1.axhline(y=losses[min_idx], color=color_loss, linestyle=':', alpha=0.5)
ax1.text(steps[min_idx]/1000, losses[min_idx], f'  Min: {losses[min_idx]:.2f}',
         fontsize=9, va='bottom', color=color_loss)

ax1.set_xlabel('Training Step (k)', fontsize=12)
ax1.set_ylabel('Actual Loss (CrossEntropy)', fontsize=12)
ax1.set_title(f'NTP Loss: {losses[0]:.2f} → {losses[-1]:.2f}  (↓{losses[0]-losses[-1]:.2f})',
              fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Target Prob
ax2 = axes1[1]
ax2.plot(steps/1000, tprobs * 1000, alpha=0.3, color=color_tp)
ax2.plot(steps/1000, smooth(tprobs * 1000), color=color_tp, linewidth=2.5, label='Target Prob (×1e-3)')
ax2.fill_between(steps/1000, tprobs * 1000, alpha=0.2, color=color_tp)

ax2.axhline(y=tprobs[-1]*1000, color=color_tp, linestyle='--', alpha=0.7)
ax2.text(steps[-1]/1000, tprobs[-1]*1000, f'  Final: {tprobs[-1]*1000:.3f}×1e-3',
         fontsize=10, va='center', color=color_tp, fontweight='bold')

max_idx = np.argmax(tprobs)
ax2.text(steps[max_idx]/1000, tprobs[max_idx]*1000, f'  Max: {tprobs[max_idx]*1000:.3f}×1e-3',
         fontsize=9, va='bottom', color=color_tp)

ax2.set_xlabel('Training Step (k)', fontsize=12)
ax2.set_ylabel('Target Prob (×1e-3, higher is better)', fontsize=12)
ax2.set_title(f'Target Prob: 0 → {tprobs[-1]*1000:.3f}×1e-3  (rising)',
              fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/pt_loss_tprob.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/pt_loss_tprob.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: pt_loss_tprob.png/pdf")

# ==================== 图2: Valid Rates 辅助图 ====================
fig2, ax3 = plt.subplots(1, 1, figsize=(10, 5))
fig2.text(0.5, 0.97,
          'v12_new PT Stage - URL Tokenizer Hit Rate Across Training',
          ha='center', fontsize=14, fontweight='bold')

ax3.plot(steps/1000, mvrs, alpha=0.25, color=color_mvr)
ax3.plot(steps/1000, smooth(mvrs), color=color_mvr, linewidth=2.5,
         label=f'Mask Valid Rate (any-layer hit, mean ≈ {mvrs.mean():.4f})')

ax3.plot(steps/1000, fvrs, alpha=0.25, color=color_fvr)
ax3.plot(steps/1000, smooth(fvrs), color=color_fvr, linewidth=2.5,
         label=f'Final Valid Rate (all-3-layer hit, mean ≈ {fvrs.mean():.4f})')

ax3.fill_between(steps/1000, mvrs, fvrs, alpha=0.15, color=color_fvr)

ax3.set_xlabel('Training Step (k)', fontsize=12)
ax3.set_ylabel('Hit Rate', fontsize=12)
ax3.set_title(
    f'Mask Valid (≥1 layer hit) vs Final Valid (all 3 layers hit)\n'
    f'Both stable around training — not a learning signal, only dict coverage',
    fontsize=12, fontweight='bold', pad=10)
ax3.legend(loc='center right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.3, 0.7)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/pt_valid_rate.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/pt_valid_rate.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: pt_valid_rate.png/pdf")

# ==================== 统计 ====================
print("\n" + "="*60)
print("PT Stage Statistics:")
print("="*60)
print(f"Step range:     {steps[0]} → {steps[-1]}")
print(f"Total points:   {len(steps)}")
print(f"Actual Loss:    {losses[0]:.4f} → {losses[-1]:.4f}  (↓{losses[0]-losses[-1]:.4f}, {(1-losses[-1]/losses[0])*100:.1f}%)")
print(f"  Min loss:     {losses[min_idx]:.4f} @ step {steps[min_idx]}")
print(f"Target Prob:    {tprobs[0]:.9f} → {tprobs[-1]:.9f}")
print(f"  Max tprob:    {tprobs[max_idx]:.9f} @ step {steps[max_idx]}")
print(f"Mask Valid:     {mvrs.mean():.4f} (mean), std={mvrs.std():.4f}")
print(f"Final Valid:    {fvrs.mean():.4f} (mean), std={fvrs.std():.4f}")
