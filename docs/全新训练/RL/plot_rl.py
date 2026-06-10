#!/usr/bin/env python3
"""
v12_new RL 阶段训练曲线 - 同 PT/SFT 风格
数据源：log_grpo_neg_sample（2025-09-19 14:31→17:00，172 个采样点，每 10 batch 一行）
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

CSV = '/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/RL/rl_metrics.csv'
OUT = '/mnt_liuhuan/search_public/xby/one_rec/docs/全新训练/RL'

# columns: time_s, batch, real_bs, uniq_ratio, grpo, gbpo, pt, legal,
#          pos_num, neg_num, ar10, ar3, hitrate, empty, unique, pos_prob, neg_prob,
#          like_auc, like_after_auc, reward_like_auc
data = np.loadtxt(CSV, delimiter=',', skiprows=1)
time_s          = data[:, 0]
batch           = data[:, 1].astype(int)
real_bs         = data[:, 2]
uniq_ratio      = data[:, 3]
grpo            = data[:, 4]
gbpo            = data[:, 5]
pt_loss         = data[:, 6]
legal           = data[:, 7]
pos_num         = data[:, 8]
neg_num         = data[:, 9]
ar10            = data[:, 10]
ar3             = data[:, 11]
hitrate         = data[:, 12]
empty           = data[:, 13]
unique          = data[:, 14]
pos_prob        = data[:, 15]
neg_prob        = data[:, 16]
like_auc        = data[:, 17]
like_after_auc  = data[:, 18]
reward_like_auc = data[:, 19]

def smooth(y, box_pts=10):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

# Colors
C_AR10        = '#E63946'
C_AR3         = '#F4A261'
C_HITRATE     = '#264653'
C_LIKE        = '#2E86AB'
C_LIKE_AFTER  = '#E63946'
C_REWARD_LIKE = '#3B7A57'
C_GRPO        = '#2E86AB'
C_GBPO        = '#A23B72'
C_LEGAL       = '#3B7A57'
C_PT          = '#F18F01'
C_EMPTY       = '#A23B72'
C_UNIQ        = '#F18F01'
C_UNIQUE      = '#264653'

# ==================== 图1: Reward & Hitrate ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.text(0.5, 0.98,
         f'v12_new RL Stage - Reward & Hitrate (Batch {batch[0]}-{batch[-1]}, ~2.5 hours, 1 GPU)',
         ha='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0.945,
         'All values are cumulative running averages from print log (total / batch_count)',
         ha='center', fontsize=10, style='italic', color='#555555')

ax = axes[0]
ax.plot(batch, ar10, alpha=0.3, color=C_AR10)
ax.plot(batch, smooth(ar10), color=C_AR10, linewidth=2.5, label=f'AvgReward@10  {ar10[0]:.4f} → {ar10[-1]:.4f}')
ax.plot(batch, ar3, alpha=0.3, color=C_AR3)
ax.plot(batch, smooth(ar3), color=C_AR3, linewidth=2.5, label=f'AvgReward@3   {ar3[0]:.4f} → {ar3[-1]:.4f}')
ax.axhline(y=ar10[-1], color=C_AR10, linestyle='--', alpha=0.5)
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('Avg Reward', fontsize=12)
ax.set_title(f'AvgReward@10:  ↑ {(ar10[-1]/ar10[0]-1)*100:+.1f}%  (grpo_loss × reward signal working)', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(batch, hitrate, alpha=0.3, color=C_HITRATE)
ax.plot(batch, smooth(hitrate), color=C_HITRATE, linewidth=2.5,
        label=f'Hitrate@100  {hitrate[0]:.4f} → {hitrate[-1]:.4f}')
ax.axhline(y=hitrate[-1], color=C_HITRATE, linestyle='--', alpha=0.5)
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('Hitrate@100', fontsize=12)
ax.set_title(f'Hitrate@100:  ↓ {(1-hitrate[-1]/hitrate[0])*100:+.1f}%  — prior_loss (w=2.0) anti-popularity by design',
             fontsize=11, fontweight='bold', pad=10, color='#A23B72')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{OUT}/rl_reward.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/rl_reward.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: rl_reward.png/pdf")

# ==================== 图2: AUC ====================
fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
fig.text(0.5, 0.97, 'v12_new RL Stage - AUC Trajectories',
         ha='center', fontsize=14, fontweight='bold')

ax.plot(batch, like_auc, alpha=0.3, color=C_LIKE)
ax.plot(batch, smooth(like_auc), color=C_LIKE, linewidth=2.5,
        label=f'LikeAUC (policy before update):  {like_auc[0]:.3f} → {like_auc[-1]:.3f}')

ax.plot(batch, like_after_auc, alpha=0.3, color=C_LIKE_AFTER)
ax.plot(batch, smooth(like_after_auc), color=C_LIKE_AFTER, linewidth=2.5,
        label=f'LikeAfterAUC (policy AFTER update):  {like_after_auc[0]:.3f} → {like_after_auc[-1]:.3f}  ★ core signal')

ax.plot(batch, reward_like_auc, alpha=0.3, color=C_REWARD_LIKE)
ax.plot(batch, smooth(reward_like_auc), color=C_REWARD_LIKE, linewidth=2.5,
        label=f'RewardLikeAUC (reward model on corpus):  {reward_like_auc[0]:.3f} → {reward_like_auc[-1]:.3f}')

ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(
    f'LikeAfterAUC  Δ {like_after_auc[-1]-like_after_auc[0]:+.3f}  — RL genuinely improves policy AUC on positive labels',
    fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.45, 1.0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT}/rl_auc.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/rl_auc.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: rl_auc.png/pdf")

# ==================== 图3: Loss 组合 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.text(0.5, 0.98, 'v12_new RL Stage - Multi-Loss Components (Cumulative Running Avg)',
         ha='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0.945,
         'Total loss = 0.2·gbpo + 1.0·grpo + 1.5·legal + 2.0·prior   |   prior_loss (weight 2.0, largest) NOT in print log',
         ha='center', fontsize=10, style='italic', color='#A23B72')

# Left: grpo / gbpo / legal (量级接近)
ax = axes[0]
ax.plot(batch, smooth(grpo), color=C_GRPO, linewidth=2,
        label=f'grpo_Loss (×1.0)   {grpo[0]:.4f} → {grpo[-1]:.4f}  ≈0 by design (group-norm advantage)')
ax.plot(batch, smooth(gbpo), color=C_GBPO, linewidth=2,
        label=f'gbpo_Loss (×0.2)   {gbpo[0]:.4f} → {gbpo[-1]:.4f}')
ax.plot(batch, smooth(legal), color=C_LEGAL, linewidth=2,
        label=f'legal_Loss (×1.5)  {legal[0]:.4f} → {legal[-1]:.4f}  saturated at −1')
ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Printed loss components (3 of 4)', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='center right', fontsize=9)
ax.grid(True, alpha=0.3)

# Right: pt_loss (NTP CE, 量级大)
ax = axes[1]
ax.plot(batch, pt_loss, alpha=0.3, color=C_PT)
ax.plot(batch, smooth(pt_loss), color=C_PT, linewidth=2.5,
        label=f'pt_Loss (NTP CE)   {pt_loss[0]:.2f} → {pt_loss[-1]:.2f}')
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('NTP CE Loss', fontsize=12)
ax.set_title('pt_Loss — MONITORING ONLY, not in total loss\n(rise reflects strategy drift from SFT distribution)',
             fontsize=11, fontweight='bold', pad=10, color='#555555')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{OUT}/rl_loss.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/rl_loss.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: rl_loss.png/pdf")

# ==================== 图4: 健康度 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.text(0.5, 0.98, 'v12_new RL Stage - Health Metrics',
         ha='center', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(batch, empty, alpha=0.3, color=C_EMPTY)
ax.plot(batch, smooth(empty), color=C_EMPTY, linewidth=2.5,
        label=f'EmptyDecodingRatio (mean {empty.mean():.4f})')
ax.plot(batch, uniq_ratio, alpha=0.3, color=C_UNIQ)
ax.plot(batch, smooth(uniq_ratio), color=C_UNIQ, linewidth=2.5,
        label=f'Uniq_ratio (mean {uniq_ratio.mean():.3f})')
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('Ratio', fontsize=12)
ax.set_title('EmptyDecoding stays low + stable → no decode crash',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(batch, unique, alpha=0.3, color=C_UNIQUE)
ax.plot(batch, smooth(unique), color=C_UNIQUE, linewidth=2.5,
        label=f'Unique%  (mean {unique.mean():.2f})')
ax.set_xlabel('Batch Count', fontsize=12)
ax.set_ylabel('Unique', fontsize=12)
ax.set_title('Beam-search outputs stay nearly all-unique',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(95, 101)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUT}/rl_health.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/rl_health.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: rl_health.png/pdf")

# ==================== 统计 ====================
print("\n" + "=" * 60)
print("RL Stage Statistics:")
print("=" * 60)
print(f"Batch range:    {batch[0]} → {batch[-1]}")
print(f"Total points:   {len(batch)}")
print(f"AvgReward@10:   {ar10[0]:.4f} → {ar10[-1]:.4f}  ({(ar10[-1]/ar10[0]-1)*100:+.1f}%)")
print(f"AvgReward@3:    {ar3[0]:.4f} → {ar3[-1]:.4f}  ({(ar3[-1]/ar3[0]-1)*100:+.1f}%)")
print(f"Hitrate@100:    {hitrate[0]:.4f} → {hitrate[-1]:.4f}  ({(hitrate[-1]/hitrate[0]-1)*100:+.1f}%)")
print(f"LikeAUC:        {like_auc[0]:.4f} → {like_auc[-1]:.4f}  (Δ {like_auc[-1]-like_auc[0]:+.4f})")
print(f"LikeAfterAUC:   {like_after_auc[0]:.4f} → {like_after_auc[-1]:.4f}  (Δ {like_after_auc[-1]-like_after_auc[0]:+.4f}) ★")
print(f"RewardLikeAUC:  {reward_like_auc[0]:.4f} → {reward_like_auc[-1]:.4f}  (Δ {reward_like_auc[-1]-reward_like_auc[0]:+.4f})")
print(f"grpo_Loss:      {grpo[0]:.4f} → {grpo[-1]:.4f}")
print(f"gbpo_Loss:      {gbpo[0]:.4f} → {gbpo[-1]:.4f}  (Δ {gbpo[-1]-gbpo[0]:+.4f})")
print(f"legal_Loss:     {legal[0]:.4f} → {legal[-1]:.4f}")
print(f"pt_Loss:        {pt_loss[0]:.4f} → {pt_loss[-1]:.4f}  (Δ {pt_loss[-1]-pt_loss[0]:+.4f})")
print(f"EmptyDecoding:  mean={empty.mean():.4f}, max={empty.max():.4f}")
print(f"Unique:         mean={unique.mean():.2f}, min={unique.min():.2f}")
print(f"Mean batch_duration: {time_s.mean():.1f}s  (per 10 batches)")
print(f"Effective: ~{time_s.mean()/10:.2f}s per batch")
