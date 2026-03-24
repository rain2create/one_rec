#!/usr/bin/env python3
"""
绘制 One-Rec 训练 Loss 曲线 - 分阶段显示，RL显示为正Reward
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 读取数据
data = []
with open('/tmp/loss_data_positive.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                step = int(parts[0])
                ct_loss = float(parts[1])
                rl_reward = float(parts[2])  # 已经是正值
                if step > 0 and ct_loss > 0 and rl_reward > 0:
                    data.append((step, ct_loss, rl_reward))
            except:
                continue

# 转换为numpy数组
steps = np.array([d[0] for d in data])
ct_losses = np.array([d[1] for d in data])
rl_rewards = np.array([d[2] for d in data])

# 按step排序
sort_idx = np.argsort(steps)
steps = steps[sort_idx]
ct_losses = ct_losses[sort_idx]
rl_rewards = rl_rewards[sort_idx]

# 定义两个阶段
# 最初30天：约100-66000步
first_month_mask = steps <= 66000
steps_month1 = steps[first_month_mask]
ct_month1 = ct_losses[first_month_mask]
rl_month1 = rl_rewards[first_month_mask]

# 全部数据
steps_all = steps
ct_all = ct_losses
rl_all = rl_rewards

# 平滑函数
def smooth(y, box_pts=20):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# 颜色定义
color_ct = '#2E86AB'      # 蓝色 - NTP Loss
color_rl = '#A23B72'      # 紫色 - RL Reward

# ==================== 图1: 最初30天 ====================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# 使用 fig.text 添加标题，避免重叠
fig1.text(0.5, 0.98, 'One-Rec Training: First 30 Days (Step 100-66,000)', 
          ha='center', fontsize=14, fontweight='bold')

# NTP Loss - 最初30天
ax1 = axes1[0]
ax1.plot(steps_month1/1000, ct_month1, alpha=0.3, color=color_ct)
ax1.plot(steps_month1/1000, smooth(ct_month1), color=color_ct, linewidth=2.5, label='NTP Loss')
ax1.fill_between(steps_month1/1000, ct_month1, alpha=0.2, color=color_ct)

ax1.axhline(y=ct_month1[-1], color=color_ct, linestyle='--', alpha=0.7)
ax1.text(steps_month1[-1]/1000, ct_month1[-1], f'  Final: {ct_month1[-1]:.2f}', 
         fontsize=10, va='center', color=color_ct, fontweight='bold')

ax1.set_xlabel('Training Step (k)', fontsize=12)
ax1.set_ylabel('NTP Loss (CrossEntropy)', fontsize=12)
ax1.set_title('NTP Loss Trend', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# RL Reward - 最初30天
ax2 = axes1[1]
ax2.plot(steps_month1/1000, rl_month1, alpha=0.3, color=color_rl)
ax2.plot(steps_month1/1000, smooth(rl_month1), color=color_rl, linewidth=2.5, label='RL Reward')
ax2.fill_between(steps_month1/1000, rl_month1, alpha=0.2, color=color_rl)

ax2.axhline(y=rl_month1[-1], color=color_rl, linestyle='--', alpha=0.7)
ax2.text(steps_month1[-1]/1000, rl_month1[-1], f'  Final: {rl_month1[-1]:.2f}', 
         fontsize=10, va='center', color=color_rl, fontweight='bold')

ax2.set_xlabel('Training Step (k)', fontsize=12)
ax2.set_ylabel('RL Reward (higher is better)', fontsize=12)
ax2.set_title('RL Reward Trend', fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出顶部空间给fig标题
plt.savefig('training_first_30days.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('training_first_30days.pdf', bbox_inches='tight', facecolor='white')
print("Saved: training_first_30days.png/pdf")

# ==================== 图2: 全部数据 ====================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 使用 fig.text 添加标题，避免重叠
fig2.text(0.5, 0.98, f'One-Rec Training: All Data (Step {steps_all[0]}-{steps_all[-1]})', 
          ha='center', fontsize=14, fontweight='bold')

# NTP Loss - 全部
ax3 = axes2[0]
ax3.plot(steps_all/1000, ct_all, alpha=0.3, color=color_ct)
ax3.plot(steps_all/1000, smooth(ct_all), color=color_ct, linewidth=2.5, label='NTP Loss')
ax3.fill_between(steps_all/1000, ct_all, alpha=0.2, color=color_ct)

ax3.axhline(y=ct_all[-1], color=color_ct, linestyle='--', alpha=0.7)
ax3.text(steps_all[-1]/1000, ct_all[-1], f'  Final: {ct_all[-1]:.2f}', 
         fontsize=10, va='center', color=color_ct, fontweight='bold')

# 标注关键点
min_ct_idx = np.argmin(ct_all)
ax3.axhline(y=ct_all[min_ct_idx], color=color_ct, linestyle=':', alpha=0.5)
ax3.text(steps_all[min_ct_idx]/1000, ct_all[min_ct_idx], f'  Min: {ct_all[min_ct_idx]:.2f}', 
         fontsize=9, va='bottom', color=color_ct)

ax3.set_xlabel('Training Step (k)', fontsize=12)
ax3.set_ylabel('NTP Loss (CrossEntropy)', fontsize=12)
ax3.set_title('NTP Loss - Full Training', fontsize=13, fontweight='bold', pad=10)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# RL Reward - 全部
ax4 = axes2[1]
ax4.plot(steps_all/1000, rl_all, alpha=0.3, color=color_rl)
ax4.plot(steps_all/1000, smooth(rl_all), color=color_rl, linewidth=2.5, label='RL Reward')
ax4.fill_between(steps_all/1000, rl_all, alpha=0.2, color=color_rl)

ax4.axhline(y=rl_all[-1], color=color_rl, linestyle='--', alpha=0.7)
ax4.text(steps_all[-1]/1000, rl_all[-1], f'  Final: {rl_all[-1]:.2f}', 
         fontsize=10, va='center', color=color_rl, fontweight='bold')

# 标注最高点
max_rl_idx = np.argmax(rl_all)
ax4.axhline(y=rl_all[max_rl_idx], color=color_rl, linestyle=':', alpha=0.5)
ax4.text(steps_all[max_rl_idx]/1000, rl_all[max_rl_idx], f'  Max: {rl_all[max_rl_idx]:.2f}', 
         fontsize=9, va='bottom', color=color_rl)

ax4.set_xlabel('Training Step (k)', fontsize=12)
ax4.set_ylabel('RL Reward (higher is better)', fontsize=12)
ax4.set_title('RL Reward - Full Training', fontsize=13, fontweight='bold', pad=10)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出顶部空间给fig标题
plt.savefig('training_all_data.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('training_all_data.pdf', bbox_inches='tight', facecolor='white')
print("Saved: training_all_data.png/pdf")

# 输出统计
print("\n" + "="*60)
print("Statistics:")
print("="*60)
print(f"\nFirst 30 Days (Step {steps_month1[0]}-{steps_month1[-1]}):")
print(f"  NTP Loss: {ct_month1[0]:.4f} -> {ct_month1[-1]:.4f} (↓{ct_month1[0]-ct_month1[-1]:.4f})")
print(f"  RL Reward: {rl_month1[0]:.4f} -> {rl_month1[-1]:.4f} (↑{rl_month1[-1]-rl_month1[0]:.4f})")

print(f"\nAll Data (Step {steps_all[0]}-{steps_all[-1]}):")
print(f"  NTP Loss: {ct_all[0]:.4f} -> {ct_all[-1]:.4f} (↓{ct_all[0]-ct_all[-1]:.4f})")
print(f"  RL Reward: {rl_all[0]:.4f} -> {rl_all[-1]:.4f} (↑{rl_all[-1]-rl_all[0]:.4f})")
print(f"  RL Max: {rl_all[max_rl_idx]:.4f} at step {steps_all[max_rl_idx]}")
