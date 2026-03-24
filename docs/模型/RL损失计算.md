# One-Rec RL Loss 数据流详解

## 一、整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RL Loss 数据流 (decoder.py)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│  ├── output_logits: [batch_size, 3, 1502]  (3层码本，每层1502个token)      │
│  ├── labels: [batch_size, 3]  (目标token的3层码本ID)                        │
│  └── advantages: [batch_size]  (每个样本的advantage值)                      │
│                                                                             │
│  Step 1: 计算序列联合概率 (每个样本一个概率值)                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  probs_ = F.log_softmax(output_logits, -1)                            │ │
│  │  → shape: [B, 3, 1502]  (每层每个token的log概率)                      │ │
│  │                                                                       │ │
│  │  probs_ = gather(probs_, labels_)                                     │ │
│  │  → shape: [B, 3]  (只取出目标3个token的log概率)                       │ │
│  │                                                                       │ │
│  │  probs_ = sum(dim=-1)  ← 关键！3个token的log概率相加                 │ │
│  │  → shape: [B]  (序列的log概率 = log(p1) + log(p2) + log(p3))         │ │
│  │                                                                       │ │
│  │  probs_ = exp(probs_)                                                 │ │
│  │  → shape: [B]  (序列联合概率 = p1 × p2 × p3)                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Step 2: 计算 RL Loss                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  pi_new = probs_  → shape: [B]                                        │ │
│  │  advantages → shape: [B]                                              │ │
│  │                                                                       │ │
│  │  # Policy Gradient 核心                                               │ │
│  │  loss_per_sample = -(pi_new * advantages)  → shape: [B]               │ │
│  │                                                                       │ │
│  │  # 只保留有交互的样本 (adv != 0)                                       │ │
│  │  mask = (advantages != 0)  → shape: [B]                               │ │
│  │                                                                       │ │
│  │  # 样本级平均                                                          │ │
│  │  rl_loss = (loss_per_sample * mask).sum() / mask.sum()                │ │
│  │  → scalar (单个数值，如 -2.13)                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Shape 变化表

| 步骤 | 代码 | Shape | 说明 |
|------|------|-------|------|
| 输入 | `output_logits` | `[B, 3, 1502]` | 3层码本的logits |
| 输入 | `labels` | `[B, 3]` | 目标3层token ID |
| 输入 | `features['advantages']` | `[B]` | 每个样本的advantage |
| log_softmax | `F.log_softmax(...)` | `[B, 3, 1502]` | 每层每个token的log概率 |
| gather | `gather(..., labels_)` | `[B, 3, 1]` | 只取目标token的log概率 |
| squeeze | `squeeze(-1)` | `[B, 3]` | 去掉最后一个维度 |
| **sum** | `sum(dim=-1)` | `[B]` | **关键：3层log概率相加** |
| exp | `torch.exp(...)` | `[B]` | 得到序列联合概率 |
| 输入RL | `pi_new` | `[B]` | 序列概率 |
| ratio计算 | `pi_new / pi_safe` | `[B]` | 策略比率 |
| loss计算 | `-(ratio * advantages)` | `[B]` | 逐样本loss |
| mask | `mask = (adv != 0)` | `[B]` | 过滤中性样本 |
| **平均** | `sum() / count` | `scalar` | **样本级平均** |

---

## 三、关键代码解析

### 3.1 序列联合概率计算

```python
# decoder.py:403-405
probs_ = F.log_softmax(output_logits.view(batch_size, seq_len_minus1, vocab_size), -1)
# shape: [B, 3, 1502]

labels_ = labels.view(batch_size, seq_len_minus1)
# shape: [B, 3]

probs_ = torch.exp(
    torch.gather(probs_, dim=2, index=labels_.unsqueeze(-1))  # [B, 3, 1]
    .squeeze(-1)  # [B, 3]
    .sum(dim=-1)  # [B] ← 关键：3层log概率相加
)
# shape: [B]  ← 每个样本一个序列联合概率
```

**数学公式**:
```
seq_prob = exp(log_p1 + log_p2 + log_p3)
         = exp(log(p1)) × exp(log(p2)) × exp(log(p3))
         = p1 × p2 × p3
```

### 3.2 RL Loss 计算

```python
# decoder.py:316-354
def rl_loss(self, features: Dict[str, torch.Tensor], pi_new: torch.Tensor):
    # pi_new: [B] 序列联合概率
    # advantages: [B] 每个样本的advantage
    
    pi_safe = torch.where(features['advantages'] > 0.0, pi_new, 1.0 - pi_new)
    # pi_safe: [B] 安全策略（正样本用pi_new，负样本用1-pi_new）
    
    ratio = pi_new / pi_safe.detach()
    # ratio: [B] 新旧策略比率
    
    mask = (features['advantages'] != 0.0)
    # mask: [B] 过滤中性样本
    
    loss = -(ratio * features['advantages'])
    # loss: [B] 逐样本loss（未平均）
    
    loss = (loss * mask).sum() / (mask.sum().clamp_min(1))
    # scalar: 样本级平均
    
    return loss
```

---

## 四、序列级别 vs Token 级别

### 4.1 序列级别（当前实现）

```python
# 先算序列联合概率（3个token相乘）
seq_log_prob = log_p1 + log_p2 + log_p3  # [B]
seq_prob = exp(seq_log_prob)  # [B]

# 然后序列概率 × advantage
loss = -(seq_prob * advantage)  # [B]
```

**特点**:
- 优化目标：生成**完整序列**的概率
- 适用场景：推荐系统（关注最终视频ID是否正确）
- Loss量级：较小（因为seq_prob是3个<1的数相乘）

### 4.2 Token 级别（假设）

```python
# 每个token单独 × advantage，再相加
loss_token1 = -p1 * advantage  # [B]
loss_token2 = -p2 * advantage  # [B]  
loss_token3 = -p3 * advantage  # [B]

loss = loss_token1 + loss_token2 + loss_token3  # [B]
# 等价于: loss = -(p1 + p2 + p3) * advantage
```

**特点**:
- 优化目标：每个位置的预测准确度
- 适用场景：语言模型（每个token都重要）
- Loss量级：较大（因为p1+p2+p3 > p1×p2×p3）

### 4.3 数学区别

```
假设 p1=0.1, p2=0.2, p3=0.3, advantage=+4

序列级别:
  seq_prob = 0.1 × 0.2 × 0.3 = 0.006
  loss = -(0.006 × 4) = -0.024

Token级别:
  loss = -(0.1×4 + 0.2×4 + 0.3×4) = -(0.4 + 0.8 + 1.2) = -2.4

差异: 序列级别 loss 比 Token 级别小 100 倍！
```

---

## 五、具体例子

### 5.1 输入数据

```python
batch_size = 2

# 3层码本的预测概率
样本1: p1=0.10, p2=0.20, p3=0.30, advantage=+4.0  (正样本)
样本2: p1=0.05, p2=0.10, p3=0.15, advantage=-6.0  (负样本)

# 中间变量
样本1 seq_prob = 0.10 × 0.20 × 0.30 = 0.006
样本2 seq_prob = 0.05 × 0.10 × 0.15 = 0.00075
```

### 5.2 计算过程

| 步骤 | 样本1 | 样本2 | 说明 |
|------|-------|-------|------|
| pi_new | 0.006 | 0.00075 | 序列联合概率 |
| pi_safe | 0.006 | 0.99925 | 正样本用pi_new，负样本用1-pi_new |
| ratio | 1.0 | 0.00075 | pi_new / pi_safe |
| advantage | +4.0 | -6.0 | 预计算的奖励权重 |
| loss | -(1.0×4.0)=**-4.0** | -(0.00075×-6.0)=**+0.0045** | 逐样本loss |
| mask | 1 | 1 | 都非中性样本 |

### 5.3 最终平均

```python
rl_loss = (-4.0 + 0.0045) / 2 = -1.99775 ≈ -2.0
```

**注意**：
- 正样本的 loss 绝对值很大（-4.0），因为 ratio≈1
- 负样本的 loss 很小（+0.0045），因为 pi_new 很小，ratio 很小
- 这导致模型会优先优化正样本（让正样本的序列概率更高）

---

## 六、为什么用序列级别

### 6.1 推荐场景的特点

| 特点 | 说明 |
|------|------|
| **最终ID最重要** | 用户只看到最终推荐的视频，不关心中间层 |
| **层级依赖** | 第2层依赖第1层，第3层依赖第2层，必须同时正确 |
| **联合概率** | 3层都对的概率 = p1 × p2 × p3 |

### 6.2 如果用 Token 级别会怎样

```
假设模型学会了：
  p1 = 0.9 (第一层很准)
  p2 = 0.9 (第二层很准)
  p3 = 0.1 (第三层不准)

Token级别 Loss: -(0.9 + 0.9 + 0.1) × adv = -1.9 × adv
  → 看起来 loss 很小，模型以为学得不错

序列级别 Loss: -(0.9 × 0.9 × 0.1) × adv = -0.081 × adv
  → loss 很小（绝对值），但实际上几乎无法反解出正确URL！
```

**结论**：序列级别更贴合推荐场景，因为只有3层都对才能正确推荐。

---

## 七、调试验证

如果你想验证当前的实现，可以添加打印：

```python
# 在 decoder.py:405 后添加
print(f"seq_prob shape: {probs_.shape}")  # 应该是 [B]
print(f"seq_prob range: [{probs_.min():.6f}, {probs_.max():.6f}]")
# 典型值: [0.000001, 0.01] （非常小，因为3个<1的数相乘）

# 在 decoder.py:346 前添加
print(f"loss_per_sample: {loss}")
print(f"advantages: {features['advantages']}")
```

预期输出：
```
seq_prob shape: torch.Size([1024])
seq_prob range: [0.000001, 0.008000]
loss_per_sample: tensor([-3.5, -2.1, 0.8, ..., -4.2])
advantages: tensor([4.0, 4.0, -6.0, ..., 4.0])
```
