# One-Rec Model Architecture

## 概述

One-Rec是一个基于**Encoder-Decoder架构**的生成式推荐模型，通过生成目标物品的**语义ID(Semantic ID)**来完成推荐任务。

## 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RecModel                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ SparseModel  │ -> │   Encoder    │ -> │       Decoder        │  │
│  │  (Embedding) │    │ (Transformer)│    │ (MOE-Transformer)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 模块详解

### 1. SparseModel (`sparse_model.py`)

**功能**: 稀疏特征嵌入层，将离散特征转换为稠密向量

**处理特征类型**:
- **用户特征** (4维): `sign`, `autok`, `autoplay`, `reqtimes`
- **行为序列特征** (3组，每组30个历史行为):
  - `logsess`: 用户session行为序列
  - `logqueue`: 队列行为序列  
  - `logreallike`: 点赞行为序列

**核心设计**:
```python
# 每个特征独立Embedding，不共享
feature_embeddings = nn.ModuleDict({
    name: nn.Embedding(vocab_size, emb_dim, sparse=True)
    for name, vocab_size in feature_vocabs.items()
})

# 物品码本使用3层独立Embedding
code_books_embeddings = nn.ModuleList([
    nn.Embedding(1502, emb_dim, sparse=True) for i in range(3)
])
```

**输出**: 4个拼接后的特征向量列表 `[user_features, logsess_features, logqueue_features, logreallike_features]`

---

### 2. Encoder (`encoder.py`)

**功能**: 将用户特征和行为序列编码为上下文表示

**架构流程**:
```
输入特征 -> 各类特征投影MLP -> 拼接 -> 位置编码 -> LayerNorm -> 输出
```

**关键参数**:
- `emb_dim`: 嵌入维度 (默认128)
- `model_dim`: 模型维度 (默认256)
- `max_seq_len`: 最大序列长度 (默认128)

**序列结构** (共91个token):
- 1个用户特征token
- 30个logsess行为token
- 30个logqueue行为token
- 30个logreallike行为token

**输出**: 
- `att_emb`: 编码后的上下文表示 `[batch_size, 91, model_dim]`
- `attn_mask`: padding mask矩阵

---

### 3. Decoder (`decoder.py`)

**功能**: 基于MOE的Transformer解码器，生成目标物品的语义ID

#### 3.1 码本机制
- **3层码本** (codebook_layers=3): 每个物品由3个token表示
- **码本大小** (codebook_size=1500): 每层码本包含1500个token
- **BOS Token**: 序列起始符 (id=0)
- **权重绑定** (weight_tying): 输出层权重与对应Embedding共享

#### 3.2 MOE架构
使用 `DecoderMOE` + `DecoderLayerMOE` 结构:
```
输入 -> Self-Attention -> Cross-Attention(Encoder K/V) -> MOE/FFN -> 输出
```

**MOE配置**:
- 专家数量: 24
- Top-k路由: 2
- 辅助损失权重: 0.01

#### 3.3 生成策略
| 方法 | 描述 |
|------|------|
| `rl_generate` | Top-k/Top-p采样生成 |
| `bs_rl_generate` | Beam Search生成 |
| `bs_rl_generate_mask` | 带历史物品mask的Beam Search |

---

### 4. MOE模块 (`moe_auxloss.py`, `decoder_moe.py`)

**Router**: Top-k路由选择
```
输入 -> 线性投影 -> Top-k选择 -> Softmax -> 专家权重
```

**MOELayer**: 专家计算与聚合
```
输入 -> Router选择 -> 各专家并行计算 -> 加权聚合 -> 输出
```

**辅助损失**: 负载均衡损失，确保专家均匀使用

---

## 模型传导流程

### 训练阶段

```
输入 url_ids: [BOS, code1, code2, code3]  形状 [B, 4]

步骤1: 切片取前3个作为Decoder输入
  input_ids = url_ids[:, 0:-1] = [BOS, code1, code2]  形状 [B, 3]

步骤2: Embedding查表 (每位置独立)
  位置0 (BOS)   → embeddings[0]
  位置1 (code1) → embeddings[1]
  位置2 (code2) → embeddings[2]

步骤3: Transformer Decoder
  Self-Attention (因果mask) + Cross-Attention + MOE FFN
  输出: decoder_out [B, 3, D]

步骤4: 输出层计算 (Weight Tying)
  位置0 → output_layers[0] = embeddings[1] → 预测 code1
  位置1 → output_layers[1] = embeddings[2] → 预测 code2
  位置2 → output_layers[2] = embeddings[3] → 预测 code3

步骤5: 归一化 + 温度缩放
  logits = normalize(hidden) @ normalize(weight).T * temperature

输出:
  logits: [B, 3, V=1502]
  labels: [B, 3] = [code1, code2, code3]
```

### 生成阶段

```
Step 0: 输入 [BOS]        → 预测 code1 (ensure_matrix[0] 约束)
Step 1: 输入 [BOS, code1] → 预测 code2 (ensure_matrix[0] 约束)
Step 2: 输入 [BOS, c1, c2] → 预测 code3 (ensure_matrix[1] 约束)

输出: [code1, code2, code3] → Tokenizer.decode → URL
```

---

## 关键设计

### 1. Weight Tying (权重绑定)

```python
# 输出层权重 = 对应位置Embedding权重
# 位置对应关系:
#   预测 code1 → output_layers[0].weight = embeddings[1].weight
#   预测 code2 → output_layers[1].weight = embeddings[2].weight
#   预测 code3 → output_layers[2].weight = embeddings[3].weight

for i in range(codebook_layers):
    self.output_layers[i].weight = self.embeddings[i+1].weight
```

**优势**:
- 减少参数量，防止过拟合
- 输入空间和输出空间一致，语义更统一
- 梯度累加，两个任务共同优化同一个表示空间

### 2. ensure_matrix (码本转移约束)

```python
# 形状: [2, 1502, 1502]
# ensure_matrix[0][i][j] = 1.0 表示 code1=i 时 code2=j 合法
# ensure_matrix[1][i][j] = 1.0 表示 code2=i 时 code3=j 合法

# 生成时使用:
if i == 1 or i == 2:
    logit_bias = self.ensure_matrix[i-1][input_ids[:, -1]]
    cur_l_logit += logit_bias  # 合法位置 +1.0，非法位置 +0
```

**作用**: 保证生成的码本能解码成真实存在的URL

### 3. MOE Router 详解

```python
# Step 1: 计算专家分数
logits = w_g(x)  # [B, T, n_exp]

# Step 2: 选top-k专家
top_k_logits, top_k_indices = logits.topk(k, dim=-1)

# Step 3: 只有top-k位置有非零概率
router_probs = softmax(top_k位置)

# Step 4: 计算每个token在expert内的slot位置
exp_rank = cumsum(exp_mask) - 1

# Step 5: 容量裁剪
exp_mask *= (exp_rank < capacity)

# Step 6: 专家计算 + 加权聚合
output = exp_weight @ exp_out  # 只有top-k专家贡献
```

---

## Loss 计算

### Loss 组成

```
Total Loss = 对比学习Loss + RL Loss + MOE Aux Loss
```

### 对比学习 Loss

```python
# 处理 advantages >= 0 的样本
loss = CrossEntropy(output_logits, labels)  # -log(p)
loss_neg = -log(1 - p)  # 让概率下降的loss形式

# 根据 advantages 选择:
#   advantages > 0 (正样本): loss = -log(p)      → 提高概率
#   advantages = 0 (中性样本): loss = -log(1-p)  → 降低概率
#   advantages < 0 (负样本): 不参与对比学习
```

### RL Loss

```python
# 处理 advantages != 0 的样本
probs = P(code1) * P(code2) * P(code3)  # 整个URL的生成概率

mask = (advantages != 0.0)
loss = -(ratio * advantages)
loss = (loss * mask).sum() / mask.sum()

#   advantages > 0: loss < 0 → 提高 probs
#   advantages < 0: loss > 0 → 降低 probs
#   advantages = 0: 不参与
```

### advantages 计算 (预训练阶段)

```python
w = 0
w = where(like == '1', w + 4, w)      # 点赞 +4
w = where(dislike == '1', w - 4, w)   # 不喜欢 -4
w = where(follow == '1', w + 4, w)    # 关注 +4
w = where(finish == '1', w + 0.4, w)  # 完播 +0.4
w = where(skip == '1', w - 0.2, w)    # 跳过 -0.2
...
advantages = w + duration_adv
```

### Loss 分工

| 样本类型 | advantages | 对比学习 | RL Loss |
|---------|------------|---------|---------|
| 正样本 (点赞/关注) | > 0 | ✓ 提高概率 | ✓ 提高概率 |
| 中性样本 (曝光) | = 0 | ✓ 降低概率 | ✗ 不参与 |
| 负样本 (不喜欢/跳过) | < 0 | ✗ 不参与 | ✓ 降低概率 |

---

## 数据流示意

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 输入数据                                                                 │
│ - 用户特征: [sign, autok, autoplay, reqtimes]                           │
│ - 历史行为: logsess(30), logqueue(30), logreallike(30)                  │
│ - 目标物品: url (3层码本token)                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SparseModel                                                             │
│ - 各特征独立Embedding                                                    │
│ - 物品码本3层Embedding拼接                                               │
│ 输出: [user_emb, logsess_emb, logqueue_emb, logreallike_emb]           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Encoder（context_processor ）                                           │
│ - 特征投影MLP (4类特征各自投影)                                          │
│ - 拼接 + 位置编码 + LayerNorm                                            │
│ 输出: context_emb [B, 91, model_dim], attn_mask [B, 91]                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Decoder                                                                 │
│ 输入: [BOS, code1, code2]  (训练) / [BOS] (生成)                        │
│ - Self-Attention (causal mask)                                          │
│ - Cross-Attention (encoder K/V)                                         │
│ - MOE FFN (24专家, Top-2路由)                                            │
│ 输出: logits [B, 3, codebook_size+2]                                    │
│ Loss: 对比学习Loss + RL Loss + MOE Aux Loss                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 推理生成                                                                 │
│ Beam Search -> [code1, code2, code3] -> Tokenizer.decode -> URL        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 关键文件说明

| 文件 | 功能 | 是否使用 |
|------|------|----------|
| `rec_model.py` | 主模型，组合SparseModel+Encoder+Decoder | ✅ 核心 |
| `sparse_model.py` | 稀疏特征嵌入层 | ✅ 核心 |
| `encoder.py` | Transformer编码器 | ✅ 核心 |
| `decoder.py` | MOE Transformer解码器 | ✅ 核心 |
| `decoder_moe.py` | MOE解码器层实现 | ✅ 核心 |
| `moe_auxloss.py` | MOE路由与辅助损失 | ✅ 核心 |
| `manager.py` | MOE辅助损失管理器 | ✅ 核心 |
| `moe_auxloss_bak.py` | MOE备份版本 | ❌ 未使用 |
| `moe_auxloss_.py` | MOE旧版本 | ❌ 未使用 |
| `moe_auxloss_v1.py` | MOE v1版本 | ❌ 未使用 |

---

## 配置参数 (ModelArguments)

```python
# Encoder参数
emb_dim = 128           # 嵌入维度
model_dim = 256         # 模型维度
encoder_num_layers = 6  # Encoder层数
encoder_num_heads = 4   # 注意力头数

# Decoder参数
decoder_num_layers = 6      # Decoder层数
decoder_num_heads = 4       # 注意力头数
decoder_codebook_layers = 3 # 码本层数
decoder_codebook_size = 1500 # 码本大小
decoder_temperature = 10.0  # logits温度系数
```

---

## 使用示例

```python
from model.rec_model import RecModel
from arguments.model_args import ModelArguments

# 创建模型
args = ModelArguments()
model = RecModel.from_args(args)

# 前向传播
output_logits, loss, mask_valid, final_valid, probs = model(features)

# 推理生成
output = model.generate(features, group_num=10)
```
