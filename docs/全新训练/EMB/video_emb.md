# Video Embedding 生成模型（verttagembcf）

代码路径：`/mnt/code/tanxiaofeng/DeepNeuralRecall/`

---

## 1. 整体流程

```
视频内容标签 (md5 + tag列表)
        ↓ verttagembcf_prepare.py（数据预处理）
标签序列文件 (md5 \t tag1|tag2|tag3|...)
        ↓ bert_trainer.py（训练）
BERT 模型权重
        ↓ get_verttagemb.py（推理）
64维 L2归一化 video embedding
        ↓
上传 S3，key = "url_emb_mask"
        ↓
RQ-KMeans 聚类 → [T1, T2, T3] Semantic ID
```

---

## 2. 数据预处理

**文件**：`prepare/verttagembcf_prepare.py`

**一个视频一条样本，一条样本有多个 tag（变长）**

输入格式（原始语料）：
```
md5 \t tag_1 \t tag_2 \t tag_3 \t ...
```

输入语义：**以视频为单位，它的垂类内容标签列表作为"序列"**（不是用户行为序列）

## min_session_length = 5  max_seq_length = 20

**tag list 长度**：每个视频的 tag 数量不固定
- 少于 `min_session_length` 的视频直接过滤（序列太短没意义）
- 超过 `max_seq_length` 的截断取前 N 个
- 具体数值在 conf 文件里配置（`[prepare]` 段）

处理逻辑：
- 清洗标签（去掉逗号、竖线等特殊字符）
- 过滤 tag 数不足 min_session_length 的视频
- 低频标签上采样：若某个 `sq_` 前缀标签出现次数 < 200，整条样本重复 `200/count` 次
- 输出格式：`md5 \t tag_1|tag_2|tag_3|...`


---

## 3. 模型结构

**文件**：`models/bert.py`，类名 `Bert`

架构：**双向 Transformer Encoder（BERT 风格，不是 GPT 单向）**
## 4层block 8个head

```
输入：标签 ID 序列 [tag_1, tag_2, ..., tag_n]
        ↓ Embedding 查表（vocab_size × hidden_size=64）
        ↓ LayerNorm
        ↓ TransformerEncoderBlock × N 层
             ├── Multi-Head Self-Attention（双向，无因果掩码）
             └── FFN（inner_dim → hidden_size）
        ↓ hidden states，shape: [batch, seq_len, 64]
```

关键配置（从 config 读取）：
| 参数 | 含义 |
|---|---|
| `bert_dim` | hidden size，= 64 |
| `bert_deep` | Transformer 层数 |
| `bert_header` | attention 头数 |
| `bert_inner_dim` | FFN 中间维度 |
| `lbl_rate` | mask 比例（如 15%） |
| `max_labels` | 每条序列最多 mask 几个 |
| `neg_size` | 每个正样本对应的负样本数 |

**输出表示：CLS token 汇总**

模型输出的是每个位置的 hidden state，但最终只取 CLS 位置（位置 0）的那一个 64 维向量作为整个视频的表示：

```
输入: [CLS, 科技, 手机, 评测, 互联网, 数码]
         ↓
Transformer Encoder（双向 Self-Attention，每个位置能看到所有其他位置）
         ↓
输出: [h_CLS, h_科技, h_手机, h_评测, h_互联网, h_数码]
         ↑
       只取这一个，64维，其余丢掉
```

CLS 没有实际语义，模型将其作为"汇总槽"，通过 Self-Attention 聚合整个 tag 序列的信息。不是 concat，不是 mean pooling，就是 CLS 的单个输出向量。

**Weight Tying**：输入 Embedding 和输出打分层共享同一组权重

---

## 4. 样本构造（Masked Token Prediction）

**文件**：`trainers/bert_trainer.py`

```
原始序列：[tag_1, tag_2, tag_3, tag_4, tag_5]

Step 1：随机选 lbl_rate% 的位置（至少1个，最多 max_labels 个）
         假设选中位置 [1, 3]

Step 2：把选中位置替换为 Mask token（id = 1）
         输入序列：[tag_1, MASK, tag_3, MASK, tag_5]

Step 3：记录被 mask 的信息
         lbl_idx  = [1, 3]          ← 哪些位置被 mask 了
         lbl_ids  = [tag_2, tag_4]  ← 那些位置原始 ID
         lbl_weight = [1.0, 1.0]    ← 各位置的权重（可加权）
```

词表构建：
- `id=0`：padding（None）
- `id=1`：Mask token
- `id=2+`：实际 tag ID（按频率排序建表）

---

## 5. 目标函数（采样 Softmax） 就是单纯做CE

**文件**：`models/bert.py`，`call()` 函数

**预测机制：无 MLP，直接点积**

mask 位置的 hidden state 不经过任何额外的 MLP，直接和候选 tag 的 embedding 做点积打分：

```
h1（MASK位置的 hidden state，64维）

  ·  embedding(手机)    → score(手机)   ← 正样本（被遮住的真实tag）
  ·  embedding(游戏)    → score(游戏)   ← 负样本（从词频CDF随机采的）
  ·  embedding(美食)    → score(美食)   ← 负样本
  ...（共 1 + neg_size 个）

↓ softmax over [score(手机), score(游戏), score(美食), ...]

  P(手机) / P(游戏) / P(美食) / ...

↓ loss = -log P(手机)   最大化正样本概率

  采样 Softmax（这里用的）：
  P(手机) ≈ exp(score_手机) / (exp(score_手机) + Σ exp(score_采样的少量负样本))
                                                   ↑
                                             分母只算 1 + neg_size 个
                                             比如负样本50个，每次只算51次点积
                                             快得多


```

- embedding 和输入层共享同一个矩阵（Weight Tying），没有额外参数
- 负样本从词频分布的 power 次方 CDF 里随机采（power < 1，高频词降权，类似 word2vec）
- 本质是 CE，但分母不对全词表求和，只对采样的少量负样本求和（效率近似）

优化器：**Adagrad**

---

## 6. 推理：提取视频 Embedding

**文件**：`infer/get_verttagemb.py`，`masked_emb_inference()` 函数

```python
# 在序列最前面拼上 CLS token（id = 1）
ids = concat([[CLS=1], tag_ids])[:max_length]

# 过 Transformer Encoder（双向）
hidden = TransformerEncoder(ids)      # [batch, seq_len, 64]

# 取第 0 个位置（CLS 位置）的输出
emb = hidden[:, 0, :]                 # [batch, 64]

# L2 归一化
emb = L2_normalize(emb)               # [batch, 64]
```

输出格式：
```
md5 \t e1 e2 e3 ... e64
```

---

## 7. 与 RQ-KMeans 的关系

```
video embedding（64维 L2归一化向量）
        ↓
RQ-KMeans 第1层：1500个聚类中心 → T1（0~1499）
        ↓ 计算残差（原始向量 - 第1层量化向量）
RQ-KMeans 第2层：1500个聚类中心 → T2（0~1499）
        ↓ 计算残差
RQ-KMeans 第3层：1500个聚类中心 → T3（0~1499）
        ↓
Semantic ID = [T1, T2, T3]  ← One-Rec Decoder 的生成目标
```

语义相近的视频（标签组合相似）→ 64维向量距离近 → 被分配到相同或相邻的 Semantic ID

---

## 8. 如何评估 Embedding 质量（i2i 相似度人工抽查）

训练完成后，upload 脚本会对全量视频向量做 ANN 近似最近邻搜索，给每个视频找 top-K 个最相似的视频：

```
视频A（手机评测）→ [视频B（iPhone开箱）, 视频C（安卓评测）, 视频D（数码配件）, ...]
视频X（美食探店）→ [视频Y（火锅测评）, 视频Z（重庆小吃）, ...]
```

**判断标准**：随机抽几个视频，看它的 top-10 相似结果是否语义上合理：
- top-K 都是同类内容 → embedding 质量好，模型学到了 tag 之间的语义关系
- top-K 出现跨类内容（如"手机评测"相似出"美食探店"）→ embedding 没学好

工具：`src/infer/look_sim.py`、`src/infer/item_sim.py`，用来可视化和人工 check 相似结果。
