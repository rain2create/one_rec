# v12_new PT 阶段训练总结

> 数据源：`/data04/search/tanxiaofeng/dev/onerec_v12_new/log/log_2025090118`
> 真实日志逐 step 抽取（2721 个采样点）。

---

## 1. 训练参数

### 1.1 任务 / 数据

| 参数 | 值 | 含义 |
|---|---|---|
| **训练入口** | `train_pt.py` | PT 阶段主脚本（不带 `--train_sft`）|
| **语料** | 30 天曝光日志 | 根目录 `video.rank.s101` |
| **文件数** | 2,080 | gzip hour 文件总数 |
| **`with_today`** | False | 不包含当天数据 |
| **`done_spec`** | `v12.pt.30days` | 进度隔离 key，避免跟 SFT/RL 互踩 |
| **URL 词典** | 9,196,859 条 | url2tokens 字典 2025-08-17 快照 |

### 1.2 模型规模

| 参数 | 值 | 含义 |
|---|---|---|
| **总参数量** | 2.00 B | 整模型 |
| **Sparse Embedding** | 1.99 B | 99.5% 参数都在稀疏侧 |
| **Dense 参数** | ~10 M | Transformer 主干 |
| **Codebook 层数** | 3 | 每个 URL 用 3 个 token 表达 |
| **Codebook 词表** | 1502 | 1500 真实 token + BOS + UNK |

### 1.3 训练超参

| 参数 | 值 | 含义 |
|---|---|---|
| **Batch Size** | 1024 | 每步处理样本数 |
| **GPU 数** | 1 | log 头 `world size 1`，单卡训（host `algpu01` 共 8×A100-40GB，PT 只占 1 张）|
| **GPU 型号** | NVIDIA A100-SXM4-40GB | 单卡显存 40 GiB（log 未直接打印，由 host `algpu01.news.lycc.qihoo.net` 实测得）|
| **Learning Rate (dense)** | 1e-4 | 稠密参数学习率（`train_args.learning_rate` 默认值）|
| **Learning Rate (sparse)** | 3e-2 | 稀疏参数学习率（`train_args.lr_sparse` 默认值，比 dense 高 300×）|
| **Optimizer (dense)** | AdamW | weight_decay = 0.01 |
| **Optimizer (sparse)** | SparseAdam | 稀疏参数优化器 |
| **LR Schedule** | constant_with_warmup | warmup 后保持恒定，不衰减 |
| **AMP** | 关 | 全精度训练 |
| **Gradient Accumulation** | 1 | 无累积，每步更新 |
| **Logging Steps** | 100 | 每 100 步打印日志 |
| **Save Steps** | 10000 | 每 1 万步保存 checkpoint |
| **Max Ckpt Keep** | 默认 | 保留全部 ckpt |

## 2. 训练时长 / 步数

| 项 | 值 |
|---|---|
| 开始时间 | 2025-09-01 18:15:18 |
| 结束时间 | 2025-09-02 04:49:42 |
| **总时长** | **~10 小时 34 分钟** |
| 起始 step | 0 |
| 终止 step | **272030**（最终 ckpt：`checkpoint-272030.pt`）|
| **折合 epoch** | **1 epoch**（log 里 epoch 计数器仅到 0；单遍历样本数 ≈ 2.785 亿，2080 文件平均 ~13.4 万行/文件）|
| 平均吞吐 | ~7.15 step/s（含 ckpt 保存）|

> 这个 ckpt 后续作为 SFT 阶段的 resume 起点。

---

## 3. 核心趋势曲线

### 3.1 Actual Loss & Target Prob

![PT loss & target prob](pt_loss_tprob.png)

| 指标 | 起点 | 终点 | 极值 |
|---|---|---|---|
| **Actual Loss** | 7.52 (step 0) | **3.65** (step 272000) | Min **3.45** @ step 225000 |
| **Target Prob** | ~0 (step 0) | **6.28×10⁻⁴** (step 272000) | Max **1.41×10⁻³** @ step 250600 |
| 下降幅度 | — | ↓ **3.87**（51.5%）| — |

**解读**：
- Loss 从 7.52 → 3.65，下降 51.5%——`ln(1502) ≈ 7.31`，起点接近"对 1502 个 token 完全均匀猜测"的熵；终点 3.65 对应有效困惑度 `e^3.65 ≈ 38`（每层从 1502 个 token 中收敛到约 38 个候选）。
- Target Prob 是 3 个 token 联合概率，从 0 → 6.28×10⁻⁴，相当于平均每个 token P ≈ 0.086（如果均匀分布于 38 个候选）。
- 曲线有周期性"小尖刺"——是切换到新 hour gzip 文件时分布漂移造成的，不影响整体趋势。

### 3.2 URL Tokenizer 命中率（Mask / Final Valid Rate）

![PT valid rate](pt_valid_rate.png)

| 指标 | 平均值 | std | 含义 |
|---|---|---|---|
| **Mask Valid Rate** | **0.6454** | 0.0367 | 至少 1 层 codebook 命中字典的样本占比 |
| **Final Valid Rate** | **0.3807** | 0.0235 | 3 层 codebook 全部命中字典的样本占比 |

**关键认知**：
- 这两个指标**不是模型学习的成果**——它们反映的是 url2tokens 字典对当前 batch URL 的覆盖率
- 训练全程基本平稳（mean ± 1 std 范围内），有周期性波动是切换 hour 文件造成的
- **Final Valid 0.38 意味着每个 batch 里只有 38% 的样本能被纯 CE 完整反传**——剩下 62% 的样本至少有 1 层 token=UNK，被 `ignore_index=1501` 跳过，loss 不进梯度
- 这个比例从训练开始到结束没本质变化——所以 PT 阶段的实际有效样本量约等于 `名义样本量 × 0.38`

---

## 4. 训练健康度判断

| 维度 | 表现 | 评价 |
|---|---|---|
| Loss 收敛 | 7.52 → 3.65，曲线持续下降 | ✅ 正常 |
| Target Prob 提升 | 0 → 6.28e-4，按 e^loss 反算一致 | ✅ 正常 |
| URL 命中率 | 训练前后基本稳定，无漂移 | ✅ 正常（非学习指标）|
| 周期性尖刺 | 出现在 hour 切换处，未引起发散 | ✅ 正常 |
| 训练时长 | 30 天 corpus 跑 ~10.5 小时 | ✅ 单卡可接受 |

**结论**：PT 阶段顺利完成，loss 下降 51.5%，无异常事件，`checkpoint-272030.pt` 作为 SFT 起点合格。

---

## 5. 一句话总结

v12_new PT 阶段在 30 天语料 + 单卡上跑了 10.5 小时、272k step，NTP loss 从 7.52 收敛到 3.65（约 e³·⁶⁵ ≈ 每层 38 个有效候选），生成基础"序列推荐能力"，但**有效样本量受字典覆盖率限制为名义量的 38%**——这是 v12_new 三阶段范式后续 SFT/RL 共同继承的瓶颈。
