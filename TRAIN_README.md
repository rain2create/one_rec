# One-Rec 训练流程说明

流程总结
┌─────────────────────────────────────────────────────────────┐
│                      初始化阶段                              │
│  1. 分布式设置                                               │
│  2. 创建 tokenizer, parser, dataset, dataloader             │
│  3. 创建模型 RecModel                                        │
│  4. 加载 ensure_matrix                                       │
│  5. DDP 封装                                                 │
│  6. 创建优化器（dense: AdamW, sparse: SparseAdam）           │
│  7. 恢复 checkpoint（如果有）                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      训练循环                                │
│                                                             │
│  for epoch in epochs:                                       │
│      while True:                                            │
│          1. 获取 batch                                       │
│          2. 同步所有 rank 是否结束                           │
│          3. 处理 dummy batch（分布式填充）                    │
│          4. 前向传播 → output_logits, loss, probs           │
│          5. 反向传播 → loss.backward()                       │
│          6. 梯度累积步数到达时：                              │
│             - 梯度裁剪                                       │
│             - 更新 dense 参数                   │
│             - 更新 sparse 参数                │
│             - 更新学习率                                     │
│             - 同步 sparse 参数（分布式）                      │
│          7. 打印日志、保存 checkpoint                        │
│          8. global_step += 1                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      结束阶段                                │
│  1. 保存最终 checkpoint                                      │
│  2. 清理分布式进程组                                         │
│  3. 标记训练完成                                             │
└─────────────────────────────────────────────────────────────┘

## 详细说明
## 一、初始化阶段

```
1. 分布式设置
   - 检测是否分布式训练（通过环境变量 RANK, WORLD_SIZE）
   - 获取 local_rank，决定使用哪块 GPU

2. 数据加载
   - UrlTokenizer: URL <-> 码本编解码
   - PtParser / SFTParser: 解析原始日志数据，提取特征
   - PtDataset: 流式数据集（支持大规模语料）
   - DataLoader: batch_size=None（batch 在 dataset 内部分好）

3. 模型构建
   - RecModel: SparseModel + Encoder + Decoder
   - 加载 ensure_matrix（码本合法转移矩阵）

4. DDP 封装
   - Encoder 和 Decoder 用 DDP 封装
   - SparseModel 不用 DDP（单独同步机制）

5. 优化器
   - optimizer_dense: AdamW，优化 Transformer 参数
   - optimizer_sparse: SparseAdam，优化稀疏 Embedding

6. 恢复训练（可选）
   - 从 checkpoint 恢复模型、优化器状态
```

## 二、训练主循环

```
for epoch in epochs:
    while True:
        │
        ├─ 1. 获取 batch
        │     - try: batch = next(data_iter)
        │     - except: batch = None, local_finished = 1
        │
        ├─ 2. 同步所有 rank
        │     - all_reduce(global_finished)
        │     - 所有 rank 都结束 → break
        │
        ├─ 3. 处理 dummy batch
        │     - 数据先结束的 rank 用 dummy_batch 填充
        │     - 保持分布式同步
        │
        ├─ 4. 前向传播
        │     - output_logits, loss, mask_valid, final_valid, probs = model(batch)
        │     - loss = loss / gradient_accumulation_steps
        │
        ├─ 5. 反向传播
        │     - scaler.scale(loss).backward()
        │
        └─ 6. 梯度累积步数到达时：
              │
              ├─ unscale 梯度
              │
              ├─ 梯度裁剪（只裁剪 dense 参数）
              │
              ├─ 更新参数
              │     - optimizer_dense.step()
              │     - optimizer_sparse.step()
              │
              ├─ 更新学习率
              │     - lr_scheduler.step()
              │
              ├─ 同步 sparse 参数（分布式）
              │     - reduce_sparse_parameters()
              │
              ├─ 打印日志（主进程）
              │     - loss, probs
              │     - pos_pi_new, neg_pi_new, cts_pi_new
              │     - ct_loss, rl_loss
              │
              └─ 保存 checkpoint（主进程）
```

## 三、结束阶段

```
1. 保存最终 checkpoint
2. 清理分布式进程组（cleanup）
3. 标记训练完成（after_train_process）
```

## 四、模型前向传播

```
model(batch) 返回:
  - output_logits: [B, 3, V]  每层码本的预测 logits
  - loss: 总损失（对比学习 + RL + MOE aux_loss）
  - mask_valid: 有效样本数
  - final_valid: 有效样本比例
  - probs: 目标物品的概率

Loss 组成:
  Total Loss = 对比学习 Loss + 强化学习 Loss + MOE Aux Loss

  | 样本类型       | advantages | 对比学习        | RL Loss     |
  |---------------|------------|----------------|-------------|
  | 正样本 (点赞)   | > 0        | 提高 target 概率 | 提高 target 概率 |
  | 中性样本 (曝光) | = 0        | 降低非 target 概率 | 不参与 |
  | 负样本 (不喜欢) | < 0        | 不参与          | 降低 target 概率 |

advantages 计算:
  w = 0
  w = where(like == '1', w + 4, w)      # 点赞 +4
  w = where(dislike == '1', w - 4, w)   # 不喜欢 -4
  w = where(follow == '1', w + 4, w)    # 关注 +4
  w = where(finish == '1', w + 0.4, w)  # 完播 +0.4
  w = where(skip == '1', w - 0.2, w)    # 跳过 -0.2
  advantages = w + duration_adv
```

## 五、分布式训练相关

```
术语:
  - rank: 当前进程编号 (0, 1, 2, ...)
  - local_rank: 当前节点内的 GPU 编号
  - world_size: 总进程数
  - rank 0: 主进程，负责日志、保存模型
  - all_reduce: 所有进程汇总数据
  - barrier: 所有进程在此等待同步

流程示意:

  Rank 0 (主进程)    Rank 1          Rank 2          Rank 3
  ┌─────────┐      ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ GPU 0   │      │ GPU 1   │     │ GPU 2   │     │ GPU 3   │
  │ 数据 A  │      │ 数据 B  │     │ 数据 C  │     │ 数据 D  │
  │ 梯度 g0 │      │ 梯度 g1 │     │ 梯度 g2 │     │ 梯度 g3 │
  └────┬────┘      └────┬────┘     └────┬────┘     └────┬────┘
       │                │               │               │
       └────────────────┴───────┬───────┴───────────────┘
                               ↓
                  all_reduce (梯度平均)
                  avg_grad = (g0+g1+g2+g3) / 4
                               ↓
                  所有 rank 用相同的梯度更新模型
```

## 六、关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `learning_rate` | dense 参数学习率 | 2e-4 |
| `lr_sparse` | sparse 参数学习率 | 1e-2 |
| `gradient_accumulation_steps` | 梯度累积步数 | - |
| `max_grad_norm` | 梯度裁剪阈值 | - |
| `warmup_steps` | 学习率 warmup 步数 | - |
| `save_steps` | 保存 checkpoint 间隔 | - |
| `logging_steps` | 打印日志间隔 | - |
| `epochs` | 训练轮数 | - |