# 案例：Decode Attention Vector 合轴 + 合图优化

## 场景

GQA decode attention 算子中，softmax 及前后的 vector 操作（mul/amax/sub/exp/sum/div/cast）在 3D shape `[8, 4, 2048]` 下执行，产生 6 个独立 vector 子图，调度开销大。

## 核心优化

1. matmul 输出的 3D tensor reshape(inplace) 为 2D
2. 所有 vector 操作在 2D 下执行 + sg_set_scope 合图
3. reshape 回 3D 传给下一个 matmul

## 代码对比

### 优化前（3D，6 个子图）

```python
scores_fp32 = pypto.matmul(q_grouped, k_cache, pypto.DT_FP32, b_trans=True)
scores_scaled = pypto.mul(scores_fp32, scale)       # [8,4,2048] FP32
row_max = pypto.amax(scores_scaled, dim=-1, keepdim=True)
scores_sub = pypto.sub(scores_scaled, row_max)
exp_scores = pypto.exp(scores_sub)
exp_sum = pypto.sum(exp_scores, dim=-1, keepdim=True)
attn_weights_fp32 = pypto.div(exp_scores, exp_sum)
attn_weights = pypto.cast(attn_weights_fp32, pypto.DT_BF16)
# → 传给 matmul(attn_weights, v_cache)
```

### 优化后（2D，1 个合图子图）

```python
scores_fp32 = pypto.matmul(q_grouped, k_cache, pypto.DT_FP32, b_trans=True)
scores_2d = pypto.reshape(scores_fp32, [32, 2048], inplace=True)
pypto.set_vec_tile_shapes(8, 2048)

pypto.set_pass_options(sg_set_scope=1)
scores_scaled = pypto.mul(scores_2d, scale)
row_max = pypto.amax(scores_scaled, dim=-1, keepdim=True)
scores_sub = pypto.sub(scores_scaled, row_max)
exp_scores = pypto.exp(scores_sub)
exp_sum = pypto.sum(exp_scores, dim=-1, keepdim=True)
attn_fp32 = pypto.div(exp_scores, exp_sum)
attn_bf16 = pypto.cast(attn_fp32, pypto.DT_BF16)
pypto.set_pass_options(sg_set_scope=-1)

attn_weights = pypto.reshape(attn_bf16, [8, 4, 2048], inplace=True)
# → 传给 matmul(attn_weights, v_cache)
```

## 迭代过程

| 尝试 | vec_tile_shapes | 结果 | 原因分析 |
|------|----------------|------|---------|
| 1 | `(32, 512)` | 321.18 us (+16.6% 回退) | 归约轴(2048)被切为 4 块(512)，产生跨子图 reduce 开销 |
| 2 | 未设置 | 编译失败 | reduce op 要求尾轴 32B 对齐，未设 vec_tile_shapes 导致框架无法推导 |
| 3 | `(32, 2048)` | 编译失败 | 32×2048×4×多tensor 超出 UB 容量，`Run pass failed` |
| 4 | `(8, 2048)` + `sg_set_scope` | **258.98 us (-6.0%)** | 归约轴不切分 + 合图减少子图数，任务数 168→137 |

## 收益

- 执行时间：275.44 → 258.98 us（-6.0%）
- 任务数：168 → 137（-18.5%）
- 子图数：6 个独立 vector 子图 → 1 个合图子图
- 精度：Max difference 0.000031，无变化

## 关键经验

1. **归约轴必须不切分**：vec_tile_shapes 第二维应等于实际归约轴长度，切分会产生跨子图 reduce 开销
2. **UB 容量决定第一维上限**：第一维 × 第二维 × dtype × tensor 总数 不能超出 UB（约 128KB 保守估计），否则编译失败
3. **必须显式设 vec_tile_shapes**：合轴后维度变化，不设会导致 reduce op 编译失败（32B 对齐约束）
4. **sg_set_scope 合图**：将连续 vector 操作合并为单个子图，减少调度开销

## 常见失败模式

| 报错信息 | 原因 | 修复方法 |
|---------|------|---------|
| `Reduce op: the tileShape of last axis need to 32Byte align!` | 未设 vec_tile_shapes 或尾轴非 32B 对齐 | 显式设置 `set_vec_tile_shapes(M, N)`，FP32 下 N 为 8 的倍数 |
| `Run pass failed` | tile 数据量超出 UB | 减小第一维：`第一维 × 第二维 × dtype字节数 × 3 ≤ 128KB` |
| 性能回退 | 归约轴被 vec_tile_shapes 切分 | 第二维 = 实际归约轴长度，不切分 |
