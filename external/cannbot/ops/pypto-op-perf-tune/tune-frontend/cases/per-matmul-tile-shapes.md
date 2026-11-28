# 案例：多 Matmul 独立 TileShape 优化

## 场景

Decode Attention 算子中有 3 个 matmul，shape 特征各不相同。原来用统一的 `set_cube_tile_shapes([16, 256], [128, 256], [256, 256])` 应用于所有 matmul，导致部分 matmul 的 L1 tile 超过实际轴长，浪费 L1 空间。

## 核心原则

1. **L1 不超过实际轴长**：如果 K=128，则 kL1 设 256 无意义（实际数据只有 128），反而浪费 L1 空间
2. **小轴不切**：轴值较小时 L0=L1=实际值
3. **大轴大 tile**：轴值大时用大 L1 减少任务数
4. **每个 matmul 前独立设置**：不同 shape 的 matmul 最优 tile 不同

## 三个 Matmul 的 Shape 分析

| matmul | M | K | N | 特点 |
|--------|---|---|---|------|
| Q@K^T | 4 | 128 | 2048 | K 小，N 大 |
| attn@V | 4 | 2048 | 128 | K 大，N 小 |
| output_proj | 1 | 4096 | 4096 | K/N 都大 |

## 代码对比

### 优化前（统一 tile）

```python
pypto.set_cube_tile_shapes([16, 256], [128, 256], [256, 256])
scores_fp32 = pypto.matmul(q_grouped, k_cache, pypto.DT_FP32, b_trans=True)
# ... vector ops ...
attn_output = pypto.matmul(attn_weights, v_cache, pypto.DT_BF16)
# ...
result = pypto.matmul(attn_output_flat, o_weight, pypto.DT_BF16)
```

问题：
- Q@K^T：K=128，但 kL1=256 浪费 L1
- attn@V：N=128，但 nL1=256 浪费 L1
- 所有 matmul 用同一配置，无法按各自特征优化

### 优化后（独立 tile）

```python
# Q@K^T: K=128小不切，N=2048大tile
pypto.set_cube_tile_shapes([16, 16], [128, 128], [256, 256])
scores_fp32 = pypto.matmul(q_grouped, k_cache, pypto.DT_FP32, b_trans=True)

# ... vector ops ...

# attn@V: K=2048大tile，N=128小不切
pypto.set_cube_tile_shapes([16, 16], [256, 256], [128, 128])
attn_output = pypto.matmul(attn_weights, v_cache, pypto.DT_BF16)

# output_proj: K/N都大，均用大tile
pypto.set_cube_tile_shapes([16, 16], [128, 256], [256, 256])
result = pypto.matmul(attn_output_flat, o_weight, pypto.DT_BF16)
```

## 迭代过程

| 尝试 | 配置方式 | 结果 | 原因分析 |
|------|---------|------|---------|
| 1 | 统一 `[16,256],[128,256],[256,256]` | 257.12 us（基线） | 大轴浪费 L1，小轴 tile 不匹配 |
| 2 | 分设但 mL1=256 | 275.08 us（+7% 回退） | M 极小(4/1)时 mL1=256 浪费 L1，挤占 K/N 空间 |
| 3 | 分设 mL1=16 | **237.12 us（-7.8%）** | M 极小不浪费 L1，空间让给 K/N 大轴 |

## 收益

- 执行时间：257.12 → 237.12 us（-7.8%）
- 累计（含前置优化）：439.54 → 237.12 us（-46.1%）
- 精度：Max difference 0.000031，无变化

## 关键经验

1. **L1 不超实际轴长**：K=128 时 kL1=256 没有意义，实际只有 128 个元素
2. **M 极小时 mL1 也要小**：M=4 或 M=1 时 mL1=256 浪费 L1，挤占 K/N 的 L1 空间
3. **分设时要先设统一值确认编译通过**：先每个 matmul 前都设成同一个值，确认编译通过后再分别调整
4. **注意 L0/L1 约束**：`kL0 <= kL1 && kL1 % kL0 == 0`，违反会编译失败
