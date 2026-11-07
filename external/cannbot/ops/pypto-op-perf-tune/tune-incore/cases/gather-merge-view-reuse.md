# 案例：合并 gather + view 复用消除 DDR 往返

## 场景

sparse_flash_attention_quant（BF16，A5 平台 `DAV_3510`）。Decode 场景，算子含 Q@K^T → softmax → P@V 的 CV 交替结构。前端+泳道图调优后 AICore E2E Time = 1019us，S-14 Mix合图配置后性能无收益。

经 §4.7 Mix合图失败诊断子流程分析，CV 通路存在 2 个 DDR 断点（独立 gather 打断 CV 通路连续性），须执行 I-10（合并 gather）+ I-11（view 复用）修复后重试 S-14。

## 诊断

### 原始实现（3 次独立 gather）

```python
# V0 阶段：分两次 gather 取 key 的不同列段
kn = pypto.gather_in_ub(key_nope_2d, indices, block_table, block_size, -2)  # 512 维，DDR→UB
kr = pypto.gather_in_ub(key_rope_2d, indices, block_table, block_size, -2)  # 64 维，DDR→UB

# C2 阶段：第三次 gather 取 vj（vj 与 kn 内容完全重叠）
vj = pypto.gather_in_ub(key_nope_2d, indices, block_table, block_size, -2)  # 512 维，DDR→UB（重复搬运）
```

### DDR 断点分析（program.json 解析）

| 断点 | 位置 | 根因 | 分类 | 修复模式 |
|------|------|------|------|---------|
| 1 | V0 阶段 kn + kr 两次独立 gather | CV 序列中间插入独立 gather，打断通路连续性 | 可修复 | I-10 合并 gather |
| 2 | C2 阶段 vj 独立 gather | vj 与已 gather 的 kn 内容完全重叠，重复搬运 | 可修复 | I-11 view 复用 |

无不可修复断点（matmul 输出 FP32 是 V1 softmax 的输入，可走 CV 通路）。

## 修复

### I-10：合并 gather（V0 阶段）

```python
# ✅ 入口拼接后一次 gather
key_2d = pypto.concat([key_nope_2d, key_rope_2d], -1)  # 576 维，GM 中完成
kj = pypto.gather_in_ub(key_2d, indices, block_table, block_size, -2)  # 一次 DDR→UB
kn = pypto.view(kj, [s2_tile, dn], [0, 0])   # UB 内 view 切出 kn（512 维）
kr = pypto.view(kj, [s2_tile, dr], [0, dn])  # UB 内 view 切出 kr（64 维）
```

### I-11：view 复用（C2 阶段）

```python
# ✅ view 复用已有 kn，零搬运
vj = pypto.view(kn, [s2_tile, dn], [0, 0])  # vj 与 kn 同源同段，直接 view
```

## 迭代过程

| 步骤 | 操作 | AICore E2E Time | CV 通路状态 | 说明 |
|------|------|----------------|------------|------|
| 0 | S-14 Mix合图（原始 3 次 gather） | 1019us | 断点 2 处 | Mix合图无收益 |
| 1 | I-10 合并 kn+kr gather | — | 断点 1 修复 | 精度 PASS |
| 2 | I-11 view 复用 vj | — | 断点 2 修复 | 精度 PASS |
| 3 | 重新评估 S-14 Mix合图（scope=5001） | **611us** | 全部闭合 | Mix合图生效 |

## 收益

- AICore E2E Time：1019 → 611 us（**-40%**）
- DDR→UB 往返：3 次降至 1 次
- CV 通路：2 个断点全部修复，Mix合图生效

## 关键经验

1. **独立 gather 是 CV 通路断点的常见根因**：Mix合图要求 CV 序列连续，中间插入的独立 gather 会迫使数据走 DDR 中转，打断通路
2. **I-10 和 I-11 须配合使用**：I-10 合并同源 gather 消除断点 1，I-11 复用已有数据消除断点 2，两者协同才能使 CV 通路全部闭合
3. **修复后必须重新评估 S-14**：I-10/I-11 修复 CV 断点后，原先不生效的 Mix合图可能变为生效，须重新配置 scope 并验证
4. **拼接须用 `pypto.concat`**：kernel 函数内不能使用 `torch.cat`（host 侧 API），须用 `pypto.concat` 在 GM 中完成拼接
5. **gather 结果受 UB 248KB 限制**：拼接后单次 gather 的结果（ND+NZ）须 < 248KB，否则数据走 DDR 中转，优化失效

## 常见失败模式

| 现象 | 原因 | 修复方法 |
|------|------|---------|
| 拼接后 gather 结果走 DDR | 单 tensor ND+NZ > 248KB | 减小 gather tile_shape |
| view 切出数据布局与下游不匹配 | NZ/ND 格式差异 | 检查 view 后 tensor 布局，必要时 reshape |
| vj 复用 kn 后精度异常 | kn 生命周期已结束被覆盖 | 在 kn 被覆盖前复制或延长生命周期 |
| I-10/I-11 修复后 Mix合图仍不生效 | 存在其他不可修复断点 | 重新走 §4.7 诊断流程，排查 UB 超限/M:N 依赖等架构约束 |
