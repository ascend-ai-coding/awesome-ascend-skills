# Reduction 族 — 通用 Tiling 流程

> 归约类算子的五模板决策树与 UB 预算推导。所有归约算子共享同一套模板语言；
> 不同算子仅在 UB 预算的分母（中间 buffer 数量）上有差异。

---

## 1. 核心概念

| 术语 | 含义 |
|------|------|
| **合轴** | 将原始 shape + axes 合并为 `(A1, R, A0)` 三元组 |
| **A1** | 归约轴 R **之前**所有保留轴的乘积 |
| **R** | 归约轴长度 |
| **A0** | 归约轴 R **之后**所有保留轴的乘积（AR 模式下 A0=1） |
| **模板** | 确定的 UB 切分与多核策略，共五种 |
| **FullLoad** | R 方向数据一次全部载入 UB |
| **Recompute** | R 方向超出 UB 容量，分块处理并在块间合并 |
| **SmallR** | R 极小时通过转置布局优化访存（仅 AR 模式） |

---

## 2. 合轴（Step 0）

```
给定: shape, axes

1. 标记每个维度为 A(保留) 或 R(归约)
2. 消除 size=1 的冗余维
3. 合并相邻同类型轴

输出: (A1, R, A0)
```

多轴归约（如 axes=[1,2]）时，相邻 R 轴合并为一个 R。

---

## 3. 五模板决策树（Step 1–2）

```
输入: (A1, R, A0), dtype, ub_size, core_num

┌─ A0 == 1 ──→ AR 族（最内维归约，等价 2D [A1, R]）
│
│   r_align = R 按 32B 对齐后的长度
│   r_small_threshold = dtype 相关的小 R 阈值（FP32 约 16，FP16 约 32）
│
│   ├─ R ≤ r_small_threshold 且 UB 预算通过
│   │     → 【AR-SmallR】
│   │       转置为 (R, A1) 布局，沿 A1 方向切 tile
│   │       适用: R 极小、A1 较大，转置后可向量化处理
│   │
│   ├─ R 可整段载入 UB（FullLoad 预算 ≥ 1）
│   │     → 【AR-FullLoad】
│   │       R 全量驻留 UB，A1 方向多核切分
│   │       适用: 标准归约、Softmax 等 R 中等的场景
│   │
│   └─ R 超出 UB 容量
│         → 【AR-Recompute】
│           R 方向分块，每块独立归约后在 UB 内合并
│           适用: 大 R 归约；Softmax 等需分轮重读原数据
│
└─ A0 > 1 ──→ ARA 族（非尾轴归约，等价 3D [A1, R, A0]）
    │
    │   a0_tile_unit = 64(FP32) / 128(FP16)  // 向量寄存器宽度对应的元素数
    │
    ├─ R 可整段载入 UB（FullLoad 预算 > 0）
    │     → 【ARA-FullLoad】
    │       沿 A0 切 tile，每个 tile 内 R 行全量驻留
    │       R ≤ 8: 直接多行累加
    │       R > 8: 启用二分累加（BinaryAdd）做行归约
    │
    └─ R 超出 UB 容量
          → 【ARA-Recompute】
            沿 A0 切 tile，R 方向按固定粒度（默认 128 行）分块循环
            跨块用 cache buffer 做二分累加合并
```

### 模板速查

| 模板 | 维度 | R 条件 | 关键策略 |
|------|------|--------|---------|
| AR-SmallR | A0=1 | R 极小 | 转置布局，沿 A1 切 tile |
| AR-FullLoad | A0=1 | R 可全载 | R 驻留 UB，A1 切分 |
| AR-Recompute | A0=1 | R 过大 | R 分块 + 块间合并 |
| ARA-FullLoad | A0>1 | R 可全载 | A0 切 tile，R 全载 |
| ARA-Recompute | A0>1 | R 过大 | A0 切 tile，R 分块循环 |

---

## 4. UB 预算公式（Step 3）

以下公式以 **标准单次归约**（1 份输入 + 1 份输出 + Reduce API 临时区）为基准。
多遍流水算子（Softmax、Norm）需在分母中追加中间 buffer，见第 7 节。

### 4.1 AR-SmallR

```
条件: R ≤ r_small_threshold

r_small_threshold = r_tile_unit × 2
  r_tile_unit = 8 (FP32) / 16 (FP16)

每 tile 的 UB 开销（转置后）:
  per_tile = r_align × (输入×2 + 输出×2 + 中间FP32×2)

a0_tile_unit = 64  // FP32 向量宽度
max_a1_tiles = ub_size / (a0_tile_unit × (per_tile + 4))
a1_tile_count = min(max_a1_tiles, ceil(A1 / a0_tile_unit))
a1_tile_len = a1_tile_count × a0_tile_unit

多核: 按 a1_outer = ceil(A1 / a1_tile_len) 均分到 core_num
```

### 4.2 AR-FullLoad

```
条件: rows_per_ub ≥ 1

r_align = ceil(R, 32B/dtype) × (32B/dtype)

rows_per_ub = (ub_size - 固定预留) / per_row_bytes
  固定预留 = 1024 + 512   // 通用预留 + 二分累加临时区
  per_row_bytes = r_align × (FP32中间量 + 输入×2 + 输出×2)

rows_per_core = ceil(A1 / core_num)
rows_per_ub = min(rows_per_ub, rows_per_core)
```

标准归约的简化版（无多遍中间量）:

```
r_max = (ub_size - tmp_buf - 64) / (2 × dtype_bytes)
FullLoad 条件: R ≤ r_max
```

### 4.3 AR-Recompute

```
条件: R > AR-FullLoad 阈值

可用 UB = ub_size - max_buf - sum_buf - binary_cache
  max_buf = 32, sum_buf = 32, binary_cache = 2048

每元素开销 = 输入×3 + 输出×2 + FP32×1   // 标准归约; Softmax 等需追加
r_chunk = floor(可用UB / 每元素开销)，按 32B 对齐

r_loop_count = ceil(R / r_chunk)
尾块 r_chunk_tail = R - r_chunk × floor(R / r_chunk)

块间合并: 二分累加
  fold_base = 小于 r_loop_count 的最大 2 的幂
  fold_remain = floor(R/r_chunk) - fold_base
```

### 4.4 ARA-FullLoad

```
a0_tile_unit = 64 (FP32) / 128 (FP16)

每 tile UB 开销:
  per_tile = R × (输入×2 + FP32中间量×2 + FP32输出) + 固定 8B

max_a0_inner = ub_size / per_tile / a0_tile_unit
a0_factor_max = ceil(A0 / a0_tile_unit)
total_tiles_max = A1 × a0_factor_max

a0_inner = min(
  ceil(total_tiles_max / core_num),  // 多核均衡
  max_a0_inner,                       // UB 容量
  a0_factor_max                       // A0 维度
)
a0_tile_len = a0_inner × a0_tile_unit

R ≤ 8: 直接逐行累加，无需二分参数
R > 8: 启用 BinaryAdd（将 R 行两两分组做树形归约）
```

### 4.5 ARA-Recompute

```
r_bin_size = 128   // R 方向默认分块粒度

bin_loop_count = ceil(R / r_bin_size)
bin_tail = R - (bin_loop_count - 1) × r_bin_size

cache_layers = log2(bin_loop_count)  // 跨 bin 二分累加的 cache 层数

per_tile = r_bin_size × (输入×2 + FP32) + FP32 × (11 + cache_layers)
max_a0_inner = ub_size / a0_tile_unit / per_tile
// 后续 a0_inner / a0_tile_len / 多核分配同 ARA-FullLoad
```

---

## 5. 多核切分

| 模板 | 切分轴 | 分配方式 |
|------|--------|---------|
| AR-SmallR | A1（转置后的"列"） | `tiles_per_core = ceil(a1_outer / core_num)` |
| AR-FullLoad | A1 | `rows_per_core = ceil(A1 / core_num)` |
| AR-Recompute | A1 | 同 AR-FullLoad |
| ARA-FullLoad / ARA-Recompute | A1 × A0 tile | `total_tiles = A1 × a0_outer`，均分到 core_num |

`used_core_num = ceil(total_work / per_core_work)`，保证 ≥ 1。

---

## 6. 可选增强（Step 4）

在基础五模板之上，按算子特性叠加：

| 增强 | 触发条件 | 作用 |
|------|---------|------|
| **Group Reduce** | R 超出单核容量 且 A 维乘积 < core_num | R 方向跨核分块，workspace 合并 |
| **Welford Online** | op_type 为 var/std/norm 且 Recompute 模式 | 分块流式计算 mean + var |
| **二分累加** | Sum 精度敏感 | 块间累加用树形而非线性 |
| **With-Index** | ArgMax / ArgMin | 归约时同步跟踪极值位置 |

---

## 7. 算子差异：如何修正 UB 预算

通用模板不变，修正的是 **per_row_bytes** 或 **每元素开销** 中的 buffer 项：

| 算子 | 沿 R 轴中间量 | buffer 修正 |
|------|-------------|------------|
| ReduceSum/Max | 无额外中间量 | 标准公式 |
| Softmax | max, exp, sum | 输入×2 + 输出×2 + FP32×2（FullLoad）；Recompute 需重读原数据 |
| LayerNorm | mean, var, normalize | 两个关联统计量 → Recompute 时启用 Welford |
| RMSNorm | mean_sq, normalize | 类似 LayerNorm，少一个 mean 缓冲 |

**Agent 推导步骤**：先确定算子数学流程 → 数清 UB 内同时存在的 buffer 份数 → 代入对应模板的预算公式。

---

## 8. Reduce API 临时区

使用硬件 Reduce 指令时需预留临时 buffer：

```
elements_per_repeat = 256 / dtype_bytes
elements_per_block = 32 / dtype_bytes
repeat_count = ceil(r_align / elements_per_repeat)
tmp_buf_size = ceil(repeat_count / elements_per_block) × elements_per_block × dtype_bytes
tmp_buf_size = max(tmp_buf_size, 4096)
```

---

## 9. 对齐规则

| 场景 | 对齐粒度 | 说明 |
|------|---------|------|
| R 方向 | 32B / dtype_bytes 个元素 | DataCopyPad 要求 |
| A0 切分 | 64(FP32) / 128(FP16) | 向量寄存器宽度 |
| UB Buffer | 32B | 最小搬运单位 |
| tmp_buf_size | ≥ 4096B | Reduce API 下限 |

---

## 10. 有效长度 vs 对齐长度

| 用途 | 用 R（有效长度） | 用 r_align（对齐长度） |
|------|:---:|:---:|
| DataCopyPad 搬运长度 | ✅ | ❌ |
| Reduce API 元素计数 | ✅ | ❌ |
| UB 内行偏移计算 | ❌ | ✅ |
| Buffer 大小分配 | ❌ | ✅ |
