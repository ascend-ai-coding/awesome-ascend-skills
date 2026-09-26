# MatMul 族 — 变体差异说明

> MatMul 族各变体与通用 Tiling 流程（见 [tiling-flow.md](tiling-flow.md)）的差异对照。
>
> 所有变体共享同一套 SWAT → FullLoad → StreamK 推导链，仅在初始参数、Cube 粒度、数据布局和特有约束上有差异。

---

## 1. 变体总览

| 维度 | a16w16 | mxfp4/8 | batch_matmul | group_matmul |
|------|--------|---------|-------------|-------------|
| dtype (A/B) | FP16 / BF16 | FP4(0.5B) / FP8(1B) | 随子变体 | 随子变体 |
| Scale | 无 | UE8(1B), per-group=32 | 随子变体 | 随子变体 |
| 特有维度 | — | — | batch (B) | group (g), M_i 不等 |
| baseK | 16 | 32 | 随子变体 | 随子变体 |

---

## 2. a16w16（基线变体）

**差异**：无。a16w16 是通用 Tiling 流程的标准实现，SWAT 七步推导完全适用。

- A、B：FP16 / BF16，2B/element
- Cube 粒度：m0=16, k0=16, n0=16
- SWAT 机制 A（BLOCK_TABLE 负载均衡）为 a16w16 专属
- TilingData 字段名：`l0cDB`（其他变体 `dbL0C`，语义一致）
- stepKa/stepKb 不下发给 Kernel（Kernel 用 kL1/baseK 还原）

---

## 3. mxfp4/8（低精度 + Scale 变体）

MX 量化变体。mxfp4 与 mxfp8 共享同一套 Tiling 流程，仅 dtype 字节宽度和 baseK 上限不同。

| 项目 | mxfp4 | mxfp8 |
|------|-------|-------|
| dtype (A/B) | FP4 (0.5B) | FP8 (1B) |
| Scale | UE8 (1B), per-group=32 | 同左 |
| baseK 上限 | 256 | 128 |

### 3.1 Scale 维度建模

Scale 是 MX 量化的元数据：每个 group（32 个 K 元素）配一个 scale 值用于反量化。Scale 数据量约为计算数据的 1/32，但小数据 MTE2 搬移效率低，需要特殊处理。

**Scale 几何**：

```
scale_per_K = CeilDiv(K, MX_GROUP_SIZE(=32))
total_scaleA = baseM × scale_per_K × sizeof(UE8)
total_scaleB = baseN × scale_per_K × sizeof(UE8)
```

**Scale 复用因子**（scaleFactorA / scaleFactorB）：控制单次 Scale 搬移量，防止小数据占比过高。受 `SCALE_FACTOR_MAX=127` 和 `MTE2_MIN_LOAD_SIZE=32KB` 约束。

**TilingData 影响**：scaleKL1 覆盖的 K 范围需与 kL1 对齐；全载路径需为 Scale 预留 L1 空间。

### 3.2 字段差量（相对 a16w16）

| 字段 | a16w16 | mxfp4/8 |
|------|--------|---------|
| `stepKa`, `stepKb` | 不下发 Kernel | **下发**（Scale 因子推导依赖） |
| `scaleFactorA`, `scaleFactorB` | **不存在** | Scale 复用因子 |
| `scaleKL1` | **不存在** | Scale 覆盖 K 范围 |
| `dbL0c` | 字段名 `l0cDB` | 字段名 `dbL0C` |
| baseK | 16（FP16） | 32（与 INT8 相同粒度） |

---

## 4. batch_matmul（批处理变体）

(B, M, K) × (B, K, N) → (B, M, N)。多核切分在 B × M × N 三维上进行。batch 维度天然适合并行 —— 不同 batch 分给不同核，无跨核同步。

### 4.1 多核切分（三维决策树）

```
totalTiles = B × mCnt × nCnt

决策树:
  ├─ totalTiles ≥ aicNum
  │   优先 batch 维分核（batch 独立 → 不需跨核同步）
  │   singleCoreBatch = CeilDiv(B, batchAxisCoreNum)
  │   剩余核继续切 (M, N) 平面
  │
  ├─ totalTiles < aicNum
  │   退化：以单 batch 为粒度复制到所有核
  │   singleCoreBatch = 1，每核处理 (M, N) 部分区块
  │
  └─ B > aicNum
      纯 batch 切分 (M, N 各 1 核内完成)
      usedCoreNum = aicNum，每核负责 CeilDiv(B, aicNum) 个 batch
```

### 4.2 batchAxisCoreNum 启发

若 mCnt×nCnt ≥ aicNum（MN 方向自己能满核），batch 维不参与切核；若 MN tile 很少，batch 维多分配核。

### 4.3 与通用流程的差异

- 多核切分增加 batch 维度因子
- **默认不启用 StreamK**（B ≥ 2 时与 batch 并行语义冲突）
- 其余 K 轴流水与单 matmul 一致

---

## 5. group_matmul（分组变体）

g groups of (M_i, K, N)。各 group 共享 K 和 N，但 M_i 可不相等。常用于 MoE：每个 expert 做矩阵乘，g = expert 数量。

### 5.1 子路由：split-M vs split-N

| 条件 | split-M | split-N |
|------|:------:|:------:|
| M_i 较大（≥ baseM×2），核间负载容易均衡 | ✅ | |
| M_i 很小，N 方向大 | | ✅ |
| A=BF16 无 Scale, B=mxfp 有 Scale（weight-quant） | | ✅ |
| A/B 均为 mxfp（全量对称量化） | ✅ | |

**split-M**：沿累积 M 方向切核，每核负责一段连续 M 区间（可能跨多个 group）。使用 groupListGm（累积 M 索引）定位各 group 起止位置。

**split-N**：沿 N 方向切核，N 切成 mainBlock + tail 三段结构。适用于 M_i 很小但 N 很大的 weight-quant 场景。

### 5.2 字段差异

- kL1 拆分为 kAL1/kBL1（stepKa 可能 ≠ stepKb）
- 不下发 nBufferNum（pingpong 深度由 Kernel 编译期常量控制）
- 尾块由 Kernel 端按 groupListGm 动态推导，不需 Host 下发

### 5.3 与通用流程的差异

- **默认不启用 StreamK**
- SWAT 机制 B 的边缘合并需分组独立执行（各组 M_i 不同）
- 多核切分需平衡各组计算量；split-M 下按累积 M 分配，split-N 下按 N 分配

---

## 6. 字段命名差异

详见 [tiling-fields.md](tiling-fields.md) 的变体注记。

| 语义 | a16w16 | mxfp4/8 | group_matmul |
|------|--------|---------|-------------|
| L0C 双缓冲 | `l0cDB` | `dbL0C` | `dbL0C` |
| K 步进下发 | 不下发 | 下发 `stepKa/stepKb` | 下发 |
| Scale 复用因子 | 不存在 | `scaleFactorA/B` | 按子路径 |
