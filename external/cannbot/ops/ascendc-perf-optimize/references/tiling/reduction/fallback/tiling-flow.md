# Reduction 族 — 通用 Tiling 流程

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

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
| **多轴交错** | 合轴后仍为多个交替 A/R（A-R-A-R…，不可合成单个 R），如 8 轴 ARARARAR / 7 轴 ARARARA |
| **交错连续载入** | 交错布局专用模板（tail-R fast / tail-A fast-RARA），按内存物理连续顺序做大 burst 载入，把 A/R 分离从 HBM 碎读搬到 UB 内链式规约 |
| **R 轴分核 group（BlockSplitR）** | 大 R + 小 A 时**跨核切归约轴 R**：各核算一段 R 的部分和写 workspace，SyncAll 后 Phase2 合并。与"切 A"的五模板/normal 并列，见 [§3.2](#32-r-轴分核-groupblocksplitr-大-r--小-a-的一等分支) |

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

**合轴后可能仍是交错序列。** 当保留轴与归约轴在内存上交替出现（如 shape `[4,2,2,640,32,32,2,2]` axes=[1,3,5,7]，标记后为 `A-R-A-R-A-R-A-R`），相邻同类型轴合不掉、size=1 也去不掉，最终 **无法压成单个 R**。此时不能直接进五模板（五模板假设合轴后是简单 AR/ARA，单个 R）——需先过 [§3.1 多轴交错前置判定](#31-多轴交错前置判定step-05)。判定用 `axisNum`（严格交替后的轴数）与 `isTailR`（轴数为偶=尾轴是 R）区分。

---

## 3.1 多轴交错前置判定（Step 0.5）

> 位于合轴之后、五模板决策树之前。仅处理 **不可合成单 R 的交错布局**；不命中则原样落入五模板。

**背景问题**：交错 case 若走五模板，CopyIn 会边搬边转置把散落的 R 聚拢，导致最内 burst 极小（如 #18 内块最内 R 长度=2 → burst **8B**）、MTE2 打满（ratio≈0.995）、有效带宽仅 ~145GB/s。

**核心洞察**：平方和（及任何可交换归约）**不要求按 R 逻辑顺序累加**，可按内存**物理连续顺序**做大 burst 载入（16–64KB），把 A/R 分离从 HBM 碎读搬到 UB 内做**链式规约**。arch35 硬件 `ReduceSum` 只支持 AR/RA 两种排布，故多轴交错在 UB 内用**两步 AR/RA 链式分解**完成。

```
输入: 严格交替后的 axisShape[], axisNum, isTailR, dtype, axisStride[], coreNum, ubSize

┌─ 不是纯交错病态？（任一不满足 → 回退五模板）
│    dtype ∉ {fp32, fp16, bf16}             → 五模板  // int32 平方需精确整数，不走 fast
│    axisNum < 6                            → 五模板
│    内块（末 4 轴）非物理连续              → 五模板
│         校验: 最内轴 stride==1
│                内块前的分块轴 stride==内块元素数
│    内块整块字节非 32B 对齐                → 五模板  // kernel 连续 DataCopy 载入要求
│
├─ axisNum 偶数且 isTailR（ARAR...AR，尾轴 R，S=axisNum-4）
│     → 【交错-tail-R / fast】两段式，见 §11.1
│       附加门控（全满足才命中）:
│         R1(最内 R)×dtypeSize ≤ 16B        // ★ burst 病态判据（连续化收益区）
│         rowsSize ≥ 8                      // ★ rows 太小则两步规约+屏障反超增益
│         chunkA0 ≥ 1 且 chunkA0 == szA0   // 内块 A0 无需分块（需分块实测负收益 #17 -9%）
│         groupCnt = ∏(外层 A 轴) ≤ coreNum
│       否则 → 回退五模板（如 #17 内块需分块，chunkA0=14<32）
│
├─ axisNum 奇数且 !isTailR（ARAR...A，尾轴 A，S=axisNum-4）
│     → 【交错-tail-A / fast-RARA】无 workspace，见 §11.2
│       附加门控（全满足才命中）:
│         slabCnt = ∏(内块外 A 轴) ≥ 1
│         slabCnt × 2 ≥ coreNum            // slab 太少并行度不足（#9 slabCnt=8 判负）
│         innerRowElems ≥ 512              // ★ 内块每行太小则连续载入无收益（#13/#15 回退）
│       否则 → 回退五模板/group
│
└─ 其余交错形态（其它轴数 / dtype 不符 / 内块不连续 / 非 32B 对齐）
      → 回退五模板（normal/group）
```

> ⚠️ **判据不是 `mte2_ratio`**。真正决定赢面的是 **基线最内 burst 大小**（tail-R：内块最内 R 长度）与 **内块每行大小**（tail-A：innerRowElems）。详见 [§13 交错模板门控判据与 mte2 教训](#13-交错模板门控判据与-mte2-会误导的教训)。

---

## 3. 五模板决策树（Step 1–2）

> 前置：先过 [§3.1 多轴交错前置判定](#31-多轴交错前置判定step-05)。命中交错模板则直接返回；未命中（合轴后是简单 AR/ARA，单个 R）才进入本节。

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

## 3.2 R 轴分核 group（BlockSplitR）—— 大 R + 小 A 的一等分支

> **一等分支，不是"可选增强"。** 五模板/交错模板**都在切 A（行/输出维），不切 R**。但当 **R 大到单核放不下
> 且 A 维乘积 < core_num（单纯切 A 只能用满 A 个核，其余闲置）** 时，必须把**归约轴 R 本身跨核切分**，各核
> 算一段 R 的部分和再合并，才能用满核。这就是 **R 轴分核 group（BlockSplitR，等价 batch_norm 的 BlockSplitR）**。
> 参考实现见 EuclideanNorm host tiling `ShouldUseGroup` / `ComputeGroupSplit`；实践见
> [example/euclidean_norm/experience.md](example/euclidean_norm/experience.md)（20 case 里 **#1–9 共 9 个走 group**，最主力路径）。

**位置**：合轴 → 交错前置判定（不命中）→ 五模板选出 UB 切分（含 R 是否分块）→ **本判定**（决定 A 轴分核 normal
还是 R 轴分核 group）。

```
输入: aLoopCntTotal（A 维扁平计数）, rLoopCntTotal（R 分块后的外层 R loop 计数）, coreNum

┌─ aLoopCntTotal > coreNum / 2      → 【A 轴分核 normal】   // 切 A 已够满核，无需跨核切 R
├─ rLoopCntTotal <= 1               → 【A 轴分核 normal】   // R 未分块（R 全载 UB），无跨核 R 可切
└─ 否则（大 R + 小 A）             → 【R 轴分核 group / BlockSplitR】
      Phase1 各核部分和 → workspace → SyncAll → Phase2 合并 + 输出
```

**两段式结构（可交换归约才成立：平方和/求和/…）**：

```
分核网格: 2D [aLoopCntTotal, rGroupCnt]，usedCoreNum = aLoopCntTotal × rGroupCnt
Phase1（每核）: 处理分到本核的一段 R chunk → 局部部分和 → 写 workspace[rChunkIdx, 该A列]
SyncAll()                          # 跨核屏障，等所有核 workspace 写完
Phase2（RA mini-kernel）: workspace[rGroupCnt, aTotal] 沿 rGroupCnt 维 ReduceSum 合并 → sqrt → 输出
```

### UB 预算 / 多核公式（从 `ShouldUseGroup` / `ComputeGroupSplit` 提炼）

```
前置（五模板已算好）:
  aLoopCntTotal = ∏(外层 A 整根) × CeilDiv(A_split 轴, aUbFactor)   // A 维扁平计数
  rLoopCntTotal = ∏(外层 R 整根) × CeilDiv(R_split 轴, rUbFactor)   // R 分块后的外层 R loop 计数
  aTotal        = ∏(所有 A 轴 size)                                 // 输出元素数 = workspace 列数

门控（两条都不满足才用 group）:
  ShouldUseGroup = (aLoopCntTotal <= coreNum/2) AND (rLoopCntTotal > 1)

R 分核多核（ComputeGroupSplit）:
  totalOuter = aLoopCntTotal × rLoopCntTotal
  perCoreNum = CeilDiv(totalOuter, coreNum)
  numBlocks  = CeilDiv(totalOuter, perCoreNum)
  numBlocks  = (CeilAlign(numBlocks, aLoopCntTotal) <= coreNum)      // 对齐到 A 维整数倍
               ? CeilAlign(numBlocks, aLoopCntTotal)                 //   （每 A 列 rGroupCnt 相等，aPerCore=1）
               : FloorAlign(numBlocks, aLoopCntTotal)
  usedCoreNum = numBlocks
  rGroupCnt   = numBlocks / aLoopCntTotal                            // 每 A 列切成 rGroupCnt 个 R 分组

workspace:
  workspaceBytes = rGroupCnt × aTotal × 4          // fp32 dense，布局 [rGroupCnt, aTotal]

Phase2（kernel 侧现算，R 全载 rGroupCnt 行进 UB）:
  aUbFactorP2 = floor(preInElems / rGroupCnt) → FloorAlign(8) → clamp(outBuf/tmpBuf/aTotal 三上界)
  ReduceSum<fp32, RA, srcInnerPad=true>  srcShape={rGroupCnt, aUbFactorP2}  // 消解 rGroupCnt，保留 A
```

> **验证锚点**（g++ 纯 host，coreNum=72, UB=192KB）：#1 `[64000,32]` axes[0] → `templateType=1`（group），
> `rGroupCnt=61`、`usedCoreNum=61`、`workspaceBytes=61×32×4=7808`；#6 `[4,3200,24]` axes[1] → `rGroupCnt=21`、
> ws=8064；#8 `[2,320,14,320]` axes[1,3] → `rGroupCnt=2`、ws=224。

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
| **group / BlockSplitR** | **R（跨核切 R）** | 2D 网格 `[aLoopCntTotal, rGroupCnt]`，`usedCoreNum = aLoopCntTotal × rGroupCnt`，见 [§3.2](#32-r-轴分核-groupblocksplitr-大-r--小-a-的一等分支) |

`used_core_num = ceil(total_work / per_core_work)`，保证 ≥ 1。

---

## 6. 可选增强（Step 4）

在基础五模板之上，按算子特性叠加：

| 增强 | 触发条件 | 作用 |
|------|---------|------|
| **Welford Online** | op_type 为 var/std/norm 且 Recompute 模式 | 分块流式计算 mean + var |
| **二分累加** | Sum 精度敏感 | 块间累加用树形而非线性 |
| **With-Index** | ArgMax / ArgMin | 归约时同步跟踪极值位置 |

> ⚠️ **`Group Reduce`（R 轴分核 / BlockSplitR）已从"可选增强"提升为一等分支**，见
> [§3.2 R 轴分核 group（BlockSplitR）](#32-r-轴分核-groupblocksplitr-大-r--小-a-的一等分支)。它不是叠加在单模板上的小优化，
> 而是与"A 轴分核 normal"并列的多核策略选择（大 R + 小 A 时跨核切 R + workspace 合并 + SyncAll + Phase2）。

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

---

## 11. 多轴交错连续载入模板（细节）

> 对应 [§3.1 前置判定](#31-多轴交错前置判定step-05) 命中的两条路径。参考实现见 EuclideanNorm host tiling
> `euclidean_norm_tiling_host.h`（`TryComputeFastTiling` / `TryComputeFastRARATiling`），
> 实践案例见 [example/euclidean_norm/experience.md](example/euclidean_norm/experience.md)。

**共同思想**：把交错张量的**末 4 轴当作"内块"**（tail-R 为 `[A0,R0,A1,R1]`，tail-A 为 `[R0,A0,R1,A1]`），沿内存物理连续方向做大 burst `DataCopy`（不转置、不碎读），载入后在 UB 内用**两步 AR/RA 链式规约**把 A/R 分离。外层 A 轴（内块之外）用于分核。

### 11.1 交错-tail-R（fast，两段式）

适用 `axisNum` 偶数≥6 严格 `ARAR...AR`（`isTailR==true`）。因外层 A 轴少（groupCnt≤coreNum），需多核共同切分 `rows` 轴（内块前的 R 轴，索引 S-1），故采用**两段式 + workspace**（`coresPerGroup==1` 时免 workspace 直出）：

```
布局: [...外层 | A0,R0,A1,R1]   内块=末 4 轴(S..N-1)，S=axisNum-4
分核: group = ∏(外层A轴)  ;  同 group 内 coresPerGroup 核切 rows 轴(S-1)

Phase 1（每核）:
  沿 rows 轴分到本核的区段，连续 DataCopy 载入内块切片（burst=sliceStride×dtypeSize）
  (Cast 升 fp32) VF 平方累加 → 内块两步规约:
    step1  沿最内 R1 规约（AR）            → 每 [A0,R0,A1] 面得部分平方和
    step2  逐 (a0,a1) 标量累加 R0          → 每 [A0,A1]=outLanes 得部分和（规避 adv_api AR 边界高估）
  本核部分和写 workspace
if coresPerGroup == 1:  Sqrt(part) → (Cast→D_T) → yGm   // 免两段式同步
Phase 2（coresPerGroup>1，SyncAll 后）:
  跨核合并同一 group 的部分和 → sqrt → (Cast→D_T) → 输出

workspace 字节 = groupCnt × coresPerGroup × outLanes × 4   （fp32 部分和；outLanes = A0 × A1）
```

### 11.2 交错-tail-A（fast-RARA，无 workspace）

适用 `axisNum` 奇数≥7 严格 `ARAR...A`（`isTailR==false`）。外层 A 轴（slab）足够多（slabCnt×2≥coreNum），**每核独占若干 slab、无跨核切 R**，故**无 workspace、无 Phase2 屏障**：

```
布局: [...外层 | R0,A0,R1,A1]   内块=末 4 轴(S..N-1)，S=axisNum-4
分核: slab = ∏(外层A轴)  ;  每核独占 min(slabCnt,coreNum) 分配后的若干 slab

每 slab（单核内）:
  step1  沿内块最外 R0 连续载入 + (Cast 升 fp32) 平方累加（burst=innerRowElems×dtypeSize，无 gather）
         → acc[innerRowElems] = Σ_R0 x²，innerRowElems = A0×R1×A1
  step2  逐 A 平面沿 R1 做 RA 规约 → 每 [A0,A1]=outLanes 得和 → sqrt → (Cast→D_T) → 输出
```

**tail-R vs tail-A 对比**：

| 维度 | tail-R（fast） | tail-A（fast-RARA） |
|------|---------------|---------------------|
| 轴数 / 尾轴 | 偶数≥6 / R | 奇数≥7 / A |
| dtype | fp32/fp16/bf16 | fp32/fp16/bf16 |
| 内块 | `[A0,R0,A1,R1]`（末 4 轴 S..N-1） | `[R0,A0,R1,A1]`（末 4 轴 S..N-1） |
| 分核对象 | rows 轴（S-1），group 内多核切 | slab（∏外层A轴），每核独占 |
| workspace | 需要（coresPerGroup>1）；coresPerGroup==1 免 | **无** |
| Phase2 / SyncAll | 需要（coresPerGroup>1） | 无 |
| 内块规约 | step1 规约 R1 → step2 逐(a0,a1)标量累加 R0 | step1 连续载入累加 R0 → step2 逐平面 RA 规约 R1 |
| 实测代表 | #18 **8.29×**（1128→136µs，burst 8B，fp32） | #11 **1.39×**（171→123µs，内块 17KB，fp32） |

---

## 12. 交错模板 UB 预算公式

> 常量取自 host tiling：`tmp`(共享 Reduce 临时区)=16KB，`DB`(double buffer)=2，`dtypeSize`=4(fp32)/2(fp16,bf16)。acc/red/tmp/part 恒为 fp32；inQ/outQ 用 D_T 精度。

### 12.1 tail-R（fast）—— 求内块分块 chunkA0（`TryComputeFastTiling`）

```
记内块 [A0,R0,A1,R1]（轴 S..N-1，S=axisNum-4）:
  sliceStride = R0 × A1 × R1           // 一个 A0 面的元素数
  outLanes    = A0 × A1                // 输出通道数
  innerElems  = A0 × R0 × A1 × R1      // 内块总元素（连续性校验 axisStride[S-1]==innerElems）

固定开销（与 chunkA0 无关）:
  fixedIndep = 16384 + CeilAlign(outLanes, 8) × 4 × 2   // tmp(16K) + part + outQ(fp32)

与 chunkA0 成比例的每单位字节:
  perA0 = 2×sliceStride×dtypeSize   (inQ, DB=2, D_T)
        + sliceStride×4             (acc, fp32)
        + R0×A1×4                   (step1 中间 red, fp32)
        + R0×4                      (step2 plane，保守预留)
        + 4                         (dstA4，保守预留)

chunkA0 = (ubSize - fixedIndep) / perA0，clamp 到 [1, A0]

★ 命中门控（全满足才用，否则回退五模板）:
    innerElems×dtypeSize % 32 == 0      // 连续 DataCopy 32B 对齐
    R1×dtypeSize ≤ 16                   // ★ burst 病态判据
    rowsSize ≥ 8                        // ★ rows 太小则两步规约+屏障反超增益
    chunkA0 == A0                       // 内块整块放得下（无需分块）
    groupCnt = ∏(外层A轴) ≤ coreNum
   余量提 rowsPerLoad（增大 burst）: rowsPerLoad += 剩余UB / (2×chunkElems×dtypeSize)，clamp [1, min(8, rows轴长)]
```

### 12.2 tail-A（fast-RARA）—— 求内块分块 chunkR0（`TryComputeFastRARATiling`）

```
记内块 [R0,A0,R1,A1]（轴 S..N-1，S=axisNum-4）:
  innerRowElems = A0 × R1 × A1        // 固定 R0 的一行（连续性校验 axisStride[S]==innerRowElems）
  outLanes      = A0 × A1

固定开销:
  fixedBytes = innerRowElems×4         (acc, fp32)
             + 16384                   (tmp)
             + R1×A1×4                 (plane)
             + CeilAlign(outLanes,8)×4×2   (part + outQ, part fp32 / outQ D_T)

chunkR0 = (ubSize - fixedBytes) / (2 × innerRowElems × dtypeSize)   // DB=2，clamp 到 [1, min(R0, 8)]

★ 命中门控: innerRowElems×dtypeSize % 32 == 0  (32B 对齐)
            slabCnt = ∏(外层A轴) ; slabCnt×2 ≥ coreNum ; innerRowElems ≥ 512
   usedCoreNum = min(slabCnt, coreNum)  ; workspace = 0
```

### 12.3 与五模板 UB 预算的差异

| 项 | 五模板 | 交错模板 |
|----|--------|---------|
| 分块对象 | R（FullLoad/Recompute） | 内块 A0（tail-R）/ 内块 R0（tail-A） |
| 载入方向 | 转置聚 R | 内存物理连续，大 burst（连续 `DataCopy`，非 `DataCopyPad`） |
| dtype | 任意 | fp32/fp16/bf16（载入即升 fp32，int32 不走 fast） |
| 共享临时区 | tmp_buf_size ≥ 4096 | 固定 16KB（`kCacheBufBytes`） |
| 跨核合并 | 二分累加 / Group workspace | tail-R workspace 两段式（coresPerGroup==1 免）；tail-A 无 |

---

## 13. 交错模板门控判据与 mte2 会误导的教训

> 全部经 arch35 上板实测校准（EuclideanNorm 20 case，见
> [example/euclidean_norm/experience.md](example/euclidean_norm/experience.md) §5）。

**核心判据：`mte2_ratio` 单指标会误导，不能作为选型依据。** 真正决定赢面的是：

| 路径 | 真实判据 | 阈值 | 依据 case |
|------|---------|------|----------|
| tail-R（fast） | 基线最内 burst 大小（=内块最内 R 长度×dtypeSize）+ rows 轴规模 | burst **≤16B** 且 **rowsSize≥8** 才大赢 | #18 burst 8B, rows=640 → **8.29×** |
| tail-A（fast-RARA） | 内块每行大小 innerRowElems | **≥ 512 元素（≥1KB fp16 / ≥2KB fp32）** 才值得 | #11 4320 元素 → 1.39×；#12 3840 → 微赢 |

> 两条路径还要求 `innerElems×dtypeSize`（tail-R）/ `innerRowElems×dtypeSize`（tail-A）32B 对齐（kernel 连续 `DataCopy` 载入要求），非对齐直接回退。

**反例（必须回退五模板）**：

| case | 症状 | 现象 | gate 动作 |
|------|------|------|----------|
| #17 arararar_0001 | 内块最内 R=16 → burst **64B**（中等），内块需分块 chunkA0=14<32 | fast 分块屏障+逐平面规约吃掉增益，实测 **-9%** | `R1×dtypeSize≤16` / `chunkA0==A0` 不满足 → 回退 normal |
| #13 ararara_0006 | innerRowElems=80（**320B**/行） | 误命中 RARA 慢 **28%**（92.6 vs 75.2µs） | `innerRowElems ≥ 512` 不满足 → 回退 group |
| #15 ararara_0008 | innerRowElems=8（**32B**/行），slabCnt=18000 | 误命中 RARA 慢 **10×**（229 vs 22.8µs），大量 slab 的 step2 屏障+小规约开销拖垮 | 同上 → 回退 normal |

**教训**：
1. 中等 burst（64B）或小内块（≤320B）时，模板的固定开销（分块屏障、逐平面规约、每 slab 的 step2 屏障+小规约）反被基线反超。
2. `innerRowElems ≥ 512` 阈值是 **上板抓出 #13/#15 退化后新增的**（初版 RARA gate 太宽，只判 slabCnt，误命中）。
3. 交错模板是"精准打击"型优化：20 case 里仅 #11/#12/#18 真正获益，其余 17 个必须干净回退，零退化。**任一门控不满足即 `return false` 落回五模板/group，回退安全。**

---
