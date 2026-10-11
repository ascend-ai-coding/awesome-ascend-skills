# 流水配平分析框架

> 根据 Bound 类型（R 值）和硬件核利用率约束，确定最优非均匀切分方案。

---

## 通算切分原理

MC² 算子按 matmul 结果矩阵的行（M）切分为多个 tile，通信和计算做掩盖。长短块配平通过非均匀切分（长块 + 短块），使短块与中间长块完全掩盖，流水线总时间最短。

### 长短块排布

TilingData 包含长块数量、长块长度、短块数量、短块长度、短块在前或在后的 flag，控制 kernel 的分块执行：

```
短块在前: 1 个短块（shortMSize） + longBlockCnt 个长块（longMSize）
短块在后: longBlockCnt 个长块（longMSize） + 1 个短块（shortMSize）
```

短块位置由执行顺序 × Bound 类型决定（见下方策略矩阵）。

> 长块和短块使用**各自独立的完整 matmul tiling data**（baseM/baseN/baseK/L1 参数等均不同）。

---

## AIC/AIV 分离架构

MC² 通算融合算子采用 AIC（计算核）+ AIV（向量核）分离架构，AIC 负责 matmul 计算，AIV 负责跨卡通信（AllToAll via UDMA / shmem）。两者通过 CrossCore 事件同步实现通信与计算的流水掩盖。

通信与计算的执行顺序不同，同步方向完全相反：

### 通信后计算（Pattern A，如 alltoall + matmul）

AIV 先行通信，AIC 等待数据就绪后计算。同步方向：**AIV → AIC**。

```
AIV: PutToAllRanks(块0) → SetFlag(0) → PutToAllRanks(块1) → SetFlag(1) → ...
AIC:                         WaitFlag(0) → DoMatmul(块0) → WaitFlag(1) → DoMatmul(块1) → ...
```

| 核类型 | 职责 | 同步机制 |
|--------|------|---------|
| **AIV** | 先通信：将本卡数据 Put 到远端卡 Win 区 | `CrossCoreSetFlag` 通知 AIC 数据已就绪 |
| **AIC** | 后计算：从 Win 区读取远端数据做 matmul | `CrossCoreWaitFlag` 等待 AIV 数据就绪 |

> 首块为 fill（AIV 通信，AIC 空闲等待），尾块为 drain（AIC 计算，AIV 已完成）。

### 计算后通信（Pattern B，如 matmul + alltoall）

AIC 先行计算，AIV 等待计算完成后通信。同步方向：**AIC → AIV**。

```
AIC: DoMatmul(块0) → NotifyComputeComplete(0) → DoMatmul(块1) → NotifyComputeComplete(1) → ...
AIV:                  WaitComputeComplete(0) → BarrierAll → AlltoAll(块0) → WaitComputeComplete(1) → ...
```

| 核类型 | 职责 | 同步机制 |
|--------|------|---------|
| **AIC** | 先计算：本地 matmul，输出到 Win 区 | `NotifyComputeComplete` 通知 AIV 计算已完成 |
| **AIV** | 后通信：从远端卡 Win 区 GET 计算结果 | `WaitComputeComplete` 等待 AIC 计算完成，再 `BarrierAll` 对齐所有 rank 后执行 AllToAll |

> 首块为 fill（AIC 计算，AIV 空闲等待），尾块为 drain（AIV 通信，AIC 已完成）。Pattern B 的 AIV 在每次 AllToAll 前需 `BarrierAll` 确保所有 rank 的 AIV 对齐。

### 流水线三段结构

流水配平的目的是让短块与中间长块完全掩盖，流水线总时间由三段组成：

```
T_total = T_fill + T_steady + T_drain
```

| 阶段 | 含义 | 耗时 |
|------|------|------|
| T_fill | 流水线填充（首块，无重叠） | 首块操作 × M_first |
| T_steady | 稳态重叠（中间块，通信与计算重叠） | max(α, β) × M_steady |
| T_drain | 流水线排空（尾块，无重叠） | 尾块操作 × M_last |

短块放在 fill 端或 drain 端，取决于哪种操作无法被掩盖：
- **计算 Bound**（计算掩盖通信）：首/尾块的**通信**无法被掩盖，短块放在通信不可掩盖的一端，使该端通信耗时最小
- **通信 Bound**（通信掩盖计算）：首/尾块的**计算**无法被掩盖，短块放在计算不可掩盖的一端，使该端计算耗时最小

配平的目标是使短块与相邻长块完全掩盖——即短块的被掩盖操作耗时 ≥ 相邻长块的瓶颈操作耗时。

---

## 短块位置策略矩阵

短块位置由执行顺序 × Bound 类型决定：

### 计算后通信的 MC² 融合算子（Pattern B，如 matmul + alltoall）

**计算 Bound**：计算要掩盖通信，最后一块的通信无法被掩盖，因此**短块放最后（drain 端）**。短块通信耗时要尽可能短，对应的计算短块耗时要掩盖住通信的长块耗时。整体上第 N 块的计算耗时 ≥ 第 N-1 块的通信耗时，才能达到掩盖率最优。

```
短块在后:
  长块1计算 → 长块2计算 → ... → 长块N计算 → 短块计算
     ↕掩盖      ↕掩盖              ↕掩盖       ↕(尾通信无法掩盖)
  (无通信)   长块1通信    ...   长块N-1通信   长块N通信

约束：compute[long_k] ≥ comm[long_k-1]，短块 comm 尽量短
```

**通信 Bound**：通信要掩盖计算，第一块计算无法被掩盖，因此**短块（matmul）在最前面（fill 端）**。短块 matmul 的耗时要尽可能短，通信的短块耗时要掩盖住计算的长块耗时。整体上第 N 块的通信耗时 ≥ 第 N+1 块的计算耗时，才能达到掩盖率最优。

```
短块在前:
  短块计算 → 长块1计算 → 长块2计算 → ... → 长块N计算
     ↕(首计算无法掩盖)  ↕掩盖      ↕掩盖
  短块通信   长块1通信   长块2通信  ...   长块N通信

约束：comm[k] ≥ compute[k+1]，短块 compute 尽量短
```

### 通信后计算的 MC² 融合算子（Pattern A，如 alltoall + matmul）

> 不需要通信的 Local 数据可以提前展开，在首次做通信时可以与 Local matmul 做掩盖。

**计算 Bound**：计算要掩盖通信，并且 matmul 计算不能被通信中断。由于有 Local matmul 的存在，第一次通信可以和 Local matmul 做掩盖。在第一块通算切分 matmul 计算时，一定要保证首块通信完成，因此 **Local matmul 的耗时要掩盖住通信**，才能保证 matmul 的计算连续性。从配平角度，首块的 matmul 耗时（非 local 块，而是通算切分块）≥ 通信第二块的耗时。整体上第 N 块的计算耗时 ≥ 第 N+1 块的通信耗时。**短块在后（drain 端）**。

```
Local matmul ──┐
               ↕掩盖（保证首块通信完成，matmul 连续）
通信块1 ──→ 通信块2 ──→ ... ──→ 通信块N（短块, drain 端）
   ↕掩盖      ↕掩盖
matmul块1   matmul块2  ...    短matmul（尾, 尽量短）

约束：
  1. local_matmul ≥ comm[1]（保证首块通信完成，matmul 连续）
  2. compute[k] ≥ comm[k+1]
```

**通信 Bound**：通信要掩盖计算，由于有 Local matmul 的存在，在首次通信就能启动 matmul。一般而言，通信可以连发不中断，只需**尾块在最后（drain 端）**，并且保证尾块 matmul 的耗时最小，即可完成比较优的掩盖率。注意要保证尾块 matmul 不能太小，要保证一定的 Mac 利用率。

```
通信块1 ──→ 通信块2 ──→ ... ──→ 通信块N（短块, drain 端）
   ↕掩盖      ↕掩盖              ↕
matmul块1   matmul块2  ...    短matmul（最小但保 Mac 利用率）

约束：comm 连发不中断，尾块 matmul 最小化但保 Mac 利用率
```

### 通信和计算相近

鼓励对算子多做切分，让通信和计算充分掩盖。注意切分的最小粒度要保证 Mac 利用率尽可能的高，让算子的端到端整体性能达到最优。

### 策略速查

| 执行顺序 | Bound 类型 | 短块位置 | 配平目标 |
|---------|-----------|---------|---------|
| 计算后通信 | 计算 Bound | 后（drain 端） | compute[N] ≥ comm[N-1]，尾通信尽量短 |
| 计算后通信 | 通信 Bound | 前（fill 端） | comm[N] ≥ compute[N+1]，首计算尽量短 |
| 通信后计算 | 计算 Bound | 后（drain 端） | local_matmul ≥ comm[1]; compute[N] ≥ comm[N+1] |
| 通信后计算 | 通信 Bound | 后（drain 端） | comm 连发不中断，尾 matmul 最小但保 Mac 利用率 |
| 任意 | 完美平衡 | — | 充分切分，最小粒度保 Mac 利用率 |

---

## 配平公式

### 配平条件

使短块与相邻长块完全掩盖——短块的被掩盖操作耗时 ≥ 相邻长块的瓶颈操作耗时：

```
α × M_short ≥ β × M_long    （计算 Bound，短块通信掩盖长块计算）
β × M_short ≥ α × M_long    （通信 Bound，短块计算掩盖长块通信）
```

### 配平初值

联立约束条件 `longBlockCnt × M_long + shortBlockCnt × M_short = M` 和配平条件（取等号）求解。以计算 Bound（短块在后，1 个短块）为例，配平条件为 `α × M_short = β × M_long`：

```
M_long  = α × M / (α × longBlockCnt + β) = M / (longBlockCnt + R)
M_short = β × M / (α × longBlockCnt + β) = R × M / (longBlockCnt + R)
```

> 配平初值仅为满足配平条件的一个解，不代表全局最优。

### 配平可行域

配平条件是 `≥`（短块的被掩盖操作耗时 ≥ 相邻长块的瓶颈操作耗时），因此满足配平条件的解为一个**可行域**而非单一值。以计算 Bound（β > α，配平比 ρ = R）为例：

```
配平条件: α × M_short ≥ β × M_long  →  M_short ≥ R × M_long

约束条件: longBlockCnt × M_long + M_short = M

联立得可行域:
  M_long ≤ M / (longBlockCnt + R)
  M_short = M - longBlockCnt × M_long ≥ R × M_long
```

可行域内所有满足条件的 (M_long, M_short) 组合均为候选解。增大 M_short（减小 M_long）会提高掩盖率但可能增大 drain 开销；减小 M_short（增大 M_long）会降低掩盖率但提升长块 Mac 利用率。需在可行域内扫描所有候选解，按总时间择优。

通信 Bound（α > β，配平比 ρ = 1/R）对称：

```
配平条件: β × M_short ≥ α × M_long  →  M_short ≥ (1/R) × M_long
```

### 16 字节对齐

```
M_long_aligned = floor(M_long / 16) × 16
M_short_aligned = M - longBlockCnt × M_long_aligned
```

若 M_short < 16，则减小 M_long 直到 M_short ≥ 16。

---

## 核利用率修正

### matmul 多核切分机制

matmul 计算按 baseM×baseN 的 block 粒度切分到多核并行执行：

```
mBlockCnt = ceil(longMSize / baseM)           // M 维度 block 数
nBlockCnt = ceil(N / baseN)                   // N 维度 block 数
totalBlocks = mBlockCnt × nBlockCnt
effectiveCores = min(totalBlocks, N_core)     // N_core 为 AIC 核数
utilization = effectiveCores / N_core
```

### 可行区间推导

```
M_long_min = (feasible_mBlk - 1) × baseM + 16
M_long_max = floor((M - 16) / longBlockCnt / 16) × 16
```

当核利用率不足时，向下回退 mBlockCnt，找到最大可行值。

---

## TilingData 依赖

搜索算法依赖 baseM、baseN、usedCoreNum 等 matmul tiling 参数来计算核利用率和搜索范围。这些参数由 host 侧 tiling 引擎生成，**必须从 host 代码打印输出中获取**。

### 必需参数

| 参数 | 来源 | 用途 |
|------|------|------|
| `baseM` | TilingData 打印 | 计算 mBlockCnt = ceil(longMSize / baseM)，推导核利用率 |
| `baseN` | TilingData 打印 | 计算 nBlockCnt = ceil(N / baseN)，推导核利用率 |
| `usedCoreNum` | TilingData 打印 | 确认实际可用核数 N_core |
| `baseK` | TilingData 打印 | 辅助分析 L1 切分合理性 |

### 打印方法

host 代码的 tiling 引擎通常包含 `PrintTilingData` 方法但被注释掉。参考 `quant_matmul_tiling_base.h` 中的实现：

```cpp
// 在 tiling 引擎的 GetTilingData 方法中，取消注释 PrintTilingData 调用：
void GetTilingData(...) {
    ...
    DoOpTiling(tilingData);
    PrintTilingData(tilingData);  // 取消注释，或添加此行
}

void PrintTilingData(const QuantMatmulTilingData& tilingData) const {
    printf("[TilingData] usedCoreNum=%u, m=%u, n=%u, k=%u, baseM=%u, baseN=%u, baseK=%u, stepK=%u, ...\n",
           tilingData.usedCoreNum, tilingData.m, tilingData.n, tilingData.k,
           tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.stepK);
}
```

> ⚠️ **若算子 host 代码未打印 TilingData**，须在**源码副本**中添加打印语句后重新运行，采集 baseM/baseN/usedCoreNum 等参数。不得在无 TilingData 数据的情况下进行搜索——缺少 baseM/baseN 将导致核利用率计算失真，搜索结果不可靠。

> 长块和短块各有独立的 matmul tiling data（不同 M 大小 → 不同 baseM/baseN）。须分别打印长块和短块的 tiling data，用于各自的核利用率评估。

---

## 自适应搜索算法

### 设计原则

搜索算法的目标是在配平可行域内**系统性地遍历所有候选切分方案**，通过理论时间估算排序后，输出 Top-N 候选供 msprof 实测验证。以下四条剪枝规则将搜索空间从 O(M/16) 缩减到可接受范围：

| # | 剪枝规则 | 原理 | 阈值 |
|---|---------|------|------|
| P1 | **尾块下限**：shortMSize ∈ (0, 128) 直接排除 | 尾块过小导致 Mac 利用率极低，计算膨胀抵消掩盖收益 | TAIL_MIN = 128 |
| P2 | **核利用率硬约束**：长块 totalBlocks / N_core < 阈值直接排除 | 用核不满的切分实际性能必然劣化 | 计算 bound ≥ 80%，通信 bound ≥ 50% |
| P3 | **tileCnt 范围**：tc < tcFloor 或 tc > tcCap 直接排除 | 块数过少无流水收益，过多通信膨胀过大 | tcFloor=2, tcCap 由 R 值决定 |
| P4 | **配平条件**：shortMSize < ρ × longMSize 排除（松约束可降级为惩罚） | 短块无法掩盖相邻长块的瓶颈操作 | ρ = max(R, 1/R) |

### 搜索范围推导

搜索的自变量为 `hs`（长块大小 longMSize），范围为 [hs_min, hs_max]，步进 16：

```
// 已知参数（从 TilingData 打印和隔离测试获取）
nBlockCnt  = ceil(N / baseN)
N_core     = usedCoreNum          // 从 TilingData 打印获取
ρ          = max(R, 1/R)          // 配平比

// 1. 由核利用率约束推导 hs 下限
//    要求: ceil(hs / baseM) × nBlockCnt ≥ utilThreshold × N_core
utilThreshold = (R >= 1) ? 0.80 : 0.50    // 计算 bound 硬约束 80%，通信 bound 软约束 50%
minTotalBlocks = ceil(utilThreshold × N_core)
minMBlockCnt   = max(1, ceil(minTotalBlocks / nBlockCnt))
hs_min = max(baseM, (minMBlockCnt - 1) × baseM + 16)   // 保证 mBlockCnt ≥ minMBlockCnt
hs_min = align16(hs_min)                                // 16 对齐

// 2. hs 上限：至少 2 块（流水线需要 fill + steady + drain）
hs_max = floor16(M / 2)

// 3. tileCnt 范围（由 R 值决定）
tcFloor = 2
if R >= 1:   tcCap = min(ceil(R) + 2, 8)       // 计算 bound：块数过多收益递减
else:        tcCap = min(ceil(1/R) + 2, 16)     // 通信 bound：更多块数有助掩盖
```

> 通过 P1-P4 剪枝和范围推导，搜索空间通常缩减到 10-50 个候选解（而非 M/16 ≈ 数百个），每个候选的性能差异可达 5-10us，只取几个代表点会遗漏最优点。

### 统一搜索流程

不论计算 bound 还是通信 bound，均以 hs 为自变量统一扫描。区别仅在剪枝阈值和理论时间公式：

```
// ============ 输入 ============
// M, N: 算子 shape
// baseM, baseN, N_core: 从 TilingData 打印获取
// α, β, R: 从隔离测试获取（bound_diagnosis.md）
// ρ = max(R, 1/R)

// ============ Step 1: 计算搜索范围 ============
nBlockCnt  = ceil(N / baseN)
utilThresh = (R >= 1) ? 0.80 : 0.50
minTotalBlk = ceil(utilThresh × N_core)
minMBlkCnt  = max(1, ceil(minTotalBlk / nBlockCnt))
hs_min = align16(max(baseM, (minMBlkCnt - 1) × baseM + 16))
hs_max = floor16(M / 2)
tcFloor = 2
tcCap   = (R >= 1) ? min(ceil(R) + 2, 8) : min(ceil(1/R) + 2, 16)

// ============ Step 2: 扫描所有候选 hs ============
candidates = []
for hs = hs_min to hs_max step 16:
    hc   = M / hs                    // 长块数量
    tail = M % hs                    // 尾块大小 (= shortMSize)

    // --- 剪枝 P1: 尾块下限 ---
    if tail > 0 and tail < 128:
        continue                     // 尾块过小，Mac 利用率极低

    // --- 剪枝 P3: tileCnt 范围 ---
    tc = hc + (tail > 0 ? 1 : 0)    // 总块数
    if tc < tcFloor or tc > tcCap:
        continue

    // --- 剪枝 P2: 核利用率（主轮 = 长块）---
    mBlockCnt  = ceil(hs / baseM)
    totalBlk   = mBlockCnt × nBlockCnt
    utilization = min(totalBlk, N_core) / N_core
    if utilization < utilThresh:
        continue                     // 用核不满，实际性能劣化

    // --- 剪枝 P4: 配平条件 ---
    shortMSize = tail
    if shortMSize > 0 and shortMSize < ρ × hs:
        continue                     // 短块无法掩盖相邻长块瓶颈操作

    // --- 理论时间估算 ---
    T = estimate_pipeline_time(hs, hc, shortMSize, tc, α, β, R)
    candidates.append({hs, hc, shortMSize, tc, T, utilization})

// ============ Step 3: 排序并输出 Top-N ============
sort candidates by T ascending
return candidates[:TOP_N]            // TOP_N = 15，供 msprof 实测验证
```

### 理论时间估算公式

根据 bound 类型使用不同的流水线时间模型：

**计算 bound（β > α，短块在 drain 端）**：

```
T_fill      = α × hs                              // 首块通信（无重叠）
T_steady    = (hc - 1) × β × hs                   // 稳态：计算为瓶颈
T_transition = max(α × shortMSize, β × hs)        // 末长块→短块切换
T_drain     = β × shortMSize                       // 短块计算（无重叠）
T_total     = T_fill + T_steady + T_transition + T_drain
```

**通信 bound（α > β，短块在 fill 端）**：

```
T_fill       = β × shortMSize                      // 首短块计算（无重叠）
T_transition = max(α × hs, β × shortMSize)        // 短块→首长块切换
T_steady     = (hc - 1) × α × hs                   // 稳态：通信为瓶颈
T_drain      = α × hs                              // 末块通信（无重叠）
T_total      = T_fill + T_transition + T_steady + T_drain
```

**无短块（tail == 0，均匀切分）**：

```
T_total = α × hs + (tc - 1) × max(α, β) × hs + max(α, β) × hs
```

> 理论时间仅为排序用代理指标，实际性能受 cache、流水 stall 等硬件因素影响。**必须对 Top-N 候选做 msprof 实测**才能确定最优解——理论最优不一定是实测最优。

### 多候选实测验证

搜索算法输出 Top-N（默认 15）个候选切分方案，而非仅返回理论最优。原因：

1. 理论模型未考虑计算膨胀（流水断流、L1 cache miss）和通信膨胀（头开销、带宽下降）的精确量
2. 不同 hs 值的膨胀特性不同，理论时间相近的候选实测可能差异显著
3. Top-N 候选已通过四条剪枝规则过滤，均为可行解，msprof 实测成本可控

**实施要求**：host 代码须支持通过命令行参数指定 `longMSize`，以便对 Top-N 候选逐一实测：

```cpp
// host 侧支持命令行指定 longMSize
if (longMSizeArg > 0) {
    longMSize = longMSizeArg;
    longBlockCnt = m / longMSize;
    shortMSize = m % longMSize;
    shortBlockCnt = (shortMSize > 0) ? 1 : 0;
    tileCnt = longBlockCnt + shortBlockCnt;
}
```

> 实施阶段对 Top-N 候选分别设置 longMSize 参数运行 msprof，按实测 aiv_time 选最优。

---

## R 值策略决策表

| R 范围 | 核利用率 | 参考策略 | 典型配置 |
|--------|---------|---------|---------|
| ≈1 (0.9~1.1) | 满核 | 等分 n=4 | longMSize=512, tc=4 |
| 1.3~2.0 | **不满 (<80%)** | 长短块，longMSize 使 mBlk 增大 | longMSize=576, tc=4 |
| 1.3~2.0 | 满核 | 长短块，增大 tileCnt | longMSize=352~432, tc=5~6 |
| >2 且恒满核 | 满核 | 等分 n=2 | longMSize=1024, tc=2 |
| >2 且不满 | 不满 | 长短块，longMSize 使 mBlk 增大 | longMSize=576, tc=4 |

### 搜索强制执行

所有场景**必须执行完整搜索**，不得跳过。上表中的"典型配置"仅作为 R 值区间的参考预期，实际 longMSize 须由搜索算法在配平可行域内系统遍历后取 Top-N 候选 msprof 实测确定。即使 R ≈ 1 或 R > 2 且恒满核，搜索仍须执行——搜索过程中可能发现核利用率约束或配平条件排除了预期方案，理论最优与实际最优常有偏差。

> 搜索的 Top-N 候选须全部 msprof 实测后择优。

---

## 输出

```
operator_order: "compute_then_comm" | "comm_then_compute"
bound_type: "balanced" | "compute" | "strong_compute" | "comm" | "strong_comm"
R: <β/α 比值>
α: <每行通信耗时 μs/row>
β: <每行计算耗时 μs/row>
tiling_data_ref:
  baseM: <从 host 打印获取>
  baseN: <从 host 打印获取>
  N_core: <usedCoreNum, 从 host 打印获取>
search_params:
  hs_min: <搜索下限>
  hs_max: <搜索上限>
  tcFloor: <块数下限>
  tcCap: <块数上限>
  utilThreshold: <核利用率阈值>
  candidate_count: <剪枝后候选总数>
top_candidates:                      // Top-N 候选，按理论时间排序
  - rank: 1
    long_m_size: <长块 M 维大小>
    long_block_cnt: <长块数量>
    short_block_cnt: <短块数量, 0 或 1>
    short_m_size: <短块 M 维大小>
    tile_cnt: <总块数>
    short_block_pos: 0 | 1   # 0=front, 1=back
    est_time: <理论估算时间 μs>
    utilization: <核利用率>
    key_constraint: <核心配平约束>
  - rank: 2
    ...
  - rank: N
    ...
recommended: <rank 1 的配置，作为首选实施方案>
tiling_correction:
  - field: <需修正的 tiling 字段>
    reason: <修正原因>
    value: <建议值>
```

> 输出的 Top-N 候选须全部交由实施阶段做 msprof 实测验证，按实测 aiv_time 选最优。理论排序仅用于缩小实测范围。
