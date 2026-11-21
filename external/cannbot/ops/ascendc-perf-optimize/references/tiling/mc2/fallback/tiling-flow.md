# MC² 通算融合族 — 均分切分 Tiling 流程

> 基于 matmul tiling 的 baseM/baseN，沿 M 维均分切分，使核利用率最高且切分块数最多的默认 Tiling 推导流程。
>
> **前置依赖**：baseM、baseN、usedCoreNum 须从 host 侧 TilingData 打印获取（见 [pipeline_balancing.md](../../../comm-compute/pipeline_balancing.md) 的「TilingData 依赖」章节）。

---

## 1. 设计原则

| 原则 | 说明 |
|------|------|
| **均分优先** | 所有 tile 等大（shortBlockCnt=0），流水调度对称，避免短块 Mac 利用率风险 |
| **利用率最高** | 主目标为满核（100%），无法满核时取可达到的最高利用率 |
| **切分最多** | 满核前提下取最小 longMSize → 最多 tileCnt → 最大化通信与计算掩盖机会 |
| **粒度对齐** | longMSize 须为 baseM 的整数倍（首选）或 16 对齐（降级），避免 Cube 单元浪费 |

> **均分 vs 长短块**：长短块配平（pipeline_balancing.md）通过非均匀切分使短块与长块完全掩盖，理论最优但依赖 α/β/R 精确值。均分无需隔离测试数据，作为兜底基线更稳健；当隔离测试数据齐备时，可切换至长短块搜索算法获取更优解。

---

## 2. 关键参数

| 参数 | 来源 | 含义 |
|------|------|------|
| `M` | 算子 shape | matmul 结果矩阵的 M 维总大小 |
| `N` | 算子 shape | matmul 结果矩阵的 N 维总大小 |
| `baseM` | TilingData 打印 | Cube 单元在 M 方向的基本块大小（16 对齐） |
| `baseN` | TilingData 打印 | Cube 单元在 N 方向的基本块大小（16 对齐） |
| `N_core` | TilingData 打印 (usedCoreNum) | 实际使用的 AIC 核数 |

### 派生参数

| 参数 | 公式 | 含义 |
|------|------|------|
| `nBlockCnt` | `ceil(N / baseN)` | N 方向的 block 总数（固定值，由 baseN 决定） |
| `mBlockCnt` | `ceil(longMSize / baseM)` | 单个 tile 在 M 方向的 block 数（随 longMSize 变化） |
| `totalBlocks` | `mBlockCnt × nBlockCnt` | 单个 tile 的总 block 数 |
| `utilization` | `min(totalBlocks, N_core) / N_core` | 单个 tile 的核利用率 |

---

## 3. 四步推导

```
EqualSplitTiling():
  ├── _calc_n_block_cnt()           # §3.1 计算 N 方向 block 数
  ├── _find_optimal_uniform()       # §3.2 求解最优均分（主路径）
  ├── _fallback_degrade()           # §3.3 降级路径（无完美均分解时）
  └── _build_tiling_data()          # §3.4 组装通算切分参数
```

### 3.1 计算 N 方向 block 数

N 方向的 block 数由 baseN 决定，与切分方案无关：

```
nBlockCnt = ceil(N / baseN)
```

- 若 `nBlockCnt ≥ N_core`：M 方向只需 1 个 block 即可满核，minMBlockCnt = 1
- 若 `nBlockCnt < N_core`：需要 M 方向多个 block 凑满核，minMBlockCnt = ceil(N_core / nBlockCnt)

### 3.2 求解最优均分（主路径）

**核心思路**：枚举所有满足约束的 longMSize，按 (利用率, tileCnt) 字典序选最优。

```
输入: M, baseM, nBlockCnt, N_core

Step 1 — 计算满核所需的最小 M 方向 block 数:
  minMBlockCnt = max(1, ceil(N_core / nBlockCnt))

Step 2 — 计算最小 longMSize（满核下限）:
  minLongMSize = minMBlockCnt × baseM

Step 3 — 枚举候选 longMSize:
  候选集 = { k × baseM | k ∈ [minMBlockCnt, M/baseM], (k × baseM) 整除 M }

  即: longMSize 是 baseM 的倍数，且能整除 M（保证均分无尾块）

Step 4 — 按目标排序选最优:
  对每个候选 longMSize:
    mBlockCnt  = longMSize / baseM
    totalBlk   = mBlockCnt × nBlockCnt
    util       = min(totalBlk, N_core) / N_core
    tileCnt    = M / longMSize

  排序: 主键 util 降序, 次键 tileCnt 降序（即 longMSize 升序）
  选择: 排序后第一个候选即为最优
```

**为什么这样选**：
- 主键选利用率最高：满核时计算资源不浪费，是性能的基础保障
- 次键选 tileCnt 最多：更多切分 → 通信与计算的 overlap 窗口更多 → 流水掩盖更充分

**典型场景**：

```
M=4096, N=4096, baseM=128, baseN=128, N_core=8

nBlockCnt    = ceil(4096/128) = 32
minMBlockCnt = ceil(8/32) = 1
minLongMSize = 1 × 128 = 128

候选 longMSize（baseM 倍数且整除 M）:
  128  → mBlk=1,  totalBlk=32,  util=100%, tileCnt=32
  256  → mBlk=2,  totalBlk=64,  util=100%, tileCnt=16
  512  → mBlk=4,  totalBlk=128, util=100%, tileCnt=8
  ...

最优: longMSize=128, tileCnt=32（满核 + 最多切分）
```

```
M=2048, N=256, baseM=128, baseN=128, N_core=8

nBlockCnt    = ceil(256/128) = 2
minMBlockCnt = ceil(8/2) = 4
minLongMSize = 4 × 128 = 512

候选 longMSize:
  512  → mBlk=4,  totalBlk=8,  util=100%, tileCnt=4
  1024 → mBlk=8,  totalBlk=16, util=100%, tileCnt=2
  2048 → mBlk=16, totalBlk=32, util=100%, tileCnt=1

最优: longMSize=512, tileCnt=4（满核 + 最多切分）
```

### 3.3 降级路径（无完美均分解时）

当 M 不存在满足「baseM 倍数 + 整除 M」的因子时，按以下优先级降级：

#### 降级 A：放宽对齐到 16

```
候选集 = { d | d 整除 M, d ≥ minLongMSize, d % 16 == 0 }

即: 不再要求 baseM 倍数，放宽到 16 对齐
注意: longMSize 非 baseM 倍数时，最后一个 M 方向 block 为部分块（partial block），
      Mac 利用率略有损失，但整体仍优于短块方案
```

#### 降级 B：允许短块

```
无法找到整除 M 的 longMSize，允许 1 个短块:

longMSize   = minLongMSize（或更大，按 tileCnt 目标选）
longBlockCnt = floor(M / longMSize)
shortMSize   = M - longBlockCnt × longMSize
shortBlockCnt = 1（若 shortMSize > 0）
shortBlockPos = 1（默认放后，drain 端）

约束: shortMSize ≥ 16（否则减小 longMSize 或 tileCnt）
```

> 短块位置默认 drain 端（后）。具体位置应根据执行顺序 × Bound 类型调整（见 [pipeline_balancing.md](../../../comm-compute/pipeline_balancing.md) 策略矩阵），但作为兜底默认放后。

#### 降级 C：放宽利用率阈值

```
若 minLongMSize > M（M 太小无法满核）:

放宽利用率阈值至 80%（或更低）:
  minMBlockCnt = max(1, ceil(0.8 × N_core / nBlockCnt))
  minLongMSize = minMBlockCnt × baseM

若 M < baseM: 单块切分（tileCnt=1），无流水掩盖，仅保证功能正确
```

### 3.4 组装通算切分参数

```
输出:
  tileCnt        = M / longMSize           # 均分时 = longBlockCnt
  longBlockCnt   = tileCnt                  # 均分时无短块
  longMSize      = <最优值>
  shortBlockCnt  = 0                        # 均分
  shortMSize     = 0                        # 均分
  shortBlockPos  = 0                        # 无短块
  utilization    = min(mBlockCnt × nBlockCnt, N_core) / N_core

引用:
  long_qbmm_tiling  → matmul/fallback/ 的 SWAT/FullLoad/StreamK（按 baseM 大小选择）
  short_qbmm_tiling → 无（shortBlockCnt=0）
  local_qbmm_tiling → comm-compute/local_matmul.md（全量 M 的 matmul tiling）
```

---

## 4. 算法选择决策树

```
给定: M, N, baseM, baseN, N_core

Step 1 — 计算 nBlockCnt 和 minMBlockCnt
  ├─ nBlockCnt = ceil(N / baseN)
  └─ minMBlockCnt = max(1, ceil(N_core / nBlockCnt))

Step 2 — 主路径：均分满核
  ├─ 存在 longMSize = k × baseM 整除 M 且 ≥ minMBlockCnt × baseM
  │   └─ 选最小满足者 → 均分满核（最优）
  └─ 不存在 → Step 3

Step 3 — 降级 A：放宽对齐
  ├─ 存在 longMSize 整除 M，16 对齐，≥ minMBlockCnt × baseM
  │   └─ 选最小满足者 → 均分放宽
  └─ 不存在 → Step 4

Step 4 — 降级 B：允许短块
  ├─ M ≥ minMBlockCnt × baseM
  │   └─ longMSize = minMBlockCnt × baseM, shortBlockCnt = 1
  └─ M < minMBlockCnt × baseM → Step 5

Step 5 — 降级 C：放宽利用率
  └─ 放宽阈值至 80%，M < baseM 则单块切分
```

---

## 5. 与长短块配平的关系

均分切分是 MC² Tiling 的**兜底基线**，长短块配平是**精细优化路径**：

| 维度 | 均分（本算法） | 长短块配平（pipeline_balancing.md） |
|------|--------------|-----------------------------------|
| 前置依赖 | 仅需 baseM/baseN/N_core | 额外需 α/β/R（隔离测试采集） |
| 切分方式 | 所有 tile 等大 | 长块 + 1 短块（非均匀） |
| 优化目标 | 满核 + 最多切分 | 短块与长块完全掩盖，流水总时间最短 |
| 适用场景 | 缺少隔离数据 / 基线对照 | 有完整隔离数据 / 追求极致性能 |
| 风险 | 无（保守稳健） | 短块过小导致 Mac 利用率低 |

**推荐流程**：
1. 先用均分切分作为基线，采集 aiv_time
2. 再用隔离测试采集 α/β/R（见 [bound_diagnosis.md](../../../comm-compute/bound_diagnosis.md)）
3. 用长短块搜索算法（[pipeline_balancing.md](../../../comm-compute/pipeline_balancing.md)）搜索 Top-N 候选
4. msprof 实测对比，选最优

> 均分切分的结果也可作为长短块搜索的 `hs_min` 下限参考——均分的 longMSize 通常是搜索空间中「满核 + 最大 tileCnt」的解。
