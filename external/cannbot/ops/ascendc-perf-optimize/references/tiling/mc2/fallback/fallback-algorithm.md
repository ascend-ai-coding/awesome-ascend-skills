# MC² 通算融合族 — 兜底算法（均分切分）

> MC² 通算融合算子所有变体（matmul_all_reduce / allgather_matmul / matmul_reducescatter / alltoall_matmul）共享的默认 Tiling 兜底算法。
>
> 核心思想：**以 matmul tiling 的 baseM/baseN 为基本粒度，沿 M 维均分切分，使每块的核利用率最高，并在满核前提下取最多切分块数**，最大化通信与计算的流水掩盖机会。

---

## 1. 算法定位

MC² Tiling 分为三部分，本算法仅负责**通算切分参数**部分，其余复用既有建模：

| 部分 | 来源 | 说明 |
|------|------|------|
| 基础 matmul tiling | [matmul/fallback/](../../matmul/fallback/) | baseM/baseN/baseK/L1/L0 参数，由 host tiling 引擎生成 |
| **通算切分参数** | **本目录** | longMSize / tileCnt / shortBlock 等参数，**均分策略** |
| local matmul tiling | [comm-compute/local_matmul.md](../../../comm-compute/local_matmul.md) | 本 rank 全量 M 的独立 matmul tiling |

> **为何选择均分作为兜底**：长短块配平（见 [pipeline_balancing.md](../../../comm-compute/pipeline_balancing.md)）依赖隔离测试采集的 α/β/R 值，属于精细优化路径。当缺少这些参数或作为基线对照时，均分切分是最稳健的默认选择——每块大小一致、流水调度对称、不引入短块 Mac 利用率风险。

---

## 2. 路由流程

```
给定：M, N, baseM, baseN, N_core(usedCoreNum)

Step 0 — 前置检查：
  ├─ baseM/baseN/N_core 缺失 → 返回「需先打印 TilingData」
  └─ 齐备 → 进入 Step 1

Step 1 — 计算固定参数：
  ├─ nBlockCnt = ceil(N / baseN)
  └─ minMBlockCnt = max(1, ceil(N_core / nBlockCnt))   # 满核所需最小 M 方向 block 数

Step 2 — 求解最优均分：
  ├─ 枚举 longMSize = k × baseM，满足 longMSize | M 且 longMSize ≥ minMBlockCnt × baseM
  ├─ 主目标：utilization = min(ceil(longMSize/baseM) × nBlockCnt, N_core) / N_core → 取最大
  └─ 次目标：tileCnt = M / longMSize → 取最多（即 longMSize 最小）

Step 3 — 降级路径（无完美均分解时）：
  ├─ 降级 A：放宽 baseM 对齐为 16 对齐
  ├─ 降级 B：允许 1 个短块（shortBlockCnt=1，非均分）
  └─ 降级 C：放宽利用率阈值至 80%

Step 4 — 输出通算切分参数 + 引用 matmul/local tiling
```

---

## 3. 算法策略

| 策略 | 适用条件 | 核心思想 |
|------|---------|---------|
| **均分满核**（默认） | M 存在满足约束的因子 | 所有 tile 等大，满核利用，tileCnt 最大化 |
| **均分放宽**（降级 A） | M 无 baseM 倍数因子，但有 16 对齐因子 | 放宽对齐，保持均分 |
| **短块兜底**（降级 B） | M 为质数或无合适因子 | 1 短块 + N 长块，尽量贴近均分 |

---

## 4. 索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 均分切分四步推导 + 降级路径 + 选型决策 |
| [tiling-fields.md](tiling-fields.md) | 通算切分参数字段语义与取值规则 |
| [script/mc2_tiling.py](script/mc2_tiling.py) | Python 均分切分脚本（支持满核/降级路径） |
