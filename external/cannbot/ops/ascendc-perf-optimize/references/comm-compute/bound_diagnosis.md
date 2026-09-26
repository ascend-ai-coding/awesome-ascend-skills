# Bound 判定方法

> MC² 算子通过通信和计算的掩盖达成性能提升。不同数据量下通信和计算耗时比值 R 差异巨大（实测 R 从 0.3 到 9.8），需首先通过 α/β 分解精确判定 Bound 类型，以确定长短块配平策略和搜索方向。

---

## Bound 类型

通过通信计算比值 R = β/α = T_compute / T_comm 判定：

| R 值范围 | Bound 类型 | 物理含义 | 优化方向 |
|---------|-----------|---------|---------|
| R ≈ 1 (0.9~1.1) | 完美平衡 | 通信与计算耗时接近 | 等分切分即最优 |
| 1.1 < R ≤ 2 | 计算 bound | 计算耗时 > 通信耗时 | 用通信掩盖计算，增大 tileCnt |
| R > 2 | 强计算 bound | 计算远大于通信 | 增大 tileCnt 收益极小，倾向少切分；须搜索确认 |
| R < 1 | 通信 bound | 通信耗时 > 计算耗时 | 用计算掩盖通信，增大 tileCnt |
| R < 0.5 | 强通信 bound | 通信远大于计算 | 从短块出发搜索，最大化 tileCnt |

> 配平比 ρ = max(R, 1/R)，即长块与短块的 M 大小之比。

---

## α/β 采集方法

### 隔离测试法

在 **tileCnt=1**（无流水线）的条件下，通过注释算子中的通信或计算 process 代码，直接隔离测试纯通信耗时和纯计算耗时：

> ⚠️ **隔离测试须在源码副本中进行**，不修改用户原目录。测试产物（profiling 数据）保留供分析，代码副本测完丢弃。

> ⚠️ **采集方法**：MC² 算子须按 `comm-compute/index.md` 的「性能采集方法」多 rank 并行采集——所有 rank 分别注释通信或计算 process，循环至少 10 次（前 5 次 warm-up），取各 rank 后 5 次 Task Duration 均值的**最大值**作为 T_comm 或 T_compute。

```
Step 0 — 复制源码到临时目录（如 {code_dir}_diag/）
Step 1 — 注释计算 process，仅保留通信 → 测得纯通信耗时 T_comm
Step 2 — 注释通信 process，仅保留计算 → 测得纯计算耗时 T_compute
Step 3 — 计算 R = T_compute / T_comm
```

### 计算 α 和 β

```
α = T_comm / M    （每行通信耗时，μs/row）
β = T_compute / M  （每行计算耗时，μs/row）
R = β / α = T_compute / T_comm
```

---

## 输出

```
bound_type: "balanced" | "compute" | "strong_compute" | "comm" | "strong_comm"
R: <β/α 比值>
α: <每行通信耗时 μs/row>
β: <每行计算耗时 μs/row>
T_comm: <纯通信耗时 μs>
T_compute: <纯计算耗时 μs>
ρ: <配平比 = max(R, 1/R)>
```

Bound 判定结论驱动 [流水配平策略](pipeline_balancing.md) 的三因子联合优化。

---

## MC² 场景 MTE2 污染判定

> MC² 通算融合算子中，`CrossCoreWaitFlag` 通常挂在 AIC 的 MTE2 流水线上等待 AIV 通信完成。这会导致 profiling 中 `aic_mte2_ratio` **含通信等待空转**，而非真实访存瓶颈。

隔离测试 Step 2（纯计算，通信已注释）的 profiling 数据天然排除了通信等待。通过对比完整算子与纯计算的 MTE2 ratio，可判定是否被污染：

```
Step 1 — 取完整算子 profiling 的 aic_mte2_ratio（含通信等待）
Step 2 — 取隔离测试 Step 2（纯计算）profiling 的 aic_mte2_ratio（无通信等待）
Step 3 — 判定：
         若 ratio_full - ratio_pure > 0.1（或 ratio_pure / ratio_full < 0.8）
         → mte2_polluted = true，MTE2 ratio 被通信等待污染
         否则 → mte2_polluted = false
```

**污染为 true 时的路由影响**：

- 核内 Bound 诊断**不应**路由到 `single-core-pipeline/memory.md`（访存优化策略对通信等待空转无效）
- 应路由到 `comm-compute/` 下的通信掩盖策略（如 [local_matmul.md](local_matmul.md)、[pipeline_balancing.md](pipeline_balancing.md)），通过消除/减少通信等待来降低 MTE2 占比

输出新增字段：

```
mte2_polluted: true | false
mte2_ratio_full: <完整算子 aic_mte2_ratio>
mte2_ratio_pure: <纯计算 aic_mte2_ratio>
```
