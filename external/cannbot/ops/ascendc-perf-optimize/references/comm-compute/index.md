# 通算演算优化策略

> MC² 通算融合算子的卡间流水分析。Local Matmul 是首选优化策略（不依赖 Bound 判定），流水配平在此基础上进一步优化长短块排布。两者结合使用，推荐优先分析和实现 Local Matmul。计算 Bound 场景下可进一步叠加 matmul 计算效率优化（优先级最低）。

---

## 性能指标

MC² 算子的性能目标通过以下两个指标衡量：

### 掩盖率

衡量通信与计算流水掩盖的程度：

```
掩盖率 = (融合前耗时 - 融合后耗时) / min(计算耗时, 通信耗时)
```

- **融合前耗时**：tileCnt=1 时的算子总耗时（通信与计算串行）
- **融合后耗时**：优化后（流水配平等）的算子总耗时
- **计算耗时 / 通信耗时**：在 tileCnt=1 下，注释掉通信或计算过程分别测得

掩盖率越高，说明通信被计算掩盖得越充分。理论上限为 1.0（完全掩盖）。

### 加速比

衡量通算融合带来的端到端性能提升：

```
加速比 = 融合前耗时 / 融合后耗时
```

- **融合前耗时**：tileCnt=1 时的算子总耗时
- **融合后耗时**：优化后的算子总耗时

---

## 性能采集方法

MC² 算子是多 rank 协同的通算融合算子，其性能采集与单卡算子有本质区别——通信开销需要跨卡交互才能真实体现，且多 rank 间的同步（BarrierAll / CrossCoreFlag）使整体性能受制于最慢的 rank。

> ⚠️ **禁止使用 host 侧计时（std::chrono / gettimeofday 等）替代 msprof 采集**。MC² 算子的所有性能数据（基线、隔离测试、方案对比）**必须**通过 msprof 采集，从 `op_summary_*.csv` 的 `Task Duration(us)` 字段提取。host 侧计时包含 launch 开销且精度不足，不得作为性能指标。

### 采集工具：msprof 多 rank 采集

msprof 包裹 fork 多 rank 程序时，会自动为每个活跃设备分别产出 profiling 数据。每个设备一个 `PROF_` 目录，各自含独立的 `op_summary_*.csv`。

```bash
# 采集命令（程序内部 fork 多 rank，msprof 自动采集所有设备）
msprof --output={prof_dir} \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       {code_dir}/build/{exe} {m} {k} {n} {rankNum} perf {tileCnt}
```

### 多 rank 并行采集

每个 rank 绑定一张 NPU 卡，所有 rank **并行启动**（如 fork 子进程），各自运行完整的算子逻辑。msprof 自动为每个设备产出独立的 `PROF_` 目录。

### 稳态取值

perf 模式下循环执行至少 10 次算子调用，**前 5 次为 warm-up**（L2 cache 预热、DVFS 频率爬升），**取后 5 次 Task Duration 的均值**作为该 rank 的稳态性能。

**Task Duration 提取方法**：从每个设备的 `PROF_*/mindstudio_profiler_output/op_summary_*.csv` 中，读取 `Task Duration(us)` 列。该 CSV 中每行对应一次 kernel launch（10 轮 = 10 行），丢弃前 5 行（warm-up），取后 5 行均值。

### 跨 rank 取最大值

多 rank 协同算子存在**木桶效应**——所有 rank 通过 BarrierAll / CrossCoreFlag 同步，整体性能由**最慢的 rank** 决定。因此最终性能指标取所有 rank 稳态均值的**最大值**，而非平均值。

```
T_task = max(rank_0_steady_avg, rank_1_steady_avg, ..., rank_N_steady_avg)
```

> 此采集方法适用于 MC² 算子的所有性能测试场景：基线 profiling、隔离测试（COMM_ONLY / COMPUTE_ONLY）、方案对比测试。隔离测试时每个 rank 分别注释通信或计算 process，再按同样方式取各 rank 稳态均值的最大值。

---

## 适用算子

通算融合算子（集合通信 + Cube/Vector 计算），典型场景：

| 融合模式 | 通信原语 | 计算原语 | 执行顺序 |
|---------|---------|---------|---------|
| TP 权重聚合 | AllGather | Matmul | 通信→计算 |
| TP 输出分发 | ReduceScatter | Matmul | 计算→通信 |
| 全局归约 | AllReduce | Matmul | 计算→通信 |
| EP 专家分发 | AllToAllv | GroupedMatmul | 通信→计算 |
| 量化+集合通信 | AllReduce/ReduceScatter | Quant+Matmul | 计算→通信 |

> 纯通信算子（无计算融合）不适用本策略，走常规通信优化。

---

## AIC/AIV 分离架构

MC² 通算融合算子采用 AIC（计算核）+ AIV（向量核）分离架构，AIC 负责 matmul 计算，AIV 负责跨卡通信（AllToAll via UDMA / shmem）。通信与计算的执行顺序不同，同步方向完全相反：

**通信后计算（Pattern A，如 alltoall + matmul）**：AIV 先行通信，AIC 等待数据就绪后计算。同步方向 AIV → AIC：AIV `CrossCoreSetFlag` 通知数据就绪，AIC `CrossCoreWaitFlag` 等待。

**计算后通信（Pattern B，如 matmul + alltoall）**：AIC 先行计算，AIV 等待计算完成后通信。同步方向 AIC → AIV：AIC `NotifyComputeComplete` 通知计算完成，AIV `WaitComputeComplete` 等待后再 `BarrierAll` 对齐所有 rank 执行 AllToAll。

> 详见 [pipeline_balancing.md](pipeline_balancing.md) 的 AIC/AIV 分离架构章节。

---

## 分析流程

```
给定：算子类型、计算流程、profiling 数据、仿真图（可选）

Step 0 — TilingData 采集（MC² 必选前置）：
  确认 host 代码打印 baseM/baseN/baseK/usedCoreNum 等 tiling 参数
  若未打印 → 在源码副本中启用 PrintTilingData → 重新运行采集
  → pipeline_balancing.md「TilingData 依赖」章节

Step 1 — Local Matmul 策略（首选，不依赖 Bound 判定）：
  本 rank 数据无需通信，独立计算以与通信重叠
  → local_matmul.md

Step 2 — Bound 判定：
  通过 tileCnt=1 下分别测试通信和计算的纯耗时，判定 Bound 类型
  （按「性能采集方法」多 rank 并行采集，取各 rank 稳态均值最大值）
  ├─ 计算耗时 >> 通信耗时 → 计算 Bound → bound_diagnosis.md
  ├─ 通信耗时 >> 计算耗时 → 通信 Bound → bound_diagnosis.md
  └─ 通信 ≈ 计算 → 相近 → bound_diagnosis.md

Step 3 — 流水配平策略（在 Local Matmul 基础上叠加）：
  根据 Step 0 的 TilingData (baseM/baseN/N_core) 和 Step 2 的 R 值
  执行统一搜索算法（四条剪枝规则 + Top-N 候选输出）
  → pipeline_balancing.md「自适应搜索算法」章节
  → 膨胀影响评估：expansion_analysis.md

Step 4 — matmul 计算效率优化（仅计算 Bound，优先级最低，正交可叠加）：
  参考 matmul 族的优化策略（pingpong/streamk/fullload 等），提升单块 matmul 效率
  → ../../../ascendc-performance-best-practices/reference/matmul/guide.md

Step 5 — 性能指标核算（必选输出）：
  基于 Step 2 隔离测试的 T_comm / T_compute 和各方案实测耗时，计算加速比与掩盖率
  → 见上方「性能指标」章节的公式
```

> ⚠️ Step 0 为 MC² 算子的必选前置步骤。缺少 baseM/baseN/usedCoreNum 将导致核利用率计算失真，搜索算法无法正确剪枝。Step 5 为必选步骤。隔离测试已采集 T_comm 和 T_compute，加速比与掩盖率的计算只需代入公式，成本极低但对方案择优和收敛判断至关重要——掩盖率直接反映通信被计算掩盖的程度，是 MC² 算子区别于普通算子的核心评价指标。

## 阶段映射与输出要求

> 上述 Step 0-5 是逻辑分析流程，在实际调优工作流中须映射到「数据采集」和「分析」两个阶段执行。以下编排要点由调用方（如 ascendc-perf-analysis-expert）遵循，本节明确供 skill 自洽使用。

### 阶段映射

| skill 内步骤 | 归属阶段 | 说明 |
|---|---|---|
| Step 0（TilingData 采集） | **数据采集阶段** | 须在 profiling 采集前完成，从 host 打印输出中获取 baseM/baseN/baseK/usedCoreNum |
| Step 2（Bound 判定 / 隔离测试） | **数据采集阶段** | 须在 profiling 采集后、性能分析前完成，实测 T_comm / T_compute 计算 R 值 |
| Step 1 / Step 3 / Step 4（策略分析） | **分析阶段** | 基于数据采集阶段的 TilingData 和 R 值展开 |
| Step 5（性能指标核算） | **分析阶段** | 基于已采集的 T_comm / T_compute 核算 |

> 若分析阶段发现 TilingData 或隔离测试数据缺失，须返回数据采集阶段补采。

### per-case 记录要求

每个 case 的以下数据须**独立记录**，不得仅归组汇总：

- TilingData 参数（baseM / baseN / baseK / usedCoreNum，含长块/短块各自的 tiling data）
- 隔离测试结果（T_comm / T_compute / R 值）
- aiv_time / bound 类型 / 关键瓶颈指标

### 输出汇总

数据采集阶段结束时，须输出：
- 全部 case 的 aiv_time 汇总表
- **MC² 算子额外**：每个 case 的 TilingData 参数 + 隔离测试结果汇总

### 硬约束

| # | 约束 |
|---|------|
| MC-C1 | **缺失 TilingData 时不得进行流水配平分析**。若 host 代码未打印，须在源码副本中启用 PrintTilingData 后重新运行 |
| MC-C2 | **流水配平方案必须通过系统搜索确定切分参数**（四条剪枝 + Top-N 候选），禁止仅取 1-2 个经验值（如 longMSize=512）不做搜索就交付 |
| MC-C3 | **隔离测试须在源码副本中进行**，不修改用户原目录。测试产物保留供分析，代码副本测完丢弃 |

## 文档路由

| 分析维度 | 内容 | 文档 |
|---------|------|------|
| TilingData 采集（MC² 必选前置） | host 代码打印 baseM/baseN/usedCoreNum 的方法 | [pipeline_balancing.md](pipeline_balancing.md)「TilingData 依赖」章节 |
| Local Matmul（首选） | 本 rank 数据独立计算策略（前置/后置/融合） | [local_matmul.md](local_matmul.md) |
| Bound 判定 | 三类 Bound 的诊断方法与判定标准 | [bound_diagnosis.md](bound_diagnosis.md) |
| 流水配平 | 长短块排布 + AIC/AIV 架构 + 统一搜索算法（四条剪枝 + Top-N） | [pipeline_balancing.md](pipeline_balancing.md) |
| 膨胀分析 | 切分膨胀对端到端性能的影响与最小化策略 | [expansion_analysis.md](expansion_analysis.md) |
| matmul 计算效率优化（计算 Bound，最低优先级） | 参考 matmul 族优化策略提升单块 matmul 效率 | [matmul/guide.md](../../../ascendc-performance-best-practices/reference/matmul/guide.md) |
