# MC² 通算融合类算子性能优化策略索引

按策略类型查找对应的适用场景与详细设计文档。

适用于 MC² 通算融合族所有变体：`matmul_all_reduce`、`allgather_matmul`、`matmul_reducescatter`、`alltoall_matmul` 等。

> MC²（Matrix Computation & Communication）是昇腾通算融合算子框架，将集合通信与计算（Cube 计算、量化计算等）融合为单一算子，实现通信与计算流水线并行。

| 策略 | 优先级 | 适用场景 | 核心手段 | 详细文档 |
|------|--------|---------|---------|---------|
| **local_matmul** | 首选 | 所有 MC² 通算融合算子 | 将本 rank 数据的 matmul 从通信块循环中分离，前置/后置独立执行以与通信重叠 | [local_matmul_design.md](local_matmul_design.md) |
| **pipeline_balancing** | 中 | 所有 MC² 通算融合算子（在 Local Matmul 基础上叠加） | 按 Bound 类型确定长短块排布，**TilingData 驱动的系统搜索**（四条剪枝 + Top-N 候选实测）选最优切分 | [pipeline_balancing_design.md](pipeline_balancing_design.md) |
| **matmul 计算效率优化** | 最低 | 仅计算 Bound（正交可叠加） | 参考 matmul 族优化策略（pingpong/streamk/fullload 等），提升单块 matmul 效率 | [matmul/guide.md](../matmul/guide.md) |

> Local Matmul 不依赖 Bound 判定，推荐优先实现。流水配平在 Local Matmul 基础上进一步优化长短块排布——**须先打印 TilingData (baseM/baseN/usedCoreNum) 并执行隔离测试获取 R 值**，再通过系统搜索算法（非经验值）确定最优 longMSize。计算 Bound 场景下可进一步叠加 matmul 计算效率优化。三者可结合使用。

## 融合模式分类

| 融合模式 | 执行顺序 | 典型算子 |
|---------|---------|---------|
| TP 权重聚合 | 通信→计算 | allgather_matmul |
| TP 输出分发 | 计算→通信 | matmul_reducescatter |
| 全局归约 | 计算→通信 | matmul_all_reduce |
| EP 专家分发 | 通信→计算 | alltoallv_grouped_matmul |
| EP 反向分发 | 计算→通信 | grouped_matmul_alltoallv |

> 纯通信算子（无计算融合）不属于 MC² 族，不适用本族优化策略。

---

## 与分析层 skill 的关系

本文档为**实现层**设计指南，提供 TilingData 结构、Host 侧参数计算、Kernel 核心循环和修改点等实施细节。对应的**分析层**框架（Bound 判定、搜索算法、膨胀分析、性能采集方法、性能指标核算）详见 `/ascendc-perf-optimize` skill 的 `references/comm-compute/` 目录：

| 分析维度 | 分析层文档（ascendc-perf-optimize） |
|---------|----------------------------------|
| 性能采集方法 / 多 rank 采集 / 稳态取值 | [comm-compute/index.md](../../../ascendc-perf-optimize/references/comm-compute/index.md)「性能采集方法」 |
| Bound 判定（α/β 分解、隔离测试法、MTE2 污染） | [comm-compute/bound_diagnosis.md](../../../ascendc-perf-optimize/references/comm-compute/bound_diagnosis.md) |
| 流水配平搜索算法（四条剪枝 + Top-N 候选） | [comm-compute/pipeline_balancing.md](../../../ascendc-perf-optimize/references/comm-compute/pipeline_balancing.md)「自适应搜索算法」 |
| 膨胀分析 | [comm-compute/expansion_analysis.md](../../../ascendc-perf-optimize/references/comm-compute/expansion_analysis.md) |
| Local Matmul 分析 | [comm-compute/local_matmul.md](../../../ascendc-perf-optimize/references/comm-compute/local_matmul.md) |
| 性能指标（加速比 / 掩盖率） | [comm-compute/index.md](../../../ascendc-perf-optimize/references/comm-compute/index.md)「性能指标」 |
| 硬约束（MC-C1/C2/C3） | [comm-compute/index.md](../../../ascendc-perf-optimize/references/comm-compute/index.md)「硬约束」 |

> 实施前须先通过分析层完成 TilingData 采集和隔离测试，获取 baseM/baseN/usedCoreNum 和 R 值后，再参考本文档进行代码实现。
