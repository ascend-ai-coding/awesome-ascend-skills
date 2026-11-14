# SHMEM 性能采集流程与规范

本文规定 `shmem-ops-performance-eval` 在 correctness 通过后如何采集性能、分析瓶颈并输出结构化数据。所有性能结论 **MUST** 来自实际命令输出或明确标注为假设。

## 核心原则

- **性能阶段是强制性阶段**：仅当 `meta.performance_required: true` 时，正确性通过后 **NEVER** 直接停止，**MUST** 进行性能采集、baseline 接入和瓶颈分析
- **有 baseline 时 **MUST** 接入**：有 HCCL/aclnn baseline 时 **MUST** 接入并对比；通算融合按 [baseline-selection.md](baseline-selection.md) P1 fused → P2 stitched；默认 current ≥ baseline 的 **80%**
- **规模门禁**：Phase 6 **MUST** 采集 **S 档 + L 档**（见本文 §4）；PE 数推荐 8PE，资源受限时可降级为 4PE/2PE 并在报告标注
- **无 baseline 时 **MUST** 指标验收**：通信算子测量带宽利用率或端到端时延，能计算带宽利用率时 **NEVER** 低于 20%
- **达标线**：有 baseline 时默认 current ≥ baseline **80%**；无 baseline 时通信算子带宽利用率 ≥ **20%**（详见 §4.2）
- **功能完善不算性能优化**：例如从 Host RMA 改为 Device kernel 是功能完善，不是性能优化
- **并发补齐不算性能优化**：例如从临时 `block_dim=1` 改为 design 要求的多 block，是 correctness 实现补齐
- **只采集当前实现时间不算完成**：**MUST** 有 baseline 对比并达标，或 在 metric_only 下完成指标测量并达标

## 1. 前置条件

进入性能阶段前 **MUST** 满足：

- 编译成功
- functional/correctness contract 已通过
- implementation-vs-design review 已通过：编译 target、transport API、launch block_dim、phase、tile/chunk/tail 与 design 一致
- 中等规模 correctness case 已通过，**NEVER** 只依赖 smoke 小 shape
- 输出可见性、同步不变量、tail case 已验证或明确未验证原因
- `design.md` 的 performance contract 已读取

correctness 未通过时，性能数据无效。

## 2. 性能 contract 字段

`design.md` 应提供或由实现阶段补齐：

| 字段 | 说明 |
| --- | --- |
| metric | `latency_us`、`algo_bandwidth_GBps`、`e2e_bus_bandwidth_GBps`、`kernel_bus_bandwidth_GBps`、`bandwidth_utilization_percent`、`effective_flops`、`compute_utilization_percent`、`cycles` |
| baseline | HCCL（见 baseline-selection.md §1.1）、aclnn（见 baseline-selection.md §1.2）、拼接（见 baseline-selection.md §2）、已有 SHMEM example、用户参考实现、或 metric_only |
| baseline_target | 有 baseline 时默认 current ≥ 80% baseline；无 baseline 时写 metric_only 指标目标，通信算子能计算带宽利用率时 **NEVER** 低于 20% |
| cases | shape、dtype、PE count、engine、scope |
| min_scale | 集合通信至少 256MB 级通信数据量；无法满足时说明原因 |
| repeats | 预热次数、统计次数 |
| profiler | msprof、torch_npu profiler、SHMEM cycle profiling、example 内部计时 |
| target | 目标改善方向或验收阈值 |

如果 contract 不完整，先补到 `design.md` 或在实现日志中显式写出假设。

## 3. 性能结果表

每个 case 至少输出：

| 字段 | 含义 |
| --- | --- |
| case_id | 唯一 case 名称 |
| shape | 输入输出 shape |
| dtype | 数据类型 |
| n_pes | PE 数 |
| engine | MTE、SDMA、RDMA/RoCE 或 default |
| metric | 主指标 |
| baseline | baseline 数值或 N/A |
| current | 当前实现数值 |
| delta | 与 baseline 差异 |
| target | baseline 80% 达标线，或 metric_only 指标目标 |
| pass | 是否达标 |
| notes | 关键现象 |

如果没有 baseline，仍需记录 current、方法、baseline 搜索过程、metric_only 目标和达标判断，不要伪造对比。

通信算子额外输出：

| 字段 | 含义 |
| --- | --- |
| logical_payload_bytes | 算法语义数据量，说明单 PE/全局口径 |
| algo_bandwidth_GBps | 见 [timing-and-metrics-standard.md §4](timing-and-metrics-standard.md) |
| kernel_bus_bandwidth_GBps | 达标主指标；见 [timing-and-metrics-standard.md §4](timing-and-metrics-standard.md) |
| peak_bandwidth_GBps | 硬件链路或 fabric 峰值及来源，决策表见 [timing-and-metrics-standard.md §4.4](timing-and-metrics-standard.md) |
| bandwidth_utilization_percent | 见 [timing-and-metrics-standard.md §4](timing-and-metrics-standard.md) |

计算算子额外输出：

| 字段 | 含义 |
| --- | --- |
| op_count_flops | 有效 FLOPs 公式和数值 |
| compute_latency_us | compute frame 耗时；若使用端到端耗时 **MUST** 标注 |
| effective_flops | `op_count_flops / compute_latency_s` |
| peak_flops | 当前 SoC + dtype 的硬件峰值及来源 |
| compute_utilization_percent | `effective_flops / peak_flops * 100` |

关键片段占比额外输出：

| frame_id | phase | avg_us | max_core_us | count | percent_of_e2e | bottleneck_note |
| --- | --- | --- | --- | --- | --- | --- |

## 4. 性能 case 规模要求

**Hard Gate**：Phase 6 **MUST** 覆盖 **S 档 + L 档** 各至少 1 case

| 算子类型 | S 档（小数据量） | L 档（大数据量） |
| --- | --- | --- |
| 通信算子 | 64KB~1MB/PE；时延类主对比档 | ≥64MB/PE；带宽类用 `kernel_bus_bandwidth_GBps` |
| 通算融合 | 小 MatMul + 小通信 payload；**端到端时延** | 大 MatMul + 大通信；**端到端 e2e**（分段为可选诊断） |

- smoke/XS 只用于启动和 sanity check，**NEVER** 作为唯一性能结论
- Round 0 **MUST** 覆盖 S + L；中间优化轮次以 L 档为准；最终轮重新采集 S + L
- 集合通信 L 档优先 8PE、通信总量优先 ≥256MB（资源受限时可降级，**MUST** 在报告注明实际 PE/规模）
- 达不到 S+L 门禁时，最终报告 **MUST** 写「未满足性能规模门禁」，**NEVER** 写成完整性能完成

### 4.2 达标口径（统一）

| 场景 | 主指标 | 达标线 |
| --- | --- | --- |
| 通信算子有 baseline | `kernel_bus_bandwidth_GBps`（或约定时延比） | current ≥ baseline **80%** |
| 通算融合有 baseline（P1 fused / P2 stitched） | **端到端 `e2e_us`**（带宽为辅） | current ≥ baseline **80%**（时延：SHMEM e2e ≤ baseline_e2e / 0.8） |
| 无 baseline（metric_only） | 带宽利用率 / 约定时延 | 通信算子利用率 **NEVER** 低于 **20%** |

> **不再**使用按算子定制的独立目标表（如历史「某算子 ≥105%」）；特例若 design 声明 `baseline_target`，以 design 为准并在报告注明。

### 4.3 PE 数

- **推荐 8PE**（单机标准配置），**不强制**
- 资源不足可用 design / Phase 0 确认的 4PE 或 2PE，报告 **MUST** 标注实际 PE 数
- 降级后的 S+L 结果仍可作为 Phase 6 依据，不因未满 8PE 单独判 blocked

## 5. 采集命令（custom-ops 默认）

**MUST** 按 [perf-workflow.md](perf-workflow.md) 执行（custom-ops 为 skill 生成交付物；命令以该 md 代码段为准）。HCCL 与 SHMEM **分阶段、分 shell**，对比离线进行。

性能前后正确性回归：见 [custom-ops-entrypoints.md §3](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)。

**alltoallv 参考**：两次独立 `docker exec`，容器内粘贴 perf-workflow §1 阶段 A/B，再 §1 阶段 C 离线对比。

Agent 完成 perf 采集后 **MUST** 在聊天中摘要：
1. **S 档 + L 档性能表**（kernel_bus_bandwidth_GBps、utilization%）
2. **SHMEM vs baseline 对比**（kernel_bus_bandwidth_GBps 比值、PASS/FAIL）
3. **NEVER** 只报单点 e2e 时延或单边数据

详见 [baseline-compare-workflow.md](baseline-compare-workflow.md)。

in-tree example 仍使用算子目录内 `scripts/perf.sh` 或 `scripts/run.sh --perf`。
