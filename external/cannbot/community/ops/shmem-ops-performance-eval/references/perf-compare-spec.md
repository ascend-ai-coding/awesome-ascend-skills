# 性能对比规范

Agent 在 Phase 6 完成后 **MUST** 在聊天输出与 `design.md` 目标对齐的对比结果（数值来自 `[PERF]` / `[BASELINE_PERF]`，**NEVER** 编造）。

> **禁止在 skill 中硬编码算子标杆**：达标阈值、采集规模、HCCL baseline 数值 **MUST** 写入 `design.md`（`performance.baseline_target`、`performance.target_cases`）。skill 只规定对比方法与达标口径。
>
> **许多算子不存在打满带宽的场景**（小 payload、时延主导、稀疏通信等）。此类算子 **无需** 探测带宽饱和区；**SHMEM 与 baseline 使用相同数据量对比即可**。

## 目标来源（MUST Read design.md）

| 字段 | 用途 |
| --- | --- |
| `performance.baseline_target` | 达标阈值（如 `kernel_bus_bandwidth_GBps >= 80% baseline`、`HCCL/SHMEM latency >= N×`） |
| `performance.target_cases` | 性能采集 case（shape、dtype、PE count、参数）—— **SHMEM 与 baseline 必须一致** |
| `performance.min_scale` | 最小测试规模（按算子语义在 design 阶段约定） |

Agent **MUST** 在聊天对比表中引用 `design.md` 的 `baseline_target` 作为目标列。

## 同量对比原则（Hard Gate）

1. **参数一致**：SHMEM `[PERF]` 与 baseline `[BASELINE_PERF]` **MUST** 使用 `performance.target_cases` 中同一 case 的 shape、dtype、PE 数、sendcounts 等
2. **分阶段采集**：HCCL 与 SHMEM **NEVER** 同 shell 混跑（见 [perf-workflow.md §1](perf-workflow.md)）
3. **指标口径**：
   - 带宽类：`kernel_bus_bandwidth_GBps`（kernel 口径）作对比主指标；e2e 带宽仅参考
   - 时延类：`comm_only_us` 或 `kernel_us`（见 [timing-and-metrics-standard.md §2.4](timing-and-metrics-standard.md)）
4. **不要求带宽饱和**：小 payload / 时延算子 **不得** 因「未打满链路带宽」而判定测试无效；在 design 指定 case 上对比即有效

## 采集流程

对 `performance.target_cases` 中每个 case 执行 [perf-workflow.md §1](perf-workflow.md) 三阶段（A baseline → B SHMEM → C 离线对比）。

输出含「带宽表 + 时延表」；Agent **MUST** 将对比 Markdown 表贴入聊天（格式见 [perf-chat-output-spec.md](perf-chat-output-spec.md)）。

## 可选：带宽饱和探测（sweep）

**仅当** `design.md` 明确需要表征「随 payload 增大带宽如何变化」（如大消息集合通信算子）时，**MAY** 按 [perf-workflow.md §2](perf-workflow.md) 做倍增 sweep。**NEVER** 作为所有算子的默认必做步骤。

## 达标口径

| 算子类型 | 指标 | 比值公式 | 目标来源 |
| --- | --- | --- | --- |
| 带宽类 | `kernel_bus_bandwidth_GBps` | SHMEM_kernel_bus / HCCL_kernel_bus | `design.md` `performance.baseline_target` |
| 时延类 | `comm_only_us` 或 `kernel_us` | HCCL / SHMEM | `design.md` `performance.baseline_target` |

补充规则：

- HCCL baseline 未采到 → 标注「未采到」并 retry baseline
- 无 `comm_only_us` 时，时延算子可用 `kernel_us`（须在报告中注明口径）
- 有 baseline 但未写明 `performance.baseline_target` → 默认带宽类 `current >= 80% baseline`

## peak_bandwidth 参考

计算带宽利用率时 **MAY** 参考 [hardware-architecture.md §2.6](../../shmem-ops-design/references/hardware-architecture.md)。利用率 >100% 须在报告中注释口径；**不得**因未达 peak 而否定同量对比结论。
