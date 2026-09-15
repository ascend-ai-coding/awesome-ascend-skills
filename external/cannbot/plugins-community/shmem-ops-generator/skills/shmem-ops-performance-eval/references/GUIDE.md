# 参考资料索引

本目录包含 `shmem-ops-performance-eval` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [performance-eval-guide.md](performance-eval-guide.md) | 采集流程概览、contract 字段、结果表格式、规模要求 | 步骤 1-6 全过程 |
| [baseline-selection.md](baseline-selection.md) | baseline 选择策略、HCCL/aclnn 清单、决策树 | 步骤 1 baseline 选择时 |
| [perf-compare-spec.md](perf-compare-spec.md) | 同量对比 / 达标口径 / 非饱和规则 | Phase 6 对比时 |
| [hccl-expansion-mode.md](hccl-expansion-mode.md) | HCCL `HCCL_OP_EXPANSION_MODE`：优先 AIV，回退 AI_CPU | 阶段 A baseline 前 |
| [baseline-compare-workflow.md](baseline-compare-workflow.md) | baseline 接入 / perf_compare / 聊天对比表 | 步骤 1-3 |
| [perf-chat-output-spec.md](perf-chat-output-spec.md) | **聊天自动输出**、Phase 6.5 每轮 Δ% | 采集/优化完成后 |
| [timing-and-metrics-standard.md](timing-and-metrics-standard.md) | 时间打点标准、指标计算公式、Device 打点、计算指标、采集方法 | 步骤 2-4 性能采集和对比时 |
| [profiling-tools.md](profiling-tools.md) | msprof、SHMEM cycle profiling 使用方法 | 步骤 2 性能采集时 |
| [device-profiling-guide.md](device-profiling-guide.md) | SHMEMI_PROF 完整操作指南 | 步骤 2 Device 打点采集时 |
| [perf-workflow.md](perf-workflow.md) | 性能三阶段采集命令代码段 | Phase 6 采集时 |
| [skill-performance-review.md](skill-performance-review.md) | **维护者备忘**（历史修正；Phase 6 不必 Read） | 维护 skill 时 |
| [../../shmem-ops-dev/references/shmem-repo-resolution.md](../../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO` | 读仓内 profiling 文档前 |

跨 skill 参考：

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [hardware-architecture.md](../../shmem-ops-design/references/hardware-architecture.md) | 昇腾硬件参数速查（核数、P2P 带宽等） | 计算带宽利用率时 |

模板：

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [performance-report.md](../templates/performance-report.md) | 性能报告模板（7 段式） | 输出报告时 |
