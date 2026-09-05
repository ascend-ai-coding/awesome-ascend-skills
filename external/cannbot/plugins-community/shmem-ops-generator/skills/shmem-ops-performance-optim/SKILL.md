---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-performance-optim
description: 基于 SHMEM 的算子性能优化迭代，固定 5 轮机制优化，每轮输出 Δ% 对比。触发：用户需要定位 SHMEM 算子瓶颈、调优通信机制或执行自动性能优化时使用。关键词：基于SHMEM性能优化、调优、optimization、机制优化、性能调优。
original-name: shmem-ops-performance-optim
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子性能优化

**Skill类型**：分析推荐型（读性能数据 → 瓶颈分析 → 输出修改意见 → 委托子 skill 执行改动 → 验证 → 决策）。**禁止**自行修改算子代码或 design.md。

> **中文写作要求**：performance_report.md **MUST** 使用中文撰写。仅 API 名称、代码片段、指标名等技术术语保留英文原文。

正确性通过、Phase 6 性能采集完成且编排器因 **`meta.performance_auto_optim: true`** 且 **未达标** 进入 Phase 6.5 后执行（`false` 或已达标时本 skill **不得**被调用；用户明确指定优化轮次时无论是否达标都进入）。进入后 **MUST** 跑满 `max_opt_rounds`（默认 5）轮：**Round 1~5 均为机制优化轮**（每轮一项代码/机制改动 + 可选参数扫描）。**达标 不是 提前停止理由**——进入后 MUST 完成全部轮次（默认 5 或用户指定 N），以充分挖掘优化空间。

**通算融合 Hard Gate**（见 [fused-compute-phase-gates.md §5、§6](../shmem-ops-dev/references/fused-compute-phase-gates.md)）：
- 每轮 **MUST** 复跑 `run_case_matrix.py` + Phase 6 指标复采
- 有 HCCL baseline 时每轮对比表 **MUST** 含 SHMEM vs HCCL kernel_bus_bandwidth_GBps Δ%
- **禁止**将 H2D 移出循环、block_dim 扫参等参数项拆成 5 轮凑数；每轮须有实质机制差异（overlap、ping-pong、CoC 调优等）

performance_report.md 由 `shmem-ops-performance-eval` 在每轮采集后写入/追加。优化过程中 **NEVER** 直接编辑该报告——填表由 eval skill 负责，optim skill 只读取。

## 优化优先级（MUST）

1. **机制优化**（流水、分核、同步粒度、engine、overlap）— 默认从 Round 1 起即优先尝试
2. **参数调优**（block_dim、chunk_size 等）— 在机制改动后的同一轮内扫描，或机制无法推进时的补充
3. **大小分支** — **禁止**作为默认优化手段；仅 unified 路径 profiling 后证明 **≥5%** 收益才引入/保留 `big_data`/`large` 路径

### 统一实现原则

| 原则 | 说明 |
| --- | --- |
| 默认 single path | 一套 kernel 逻辑 + UB 分块，覆盖平台 S/L 档 |
| 外层 tile | 仅因 `GVA_BUFF_MAX_SIZE` / symm 容量触发的分块循环，可保留 |
| 证明后分拆 | size 分支须附 Round 对比数据，未达标则合并删除 |
| 死代码 | `if (0)`、`INT64_MAX` 禁用的 large 路径 **MUST** 在合并后删除 |

---

## 轮次模型

| 轮次 | 性质 | 内容 |
| --- | --- | --- |
| Round 1 | **机制优化轮**（默认） | 一项代码/机制改动（流水、分核、signal 等）+ 可选参数扫描 |
| Round 2-5 | **机制优化轮** | 每轮一项新机制改动，改动后再扫参数找最优 |

**参数 vs 机制的区分**：

| 类别 | 定义 | 示例 |
| --- | --- | --- |
| 参数调节 | 改配置值，不改代码结构 | block_dim、chunk_size、TileShape、Swizzle offset/direction、transfer_size、UB buffer 大小 |
| 机制改动 | 改代码逻辑/算法 | 加 ping-pong double buffer、实现 sender/receiver 分核、barrier→signal/wait、增加流水级数、engine 切换（MTE→SDMA）、通信-计算 overlap |

## 核心工作流

```
┌──────────────────────────────────────────────────────────────┐
│  OptimStep 1: 瓶颈定位（读 performance_report + design）      │
│  OptimStep 2: 基线锁定（确认全部基线指标）                    │
│  OptimStep 3: 输出修改意见                                    │
│    ├── 设计级改动 → 委托 shmem-ops-design 修订 design.md     │
│    │                 → 跳过 testcase-gen                      │
│    └── 代码级改动 → 委托 shmem-ops-code-gen 修改代码          │
│                      → 委托 shmem-ops-compile-debug 编译      │
│                        （compile-debug 不改代码，失败回 code-gen）│
│                      → 委托 shmem-ops-correctness-eval 正确性 │
│  OptimStep 4: 性能验证（调用 shmem-ops-performance-eval 采集并填入报告）│
│  OptimStep 5: 迭代决策（keep / revert）                       │
│               ↓ < 5 轮 → 回 OptimStep 1（无论是否达标）       │
│  OptimStep 6: 最终确认（检查报告完整性）                      │
└──────────────────────────────────────────────────────────────┘
```

## 必读资料

| 文件 | 阅读时机 | 用途 |
| --- | --- | --- |
| [references/optimization-patterns.md](references/optimization-patterns.md) | OptimStep 1 定位瓶颈后选择对应优化 | 通信、流水线、同步、内存、分核优化手段 |
| [references/compute-optimization.md](references/compute-optimization.md) | 当瓶颈在 compute 部分时 | 通算融合中 Matmul/Compute 调优 |
| [../shmem-ops-performance-eval/references/device-profiling-guide.md](../shmem-ops-performance-eval/references/device-profiling-guide.md) | OptimStep 4 性能验证采集 frame 数据 | SHMEMI_PROF 打点操作方法 |
| [../shmem-ops-performance-eval/references/perf-chat-output-spec.md](../shmem-ops-performance-eval/references/perf-chat-output-spec.md) | OptimStep 4 完成后、OptimStep 6 交付前 | **每轮聊天自动输出**：Round Δ% 表、累计总览、最终对比 |
| [../shmem-ops-performance-eval/templates/performance-report.md](../shmem-ops-performance-eval/templates/performance-report.md) | OptimStep 1 读取报告、OptimStep 6 最终确认 | 性能报告模板（由 eval skill 维护） |
| [../shmem-ops-design/references/hardware-architecture.md](../shmem-ops-design/references/hardware-architecture.md) | OptimStep 1 和每轮参数调优时 | 昇腾硬件参数速查（核数、带宽）——调参时 **NEVER** 超出硬件上限 |
| [../shmem-ops-compile-debug/references/custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) | OptimStep 3 委托 compile-debug 编译时 | custom-ops 编译入口命令 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | **读仓内文件前** | 定位 `SHMEM_REPO` |
| [../shmem-ops-dev/references/docker-exec-contract.md](../shmem-ops-dev/references/docker-exec-contract.md) | **用户指定 Docker 时** | 全部 build/run/perf 必须 docker exec，禁止宿主机裸跑 |
| [../shmem-ops-code-gen/SKILL.md](../shmem-ops-code-gen/SKILL.md) | OptimStep 3 委托 code-gen 时 | code-gen 入口与输入契约 |
| [../shmem-ops-dev/references/fused-compute-phase-gates.md](../shmem-ops-dev/references/fused-compute-phase-gates.md) | 通算融合 Phase 6.5 | §5 五轮、§6 HCCL 对比 |
---

## 前置条件

- 算子已通过正确性验证（`shmem-ops-correctness-eval`）
- `design.md` 中 `meta.performance_required: true` 且 `meta.performance_auto_optim: true`
- Phase 6 性能采集已完成且 **未达标**（由 `shmem-ops-dev` 编排器门控后调用本 skill）
- `design.md` 中有明确性能目标

---

## 达标规则

达标线用于评估结论，**NEVER** 作为提前停止条件——进入 Phase 6.5 后 **MUST** 完成全部 5 轮。

| 情况 | 达标线（通信算子） |
| --- | --- |
| 有 baseline（HCCL/已有实现） | **kernel_bus_bandwidth_GBps** 达标线：默认 ≥ baseline 的 **80%** |
| 无 baseline，通信算子 | kernel_bus_bandwidth_GBps 对应带宽利用率 ≥ 20% |
| 无 baseline，通算融合 | 达到 design 指标目标 |

> e2e_latency 仅作参考；Round 间性能变化对比 **MUST** 用 **kernel_bus_bandwidth_GBps**，**NEVER** 用 e2e 带宽。

---

## OptimStep 1：瓶颈定位

**Round 0（首次）**：Round 0 基线数据已由 Phase 6 eval 生成。若 performance_report.md 不存在（异常中断恢复），则调用 eval 补齐 §1-§3。

**Round 1+**：已有前序轮次的性能报告。
1. 读取 performance_report.md，重点看上一轮最优 step 的 Device Frame Table 和 §5 轮次总览
2. 对比当前瓶颈与前序轮次的变化，确认本轮优化方向

### 算法级排查（SHMEM/baseline < 50% 时 **MUST** 执行）

当性能差距极大（与 baseline 相比不到 50%，或无 baseline 但 bus_bandwidth 利用率极低）时，优先检查**算法选择**本身是否匹配硬件拓扑，再进入细粒度排查。

步骤：

1. **确认硬件拓扑**：查阅 [hardware-architecture.md §2.4-2.5](../shmem-ops-design/references/hardware-architecture.md) 确认当前 SoC 的互联拓扑（如 910B3 = 8 卡 full-mesh，每卡 7 条 HCCS 直连）
2. **分析 baseline 算法**：HCCL 在 full-mesh 拓扑下使用 **mesh 算法**，所有 peer 链路并发通信；ring 算法仅在 ring 拓扑下使用
3. **对比当前算法**：当前实现是否利用了拓扑的并发能力？例如 full-mesh 下 peer 串行 get 只用了 N-1 条链路中的 1 条
4. **判断优化方向**：
   - 算法不匹配拓扑 → 机制优化轮应优先做算法级重构（如串行 fetch → 并发 peer 通信 + 通信/计算 overlap）
   - 算法已匹配拓扑 → 进入下方 5 类细粒度排查

输出：在瓶颈列表最前面标注算法级判断。

### 5 类细粒度排查

按以下 5 类系统排查：

| 排查类别 | 指标/现象 | 典型瓶颈信号 |
| --- | --- | --- |
| 通信带宽 | algo_bandwidth 远低于理论带宽 | chunk 太小、engine 选择不当、无 overlap |
| 同步开销 | kernel_us 中大量时间在 barrier/wait | 全局 barrier 过多、signal 粒度太粗 |
| 核利用率 | 部分核空闲或 tail 核负载翻倍 | 分核比例不当、负载不均衡 |
| 内存效率 | 多余 GM scratch 读写、未对齐搬运 | UB 未充分利用、buffer 未对齐 |
| 流水深度 | 通信和计算串行、无 double buffer | 缺少 ping-pong、phase 间无 overlap |

输出：按优先级排列的瓶颈列表，每项标注排查类别和预期收益。

---

## OptimStep 2：基线锁定

确认基线数据（来自 `shmem-ops-performance-eval` 生成的 performance_report.md §3）：

| 指标 | 适用 | 值 |
| --- | --- | --- |
| e2e_latency_us | 全部 | [从 §3.1 获取] |
| kernel_latency_us | 全部 | [从 §3.1 获取] |
| algo_bandwidth_GBps | 全部 | [从 §3.1 获取] |
| kernel_bus_bandwidth_GBps | 全部 | [从 §3.1 获取] |
| bandwidth_utilization_percent | 通信算子 | [从 §3.2 获取] |
| compute_utilization_percent | 通算融合算子 | [从 §3.3 获取]；纯通信标 N/A |

---

## OptimStep 3：输出修改意见与委托执行

**本 skill 禁止自行修改算子代码或 design.md。** 所有改动必须委托对应子 skill 执行。

### 3a. 输出修改意见

根据 OptimStep 1 瓶颈分析，将本轮优化方向分类并输出结构化意见：

| 分类 | 改动范围 | 委托目标 |
| --- | --- | --- |
| **设计级** | 分核策略、调度 phase 重构、算法变更（ring→mesh）、topology 适配 | `shmem-ops-design` 修订 design.md |
| **代码级** | engine 切换、double buffer、signal/wait 替换、block_dim/chunk_size 调整 | `shmem-ops-code-gen` 修改代码 |

每条意见 **MUST** 包含：改什么、为什么（关联 OptimStep 1 的瓶颈）、预期收益（Δ% kernel_bus_bandwidth_GBps）。

机制改动参考（从 `optimization-patterns.md` 或 `compute-optimization.md` 选择）：

| 机制改动 | 说明 | 分类 | 参考 |
| --- | --- | --- | --- |
| Ping-pong double buffer | 增加双缓冲流水，掩盖搬运气泡 | 代码级 | optimization-patterns.md §2.1 |
| Copy-to-symmetric overlap | sender/receiver 分核 + 逐 chunk signal | 代码级 | optimization-patterns.md §1.1 |
| Engine 切换 | MTE→SDMA 或 MTE→RDMA | 代码级 | optimization-patterns.md §1.2 |
| Barrier→signal/wait | 全局 barrier 替换为点对点同步 | 代码级 | optimization-patterns.md §3.1 |
| 通信-计算 overlap | chunk k 通信与 chunk k-1 计算重叠 | 代码级 | optimization-patterns.md §2.2 |
| 分核策略调整 | core partition 比例、send/recv 核数分配 | 设计级 | optimization-patterns.md §5 |
| 算法级重构 | 串行 fetch→并发 peer 通信、ring→mesh | 设计级 | OptimStep 1 算法级排查 |

参数扫描（block_dim、chunk_size、transfer_size 等）属于代码级改动，同样委托 code-gen 执行。每次参数变更记录为一个 **step**（step N.1、N.2...）。

### 3b. 委托执行链

修改意见输出后，按以下顺序委托子 skill 执行。**本轮所有 step 共享同一委托链**——每个 step 变更参数后重新走 code-gen → compile → correctness 循环。

1. **设计级改动**（如有）→ 委托 `shmem-ops-design` 修订 design.md。修订完成后 **跳过 testcase-gen**（测试用例不变），进入步骤 2
2. **代码级改动** → 委托 `shmem-ops-code-gen` 修改代码
3. **编译** → 委托 `shmem-ops-compile-debug` 执行构建与诊断（compile-debug **只诊断不改代码**，编译失败回 code-gen 修复后重试）。编译命令见 [custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1
4. **正确性复测** → 委托 `shmem-ops-correctness-eval`。正确性失败的 step **NEVER** 采纳为性能结论

### 3c. Step 记录规范

每个 step **MUST** 记录：

| 字段 | 说明 | 格式约束 |
| --- | --- | --- |
| Step ID | `轮次.序号` | **MUST** 为 `1.1`、`2.3` 等格式，**NEVER** 使用其他格式 |
| 变更描述 | 具体改了什么 | **MUST** 包含具体参数值（如 `block_dim: 16→32`），**NEVER** 使用笼统描述 |
| Correctness | 正确性结果 | 只允许 PASS / FAIL，**NEVER** 使用模糊值 |
| e2e_us | 端到端延迟 | **MUST** 为数值 |
| kernel_us | 纯 kernel 延迟 | **MUST** 为数值 |
| kernel_bus_bandwidth_GBps | 总线带宽（kernel 口径） | **MUST** 为数值；**Round 对比主指标** |
| algo_bandwidth_GBps | 算法带宽 (GBps) | **MUST** 为数值 |
| e2e_bus_bandwidth_GBps | e2e 总线带宽 (GBps) | 参考；**NEVER** 作 round 对比主指标 |
| compute_utilization% | 计算利用率 | 通算融合算子 **MUST** 填写，纯通信算子标记 N/A |
| 备注 | 观察到的现象 | 如"核利用率从 60%→90%" |

> Step 表列名与 `timing-and-metrics-standard.md` 标准字段对齐。table headers in §4 of performance_report.md MUST use these exact names.

---

## OptimStep 4：性能验证

调用 `shmem-ops-performance-eval` 重新采集性能数据，由 eval skill 将结果写入 performance_report.md 对应 Round 的 §4 子节。

**L 档固定 case**：优化轮次 **MUST** 使用 Round 0 锁定的平台区 `base_count`/dtype/PE，**NEVER** 每轮换 payload（否则不可比）。

### 聊天自动输出（Hard Gate）

每轮最优 step 确定后（`is_optimal_step=true`），eval 与 optim **MUST** 在聊天自动输出（无需用户追问）：

1. **本轮 Step 摘要表**（多 step 时）
2. **Round N 最优 vs Round0 / vs 上轮最优**（含 kernel_bus_bandwidth_GBps Δ%）
3. **累计轮次总览表**（0~N 行，每轮结束后更新）

模板见 [perf-chat-output-spec.md §3](../shmem-ops-performance-eval/references/perf-chat-output-spec.md)。**NEVER** 只在 Round 5 结束才输出变化。

### 调用上下文（optim → eval）

optim 调用 eval 时 **MUST** 附带以下上下文，使 eval 知道写入报告的位置：

| 参数 | 含义 | 值 |
| --- | --- | --- |
| `target_section` | 写入目标区域 | `round_0`（基线采集）、`round_N_step_M`（如 `round_1_step_3`） |
| `round_label` | 轮次标签 | `Round 0 基线`、`Round 1 机制优化：并发 peer put`、`Round 2 机制优化：double buffer` |
| `is_optimal_step` | 是否为该轮最优 step | `true`（需输出 Device Frame）、`false`（仅填 step 表行） |

**eval skill 根据上下文写入的位置**：

| target_section | is_optimal_step | 写入内容 |
| --- | --- | --- |
| `round_0` | — | §2 Baseline 详细信息 + §3.1~§3.5 L 档基线数据 + S 档数据（供 §6.2） |
| `round_N_step_M` | `false` | 在 §4 Round N 的 Step 表中追加一行（仅 L 档） |
| `round_N_step_M` | `true` | Step 表行 + 性能验证对比表 + Device Frame Table（均仅 L 档） |
| `round_N_step_M` | `true`（且为最终轮次） | 上述内容 + §5 轮次总览行 + §6.1 L 档对比 + §6.2 S 档对比 + §7 结论 |

### 规则
- 使用与基线相同的 case、参数和环境
- **S 档仅在 Round 0 采集一次**，§4 优化轮（Round 1~5）**NEVER** 记录 S 档数据，仅记录 L 档
- eval skill **MUST** 将以下内容写入 performance_report.md §4 对应 Round：
  1. **性能验证对比表**（Baseline vs Round N 最优 | Δ%）—— 简表格式
  2. **Device Frame Table**（最优 step 的 frame 数据）—— 若该轮包含多个 step（参数扫描），只输出最优 step 的 Device Frame
- 纯通信算子：不需要输出 compute_utilization_pct
- 通算融合算子 **MUST** 同时输出 compute_utilization_pct
- **MUST** 输出与基线的 Δ% 对比

optim skill 从 OptimStep 5 得到的结构化数据格式：

**性能验证对比表**（由 eval 填入 §4 各 Round 末尾）：

| 指标 | Baseline | Round N 最优 | Δ% |
| --- | --- | --- | --- |
| kernel_bus_bandwidth_GBps | | | |
| e2e_latency_us | | | |
| kernel_latency_us | | | |
| algo_bandwidth_GBps | | | |
| e2e_bus_bandwidth_GBps | | | |
| compute_utilization_pct | | | |

> compute_utilization_pct 行仅通算融合算子填写；纯通信标 "N/A"

**Device Frame Table**（由 eval 填入 §4 各 Round "Device Frame（最优 step）" 子节）：

| Frame | Phase | Avg us | Max core us | Count | % E2E | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- |

> 若该轮仅一个 step，取该 step 的 frame 数据；若多个 step，取最优 step 的 frame 数据。

---

## OptimStep 5：迭代决策

| 决策 | 条件 |
| --- | --- |
| keep | 本轮最优 step 性能提升且正确性通过 → 采纳该 step 配置作为新基线 |
| revert | 本轮所有 step 性能退步或正确性失败 → 恢复上一轮最优配置 |

决策后：若轮次 < `max_opt_rounds`（默认 5）→ 回 OptimStep 1 开始下一轮（达标 **不是** 提前停止理由；用户明确指定轮次时 MUST 跑满指定轮次）。

---

## 停止条件

- 已完成 `max_opt_rounds`（默认 5）轮（**唯一正常停止条件**——达标 **不是** 提前停止理由；用户明确指定轮次时跑满指定轮次，即使已达标）
- 用户明确要求继续优化 → 可超过指定轮次，每额外轮次仍遵循完整的 OptimStep 1-6 流程，并在 performance_report.md 中标注"用户要求的额外优化轮次"；模板 §4/§5 按 Round 2 格式续写，无上限
- 进一步优化需修改 SHMEM 核心库但无 gap analysis 授权（异常停止，需在报告中说明）

---

## OptimStep 6：最终确认与格式走读

所有轮次完成后（无论达标与否），按以下步骤最终确认：

1. **格式走读**：逐节对照 [模板](../shmem-ops-performance-eval/templates/performance-report.md) 检查 performance_report.md：

| 检查项 | 内容 |
| --- | --- |
| 节号与模板一致 | §1~§7 完整，无自定义新增章节（如"采集命令"、"指标公式"、"正确性回归"） |
| 节名与模板一致 | 使用模板定义的标准节名（"算子信息"、"Baseline 详细信息"等），不自行重命名 |
| 表结构一致 | 列名、列数、列顺序与模板完全相同 |
| §3.2 / §3.3 适用性标注 | 通信算子 §3.3 标注 "N/A"，通算融合 §3.2 不得标 N/A |
| Device Frame Table | §3.4 和 §4 每个 Round 最优 step 均存在，无空表 |
| §4 每个 Round 结构完整 | Step 表 + 性能验证对比表 + Device Frame + 本轮最优配置 + 决策 |
| §5 轮次总览 | 行数与实际轮次数一致，含 Round 0 |
| compute_util% 列 | 纯通信算子标 "N/A"，通算融合算子必须填写 |

2. **完整性确认**：report **MUST** 包含以下全部内容 → 对应模板位置：

1. 算子信息（名称、shape、dtype、PE 数、目标指标）→ §1
2. Baseline 详细信息 → §2
3. 基线数据（Round 0 的完整指标 + 通信指标表 + 计算指标表 + Device Frame Table + 性能对比表）→ §3
4. 每轮每步的完整记录（Step 表 + 性能验证对比表 + Device Frame Table）→ §4 各 Round
5. 轮次总览表 → §5
6. 最终对比表（baseline vs final，含 Δ%）→ §6
7. 结论（达标/未达标 + 瓶颈总结 + 优化路径）→ §7

3. **交付**：确认通过后输出 `docs/performance_report.md` 路径。

performance_report.md 由 `shmem-ops-performance-eval` 在每轮执行时维护，位于算子目录 `docs/performance_report.md`。

---

## 输出

### 两级记录表

每轮记录（Round 级）：

| Round | 性质 | 机制改动 | 最优 Step | 最优 kernel_bus_bandwidth_GBps | vs R0 | vs baseline | 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | — | — | ... | ... | ... | — |
| 1 | 机制优化 | 并发 peer 通信 | 1.3 | ... | ... | ... | keep |
| 2 | 机制优化 | 加 ping-pong double buffer | 2.2 | ... | ... | ... | keep |

> compute_util% 列仅通算融合算子填写；纯通信标 N/A

每步记录（Step 级，嵌套在 Round 内）：

| Step | 变更 | Correctness | kernel_bus_bandwidth_GBps | e2e_us | kernel_us | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | block_dim: 16→32 | PASS | ... | ... | ... | ... | N/A | 核利用率提升 |
| 1.2 | block_dim: 32→48 | PASS | ... | ... | ... | ... | N/A | 无明显提升 |
| 1.3 | chunk_size: 64KB→190KB | PASS | ... | ... | ... | ... | N/A | 带宽提升 20% |

### performance_report.md

最终交付 **MUST** 在算子目录下生成 `docs/performance_report.md`，由 `shmem-ops-performance-eval` 按模板格式维护。

---

## 反模式（NEVER DO THESE）

- ❌ `performance_auto_optim: false` 时擅自进入本 skill 或改 kernel
- ❌ 将 Host RMA → Device kernel 当作性能优化（这是功能完善）
- ❌ 无瓶颈分析就随机改参数
- ❌ 未经用户明确要求超过 5 轮（用户主动要求继续不受此限制）
- ❌ 未完成 5 轮就声称优化完成（达标不是停止理由）
- ❌ 正确性未复测就采纳性能结果
- ❌ 未证明 ≥5% 收益就引入或恢复 `big_data`/`large` 与 unified 路径并行维护
- ❌ 将 UB 内部分块误判为「大小两套实现」从而拒绝合并代码
- ❌ Round 1 仅扫 block_dim 而无机制改动（除非机制已全部穷尽且文档说明）
- ❌ 不输出 performance_report.md 就声称优化完成
- ❌ step 记录中省略具体参数值或度量结果
- ❌ 将用户 input 直接构造在对称内存（symmetric memory / GVA）上以跳过输入到对称堆的拷贝来提升性能。真实使用场景中，用户 input 来自 Device GM buffer，**MUST** 经过 `输入到对称堆的拷贝 → symmetric memory → kernel` 的完整流程。跳过输入到对称堆的拷贝的"优化"只是在测试中作弊，不反映实际部署性能
- ❌ 将输入到对称堆的拷贝移到性能循环外（如循环前做一次预拷贝），导致 `e2e_latency_us` 不包含输入到对称堆的拷贝时间。e2e 性能循环 **MUST** 每轮都执行 `input(GM) → 输入到对称堆的拷贝 → symmetric memory → barrier → kernel` 完整流程
- ❌ performance_report.md 使用模板 §1~§7 之外的节号（如 §8、§9、§10）或自定义节名（如"采集命令"、"指标公式"、"正确性回归"）
- ❌ 自行修改算子代码（`.cpp`/`.h`/`CMakeLists.txt`）——必须委托 `shmem-ops-code-gen`
- ❌ 自行修改 `design.md` ——必须委托 `shmem-ops-design`
- ❌ Phase 6.5 只在最后一轮才在聊天报性能变化
- ❌ Round 间用 e2e 带宽对比（应使用 kernel_bus_bandwidth_GBps）
- ❌ 优化轮次更换 payload 规模导致不可比
- ❌ §4 优化记录只输出一个扁平表格而不按 Round 分子节（每个 Round 必须包含 Step 表 + 性能验证 + Device Frame + 最优配置 + 决策）

---

## 最佳实践

- ✅ 先定位瓶颈再选优化手段
- ✅ 每个 step 记录具体参数值和 6 指标结果
- ✅ 正确性通过后再看性能数据
- ✅ 退步时 revert 而非继续叠加
- ✅ 机制优化轮（Round 1~5） **MUST** 包含代码级改动
- ✅ 固定完成 5 轮优化，达标后继续挖掘优化空间
- ✅ 最终输出 performance_report.md（由 eval skill 按模板维护）
- ✅ 最终对比表包含 Δ%

---

## MUST检查

- [ ] 完成了全部 5 轮优化（未因达标提前停止）
- [ ] Round 0：已调用 shmem-ops-performance-eval 生成基线报告 §1-§3
- [ ] 所有代码/设计改动均通过委托子 skill 执行，optim 未自行修改文件
- [ ] OptimStep 1 有明确的瓶颈分析
- [ ] Round 1~5 每轮有代码级机制改动（纯参数扫描不得单独构成一轮）
- [ ] 每个 step 记录了具体参数值和 6 指标结果（含 compute_util%）
- [ ] step 记录格式约束已遵守（Step ID 格式、变更描述含具体值、Correctness 只允许 PASS/FAIL、数值不允许空值）
- [ ] 正确性在每个 step 验证
- [ ] 每轮性能验证已调用 shmem-ops-performance-eval
- [ ] 每轮性能验证输出了对比表 + Device Frame Table
- [ ] 所有轮次和步骤记录在 performance_report.md 中
- [ ] 每轮最优 step 后聊天已自动输出 Round Δ% 表（perf-chat-output-spec §3）
- [ ] 5 轮结束后聊天已输出最终对比表（perf-chat-output-spec §4）
- [ ] 全部轮次使用 Round 0 锁定的 L 档平台区 case
- [ ] 最终对比表包含双指标延迟（e2e_us + kernel_us）+ compute_utilization_pct
- [ ] 输入到对称堆的拷贝始终在 e2e 性能循环内（未经任何轮次优化移出循环）
- [ ] performance_report.md 已按模板格式生成（由 eval skill 维护）
