---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-performance-eval
description: 基于 SHMEM 的算子性能采集、baseline 对比、瓶颈分析和性能报告生成。触发：用户需要采集 SHMEM 算子时延/带宽、对比 HCCL
  或 aclnn baseline 时使用。关键词：基于SHMEM性能评估、性能采集、performance、baseline、bandwidth、性能报告。
original-name: shmem-ops-performance-eval
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子性能采集

**Skill类型**：评测型（采集性能数据，对比 baseline，将结构化数据写入 performance_report.md）

> **中文写作要求**：性能报告 **MUST** 使用中文撰写。仅 API 名称、代码片段、指标名（如 `e2e_latency_us`）等技术术语保留英文原文。

正确性验证通过后执行。**仅当** `design.md` `meta.performance_required: true`（Phase 0 #4 用户已确认）时由编排器调用 Phase 6；**NEVER** 在用户未要求性能工作时擅自采集。

性能**优化**（Phase 6.5）**仅当** `meta.performance_auto_optim: true` **且 Phase 6 未达标**时由编排器在 Phase 6 后自动调用 `shmem-ops-performance-optim`（进入后 **MUST** 跑满 `max_opt_rounds` 默认 5 轮，达标 **不是** 提前停止理由；用户明确指定优化轮次时跑满指定轮次，即使 Round 0 已达标）；`performance_auto_optim: false` 时 eval 只采集与报告，**禁止**擅自改 kernel。

采集性能数据、接入 baseline、分析瓶颈，将结构化数据写入 performance_report.md（模板：[templates/performance-report.md](templates/performance-report.md)）。**统计指标字段保持既有定义不变**（如 `e2e_us`、`kernel_us`、`algo_bandwidth_GBps`、`kernel_bus_bandwidth_GBps` 等）；采集方式扩展为：SHMEM 保留 `source=inline_event`（原有前后打点 `[PERF]`）的完整指标，其中 `e2e_us` 只使用该口径；同时额外用 `source=msprof_task_time` 采集 SHMEM `kernel_us`。baseline 只要求 msprof 口径。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/performance-eval-guide.md](references/performance-eval-guide.md) | 采集流程概览、contract 字段、结果表格式、规模要求 |
| [references/baseline-selection.md](references/baseline-selection.md) | baseline 选择策略、HCCL/aclnn 清单、决策树；C++ 模板、CMake 集成 |
| [references/baseline-compare-workflow.md](references/baseline-compare-workflow.md) | baseline C++ 接入流程、日志标签约定、离线对比 |
| [references/perf-workflow.md](references/perf-workflow.md) | **三阶段采集命令代码段**（baseline + SHMEM + 离线对比） |
| [references/perf-chat-output-spec.md](references/perf-chat-output-spec.md) | **聊天自动输出规范** |
| [references/timing-and-metrics-standard.md](references/timing-and-metrics-standard.md) | **性能评估唯一标准**：时间打点、双指标方案、指标计算公式、Device 打点、计算指标 |
| [references/profiling-tools.md](references/profiling-tools.md) | msprof、SHMEM cycle profiling、warmup 规范（前 N 轮不计入统计）、多 PE 取平均 |
| [references/device-profiling-guide.md](references/device-profiling-guide.md) | SHMEMI_PROF 完整操作指南（kernel 打点 → Host 导出 → 填报告 → 瓶颈分析） |
| [templates/performance-report.md](templates/performance-report.md) | 性能报告模板（eval 写入数据的格式规范） |
| [../shmem-ops-design/references/hardware-architecture.md](../shmem-ops-design/references/hardware-architecture.md) | 昇腾硬件参数速查（计算带宽利用率时 **MUST** 参考） |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO`（读仓内和外链路径如 `custom-ops/<op>/` 前） |
| [../shmem-ops-dev/references/fused-compute-phase-gates.md](../shmem-ops-dev/references/fused-compute-phase-gates.md) | 通算融合：性能采集与 baseline（§3、§6） |

## 输入

- design.md 的 perf contract（metric、baseline、target_cases、最小规模）
- 已通过正确性验证的算子二进制
- 算子已包含原有 `--perf` 前后打点和 `[PERF]` 输出，同时可被 msprof 包裹（由 `shmem-ops-code-gen` 按 timing-and-metrics-standard.md 生成）
- 调用上下文（由 `shmem-ops-performance-optim` OptimStep 5 传入）：

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `target_section` | 写入目标区域 | `round_0`（基线采集）、`round_1_step_3`（Round 1 step 3） |
| `round_label` | 轮次标签 | `Round 0 基线`、`Round 1 机制优化：并发 peer put`、`Round 2 机制优化：double buffer` |
| `is_optimal_step` | 是否为该轮最优 step | `true`（需输出 Device Frame Table）、`false`（仅填 step 表行） |

> 独立调用 eval 时（非 optim 驱动），默认 `target_section = round_0`，仅写入 §2 + §3 + §6.2 baseline 列。

### 工作流

```
步骤 1  选择 baseline + 确认 baseline C++ 代码已由 code-gen 生成、compile-debug 编译通过 + 用 msprof 采集 baseline task_time
步骤 2  运行 SHMEM perf：保留 `[PERF]` 打点输出，同时用 msprof 包裹采集 task_time + 离线解析/对比
步骤 3  瓶颈分析 + 写入 performance_report.md + 聊天自动输出
```

所有命令代码段见 [perf-workflow.md](references/perf-workflow.md)；聊天输出格式见 [perf-chat-output-spec.md](references/perf-chat-output-spec.md)。

> **路径说明**：`custom-ops/<op>/` 类路径均相对于 `SHMEM_REPO`（定位规则见 [shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md)），且为**本 skill 工作流生成**（非 upstream 自带，**禁止**当作参考实现引用）。命令代码段唯一参考：[perf-workflow.md](references/perf-workflow.md)。

---

## 步骤 1：Baseline 选择与实现

**MUST** 按以下顺序逐步查找，声称"无 baseline"前每步都要排查（详见 [references/baseline-selection.md](references/baseline-selection.md)）：

1. **HCCL 集合通信算子**：AllGather、AllReduce、ReduceScatter、AllToAll、Broadcast、Reduce、Scatter、Gather（完整清单见 baseline-selection.md §1.1）
2. **aclnn 扩展算子**：aclnnAlltoAllV 等（完整清单见 baseline-selection.md §1.2）
3. **指标测试（无直接 baseline）**：仅测量利用率指标（见 baseline-selection.md §2）

`peak_bandwidth` 决策表见 [timing-and-metrics-standard.md §4.4](references/timing-and-metrics-standard.md)。

### Baseline 实现要求

- baseline C++ 代码 **MUST** 由 `shmem-ops-code-gen` 生成（由 dev 协调）；编译 **MUST** 通过 `shmem-ops-compile-debug` 完成；本 skill 负责运行、采集、对比
- 实现模板和 CMake 集成见 [baseline-selection.md §1.3](references/baseline-selection.md)；baseline runner 只负责稳定执行 warmup/measure，最终性能由 msprof 解析得到
- 源码与脚本 **MUST** 放在 `custom-ops/<op>/baseline/` 下，算子根 `CMakeLists.txt` 通过 `add_subdirectory(baseline)` 编译
- 产物布局见 [baseline-compare-workflow.md §2](references/baseline-compare-workflow.md)

### Baseline 采集

- **MUST** 使用与 SHMEM 相同的 case 和环境
- 采集命令见 [perf-workflow.md §1 阶段 A](references/perf-workflow.md)
- AllToAllV 类算子：CANN 无独立 `aclnnAlltoAllV` 头文件时，**MUST** 回退 `HcclAlltoAllV` 并记录原因（见 [baseline-compare-workflow.md §5](references/baseline-compare-workflow.md)）
- **MUST** 分三阶段执行（baseline → SHMEM → 离线对比），**NEVER** 同 shell 混跑（见 [perf-workflow.md](references/perf-workflow.md)）

### 通算融合算子附加规则（`op_kind=fused_compute_comm`）

见 [fused-compute-phase-gates.md §3、§6](../shmem-ops-dev/references/fused-compute-phase-gates.md) 与 [baseline-selection.md §1.2 / §2](references/baseline-selection.md)：

1. **MUST** 采集 **S 档 + L 档**；PE 数**推荐 8PE**（不强制），资源不足可用 4PE/2PE 并在报告标注（[performance-eval-guide.md §4](references/performance-eval-guide.md)）
2. baseline **MUST** 按优先级：P1 aclnn fused → P2 stitched（计算+HCCL）→ P3 metric_only；**禁止**有 P1 却用单一 HCCL 代替全流程
3. **MUST** 输出端到端性能表（SHMEM vs baseline）；分段耗时（matmul/comm）为**可选诊断**，**不**强制 phase 拆解或单独 `phase_timing_report.md`
4. 未实现 P1/P2 baseline（或未尽 P3 搜索）或未写 S+L 对比表 → Phase 6 未完成
5. `proxy baseline` 仅可用于阻塞定位；**MUST NOT** 作为 Phase 6 达标依据；报告必须标注 `blocked`

---

## 步骤 2：SHMEM 性能采集

### 前置：确认打点 + msprof runner

SHMEM 算子的 perf runner **MUST** 由 `shmem-ops-code-gen` 生成（由 dev 协调）。执行采集前 **MUST** 确认：

- `main.cpp` 包含 `--perf` flag、warmup、measure 循环和 `[PERF]` 输出逻辑
- `[PERF]` 行保留既有字段：`e2e_us`、`kernel_us`、`algo_bandwidth_GBps`、`e2e_bus_bandwidth_GBps`、`kernel_bus_bandwidth_GBps`、`bandwidth_utilization_pct`、`payload_bytes`、`bus_factor`、`peak_bandwidth_GBps`
- `scripts/perf.sh` 存在，可直接运行得到 `[PERF]`，也可被 `msprof --task-time=l2 ... bash scripts/perf.sh ...` 直接包裹
- kernel 包含 `SHMEMI_PROF_START/END` 宏对（至少 5 个 `frame_id`），Host 侧可调用 `aclshmemx_get_prof(nullptr, true)` 导出 frame 数据

若未包含，由 dev 编排器调用 `shmem-ops-code-gen` 添加打点 + msprof-compatible runner 后重新编译再采集。

### 采集命令

**MUST** 按 [perf-workflow.md §1 阶段 B](references/perf-workflow.md) 执行 SHMEM 采集：同一次 runner 输出 `[PERF] source=inline_event`，同时由 msprof 生成 `task_time_*.csv`。随后按 [perf-workflow.md §1 阶段 C](references/perf-workflow.md) 解析 `[PERF]` 的完整指标和 msprof profile 的 `kernel_us` 指标。采集规范（warmup、repeats、多 PE envelope）见 [profiling-tools.md](references/profiling-tools.md)。

### 时间打点与指标

时间数据有两种来源。本 skill 负责验证输出：确认 `source=inline_event` 来自 `[PERF]` 行，包含完整既有字段（`e2e_us`、`kernel_us`、带宽、payload/bus_factor/peak_bandwidth 等）；确认 `source=msprof_task_time` 来自 `task_time_*.csv`，只补充 `kernel_us` 及基于 msprof `kernel_us` 计算的 kernel 口径带宽/利用率。baseline 只要求 msprof 口径。

与 baseline 对比时 **主对比使用 SHMEM msprof kernel_us / kernel_bus_bandwidth_GBps vs baseline msprof kernel_us / kernel_bus_bandwidth_GBps**；SHMEM inline_event 的 `e2e_us` 和 `kernel_us` 作为保留指标并列展示，用于延续历史口径和定位差异。指标计算公式见 [timing-and-metrics-standard.md §4](references/timing-and-metrics-standard.md)。

### MUST 记录字段

| 字段 | 内容 |
| --- | --- |
| case | shape、dtype、PE count、engine、scope |
| command | 完整运行命令、msprof 命令、runner 参数和环境 |
| profile | baseline/SHMEM 的 msprof 输出目录、`task_time_*.csv` 路径、解析过滤规则 |
| inline_event | SHMEM `[PERF]` 原始日志路径、每个字段解析值 |
| metric | SHMEM inline_event: e2e_us、kernel_us、algo_bandwidth_GBps、e2e_bus_bandwidth_GBps、kernel_bus_bandwidth_GBps、bandwidth_utilization_pct；SHMEM msprof_task_time: kernel_us、kernel_bus_bandwidth_GBps、bandwidth_utilization_pct；baseline msprof: kernel_us、kernel_bus_bandwidth_GBps |
| device frame | phase、cycles、avg_us、percent_of_e2e（采集方法见 [device-profiling-guide.md](references/device-profiling-guide.md)） |
| compute | effective_flops、compute_utilization_percent（标注 N/A） |

注意：`algo_bandwidth` **NEVER** 乘 2，统一按 input size 计算（NCCL `algBw` 惯例）。

### 规模要求

性能 case 要求（规模分档见 [shmem-ops-testcase-gen/references/testcase-scale-standard.md](../shmem-ops-testcase-gen/references/testcase-scale-standard.md)）：
- 性能采集 **MUST** 覆盖 **S 档 + L 档** 两种规模，**NEVER** 只采 L 档；PE 数：通信算子 **8PE 必测**，通算融合 **8PE 推荐**（见上「通算融合算子附加规则」）
- L 档：集合通信 ≥256MB 数据量（全 PE 总量）
- S 档：按 testcase-scale-standard.md 定义
- **NEVER** 只用 smoke 小 shape

有 baseline 时，计算 delta（当前实现 vs baseline）和利用率；无 baseline 时，计算 metric_only 达标判断。

---

## 步骤 3：瓶颈分析与报告写入

### 瓶颈分析

基于 profiler frame table 判断（分析方法论见 [device-profiling-guide.md §6.2](references/device-profiling-guide.md)）：

- 哪个 phase 占比最高 → 对应优化方向
- `max_core_us / avg_us` 比值 → 是否存在长尾 block/PE
- signal_wait / barrier 占比 → 同步效率是否合理
- 通信 frame 的 algo_bw / peak_bw → 带宽利用率是否达标
- 是否存在不合理的 wait/sync 占比

### 写入报告

根据 `target_section` 和 `is_optimal_step` 决定写入 `docs/performance_report.md` 的位置：

| target_section | is_optimal_step | 写入位置 | 具体内容 |
| --- | --- | --- | --- |
| `round_0` | — | §2 + §3.1~§3.5 + §6.2 baseline 列 | Baseline 详细信息、L 档性能结果/通信/计算/Device Frame/性能对比表；S 档 baseline 直接写入 §6.2 baseline 列，final 列暂空 |
| `round_N_step_M` | `false` | §4 Round N Step 表 | 在已有 Step 表中追加一行（以 L 档数据为准） |
| `round_N_step_M` | `true` | §4 Round N | Step 表行（以 L 档为准）+ 性能验证对比表 + Device Frame Table |
| `round_N_step_M` | `true`（最终轮） | §4 + §5 + §6 + §7 | 上述 + §5 轮次总览行 + §6.1 L 档对比 + §6.2 S 档 final 列（**MUST 重新采集 S 档**）+ §7 结论 |
| — | 仅 S 档 | §6.2 | S 档在 Round 0 直接写入 §6.2 baseline 列；最终轮后 **MUST 重新采集** S 档数据（非复用 Round 0 数据），填入 final 列与 Δ%。**NEVER** 写入 §4 优化记录 |

**节号强制约束**：写入时 **MUST** 使用模板定义的固定节号（§1~§7）和节名。以下行为 **NEVER** 允许：
- 在 §1~§7 之外新增自定义章节（如"采集命令"、"指标公式"、"正确性回归"）
- 辅助信息归入已有章节：采集命令 → §2 Baseline 详细信息；指标公式 → §3.2/§3.3 表头备注
- §4 每个 Round **MUST** 包含完整子结构：Step 表 → 性能验证对比表 → Device Frame Table → 本轮最优配置 → 决策

### 聊天自动输出（Hard Gate）

脚本跑完后 **MUST** 在同轮对话自动输出带宽表 + 时延表，**NEVER** 只报时延或等用户追问。输出格式和模板见 [perf-chat-output-spec.md](references/perf-chat-output-spec.md)。

---

## 输出字段规范

eval 输出以下结构化数据，写入 performance_report.md。缺失任一项视为不完整。

### MUST 输出

| 输出项 | 字段 | 说明 |
| --- | --- | --- |
| Baseline 信息 | type, source_api, search_process, measurement | 逐步排查记录、baseline msprof 测试数据 |
| 性能结果表 | case_id, shape, dtype, n_pes, e2e_us, kernel_us_inline, kernel_us_msprof, algo_bandwidth_GBps, kernel_bus_bandwidth_GBps_inline, kernel_bus_bandwidth_GBps_msprof, utilization% | 每个采集 case 的完整指标；SHMEM 展示 inline 与 msprof kernel 两种口径 |
| 通信指标 | logical_payload_bytes, bus_factor, peak_bandwidth_GBps, bandwidth_utilization_percent | 通信算子 **MUST** |
| 计算指标 | op_count_flops, compute_latency_us, effective_flops, peak_flops, compute_utilization_percent | 标注 "N/A" |
| Device Frame Table | frame_id, phase, avg_us, max_core_us, count, percent_of_e2e, bottleneck_note | **MUST** 存在；采集方法见 [device-profiling-guide.md](references/device-profiling-guide.md) |
| 瓶颈判断 | 主要耗时来源、带宽利用率是否达标、计算利用率是否达标 | 基于 frame table 分析（方法论见 [device-profiling-guide.md §6.2](references/device-profiling-guide.md)） |
| Delta vs Baseline | kernel_bus_bandwidth_GBps 对比、达标判断 | 有 baseline 时 **MUST** |
| 聊天自动输出 | 带宽表+时延表、（optim 驱动时）轮次 Δ% 表 | **MUST**，见 perf-chat-output-spec.md |

### 调用方差异

| 调用方 | target_section | 聊天 MUST 额外输出 |
| --- | --- | --- |
| Phase 6 独立 eval | `round_0` | §2 perf-chat-output-spec：S+L 表 + 对比表 |
| optim OptimStep 5（每 step） | `round_N_step_M` | 无（仅写报告 Step 行） |
| optim OptimStep 5（round 最优） | `round_N_step_M` + `is_optimal_step=true` | §3 perf-chat-output-spec：本轮最优表 + 更新累计总览 |
| optim 全部完成 | 最终轮最优 | §4 perf-chat-output-spec：最终对比表 |

### 输出格式

性能结果以表格形式在聊天界面展示，关键指标 **MUST** 摘要输出，**NEVER** 仅输出路径。

**有 baseline 时，聊天回复 MUST 包含带宽表与时延表**（kernel_bus_bandwidth_GBps、比值、目标、达标），格式见 [perf-chat-output-spec.md §2](references/perf-chat-output-spec.md)。**NEVER** 只粘贴单行 `[PERF]`、只报 SHMEM 单边，或只报 e2e 时延。

示例（结构示意，数值须来自实际采集）：

| 指标 | SHMEM | HCCL baseline | SHMEM/baseline |
| --- | --- | --- | --- |
| bus_GBps (M_fp16_8pe) | 30.86 | 10.61 | 290.7% PASS |

---

## MUST检查

- [ ] baseline 已选择或记录搜索过程
- [ ] baseline C++ 程序已由 code-gen 生成、compile-debug 编译通过，且 baseline msprof profile 已采集并解析（baseline 不要求 `[BASELINE_PERF]`）
- [ ] SHMEM `[PERF] source=inline_event` 已采集并解析，字段保持既有指标不变
- [ ] SHMEM msprof profile 已采集并解析为 `kernel_us_msprof` / `kernel_bus_bandwidth_GBps_msprof`（Round 0: S 档 + L 档；中间轮次: L 档；最终轮: S 档 + L 档）
- [ ] 瓶颈已分析（基于 Device Frame Table；见 [device-profiling-guide.md §6.2](references/device-profiling-guide.md)）
- [ ] 输出字段规范中全部 **MUST** 项已输出
- [ ] `docs/performance_report.md` §3.5 对比表已填写
- [ ] 聊天已按 [perf-chat-output-spec.md](references/perf-chat-output-spec.md) 自动输出（非用户追问才补）
- [ ] optim 驱动时：每轮最优 step 后已输出 Round Δ% 表与累计总览

---

## 反模式（NEVER DO THESE）

- ❌ 有可用 baseline 却不接入
- ❌ 只采集时间不分析瓶颈
- ❌ 用 smoke 小 shape 作为性能结论
- ❌ 没有硬件却编造性能数据
- ❌ 通信算子不测量带宽利用率
- ❌ 只输出 algo_bandwidth 不输出 bus_bandwidth 和 utilization
- ❌ 报告中缺少 Device Frame Table
- ❌ baseline 字段写 `"none"` 但不记录搜索过程
- ❌ 输出指标不完整
- ❌ 写入 performance_report.md 时新增模板未定义的章节（如"采集命令"、"指标公式"、"正确性回归"），所有信息必须归入 §1~§7 对应子节
- ❌ custom-ops 性能采集在文档/回复中首选裸 `cd custom-ops/<op> && ...`（用 [perf-workflow.md](references/perf-workflow.md)）
- ❌ 性能测试完成后不在聊天中自动输出结果，等用户引导
- ❌ Phase 6.5 只在第 5 轮结束才报性能变化
- ❌ 只报 SHMEM `[PERF]`，不报 SHMEM msprof 与 baseline msprof 对比表
- ❌ 删除、改名或遗漏 SHMEM `[PERF]` 原有字段（`e2e_us`、`kernel_us` 等）
- ❌ baseline 依赖 `[BASELINE_PERF]` 才能完成；baseline 最终必须能只通过 msprof profile 得到指标
- ❌ 只保存 msprof 目录但不解析 `task_time_*.csv`
- ❌ 无离线对比流程却声称已完成 baseline 对比
- ❌ **同一 shell 内混跑 HCCL baseline 与 SHMEM perf**（须分阶段 + 离线 compare）
- ❌ AllToAllV 未排查 `HcclAlltoAllV` 就直接写 metric_only
- ❌ baseline 与 SHMEM 使用不同 gen_data / sendcounts
- ❌ **通算融合**：只报 SHMEM `[PERF]`、无 SHMEM msprof vs baseline msprof 对比即声称 Phase 6 完成
- ❌ **通算融合**：存在 AllToAll/AllReduce 等 HCCL 原语却选 metric_only
- ❌ 用 `proxy baseline` 数据替代真实 HCCL/aclnn/CATLASS 拼接对比并给出“达标/完成”结论
