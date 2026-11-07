# 通算融合算子开发规范

## 适用范围

本规范适用于 `meta.op_kind: fused_compute_comm` 的算子（Cube MatMul/GEMM + SHMEM 跨 PE 通信，CoC 模式）。

编排器在判定为通算融合后 **MUST** Read 本文件，并与 `shmem-ops-dev/SKILL.md` Phase 0–7 工作流一并执行。本规范中的 **MUST / MUST NOT / NEVER** 为 Hard Gate，优先级不低于主编排器。

---

## 1. Phase 门控

通算融合算子 **MUST** 按序完成各 Phase，**MUST NOT** 跳阶段、补文档式交付或 smoke 通过后直接进入性能/交付。

| Phase | 子 skill | 进入条件 | 必须产出 | 未满足时 |
| --- | --- | --- | --- | --- |
| 0 | — | 用户 @ `shmem-ops-dev` | `phase0_intake`（五项 intake 信息，先探测缺项再问） | **停止** |
| 1 | `shmem-ops-design` | Phase 0 完成 | `docs/design.md`（Canonical DSL、`schedule.compute`、baseline 声明） | 不得 codegen |
| 2 | `shmem-ops-testcase-gen` | Phase 1 门禁 PASS | `cases.json`、`run_case_matrix.py`、`gen_data`/`check_result`/`run.sh` | 不得 codegen |
| 3 | `shmem-ops-code-gen` | Phase 2 完成 | `src/`、`CMakeLists.txt`、`README.md`；`baseline/` 骨架（§6，优先 P1 aclnn fused） | 不得编译验收 |
| 4 | `compile-debug` + `correctness-eval` | Phase 3 编译成功 | 全量 case matrix PASS（§2）、`correctness_report.md`、`case_matrix_report.md` | **不得 Phase 5** |
| 5 | `shmem-ops-code-review` | Phase 4 全 case PASS | `review-report.md`（interim） | **不得 Phase 5.5/6** |
| 5.5 | `shmem-ops-torch-bind` | `torch_required: true` | `aclshmem_torch.so` + 测试 | 跳过须记录 |
| 6 | `shmem-ops-performance-eval` | `performance_required: true` 且 Phase 5 PASS | `performance_report.md`、baseline 对比（§6）、聊天带宽+时延表 | **不得 Phase 6.5/7** |
| 6.5 | `shmem-ops-performance-optim` | Phase 6 未达标 + `performance_auto_optim: true` | Round 1~5 全部完成（§5） | 未满 5 轮不得 Phase 7 |
| 7 | 编排器 | 上述条件满足 | `delivery_summary.md`、review final | — |

**编排器自检**：进入 Phase N 前 **MUST** 列出 Phase N−1 检查点并逐项勾选；任一项未勾选 **MUST NOT** 进入下一阶段。

**禁止行为**：
- ❌ 先写代码后补 `design.md` / `review-report.md`
- ❌ 跳过 Phase 5 走读直接进入性能或交付
- ❌ 缺 §7 交付物清单中任一项即声称 Phase 7 完成

---

## 2. 正确性门控（Case Matrix）

### 2.1 执行要求

- `cases.json` 中 **category ≠ performance** 的 case **MUST** 全部执行
- **MUST** 运行：`python3 scripts/run_case_matrix.py`（或 [custom-ops-entrypoints.md §3](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)）
- **MUST NOT** 仅用 `CASE_ID=smoke_* bash scripts/run.sh` 代替 matrix 验收
- performance case 可在 matrix 中 SKIP，但 **MUST** 在 `correctness_report.md` 标注，且 **MUST** 在 Phase 6 按 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) 单独采集（算子目录通常为 `scripts/perf.sh`）

### 2.2 通过率

| 档位 | 要求 |
| --- | --- |
| smoke（XS） | 100% PASS |
| functional + boundary + stress | 100% PASS（或逐项 DEFER 并写明原因） |
| M 档 | design 声明的 M case **MUST** 执行；未通过 **MUST NOT** 进 Phase 5 |

### 2.3 交付物

- `docs/case_matrix_report.md`：**每一个** case_id 与 status
- `docs/correctness_report.md`：汇总表 + 全量 case 表 + invariant 表

### 2.4 禁止行为

- ❌ smoke 通过即声称 Phase 4 完成
- ❌ 无 `run_case_matrix.py` 却手写 correctness 报告
- ❌ 报告只列 PASS case，省略 FAIL/SKIP
- ❌ 未跑 matrix 即进入 Phase 5/6/7

---

## 3. 性能采集门控（Phase 6）

当 `meta.performance_required: true` 时，**MUST** 完成以下全部项，否则 **MUST NOT** 进入 Phase 6.5 或输出 Phase 7 性能结论。

### 3.1 规模覆盖（Hard Gate）

**MUST** 覆盖 **S 档 + L 档** 各至少 1 case（[performance-eval-guide.md §4](../../shmem-ops-performance-eval/references/performance-eval-guide.md)、[testcase-scale-standard.md §1.2](../../shmem-ops-testcase-gen/references/testcase-scale-standard.md)）。

> **PE 数：推荐 8PE** 但资源受限时可用 design/phase0 确认的 PE 数（如 2PE/4PE），**MUST** 在报告中标注实际 PE 数，该结果可作为 Phase 6 依据，不作为阻塞项。

### 3.2 采集与报告

通算融合性能指标 **以端到端为准**：

1. 聊天 **带宽表 + 时延表**（S + L 档；[perf-chat-output-spec.md](../../shmem-ops-performance-eval/references/perf-chat-output-spec.md)）
2. `docs/performance_report.md` 端到端对比表（SHMEM vs P1 fused / P2 stitched baseline）
3. **主指标**：端到端 `e2e_us`（S + L）；带宽类同时报告 `algo_bandwidth_GBps` / `e2e_bus_bandwidth_GBps` / `kernel_bus_bandwidth_GBps`（P1 fused 可用 e2e 归一化作主对比，**MUST** 标注口径）
4. **可选诊断**：SHMEM 侧 `comm_us` / `kernel_bus_bandwidth_GBps`、`matmul_us`、`compute_util%` 等分段数据 — **不作为 Phase 6 门禁**，也不要求单独的 `phase_timing_report.md`

采集 **MUST** 按 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) 分三阶段（A baseline → B SHMEM → C 离线），**NEVER** 同 shell 混跑。（此处「分三阶段」指 baseline 与 SHMEM 分 shell 采集，与上文「不按 phase 拆解耗时」是两回事。）

**禁止**：只采 S 档或只采 L 档即声称 Phase 6 完成。

---

## 4. 计算实现规范

通算融合算子的计算子系统 **MUST** 按下列分级选型。

| 级别 | 实现 | 用途 | 交付默认 |
| --- | --- | --- | --- |
| **L0** | Device 三重循环 / 逐元素标量 MatMul | bring-up 原型（不得提交） | ❌ **禁止** |
| **L1** | Host `aclnnMatmul` + 独立 Comm kernel | 正确性过渡 | ⚠ 仅 `compute_path: host_aclnn` 且 design 授权 |
| **L2** | AIC `CATLASS BlockMmad` + AIV `CommBlockEpilogue`（CoC） | 标准交付路径 | ✅ **默认** |
| **L3** | L2 + ping-pong、Split-K、Swizzle、compute-comm overlap | Phase 6.5 机制优化 | ✅ 优化轮 |

**CATLASS 源码参考**：L2/L3 计算段实现 **MUST** 优先参考仓内 `${SHMEM_REPO}/3rdparty/catlass`；若该目录不存在或内容不完整，执行 `git clone https://gitcode.com/cann/catlass.git ${SHMEM_REPO}/3rdparty/catlass` 后再检索。检索顺序：

1. `3rdparty/catlass/examples/`：按算子语义找 `basic_matmul`、`grouped_matmul`、`matmul_gelu/silu`、`small_matmul`、`optimized_matmul_tla` 等相近样例
2. `3rdparty/catlass/include/catlass/gemm/`：确认 `BlockMmad`、`DispatchPolicy`、`GemmType`、`GemmShape` 的真实接口
3. SHMEM 仓内 `examples/matmul_*`、`examples/allgather_matmul*`：确认 CATLASS 计算段与 CATCOC/SHMEM 通信段的协同方式

CATLASS 只作为 compute 段高性能实现参考；通信 epilogue、跨 PE 同步和 symmetric buffer 仍以 SHMEM/CATCOC 样例为准。

### 4.1 设计与编码要求

1. Phase 1 design **MUST** 在 DSL `schedule.compute` 声明 `compute_path`：`device_catlass_coc` | `host_aclnn`
2. 选 `host_aclnn` 时 **MUST** 在 Capability Mapping 写明过渡理由及升级至 L2 的条件（`upgrade_plan`）
3. code-gen **MUST** Read [compute-optimization.md](../../shmem-ops-performance-optim/references/compute-optimization.md) 与 [fused-compute/GUIDE.md](../../shmem-ops-code-gen/templates/fused-compute/GUIDE.md)，并按上面的 CATLASS 源码参考规则查真实样例/头文件
4. code-review **MUST** 检查：无 L0 三重循环；若仅为 L1，Section 4 风险表 **MUST** 记录

### 4.2 禁止行为

- ❌ Device 侧 `for i for j for k` 标量 MatMul 作为交付实现
- ❌ 未经 design 授权默认 `host_aclnn` 路径
- ❌ AIV 标量/向量点乘替代 AIC Cube 计算作为性能路径

---

## 5. 性能优化门控（Phase 6.5）

当 `meta.performance_auto_optim: true` 且 Phase 6 未达标时，**MUST** 进入 Phase 6.5 并完成下列全部要求。

### 5.1 轮次要求

- **MUST** 完成 Round 1~5（达标 **不是** 提前停止理由）
- 每轮 **MUST** 包含：一项机制改动 + `run_case_matrix.py` 复测 + Phase 6 指标复采 + 聊天 Δ% 表
- 每轮机制改动 **MUST** 有实质差异（overlap、ping-pong、CoC 调优、sender/receiver 分核等）

### 5.2 禁止行为

- ❌ 未满 5 轮即进入 Phase 7
- ❌ 将 H2D 移出循环、block_dim 扫参等参数项拆成 5 轮凑数
- ❌ 有 P1/P2 baseline 却不在每轮对比表中报告 SHMEM vs baseline e2e（及可选通信段）Δ%
- ❌ 仅口头声明“做了 5 轮”但缺少 Round 表、每轮复测结果与 keep/revert 决策证据

---

## 6. Baseline 规范

通算融合 baseline **MUST** 按 [baseline-selection.md](../../shmem-ops-performance-eval/references/baseline-selection.md) **优先级**选型，**MUST NOT** 跳过更高优先级直接退到拼接或 metric_only。

### 6.0 选型优先级（Hard Gate）

| 优先级 | 类型 | 何时使用 | 来源 |
| --- | --- | --- | --- |
| **P1** | **aclnn 融合算子** | CANN 有语义对应的 fused API（如 `aclnnMatmulAllReduce`、`aclnnMatmulAllToAll`、`aclnnAllGatherMatmul`、`aclnnMatmulReduceScatter` 等） | [baseline-selection.md §1.2](../../shmem-ops-performance-eval/references/baseline-selection.md) |
| **P2** | **小算子拼接（stitched）** | 无合适 aclnn fused，或用户明确要求不用已弃用 fused API | [baseline-selection.md §2](../../shmem-ops-performance-eval/references/baseline-selection.md) |
| **P3** | **metric_only** | HCCL 清单、aclnn 清单、拼接两段均无法落地，且 `baseline_search` 穷尽 | [baseline-selection.md §3](../../shmem-ops-performance-eval/references/baseline-selection.md) |

**弃用接口**：若在线文档标注 aclnn fused「后续版本会废弃」，**MUST** AskQuestion 用户：仍用该 API，或改用 P2 拼接。

**禁止**：有对应 aclnn fused 却默认只用单一 HCCL 原语（如仅 `HcclAllReduce`）代替全流程 baseline。

### 6.1 实现与采集要求

| 要求 | 说明 |
| --- | --- |
| **实现位置** | `baseline/` 独立工程（`baseline/src/*_baseline.cpp` + `baseline/CMakeLists.txt` + `scripts/run_baseline.sh`）；**NEVER** 混入算子 `src/` |
| **形态** | C++ 可执行文件（**NEVER** Python）；输出 `[BASELINE_PERF]`（含 `e2e_us`；有通信 payload 时同时报 `algo_bandwidth_GBps` / `e2e_bus_bandwidth_GBps` / `kernel_bus_bandwidth_GBps`） |
| **P1 fused** | 直接调用对应 `aclnn*GetWorkspaceSize` + `aclnn*`；HCCL group 名与 `HcclComm` 对齐 |
| **P2 stitched** | 计算段：CATLASS BlockMmad 或 `aclnnMatmul` / GroupedMatmul（与 design `compute_path` 对齐）+ 通信段：HCCL 对应原语，**顺序拼接、无 overlap** |
| **采集** | 分阶段 A baseline → B SHMEM → C 离线（baseline 与 SHMEM 分 shell）；S/L 档 **MUST** 与 SHMEM case 同参 |
| **对比主口径** | **端到端 e2e**：SHMEM vs baseline（P1 fused 或 P2 stitched）；通信段 `kernel_bus_bandwidth_GBps` / 计算段 util 仅可选诊断 |
| **达标线** | 默认 current ≥ baseline 的 **80%**（时延：SHMEM e2e ≤ baseline_e2e / 0.8；带宽：等价比值 ≥ 0.8）；见 [baseline-selection.md](../../shmem-ops-performance-eval/references/baseline-selection.md) 与 [performance-eval-guide.md](../../shmem-ops-performance-eval/references/performance-eval-guide.md) |
| **记录** | `performance_report.md` / `baseline_search` **MUST** 写明 `baseline.type`（`cann_operator` / `stitched` / `metric_only`）、API、CANN 版本 |

Phase 3 code-gen **MUST** 生成 `baseline/` 骨架（优先按 P1 API 名，否则 P2 stitched 骨架）。

### 6.2 Proxy baseline 限制（Hard Gate）

`stitched_proxy`（如 host 侧 memcpy/模拟通信拼接）仅用于环境受阻时的临时趋势观察：

1. 报告 **MUST** 明确标注 `baseline_kind=proxy` 与阻塞原因（错误码/日志）
2. **MUST NOT** 用 proxy 数据宣称“已完成 HCCL/aclnn 对比”或“Phase 6 达标”
3. 若仅有 proxy baseline，Phase 结论 **MUST** 标记为 `Phase 6 blocked by baseline environment`
4. 未补齐真实 P1/P2 baseline 前，**MUST NOT** 进入 Phase 7 完成交付

**禁止行为**：
- ❌ 通算融合有 aclnn fused（P1）却无问询直接跳过、或用单一 HCCL 代替全流程
- ❌ 只报 SHMEM `[PERF]`、无 `[BASELINE_PERF]`（P3 metric_only 除外）即声称 Phase 6 完成
- ❌ baseline 与 SHMEM 同 shell 混跑
- ❌ 只采 L 档或只采 S 档即声称性能验证完成
- ❌ 用 proxy baseline 替代真实 P1/P2 并声称 Phase 6 完成
- ❌ P3 metric_only 未写明已穷尽 §1.1/§1.2/§2 搜索过程
---

## 7. 交付物清单

| 文档 / 目录 | 产出 Phase | `performance_required: false` | `performance_required: true` |
| --- | --- | --- | --- |
| `docs/design.md` | 1 | ✅ | ✅ |
| `docs/case_matrix_report.md` | 4 | ✅ | ✅ |
| `docs/correctness_report.md` | 4 | ✅ | ✅ |
| `docs/review-report.md` | 5 / 7 | ✅ | ✅ |
| `README.md` | 3 | ✅ | ✅ |
| `baseline/`（P1 fused 或 P2 stitched） | 3 / 6 | — | ✅ |
| `docs/performance_report.md` | 6 | — | ✅ |
| `docs/delivery_summary.md` | 7 | ✅ | ✅ |

**缺一即 Phase 7 未完成。**

## 7.1 元数据一致性门禁（新增 Hard Gate）

`design.md` 的 DSL 元数据与交付物必须一致：

- `meta.torch_required: true` → **MUST** 存在 Torch 绑定与测试产物（`aclshmem_torch.so`、`torch_test_*.py` 及结果）
- `meta.performance_required: true` → **MUST** 完成 §3 和 §6 的全部门禁
- 若实际执行与 DSL 不一致，**MUST** 先修订 design DSL 并记录偏差原因，修订前 **MUST NOT** 进入下一 Phase

---

## 8. 关联文档

| 文档 | 用途 |
| --- | --- |
| [shmem-ops-dev/SKILL.md](../SKILL.md) | Phase 0–7 主编排 |
| [agent-execution-contract.md](agent-execution-contract.md) | 执行契约（禁止中断、聊天双表） |
| [op-dsl.md](../../shmem-ops-design/references/op-dsl.md) | `schedule.compute` 字段定义 |
| [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) | 三阶段性能采集 |
| [baseline-selection.md](../../shmem-ops-performance-eval/references/baseline-selection.md) | baseline 优先级（aclnn fused → stitched → metric_only） |
| [performance-eval-guide.md](../../shmem-ops-performance-eval/references/performance-eval-guide.md) | 规模门禁、contract、结果表 |
| [perf-chat-output-spec.md](../../shmem-ops-performance-eval/references/perf-chat-output-spec.md) | Phase 6/6.5 聊天双表 |
