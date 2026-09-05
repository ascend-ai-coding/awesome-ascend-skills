---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-code-gen
description: 根据 design.md 生成基于 SHMEM 的算子代码、CMake、README 和目录结构。触发：用户需要基于 SHMEM 生成算子实现、kernel、CMake
  或工程目录时使用。关键词：基于SHMEM代码生成、算子实现、code-gen、kernel、CMake。
original-name: shmem-ops-code-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子代码生成

**Skill类型**：代码生成型（读取设计文档，输出可编译代码）

> **中文写作要求**：README.md 等交付文档必须使用中文撰写。仅 API 名称、代码片段、命令行示例等技术术语保留英文原文。

消费通过质量门禁的 `design.md`，生成 SHMEM 通信算子或通算融合算子的完整代码。不做需求设计（回 `shmem-ops-design`），不做正确性验证（交给 `shmem-ops-correctness-eval`），不做性能优化（交给 `shmem-ops-performance-optim`）。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/internal-api-boundary.md](references/internal-api-boundary.md) | **禁止** `aclshmemi_*`、deprecated barrier、quiet 误用 |
| [references/api.md](references/api.md) | SHMEM API 选择参考 |
| [references/code-patterns.md](references/code-patterns.md) | Host/Device 代码组织、RMA 模式、chunk/tail |
| [references/atomic-add-pattern.md](references/atomic-add-pattern.md) | `SetAtomicAdd<T>()` 累加：安全顺序、边界、风险。实现 reduce/累加时必读 |
| [references/code-style.md](references/code-style.md) | C/C++ 代码规范 |
| [references/readme-spec.md](references/readme-spec.md) | README.md 格式规范 |
| [../shmem-ops-compile-debug/references/custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) | custom-ops 编译/运行/matrix/Torch 入口命令；README 运行入口 **MUST** 以此为准 |
| [../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内 `docs/`（**只读**）、`install/shmem/include/`、examples 参考 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO`（**读仓内文件前**） |

读 `${SHMEM_REPO}/docs/` **只读**；**MUST** 先定位 `SHMEM_REPO`（shmem-repo-resolution），**NEVER** 向 `docs/` 追加或修改内容。

模板（分文件承载，按生成步骤按需读取，**NEVER** 一次读完全部模板）：

| 文件 | 阅读时机 | 内容 |
| --- | --- | --- |
| [templates/communication/GUIDE.md](templates/communication/GUIDE.md) | 步骤 2 选模板时 | 索引、路径映射、约束 |
| [templates/communication/templates-cmake-main.md](templates/communication/templates-cmake-main.md) | 步骤 4 子步骤 1 | `CMakeLists.txt`、`main.cpp` |
| [templates/communication/templates-kernel.md](templates/communication/templates-kernel.md) | 步骤 4 子步骤 2–4 | `*_kernel.h`、`*_kernel.cpp` |
| [templates/communication/templates-scripts.md](templates/communication/templates-scripts.md) | 步骤 4 子步骤 5（若需补齐 scripts） | `gen_data.py`、`check_result.py`、`scripts/run.sh` |
| [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md) | `op_kind=fused_compute_comm` 时步骤 2–4 | 通算融合（CMake + AIC/AIV kernel + main.cpp + scripts） |
| [../shmem-ops-dev/references/fused-compute-phase-gates.md](../shmem-ops-dev/references/fused-compute-phase-gates.md) | `op_kind=fused_compute_comm` 时 **MUST** | 通算融合开发规范（§4 计算实现、§6 HCCL） |
| [../shmem-ops-performance-optim/references/compute-optimization.md](../shmem-ops-performance-optim/references/compute-optimization.md) | 通算融合 codegen 时 | CATLASS TileShape、CoC、业界高阶模式 |

必要时先定位 `SHMEM_REPO`，再查阅仓内文档与头文件（见 shmem-repo-docs-index、shmem-repo-resolution）：`${SHMEM_REPO}/docs/`、`${SHMEM_REPO}/install/shmem/include/`、`${SHMEM_REPO}/examples/`。真实代码与文档优先于记忆。

通算融合计算段还必须把 CATLASS 作为参考来源：优先使用 `${SHMEM_REPO}/3rdparty/catlass`；若不存在或不完整，可执行 `git clone https://gitcode.com/cann/catlass.git ${SHMEM_REPO}/3rdparty/catlass`。先查 `examples/` 中相近 matmul/grouped-matmul/activation/optimized TLA 样例，再查 `include/catlass/gemm/` 真实接口；

---

## 输入门禁

开始生成前必须验证 `design.md`：

- [ ] 存在完整的 Canonical DSL `yaml` 代码块
- [ ] `source.user_confirmations` 记录了确认的 `op_name` 和 `dtypes`
- [ ] capability mapping 覆盖六类：lifecycle、memory、transport、sync、compute、scheduler
- [ ] DSL `schedule` 的 core_partition/tiling/phases 足够具体
- [ ] compile contract 写明 `cann_env` 和 `build_mode`
- [ ] **Design Review Before Handoff（section 5）已填写且无 FAIL 项**
- [ ] DSL `performance.baseline` 不为空或 `"none"`（必须为具体 baseline 来源——HCCL/aclnn/metric_only，且附 `baseline_search` 搜索记录）

门禁失败时停止，要求先用 `shmem-ops-design` 修订。

### 门禁执行规则

门禁检查是阻断条件，不是建议。执行方式：

1. 逐项检查上述 checklist，结果必须显式输出给用户（列表形式）
2. 任一项 FAIL → 立即停止，向用户报告缺失项，要求回 `shmem-ops-design` 补齐
3. 不得以"先跑通再补"、"设计基本够用"、"benchmark 不需要完整设计"等理由跳过 FAIL 项
4. 如果 design.md 缺少 Canonical DSL 的 `schedule` / `correctness` / `performance` section，直接判定 FAIL
5. 如果 Design Review Before Handoff（section 5）不存在或有 FAIL 项，直接判定 FAIL

---

## 工作流

```
步骤 1  提取设计契约
步骤 2  选择模板和参考 example
步骤 3  制定实现计划
步骤 4  渐进式代码生成
步骤 5  调用 shmem-ops-compile-debug 编译验证
```

---

## 步骤 1：提取设计契约

从 `design.md` 提取：

| 内容 | DSL 字段 |
| --- | --- |
| 算子身份 | `meta.op_name`、`op_kind`、target SoC、scope |
| 接口 | inputs/outputs、dtype、shape、visibility |
| 语义 | local_compute、communication、finalize |
| 拓扑 | team、peer_model、addressing |
| 内存 | buffers、symmetric_layout、signal/state |
| 调度 | phases、tile/chunk/tail、core_partition、overlap |
| 正确性 | oracle、tolerance、invariants、case_matrix |
| 性能 | metric、baseline、target_cases |

---

## 步骤 2：选择模板和参考 example

| `op_kind` 或语义 | 模板目录 |
| --- | --- |
| `transport` / `collective` / `compute` / 纯 put/get/exchange | `templates/communication` |
| `fused_compute_comm`（Matmul/GEMM + 跨 PE 通信，AIC/AIV CoC） | `templates/fused-compute` |

> **`compute`**：单 PE 或 Device 内本地计算、无跨 PE 通信语义时仍用 `templates/communication`（通常仅 Host + 空/轻量 kernel）；**禁止**因存在 `local_compute` 字段就路由到 fused-compute。
> **`fused_compute_comm`**：必须同时含 Cube matmul（CATLASS BlockMmad）与 SHMEM 跨 PE 通信（CommBlockEpilogue），见 [core-allocation.md §4](../shmem-ops-design/references/core-allocation.md)。

选定模板后，**仅读取当前生成步骤对应的模板文件**（见上表）。纯通信按 [templates/communication/GUIDE.md](templates/communication/GUIDE.md) 分步读取子模板；通算融合从 [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md) 按章节标题提取 fenced code block，写入目标路径，替换 `<op_name>`/`<OpName>`/`<OP_NAME>` 占位符。

**通算融合（`fused_compute_comm`）MUST**：
1. Read [fused-compute-phase-gates.md §4](../shmem-ops-dev/references/fused-compute-phase-gates.md) 确认 `compute_path`
2. Read [compute-optimization.md](../shmem-ops-performance-optim/references/compute-optimization.md) 与 [fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md)
3. 对 L2/L3 计算段，MUST 在 `${SHMEM_REPO}/3rdparty/catlass`（缺失时 clone `https://gitcode.com/cann/catlass`）中检索相近实现，避免凭记忆臆造 CATLASS 类型/接口

参考 example 选择：
- allgather / put-get / SDMA / RDMA → `${SHMEM_REPO}/examples/allgather`、`sdma`、`rdma_demo`（先定位 `SHMEM_REPO`）
- matmul allreduce / reduce-scatter → `${SHMEM_REPO}/examples/matmul_allreduce`、`matmul_reduce_scatter`
- KV / dispatch combine → `${SHMEM_REPO}/examples/kv_shuffle`、`dispatch_gmm_combine`

选定后记录路径和复用理由。

---

## 步骤 3：制定实现计划

编码前写出：
- 目标目录和文件列表（**默认** `custom-ops/<op_name>/`，非 `examples/`）
- 复用的 API、example、template
- `main.cpp` 与 Host helper 模块的职责边界
- 构建模式（`independent_project` 默认）与编译命令（[custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1 编译）
- README.md 覆盖范围

---

## 步骤 4：渐进式代码生成

**模板分支（MUST）**：
- `meta.op_kind == fused_compute_comm` → **只读** [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md)，按章节标题提取代码块；**NEVER** 读 `templates/communication/templates-*.md`
- `transport` / `collective` / `compute` → 只读 communication 子模板（下表）

按"最小正确路径 → 完整正确性 → 性能路径"顺序。**每子步骤只读当前分支对应模板文件**：

1. **CMake + Host**（communication：`templates-cmake-main.md`；fusion：fused-compute GUIDE 对应章节）
2. **Kernel 声明**（communication：`templates-kernel.md`；fusion：fused-compute GUIDE）：`src/<op_name>_kernel.h`
3. **lifecycle + memory**（同上）：初始化、symmetric allocation、释放
4. **transport + sync + compute**（communication 分支；fusion 在 fused-compute kernel 章节实现）：
     - **通算融合 — 计算实现分级**（见 fused-compute-phase-gates §4）：
       - **L0 禁止**：Device 三重循环 / 逐元素标量 MatMul，或者在 AIV 上用标量/向量点乘替代 AIC 计算 → **NEVER** 作为交付代码
       - **L1 过渡**（仅 `compute_path: host_aclnn` 且 design 明确授权）：Host `aclnnMatmul` + 独立 Comm kernel；**MUST** 在 README 标注过渡方案
       - **L2 推荐**（默认）：AIC `CATLASS BlockMmad` + AIV `CommBlockEpilogue`（CoC）；生成前先查 `${SHMEM_REPO}/3rdparty/catlass` 或 `https://gitcode.com/cann/catlass`
       - **L3 进阶**：L2 + ping-pong、Split-K、Swizzle、compute-comm overlap（Phase 6.5）
     - **纯通信**：按 design.md 实现 local compute（如有）
      - **GM 累加方式选择**：当 kernel 需要将远端数据累加到 output 时，按 [references/atomic-add-pattern.md](references/atomic-add-pattern.md) §12 决策优先级表选择
5. **测试脚本**（communication：`templates-scripts.md`；fusion：fused-compute GUIDE；若 Phase 2 已生成则跳过）
6. **scheduler**（communication：kernel.cpp；fusion：fused-compute kernel 章节）：phase、tile/chunk/tail、core partition、overlap
7. **README.md**（按 references/readme-spec.md）

> **性能打点 + msprof runner（Phase 3 跳过）**：模板中的 `--perf` 模式、内置前后打点、`[PERF]` 输出、`scripts/perf.sh` **MUST 在 Phase 3 跳过**。这些代码段仅当 Phase 6 dev 显式调用 code-gen 添加性能采集入口时写入。Phase 3 生成的是纯 correctness 代码（perf_times 默认 0）。Phase 6 生成的 perf runner **必须保留原有内置打点指标**，同时可被 `msprof --task-time=l2` 包裹，供 performance-eval 额外采集 msprof 指标。

### 关键约束

- **算子必须在 Device 执行**，禁止 Host RMA 作为主通信路径
- **main.cpp 只做单 PE Host 编排**：参数解析、lifecycle、I/O、launch、cleanup
- 复杂 Host 逻辑拆到独立 `.cpp/.h`（如 `op_host_plan.cpp`）
- 跨 PE 传输必须使用 `aclshmem_*` 或 `aclshmemx_*` 接口
- symmetric allocation 顺序和大小在所有 PE 一致
- **通算融合 — MatMul 默认 L2**：AIC + CATLASS BlockMmad，AIV CommBlockEpilogue；生成前必须检索 `${SHMEM_REPO}/3rdparty/catlass`（缺失时 clone `https://gitcode.com/cann/catlass`）获取相近实现和真实接口；**禁止** L0 三重循环；L1 `host_aclnn` 须 design 授权并在 review 记风险
- **通算融合 — 有通信语义时**：Phase 3 **MUST** 生成 `baseline/` 骨架，按 [baseline-selection.md](../shmem-ops-performance-eval/references/baseline-selection.md) 优先级选型（**P1** aclnn fused §1.2 → **P2** 计算段 CATLASS/aclnn + 通信段 HCCL 拼接 §2）；弃用接口须先 AskQuestion。详见 [fused-compute-phase-gates.md §6](../shmem-ops-dev/references/fused-compute-phase-gates.md)
- `block_dim=1` 仅临时调试用；首版 correctness 必须落地 design 的并发
- 新增核心能力必须对应 gap analysis

### 性能输出要求

main.cpp 的 `--perf` 模式必须保留原有 `[PERF]` 双指标延迟和带宽输出，同时 `scripts/perf.sh` 必须能被 msprof 包裹（严格按照 [shmem-ops-performance-eval/references/timing-and-metrics-standard.md](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md) 执行）：

| 指标 | 说明 | 公式 |
| --- | --- | --- |
| `e2e_us` | 端到端延迟：做法 A 下 ≈kernel_us（搬运在 kernel 内）；做法 B 下含 aclrtMemcpy + barrier + kernel | 做法 A：kernel launch 前到 stream sync 后；做法 B：aclrtMemcpy 前到 stream sync 后 |
| `kernel_us` | kernel 执行时间 | kernel launch 前到 stream sync 后 |
| `algo_bandwidth_GBps` | 算法带宽（基于 e2e_us，参考） | `logical_payload_bytes / (e2e_us * 1e-6) / 1e9` |
| `e2e_bus_bandwidth_GBps` | 总线带宽（e2e 参考） | `algo_bandwidth * bus_factor`（bus_factor 见 [timing-and-metrics-standard.md §4.3](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md)） |
| `kernel_bus_bandwidth_GBps` | kernel 口径带宽 | `logical_payload_bytes / (kernel_us * 1e-6) / 1e9 * bus_factor` |
| `bandwidth_utilization_pct` | 带宽利用率（基于 kernel_bus_bandwidth_GBps） | `kernel_bus_bandwidth_GBps / peak_bandwidth * 100`（peak_bandwidth 按通信模式确定，见下文） |

`bus_factor` 按算子语义确定（不区分拓扑，NCCL 惯例的通信量标准化系数）：AllReduce: `2*(n-1)/n`，ReduceScatter/AllGather: `(n-1)/n`，AllToAll/Shuffle: `(n-1)/n`，Broadcast/P2P: 1。

`peak_bandwidth` 按通信模式确定（参考 [hardware-architecture.md §2.6](../shmem-ops-design/references/hardware-architecture.md)）：
- P2P 点对点：28 GB/s（单条 HCCS 链路单向）
- 集合通信（AllReduce/AllGather 等）：聚合带宽，如 910B3 8 卡 full-mesh 为 7 × 28 = 196 GB/s

perf 循环结构要求：
- **内置打点循环（保留）**：每轮覆盖从用户 input(GM) 到结果 output(GM) 的全部数据搬运，并打印原有 `[PERF]` 字段。做法 A（MTE/SDMA/UDMA kernel 内放置）下 e2e 循环 = kernel launch 循环，搬运在 kernel 内自然完成；做法 B（RDMA）下需包含 aclrtMemcpy + barrier + kernel launch。数据放置方式详见 [timing-and-metrics-standard.md §1.2](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md)
- **msprof 采集兼容**：同一个 `scripts/perf.sh` 必须可被 `msprof --task-time=l2` 包裹。msprof 解析时按 `WARMUP` 跳过预热轮，正式轮数与 `[PERF]` measure 轮数一致
- **e2e 计时口径**：做法 A 下 e2e_us≈kernel_us 是正常的（搬运在 kernel 内），代码中 **MUST** 有注释说明引擎选择（如 `// MTE put_nbi src=local GM, no Host-side memcpy needed`）；做法 B 下 e2e_us **MUST** > kernel_us，差值等于 kernel 外搬运时间。**禁止**为制造 e2e > kernel 假象而在做法 A 路径上加无意义的 kernel 外搬运；**禁止**在性能循环前预做搬运使 e2e 口径缩水

注意：`algo_bandwidth` 不乘 2，统一按 input size 计算（NCCL `algBw` 惯例）。`logical_payload_bytes` 必须在输出中注明口径（单 PE 还是全局）。Phase 6 报告必须保留 `source=inline_event` 的 `[PERF]` 指标；其中 `e2e_us` 只使用 inline_event 口径。`kernel_us` 需要同时给出 `inline_event` 和 `msprof_task_time` 两种来源；与 baseline 达标对比优先使用 msprof 的 `kernel_us` / `kernel_bus_bandwidth_GBps` 口径。

如果算子包含 `SHMEMI_PROF_START/END` 打点，`--perf` 模式还应调用 `aclshmemx_get_prof(nullptr, true)` 输出 Device 帧数据。

---

## 步骤 5：编译验证

1. 调用 `shmem-ops-compile-debug`，传入 compile contract，由 compile-debug 执行构建并诊断失败
2. compile-debug 返回诊断结果后：
   - compile / link / launch 失败 → **本 skill 修复代码**（compile-debug 只诊断不改代码），修复后重新调用 compile-debug
   - correctness 失败 → 先判断是代码问题还是 Phase 2 测试脚本不匹配（如 gen_data 布局、check_result 容差）；代码问题由本 skill 修复，脚本问题委托 `shmem-ops-testcase-gen` 修正
   - runtime / environment 失败 → compile-debug 自行修复环境后重试
3. 该循环持续直到编译通过且 smoke case 运行通过，或确认为环境阻塞 / 设计缺陷

---

## 最终目标目录结构

**默认根路径**：`custom-ops/<op_name>/`（独立工程）。in-tree 时为 `examples/<op_name>/`。

最终交付目录结构 **MUST** 严格遵循以下布局（以下以 `<op_root>/` 表示算子根目录）：

```
op_name/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── design.md                  # shmem-ops-design
│   ├── review-report.md           # shmem-ops-code-review
│   ├── correctness_report.md      # shmem-ops-correctness-eval
│   ├── performance_report.md      # shmem-ops-performance-eval / shmem-ops-performance-optim
│   └── case_matrix_report.md      # shmem-ops-testcase-gen
├── src/
│   ├── main.cpp
│   ├── op_name_kernel.cpp
│   ├── op_name_kernel.h
│   ├── op_host_plan.cpp (可选)
│   └── op_host_plan.h (可选)
├── scripts/
│   ├── gen_data.py
│   ├── check_result.py
│   ├── run.sh
│   ├── run_case_matrix.py
│   ├── perf.sh                      # Phase 6；实现见 perf-workflow.md
│   └── perf_compare.sh              # 有 baseline 时；或统一用 perf-workflow §1 阶段 C
└── baseline/                       # 有 baseline 时 MUST 存在
    ├── CMakeLists.txt              # 独立的 baseline 编译 target（add_subdirectory）
    ├── src/
    │   └── op_name_baseline.cpp    # HCCL/aclnn baseline 源码
    └── scripts/
        └── run_baseline.sh         # baseline runner，供 msprof 包裹
```

---

## MUST检查（全部通过方可通过本阶段，禁止以"先跑通再补"跳过任一项）

### P0（阻断级——任一 FAIL 则禁止进入 Phase 4）

- [ ] **目录结构**：`docs/` 承载所有阶段产出的 `.md`、`src/` 含全部算子 `.cpp/.h`、`scripts/` 含全部测试脚本；`baseline/` 含全部 baseline 源码和编译配置（**Phase 6 由 dev 调用 code-gen 按需生成，Phase 3 不检查此项；`metric_only` 或 `performance_required: false` 时标记 N/A**）
- [ ] **CMake**：target 命名与 op_name 一致；使用 target-scoped include/link/compile options；禁止全局 `link_libraries()`（见 code-style.md §10.1）
- [ ] **代码遵循 design.md**：phase/transport/sync/partition 全部与 DSL `schedule` 一致
- [ ] **默认 unified single path**：无 profiling ≥5% 收益证据则不引入 size 分支（见 code-patterns.md §6.4 / implementation-boundary.md）
- [ ] **GM累加方式**：远端数据累加到 output 时，**MUST** 按 `atomic-add-pattern.md §12` 决策优先级表选择累加方式；reduce-scatter/allreduce RS 阶段 **MUST** 使用 `SetAtomicAdd<T>()`（MTE 批量）+ `mte_get_nbi`（优先级 1），**禁止**以串行 UB 累加（优先级 4/5）作为交付路径
- [ ] **性能指标计算**：整个计算链路 **MUST** 正确——(1) `logical_payload_bytes` = `elements × sizeof(dtype)`（单 PE 语义数据量）；(2) inline `algo_bandwidth_GBps` = `logical_payload / (e2e_us × 1e-6) / 1e9`（基于 inline_event 的 e2e_us，不乘2）；(3) `bus_factor` 从 [timing-and-metrics-standard.md §4.3](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md) 唯一参照表取值；**禁止**使用 `n_pes-1` 或其他错误值；(4) `e2e_bus_bandwidth_GBps` = `algo_bandwidth × bus_factor`（e2e 口径，仅参考）；(5) inline `kernel_bus_bandwidth_GBps` = `logical_payload / (kernel_us_inline × 1e-6) / 1e9 × bus_factor`；(6) msprof `kernel_bus_bandwidth_GBps` = `logical_payload / (kernel_us_msprof × 1e-6) / 1e9 × bus_factor`；(7) `bandwidth_utilization_pct` = `kernel_bus_bandwidth_GBps / peak_bandwidth × 100`。`[PERF]` 输出 **MUST** 包含既有 `e2e_us`、`kernel_us`、`algo_bandwidth_GBps`、`e2e_bus_bandwidth_GBps`、`kernel_bus_bandwidth_GBps`、`bandwidth_utilization_pct`；msprof 解析 **MUST** 输出 `kernel_us`、`kernel_bus_bandwidth_GBps`、`bandwidth_utilization_pct` 并标注 `source=msprof_task_time`
- [ ] **License 头**：所有 `.cpp`、`.h`、`.py`、`.sh` 文件 **MUST** 包含 CANN Open Software License 头（见 code-style.md §7.2）；交付不可省略
- [ ] **e2e 计时口径**：做法 A（MTE/SDMA/UDMA kernel 内放置）下 e2e_us≈kernel_us 正常，代码中 **MUST** 有注释说明引擎选择；做法 B（RDMA 或显式 Host 侧搬运）下 e2e_us **MUST** > kernel_us，kernel 外搬运 **MUST** 在 e2e/measure 循环内。**禁止**为制造 e2e>kernel 假象而在做法 A 路径上加无意义的 kernel 外搬运
- [ ] **跨 PE 传输**：**MUST** 使用 `aclshmem_*` / `aclshmemx_*` 数据面接口；**禁止** DataCopy 直接写远端地址
- [ ] **有状态 API 返回值检查**：有返回值的 ACL/SHMEM/RT 调用 **MUST** 检查返回值（如 `aclshmem_put_nbi`、`aclshmem_get_nbi`、`aclshmem_signal_wait_until`、`aclrtMalloc` 等）。**禁止**静默忽略返回值（见 code-style.md §1.4）
- [ ] **void 同步 API 对称性**：`aclshmem_barrier_all()`、`aclshmem_finalize()` 等返回 `void` 的同步 API **MUST** 检查：所有 PE 对称调用（无分支遗漏）、barrier 后无 data race、finalize 前数据已可见。**禁止**检查不存在的返回值
- [ ] **SHMEM 接口约束**：算子代码中的 SHMEM 调用 **MUST** 只使用对外 API（`aclshmem_*` / `aclshmemx_*` 开头）。若使用了 `aclshmemi_*` 内部接口：(1) 有公开等价 API 时 **MUST** 改为公开 API；(2) 无公开等价 API 时 **MUST** 在 Capability Mapping / gap analysis 中登记，写明无替代方案和风险评估（见 internal-api-boundary.md）

### P1（严重级——任一 FAIL 必须先修复再进入 Phase 4）

- [ ] **错误处理用宏**：**MUST** 使用 `ACL_CHECK` / `ACL_CHECK_WITH_RET` / `CHECK_SHMEM` 等宏进行错误检查，**禁止**裸写 `if (status != ACL_ERROR_NONE)` 模式（见 code-style.md §1.1 / §1.2）
- [ ] **README.md 中文 + 结构合规**：**MUST** 使用中文撰写；**MUST** 遵循 readme-spec.md 完整结构——含算子介绍、环境要求、目录结构、参数说明、编译项目、运行算子、验证结果、性能采集、注意事项等 section（见 shmem-ops-dev 原则#0；readme-spec.md §2）
- [ ] **无硬编码路径**：脚本/文档/CMake 中 **MUST NOT** 硬编码用户机器路径（如 `/home/<user>/...`），**MUST** 使用环境变量或相对路径（见 code-style.md §10.1）
- [ ] **同步 API**：phase 同步 **MUST** 优先使用 `aclshmem_barrier_all()` 或 `aclshmem_barrier(team)`；**禁止**无理由使用 `aclshmemx_barrier_all_vec()` 替代（仅对齐既有 legacy example 时可保留，见 code-style.md §6.2）
- [ ] **signal/wait 配对**：每个 signal 有对应 wait，magic/epoch 策略一致（见 code-review-checklist.md §同步正确性）
- [ ] **symmetric 分配顺序**：所有 PE 的 `aclshmem_malloc` 调用顺序和大小完全一致（见 code-review-checklist.md §内存与Buffer）
- [ ] **main.cpp 不含复杂逻辑**：route/payload/tiling/packing/golden/checker 逻辑 **禁止** 在 `main.cpp` 中；复杂 Host 逻辑 **MUST** 拆到独立 `.cpp/.h`（见 code-style.md §5.2）
- [ ] **编译成功**：或记录环境阻塞原因
- [ ] **记录所有生成/修改的文件**

### P2（建议级——MUST在交付前修复）

- [ ] **code-style.md §12 代码审查清单**：逐项对照完整清单（~32项），每项 **MUST** 标记通过/不适用/FAIL；**禁止**笼统声称"已检查"而不逐项记录。覆盖范围包括但不限于：License 头、include guard、缩进 4 空格、指针左对齐 `int *ptr`、`UNUSED_PARAM` 标记、UB/event 命名常量、资源释放顺序、注释解释"为什么"、命名风格一致性、`main.cpp` 边界等（详见 code-style.md §12）

**P0 全部 PASS 且 P1 全部 PASS → 进入 Phase 4；否则 MUST 修复后重新检查。**

### 条件性需求：性能打点代码（Phase 6 按需生成）

以下需求 **仅当** `meta.performance_required: true` 且 dev 在 Phase 6 显式调用 code-gen 添加性能打点时才生效。**Phase 3 不检查、不生成**。

- [ ] **--perf 模式**：main.cpp **MUST** 包含 `--perf` flag + warmup + timing loop + `[PERF]` 输出逻辑（见 templates-cmake-main.md perf 代码段）
- [ ] **perf.sh**：`scripts/perf.sh` **MUST** 存在且可执行，既能直接输出 `[PERF]`，也可被 `msprof --task-time=l2 ... bash scripts/perf.sh ...` 直接包裹（实现见 perf-workflow.md）
- [ ] **SHMEMI_PROF 分 phase frame**：kernel 中 **MUST** 使用 `SHMEMI_PROF_START(frame_id)` / `SHMEMI_PROF_END(frame_id)`，至少 5 个独立 `frame_id` 覆盖 copy_in / remote_put_get / signal_or_barrier_wait / local_compute / finalize（见 timing-and-metrics-standard.md §7）
- [ ] **Host 侧 show_prof**：stream sync 后 **MUST** 调用 `aclshmemx_get_prof(nullptr, true)` 导出 Device frame 数据（见 timing-and-metrics-standard.md §7）

---

## 反模式（NEVER DO THESE）

- ❌ Host RMA 作为 correctness 实现
- ❌ main.cpp 包含 route/payload/tiling/golden 逻辑
- ❌ main.cpp fork/spawn 多 PE
- ❌ DataCopy 直接写远端地址
- ❌ 跳过 Device kernel 实现
- ❌ 不读 design.md 就生成代码
- ❌ 一次性读取全部 communication 模板文件（必须按步骤只读当前子文件）
- ❌ design.md 门禁有 FAIL 项仍继续生成代码（"先跑通再补设计"是最常见的违规路径）
- ❌ GM 标量循环累加作为性能路径交付（按 atomic-add-pattern.md §12 决策表选择正确方式）
- ❌ reduce-scatter/allreduce RS 阶段使用串行 UB 累加（get→UB→Add）替代 `SetAtomicAdd<T>()` + `mte_get_nbi` 批量并行模式（见 atomic-add-pattern.md §5.1 / §12 优先级1）
- ❌ bus_factor 取值错误（**MUST** 从 `timing-and-metrics-standard.md §4.3` 唯一参照表取值，如 AllReduce 为 `2*(n-1)/n`，**禁止**用 `n-1`）
- ❌ License 头缺失即作为交付
- ❌ 错误处理裸写 `if (status != ACL_ERROR_NONE)` 而不使用 ACL_CHECK / CHECK_SHMEM 宏
- ❌ README.md 非中文或不遵循 readme-spec.md 结构
- ❌ 脚本/文档硬编码用户机器路径
- ❌ 删除或改名原有 `[PERF]` 指标字段（`e2e_us`、`kernel_us`、`algo_bandwidth_GBps` 等必须保留）
- ❌ Phase 3 生成 SHMEMI_PROF / --perf 打点代码（性能入口由 Phase 6 按需添加，code-gen 模板中 perf 代码段仅当 dev 显式调用时才写入）
- ❌ SHMEMI_PROF 只用单个 frame_id 包裹整个 kernel
- ❌ 只生成 msprof runner 而不保留内置前后打点和 `[PERF]` 输出
- ❌ 无理由使用 `aclshmemx_barrier_all_vec()` 替代 `aclshmem_barrier_all()`
- ❌ `(void)cast` 替代 `UNUSED_PARAM(x)`
- ❌ UB buffer 偏移量散落裸 magic value
- ❌ `aclshmem_barrier_all()` / `aclshmem_finalize()` 不对称调用（某 PE 跳过 barrier/finalize，导致死锁或资源泄漏）
- ❌ 算子代码中无正当理由调用 `aclshmemi_*` 内部接口（除非无公开等价 API 且已在 gap analysis 中登记）
- ❌ 修改 SHMEM 核心库却无 gap analysis 授权
- ❌ 将数据搬运排除在 e2e 性能循环外（性能循环前预搬运使 e2e 口径缩水）；或在做法 A 上加无意义搬运制造 e2e>kernel 假象
- ❌ `.md` 文档散放在算子根目录（必须归入 `docs/`）
- ❌ `.cpp/.h` 源文件散放在算子根目录（必须归入 `src/`）
- ❌ baseline 源码放在 `src/` 下或算子根目录（必须归入 `baseline/src/`，编译 target 必须在 `baseline/CMakeLists.txt`）
- ❌ Phase 3 自行生成 baseline 代码（baseline 在 Phase 6 确定后才由 dev 调用 code-gen 按需生成）
- ❌ 用户未要求 in-tree 时将算子生成到 `examples/`（默认 `custom-ops/<op_name>/`）
- ❌ 通算融合算子在 AIV 上用标量/向量点乘替代 AIC + CATLASS 高性能计算（即使以"正确性验证"为由也不允许——须使用 CATLASS BlockMmad + CommBlockEpilogue 的 AIC/AIV 分工模式）
- ❌ **通算融合**：Device 侧三重循环 MatMul（`for i for j for k` 点乘）作为交付实现
- ❌ **通算融合**：未经 design `compute_path: host_aclnn` 授权就默认 Host `aclnnMatmul` 路径
- ❌ **通算融合**：存在 HCCL 等价通信却不生成 `baseline/` 目录
