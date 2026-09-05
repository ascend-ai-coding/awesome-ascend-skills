---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-design
description: 基于 SHMEM 设计通信算子与通算融合算子，将需求转化为 design.md。触发：用户需要分析通信算法、PE、内存、同步或制定 SHMEM
  算子设计方案时使用。关键词：基于SHMEM算子设计、算子设计、设计文档、design、Canonical DSL、capability mapping。
original-name: shmem-ops-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子设计

**Skill类型**：文档生成型（需求分析 → 设计文档输出）

> **中文写作要求**：design.md 必须使用中文撰写。仅 API 名称、代码片段、变量名、DSL 字段名等技术术语保留英文原文。

将用户需求转化为可被 `shmem-ops-code-gen` 直接消费的 `design.md`。本 skill 只产出设计文档，不生成实现代码。

## 必读资料

设计前必须读取本 skill 的 reference：

| 文件 | 阅读时机 | 用途 |
| --- | --- | --- |
| [references/op-dsl.md](references/op-dsl.md) | **始终 — 最先读取** | Canonical DSL 字段、归一化规则和最小要求 |
| [references/capacity.md](references/capacity.md) | **始终** | SHMEM 已有能力、功能域和边界 |
| [references/core-allocation.md](references/core-allocation.md) | **始终** | 分核、phase、chunk、lane、通算 overlap 策略 |
| [references/design-template.md](references/design-template.md) | **始终** | `design.md` 输出模板 |
| [references/implementation-boundary.md](references/implementation-boundary.md) | **始终** | Device/Host/Python 实现边界和禁止项 |
| [references/hardware-architecture.md](references/hardware-architecture.md) | **始终** | AI Core、存储层次、传输引擎、同步机制 |
| [../shmem-ops-dev/references/intake-checklist.md](../shmem-ops-dev/references/intake-checklist.md) | **设计前** | `phase0_intake` 字段定义（op_name、dtype、build_mode、torch_required 等） |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | **读仓内文件前** | 定位 `SHMEM_REPO` |

当 API 名、能力边界、示例路径或约束不明确时，先定位 `SHMEM_REPO`（[shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md)），再检查 `${SHMEM_REPO}/include/`、`src/`、`examples/`、`tests/`、`docs/` 下的真实代码和文档。

## 核心原则

1. **语义优先**：先统一语义，再选择 API、模板和调度策略
2. **名称与类型先行**：设计开始前 **MUST** 从编排器传入的 `phase0_intake` 中读取 `op_name` 和 `dtype/dtypes`；未读取前不得进入 Canonical DSL、能力映射或执行方案设计
3. **语义与调度分离**：区分 semantics 和 schedule——semantics 描述必须发生什么，schedule 描述如何高效执行
4. **缺失不脑补**：缺失信息必须写入 assumptions、open questions 或 gap analysis，不能静默脑补 topology、visibility、sync、dtype、shape
5. **复用优先**：优先复用 SHMEM 已有 API、模板和 examples；新增模块必须说明原因、风险和验证方式
6. **面向代码生成**：输出必须面向代码生成，避免只写概念性 prose

## Phase 0 Intake 读取

编排器在 Phase 0 已确认以下五项 intake 信息（先探测缺项再问，记录在 `phase0_intake` 中；字段定义见 [intake-checklist.md](../shmem-ops-dev/references/intake-checklist.md)）。本 skill **MUST** 直接从 `phase0_intake` 读取，**禁止**就已确认项向用户再次提问。

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `op_name` | phase0_intake | 目录名、文件名前缀、kernel/target 命名 |
| `dtype` | Phase 0 Step 0.2 需求收集 | 输入/输出/中间计算 dtype |
| `build_mode` | phase0_intake | `meta.build_mode` |
| `torch_required` | phase0_intake | `meta.torch_required` |
| `performance_required` | phase0_intake | `meta.performance_required` |
| `performance_auto_optim` | phase0_intake | `meta.performance_auto_optim` |
| `shmem_repo` | phase0_intake | 仓内文件定位 |
| `docker_container` | phase0_intake | 是否 `docker exec` |
| 网络拓扑 | Phase 0 Step 0.2 需求收集 | SoC 型号 → 推断默认拓扑（如 910B3 → 8 卡 full-mesh） |

读取后 **MUST** 在 `design.md` 的 `source.user_confirmations` 中记录各字段值，来源标注 `phase0_intake`。

`design.md` 中不得留下阻塞的 dtype/open name/拓扑问题；否则不能交给 `shmem-ops-code-gen`。

---

## 步骤 1：输入规范化与语义提升

把自然语言、伪代码、异构参考实现统一成 Canonical DSL。

### 伪代码处理（用户提供伪代码时 MUST 执行）

用户提供伪代码时，**MUST** 完整通读并逐段拆解，**NEVER** 跳过或粗略浏览后直接照搬：

1. **逐段拆解**：识别伪代码中的局部计算段、跨 PE 数据搬运段、同步等待段
2. **数据搬运映射**：伪代码中隐式的跨 PE 数据访问必须显式映射为 SHMEM 原语：
   - "从其他 PE 读取数据" → `aclshmem_*_get`（MTE/SDMA/RDMA 择一，设计阶段不需要锁定具体引擎）
   - "向其他 PE 写入数据" → `aclshmem_*_put_nbi`
   - "广播/收集/散布" → 对称堆 + get/put + signal/wait 或 barrier 的组合
   - 所有跨 PE 搬运均经对称堆 staging：input(GM) → copy_in(symmetric) → put/get → copy_out(GM)
   - 伪代码中的"共享变量""全局数组"在 SHMEM 中不存在，不可直接引用为可寻址内存
3. **同步映射**：伪代码中的 ready/done/flag → signal/wait/barrier/fence/quiet
4. **禁止机械照搬**：**NEVER** 将伪代码的变量名、循环结构、内存模型直接当作 SHMEM 实现。必须经过语义提升后再设计。
5. **分核解耦**：伪代码通常是单线程视角，设计时必须按 core-allocation.md 将单线程逻辑映射为多 AIV/AIC 的分组执行方案

### 执行要求

1. 使用 `phase0_intake` 中读取的 `op_name`、dtype 和网络拓扑，并抽取 `op_kind`、target SoC、scope、shape、attrs、输出可见性。
2. 将需求提升为三类语义：
   - local compute：每个 PE 本地计算什么
   - communication：跨 PE 交换、搬运、规约或同步什么
   - finalize/result：最终输出由谁持有，replicated/sharded/owner-only 如何定义
3. 将异构参考实现转成 SHMEM 语义，而不是机械照搬 API：
   - send/recv -> put/get/exchange
   - reduce/allreduce/reducescatter/allgather -> collective semantic，再映射到 SHMEM 能力
   - ready/done/flag -> signal/wait/barrier/fence/quiet
4. 按 [references/op-dsl.md](references/op-dsl.md) 填写 Canonical DSL；`meta.op_kind` 判定规则见 op-dsl §3，并决定 code-gen 模板：`fused_compute_comm` → fused-compute GUIDE，其余 → communication 子模板。

**通算融合（`fused_compute_comm`）附加 MUST**：
- Read [fused-compute-phase-gates.md](../shmem-ops-dev/references/fused-compute-phase-gates.md)
- DSL `schedule.compute.compute_path` 声明实现路径（默认 `device_catlass_coc`）
- `performance.baseline` 存在 HCCL 原语时 **MUST** 选 HCCL，不得 metric_only

DSL 必须显式包含：interface、semantics、topology、memory、schedule、correctness、performance。

DSL `topology` 必须写明：
- `deployment`：`intra_node`（单服务器内）或 `inter_node`（跨服务器）
- `intra_node_topology`：full-mesh / switch / ring（来自 `phase0_intake`）
- `intra_node_link_bandwidth_gbps`：单条链路单向带宽（如 28）
- `intra_node_links_per_npu`：每 NPU 的 HCCS 链路数（如 7）
- 跨服务器时还需写明 `inter_node_fabric`（RoCE / IB）和 `inter_node_bandwidth_gbps`

---

## 步骤 2：SHMEM 能力盘点与复用映射

目标不是列 API，而是判断每个需求点如何落到 SHMEM 现有能力。

对每个需求点给出唯一分类：

| 分类 | 含义 | 必填信息 |
| --- | --- | --- |
| 可复用 | 已有 API、模板或 example 可直接使用 | 文件/API/example 路径，以及为什么匹配 |
| 需适配 | 已有能力基本匹配，但需要参数化、封装或调度调整 | 适配点、约束和风险 |
| 需新实现 | 现有能力不能覆盖 | 缺口、候选设计、fallback 和验证方式 |

能力映射至少覆盖六类：

- lifecycle：init/finalize、PE/team、bootstrap、资源生命周期
- memory：对称内存、地址翻译、本地/远端/UB buffer、输出可见性
- transport：MTE、RDMA/RoCE、SDMA、RMA put/get、非连续或 chunk 搬运
- sync：signal/wait/test、barrier/sync、quiet/fence、顺序约束
- compute：本地 AscendC/CATLASS 计算、dtype cast、accumulation、reduce
- scheduler：phase 顺序、tile/chunk、分核、stream/async/overlap

`design.md` 必须包含 Capability Mapping & Gap Analysis 合并表。凡是”需新实现”，都必须在同行写明 gap 原因、设计方案、验证方式和 correctness/performance risk。

---

## 步骤 3：执行方案设计

在 DSL 的 `schedule`、`correctness` 和 `performance` 中填写完整执行方案。

设计前必须读取 [references/core-allocation.md](references/core-allocation.md)，并优先复用其中总结的分核模式（A/B/C）、examples 分核经验和动态 tiling 参数。

**MUST** 在进入 core_partition 设计前，完成拓扑并行度分析：
1. 读取 DSL `topology`：明确 `intra_node_topology`（full-mesh/switch/ring）、`intra_node_links_per_npu`、`intra_node_link_bandwidth_gbps`
2. 计算拓扑的**理论最大并行链路数**（如 full-mesh 8 卡 = 7 条独立 HCCS 链路每 PE）
3. 设计时确保通信 phase 中**同时活跃的链路数**尽量接近最大并行链路数——若设计中 peer 维度串行（如 `for peer in range(n_pes-1)` 每次只 `get_nbi` 一个），必须显式标注为瓶颈并在 Design Review 中记录原因
4. 优先方案：SetAtomicAdd（MTE 批量 in-flight）、按 peer 分核独立搬运、多 lane 交错等

DSL `schedule` **按以下顺序设计**（先拓扑分析 → 分核 → 再 tiling → 最后 phases）：

1. **core_partition（最先设计）**：
   - `mode`：引用 core-allocation.md 的模式（A 全员协作 / B 按 PE 分组 / C1 生产者消费者 / C2 维度分组 / C3 引擎分组 / CoC 通算融合）
   - `groups`：每组核的编号范围、职责、使用的数据接口。**职责描述必须精确到以下三要素，禁止笼统概括**：
     1. **AIV→数据区间映射**：每个 AIV 负责哪段数据，写出 `offset` 和 `len` 公式（及尾块处理方式，参考 core-allocation.md §6）
     2. **AIV→PE 映射**：每个 AIV 负责哪个/哪些 PE 的数据（写出 PE 编号映射公式或迭代顺序，如 `for peer in [my_pe, (my_pe+1) % n_pes, ...]`）
     3. **本地 vs 远端路径**：self PE 使用本地快路径还是与其他 PE 走同一路径，明确写出
   - 设计时必须参照 core-allocation.md §2 对应模式的分核公式（如模式 B 的 `pe_idx = aiv_idx / core_per_rank`）填写上述三要素
   - `overlap`：组间如何重叠（并行 / 2-stage pipeline / 无）
   - `sync_between_groups`：组间同步方式（cross-core flag / barrier / signal-wait / quiet）
   - `rationale`：引用 core-allocation.md 或 shmem/examples 路径
   - `cross_pe_dataplane`：跨 PE 数据面接口（必须是 `aclshmem_*` 或 `aclshmemx_*`）

2. **tiling（分核之后设计）**：每组核的 tile、chunk、tail 处理方式

3. **phases（最后设计）**：device 侧大体执行流程，每个 phase 仅需 `name` + `action`（一句话中文描述），不需要写 input/output/dependency/sync 等微观细节

DSL `correctness.invariants` 每项必须含 invariant、test_method、case。

DSL `performance` 必须填写 baseline_search、baseline_target、min_scale。

**baseline_search 必须记录的搜索步骤**（参考 [../shmem-ops-performance-eval/references/baseline-selection.md](../shmem-ops-performance-eval/references/baseline-selection.md) 的 Baseline 选择决策树）：

1. 查找 HCCL 集合通信算子：AllGather、AllReduce、ReduceScatter、**AllToAll**、Broadcast、Reduce、Scatter、Gather
2. 查找 aclnn 扩展算子：aclnnAlltoAllV 等
3. 是否可用 HCCL + CATLASS/AscendC 串行拼接（如 AllToAll + MatMul）
4. 以上均无 → 填写 `"metric_only"` 并记录搜索过程

> **常见遗漏**：HCCL 库包含 AllToAll 算子（`HcclAlltoAll`）。设计时不要误认为这些算子没有 baseline。

### 算子实现边界

设计文档必须明确 Device kernel、`main.cpp`、Host helper、Python scripts 四层职责边界，且 kernel **默认 unified single path**（见 [references/implementation-boundary.md](references/implementation-boundary.md)）。

`schedule.core_partition` 必须说明 Device kernel 的分核策略，不能写成"Host 侧执行"。

---

## 步骤 4：生成契约输出

按 [references/design-template.md](references/design-template.md) 输出到算子目录的 `docs/design.md`（`docs/` 目录若不存在则创建）。如果用户指定路径，写到指定路径；否则写到当前算子/工程目录的 `docs/design.md`，无法判断时写到当前 workspace 的 `docs/design.md`。不得将 `design.md` 放置在算子根目录。

输出前必须完成设计走读，走读项见 [references/design-template.md](references/design-template.md) section 5。走读失败时先修订 DSL 的 schedule、correctness 或 performance 字段，再输出 `design.md`。

### 语言规范

描述性字段使用中文，API/dtype/字段名/代码标识符使用英语。详见 [references/op-dsl.md](references/op-dsl.md) 的语言使用规则。

`design.md` 必须包含：

1. 算子接口与 assumptions
2. Canonical DSL（扩充后的 yaml block，唯一结构化信息源）
3. Capability Mapping & Gap Analysis（合并表）
4. 实现边界（Device/Host/Python 职责）
5. **Design Review Before Handoff**（12 项检查表，必须逐项填写 pass/fail 和理由）
6. compile/test contract
7. handoff checklist for `shmem-ops-code-gen`

Section 5 不可省略。缺少此 section 的 design.md 不能交给 `shmem-ops-code-gen`。

### 结构完整性自检

输出 design.md 前，逐项检查以下必须存在的结构块：

| 结构块 | 标识 | 必须包含 |
| --- | --- | --- |
| Canonical DSL yaml | ` ```yaml ` 代码块 | interface、semantics、topology（含 deployment/拓扑类型/链路带宽）、memory、schedule（含 core_partition/tiling/phases）、correctness（含 invariants/case_matrix）、performance（含 baseline_search/baseline_target/min_scale） |
| Capability Mapping | 六列表格 | 覆盖六个能力域：lifecycle/memory/transport/sync/compute/scheduler（各域通过 `schedule.core_partition`、`memory`、`semantics` 等 DSL 字段表达，sync 和 transport 不设独立 DSL section） |
| Design Review | section 5 标题 | 12 项检查表，逐项 pass/fail |
| Compile/Test Contract | contract section | cann_env、build_mode、test_command |
| Handoff Checklist | section 7 | 逐项检查 |

任一结构块缺失 → 补齐后再输出。不得以"简化版"、"benchmark 不需要"等理由省略。

### Canonical DSL

必须用 fenced `yaml` block 输出完整 DSL，不能只用 prose 概括。DSL 是设计的唯一结构化信息源：接口、语义、拓扑、内存、调度（core_partition/tiling/phases）、正确性（invariants 含 test_method/case、case_matrix 含 stress case）、性能（含 baseline_search/baseline_target/min_scale）全部在 DSL 中定义，不在 DSL 之外另设重复的 prose 表格。

### Capability Mapping & Gap Analysis

必须包含如下列：

| Requirement | SHMEM capability/source | Classification | Adaptation / Gap | Risk |
| --- | --- | --- | --- | --- |

`Classification` 只能是 `可复用`、`需适配`、`需新实现`。"需新实现"行必须在 Adaptation / Gap 列写明 gap 原因、设计方案和验证方式。

映射至少覆盖六个能力域。sync 和 transport 在 DSL 中通过 `schedule.core_partition.sync_between_groups` / `cross_pe_dataplane` 表达，不设独立顶层 section。

`design.md` 必须包含 Capability Mapping & Gap Analysis 合并表。凡是"需新实现"，都必须在同行写明 gap 原因、设计方案、验证方式和 correctness/performance risk。

---

## 质量门禁

按 [references/design-template.md](references/design-template.md) section 7 的 Handoff Checklist 逐项检查。额外要求：

- DSL `correctness.case_matrix` 和 `performance.target_cases` 含中等规模 case（集合通信 256MB 级，计算/通算融合 hidden size 上千级），或标记未满足。
- DSL `schedule.core_partition.rationale` 已参考 [references/core-allocation.md](references/core-allocation.md)，或说明不适用原因。
- DSL `schedule.core_partition.groups[].role` 包含精确的 AIV→数据区间映射公式（含尾块处理）、AIV→PE 映射公式（或迭代顺序）、self/remote 路径差异，不可仅为模糊描述。
- assumptions、open questions、gap 没有隐藏在模糊 prose 中。

---

## MUST检查

- [ ] `op_name`、dtype、网络拓扑、`meta.torch_required`、`meta.performance_required`、`meta.performance_auto_optim` 均已从 `phase0_intake` 读取
- [ ] design.md 存在且非占位
- [ ] 完整 Canonical DSL yaml block（含 interface、semantics、topology、memory、schedule、correctness、performance）
- [ ] DSL `topology` 含 deployment、拓扑类型、链路带宽（来自 `phase0_intake`）
- [ ] Capability Mapping 覆盖六个能力域（lifecycle/memory/transport/sync/compute/scheduler），各域在 DSL 中有对应字段表达
- [ ] `schedule` 默认 unified single path；无 profiling ≥5% 收益证据则不在设计中规划 size 分支（见 implementation-boundary.md）
- [ ] Capability Mapping & Gap Analysis 合并表完整
- [ ] correctness invariants 每项含 test_method 和 case
- [ ] performance baseline_search 已逐步排查
- [ ] compile/test/perf contract 可执行
- [ ] Design Review Before Handoff（section 5）12 项检查已填写且无 FAIL
- [ ] 结构完整性自检通过（DSL yaml、Capability Mapping、Design Review、Contract、Handoff Checklist）

**全部通过 → 可交给 `shmem-ops-code-gen`**

---

## 反模式（NEVER DO THESE）

- ❌ 未从 `phase0_intake` 读取 `op_name` 或 dtype 就进入设计
- ❌ `phase0_intake` 中 `op_name` 或 dtype 缺失时自行脑补而非报告编排器
- ❌ 把缺失的关键语义写进 `assumptions` 替代从 `phase0_intake` 读取
- ❌ 只写概念性 prose 不输出 Canonical DSL yaml block
- ❌ 在 DSL 之外另建重复的 Phase 表、Tiling 节、分核节或 Correctness Invariants 节
- ❌ 通信模块写 `DataCopy` 直接读写远端 PE 地址
- ❌ `core_partition` 写"后续优化再并行"或"Host 侧执行"
- ❌ `groups.role` 只写模糊描述如"所有 AIV 并行处理""每个 AIV 负责连续区间"，不给出 AIV 编号→数据范围→PE 映射的具体公式
- ❌ `groups.role` 不区分 self PE 本地快路径和 remote PE 远路径
- ❌ `main.cpp` 设计为 fork/spawn 多 PE 或承载复杂逻辑
- ❌ 性能路径全部标为后续 gap 但不说明本轮为何仍可交付
- ❌ 省略 Design Review Before Handoff（section 5）
- ❌ DSL `performance.baseline` 写 `"none"` 或留空不记录搜索过程
- ❌ DSL `performance.baseline` 未逐步排查 HCCL 清单、aclnn 清单就直接写 `"metric_only"`
- ❌ DSL `performance.metric` 缺少 **`kernel_bus_bandwidth_GBps`**（Phase 6 达标主指标）
- ❌ DSL `performance.metric` 只写 `latency_us` 不包含 `kernel_bus_bandwidth_GBps`
- ❌ DSL `performance.metric` 不区分 `e2e_latency_us` 和 `kernel_latency_us`（通信算子必须双指标）
- ❌ 用户提供伪代码时粗略浏览后直接照搬实现，不逐段拆解和 SHMEM 语义映射
- ❌ 将伪代码中的"共享变量""全局数组"直接当作 SHMEM 可寻址内存
