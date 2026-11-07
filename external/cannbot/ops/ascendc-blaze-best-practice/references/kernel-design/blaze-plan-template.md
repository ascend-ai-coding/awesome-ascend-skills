# {operator_name} 算子开发计划

> `{operator_name}` → 实际算子名称。本文档在开发流程中持续更新。

本模板由 `/ascendc-blaze-best-practice` Step 3 在可执行 DESIGN 冻结后使用，生成项目级 `operators/<operator_name>/docs/PLAN.md`。第 1、3 章是设计基线，第 9、10 章是冻结开发合同，第 11 章只追加执行记录；第 2、4--8 章由实施阶段持续更新。

> 仅 `implementation_route=blaze_native` 或 `implementation_route=blaze_custom` 可以生成 PLAN。`unsupported` 只生成阻塞 DESIGN；一次补充 Investigation 后仍缺关键事实时不生成 DESIGN/PLAN。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | {operator_name} |
| 数学公式 | y = f(x) |
| 输入 | x1: shape=[...], dtype=... |
| 输出 | y: shape=[...], dtype=... |
| 算子类别 | MatMul / Batch MatMul / Quantized MatMul / MatMul Fusion / ... |
| 需求类型 | 特定用例 / 通用 |

本章是 DESIGN §0.2、§0.4 和 §1.2 的人读投影。需求语义由设计阶段所有；实施阶段发现变更时在第 5、11 章记录 `design_issue` 并返回设计阶段，不直接改写本章。

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/{operator_name}_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `kernel/{operator_name}_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/{operator_name}.asc` — Host + main 入口 | ⬜ |
| `torch_library/{operator_name}_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

Step 3 按当前 DESIGN 和 §9.4 实例化实际文件行，不保留不适用的示例文件。实施阶段更新状态，并可追加为实现冻结设计产生的实际项目文件；新增项同时记录到第 11 章。每一行映射 §9.4 的 `target_file_manifest.target_file`。

---

## 3. 测试计划

精度标准：来自 DESIGN §3.5 的 `comparison_threshold_and_nonfinite_gate`。

**Golden 计算**：实现路径和 dtype/计算顺序来自 DESIGN §3.5，并在 §9.7 `data_and_golden_wiring` 中绑定实际文件。

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：
测试用例必须可追溯到 DESIGN.md §0.2 用户原始需求（覆盖率需 100%）。Tn/Pn 使用相同逻辑输入和同一 Golden；不适用 PyTorch 时仍保留 Pn 和对应章节，状态填写 `N/A` 并说明跳过原因。

| 编号 | 用例 | 需求追溯 | 输入 | 预期输出 |
|-----|------|---------|------|---------|
| T1/P1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | randn(...) | y=f(x) |
| T2/P2 | 零值 | （同上） | zeros(...) | y=f(0) |
| T3/P3 | 边界/tail | （同上） | 按 DESIGN 边界合同 | y=f(x) |

本章是 DESIGN §3.5 与 §9.8 `validation_checkpoints` 的人读投影，测试范围、追溯和验收标准由设计阶段所有。需要改变时记录 `design_issue`，不得由实施阶段自行扩缩支持域。

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T3 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.{operator_name}()` 可调用 + P1-P3 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

本章由实施阶段持续更新，汇总 §9.5 actions、§9.8 checkpoints 和 §9.9 deliverables 的执行进度；不在冻结合同中维护第二套状态。

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|

实施阶段持续追加问题、决策、`design_issue` 和偏离，不覆盖历史行。改变路线、Blaze 组装方案、接口/ABI、支持范围或验证标准的问题必须返回设计阶段。

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 用例 | 需求追溯 | 结果 | Max Diff |
|-----|------|---------|------|----------|
| T1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | ⬜ | |
| T2 | 零值 | （同上） | ⬜ | |
| T3 | 边界/tail | （同上） | ⬜ | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 用例 | 需求追溯 | 结果 | Max Diff |
|-----|------|---------|------|----------|
| P1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | ⬜ | |
| P2 | 零值 | （同上） | ⬜ | |
| P3 | 边界/tail | （同上） | ⬜ | |

PyTorch 通路不适用时不得删除本节；逐行填写 `N/A`，并在 §6.3 和第 5 章记录跳过原因。

### 6.3 产物 & 执行状态

- [ ] `build/{operator_name}` 可执行文件存在
- [ ] `build/lib{operator_name}_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.{operator_name}` 可调用

| 通路 | 状态 | 运行时间 | 跳过原因 |
|------|------|---------|---------|
| 可执行文件 | ⬜ | | |
| PyTorch | ⬜ | | |

---

## 7. 性能验收

**状态**: ⬜ | **数据**: docs/perf/round_NNN/

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | | |
| Block Dim | | |
| 主导流水 | | |

**达标判定**: ⬜ | **理由**:

本章映射 §9.8 的性能 checkpoint；实施阶段只填写实际证据，不改变 DESIGN 的性能验收标准。

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| 可执行文件 | | | | ⬜ |
| PyTorch | | | | ⬜ |
| 性能 | | | | ⬜ |

本章只汇总第 6、7 章，不维护独立结果事实。

---

## 9. Blaze 开发合同

### 9.1 PLAN Metadata 与 DESIGN 绑定

```text
plan_template_provider: /ascendc-blaze-best-practice
plan_template_path: /ascendc-blaze-best-practice/references/kernel-design/blaze-plan-template.md
project_root:
operator_name:
operator_root: <project-root>/operators/<operator_name>/
project_contract_id:
design_contract_id:
plan_contract_id:
design_path: operators/<operator_name>/docs/DESIGN.md
investigation_report: operators/<operator_name>/docs/blaze/blaze-investigation-report.md
implementation_route: blaze_native | blaze_custom
selected_scenario: <only blaze_custom>
design_guide_source: <only blaze_custom>
development_guide_source: <only blaze_custom>
selected_candidate_ids:
source_version_status:
checkout_consistency: confirmed | blocking
design_plan_consistency: confirmed | blocking
design_consistency_marker:
plan_consistency_marker:
selected_design_binding_refs:
kernel_abi_contract_refs:
abi_crosswalk_refs:
source_backed_signature_skeleton_refs:
abi_crosswalk_delta_refs: <only blaze_custom when applicable>
allowed_change_scope_ref:
forbidden_change_scope_ref:
verification_contract_ref:
plan_owner: design_stage
execution_owner: implementation_stage
frozen_plan_status: ready | blocking
```

字段说明：`assembly_witness` 表示 Blaze 组装方案真实证据；`assembly_members` 表示具体成员；`candidate_evaluation_id` 表示候选组装方案评估标识。PLAN 只引用冻结 DESIGN 的首选合同，不能产生新路线或新候选。

`frozen_plan_status=ready` 只表示第 9、10 章和第 1、3 章设计基线已冻结，不表示整份 PLAN 只读。

### 9.2 来源、阅读清单与依赖 Skill

`reading_manifest` 每项包含：

```text
reading_id
source_ref
canonical_path
location
read_before_action_ids
purpose
required_or_conditional
```

依赖 Skill 先登记根入口，再登记必要叶子。`blaze_native` 不登记场景资料；`blaze_custom` 只登记唯一场景和 DESIGN 已选择路线所需资料。`reading_manifest` 是 Step 4 的初始阅读基线，不是临时诊断和实现修复的封闭资料清单；新增读取只作为执行证据记录，不能改变冻结设计。

### 9.3 路线、前置条件与禁止项

```text
selected_design_binding_refs
scenario_guidance_compliance
preconditions
abi_preconditions
abi_action_coverage
blocking_conditions
forbidden_decisions_in_step4
forbidden_files
```

`abi_preconditions` 必须确认 DESIGN 的 `abi_mapping_draft`、每个首选 binding 的 source-backed `kernel_abi_contract`、`abi_crosswalk`、`source_backed_signature_skeleton` 和（适用时）`abi_crosswalk_delta` 已闭合。`abi_action_coverage` 必须按 `design_binding_ref` 和 `crosswalk_row_id` 将每个必需 crosswalk 行映射到其 buffer、Tiling/Params、Wrapper、Kernel entry、Launcher 或验证 action/checkpoint；场景增量同时记录 `delta_crosswalk_row_id`。

`blaze_native` PLAN 不包含场景 action、custom fallback 或场景依赖。`blaze_custom` PLAN 将已选 design/development guide 的每项激活要求映射到 DESIGN 合同、action、checkpoint、deliverable 或有依据的 `not_applicable`。场景是否消费 `matmul_base_analysis` 由 DESIGN 的 `consumed_contracts` 决定，PLAN 不自动加入。

项目内 `blaze/`、`tensor_api/` 副本必须满足以下前置之一：已由 Step 1 建立且同源一致；或 DESIGN 对缺失副本明确授权一次 `create_only_exact_read_only_materialization` action。不得把副本缺失解释为允许直接从根源码构建、重新 clone 或修改官方内容。

### 9.4 目标文件与交付范围

`target_file_manifest` 每项包含：

```text
file_id
target_file
file_role
action_type: create | copy_and_adapt | modify | delete | read_only | forbidden
source_refs
design_contract_refs
allowed_change_scope_ref
expected_artifact
```

不预设固定工程树或脚本数量。PLAN 中的 `target_file` 和 action `target_files` 是初始计划与审计基线，必须以 `operators/<operator_name>/` 开头；Step 4 为实现冻结设计新增或调整的项目文件更新第 2 章并写入 `execution_record`，不要求先回写冻结合同。项目外 Skill 文档、Asset 和 Blaze 根源码只能登记为只读来源；项目内官方副本除已授权的缺失目录 create-only 原样物化外为 read_only/forbidden，物化完成后禁止内容修改。

缺少项目内官方副本时，分别登记精确的 `action_type: create` 目标，并引用当前 Investigation、`blaze_source_root` 和 DESIGN 的 create-only scope；`expected_artifact` 必须是与当前根源码同源、内容未适配的只读副本或绑定。副本已存在且核对通过时登记为 `read_only`，相应物化 action 写有依据的 N/A。不得使用 `copy_and_adapt` 处理官方副本。

### 9.5 有序开发动作

`ordered_actions` 每项包含：

```text
action_id
sequence
design_contract_refs
abi_contract_refs: <required for buffer/Tiling/Params/Wrapper/Kernel/Launcher actions>
  design_binding_ref
  kernel_abi_contract_ref
  crosswalk_row_refs
  source_backed_signature_skeleton_ref: <required for Wrapper/Kernel entry/Launcher binding>
  delta_crosswalk_row_refs: <only active blaze_custom additions>
source_refs
read_before_action_ids
target_files
action
prerequisites
expected_output
verification
failure_rollback
```

顺序和前置关系必须确定。初始动作只能消费首选合同；不能只写“参照文档实现”。所有涉及物理 buffer、Tiling/Params、Wrapper、Kernel entry 或 Launcher 的计划动作必须引用其 `design_binding_ref`、`kernel_abi_contract` 和 `crosswalk_row_id`；涉及 Wrapper、Kernel entry 或 Launcher 启动绑定的动作还必须引用相应 `source_backed_signature_skeleton`。场景增量还必须引用相应 `delta_crosswalk_row_id`。PLAN 中的复制、建目录、生成、修改和删除动作应显式描述预期工作，但 Step 4 为闭合冻结设计进行的额外项目内修复不要求补 action。CMake configure/build action 使用同一组通用字段，写明构建目标、预期结果、verification 和失败后的排障入口。

若存在官方副本物化 action，它必须先于任何读取项目副本、构建或实现 action：只从当前 `blaze_source_root` 原样复制或绑定，核对来源与内容一致，设置只读状态，并记录证据。失败时返回 Step 1；不得就地修补副本或改用其他源码根。

### 9.6 实现字段与跨文件接线

```text
implementation_wiring_contract:
  kernel_entry_and_wrapper
  template_and_specialization_binding
  component_and_policy_binding
  params_and_tilingdata_mapping
  host_tiling_provider_or_fixed_legal_values
  memory_layout_and_offset_mapping
  workspace_and_output_lifecycle
  abi_crosswalk_binding
  sync_and_event_wiring
  launcher_argument_and_buffer_wiring
  custom_copy_and_adaptation_scope
```

这些是 DESIGN 首选合同的实现投影，不得产生新设计。

若 PLAN 包含 CMake configure 或 build action，应把对应项目构建文件列入 `target_file_manifest` 作为初始计划。构建目标、预期产物、验证和失败处理直接写入 `ordered_actions` 与 `validation_checkpoints`，不增加专用编译合同、状态或 schema；Step 4 可以在项目根内调整或新增实现所需构建文件，并记录实际变更。

### 9.7 构建、Tiling、Launcher、数据与 Golden

```text
build_tiling_launcher_wiring
data_and_golden_wiring
```

明确构建入口和判据、Tiling/Params、Kernel/Wrapper/Launcher ABI、输入生成、逻辑到物理转换、CPU Golden、输出比较、临时产物和固定合法 Scheduler-control 值的实现与校验。构建 action 只冻结目标和验收结果；Step 4 可根据真实编译证据在项目根内调整或补充构建配置，但不得改变需求语义、路线、Blaze 组装方案、接口/ABI、支持域或验证标准。`abi_crosswalk_binding` 必须保留每个逻辑对象到物理 buffer、Launcher、Wrapper、Kernel 参数、Tiling/Params 和最终设备消费者的实施映射；不能只在文字说明中引用 ABI。

### 9.8 分层验证 Checkpoint

`validation_checkpoints` 每项包含：

```text
checkpoint_id
after_action_ids
design_verification_refs
test_scope
inputs
expected_result
evidence_to_record
failure_rollback
```

覆盖当前 DESIGN 要求的静态 ABI、构建、功能、边界/tail、布局/offset、资源/生命周期、场景诊断、设备精度、重复运行和最终回归。

每个涉及精度或设备的 checkpoint 必须在 `evidence_to_record` 中区分原始失败、
前置风险和环境阻塞，并保存执行上下文（sandbox/device-visible、设备节点、
架构、命令、返回码）。只有真实设备路径的完整构建和验证证据才能标记
`device_verified`；没有同合同性能基线时性能字段固定为 `NOT_EVALUATED`。

第一个相关 checkpoint 必须静态核对：必需逻辑参数和辅助 ABI 对象的物理字节/offset 单位、TilingData/Params、workspace、grid/usedCore、Wrapper/entry 绑定以及设备消费者均有闭合 crosswalk；缺失时不允许启动 Kernel。

### 9.9 最终交付件与清理

`deliverable_manifest` 每项包含：

```text
deliverable_id
path_or_artifact
source_action_ids
acceptance_checkpoint_ids
cleanup_requirement
submission_status_expected
```

`cleanup_contract` 明确计划内临时诊断代码、数据、构建产物和日志的处理动作与最终回归；Step 4 额外产生的临时文件也必须写入 `execution_record` 并在最终回归前清理。

### 9.10 失败停止与回退

- Blaze 源码版本不一致：停止并回 Step 1；
- Blaze 源码事实被推翻：保留证据并回 Step 2；
- DESIGN/PLAN 缺少启动实现所必需的设计事实：回 Step 3；
- canonical 场景资料、注册合同或依赖 Skill 自身冲突且阻碍执行：进入 Blaze skill 维护；
- 编译、链接、精度、边界或普通实现失败：Step 4 读取实际证据，在项目根内诊断、修复并重新执行受影响 checkpoint；实际问题、修改和结果追加到 `execution_record`；
- 实现修复不设置固定重试次数；只要仍有实现层证据和可执行的下一步，就继续修复闭环；暂时无法解决时记录 `failed`；
- 修复显示必须改变实现路线、Blaze 组装方案、接口/ABI、Tiling/Params 语义、支持范围或验证标准：回 Step 3；需要新的 Blaze 源码事实时先回 Step 2；
- 工具链、设备或权限暂时不可用：Step 4 记录 `blocked`，不把环境问题伪装成设计回退；
- 用户需求改变：停止执行，回需求合同和 Step 3。

不自动切换 DESIGN 中的备选。

**章节所有权**：第 1 章需求语义和第 3 章测试范围/追溯/验收标准由设计阶段所有；第 2 章文件状态、第 4 章进度、第 5 章问题决策、第 6--8 章结果由实施阶段持续更新；第 9、10 章冻结；第 11 章只追加。第 2 章映射 `target_file_manifest`，第 3 章映射 `validation_checkpoints`，第 4 章汇总 actions/checkpoints/deliverables，第 6 章复用第 3 章 T/P ID，第 7 章映射性能 checkpoint，第 8 章只汇总第 6、7 章。设计基线变化必须记录 `design_issue` 并返回设计阶段。

## 10. Step 3 冻结与 Readiness 门禁

- `implementation_route` 只能为 `blaze_native` 或 `blaze_custom`；
- 所有 action 字段完整，依赖顺序无环；
- 每个首选合同有 action 或无需动作的说明；
- 每个验证合同有 checkpoint；
- 每个必需 ABI crosswalk 行均有 action/checkpoint 覆盖；所有 ABI 相关 action 都已填写 `abi_contract_refs`；
- 存在 CMake configure/build action 时，PLAN 已提供项目构建入口、目标、预期结果、构建 checkpoint 和失败处理基线；
- 初始目标文件和资料路径均位于项目边界内，官方源码区和 Asset 原文件保持只读；
- 项目内官方副本已同源确认，或缺失副本具有先于构建/实现的 create-only 原样物化 action；
- PLAN 的资料与初始 action 具备 `source_refs`/`read_before_action_ids` 追溯关系，Step 4 追加的诊断资料和实际文件记录在 `execution_record`；
- 所有交付件可追溯；
- 无 TBD、未选分支、场景组合、备选自动回退或 unresolved blocking。

## 11. 执行和变更记录

Step 3 只建立空集合。Step 4 只能追加：

```text
execution_record:
  - execution_id
    action_id_or_checkpoint_id
    start_and_end_marker
    actual_files_changed
    actual_output
    result: completed | failed | blocked
    evidence_refs
    deviation
    rollback_performed
    next_action_or_return_target
```

任何改变需求语义、路线、Blaze 组装方案、接口/ABI、验证标准或支持域的 deviation 都必须停止；项目根内新增文件、实现步骤、action 执行调整、Tiling 合法值、资源/同步实现或配置修复应更新第 2、4--8 章并追加执行记录，不修改第 9、10 章或 DESIGN。
