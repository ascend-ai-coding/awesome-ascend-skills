# Step 2: Blaze Investigation

> **定位**：从最高层 `project_root` 发现并只读验证 `<project-root>/ops-tensor/`，围绕需求语义建立可核验的源码事实报告。设计请求可以跳过 Step 1；Step 2 不依赖 Step 1 文件产物，不推荐项目方案、不匹配场景、不选择路线、不编译或运行设备。

## 1. 输入、输出与边界

```text
输入：project_root、operator_name、用户需求、target_chip、npu_arch；可选 cann_version、Step 1 抽象版本记录、Step 3 语义化补充调查问题
blaze_source_root：<project-root>/ops-tensor/
operator_root：<project-root>/operators/<operator_name>/
输出：operators/<operator_name>/docs/blaze/blaze-investigation-report.md
模板：/ascendc-blaze-best-practice/references/blaze-investigation/investigation-report-template.md
```

- 从 `project_root` 和 `operator_name` 派生全部路径。没有 Step 1 handoff 时，自行只读建立相同的抽象源码一致性记录；有 handoff 时核对当前 checkout 与记录一致。
- 写报告前只创建 `<project-root>/operators/<operator_name>/docs/blaze/` 及缺失的父目录；这不建立 Step 1 的工程骨架，也不创建任何实现文件。
- 确认 `ops-tensor/` 和 `operators/` 同级，remote 指向授权仓库，当前 worktree/branch 可核验，必要递归 submodule 已初始化且与父仓 gitlink 一致。
- 源码或 submodule 缺失、不一致时停止，给出由 Step 1 或 direct init 恢复的条件；不得在 Step 2 clone、pull、checkout、切换版本或执行 submodule update。
- 只读核对项目内 `op_kernel/include/blaze/`、`tensor_api/` 是否存在且与当前根源码同源。设计请求跳过 Step 1 时，副本缺失不阻塞 Investigation，也不由 Step 2 复制；在报告中记录 `missing_for_exact_materialization`，由 Step 3 为实施阶段编译一次性原样物化 action。已存在但来源或内容不一致则记录 `blocking`。
- 不回退搜索 `operators/<operator_name>/`、其他项目或历史工程中的 ops-tensor。
- 调查范围来自数学语义、dtype、format、layout、topology、shape、额外输入输出和明确约束，而不是预设场景名称。
- Blaze 源码事实只来自当前 `<project-root>/ops-tensor/` 的源码、官方文档、example 和 UT。Asset、项目实现、其他 Skill 和历史验证工程都不是 Blaze 源码能力证据。
- 不读取 `references/scenarios/index.md`、任何场景设计/开发指导或依赖 Skill；Step 2 不知道项目是否将采用定制场景。
- 不读取 `environment.md`、外部 workflow manifest、DESIGN/PLAN 模板或其他 direct invoke 产物。芯片事实必须由输入直接提供，不追溯外部工作流文件。
- 根 `ops-tensor` 始终只读；项目内官方副本存在时只读核对，缺失时只记录物化状态。候选扩展必须有边界；同一 Investigation 报告最多接受一次来自 Step 3 的补充调查。
- 报告版本字段使用当前自检或 Step 1 handoff 的抽象版本记录；不得写实际提交 ID、manifest/hash、文件哈希、构建时间戳或具体提交节点。源码变化会使已有 Investigation 失效。

## 2. 五阶段 Runbook

| 阶段 | 必读文档 | 动作 | 报告写入 | 停止条件 |
|---|---|---|---|---|
| 0. 建立调查 Brief | [报告模板](../blaze-investigation/investigation-report-template.md) | 派生路径并只读自检 Blaze 源码；将需求投影为分区、约束、问题和读取边界 | 路径/版本状态、需求投影、调查范围、补充范围 | 源码缺失/不一致或需求语义不足时停止 |
| 1. 候选组装方案发现 | [Blaze 源码调查方法](../blaze-investigation/source-investigation-method.md) | 先从 example/UT/真实 Kernel 入口找种子，再回到定义与 specialization | 候选目录、发现证据、读取边界 | 每个已调查语义有 found/not_found/unknown 记录 |
| 2. Blaze 组装方案闭合 | [Blaze 组装方案恢复方法](../blaze-investigation/assembly-recovery-method.md) | 依真实调用恢复 Kernel、Policy、BlockMmad、BlockScheduler、可选 Epilogue、Tiling/Params | 具体证据、成员链、结构状态 | 每条候选标为 complete/partial；禁止名称拼接 |
| 3. 依赖与物理合同闭合 | [依赖追溯方法](../blaze-investigation/dependency-trace-method.md) | 只追踪候选实际消费的 Params、Tensor API、物理数据和 host/device ABI | 字段语义、API、物理数据、ABI、已观察限制和未闭合事实 | 调用点/定义点/字段语义闭合或明确 unknown |
| 4. 证据审计与报告冻结 | [调查合同审计方法](../blaze-investigation/design-contract-method.md)、报告模板 | 审计证据、冲突、读取边界和未闭合事实 | 唯一事实报告、证据账本、补充记录 | 不产生支持、路线或场景结论 |

只在进入对应阶段时读取该阶段文档。Step 2 不提前加载 DESIGN 模板、PLAN 模板、场景指导或依赖 Skill。

## 3. 阶段 0：建立调查 Brief

先只读核对根级源码：

```bash
git -C <project-root>/ops-tensor remote -v
git -C <project-root>/ops-tensor status --short --ignore-submodules=none
git -C <project-root>/ops-tensor submodule status --recursive
git -C <project-root>/ops-tensor ls-tree HEAD include/tensor_api
```

不得在 Step 2 执行会改变 checkout 的命令。缺仓、remote 不匹配、未初始化 submodule、gitlink 不一致或布局不是 `ops-tensor/` 与 `operators/` 同级时停止，并将恢复目标指向 Step 1/direct init。

从需求提取：

```text
request_summary
math_contract
tensor_roles_and_extra_io
dtype_format_layout_constraints
topology_and_shape_predicates
runtime_compile_time_axes
demand_partitions
hard_constraint_manifest
required_source_questions
out_of_scope
candidate_expansion_budget: 1
supplement_scope: absent | one semantic request
```

Basic、Batch、Grouped、Quantized、MX、layout、transpose 和 shape 是可组合的需求画像。只有会改变真实入口、specialization、ABI 或物理数据合同的轴才拆分 demand partition；shape 用谓词域描述，不枚举用户未要求的数值。发现新的 compile-time 轴时，记录 `brief_amendment`，使受影响事实重新核验，不能借拆分缩小用户范围。

“额外输出后处理”“额外 operand”“广播关系”等只是需求语义探针：它们要求调查相关数据流、物理映射或 AIC/AIV 协作事实，但不对应注册场景，也不预测路线。

### 补充调查

Step 3 仅在缺少决定性 Blaze 源码事实时，向下一轮同一报告写入一次 `supplement_scope`。它必须包含：

```text
requested_by: Step 3
attempt: 1
affected_requirement_ids
semantic_questions
required_blaze_source_frontier
why_decision_is_blocked
```

问题只描述待确认的需求语义、接口、物理数据或调用关系；不得出现场景 ID、场景文档路径、custom 路线或实现建议。补充完成后仍缺关键事实时，报告照实记录 `unresolved_source_facts`，由 Step 3 停止等待用户澄清。

## 4. 阶段 1：候选组装方案发现

按 [Blaze 源码调查方法](../blaze-investigation/source-investigation-method.md)：

1. 从与需求语义相近的 example、UT 和真实 Kernel 入口寻找 using、alias、实例化及调用点。
2. 回到公开定义解析 specialization、模板参数、Params 和架构守卫。
3. 未找到执行种子时，才从目标公开 Kernel/Policy/Block 反向寻找实际 caller。
4. 仅为解释真实调用而读取绑定官方 API 文档和构建文件。
5. 不按相似文件名无边界扫描，也不把未引用组件拼成候选。

每个候选记录：

```text
candidate_id
investigated_partition_ids
candidate_result: found | not_found | unknown
concrete_entry_seed
coverage: indexed | deep | out_of_scope
applicability_observations
source_refs
```

`not_found` 表示在已声明读取边界没有发现对应种子，不等于 Blaze 官方库不支持；`unsupported` 只能记录源码或官方约束明确拒绝的准确形态。

## 5. 阶段 2：Blaze 组装方案闭合

沿同一真实入口恢复实际调用链：

```text
Kernel entry
  -> Policy / specialization / dispatch
  -> BlockMmad
  -> BlockScheduler
  -> optional Epilogue
  -> TilingData / Params / Tiling entry
```

每条候选绑定具体 witness、成员、适用 partition、结构状态和 source relationships。Grouped Plain、Grouped Quantized、Grouped MX 必须分别取得与激活分区一致的 witness，不能用 Grouped MX 代替其他模式。

候选不能闭合时，只允许一次有边界的扩展：先检查同一真实入口的另一 specialization，再检查该入口实际引用的同族组件。扩展后仍不能确认时保留 `partial` 或 `unknown`，不猜测模板参数或组件组合。

## 6. 阶段 3：依赖、物理数据与 ABI

只追踪已闭合候选实际消费的依赖。每个 Scheduler/Block/Policy/Kernel/Epilogue 字段记录：

```text
field_path
field_type_and_unit
consumer_and_semantics
legal_domain_or_predicate
cross_field_constraints
tilingdata_mapping
observed_value_scope
evidence_refs
evidence_status: source_observed | documented | example_assembled | conflict | unknown | unsupported
```

同时记录实际 Tensor API 调用点与定义、Routing、location/layout/dtype、shape/offset/stride/extent/alignment 单位、逻辑/物理转换、buffer/workspace 生命周期、Kernel entry、GM 参数、TilingData、grid 和 host ownership。不要建立 API 百科。

没有现成 Blaze 源码 host Tiling 不阻塞调查。要闭合的是 Scheduler Params 的语义、合法域、约束和 ABI；`observed_value_scope` 只能说明源码观察到的值，不能替项目选择固定值。

若需求要求 MatMul 输出后逐元素或 broadcast 数据处理，调查与已发现候选实际相关的输出位置、物理 mapping、完成时机、C+V/adapter/同步事实。不能从需求名称推断这些实现存在，也不能将其组织为场景 bundle。

## 7. 阶段 4：证据审计与报告冻结

按 [调查合同审计方法](../blaze-investigation/design-contract-method.md) 审计：

- 每个调查问题都有事实、来源明确的限制、或 `unknown`；
- 每个 concrete candidate 有真实入口、组件链、依赖与适用边界；
- 已观察限制与未闭合事实分开记录；
- `coverage` 只表示调查深度，`evidence_status` 只表示原子证据状态，`object_readiness` 只表示候选或证据对象的局部事实闭合度；三者均不构成官方支持、项目可执行性或项目路线结论；
- 报告清楚列出实际读取根、未读取范围、候选扩展和补充调查历史。

报告不得聚合为官方支持结论、可执行性、候选组合、场景 ID、场景 bundle 或项目路线。Step 3 依据精确需求重新评估这些事实；未调查不等于不支持，源码证据充分也不等于设备精度通过。

## 8. 完成

关闭报告前确认至少绑定：

```text
project_root
operator_name
operator_root
report_path
blaze_source_root
target_chip
npu_arch
cann_version: <optional>
investigation_id
source_version_status
checkout_consistency
read_only_source_regions
read_only_source_region_status
read_only_source_region_evidence_refs
source_refs
candidate_evaluation_id
assembly_witness
assembly_members
dependency_and_physical_data_facts
abi_and_signature_facts
unresolved_source_facts
```

报告只记录事实和缺失事实，不写 native/custom/unsupported 结论。核对模板、链接、读取边界、证据账本、未闭合事实和补充次数后进入 [Step 3: Kernel Design](step3-kernel-design.md)。
