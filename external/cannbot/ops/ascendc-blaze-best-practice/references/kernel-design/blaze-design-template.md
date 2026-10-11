# {operator_name} Blaze 算子设计文档

本模板是 `/ascendc-blaze-best-practice` 的唯一 DESIGN 模板，既用于 Skill 独立设计，也用于 direct invoke 的 Blaze 设计路线。Step 3 使用它生成：

```text
<project-root>/operators/<operator_name>/docs/DESIGN.md
```

本文中的 DESIGN 和 PLAN 分别指同一算子目录下的 `docs/DESIGN.md` 和 `docs/PLAN.md`。生成文档时必须替换全部占位内容，只记录当前用户需求、当前 Investigation 和已读取场景指导能够证明的事实；不得把模板示例、Asset、旧项目或候选名称当作能力、ABI 或验证证据。

- `blaze_native` 或 `blaze_custom`：完成本 DESIGN，通过门禁后使用 `blaze-plan-template.md` 编译 PLAN。
- `unsupported`：完成阻塞 DESIGN，不生成 PLAN。
- 一次补充 Investigation 后仍缺决定性源码事实：停止并说明待澄清项，不生成最终 DESIGN 或 PLAN。

---

## 0. 概述

### 0.0 需求类型判断

| 判断项 | 结论 | 依据 |
|-------|------|------|
| 需求类型 | 特定用例 / 通用 | 特定用例明确给出 shape 和 dtype；否则按通用需求处理 |
| 用户目的 | 算子设计 / 完整算子开发 | 只影响 Skill 流程路由，不改变本 DESIGN 合同 |
| 需求是否明确 | 是 / 否 | 需求不明确时先请求澄清，不得将其判为 `unsupported` |

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | `{operator_name}` |
| 算子类别 | `<按需求填写>` |
| 需求类型 | 特定用例（shape=`<...>`，dtype=`<...>`） / 通用 |
| 支持数据类型 | `<输入、累加、中间值和输出 dtype>` |
| 支持 shape/layout | `<逻辑 shape、物理 layout、transpose 和动态轴>` |
| 目标芯片 | `<target_chip>` |
| NpuArch | `<npu_arch>` |
| 编译参数 | `--npu-arch=<npu_arch>` |
| CANN 版本 | `<cann_version 或未指定>` |
| 特殊约束 | `<精度、性能、资源、接口或部署约束>` |
| Investigation | `operators/<operator_name>/docs/blaze/blaze-investigation-report.md` |
| 待澄清项 | `<无，或列出阻塞项>` |

### 0.2 用户原始需求

逐条原样记录用户需求，不加工、不裁剪、不合并。编号一经写入即保持稳定；下游 DESIGN 审查、PLAN 测试追溯、开发和验收均以本节为准。

| # | 需求内容 |
|---|---------|
| 1 | `<用户原文>` |
| 2 | `<用户原文；无则删除本行>` |

### 0.3 来源与 Investigation 绑定

以下为本 DESIGN 的唯一来源绑定合同：

```text
template_provider: /ascendc-blaze-best-practice
template_kind: blaze_design_contract
project_root: <project-root>
operator_name: <operator_name>
operator_root: <project-root>/operators/<operator_name>/
project_contract_id: <stable-project-contract-id>
design_contract_id: <stable-design-contract-id>
investigation_report: operators/<operator_name>/docs/blaze/blaze-investigation-report.md
blaze_source_root: <project-root>/ops-tensor/
investigation_id: <stable-investigation-id>
source_version_status: current_checkout | same_investigation_source | consistent
checkout_consistency: confirmed | blocking
read_only_source_regions:
  - <project-root>/ops-tensor/
  - operators/<operator_name>/op_kernel/include/blaze/
  - operators/<operator_name>/op_kernel/include/tensor_api/
investigation_fact_refs:
  - <fact-ref>
unresolved_source_fact_refs_considered:
  - <fact-ref，全部闭合时写 none>
```

每个 `source_ref` 必须稳定记录以下属性：

```text
source_ref
source_kind
canonical_path
location
purpose
observed_evidence
consuming_contract_refs
```

`assembly_witness` 表示实际观察到的 Blaze 组装证据，`assembly_members` 表示该 witness 的具体成员，`candidate_evaluation_id` 表示候选组装评估标识。它们必须引用当前 `blaze_source_root` 和当前 Investigation，不得由 Asset、项目实现、旧 DESIGN 或旧 PLAN 补证。

### 0.4 需求合同

将 §0.2 的每条原始需求拆成可验证的 required partitions。每个分区使用以下唯一合同：

```text
requirement_contracts:
  - requirement_id: <对应 §0.2 的稳定编号>
    partition_id: <stable-partition-id>
    requirement: <精确、可验证的需求>
    source_refs:
      - <用户需求或已确认技术事实引用>
    requirement_status: confirmed | assumed | unknown | blocking
    support_boundary: <支持域、拒绝域和边界>
    design_impact: <接口、实现、资源或验证影响>
```

需求合同的 partition 必须覆盖用户需求的完整算子语义。算子中的全部计算步骤（包括 matmul 及其前/后的所有非 matmul 计算）必须作为 required partition 纳入 §2.3 的官方方案覆盖性检查。算子的所有计算必须在 device 侧完成，不得拆分到 host 侧执行。

需求合同必须覆盖实际适用的以下方面，不得用场景名或候选名代替精确需求：

| 方面 | 需要冻结的内容 |
|-----|----------------|
| 数学与拓扑 | 公式、操作顺序、tensor 角色、额外输入输出 |
| 数据类型 | 输入、累加、中间值、转换/舍入顺序、输出 |
| shape 与布局 | 逻辑 shape、物理 layout、transpose、对齐、tail |
| 变化轴 | runtime/compile-time 轴、可选参数和默认值 |
| 质量目标 | 精度、性能、资源和稳定性目标 |
| 失败语义 | 拒绝条件、错误语义和明确排除范围 |

---

## 1. 算子设计

### 1.1 数学公式

先写逻辑接口和数学语义，再讨论物理数据或 Kernel ABI：

```text
输入:
  <logical_input_id>: shape=<...>, dtype=<...>, layout=<...>, role=<...>
输出:
  <logical_output_id>: shape=<...>, dtype=<...>, layout=<...>, role=<...>

数学公式:
  <逐步公式；明确广播、归约、累加和类型转换顺序>
```

| 步骤 | 数学操作 | 输入 dtype | 累加/计算 dtype | 输出 dtype | 对应 partition_id |
|-----|----------|------------|-------------------|------------|-------------------|
| `<n>` | `<操作>` | `<dtype>` | `<dtype>` | `<dtype>` | `<partition-id>` |

### 1.2 API 映射

本节同时服务人读评审与 Agent 合同。所有 API、组件和签名必须来自 §0.3 所绑定的 Investigation；不能从旧样例或预期名称推测。

| 数学操作 | 对应 API/组件 | 关键参数或 specialization | 数据布局 | 限制与拒绝条件 | source_ref |
|---------|--------------|--------------------------|---------|----------------|------------|
| `<操作>` | `<已验证 API/组件>` | `<实际参数语义>` | `<实际布局>` | `<限制>` | `<source-ref>` |

#### 1.2.1 API 语义验证

每个被选择的 API 或组件都要完成以下验证；未确认项必须返回 Step 2 补充 Investigation，不得在此猜测：

| API/组件 | 输入输出布局 | 功能需求 | 观察到的签名/组合 | 限制条件 | 匹配结论 | source_ref |
|----------|--------------|---------|-------------------|---------|---------|------------|
| `<名称>` | `<连续性、对齐、shape、layout>` | `<操作和维度>` | `<当前源码事实>` | `<dtype/shape/资源限制>` | confirmed / blocking | `<source-ref>` |

- [ ] 数据布局、连续性、对齐和物理 shape 已确认。
- [ ] 功能、操作维度、类型转换和输出形式已确认。
- [ ] 签名、specialization 和限制来自当前 Investigation。
- [ ] 需求位于 API/组件支持范围内；不匹配项已进入 `native_gaps`。
- [ ] 人读映射与下方接口/ABI 草案使用相同的 requirement、partition 和 source refs。

#### 1.2.2 算子接口与 ABI Mapping Draft

以下为本 DESIGN 唯一逻辑接口合同：

```text
operator_interface_contract:
  logical_inputs_and_outputs: <按逻辑顺序列出输入、输出和角色>
  logical_shape_dtype_layout: <逐项 shape、dtype、layout、transpose>
  runtime_and_compile_time_axes: <所有变化轴及所有者>
  optional_arguments_and_defaults: <可选参数、默认值、缺省语义>
  error_and_rejection_semantics: <host 校验、拒绝条件和错误行为>
  host_call_contract: <用户侧调用顺序、返回值和同步语义>
  physical_data_prerequisites: <进入 device 前必须满足的转换、打包和对齐>
  abi_mapping_draft:
    - abi_mapping_draft_row_id: <stable-row-id>
      logical_argument_id: <logical input/output or auxiliary object>
      logical_role_and_requiredness: <role, required | optional | auxiliary>
      logical_shape_dtype_layout: <logical view>
      physical_buffer_role_and_storage_rule: <physical view and storage>
      byte_extent_offset_and_unit_rule: <extent, offset, unit>
      host_owner_and_lifetime: <owner and lifetime>
      runtime_or_compile_time_owner: <owner>
      intended_device_destination_role: <device destination or consumer role>
      rejection_condition: <invalid input or state>
      evidence_refs:
        - <source-ref>
```

先冻结逻辑接口，再设计 device ABI。`abi_mapping_draft` 覆盖每个逻辑输入、输出以及 workspace、TilingData、grid/usedCore、stream/dispatch 等必要辅助对象，但不猜测具体 GM 参数顺序、入口修饰符或 Wrapper；这些由 §3.1 的真实 witness 冻结。

### 1.3 数据流

以 §1.2 的相同参数 ID 和物理对象描述完整数据流：

```text
<逻辑输入>
  -> <host 转换/打包，若适用>
  -> <GM physical buffer>
  -> <Blaze component / memory level，按当前 witness 填写>
  -> <中间值与生命周期>
  -> <输出转换/写回>
  -> <逻辑输出>
```

| 顺序 | 数据对象 | 来源位置 | 目标位置/消费者 | shape/dtype/layout | offset/单位 | 生命周期 | contract ref |
|-----|----------|---------|-----------------|--------------------|-------------|---------|--------------|
| `<n>` | `<对象>` | `<位置>` | `<位置/组件>` | `<物理属性>` | `<规则>` | `<范围>` | `<draft/crosswalk ref>` |

### 1.4 核心计算步骤(复杂算子)

本节给出精简的总体步骤，不重复 §2.4 的分支伪代码：

| 顺序 | 核心步骤 | 输入/输出 | 使用的 API/组件 | 关键资源或同步 | requirement/partition refs |
|-----|----------|----------|----------------|----------------|----------------------------|
| `<n>` | `<步骤>` | `<对象>` | `<已验证项>` | `<资源/同步>` | `<refs>` |

若存在多个实际分支，列出差异：

| 分支 ID | 选择条件 | 差异步骤 | 对应 design_binding_ref | 验证要求 |
|--------|----------|---------|--------------------------|---------|
| `<branch-id>` | `<条件>` | `<差异>` | `<binding-ref>` | `<verification ref>` |

### 1.5 内存管理(Buffer 规划)

Buffer、workspace 和输出生命周期必须与 §1.3 数据流及 §3 的 ABI/crosswalk 一致：

| Buffer/对象 ID | 物理位置 | 用途与消费者 | shape/dtype/layout | 大小计算及单位 | offset/对齐 | 所有者/生命周期 | 复用/覆盖条件 | contract ref |
|---------------|---------|-------------|--------------------|----------------|-------------|----------------|--------------|--------------|
| `<id>` | GM/L1/L0/UB/workspace | `<用途>` | `<属性>` | `<由实际 shape 和证据导出的公式>` | `<规则>` | `<owner/lifetime>` | `<规则>` | `<crosswalk ref>` |

| 资源项 | 预算或合法域 | 计划占用 | 证据 | 结论 |
|-------|--------------|---------|------|------|
| UB | `<目标环境事实>` | `<动态计算>` | `<source/design ref>` | confirmed / blocking |
| L1/L0 | `<目标环境事实>` | `<动态计算>` | `<source/design ref>` | confirmed / blocking |
| workspace | `<接口与生命周期>` | `<动态计算或 not_applicable>` | `<source/design ref>` | confirmed / blocking |

不得预填固定 Buffer 名、固定容量、固定 UB 比例、固定 offset、固定 event ID 或特定场景公式。

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 设计结论 | 证据/合同引用 |
|-----|---------|---------------|
| 切分维度及顺序 | `<按需求与 witness 填写>` | `<refs>` |
| Scheduler/Dispatch 链 | `<实际组件和语义>` | `<refs>` |
| grid/block/usedCore | `<运行时或编译时所有者及动态计算规则>` | `<refs>` |
| 单核任务和 tail | `<范围、边界和空任务行为>` | `<refs>` |
| 负载均衡 | `<策略与拒绝条件>` | `<refs>` |

### 2.2 UB 切分策略

| 项目 | 设计结论 | 证据/合同引用 |
|-----|---------|---------------|
| UB 分区与复用 | `<对象、顺序、对齐、生命周期>` | `<refs>` |
| tile/chunk 策略 | `<由实际 shape/layout 导出的规则>` | `<refs>` |
| TilingData/Params | `<字段语义、合法域、host/device 映射>` | `<refs>` |
| host Tiling | `<兼容引擎、固定合法值或独立计算策略>` | `<refs>` |
| workspace/final/partial/output | `<所有权与交接>` | `<refs>` |
| 同步与 event | `<语义、生产者、消费者、首轮/复用/final drain>` | `<refs>` |
| 资源拒绝条件 | `<UB/L1/L0/workspace 等边界>` | `<refs>` |

### 2.3 分支场景覆盖

每个 required partition 必须被一个首选方案覆盖或进入有证据的 unsupported 处置：

| 分支 ID | requirement/partition IDs | shape/dtype/layout/tail 条件 | 处理策略 | 首选 binding | 支持状态 | 验证引用 |
|--------|---------------------------|-------------------------------|---------|-------------|---------|---------|
| `<branch-id>` | `<refs>` | `<精确条件>` | `<策略>` | `<binding-ref 或 none>` | supported / rejected / unsupported | `<verification ref>` |

记录 runtime specialization、可选参数、空 tensor、极小/极大 shape、对齐前后及实际适用的边界；不得以单一对齐样例代表通用支持。

### 2.4 类别特有设计

本节必须直接包含每个实际分支的可实现级核心流程伪代码，不能写“见 §1.4/§3”或只给公式。伪代码中的接口、API/组件、参数顺序、Buffer、offset、Tiling/Params 和同步必须与 §1.2、§1.5 和 §3 合同一致；模板不预设具体 API、event、SplitM、Epilogue 或 Tiling recipe。

**分支 `<branch-id>`**

- 适用条件：`<requirement/partition refs 和精确谓词>`
- 设计绑定：`<design_binding_ref>`
- 输入输出：`<logical/physical argument IDs>`
- 关键边界：`<tail、空任务、资源和拒绝条件>`

```cpp
// 使用当前 Investigation 已验证的实际接口名替换全部占位内容。
ValidateHostInputs(<logical arguments and runtime axes>);
PreparePhysicalBuffers(<packing, layout, byte extent and offsets>);
PrepareTilingAndParams(<legal values and ownership>);

LaunchSelectedKernel(<ordered ABI from design_binding_ref>) {
    <bind selected Blaze specialization and component chain>;
    for (<actual tile or dispatch iteration>) {
        <load according to physical data contract>;
        <compute using APIs/components verified in section 1.2>;
        <apply required synchronization and lifecycle transitions>;
        <handle tail, empty work and rejection paths>;
        <write final/partial/output according to the frozen contract>;
    }
}

FinalizeAndExposeLogicalOutput(<unpacking or conversion if required>);
```

存在多个分支时，按以上格式在本节继续展开，每个分支均需独立给出伪代码和合同引用。

---

## 3. Blaze 设计合同

### 3.1 基于 Blaze 官方库的方案分析

本节始终填写。它使用 §0.4 需求合同评估 Investigation 事实，不复制 Step 2 的路线结论：

```text
matmul_base_analysis:
  candidate_evaluation_ids_considered:
    - <candidate-evaluation-id>
  concrete_blaze_composition_and_specialization: <observed composition>
  kernel_policy_block_scheduler_chain: <observed chain>
  abi_bindings:
    - design_binding_ref: <stable-binding-ref>
      partition_ids:
        - <partition-id>
      assembly_witness_ref: <single-witness-ref>
      kernel_abi_contract:
        entry_modifier_and_linkage: <observed modifier and linkage>
        entry_symbol: <observed symbol or designed project symbol bound to witness>
        template_and_specialization: <observed template/specialization>
        ordered_gm_parameters_and_direction:
          - order: <n>
            parameter: <name and type>
            direction: input | output | input_output
            nullability: required | optional
            byte_extent_offset_and_unit_rule: <rule>
        tilingdata_params_and_scheduler_mapping: <field-to-consumer mapping>
        workspace_grid_usedcore_and_dispatch: <ordered auxiliary ABI and ownership>
        wrapper_and_launch_binding: <wrapper, launcher and dispatch relation>
        final_partial_and_output_lifecycle: <owner, state transition and completion>
        source_refs:
          - <source-ref>
      abi_crosswalk:
        - crosswalk_row_id: <stable-row-id>
          logical_argument_id_or_auxiliary_abi_object: <object-id>
          physical_buffer_and_storage_rule: <buffer/layout/packing>
          launcher_argument: <argument or reasoned not_applicable>
          wrapper_argument: <argument or reasoned not_applicable>
          ordered_kernel_gm_parameter_or_entry_binding: <binding>
          tilingdata_or_params_field_or_not_applicable_reason: <mapping or reason>
          device_consumer: <component/operation>
          direction_nullability_owner_lifetime: <closed semantics>
          byte_extent_offset_and_unit_rule: <closed rule>
          source_refs:
            - <source-ref>
      source_backed_signature_skeleton:
        signature_source_refs:
          - <source-ref>
        observed_declaration_shape: <declaration form, not copied implementation>
        entry_modifier_and_linkage: <observed facts>
        entry_symbol: <observed or contract-bound symbol>
        template_and_specialization: <observed facts>
        ordered_parameters: <exact observed order and types>
        wrapper_invocation_or_dispatch: <observed call/dispatch form>
  tilingdata_params_and_scheduler_semantics: <closed semantics>
  host_tiling_strategy: <compatible engine, fixed legal controls, or project calculation>
  logical_and_physical_data_contract: <interface-to-buffer summary>
  final_partial_workspace_and_output_lifecycle: <closed lifecycle>
  resource_and_dispatch_contract: <resource domain and dispatch>
  runtime_specialization_and_rejection: <runtime predicates and rejection>
  evidence_refs:
    - <source-ref>

native_gaps:
  - requirement_id: <requirement-id>
    unmet_requirement: <exact unmet requirement>
    observed_limit_or_rejection: <source-backed limitation>
    evidence_refs:
      - <source-ref>
    design_impact: <route/interface/validation impact>
```

每个首选 `design_binding_ref` 必须独立绑定其 `partition_ids`、同一 `assembly_witness_ref`、`kernel_abi_contract`、`abi_crosswalk` 和 `source_backed_signature_skeleton`，不得混用不同 specialization。每个逻辑参数和 workspace、TilingData、grid/usedCore、stream/dispatch 等辅助 ABI 对象都必须贯通物理 Buffer、Launcher、Wrapper、Kernel 参数或 entry binding、Tiling/Params 以及 device consumer。

官方方案覆盖全部 required partitions 时 `native_gaps` 为空。gap 必须有明确不兼容/拒绝证据，或对精确需求穷尽已声明读取边界后的无匹配事实；孤立的 `not_found`、`indexed`、`unknown` 和未读范围不能构成 gap。接口、物理数据或 ABI 事实不足时，只能向 Step 2 发起一次纯语义补充调查。

### 3.2 最终路线与候选处置

```text
candidate_disposition_records:
  - candidate_evaluation_id: <stable-candidate-id>
    selection_status: 首选 | 备选
    applicable_partition_ids:
      - <partition-id>
    disposition: <selected, alternative, rejected, blocked or not_applicable with reason>
    evidence_refs:
      - <source-ref>

implementation_route: blaze_native | blaze_custom | unsupported
selected_scenario: <only blaze_custom; omit for other routes>
unsupported_points:
  - requirement_id: <only unsupported>
    native_gap: <native-gap-ref>
    scenario_match_result: <zero-match or multiple-match with evidence>
    evidence_refs:
      - <source-ref>
    user_recovery: <requirement change or evidence needed>
decision_evidence_refs:
  - <source/design ref>
```

- `blaze_native`：§3.1 覆盖全部 required partitions，不读取场景注册表，不填写场景 action。
- `blaze_custom`：存在证据闭合的 native gap，场景注册表恰好唯一命中，且 §3.3 前提闭合。
- `unsupported`：官方能力缺口证据充分，场景零命中或多命中；只生成本阻塞 DESIGN。

每个 required partition 最多一个“首选”。淘汰、阻塞和不适用候选必须记录处置，但不能伪装成备选；PLAN 只消费首选，不自动切换备选。

### 3.3 定制扩展场景合同

`implementation_route=blaze_native` 时只填写：

```text
custom_extension_status: not_applicable
```

仅 `implementation_route=blaze_custom` 时填写完整合同：

```text
selected_scenario: <single registered scenario>
registry_entry_source: <registry source>
design_guide_source: <canonical design guide>
development_guide_source: <canonical development guide>
consumed_contracts: <base contracts consumed by the scenario>
preserved_contracts: <base contracts preserved unchanged>
replaced_contracts: <explicitly authorized replacements>
added_contracts: <scenario additions>
required_investigation_fact_refs:
  - <fact-ref>
dependency_skills:
  - <only dependencies declared by the selected scenario>
formula_and_interface_delta: <formula/interface additions or replacements>
kernel_and_layer_contract: <custom layer composition>
tiling_resource_data_sync_contract: <tiling/resource/data/sync delta>
customization_scope: <copy-and-adapt sources and writable destinations>
scenario_validation_additions: <extra validation contract refs>
scenario_unsupported_boundary: <scenario rejection domain>
abi_crosswalk_delta:
  - delta_crosswalk_row_id: <stable-delta-row-id>
    base_design_binding_ref: <base binding>
    base_crosswalk_row_refs:
      - <base-row-ref>
    added_or_replaced_abi_object: <authorized object>
    physical_and_host_device_wiring: <physical/launcher/wrapper/kernel/device wiring>
    tiling_params_or_sync_wiring: <field/event ownership and lifecycle>
    source_refs:
      - <source-ref>
```

DESIGN 冻结前读取 `design_guide_source`；仅编译 PLAN 时读取 `development_guide_source`。`abi_crosswalk_delta` 只能追加或替换场景明确授权的对象，必须绑定基础 binding 和基础 rows，不能重写、拼接或混用基础 MatMul ABI。

### 3.4 最终资源、数据、ABI、Tiling、同步与范围

```text
final_operator_interface: <§1.2.2 的冻结逻辑接口>
final_kernel_entry_abi_and_crosswalk: <base abi bindings + authorized delta consistency view>
final_component_chain: <selected component/policy/block/scheduler chain>
final_physical_data_contract: <logical-to-physical conversion, packing, layout and byte rules>
final_tiling_and_params_contract: <host/device fields, legal values and owners>
final_resource_contract: <UB/L1/L0/workspace/grid/core constraints>
final_workspace_and_output_lifecycle: <ownership, state and completion>
final_sync_and_event_contract: <producer/consumer, first use, reuse and final drain>
allowed_change_scope:
  - operators/<operator_name>/<authorized project-relative scope>
  - operators/<operator_name>/op_kernel/include/blaze/: <create_only_exact_read_only_materialization_if_missing or omit>
  - operators/<operator_name>/op_kernel/include/tensor_api/: <create_only_exact_read_only_materialization_if_missing or omit>
forbidden_change_scope:
  - <project-root>/ops-tensor/
  - operators/<operator_name>/op_kernel/include/blaze/: <content modification always forbidden>
  - operators/<operator_name>/op_kernel/include/tensor_api/: <content modification always forbidden>
  - /ascendc-blaze-best-practice/assets/
  - <unregistered custom layers>
  - <paths outside operators/<operator_name>/>
  - <unrelated user files>
consistency_audit: <interfaces, bindings, delta, resources and validation agree>
```

`final_kernel_entry_abi_and_crosswalk` 只是 §3.1 基础 binding 与 §3.3 授权 delta 的一致性视图，不得形成第三套 ABI。PLAN 的初始目标文件必须是 `allowed_change_scope` 的子集。根 `ops-tensor` 和 Skill Asset 原文件始终只读；项目内官方头文件副本除已授权的缺失目录原样物化外不允许写入，物化完成后保持只读。

若 Investigation 表明设计请求因跳过 Step 1 而缺少项目内官方副本，`allowed_change_scope` 只为缺失目标增加一次 `create_only_exact_read_only_materialization`：来源固定为当前 `blaze_source_root`，只能原样复制或绑定，完成同源一致性核对后立即成为只读区。该例外不允许适配、补丁、筛选文件或从第二源码根取材；副本已存在时不生成例外。`forbidden_change_scope` 对副本内容修改始终生效。

### 3.5 Golden 与验证合同

```text
logical_data_and_seed: <logical input domain, deterministic seed and boundary values>
physical_conversion_and_packing: <conversion applied only to independent device-input copies>
cpu_golden_formula_and_dtype_order: <formula, accumulation, conversion and rounding order>
comparison_threshold_and_nonfinite_gate: <rtol/atol or exact rule, NaN/Inf policy>
demand_partition_and_boundary_coverage: <requirement/partition IDs and boundary matrix>
diagnostic_checks: <source-backed intermediate or invariant checks>
repeat_and_final_regression: <repeat count semantics, cleanup and final clean run>
semantic_golden_consistency: confirmed | conflict | blocking
golden_authority_and_conflict_resolution: <authority, conflict and recovery condition>
verification_status: planned | unverified
```

在冻结验证合同前，逐项比较需求正文、接口合同和可执行 Golden 的公式、dtype、
转换顺序、分支/轴语义。发现冲突时必须标记 `conflict` 或 `blocking`，记录权威
来源及恢复条件；不得静默选择一个公式继续生成。执行冻结 Golden 所需的后端或
依赖不可用时，`semantic_golden_consistency` 必须为 `blocking`；只有具备等价性证据
的后端才能作为 fallback，不得静默改变累加顺序、dtype、舍入或公式。

对普通、非序列化 MatMul，Golden 必须先从未做设备布局转换的逻辑输入计算，
再对独立副本做 device packing。若冻结合同把已编码 MX/FP8 value、E8M0 scale 或 FP4
字节作为设备输入，则必须从最终写入设备的实际字节解码后计算 Golden；量化前
逻辑 FP32 只能作为生成源，不能替代最终 Golden。通用模板不预填 Grouped、MX、
SplitM、固定诊断模式或固定矩阵；专项要求只来自 `scenario_validation_additions`。
Step 3 不实现、不运行设备，也不得写设备 PASS。

---

## 4. 实施计划

### 4.1 PLAN 绑定与内部 Step 4 Handoff

仅 `blaze_native` 或 `blaze_custom` 填写：

```text
plan_path: operators/<operator_name>/docs/PLAN.md
plan_template: /ascendc-blaze-best-practice/references/kernel-design/blaze-plan-template.md
project_contract_id: <must equal section 0.3>
design_contract_id: <must equal section 0.3>
plan_contract_id: <stable-plan-contract-id>
design_consistency_marker: <stable-marker>
plan_consistency_marker: <matching-stable-marker>
design_plan_consistency: confirmed | blocking
```

内部 Step 4 handoff 由 PLAN 编译，不在 DESIGN 中重复逐文件 action。它必须携带 route、唯一场景（如适用）、首选 binding、witness、ABI、crosswalk、signature skeleton、delta、scope、verification 和 readiness 引用。`unsupported` 不填写 PLAN 绑定，不交给 Step 4。

### 4.2 DESIGN/PLAN 一致性、所有权与回退

| 内容 | 设计阶段所有权 | 实施阶段行为 |
|-----|---------------|-------------|
| 需求、接口、route、scenario、首选 binding、ABI、scope、验证基线 | DESIGN 冻结 | 不重新设计；变化时返回 Step 3 |
| PLAN §1/§3 设计基线及 §9/§10 合同 | Step 3 冻结 | 不修改语义；问题记录为 `design_issue` |
| PLAN §2、§4–§8 | Step 3 初始化 | Step 4/Developer 持续更新 |
| PLAN §11 | Step 3 建立空记录 | 实施阶段只追加 |

回退规则：源码 checkout/submodule 不一致返回 Step 1；新增或被推翻的源码事实返回 Step 2 后再执行 Step 3；route、接口、ABI、支持范围或验证基线变化返回 Step 3；普通实现失败留在实施阶段并记录证据，不设置固定重试次数，也不自动切换备选。

---

## 5. 确认清单

### 5.1 人读检查项

- [ ] §0.2 原样、逐条记录用户需求，编号稳定。
- [ ] target chip、NpuArch、`--npu-arch`、shape、dtype、layout 和约束明确。
- [ ] 数学公式、逻辑接口、累加与类型转换顺序明确。
- [ ] §1.2 API/组件的签名、布局、限制和 source refs 已验证。
- [ ] §1.3 数据流与 §1.5 Buffer/资源/生命周期使用同一对象和合同引用。
- [ ] 多核、Scheduler/Dispatch、UB、Tiling/Params、workspace 和同步策略明确。
- [ ] 所有 required partitions 在 §2.3 有支持或阻塞处置。
- [ ] 每个实际分支的伪代码直接位于 §2.4，并与 API/ABI 合同一致。
- [ ] route 理由、支持范围、拒绝条件和已知限制可读且可追溯。
- [ ] Golden、阈值、nonfinite、边界、诊断、repeat 和最终回归计划完整。
- [ ] 文档没有固定 recipe、未证实签名、实现结果或设备 PASS。

### 5.2 Agent 完成门禁与恢复条件

- [ ] `checkout_consistency=confirmed`，当前 Investigation 与当前根 `ops-tensor` 一致。
- [ ] 需求、接口、官方方案和最终合同均可追到 Investigation 或用户原始需求。
- [ ] `abi_mapping_draft` 覆盖全部逻辑参数和必要辅助对象。
- [ ] 每个首选 binding 的 witness、ABI、crosswalk 和 signature skeleton 同源且闭合。
- [ ] 每个 crosswalk row 贯通物理 Buffer、Launcher、Wrapper、Kernel、Tiling/Params 和 device consumer，或给出可验证的 `not_applicable` 理由。
- [ ] custom delta 正确绑定 base binding/base rows，未改写未授权基础 ABI。
- [ ] native 路线没有场景 action；custom 路线只有一个场景且合同完整。
- [ ] unsupported 路线有逐项 `unsupported_points`，无 PLAN binding 和实施 action。
- [ ] 无 `blocking`、TBD、未决 route、interface、ABI、支持域或验证标准进入可执行路线。
- [ ] PLAN 初始目标属于 `allowed_change_scope`，且所有可写路径位于 `operators/<operator_name>/`。
- [ ] DESIGN/PLAN 的三个 contract ID、route、scenario、首选 binding、验证引用和两个 marker 一致。
- [ ] 每个首选合同有 PLAN action，每个验证合同有 checkpoint。
- [ ] ABI action 引用对应 binding、crosswalk、signature skeleton 和适用的 delta。
- [ ] 每个构建 action 都有入口、目标、预期产物、checkpoint 和失败返回路径。
- [ ] 根 `ops-tensor`、两个项目内官方头文件副本和 Skill Asset 原文件保持只读。

任一可执行路线门禁失败时不得生成 ready PLAN 或进入实施。一次补充 Investigation 后仍缺决定性事实时，停止并等待需求或源码事实澄清，不得用 blocking DESIGN/PLAN 伪装已完成的路线决策。
