# Blaze Investigation 报告模板

本模板由 `/ascendc-blaze-best-practice` Step 2 生成，固定写入 `<project-root>/operators/<operator_name>/docs/blaze/blaze-investigation-report.md`。文档内项目相对路径写为 `operators/<operator_name>/...`。它只记录 Blaze 源码调查事实，不给出项目路线、官方支持结论、场景匹配或实现建议。

## 0. 报告元数据与 Blaze 源码边界

```text
report_kind: blaze_source_investigation
project_root:
operator_name:
operator_root: <project-root>/operators/<operator_name>/
project_contract_id:
investigation_id:
report_path: operators/<operator_name>/docs/blaze/blaze-investigation-report.md
blaze_source_root: <project-root>/ops-tensor/
target_chip:
npu_arch:
cann_version: <optional>
source_version_status: current_checkout | same_investigation_source | consistent
checkout_consistency: confirmed | blocking
submodule_status:
read_only_source_regions:
  - <project-root>/ops-tensor/
  - operators/<operator_name>/op_kernel/include/blaze/
  - operators/<operator_name>/op_kernel/include/tensor_api/
read_only_source_region_status:
  blaze_source_root: confirmed | blocking
  project_blaze_copy: confirmed | missing_for_exact_materialization | blocking
  project_tensor_api_copy: confirmed | missing_for_exact_materialization | blocking
read_only_source_region_evidence_refs:
  blaze_source_root:
  project_blaze_copy:
  project_tensor_api_copy:
source_read_roots:
  - <project-root>/ops-tensor/
actual_source_frontiers:
unread_or_missing_areas:
```

机器字段含义：`upstream_read_roots`（如兼容记录需要）只能等于 `source_read_roots`，不得建立第二个源码根；`assembly_witness` 表示 Blaze 组装方案真实证据；`assembly_members` 表示其具体成员；`candidate_evaluation_id` 表示候选组装方案评估标识。Step 2 可以在没有 Step 1 handoff 时自行只读填写这些抽象状态，不读取 `environment.md`、外部 manifest 或其他工作流产物。项目内只读副本缺失不阻塞源码调查，但必须记录为 `missing_for_exact_materialization`，供 Step 3 编译一次性物化 action；根源码缺失或不一致仍为 `blocking`。

## 1. 需求语义投影

```text
request_summary:
math_contract:
tensor_roles_and_extra_io:
dtype_format_layout_constraints:
topology_features:
shape_and_tail_predicates:
runtime_compile_time_axes:
demand_partitions:
hard_constraint_manifest:
required_source_questions:
out_of_scope:
brief_amendments:
```

每个 requirement/constraint/source question 至少记录稳定 ID、适用 partitions、问题、重要性、来源和调查影响。Basic、Batch、Grouped、Quantized、MX 是可组合画像，不是预设路线或互斥场景。

## 2. 调查范围与补充记录

```text
candidate_expansion_budget: 1
candidate_expansion_log:
supplement_scope: absent | completed
supplement_history:
  - requested_by: Step 3
    attempt: 1
    affected_requirement_ids:
    semantic_questions:
    required_blaze_source_frontier:
    why_decision_is_blocked:
    completion_evidence_ids:
```

`supplement_history` 最多一项。问题只描述待确认的需求语义和 Blaze 源码关系，不能包含场景 ID、场景路径、`blaze_custom`、路线或实现方案。

## 3. 候选 Blaze 组装方案目录

每个 discovery：

```text
discovery_id
candidate_id
partition_id
candidate_result: found | not_found | unknown
concrete_entry_seed
coverage: indexed | deep | out_of_scope
applicability_observations
source_refs
```

每个 candidate evaluation：

```text
candidate_evaluation_id
candidate_id
partition_id
assembly_status: complete | partial
object_readiness: ready | partial | blocked | unknown
assembly_witness:
assembly_members:
  kernel_entry
  policy_and_specialization
  block_mmad
  block_scheduler
  optional_epilogue
  tilingdata_params_and_tiling_entry
applicability_facts
source_relationship_ids
source_refs
```

`object_readiness` 只描述该对象事实是否闭合，不代表官方支持、项目可执行性或最终选型。`not_found` 不等于 `unsupported`。

## 4. 依赖、Tiling 与 ABI 事实

每个字段语义记录：

```text
fact_id
candidate_evaluation_id_or_subject_id
related_requirement_ids
field_path
field_type_and_unit
consumer_and_semantics
applies_when
legal_domain_or_predicate
cross_field_constraints
tilingdata_mapping
observed_value_scope
evidence_status: source_observed | documented | example_assembled |
                 not_applicable | conflict | unknown | unsupported
source_refs
```

每个 ABI/API 记录：

```text
fact_id
candidate_evaluation_id_or_subject_id
caller_or_entry
api_or_abi_element
definition_or_specialization
routing_and_guards
arguments_returns_and_units
effects_on_output_resource_or_sync
related_requirement_ids
evidence_status
source_refs
```

## 5. 物理数据与输出数据流

对每个 Tensor/metadata/额外 I/O 记录：

```text
tensor_fact_id
role
logical_shape_dtype_layout
physical_shape_dtype_layout
packing_padding_alignment
index_stride_pitch_offset_and_units
producer_consumer_and_location
final_partial_and_lifecycle
related_candidate_or_subject
related_requirement_ids
evidence_status
source_refs
```

需求涉及输出后处理或 broadcast 时，只记录已观察到的 producer/consumer、mapping、adapter/同步（如存在）和未知事实。不得把这些记录组织为场景 bundle 或方案选择。

## 6. 已观察限制与未闭合源码事实

每个已观察限制或明确拒绝：

```text
observation_id
related_requirement_ids
candidate_or_subject_id
classification: observed_limit | explicit_rejection
fact
scope_and_predicate
source_refs
```

每个未闭合事实：

```text
unresolved_fact_id
related_requirement_ids
candidate_or_subject_id
question
searched_frontier
why_unresolved
evidence_status: unknown | conflict
source_refs
```

`unsupported` 仅用于源码或官方约束明确拒绝的准确对象。`unknown`、未读范围、`not_found` 和缺失目录不得伪装成拒绝。

## 7. 源码关系与证据账本

```text
source_relationships:
  - relationship_id
    relationship_kind: same_entry | caller_callee | type_reference |
                       shared_params | shared_tiling_field | explicit_dispatch |
                       explicit_prohibition
    left_object_id
    right_object_id
    observed_constraint
    source_refs

evidence_ledger:
  - evidence_id
    source_kind: blaze_source | official_doc | example_or_ut
    canonical_path
    location
    statement_or_observation
    evidence_basis: source_observed | documented | example_assembled
    related_object_ids
    related_requirement_ids
```

## 8. 审计结论

```text
audit_checklist:
  source_root_and_sibling_layout_confirmed:
  recursive_submodules_confirmed:
  normalized_chip_inputs_recorded:
  project_copy_status_and_evidence_recorded:
  request_projection_complete:
  candidate_identity_and_witnesses_recorded:
  dependency_and_physical_facts_recorded:
  observed_limits_separated_from_unknowns:
  read_boundaries_recorded:
  supplement_limit_respected:
  no_route_or_scene_fields:
audit_notes:
dependency_and_physical_data_facts: sections 4 and 5
abi_and_signature_facts: section 4
unresolved_source_facts: section 6
```

本报告结束时不写 `implementation_route`、`selected_scenario`、`unsupported_points`、官方支持状态、候选组合或场景匹配。Step 3 用精确需求消费本报告并作出唯一决策。
