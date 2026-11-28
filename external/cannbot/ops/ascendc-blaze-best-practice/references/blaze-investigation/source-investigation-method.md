# Blaze 源码调查方法

本文定义 Step 2 的 Brief、源码导航和候选/证据对象发现。它只建立 Blaze 源码事实，不恢复完整组装方案、不选择项目方案，也不读取场景注册表。

## 1. 调查 Brief

从 `project_root` 派生 `<project-root>/ops-tensor/`，只读复核当前 Blaze 源码版本、递归 submodule、一致性和只读边界，再将用户需求投影为以下调查 Brief。Step 1 handoff 可选；不存在时直接对当前 checkout 建立抽象状态，不读取外部工作流产物：

```text
request_summary
math_contract
tensor_roles_and_extra_io
topology_features
numeric_modes
dtype_format_layout_constraints
runtime_compile_time_axes
demand_partitions
hard_constraint_manifest
required_source_questions
out_of_scope
candidate_expansion_budget: 1
supplement_scope: absent | one semantic request
```

Basic、Batch、Grouped、Quantized、MX 不是互斥场景。按 topology、numeric、layout、transpose、dtype、shape 和额外 I/O 建立正交画像；只有会改变 specialization、真实入口、物理数据或 ABI 的轴才拆分 `demand_partition`。Shape 使用谓词域，不枚举数值，也不展开用户未要求的组合。

每项 hard constraint 和 source question 必须有稳定 ID、适用条件、作用域、重要性和预期事实。发现新的 compile-time 轴时，记录 `brief_amendment`，使受影响对象重新核验；不得借 amendment 缩小用户范围。

需求中的融合、广播或额外输出仅作为语义探针：记录需调查的数据依赖、逻辑/物理 mapping、输出时机或协作关系。它们不映射到场景 ID，也不预判开发路线。

## 2. 补充调查边界

Step 3 可向下一轮同一 Investigation 报告提出一次 `supplement_scope`。它仅包含：

```text
requested_by: Step 3
attempt: 1
affected_requirement_ids
semantic_questions
required_blaze_source_frontier
why_decision_is_blocked
```

补充问题必须以需求语义和待确认的 Blaze 源码关系表述，例如“确认某输出路径的物理 shape、地址单位和完成时机”。不得出现注册表、场景 ID、场景文档、custom 路线或实现建议。Step 2 只在原范围之外追踪该有界前沿；补充后仍未知的事实写入报告，不自行继续扩展。

## 3. 数据源和阅读顺序

Blaze 源码能力只来自当前 `<project-root>/ops-tensor/` 工作区；不得回退到 `operators/<operator_name>/`、其他项目或历史工程：

| 数据源 | 首要用途 | 证据依据 |
|---|---|---|
| `examples/`、`tests/UT/`、真实 wrapper/入口 | concrete 实例化与调用链种子 | `example_assembled` 或 `source_observed` |
| `include/blaze/` | 公开定义、specialization、Params、架构守卫 | `source_observed` |
| 实际 Tiling 目录 | TilingData/Params 消费和字段来源 | `source_observed` |
| `include/tensor_api/` | 已发现调用的 API 定义与 Routing | `source_observed` |
| 官方 API 文档 | 源码无法独立解释的语义 | `documented` |
| README、构建文件 | include、架构与入口线索 | 线索，不能单独证明能力 |

阅读顺序固定为：

1. 从完整 demand partition 对应的 example、UT、wrapper 和真实 Kernel 入口寻找 using、alias、实例化和调用点。
2. 回到公开定义解析 specialization、模板参数、Params 和架构守卫。
3. 没有 concrete 种子时，才从目标公开 Kernel/Policy/Block 反向寻找 caller。
4. 仅为解释候选实际语义读取 API 文档和构建文件。
5. 不深挖未进入候选链的组件，不按相似文件名扩大范围。

缺失目录记录为 `missing` 及其调查影响，不得写成 `unsupported`。Blaze skill Asset、历史 recipe 和项目代码不是 Blaze 源码证据。

## 4. 候选与证据对象

候选 Blaze 组装方案按需求语义和 demand partition 发现：

```text
candidate_discovery
  discovery_id
  partition_id
  candidate_result: found | not_found | unknown
  concrete_entry_seed
  coverage: indexed | deep | out_of_scope
  applicability_observations
  source_refs
```

同一 concrete candidate 在不同 partition 下生成独立 `candidate_evaluation_id`。Candidate ID 标识真实入口/specialization，不承载项目可行性或路线状态。

非组装方案证据对象使用：

```text
evidence_subject
  subject_id
  subject_kind: dataflow | physical_mapping | api | abi | sync | constraint
  related_requirement_ids
  source_frontier
  subject_result: found | not_found | unknown
  coverage: indexed | deep | out_of_scope
```

组件、接口、数据生产/消费点和同步控制流不得伪装成 Blaze 组装方案。`found` 只表示对象已经定位；原子事实仍须由依赖追溯闭合。

## 5. 动态调查问题

所有请求都调查 MatMul 基础链；再按需求添加问题：

| 需求维度 | 调查增量 |
|---|---|
| Batch | batch 语义、stride/broadcast、metadata、入口与调度 |
| Grouped | group metadata、per-group shape/offset、ordering/tail、workspace/final timing、group-aware ABI |
| Quantized | scale/zero-point、packing/alignment、累加/转换与 specialization |
| MX | scale metadata、packing/alignment、K-tail、layout 与 specialization |
| 额外 I/O 或输出后处理 | 实际输出数据流、额外 operand 映射、物理地址和相关协作事实 |

问题只要求事实，不证明不同维度能组合。Grouped Plain、Grouped Quantized、Grouped MX 必须分别在完整 partition 画像上找到 concrete witness。

## 6. 一次候选扩展

直接追踪已发现对象的真实定义和依赖不消耗额度。只有已声明问题在当前前沿没有答案、且源码显示存在有限相邻前沿时，才能消耗全报告一次候选广度扩展：

1. 冻结触发 requirement、partition、起始对象和允许入口/sibling/dispatch 列表。
2. 先检查同一真实入口显式列出的 sibling specialization。
3. 再检查 dispatch 明确引用的 fallback 或当前候选实际引用的同族入口。
4. 不按相似文件名、相邻目录或“可能兼容”扩大扫描。
5. 新对象只用于关闭冻结问题，不递归产生第二次扩展。

## 7. 产物与门禁

本阶段产出 Brief、constraint/source-question definitions、候选 discoveries、证据 subjects、实际读取根、缺失/未读范围与扩展日志。

- 每个已调查 partition 和 subject 必须有 `found`、`not_found` 或 `unknown`。
- `not_found` 只表示冻结范围内没有发现，不能升级为 `unsupported`。
- `indexed` 不能作为 Step 3 的完整设计依据；只有后续方法闭合 required facts 后才能记录 `deep`。
- concrete candidate 进入[Blaze 组装方案恢复方法](assembly-recovery-method.md)；非组装方案证据对象进入[依赖追溯方法](dependency-trace-method.md)。
