# Blaze 组装方案恢复方法

本文只对“Blaze 源码调查”阶段已发现的 `candidate_evaluation` 恢复 concrete Blaze 组装方案。Blaze 组装方案是同一真实入口或 concrete 实例化证实的组件链，不是按组件名称拼接出的笛卡尔积。

## 1. 输入与边界

每次恢复必须绑定：

```text
candidate_evaluation_id
candidate_id
partition_id
seed_or_expansion_source
required_requirement_ids
required_constraint_ids
```

本阶段只判断组装结构是否闭合。TilingData 全字段语义、Tensor API、物理数据和资源合同在 Dependency Trace 阶段闭合；不得因这些依赖尚未完成而把一个结构完整的 Blaze 组装方案改成 partial。

## 2. Concrete Witness

按以下优先级寻找 witness：

1. example、UT 中完整 using/alias/实例化和调用；
2. 真实 Kernel wrapper 中的 concrete 模板实例化；
3. 同一 Kernel 入口内可完整还原的类型链、specialization 和调用顺序。

孤立公开类定义、相似文件名、注释、不同入口中的片段或手工猜测模板参数都不是 concrete witness。没有 example/UT 并不自动阻塞，只要真实 wrapper/入口已经完整实例化。

## 3. 恢复顺序

沿同一 witness 的实际关系恢复：

```text
Kernel entry / wrapper
  -> dispatch / specialization / Policy
  -> BlockMmad
  -> BlockScheduler
  -> actual Epilogue, if present
  -> TilingData / Params type binding
```

逐层记录：

- 具体符号、namespace、头文件和定义位置；
- concrete specialization、模板参数和 `ScheduleType`；
- 架构/编译守卫和必要 include；
- caller/callee、类型引用和实例化关系；
- Kernel 入口修饰符、参数角色和调用顺序；
- Policy、Block、Scheduler、Epilogue 与 TilingData/Params 的实际绑定；
- 当前 role/partition 投影出的 topology、numeric、dtype、layout、transpose、shape 和功能约束。

Witness 中实际不存在的层可以标记 `not_applicable`，但必须由入口和类型链直接证明。空白、搜索未命中或“通常不需要”不能代替 N/A 证据。

## 4. Blaze 组装方案合同

每个 evaluation 写入：

```text
assembly_witness
  witness_id
  witness_kind
  entry_symbol
  concrete_instantiation
  source_refs

assembly_members
  member_role
  concrete_symbol
  specialization
  source_ref
  relationship_ids
  status: resolved | not_applicable | unknown | conflict

assembly_status: complete | partial
structural_gaps
```

字段含义：`assembly_witness` 表示 Blaze 组装方案真实证据；`assembly_members` 表示该组装方案的具体成员；`assembly_status` 表示组装方案结构状态。字段名保持与报告和 DESIGN 兼容。

`assembly_status=complete` 当且仅当：

- candidate 和单一 partition 绑定明确；
- 存在 concrete witness；
- 该 witness 实际需要的每个成员及其关系均为 resolved/N/A；
- concrete specialization、入口绑定和架构守卫无 unknown/conflict；
- 没有跨 witness 拼接成员。

其他情况均为 partial。`complete` 只表示结构链闭合，不等于 candidate ready，也不表示设备、精度或性能通过。

## 5. 组合需求约束

- Grouped Plain、Grouped Quantized、Grouped MX、Batch Quantized 等激活 partition 必须各自由匹配完整画像的 concrete witness 支撑。
- 不同候选组装方案评估可以使用不同 witness；是否足以覆盖同一项目需求由 Step 3 根据精确合同判断。
- 普通 MatMul candidate 内实际存在官方 AIV/Epilogue 只是一项 Blaze 源码事实，不推导项目路线。
- 非组装方案证据对象不进入本方法。

## 6. 产物与停止条件

每条 evaluation 必须得到 `complete` 或 `partial`，并列出 evidence IDs、结构缺口和下一阶段输入：

- complete：进入 [Dependency Trace](dependency-trace-method.md) 闭合直接依赖；
- partial 且冻结扩展尚未使用、并满足 Blaze 源码调查 的扩展条件：执行唯一一次有限扩展；
- partial 且无合法扩展：保留 partial/unknown，不按名称补齐。

本阶段不得直接评定 `candidate_evaluation_readiness=ready`，不得选择项目候选，也不得创建 recommendation。
