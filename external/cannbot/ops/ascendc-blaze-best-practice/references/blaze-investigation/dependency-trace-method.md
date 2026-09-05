# 依赖追溯方法

本文闭合已发现 Blaze 组装方案和语义证据对象的直接依赖。只沿实际调用、类型引用和已声明源码读取边界追踪；不重新发现候选、不建立 Tensor API 百科、不匹配场景、不选择项目实现。

## 1. 追踪顺序

对每个 candidate evaluation 或 evidence subject 按数据流执行：

1. 从 Kernel/subject 入口识别真实输入、输出、TilingData、Params、workspace 和 dispatch。
2. 沿 concrete 类型链找到 Scheduler、Policy、Block、实际 Epilogue 和 API 消费点。
3. 闭合 TilingData 到各层 Params 的逐字段映射和 device 侧字段语义。
4. 从实际 API 调用点追到唯一定义/specialization 与 Routing。
5. 归一化逻辑对象、设备物理表示、地址单位、资源和生命周期。
6. 仅在源码不能独立解释调用语义时读取对应官方 API 文档。
7. 将观察值、限制、unknown 和 evidence IDs 写回对应需求项。

直接依赖追踪不消耗候选扩展额度。无法确认的 required 原子问题保持 `unknown`；不得用默认值、Asset、旧 recipe 或相似算子填充。

## 2. Kernel 与 Device ABI

每个 candidate evaluation 记录：

- Kernel 符号、入口修饰符、concrete specialization 和架构守卫；
- GM 参数顺序、类型、方向、可选性和空值语义；
- shape、batch/group、scale、mode、layout、transpose 和 dispatch 信息的位置；
- TilingData 传递方式、字段类型、单位和 Params 映射；
- grid/used core 的来源；
- workspace 地址、大小单位、初始化、生命周期和归并关系；
- 输出是 final/partial、目的地和完成时机。

入口声明、host 传参和 device 消费必须逐项对应。只看到类型名或未实例化模板不能闭合 ABI。

## 3. Scheduler Params 语义

对 Scheduler、BlockMmad、Policy、Kernel 和实际 Epilogue 消费的字段记录：

```text
candidate_evaluation_id_or_subject_id
partition_id
field_path
field_type
unit
consumer
semantics
applies_when
legal_domain_or_predicate
cross_field_constraints
shape_and_core_dependencies
tiling_data_mapping
observed_default_or_example
observed_value_scope
evidence_ids
evidence_status: source_observed | documented | example_assembled | conflict | unknown | unsupported
```

至少解释字段对 tile、tail、core、offset、workspace、final/partial 和输出路径的影响。实际未消费字段用直接证据标记 `not_applicable`；示例值仅记录观察范围，不能成为项目默认值。

Blaze 源码没有可直接复用的 host Tiling 时，记录 `not_provided` 和已闭合的 device 语义。真正影响 Step 3 的是字段单位、适用条件、合法域、交叉约束或 TilingData 到 Params 映射未知。

## 4. Tensor API 调用闭合

每个实际调用点记录：

| 字段 | 内容 |
|---|---|
| Caller | candidate/subject、组件、符号和调用位置 |
| API | concrete 符号、参数、返回值和调用目的 |
| Definition | 唯一定义或 specialization 位置 |
| Routing | Pattern、location、layout、dtype、架构和 trait 条件 |
| Units | shape、extent、offset、stride、alignment 的单位 |
| Effects | Copy/Mmad/Fixpipe/Layout、同步、资源和输出影响 |
| Status | `source_observed`、`documented`、`conflict`、`unknown`、`unsupported` |

名称命中或公共说明不等于调用点解析。定义、specialization 或 Routing 不能唯一确定时保留 `unknown`，并绑定受影响 requirement IDs。

## 5. 物理数据合同

对每个输入、输出和 metadata Tensor 记录：

| 类别 | 必填事实 |
|---|---|
| 数学对象 | 角色、逻辑 shape、逻辑 dtype、公式轴 |
| 设备对象 | 存储 dtype、元素/打包字节、物理 shape、layout pattern |
| 变换 | transpose、ND/NZ、padding、alignment、packing、broadcast |
| 地址 | stride、pitch、slice、offset 及单位 |
| 有效范围 | tail、padding 是否参与计算、final/partial 范围 |
| 绑定 | candidate/API/Tiling/Params 字段和 evidence IDs |

逻辑值与设备字节必须分开。Scale、zero-point、group metadata、packed FP4 等对象分别建立合同，不允许只记录一个笼统“支持量化/分组”字段。

## 6. 额外 I/O 与输出数据流证据

需求中有额外 operand、输出后逐元素处理或 broadcast 时，按已发现的真实候选追踪相关对象：

```text
subject_id
related_requirement_ids
producer_and_consumer
logical_and_physical_shape
layout_dtype_and_location
index_stride_offset_units
completion_and_lifecycle
adapter_or_sync_binding_if_present
source_refs
evidence_status
```

这只记录 Blaze 源码事实。不得从需求名称推导 Block 输出、C+V、Epilogue adapter 或同步机制存在；也不得将这些对象打包成场景证据、选择 MemBase/RegBase 或判断 custom 路线。

## 7. 源码关系账本

只记录源码明确存在的关系：

```text
relationship_id
relationship_kind: same_entry | caller_callee | type_reference |
                   shared_params | shared_tiling_field | explicit_dispatch |
                   explicit_prohibition
left_object_id
right_object_id
source_refs
observed_constraint
```

缺少显式 binding 不等于不兼容。Step 2 不枚举候选笛卡尔积，也不把接口适配判断写成 source relationship；这些设计选择留给 Step 3。

## 8. 闭合门禁

Dependency Trace 完成时必须能对每个已调查对象回答：

```text
TilingData -> Params -> consumers
Scheduler field -> semantics/unit/legal domain
API call -> unique definition/specialization/Routing
logical tensor -> physical tensor -> address units
Kernel entry -> GM ABI/grid/workspace/output lifecycle
extra I/O/output dataflow -> observed facts or unknown
```

所有适用的 source questions 都必须有原子状态和证据。缺失 Blaze 源码 host Tiling 不构成事实缺口；缺失 device 语义、合法域、ABI 或 Routing 记录为 `partial` 或 `unknown`。完成后交给[调查合同审计方法](design-contract-method.md)核验，不在本阶段作候选推荐。
