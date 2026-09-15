# Elementwise/Broadcast Epilogue Fusion 设计指导

本文是 `elementwise-broadcast-epilogue-fusion` 在 Step 3 唯一命中后的设计入口。它明确消费 2.3 形成的 MatMul 基础分析，设计 MatMul 最终输出后的 elementwise/broadcast Vector Epilogue 增量；不重新选择基础 MatMul Blaze 组装方案，不编写代码、编译或上板。若报告缺少本场景要求的 Blaze 源码事实，本文只提出一次语义化补充调查并返回 Step 2。

## 目录

1. 场景范围与入口门禁
2. 输入和依赖 Skill
3. 设计流程
4. 三层增量合同
5. MemBase/RegBase 决策
6. 条件性 SplitM
7. 验证增量
8. 输出、合并与门禁

## 1. 场景范围与入口门禁

本场景只处理 `MatMul 类主计算 -> Vector 类计算` 的后融合。设 MatMul 类主计算的最终输出为 `T[m,n]`，Vector DAG 的最终输出为 `Y[m,n]`。Vector DAG 仅包含 elementwise 运算、具有明确映射规则的 broadcast 运算或二者组合。只有同时满足以下条件才进入本场景：

- MatMul 类主计算已经产生作为 Vector 输入的最终 `T`，主数据流随后执行一个或多个 Vector 节点；
- Vector DAG 的每个节点均可归类为 elementwise 运算，或具有可冻结索引映射的 broadcast 运算；同形输入按逐元素映射处理；
- 每个 `Y[m,n]` 只依赖 `T[m,n]`、映射到同一 `(m,n)` 的输入元素及其 Vector 中间结果；
- 所有参与输入和输出的 shape、dtype、layout、broadcast 轴及索引规则均可冻结；
- Step 3 已在官方 Blaze 方案分析中记录 `native_gaps`，并由[场景索引](../index.md)唯一命中本场景。

以下语义不属于本场景：reduce/reduction，以及 scan、softmax、跨轴 normalization、gather/scatter、改变元素归属的 transpose/reorder、窗口/邻域运算、额外 MatMul/Convolution、数据相关索引、随机/有状态更新及其他非 elementwise/broadcast 的 Vector 运算。零场景命中或 broadcast 关系不明确时阻塞；不得扩大本场景边界。

quant/dequant 不因类别被本场景排除；只要其 Vector DAG 仍满足上述逐元素/broadcast 合同即可进入设计。其实际 API、dtype 和精度可行性仍须由当前 Blaze 源码事实、依赖 Skill 和 DESIGN 确认，不构成预先支持承诺。

场景名称不自动证明任意 Add/Mul、dtype、broadcast 轴、链长、layout、shape、SplitM 或同步方式受支持。所有能力都由当前需求、Investigation 和 DESIGN 冻结。

## 2. 输入和依赖 Skill

### 2.1 必需输入

进入正文前必须具备：

```text
requirements_contract
operator_interface_contract
matmul_base_analysis
matmul_base_analysis.abi_bindings[]
investigation_report_facts
native_gaps
```

本场景明确要求 `blaze_custom` 路线消费 2.3 产出的 MatMul 基础合同投影。`matmul_base_analysis` 至少包含：

```text
candidate_evaluation_ids_considered
concrete_blaze_composition_and_specialization
kernel_policy_block_scheduler_chain
tilingdata_params_and_scheduler_semantics
abi_bindings[]:
  design_binding_ref
  partition_ids
  assembly_witness_ref
  kernel_abi_contract
  abi_crosswalk
  source_backed_signature_skeleton
final_partial_workspace_and_output_lifecycle
logical_and_physical_data_contract
resource_and_dispatch_contract
evidence_refs
```

本文只消费这份投影，不复制或接管 MatMul 候选选择。每个 `matmul_base_analysis.abi_bindings[]` 内的 `abi_crosswalk` 是已冻结的 MatMul ABI；本场景只能通过 `abi_crosswalk_delta` 追加额外 operand、输出、Params、Wrapper 或同步接线，且每行都必须声明稳定 `delta_crosswalk_row_id`、`base_design_binding_ref` 和 `base_crosswalk_row_refs`（基础 `crosswalk_row_id`），不能重写或混用基础行。基础合同或场景前提缺失时，不得继续场景设计；向 Step 2 提交一次精确的无场景名补充问题。一次补充后仍缺关键事实时停止等待用户澄清，不写最终路线，也不生成 DESIGN/PLAN。

### 2.2 Blaze 源码前提与补充问题

在设计 Block层、Kernel层或 Epilogue层之前，检查 Investigation 报告是否已经记录以下事实：

| 前提 | 需要的 Blaze 源码事实 |
|---|---|
| MatMul 输出数据流 | 最终/partial 时机、输出位置、逻辑/物理 shape、layout、地址单位和生命周期 |
| 广播映射 | 每路额外 operand 的逻辑/物理表示、索引、stride/offset 单位和适用条件 |
| C+V 协作（仅实际存在时） | Block 输出、Kernel adapter、producer/consumer、同步、slot/资源和 final drain 事实 |
| 接线 | 基础 ABI crosswalk、TilingData/Params 到 Block、Kernel、Epilogue/Wrapper 的字段关系，以及场景可追加的 delta 边界 |

缺少其中的决定性事实时，写入下一轮同一 Investigation 报告的：

```text
requested_by: Step 3
affected_requirement_ids
semantic_questions: <例如“确认 MatMul 输出到首个 Vector 消费者的物理位置、地址单位和完成时机”>
required_blaze_source_frontier
why_decision_is_blocked
```

不得写入本场景 ID、场景路径、MemBase/RegBase 选择或定制实现建议。只允许一次补充；补充后仍缺事实则等待用户澄清，而不是写为 `unsupported`。

### 2.3 依赖 Skill 门禁

场景设计必须先加载两个 Skill 根入口：

- [`/ascendc-api-best-practices/SKILL.md`](../../../../ascendc-api-best-practices/SKILL.md)
- [`/ascendc-regbase-best-practice/SKILL.md`](../../../../ascendc-regbase-best-practice/SKILL.md)

再按当前公式读取必要的 API、限制、pitfalls 和真实参考实现。直接打开叶子文档不能替代加载根入口。DESIGN 要记录两个 Skill 的 `source_ref`、当前 API 证据、适用限制和用于比较的真实实现来源。普通 MatMul 和其他扩展场景不继承本依赖。

## 3. 顶向下设计流程

### 3.1 冻结公式和广播合同

先写出完整公式 DAG，不以“Add/Mul Epilogue”之类名称代替数据合同：

| 对象 | 必填合同 |
|---|---|
| MatMul 输出 | 逻辑/物理 shape、dtype、累加/转换、final 时机、layout、row pitch |
| 每路额外输入 | 逻辑 shape/dtype/layout、分布、broadcast 轴、索引函数、stride/offset 单位、生命周期 |
| 每个 DAG 节点 | 操作、输入、输出、中间 dtype、执行顺序和 API 证据 |
| 最终输出 | 逻辑/物理 shape、dtype、转换/舍入/饱和、GM layout 和有效范围 |
| Golden | 与设备相同的操作和 dtype 转换顺序 |

对每个 `(m,n)` 显式写出额外 operand 的索引映射。无法证明的 scalar/row/column/full-tensor mapping 标记 blocking，不得从场景名称推断。

对每个新增 operand、最终输出、Params/Wrapper 字段或同步对象，追加一行 `abi_crosswalk_delta`：`delta_crosswalk_row_id`、`base_design_binding_ref`、`base_crosswalk_row_refs`、物理 buffer/字节规则、Launcher/Wrapper/Kernel 接线、Tiling/Params 字段或 `not_applicable` 理由、设备消费者、生命周期和 `source_refs`。不允许通过 `final_interface_delta` 隐式覆盖基础 MatMul ABI。

### 3.2 消费 MatMul 基础合同

核对：

1. MatMul 输出已经完成必要 K、workspace 和跨核归并，能作为最终 `T`；
2. 物理输出位置、shape、layout、extent、alignment、tail 和完成时机已闭合；
3. TilingData/Params、Scheduler 语义、grid/core 和资源足以计算融合生命周期；
4. C-direct-GM 或等价纯 MatMul诊断路径已被设计为可验证的基础路径；
5. 2.3 已列明无法由 Blaze 官方库覆盖的需求项；本文只设计该增量，不重新作官方路线判定。

只列本场景需要消费、保留、替换或新增的字段。不得静默改变基础 MatMul 的数学语义、激活 partitions、ABI crosswalk 行或 evidence boundary。

### 3.3 兼容性和 custom 授权

按顺序比较：

1. 2.3 已调查的 Blaze 官方库事实是否已经提供所需输出/协作接口；
2. `native_gaps` 是否能在不改变需求合同的前提下由本场景增量闭合；
3. 只有明确缺失且修改范围可隔离时，才授权项目内 Block层、Kernel层或 Epilogue层 custom 文件；
4. 每个 custom 授权必须记录来源、首个修改点、保留不变量、ABI 边界和验证门禁。

`blaze_custom` 是本场景的定制路线，不表示必须创建全部 custom 代码。Asset 只能是 DESIGN 明确授权的可选结构起点，不能证明公式、adapter、dtype、ratio、slot、同步或支持范围。

## 4. 三层增量合同

### 4.1 Block层

按 [Block层专题](block-l0c2ub-extension.md) 冻结：

- L0C 源 Tensor 的 dtype、location、layout、逻辑/物理 shape 和 Copy extent；
- UB/GM 目的 Tensor 的 dtype、layout、row pitch、有效 extent 和地址单位；
- final/partial 判断、归并完成时机和输出分支；
- 实际 Copy trait/API、alignment、tail 和错误边界；
- 使用官方 Block 或 custom Block 的证据和授权范围。

相邻 Block 的 Slice/Copy 模式只能作为候选线索。必须绑定当前 `matmul_base_analysis.abi_bindings[].assembly_witness_ref` 和 source evidence，不能写成默认规则。

### 4.2 Kernel层

实际存在 AIC/AIV 协作时，按 [Kernel层专题](fused-kernel-development.md) 冻结：

- concrete mixed Kernel 入口、AIC/AIV 角色和实际 ratio；
- Epilogue adapter 的每个参数、返回单位和地址语义；
- flag/event/pipe、producer/consumer、首轮、reuse、empty task 和 final drain；
- slot 数、slot index、容量、轮转和覆盖前等待；
- cross-core wait 到 Epilogue 首消费者 pipe 的本地交接需求；
- Params/TilingData、Wrapper 和 Launcher 的新增 ABI。

每个新增或替换的接线必须写入 `abi_crosswalk_delta`，并引用对应 `matmul_base_analysis.abi_bindings[]` 的 `kernel_abi_contract`/签名骨架和 `base_design_binding_ref`。本文不允许因融合需要重排、重命名或重新解释基础 GM 参数；需要改变基础 ABI 时回通用 Step 3。

ratio、slot、bridge、五参 adapter 等只能来自当前 witness；历史设备记录必须同时绑定同一 Investigation、Blaze 组装方案、构建和验证范围才能复用。

### 4.3 Epilogue层

冻结：

- C、每路 operand、每个中间值和输出的 dtype、location、alignment 和有效列；
- 每个 slot 的 C 区、staging 区、输出/临时区和 guard 的 byte range；
- 当前 AIV 的本地 C 起点，额外 GM operand/输出的全局行列映射；
- `curM/curN`、mask/tail、stage rows、DataCopy stride/extent 单位；
- 当前公式 DAG 使用的 API、事件和中间值生命周期；
- 运行时拒绝条件及 host/Tiling 门禁。

每路 operand 必须按自己的 dtype/alignment 计算。不得把整块 UB 同时分配给多个 slot，或从示例 Asset 继承固定容量和 guard。

## 5. MemBase/RegBase 决策

同时评估 [MemBase 专题](epilogue-membase-design.md) 和 [RegBase 专题](epilogue-regbase-design.md)，但只让一个“首选”进入 PLAN。比较维度至少包括：

| 维度 | 必须回答 |
|---|---|
| API 可用性 | 当前 CANN 中所需 API、specialization、限制和真实参考实现是否闭合 |
| 公式映射 | DAG 每个节点和 broadcast operand 如何实现 |
| UB/寄存器资源 | staging、intermediate、guard、slot-aware 上界和 tail |
| 数据搬运 | GM/UB/VF pass、stride、alignment、offset 和复用 |
| 同步 | LocalTensor pipeline、event、barrier、首消费者和生命周期 |
| 数值 | dtype、mask、cast/round/saturate 和 Golden 一致性 |
| 支持边界 | partition、shape/tail、broadcast 轴、SplitM/slot 和拒绝范围 |

“一个操作用 MemBase、多个操作用 RegBase”只能是比较线索，不是决策规则。某路线 API、资源或同步事实不完整时标记 blocking/淘汰；不得把另一公式或历史 Asset 的设备结果替代当前证据。

## 6. 条件性 SplitM

只有需求、基础 MatMul 合同和选定融合路径都明确启用 SplitM 时，才读取 [SplitM 专题](splitm-contract-and-debugging.md) 并冻结：

- 实际 CV ratio 与每个 sub 的行分配；
- odd-M、empty sub 和 `localRows=0` 的 Kernel 释放行为；
- 本地 slot C 起点与 GM operand/output 的全局 sub offset；
- 物理 row pitch、stage offset、stride/extent 单位和 tail；
- slot reuse、同步、支持范围和诊断矩阵。

没有激活时显式写 `not_applicable`。Ratio、行分配公式和地址规则必须由当前 witness/验证合同证明，不能从历史实现泛化。

## 7. 场景验证增量

使用 [精度诊断专题](precision-diagnosis.md) 将场景增量写入 DESIGN：

| 模式 | 隔离目标 |
|---|---|
| C-direct-GM | 基础 MatMul Blaze 组装方案与最终 GM 输出 |
| C-through-fusion | MatMul 输出进入融合路径、identity writeback、slot 和 C-ready |
| V-zero-C | 额外 operand、formula DAG、broadcast、mask/tail 和写回 |
| V-known-C | 非零已知 C、adapter、offset 和同步 |
| Full | 完整 MatMul + elementwise/broadcast Epilogue、reuse 和 final drain |

具体 dtype、shape、broadcast 轴、SplitM、slot、重复次数和阈值由需求与 DESIGN 冻结。至少包含：

- 逻辑输入、物理转换、CPU Golden 和全元素非有限值门禁；
- 每个声明支持的 mapping 及 alignment/tail 边界；
- 单变量负/正对照，未得到两者时不得宣称根因；
- 清理临时 known-C、Dump、故障注入和诊断入口后的 Full 回归；
- 一个跨输出依赖公式的负向路由检查，证明其不命中本场景。

Step 3 只能记录 `planned/unverified`。任何 `device_verified` 结论必须引用与当前 Investigation、Blaze 组装方案、构建和测试范围一致的外部验证记录；不在 Skill 中写验证工程路径或固定统计。

## 8. 输出、合并与门禁

DESIGN 的场景合同必须输出：

```text
selected_scenario: elementwise-broadcast-epilogue-fusion
registry_entry_source
design_guide_source
development_guide_source
consumed_contracts
preserved_contracts
replaced_contracts
added_contracts
required_investigation_fact_refs
dependency_skills
dependency_skill_evidence
formula_and_interface_delta
formula_and_broadcast_contract
kernel_and_layer_contract
block_layer_delta
kernel_layer_delta
epilogue_layer_delta
final_interface_delta
abi_crosswalk_delta
tiling_resource_data_sync_contract
tiling_resource_data_sync_delta
customization_scope
scenario_validation_additions
scenario_unsupported_boundary
```

本场景用以下关系满足通用场景合同：`formula_and_interface_delta` 由 `formula_and_broadcast_contract` 和 `final_interface_delta` 展开；`kernel_and_layer_contract` 由 Block层、Kernel层和 Epilogue层 delta 展开；`tiling_resource_data_sync_contract` 由 `tiling_resource_data_sync_delta` 展开。详细字段不替代通用字段，二者必须引用同一批合同 ID。

合并顺序固定为：

```text
requirements
  -> operator interface
  -> matmul_base_analysis
  -> abi_bindings[] within matmul_base_analysis
  -> scenario delta
  -> final operator/kernel/resource/validation contracts
```

本场景是冲突 owner：能由已闭合证据解决的冲突必须写明保留/替换关系；需要新 Blaze 源码事实时回 Step 2，需要改变需求/接口/候选时回通用 Step 3。禁止循环“校正”或现场实验。

DESIGN 冻结前不得读取开发指导正文。场景合同与通用最终合同闭合后，Step 3 写入 `implementation_route: blaze_custom` 与唯一 `selected_scenario`，再读取[场景开发指导](elementwise-broadcast-epilogue-fusion-development.md)编译项目 `operators/<operator_name>/docs/PLAN.md`。PLAN 未完整覆盖本场景 required 方法时，回 Step 3 修正，不交给实施阶段。
