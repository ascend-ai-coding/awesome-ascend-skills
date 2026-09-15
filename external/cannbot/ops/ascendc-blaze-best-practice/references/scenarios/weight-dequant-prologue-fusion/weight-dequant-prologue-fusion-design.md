# Weight-Dequant Prologue Fusion 设计指导

本文是 `weight-dequant-prologue-fusion` 在 Step 3 唯一命中后的设计入口。它明确消费 `matmul_base_analysis` 形成的 MatMul 基础分析，设计 AIV 侧反量化 Prologue 与 AIC 侧 MMAD 的 V+C 融合增量；不重新选择基础 MatMul Blaze 组装方案，不编写代码、编译或上板。若报告缺少本场景要求的 Blaze 源码事实，本文只提出一次语义化补充调查并返回 Step 2。

## 目录

1. 场景范围与入口门禁
2. 输入和依赖 Skill
3. 设计流程
4. 三层增量合同
5. Tiling/资源/同步合同
6. 场景验证增量
7. 输出、合并与门禁

## 1. 场景范围与入口门禁

本场景只处理 `AIV 先反量化低比特权重 B → AIC 后 MMAD` 的 V+C prologue 融合。设低比特权重为 `B[q,k,n]`（int8），反量化参数为 `scale[n]`/`offset[n]`（bf16/fp16），反量化后权重为 `B_dequant[k,n]`（bf16/fp16），最终输出为 `C[m,n] = A[m,k] × B_dequant[k,n]`。只有同时满足以下条件才进入本场景：

- B 输入为低比特量化权重（int8），需在 device 侧反量化后参与 MMAD；
- 反量化发生在 MMAD 之前（V+C 方向），AIV 先执行反量化写入 L1，AIC 从 L1 读取 B_dequant 做 MMAD；
- 反量化参数为 perchannel（`scale[n]`/`offset[n]`）；
- 所有参与输入和输出的 shape、dtype、layout、transpose 及索引规则均可冻结；
- Step 3 已在官方 Blaze 方案分析中记录 `native_gaps`，并由[场景索引](../index.md)唯一命中本场景。

以下语义不属于本场景：

- Blaze 原生支持的量化 MatMul（A8W8/MX scale 路径）——这些走 `blaze_native`，不进入场景匹配；
- pergroup 量化——当前只支持 perchannel；
- Weight-Quant + Vector Epilogue 后融合——Epilogue 固定为 void；
- C+V epilogue 融合（AIC 先 MMAD → AIV 后处理）——这是 `elementwise-broadcast-epilogue-fusion` 场景，数据流方向相反，不可能同时命中。

本场景不属纯 Quantized MatMul 范畴。纯 Quantized MatMul（A8W8/MX）使用 Blaze 原生 scale 路径，走 `blaze_native`；本场景的权重反量化 prologue 融合是 Blaze 无入口的 V+C 融合，走 `blaze_custom`。

场景名称不自动证明任意 dtype、layout、transpose、hasOffset、hasBias 组合受支持。所有能力都由当前需求、Investigation 和 DESIGN 冻结。

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

本场景明确要求 `blaze_custom` 路线消费 `matmul_base_analysis` 产出的 MatMul 基础合同投影。`matmul_base_analysis` 至少包含：

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

本文消费这份投影中的官方 BlockMmad/Scheduler 组件事实（Params 语义、L1/L0 布局约定、ABI crosswalk 基础行）作为 custom 组装的基础参照。场景 delta 在此基础上定义保留、替换和新增合同。基础合同或场景前提缺失时，不得继续场景设计；向 Step 2 提交一次精确的无场景名补充问题。一次补充后仍缺关键事实时停止等待用户澄清，不写最终路线，也不生成 DESIGN/PLAN。

### 2.2 Blaze 源码前提与补充问题

在设计 Block层、Kernel层或 Prologue 层之前，检查 Investigation 报告是否已经记录以下事实：

| 前提 | 需要的 Blaze 源码事实 |
|---|---|
| 官方 BlockMmad 组件 | BlockMmad Params 字段语义、L1/L0 布局约定、bias 路径、K 循环结构 |
| 官方 Scheduler | Scheduler Params 语义、蛇形遍历、尾块切分和 grid/usedCore |
| VF Cast API | 当前 CANN 中各 BType（int8/fp8）的 Cast dtype 组合是否受支持（查阅 asc-devkit `Cast.md` 数据类型组合表）；CastTrait、LoadDist、StoreDist 等具体配置由 Investigation 确认 |
| CV 同步 | HardEvent/CrossCore flag 配对、pipe、producer/consumer 和 final drain |
| Tensor API | B_dequant 写入 L1 的 Copy API、layout 和 alignment |

缺少其中的决定性事实时，写入下一轮同一 Investigation 报告的：

```text
requested_by: Step 3
affected_requirement_ids
semantic_questions: <例如"确认 int8→bf16 CastTrait 的声明形式和可用性">
required_blaze_source_frontier
why_decision_is_blocked
```

不得写入本场景 ID、场景路径、Prologue/Block/Kernel 选择或定制实现建议。只允许一次补充；补充后仍缺事实则等待用户澄清。

### 2.3 依赖 Skill 门禁

场景设计必须先加载两个 Skill 根入口：

- [`/ascendc-api-best-practices/SKILL.md`](../../../../ascendc-api-best-practices/SKILL.md)
- [`/ascendc-regbase-best-practice/SKILL.md`](../../../../ascendc-regbase-best-practice/SKILL.md)

再按当前公式读取必要的 VF Cast API、限制、pitfalls 和真实参考实现。直接打开叶子文档不能替代加载根入口。DESIGN 要记录两个 Skill 的 `source_ref`、当前 API 证据、适用限制和用于比较的真实实现来源。

## 3. 顶向下设计流程

### 3.1 冻结公式和反量化合同

先写出完整公式 DAG，不以"weight-quant matmul"之类名称代替数据合同：

| 对象 | 必填合同 |
|---|---|
| A 输入 | 逻辑/物理 shape、dtype（bf16/fp16）、layout、transpose |
| B 输入 | 逻辑/物理 shape、dtype（int8）、layout、transpose、perchannel 量化模式 |
| scale | dtype（bf16/fp16）、shape `(N,)`、perchannel 加载方式、广播映射 |
| offset | dtype、shape `(N,)`、可选语义、缺省为 0 的条件 |
| bias | dtype（float/bf16/fp16）、shape `(N,)`、可选语义、缺省为 nullptr |
| B_dequant | dtype（bf16/fp16）、L1 layout、写入时机和生命周期 |
| 最终输出 C | 逻辑/物理 shape、dtype、累加/转换/舍入、GM layout |
| Golden | 与设备相同的操作和 dtype 转换顺序 |

对每个 `(k,n)` 显式写出反量化公式：`B_dequant[i,j] = (B[i,j] + offset[j]) * scale[j]`。Cast 链：int8→fp16→bf16（两步）或 int8→fp16（一步，当目标为 fp16 时）。

对每个新增 operand、B_dequant 中间值、Prologue Params 字段或同步对象，追加一行 `abi_crosswalk_delta`：`delta_crosswalk_row_id`、`base_design_binding_ref`、`base_crosswalk_row_refs`、物理 buffer/字节规则、Launcher/Wrapper/Kernel 接线、Tiling/Params 字段或 `not_applicable` 理由、设备消费者、生命周期和 `source_refs`。

### 3.2 消费 MatMul 基础合同

核对：

1. 官方 BlockMmad 的 Params 字段语义、L1/L0 布局约定和 bias 路径已闭合；
2. 官方 Scheduler 的 Params 语义、蛇形遍历和尾块切分已闭合；
3. 官方 Blaze 组装方案的 ABI crosswalk 基础行可作为 custom 接线的参照；
4. `native_gaps` 已列明 Blaze 无法覆盖的需求项（无 V+C prologue Kernel、无 AIV→L1→AIC B 路径、无 WeightQuantMatmulPolicy 特化）；
5. 本文只设计该增量，不重新作官方路线判定。

本文按以下方式消费基础合同：

| 合同 | 消费方式 |
|---|---|
| Scheduler 语义（SWAT 蛇形 + 尾块） | **preserved**：直接复用官方 Scheduler，不修改 |
| A 侧搬运路径（GM A → L1 → L0A） | **preserved**：与官方 MatMul 一致 |
| L0C → GM Fixpipe 输出 | **preserved**：与官方 MatMul 一致 |
| BlockMmad | **replaced**：特化为 WeightQuantMatmulPolicy 版本，B 从 L1 读取（非 GM） |
| Kernel | **replaced**：自定义 mixed Kernel（AIC BlockMmad + AIV Prologue） |
| Prologue | **added**：新增层（VF 反量化 + UB 管理 + L1 写入 + CV 同步） |

不得静默改变基础 MatMul 的 A 侧数学语义、Scheduler 语义或 L0C 输出生命周期。

### 3.3 兼容性和 custom 授权

按顺序比较：

1. `matmul_base_analysis` 已调查的官方 Blaze 库事实是否提供 V+C prologue Kernel 或 AIV→L1→AIC B 路径；
2. `native_gaps` 是否能在不改变需求合同的前提下由本场景增量闭合；
3. 只有明确缺失且修改范围可隔离时，才授权项目内 Block层、Kernel层或 Prologue 层 custom 文件；
4. 每个 custom 授权必须记录来源、首个修改点、保留不变量、ABI 边界和验证门禁。

`blaze_custom` 是本场景的定制路线。Asset（`weight_quant_matmul_kernel.h`、`weight_quant_matmul_block_mmad.h`、`dispatch_policy.h`、`weight_quant_tiling.h` 等）只能是 DESIGN 明确授权的可选结构起点，不能证明公式、dtype、layout、slot、同步或支持范围。复制后必须按当前 ABI/公式/API 重写和验证。

## 4. 三层增量合同

### 4.1 Block层

按 [V+C 同步与 L1 布局专题](vc-sync-and-l1-layout-design.md) 冻结：

- `WeightQuantMatmulPolicy<NO_FULL_LOAD_MODE>` 特化的 SFINAE 约束和 source_ref；
- B 从 L1 读取（非 GM），跳过 GM→L1 B 搬运的修改点和保留不变量；
- L1 布局：lower half [B0|A0], upper half [B1|A1], bias at tail 64-byte aligned；
- BiasType 一致性合同（GM/L1/BT/Tiling 四处一致）；
- 分形轴对齐约束（transB=true: N%16==0; transB=false: K%16==0）；
- 使用官方 Scheduler（`MatmulSwatScheduler`）或 custom Block 的证据和授权范围。

每个新增或替换的接线必须写入 `abi_crosswalk_delta`，并引用对应 `matmul_base_analysis.abi_bindings[]` 的 `kernel_abi_contract`/签名骨架和 `base_design_binding_ref`。

### 4.2 Kernel层

按 [V+C 同步与 L1 布局专题](vc-sync-and-l1-layout-design.md) 冻结：

- concrete mixed Kernel 入口（`__global__ __aicore__ __mix__(1,2)`）、AIC/AIV 角色；
- AIC 侧：BlockMmad + Scheduler 初始化和 K 循环；
- AIV 侧：Prologue 初始化、UB 管理、VF 反量化、L1 写入和 CV 同步；
- CV 同步：HardEvent flag 配对、首轮预置、K-loop 内交替、final drain 消费剩余；
- slot 数、flag ID、pipe 和 producer/consumer 的当前 witness 事实；
- Params/TilingData、Wrapper 和 Launcher 的新增 ABI。

每个新增或替换的接线必须写入 `abi_crosswalk_delta`，并引用对应基础 binding 的 `kernel_abi_contract`/签名骨架和 `base_design_binding_ref`。本文不允许因 V+C 融合需要重排、重命名或重新解释基础 MatMul 的 A 侧 GM 参数；需要改变基础 ABI 时回通用 Step 3。

ratio、flag ID、bridge 等只能来自当前 witness；历史设备记录必须同时绑定同一 Investigation、Blaze 组装方案、构建和验证范围才能复用。

### 4.3 Prologue层

按 [VF 反量化链路专题](prologue-vf-dequant-design.md) 冻结：

- B 输入 dtype（int8/fp8_e4m3fn/fp8_e5m2）、B_dequant 输出 dtype（bf16/fp16）和 Cast 链（见专题 §3 已验证链路表）；
- 每步 Cast 的 CastTrait、LoadDist、StoreDist 配置（从 asc-devkit API 文档确认，编译+精度验证）；
- `__simd_vf__` + `asc_vf_call` 三层调用结构和参数打包 struct；
- 按 BType 的 `if constexpr` 分支选择中间寄存器类型和 Cast 链；
- UB ping-pong 布局：[bIn|bOut|scale|offset] × 2 half；
- UB 空间约束公式和 scale/offset 加载时机；
- hasOffset 的 `if constexpr` 编译期分支；
- Prologue K-L1 循环与 AIC K 循环的对应关系；
- CV 同步的 AIV 侧 wait/set 交替。

每路 operand 和中间值必须按自己的 dtype/alignment 计算 UB 空间。不得把整块 UB 同时分配给多个 buffer，或从 Asset 继承固定容量。

## 5. Tiling/资源/同步合同

按 [Tiling 参数合同专题](weight-quant-tiling-contract.md) 冻结：

- TilingData 逐字段语义、单位、合法域和 consumer 映射；
- Tiling Engine（`WeightQuantTilingSwat`）的入口签名和对齐校验逻辑；
- Tiling 与 BlockMmad/Prologue 的空间一致性（L1 bias 尾部、L0C 双缓冲、UB 约束、L1 总空间）；
- UB 空间约束公式必须按当前 B dtype 参数化，不得硬编码 int8 系数；
- Tiling Engine 的 GetTilingData 签名必须接受 weightElemBytes 和 dequantBElemBytes 参数；
- DESIGN 必须冻结当前项目的 weightElemBytes 值（由 B dtype 决定）；
- 分形轴对齐校验（transB=true: N%16==0; transB=false: K%16==0）；
- transB=true 时尾轮 N 方向不切分；
- host Tiling 策略：复用 Asset Engine（逐字段兼容后）、固定合法值或独立计算。

Tiling/Params 字段、同步事件和资源约束必须与 Block层、Kernel层和 Prologue层 delta 的字段交叉一致。不一致时回 Step 3 修正。

## 6. 场景验证增量

使用 [精度诊断专题](pitfalls-and-diagnosis.md) 将场景增量写入 DESIGN：

| 模式 | 隔离目标 |
|---|---|
| A-only-MMAD | AIC 侧 MMAD（使用预反量化 B_dequant 作为输入） |
| Dequant-only | AIV 侧反量化（零 MMAD，验证 VF Cast 链） |
| Full | 完整 V+C（AIV 反量化 + AIC MMAD + CV 同步） |

具体 dtype、shape、transB、hasOffset、hasBias、重复次数和阈值由需求与 DESIGN 冻结。至少包含：

- 逻辑输入、物理转换、CPU Golden（bf16 精度）和全元素非有限值门禁；
- 每个声明支持的 transB/hasOffset/hasBias 组合及 alignment/tail 边界；
- 单变量负/正对照，未得到两者时不得宣称根因；
- 清理临时诊断注入、Dump 和诊断入口后的 Full 回归；
- 一个 C+V epilogue 需求的负向路由检查，证明其不命中本场景。

Step 3 只能记录 `planned/unverified`。任何 `device_verified` 结论必须引用与当前 Investigation、Blaze 组装方案、构建和测试范围一致的外部验证记录。

## 7. 输出、合并与门禁

DESIGN 的场景合同必须输出：

```text
selected_scenario: weight-dequant-prologue-fusion
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
formula_and_dequant_contract
kernel_and_layer_contract
block_layer_delta
kernel_layer_delta
prologue_layer_delta
final_interface_delta
abi_crosswalk_delta
tiling_resource_data_sync_contract
tiling_resource_data_sync_delta
customization_scope
scenario_validation_additions
scenario_unsupported_boundary
```

本场景用以下关系满足通用场景合同：`formula_and_interface_delta` 由 `formula_and_dequant_contract` 和 `final_interface_delta` 展开；`kernel_and_layer_contract` 由 Block层、Kernel层和 Prologue层 delta 展开；`tiling_resource_data_sync_contract` 由 `tiling_resource_data_sync_delta` 展开。

合并顺序固定为：

```text
requirements
  -> operator interface
  -> matmul_base_analysis
  -> abi_bindings[] within matmul_base_analysis
  -> scenario delta
  -> final operator/kernel/resource/validation contracts
```

本场景是冲突 owner：能由已闭合证据解决的冲突必须写明保留/替换关系；需要新 Blaze 源码事实时回 Step 2，需要改变需求/接口/候选时回通用 Step 3。禁止循环"校正"或现场实验。

DESIGN 冻结前不得读取开发指导正文。场景合同与通用最终合同闭合后，Step 3 写入 `implementation_route: blaze_custom` 与唯一 `selected_scenario`，再读取[场景开发指导](weight-dequant-prologue-fusion-development.md)编译项目 `operators/<operator_name>/docs/PLAN.md`。PLAN 未完整覆盖本场景 required 方法时，回 Step 3 修正，不交给实施阶段。
