# Step 3: Kernel Design

> **定位**：依据精确需求和 Step 2 的 Blaze 源码调查，完成唯一的项目路线决策，并生成冻结的项目设计与执行计划。本步骤不复制文件、不写实现、不构建项目或运行设备。

## 1. Inputs and Outputs

必须先读取：

- 用户需求，以及调用上下文直接提供的 `project_root`、`operator_name`、`target_chip`、`npu_arch` 和可选 `cann_version`；
- `operators/<operator_name>/docs/blaze/blaze-investigation-report.md`；
- [Blaze DESIGN 模板](../kernel-design/blaze-design-template.md)；
- [Blaze PLAN 模板](../kernel-design/blaze-plan-template.md)。

Step 1 项目/版本 handoff 是完整四步流程的可选输入，不是设计请求的启动依赖。Step 3 不读取 `environment.md`、外部 workflow manifest 或其他 direct invoke 产物，也不重新大范围搜索 Blaze 源码；判断需要的源码事实缺失时，向 Step 2 提交一次精确的补充调查问题。

正常完成后，输出固定写入当前项目：

- `operators/<operator_name>/docs/DESIGN.md`：需求、接口、路线、最终组件/场景合同、资源数据同步、支持边界和验证合同；
- `operators/<operator_name>/docs/PLAN.md`：仅可执行路线的精确资料、目标文件、有序动作、接线、checkpoint、交付、清理和回退。

`unsupported` 只生成阻塞 `DESIGN.md`，不生成 `PLAN.md`。一次补充调查后仍缺关键事实时，不生成最终路线、DESIGN 或 PLAN，停止等待用户澄清。任何可写目标都不得位于 `<project-root>/operators/<operator_name>/` 之外。DESIGN/PLAN 的版本字段继承当前 Investigation 的抽象版本记录；不得写实际提交 ID、manifest/hash、文件哈希、构建时间戳或具体提交节点。

## 2. Top-Down Design Workflow

```text
需求合同
  -> 算子接口合同
  -> 基于 Blaze 官方库的方案分析
  -> 定制扩展场景匹配与设计（可选）
  -> 最终合同装配
  -> 验证合同
  -> PLAN 编译与冻结
```

每个阶段记录输入、动作、输出和停止条件。前置合同未闭合时，不得跳到后续阶段。

### 2.1 Requirements Contract

从用户需求冻结可逐项核对的合同：

- 数学公式、输入输出、dtype、累加和转换顺序；
- topology、layout/transpose、shape/tail、运行时变化维度；
- 额外输入输出、精度/资源目标、错误语义和排除范围；
- demand partitions、硬约束和支持边界。

Basic、Batch、Grouped、Quantized、MX 只是需求画像维度，不能从名称推导 Blaze 组件兼容性或项目路线。

### 2.2 Operator Interface Contract

先读取 Investigation 的物理/ABI事实、[MatMul Layout](../fundamentals/blaze-matmul-layout.md) 和 [Launcher 开发](../launcher/launcher-development.md)，再冻结外部逻辑接口及其接口到设备的映射草案。此阶段不从旧样例或未选 Kernel 签名反推用户接口。

`operator_interface_contract` 必须记录：

- 参数角色、顺序、逻辑 shape/dtype/layout；
- 必选/可选/默认值、运行时/编译时归属；
- host 调用、所有权、生命周期、错误和拒绝语义；
- 每个输入、输出和辅助对象的物理数据前置条件。

同时输出 `abi_mapping_draft`。每个逻辑参数或运行时辅助对象至少有一行：

```text
abi_mapping_draft_row_id
logical_argument_id
logical_role_and_requiredness
logical_shape_dtype_layout
physical_buffer_role_and_storage_rule
byte_extent_offset_and_unit_rule
host_owner_and_lifetime
runtime_or_compile_time_owner
intended_device_destination_role
rejection_condition
evidence_refs
```

这是一份从用户接口到设备数据流的草案：它必须明确每个逻辑对象将如何成为物理 buffer，并指出预期由 Kernel GM 参数、Tiling/Params、workspace 或其他设备消费者接收；但不得在没有真实证据时填写具体 GM 参数名、入口修饰符、Wrapper 形态、模板参数或 grid。精确设备 ABI 在 2.3 冻结。

逻辑接口本身不清时请求用户澄清；物理格式、字节规则或 ABI 前置事实不清时返回 Step 2。两类问题未闭合时不得进入 2.3。Kernel entry、GM 参数、TilingData、grid、pipe、event 和 slot 不能反向定义用户接口。

### 2.3 基于 Blaze 官方库的方案分析

本阶段对每个可进入设计的需求始终执行。它使用 Step 2 中的候选 Blaze 组装方案事实，按 2.1 的每个 requirement/partition 精确判断能否形成完整的 Blaze 官方库方案；它是路线决策的第一部分，而不是 Step 2 的复述。

按以下顺序执行：

1. 读取候选真实入口、具体证据、组件链、specialization、Tiling/Params、Tensor API、物理数据、host/device ABI、资源、final/partial 和生命周期事实；同时读取已观察限制与未闭合源码事实、[Tiling 方法](../kernel-design/tiling-selection.md) 和 [Launcher 开发](../launcher/launcher-development.md)。
2. 将每一项事实映射到精确需求，形成 `matmul_base_analysis` 和候选处置记录。不得按名称拼接类型、构造模板笛卡尔积，或以 Asset/旧工程证明 Blaze 官方库能力、固定 ABI 或支持范围。Asset 只能在兼容性证明且 DESIGN 明确选择后作为项目副本的结构起点。

每个可行候选的选择状态只能是“首选”或“备选”，且每个 required partition 最多一个首选。淘汰、阻塞和不适用候选单独记录在候选处置中；PLAN 只消费首选合同，不自动激活备选。
3. 对每个保留为设计 binding 的具体 Blaze 组装方案，以同一真实 witness 在 `matmul_base_analysis.abi_bindings[]` 中冻结独立 ABI 子合同。每条记录必须写入 `design_binding_ref`、覆盖的 `partition_ids` 和 `assembly_witness_ref`；其中的 `kernel_abi_contract` 必须包含入口修饰符和链接形式、入口符号、模板/具体 specialization、GM 参数的顺序/方向/可空语义/物理字节规则、TilingData/Params、workspace、grid/usedCore、Wrapper 绑定、dispatch，以及 final/partial 输出生命周期。每项都必须有对应 `source_ref`。
4. 为每个选定 binding 在其 `matmul_base_analysis.abi_bindings[]` 记录中输出独立 `abi_crosswalk`：

```text
logical argument
  -> physical buffer and storage rule
  -> Launcher argument
  -> Wrapper argument
  -> ordered Kernel GM parameter
  -> TilingData/Params field or not_applicable with reason
  -> Block/Kernel/Epilogue device consumer
```

   每行还必须记录稳定 `crosswalk_row_id`、方向、可空条件、字节范围/offset 单位、所有权/生命周期和 `source_refs`。TilingData、workspace、grid/usedCore、stream/dispatch 等非逻辑参数也要作为辅助 ABI 对象记录其 host 产生者、入口/Wrapper 绑定和设备消费者。不得将不同 `design_binding_ref` 的参数顺序或 crosswalk 行混用。
   若设计使用 Skill Asset 或项目副本作为结构起点，必须额外完成 Asset 能力边界核对：

   - 记录 Asset 实际提供的 Kernel、Block、Scheduler、Epilogue 和同步层；
   - 记录它明确不提供的 Host Tiling、Launcher、额外 operand、workspace、Golden 和验证层；
   - 将目标项目的适配点和首个设备消费者接入现有 ABI crosswalk、场景 delta 和 source refs。
   - 对会被项目 include 的 Host/ASC 资产执行同名全局 helper 探针；内部
     `Align`、`CeilDiv`、`FloorAlign` 等 helper 必须显式限定所属命名空间，不能
     依赖 include 顺序或调用方的 `using namespace` 消除歧义。

   任何一层未闭合都必须标记为 `adaptation_required` 或 `extension_missing`，不能因为
   Asset 文件存在或局部组件可编译就将端到端方案标记为 complete/native。
5. 从每个真实 witness 在对应 `abi_bindings[]` 记录中生成 `source_backed_signature_skeleton`，保留入口声明形态、参数顺序、Wrapper 或 dispatch 调用关系及其 `source_refs`。它是设计合同，不是可复制 Kernel recipe；只有 witness 已明确的修饰符、`__cube__`/`__mix__`、GM 参数、模板参数和调用方式才能写入，其他位置保留待项目命名的占位，不得猜测。
6. 若某一项决定性事实缺失且报告尚无已完成的补充，提出只描述需求语义和待查 Blaze 源码关系的补充问题，返回 Step 2。补充问题不得包含场景 ID、场景路径或路线建议。若同一报告已完成一次补充后仍缺关键事实，停止为“未完成设计，等待澄清”；不写最终路线，也不生成 DESIGN/PLAN。
7. 若已有事实表明某个需求不能由选定官方方案覆盖，逐项记录 `native_gaps`，再进入 2.4。可接受的依据是来源明确的不兼容/拒绝，或对该精确需求已穷尽声明读取边界后的无匹配事实；孤立 `not_found`、`indexed`、未读目录或未知事实不得写为 `native_gaps`。

   `native_gaps` 的判定对象是 §0.4 需求合同中的完整算子语义，不是其中的某个子步骤。官方方案覆盖性检查必须满足以下原则：
   - 算子的全部计算步骤（包括 matmul 及其前/后的所有非 matmul 计算）必须由单一 Blaze 官方 Kernel 方案覆盖，才算 blaze_native。多个独立 Kernel 分别覆盖不同步骤不构成 blaze_native。
   - 禁止将用户需求中的任何计算步骤排除到 Blaze 官方方案分析范围之外（如声明为"辅助预处理""host 侧步骤""非 Blaze 操作"等），从而缩小 `native_gaps` 的判定范围使剩余步骤单独满足 blaze_native 条件。
   - 算子的所有计算必须在 device 侧完成。不得将需求中声明的计算步骤放到 host 侧执行后将其结果作为中间输入传入 device Kernel。

8. 若一个完整官方方案以单一 Kernel 覆盖算子的全部计算步骤（包括 matmul 及其前/后的非 matmul 计算），且全部在 device 侧完成，写入：

```text
implementation_route: blaze_native
selected_scenario: <omit>
```

此结果不读取 `references/scenarios/index.md`、场景设计指导或场景开发指导。

Blaze 源码没有现成 host Tiling 实现本身不构成缺口。先逐字段核对可复用的项目 Tiling Engine；能证明 Params 语义、单位、合法域和 ABI 兼容时，PLAN 可复用或最小适配。不能复用但上述事实已闭合时，PLAN 可令项目 Tiling Engine 返回经过证明的固定合法控制值。需要新的 Params、specialization、合法域或 ABI 事实时，返回 Step 2。

### 2.4 定制扩展场景匹配与设计（可选）

本阶段仅在 2.3 已记录至少一个 `native_gaps` 后执行。它直接使用 2.1 的精确需求合同匹配 [场景索引](../scenarios/index.md)，不使用 Step 2 的场景预判，也不以“额外语义”之类 flag 作为前提。

1. 读取每个索引行的支持范围和准入条件；当前只允许命中一个场景，不组合多个场景。
2. 若需求合同不足以判定任一支持范围或准入条件，停止等待用户澄清；不将需求歧义写成 `unsupported`。
3. 零命中或多命中：在 `DESIGN.md` 写入：

```text
implementation_route: unsupported
unsupported_points:
  - requirement_id
    native_gap
    scenario_match_result
    evidence_refs
    user_recovery
```

   停止设计，不生成 PLAN，也不进入 Step 4。
4. 唯一命中：读取该行的设计指导。场景指导检查自己声明的 Blaze 源码前提，以及是否消费 `matmul_base_analysis`。
5. 场景前提缺失：生成一次无场景名的补充调查问题并返回 Step 2。若同一报告已完成一次补充且关键事实仍缺失，停止为“未完成设计，等待澄清”；不写最终 `implementation_route`，不生成 DESIGN/PLAN。
6. 场景前提充分：写入：

```text
implementation_route: blaze_custom
selected_scenario: <唯一 scenario_id>
```

   并由场景设计指导生成定制扩展合同。

场景是否消费 2.3 的 `matmul_base_analysis` 由该场景自身定义。当前 Elementwise/Broadcast Epilogue Fusion 明确消费该合同；后续场景不自动继承。

### 路线结果表

| 已知事实与匹配结果 | Step 3 行为 | 交付 |
|---|---|---|
| 判断需要的 Blaze 源码事实缺失，尚未补充 | 生成一次语义化补充问题，回 Step 2 | 无最终路线、无 DESIGN/PLAN |
| 一次补充后仍缺关键事实，或需求合同本身不清 | 停止等待用户澄清 | 无最终路线、无 DESIGN/PLAN |
| 官方 Blaze 方案以单一 device Kernel 覆盖算子全部计算步骤 | `implementation_route=blaze_native`，跳过注册表 | DESIGN + PLAN |
| 官方方案有 `native_gaps`，场景唯一命中且前提充分 | `implementation_route=blaze_custom` | DESIGN + PLAN |
| 官方方案有 `native_gaps`，场景零命中或多命中 | `implementation_route=unsupported` | 仅阻塞 DESIGN |

### 2.5 最终合同装配

按以下顺序装配最终合同：

```text
requirements
  -> operator interface
  -> abi_mapping_draft
  -> matmul_base_analysis (per-binding ABI contract, crosswalk, signature skeleton)
  -> selected scenario delta (only blaze_custom)
  -> final operator/kernel/resource/data/sync/validation contracts
```

需求语义由需求合同持有，外部接口由接口合同持有，官方库分析由 `matmul_base_analysis` 持有，定制增量由唯一场景合同持有。场景必须明确 `consumed_contracts`、`preserved_contracts`、`replaced_contracts` 和 `added_contracts`，不得由通用流程猜测。

若发现需要改变前置设计合同，返回对应 owner；需要新的 Blaze 源码事实时返回 Step 2。不得在 Step 3 现场实验或循环重选路线。

### 2.6 Validation Contract

冻结逻辑数据、物理转换、CPU Golden、阈值、非有限值、边界、诊断、重复运行、清理和回归要求。验证矩阵只来自需求、Investigation 和已选场景指导。

Step 3 只能记录 `planned` 或 `unverified`，不得伪造设备结果。验证合同必须规定：任何设备结论均需绑定当前 Investigation、冻结首选组装方案的真实 witness，以及实际构建和验证记录；未满足时对应组合保持 `unverified`，不得泛化。`blaze_custom` 的专项验证由场景设计指导增加；`blaze_native` 使用通用 MatMul 验证合同。

### 2.7 PLAN Compilation

仅当 `implementation_route` 为 `blaze_native` 或 `blaze_custom`，且 DESIGN 合同已冻结时，用统一模板编译 `operators/<operator_name>/docs/PLAN.md`：

1. 绑定 Investigation、DESIGN、唯一首选和验证合同；
2. `blaze_native` 仅从官方库设计和通用方法生成 action；
3. `blaze_custom` 才读取唯一索引行的开发指导，并将要求实例化为 reading manifest、actions、checkpoints、交付和回退；
4. 核对 Investigation 的项目内官方副本状态。副本已由 Step 1 建立时登记 `read_only` 并将物化 action 记为 N/A；设计请求跳过 Step 1 且副本缺失时，为每个缺失目标生成首个 `action_type: create` action，只允许从当前 `blaze_source_root` 原样复制或绑定、核对同源内容并立即设为只读，不得适配或使用第二源码根；
5. 只要 PLAN 含 CMake configure 或 build action，必须读取 [Blaze CMake 构建指导](../fundamentals/blaze-cmake-build-guide.md)，将项目构建入口、目标、预期产物、成功 checkpoint 和失败后的 Step 4 排障入口写入初始 PLAN；项目构建文件可作为初始 `target_file_manifest` 目标，但不是 Step 4 修复的封闭文件清单；
6. 冻结 PLAN 第 9、10 章以及第 1、3 章设计基线；第 2、4--8 章留给实施阶段持续更新，第 11 章建立空的追加记录；
7. 删除备选与未激活分支，执行 PLAN freeze gate。

每个初始 action 必须有 DESIGN refs、source refs、读取前置、计划目标、产物、验证和回退。CMake action 不增加专用合同字段，也不要求 Step 3 预判实际编译中才会暴露的 include 顺序、编译选项、链接参数或 `-iquote`。PLAN 不得留下需要 Step 4 决定的路线、接口、ABI、支持范围或验证标准；实现层的文件、配置和修复细节可由 Step 4 根据证据闭合。`unsupported` 和“等待澄清”路径均不产生可执行 action。

## 3. Reading Routes

| 设计节点 | 必读资料 | 条件资料 | 禁止 |
|---|---|---|---|
| 需求合同 | 用户需求、Investigation 的需求投影 | 无 | recipe、实现样例 |
| 接口合同 | Investigation 的物理/ABI事实、[MatMul Layout](../fundamentals/blaze-matmul-layout.md)、[Launcher 开发](../launcher/launcher-development.md) | 报告明确的公共 API | 未选 Kernel 签名、场景正文 |
| Blaze 官方库方案分析 | 候选 Blaze 组装方案、实际 witness、限制、未闭合事实、[Tiling 方法](../kernel-design/tiling-selection.md)、[Launcher 开发](../launcher/launcher-development.md) | 框架方法 | 场景索引、场景正文、Asset 能力推断 |
| 定制场景设计 | `native_gaps`、唯一索引行的设计指导 | 场景规定的依赖 Skill 根入口和叶子 | 其他场景、development guide |
| 最终合同与验证 | 已生成合同、验证方法 | 场景验证增量 | 新候选、新设备事实 |
| PLAN 编译 | 冻结 DESIGN、Step 4 约束 | 含 CMake action 时必读 [Blaze CMake 构建指导](../fundamentals/blaze-cmake-build-guide.md)；`blaze_custom` 的唯一 development guide | 新设计分支、路线切换 |

## 4. DESIGN/PLAN Gates

可执行交接必须满足：

- `implementation_route` 是 `blaze_native` 或 `blaze_custom`；
- Step 2 报告中影响选择的 Blaze 源码事实已闭合，或具有来源绑定的明确限制；
- `blaze_native` 有完整官方库设计，且没有场景 action；
- `blaze_custom` 有唯一 `selected_scenario`、完整场景合同和场景指导映射；
- `abi_mapping_draft` 覆盖全部逻辑参数、输出和必要辅助对象，且物理数据、字节规则、所有权和拒绝条件闭合；
- 每个首选 binding 的 `kernel_abi_contract`、`abi_crosswalk` 和 `source_backed_signature_skeleton` 均绑定同一真实 witness；每个必需对象都能追到 Launcher、Wrapper、Kernel 参数、Tiling/Params（或有依据的 `not_applicable`）和设备消费者，且不混用其他 binding 的参数顺序或入口；
- DESIGN/PLAN 的 project/design/plan IDs、路线、场景、首选 binding 和一致性 marker 相同，且 `design_plan_consistency=confirmed`；
- PLAN 初始文件范围是 DESIGN `allowed_change_scope` 的子集；Step 4 的实现修复可在项目根内扩展，但不得触碰 `forbidden_change_scope`；
- 项目内官方副本已与当前根源码同源，或 PLAN 为缺失副本提供先于构建和实现的 create-only 原样物化 action；该 action 完成后副本只读，任何内容适配仍受 `forbidden_change_scope` 禁止；
- 每个首选合同有 action，每个验证合同有 checkpoint；所有涉及 buffer、Tiling、Wrapper、Kernel entry 或 Launcher 的 action 均引用对应 ABI 合同项，其中 Wrapper、Kernel entry 或 Launcher 启动绑定还引用签名骨架；
- 存在构建 action 时，PLAN 已提供初始构建入口、configure/build 目标、预期结果、构建 checkpoint 和失败处理基线；
- 无 blocking、TBD，以及未决的路线、接口/ABI、支持范围或验证标准；
- 官方源码区和 Asset 原文件只读。

`unsupported` 只允许输出 `native_gaps`、`unsupported_points`、用户恢复路径和禁止执行范围。一次补充后仍缺事实的路径不输出最终合同。

## 5. Handoff

仅 `blaze_native` 或 `blaze_custom` 且联合门禁通过时，才能把 `operators/<operator_name>/docs/DESIGN.md` 和 `operators/<operator_name>/docs/PLAN.md` 交给实施阶段。完整四步 Blaze 流程进入 [Step 4: Implementation](step4-operator-development.md)；direct invoke Developer 直接消费相同文档，不调用 Blaze Step 4。实施阶段以第 1、3、9、10 章和 DESIGN 为设计边界，可在 `operator_root` 内调查实现问题、动态修复和记录额外文件，不重新设计路线、匹配场景或扩大支持域。
