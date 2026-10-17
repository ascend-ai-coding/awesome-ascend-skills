# Step 4: Implementation

> **定位**：仅在明确完整“开发算子”的 Blaze 四步流程中，执行 Step 3 生成的 `operators/<operator_name>/docs/DESIGN.md` 和 `operators/<operator_name>/docs/PLAN.md`。direct invoke Developer 直接消费相同文档，不调用本 Step。Step 4 不重新设计、不重新匹配场景或扩大支持域；按冻结合同推进，并可为实现设计补充算子项目内修复步骤。

## 1. 规范性输入和职责边界

Step 4 的规范性输入只有当前项目：

```text
operators/<operator_name>/docs/DESIGN.md  # 设计合同：做什么、为什么、边界和验证合同
operators/<operator_name>/docs/PLAN.md    # §1/§3/§9/§10 设计基线；§2/§4--8 状态；§11 追加记录
```

PLAN 的 `reading_manifest`、action `source_refs` 和 `read_before_action_ids` 是初始阅读基线，不是 Step 4 的封闭读取清单。Step 4 可以按实际实现和错误证据读取项目文件、生成日志、实际命令、构建缓存、相关工具链资料和排障文档；这些资料只用于实现冻结设计，不得成为第二套设计入口。

### 1.1 问题触发的排障

当前 action 或 checkpoint 失败、输出异常或结果偏离预期时，扫描 [`references/troubleshooting/`](../troubleshooting/)：

1. 枚举目录下所有 Markdown 文档；
2. 只读取每篇文档开头的“适用范围”和“触发信号”；
3. 完整读取与当前问题匹配的文档；
4. 定位并修复问题，重新执行原 action 或 checkpoint。

没有匹配文档时，继续使用常规工程方法诊断；缺少排障文档本身不构成阻塞。问题定位期间可以检查错误直接涉及的工具链头文件、项目文件和构建产物；这些内容只作为诊断证据，不能改变冻结设计。

为实现冻结设计，Step 4 可以在 `<project-root>/operators/<operator_name>/` 内新增、修改或删除实现所需文件，包括源码、构建文件、Tiling、Launcher、测试和诊断代码；可以调整 include 路径与顺序、target 级编译选项，以及当前项目和既有工具链/运行时依赖的链接配置。新增或调整实际文件时更新 PLAN 第 2 章；同步更新第 4--8 章状态/结果，并把问题、实际修改、验证结果和证据追加到第 11 章 `execution_record`。不回写 DESIGN 或 PLAN 第 1、3、9、10 章。

修复一旦需要改变实现路线、Blaze 组装方案、算子接口或 Kernel ABI、数据语义、支持范围或验证标准，停止并回 Step 3；需要新的 Blaze 源码事实时先回 Step 2。Blaze 源码、submodule 或只读副本异常回 Step 1。项目内新增文件不单独触发回退，新的外部依赖仍需回 Step 3 评估。

设计请求跳过 Step 1 且项目内官方副本缺失时，只能执行 PLAN 冻结的 `create_only_exact_read_only_materialization`：从当前 Investigation 绑定的 `blaze_source_root` 原样复制或绑定缺失目录，核对同源内容后立即设为只读，再进入其他 action。该初始化动作不是 custom 适配，不得改写、筛选或补丁官方文件；PLAN 未授权、来源不一致或核对失败时回 Step 1。

禁止：

- 重新选择 route、Blaze 组装方案、Kernel/接口/ABI 合同，或扩大支持域和验证标准；Tiling 合法值、资源使用和同步实现可在冻结语义内修复；
- 重新匹配场景、读取 registry 做路由或把场景 development guide 当独立指令；
- 从相似工程、旧 recipe、Asset 注释或资料推导新的设计、路线或支持范围；
- 将 DESIGN 中的“备选”自动切换为执行方案；
- 修改 DESIGN，或修改 PLAN 第 1、3、9、10 章；
- 修改根 `ops-tensor`、Skill Asset 原文件、项目根之外的路径，或在已登记的缺失目录原样物化之外写入项目内官方副本和 DESIGN `forbidden_change_scope`；
- 引入具体提交 ID、hash、构建时间戳或提交节点；
- 修改或引用独立验证工程。
- 在 host 侧执行算子需求中声明的任何计算步骤；算子的全部计算（包括但不限于反量化、scale/dequant、cast、activation、normalization 等）必须在 device Kernel 中完成，不得将 host 侧计算的中间结果作为 device Kernel 的输入。

## 2. DESIGN/PLAN 联合门禁

在创建、复制、修改或删除任何项目文件前，逐项核对：

```text
implementation_route: blaze_native | blaze_custom
checkout_consistency: confirmed
design_plan_consistency: confirmed
frozen_plan_status: ready
```

同时确认：

- DESIGN 和 PLAN 双向引用，project/design/plan IDs、一致性 marker、route、scenario（如有）和唯一首选一致；
- 没有 unresolved blocking、TBD、未选分支或需要 Step 4 决定的设计事项；
- PLAN `target_file_manifest` 是 DESIGN `allowed_change_scope` 的初始子集；Step 4 的实际项目文件可以在项目根内扩展，须避开 DESIGN `forbidden_change_scope`，并在 `execution_record` 记录；
- 初始 `target_file_manifest.target_file`、action `target_files` 和可写交付件必须记录为以 `operators/<operator_name>/` 开头的项目相对路径；官方源码区和 Skill Asset 只能作为只读来源；
- 项目内官方副本已同源核对，或 PLAN 具有最先执行的 create-only 原样物化 action；除该缺失目录初始化外，官方副本不允许任何写入；
- 每个初始 action 都有 DESIGN refs、source refs、读取前置、计划目标、产物、验证和 rollback；修复不要求补 action；
- 存在 CMake configure 或 build action 时，PLAN 写明构建目标、预期结果、构建 checkpoint 和失败处理基线；项目构建文件不是 Step 4 修复的封闭清单；
- DESIGN 的 `abi_mapping_draft`、每个首选 `design_binding_ref` 的 source-backed `kernel_abi_contract`、`abi_crosswalk`、`source_backed_signature_skeleton` 和（适用时）场景 `abi_crosswalk_delta` 已闭合；每个必需逻辑参数和辅助 ABI 对象都能以稳定 `crosswalk_row_id` 在同一 binding 内追到物理 buffer、Launcher、Wrapper、Kernel 参数、Tiling/Params（或有依据的 `not_applicable`）和设备消费者；
- 所有涉及 buffer、Tiling/Params、Wrapper、Kernel entry 或 Launcher 的 action 都引用相应 ABI 合同项；其中 Wrapper、Kernel entry 或 Launcher 启动绑定 action 还引用 source-backed 签名骨架，且静态 ABI checkpoint 覆盖物理字节/offset、TilingData、workspace、grid/usedCore、Wrapper/entry 绑定和输出生命周期；
- 每个验证合同、交付件、清理要求都有 checkpoint/action；
- 除已登记的 create-only 缺失目录原样物化外，只读源码区、Asset 原文件、DESIGN forbidden scope 和项目根之外的路径未被修改；物化后的官方副本内容零改动；
- `implementation_route=blaze_native` 时，PLAN 不含场景 action 或 custom fallback；
- `implementation_route=blaze_custom` 时，PLAN 的场景要求映射完整且只有一个 `selected_scenario`；
- `implementation_route=unsupported`，或没有最终路线/DESIGN/PLAN 的“等待澄清”状态，均不能进入 Step 4。

任一真正的设计或源码事实门禁失败，停止在文件修改前。缺少真实入口、GM 参数、TilingData/Params、workspace、grid/usedCore、Wrapper 或物理字节规则的 Blaze 源码事实时，回 Step 2；冻结设计合同不足以定义需求语义、接口/ABI 或验收标准时回 Step 3；若失败原因是 Blaze 源码版本不一致，回 Step 1。普通资料缺失、文件未列入初始 manifest 或 action 需要实现修复时，不单独阻塞 Step 4。

## 3. 四阶段执行 Runbook

### 3.1 阶段 0：联合交接

读取 DESIGN 及 PLAN 第 1、3、9、10 章，确认当前项目 Blaze 源码版本、只读边界、初始目标文件和 execution record 空间。在第 4 章更新交接进度，并在第 11 章追加 handoff 结果；不修改设计基线。

### 3.2 阶段 1：执行材料与文件核验

以 PLAN `reading_manifest` 和 `target_file_manifest` 为起点：

1. 加载当前 action 已登记的来源和位置，并按实现需要检查项目文件和工具链资料；
2. 当前 action 为 CMake configure 或 build 时，确认 action 的构建目标、预期结果、checkpoint 和失败处理基线完整；
3. 核对初始 `read_before_action_ids`、source refs、依赖入口和实际头文件；缺少资料时先在项目和工具链中查找，不因未登记自动回退；
4. 对当前 buffer/Tiling/Params/Wrapper/Kernel/Launcher action 核对其 `abi_contract_refs`、`design_binding_ref`、`crosswalk_row_id`（场景增量为 `delta_crosswalk_row_id`）和签名骨架与同一当前源码 witness 一致；
5. 核对每个目标文件的 action type、source、允许范围和预期产物；
6. 不创建目录、不复制文件、不修改文件。

只有发现只读文件被修改、项目根越界或冻结设计语义不足时才停止并回退；材料未登记或初始目标不完整时，优先在项目内补齐实现并记录实际变更。

### 3.3 阶段 2：有序 action 执行

严格按 `ordered_actions.sequence` 和 prerequisites：

- 初始 `create`、`copy_and_adapt`、`modify`、`delete`、构建、运行、记录和清理都必须是显式 action；修复步骤可以直接记录在 `execution_record`，不要求补 action；
- 官方副本的 create-only 物化 action 必须先执行并完成同源、内容和只读状态核对；它不能改为 `copy_and_adapt`，也不能在失败时切换来源；
- 每个初始 action 按计划目标推进，完成后执行其 checkpoint；
- action 或 checkpoint 失败、输出异常或结果偏离预期时，按 1.1 节加载匹配资料，在项目根内诊断、修复并重新验证；不要求先补 action 或重新登记文件；
- 修复循环不预设固定重试次数；只要证据仍指向实现层且能取得新进展，就继续修复并重跑失败项及受影响的下游 checkpoint；
- 不改路线、不换候选、不扩大参数/shape/dtype/场景支持域；实现修复可新增项目内文件和实现步骤；
- 复制 Asset 后只能适配项目目标副本，Asset 原文件保持零 diff；复制或适配必须保持初始 PLAN 冻结的相对 include 拓扑，或同时落实已冻结的源到目标映射和 include 改写；完成后用目标实际 compiler/语言/架构执行 include-closure 探针；
- 第 2 章更新实际文件状态，第 4--8 章更新进度和结果，第 11 章追加实际文件、输出、证据、结果和 deviation；不能改写第 9 章动作定义。

### 3.4 阶段 3：验证、清理和交付

按 PLAN `validation_checkpoints` 执行静态 ABI、构建、功能、边界、物理地址、资源/生命周期、精度、重复运行、诊断清理和最终回归。清理按 PLAN 基线执行，并覆盖执行期间产生的临时诊断文件和产物；交付只核对 `deliverable_manifest`。任何设备结论必须通过 `execution_record.evidence_refs` 关联当前 Investigation、冻结首选组装方案的真实 witness，以及实际构建和验证记录；任一关联缺失时，对应组合保持 `unverified`，不得泛化。

任何 required checkpoint、交付件、只读审计或清理后回归失败，都不能标记完成。先保留证据并继续诊断、修复和重跑；只有确认需要改变冻结设计、补充 Blaze 源码事实或修复源码基线时，才按第 5 节回 Step 3/2/1。暂时无法解决的实现问题记录为 `failed`，工具链或设备不可用记录为 `blocked`，不把它们伪装成设计回退。

## 4. PLAN 更新与 Execution Record 边界

PLAN 的所有权固定为：第 1 章需求语义和第 3 章测试范围/追溯/验收标准是设计基线；第 2 章文件状态、第 4 章进度、第 5 章问题/决策、第 6--8 章结果由实施阶段持续更新；第 9、10 章冻结；第 11 章只能追加。设计基线变化必须在第 5、11 章记录 `design_issue` 并返回 Step 3。

第 11 章每条记录至少包含：

```text
execution_id
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

如果 deviation 改变需求语义、路线、Blaze 组装方案、接口/ABI、验证标准或支持域，必须停止当前实现，记录证据并把 `next_action_or_return_target` 指向 Step 3；不得把设计回退记成环境 `blocked`。项目内新增文件、实现步骤、Tiling 合法值、资源/同步实现或配置修复更新第 2、4--8 章并追加实际修改和验证证据，不改写第 1、3、9、10 章。

## 5. 失败分类与回退

| 失败类型 | 处理 |
|---|---|
| Blaze 源码版本不一致 | 停止修改，回 Step 1 |
| Blaze 源码事实、类型链、Params、API 或物理 ABI 被当前源码推翻 | 保留证据，回 Step 2，再重做 Step 3 |
| Blaze 源码、submodule 或项目只读副本不完整/不一致 | 只读采集完整性证据，停止修改并回 Step 1 |
| 冻结设计、接口/ABI、支持范围或验证标准缺失/冲突 | 文件修改前回 Step 3 |
| CMake configure、编译、链接、精度或普通实现失败 | 按 1.1 节持续排障、修复并重新执行受影响 checkpoint |
| 修复需要改变需求语义、路线、Blaze 组装方案、接口/ABI、支持域、验证标准或引入新的外部依赖 | 保留证据，回 Step 3 |
| 执行材料缺失但设计和源码事实完整 | 在项目内补齐实现并记录实际文件，不自动回退 |
| canonical Skill/场景注册/指导自身缺失或冲突 | 源码事实不足时回 Step 2；否则记录 `failed` 并进入 Skill 维护，不伪装成设计或环境回退 |
| 普通实现错误 | 在项目根内修复并重新验证；只有触及冻结设计才回 Step 3 |
| 验证推翻设计前提 | 停止受影响 action，回 Step 3；需要新事实先回 Step 2 |
| 用户需求改变 | 停止执行，回需求合同和 Step 3 |

回退保留用户已有改动和失败证据，不做整体 Git 恢复，不覆盖只读区域，也不恢复旧 recipe 作为正式路由。

## 6. 完成门禁

Step 4 完成必须同时证明：

- PLAN 所有 required action、checkpoint、deliverable、cleanup 和 final regression 已记录结果；
- 项目文件均位于项目根内；除已登记的 create-only 缺失目录原样物化外，官方源码区、Asset 原文件和 DESIGN forbidden scope 未改写；
- 目标 route、唯一首选、接口/Kernel/Tiling/资源/同步/验证合同与实现一致；
- 已执行的 buffer、Tiling/Params、Wrapper、Kernel entry 和 Launcher 接线均符合冻结的 ABI crosswalk 和签名骨架；
- 所有 required 精度/边界/非有限值门禁通过；
- 每项已记录的设备结论均有绑定当前 Investigation、冻结首选组装方案真实 witness、构建和验证记录的 `execution_record.evidence_refs`；未验证组合保持 `unverified`，不作为交付支持范围；
- 清理后正式回归通过；
- 根 `ops-tensor`、全部 Skill Asset 原文件和独立验证工程零改动；项目内官方副本与当前根源码同源，且物化完成后内容零改动；
- 最终 execution record 明确交付或阻塞原因。

下一阶段不是新的 Step；最终交付审计属于本 Step 4。
