# Elementwise/Broadcast Epilogue Fusion 开发指导

本文是 Step 3 在场景 DESIGN 冻结后编译项目 `operators/<operator_name>/docs/PLAN.md` 的方法输入，不是 Step 4 的独立设计入口。本文不重新匹配场景、选择 MatMul Blaze 组装方案、选择 MemBase/RegBase、改变支持范围或补充 DESIGN；实施阶段按 `operators/<operator_name>/docs/DESIGN.md`/`operators/<operator_name>/docs/PLAN.md` 实现，并可在 `<project-root>/operators/<operator_name>/` 内为闭合设计补充修复步骤。

## 目录

1. PLAN 编译门禁
2. 阅读与依赖清单
3. 有序动作规则
4. 接线和项目文件合同
5. 验证、交付与清理
6. 合规映射和回退

## 1. PLAN 编译门禁

Step 3 读取本文前必须确认：

```text
implementation_route: blaze_custom
selected_scenario: elementwise-broadcast-epilogue-fusion
matmul_base_analysis: present
matmul_base_analysis.abi_bindings[]: source_backed_and_closed_for_consumed_partitions
abi_crosswalk_delta: source_backed_and_closed
design_plan_generation_phase: after_design_freeze
```

并且：

- DESIGN 已冻结唯一首选、公式/broadcast、三层增量、Tiling/资源/同步、customization scope 和验证合同；`abi_crosswalk_delta` 只追加额外 operand、输出、Params、Wrapper 或同步接线，且每行具有 `delta_crosswalk_row_id` 并引用 `base_design_binding_ref` 和基础 MatMul `crosswalk_row_id`；
- DESIGN 的 `dependency_skills` 与 `dependency_skill_evidence` 已冻结，覆盖当前首选路线的根入口、必要叶子及 API/路线证据；
- 根 `ops-tensor` 和 Blaze skill Asset 原文件始终只读；项目内官方副本只允许执行 PLAN 登记的缺失目录 create-only 原样物化，物化后只读；
- 无决定性 unknown、TBD、未选择分支、Grouped 基础分区缺口或要求 Step 4 决策的字段。

失败时回 Step 3，不生成可执行 PLAN；不得把本文默认值或历史实现写入可执行 action。

## 2. 阅读与依赖清单

Step 3 根据首选 DESIGN 生成 PLAN `reading_manifest` 作为初始阅读基线。初始资料应绑定 `read_before_action_ids` 和 action `source_refs`；未激活资料写 N/A 或不登记，Step 4 可按实际实现和错误证据读取必要资料，但不能据此改变设计。

| 设计合同 | PLAN 必读来源 | 条件 |
|---|---|---|
| Block层 delta | [Block层专题](block-l0c2ub-extension.md)、Investigation 指定的 concrete source locations | 使用/扩展 L0C2UB/GM 输出 |
| Kernel层 delta | [Kernel层专题](fused-kernel-development.md)、实际 mixed Kernel/同步定义 | 存在 AIC/AIV、slot 或 event 协作 |
| MemBase 首选 | DESIGN `dependency_skills` 记录的根入口和必要叶子、[MemBase 专题](epilogue-membase-design.md) | DESIGN 选择 MemBase |
| RegBase 首选 | DESIGN `dependency_skills` 记录的根入口和必要叶子、[RegBase 专题](epilogue-regbase-design.md) 及 DESIGN 记录的真实参考实现 | DESIGN 选择 RegBase |
| RegBase 使用普通 AscendC API | DESIGN `dependency_skills` 记录的普通 API 根入口和必要叶子 | 使用 DataCopyPad、HardEvent 等普通 API |
| SplitM | [SplitM 专题](splitm-contract-and-debugging.md) | DESIGN 激活 SplitM |
| 五模式/诊断 | [精度诊断专题](precision-diagnosis.md) | 所有本场景实现 |
| Tiling/Launcher | [Tiling 方法](../../kernel-design/tiling-selection.md)、[Launcher 方法](../../launcher/launcher-development.md) | 按 DESIGN 选择所需章节 |
| Event/pipe | [同步方法](../../fundamentals/blaze-sync-patterns.md) 和当前 CANN 实际头文件 | DESIGN 激活相关同步 |

PLAN 先登记 DESIGN `dependency_skills` 中的根入口，再登记当前首选路线需要的叶子。`reading_manifest` 只登记当前首选路线的初始资料，不能把全目录一股脑列入；Step 4 可为诊断和实现读取必要资料，但不能将其变成新的路线或支持范围。

## 3. 有序动作规则

Step 3 将下列稳定动作类别实例化为 PLAN 的 `ordered_actions`。每个初始动作必须含 DESIGN refs、source refs、计划目标文件、前置、产物、checkpoint 和 failure rollback。

### 3.1 来源与项目副本

1. 核对 DESIGN/Investigation 与当前 Blaze 源码版本的抽象一致性。
2. 对 DESIGN 选择的官方组件登记 read-only 来源；满足合同时直接引用，不创建 custom 副本。
3. 只有 `customization_scope` 明确授权时，登记从具体来源复制到项目目标文件并适配的 action，记录首个修改点和必须保持的不变量。
4. 只有 DESIGN 明确授权结构起点时，才可复制 [MemBase Asset](../../../assets/blaze_custom/epilogue/epilogue_fusion_membase.h) 或 [RegBase Asset](../../../assets/blaze_custom/epilogue/epilogue_fusion_regbase.h) 到项目文件；Asset 原文件只读，复制后必须按当前 ABI/公式/API 重写和验证。
5. 不复制整个 Blaze，不通过 include 顺序覆盖官方 specialization，不创建项目根之外或 DESIGN forbidden scope 内的目录。

### 3.2 Block层动作

按 DESIGN 冻结的 Block delta 实例化：

- L0C 源、UB/GM 目的、Copy API/trait、extent、row pitch 和 tail；
- final/partial、归并和输出分支；
- official/custom 类型链、namespace、Policy/Params 绑定；
- 结构/ABI 静态检查及必要的单变量正负对照。

未授权 custom Block 时只登记官方组件绑定和核对 action，不生成修改动作。

### 3.3 Kernel层动作

按 concrete witness 实例化：

- mixed Kernel entry、AIC/AIV 角色和 ratio；
- adapter 的参数、单位和地址语义；
- flag/event/pipe、slot 初始化/轮转/覆盖前等待、empty task 和 final drain；
- Block/Epilogue、Params/TilingData、Wrapper 的类型接线；
- DESIGN 已证明必要的 local event bridge 及其负向对照。

所有涉及新增 operand、输出、Params、Wrapper、Kernel entry 或同步的动作必须引用由 `base_design_binding_ref` 选定的 `matmul_base_analysis.abi_bindings[]` 中 `kernel_abi_contract`/`abi_crosswalk` 与对应 `abi_crosswalk_delta` 行；Wrapper、Kernel entry 或 Launcher 启动绑定还必须引用同一 binding 的 `source_backed_signature_skeleton`。不通过融合动作重新解释基础 MatMul GM 参数。

不得从专题或历史实现补入固定 ratio、slot、bridge 或 adapter。实现发现调用点不一致时回 Step 2/3，不临时改 DESIGN。

### 3.4 Epilogue层动作

只实例化 DESIGN 的唯一首选路线：

- 当前 formula DAG 和每个 broadcast operand 的索引/stride/offset；
- C、operands/intermediates/output 的 dtype、alignment、有效列和转换；
- slot-aware C/staging/output/guard 内存分区；
- DataCopy/VF/LocalTensor API、mask/tail、event 和生命周期；
- 输出写回、运行时拒绝和 host/Tiling 门禁。

MemBase 和 RegBase 不形成两个可执行分支；备选只留在 DESIGN，PLAN 不自动切换。

### 3.5 Tiling、Params、Wrapper 和 Launcher

按以下单向接线形成 action：

```text
project host Tiling
  -> TilingData
  -> Scheduler/Block/Epilogue Params
  -> concrete Kernel type and entry
  -> Wrapper/build target
  -> Launcher buffers and arguments
```

- 先逐字段验证复制后的 Blaze skill Tiling Engine 与当前 Scheduler Params 合同兼容；只有全部语义/单位/合法域一致才复用或最小适配。
- 不能复用但 device 合同已闭合时，为对应 demand partition 实例化返回固定合法控制值的项目 Tiling action。
- 需要重划 partition、新 specialization 或未知合法域时回 Step 2/3；不能让 Step 4 猜值。
- Launcher 只执行设备 Kernel 和输出结果；逻辑输入、物理转换、CPU Golden 和比较由 PLAN 指定的 host/Python 产物负责。
- 不预设固定工程树、脚本数量或文件名；PLAN 的 `target_file_manifest` 提供初始项目文件，实施阶段可在 `<project-root>/operators/<operator_name>/` 内补充实现文件，更新 PLAN 第 2 章并在第 11 章记录实际路径。
- Tiling、Wrapper、Launcher 和 buffer action 必须把额外 operand/输出的物理字节、offset 单位、Params、workspace、grid/usedCore、entry 绑定和设备消费者映射回 `abi_crosswalk_delta`；任一项缺失时回 Step 3，不启动 Kernel。

### 3.6 数据、诊断和验证入口

实例化：

1. 逻辑输入和 deterministic seed；
2. 每个 Tensor 的逻辑到物理转换、padding/packing/broadcast 映射；
3. 与 DESIGN 公式/dtype 顺序一致的 CPU Golden；
4. C-direct-GM、C-through-fusion、V-zero-C、V-known-C、Full 五模式；
5. SplitM、slot、broadcast 轴、alignment/tail 和重复运行的实际激活子集；
6. row/column/known-C 等定位数据、单变量故障注入和记录格式；
7. 诊断开关、额外输出和 Dump 的清理 action。

## 4. 接线和项目文件合同

PLAN 的 `target_file_manifest` 应逐文件记录初始目标的 `create/copy_and_adapt/modify/delete/read_only/forbidden`。至少覆盖当前项目预期需要的：

- Kernel/Wrapper 和可选 custom Block/Kernel/Epilogue；
- TilingData、host Tiling、Params 和构建接线；
- Launcher 及其 ABI/Buffer 接线；
- 逻辑数据、物理转换、Golden 和比较入口；
- 模式分发、诊断、验证和交付产物。

这些是角色，不是固定文件树。PLAN 初始文件必须属于 DESIGN `allowed_change_scope`，并记录为以 `operators/<operator_name>/` 开头的项目相对路径；实施阶段新增文件只能位于当前 `operator_root`，必须避开 DESIGN `forbidden_change_scope`，更新第 2 章并追加第 11 章。根 `ops-tensor`、Skill Asset 原文件、当前 `operator_root` 之外的路径、未注册 custom 层和无关用户文件必须列为 read_only/forbidden；项目内官方副本已存在时列为 `read_only`，缺失时先列为 `create`，完成原样物化和同源核对后转为 `read_only`。

`implementation_wiring_contract` 必须能从最终 operator interface 一路追到每个文件和参数，并显式保留基础 ABI 与场景增量的边界：

```text
logical operand
  -> physical buffer
  -> Launcher argument
  -> Wrapper argument
  -> Kernel GM parameter
  -> Params/Tiling field
  -> Block/Kernel/Epilogue consumer
  -> output buffer and Golden comparison
```

基础 MatMul 行必须引用与其 `design_binding_ref` 对应的 `matmul_base_analysis.abi_bindings[].abi_crosswalk` 和 `crosswalk_row_id`；只有本场景新增/替换的行写入 `abi_crosswalk_delta`，并写明 `delta_crosswalk_row_id`、`base_design_binding_ref` 和 `base_crosswalk_row_refs`。每行都要映射到 action、checkpoint 和 source ref，不能用泛化接线图替代。

## 5. 验证、交付与清理

PLAN checkpoints 至少包括：

1. source/ABI/模板实例化静态核对，包括基础 ABI、场景 delta、物理字节/offset、TilingData、workspace、grid/usedCore、Wrapper/entry 和设备消费者；
2. 分层构建和最小功能运行；
3. 五模式按最早失败域定位；
4. 每个声明支持的 elementwise/broadcast mapping、dtype、shape/alignment/tail；
5. SplitM/slot/reuse/empty/final drain 的条件性用例；
6. required 单变量负/正证据；
7. 非有限值和 DESIGN 阈值门禁；
8. 清理临时诊断后重新构建并执行 Full 回归；
9. 最终 Asset/官方源码只读、交付件和支持边界审计。

`deliverable_manifest` 只列当前项目必需产物。`cleanup_contract` 必须覆盖 known-C/zero/identity 注入、故障开关、额外 Kernel entry/Params、Dump、临时数据、日志和构建产物；清理后回归未通过时不得交付。

## 6. 合规映射和回退

PLAN 的 `scenario_guidance_compliance` 必须把本文每个激活要求映射到：

```text
DESIGN contract ID
reading_manifest ID
ordered_action ID
checkpoint ID
deliverable/cleanup ID
or evidence-backed not_applicable
```

每个激活的 `abi_crosswalk_delta` 行都必须映射到基础 ABI ref、初始 action、checkpoint 和 source ref。未映射 required 设计要求、只写“参见场景指导”或留下待选分支时，PLAN 为 blocking；实现阶段新增的项目文件和修复步骤不要求回写这组映射。

| 发现的问题 | 回退目标 |
|---|---|
| Blaze 源码版本不一致 | Step 1 |
| Blaze 源码类型链、Params/API/物理事实被推翻或缺失 | Step 2，再重做 Step 3 |
| DESIGN 首选、范围、场景合同或设计边界不完整 | Step 3 DESIGN |
| 本文无法表达必要 PLAN 动作/门禁 | Blaze skill 场景维护；修订指导后重新编译 PLAN |
| 实现或验证推翻设计前提 | Step 3；需要新 Blaze 源码事实时先 Step 2 |
| 普通编码、构建或验证错误 | Step 4 在项目根内持续诊断、修复和重跑；只有推翻设计前提才回 Step 3 |

Step 4 的规范性输入只有当前项目的 DESIGN 和 PLAN。它以 PLAN 登记资料为起点，也可以读取本文或其他相关资料诊断和实现，但不能据此重新匹配场景、生成新的设计路线、切换备选或扩大支持范围。
