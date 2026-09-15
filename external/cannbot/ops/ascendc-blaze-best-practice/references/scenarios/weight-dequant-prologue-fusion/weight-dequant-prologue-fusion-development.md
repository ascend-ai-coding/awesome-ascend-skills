# Weight-Dequant Prologue Fusion 开发指导

本文是 Step 3 在场景 DESIGN 冻结后编译项目 `operators/<operator_name>/docs/PLAN.md` 的方法输入，不是 Step 4 的独立设计入口。本文不重新匹配场景、选择 MatMul Blaze 组装方案、选择 Block/Kernel/Prologue 组件、改变支持范围或补充 DESIGN；实施阶段按 `operators/<operator_name>/docs/DESIGN.md`/`operators/<operator_name>/docs/PLAN.md` 实现，并可在 `<project-root>/operators/<operator_name>/` 内为闭合设计补充修复步骤。

**当前支持范围**：本场景偏重 perchannel 量化模式（scale/offset 按 N 维广播），不支持 pergroup 量化。weight dtype 支持 int8 和 fp8（fp8 的 Cast 链路见 [VF 反量化链路专题](prologue-vf-dequant-design.md) §3.1）。

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
selected_scenario: weight-dequant-prologue-fusion
matmul_base_analysis: present
matmul_base_analysis.abi_bindings[]: source_backed_and_closed_for_consumed_partitions
abi_crosswalk_delta: source_backed_and_closed
design_plan_generation_phase: after_design_freeze
```

并且：

- DESIGN 已冻结唯一首选、公式/反量化合同、三层增量（Block层替换 + Kernel层替换 + Prologue层新增）、Tiling/资源/同步、customization scope 和验证合同；`abi_crosswalk_delta` 只追加额外 operand（B/scale/offset/bias）、B_dequant 中间值、Prologue Params、Wrapper 或同步接线，且每行具有 `delta_crosswalk_row_id` 并引用 `base_design_binding_ref` 和基础 MatMul `crosswalk_row_id`；
- DESIGN 的 `dependency_skills` 与 `dependency_skill_evidence` 已冻结，覆盖当前首选路线的根入口、必要叶子及 API/路线证据；
- 根 `ops-tensor` 和 Blaze skill Asset 原文件始终只读；项目内官方副本只允许执行 PLAN 登记的缺失目录 create-only 原样物化，物化后只读；
- 无决定性 unknown、TBD、未选择分支、Grouped 基础分区缺口或要求 Step 4 决策的字段。

失败时回 Step 3，不生成可执行 PLAN；不得把本文默认值或历史实现写入可执行 action。

## 2. 阅读与依赖清单

Step 3 根据首选 DESIGN 生成 PLAN `reading_manifest` 作为初始阅读基线。初始资料应绑定 `read_before_action_ids` 和 action `source_refs`；未激活资料写 N/A 或不登记，Step 4 可按实际实现和错误证据读取必要资料，但不能据此改变设计。

| 设计合同 | PLAN 必读来源 | 条件 |
|---|---|---|
| Block层 delta | [V+C 同步与 L1 布局专题](vc-sync-and-l1-layout-design.md)、Investigation 指定的 concrete source locations | 使用/扩展 BlockMmad 特化 |
| Kernel层 delta | [V+C 同步与 L1 布局专题](vc-sync-and-l1-layout-design.md)、实际 mixed Kernel/同步定义 | 存在 AIC/AIV 协作和 CV 同步 |
| Prologue层 delta | [VF 反量化链路专题](prologue-vf-dequant-design.md)、[Prologue UB 布局专题](prologue-ub-layout-design.md) | 使用 VF 反量化 prologue |
| Cast API | DESIGN `dependency_skills` 记录的 ascendc-api-best-practices 根入口和必要叶子 | 使用 VF Cast API |
| RegBase | DESIGN `dependency_skills` 记录的 ascendc-regbase-best-practice 根入口和必要叶子 | 使用 RegBase VF 反量化 |
| Tiling | [Tiling 参数合同专题](weight-quant-tiling-contract.md)、[Tiling 方法](../../kernel-design/tiling-selection.md) | 按 DESIGN 选择所需章节 |
| Launcher | [Launcher 方法](../../launcher/launcher-development.md) | 按 DESIGN 选择所需章节 |
| Event/pipe | [同步方法](../../fundamentals/blaze-sync-patterns.md) 和当前 CANN 实际头文件 | DESIGN 激活相关同步 |
| Layout 边界 | [Layout 支持范围专题](layout-variant-boundary.md) | 确认支持域和拒绝域 |
| 诊断 | [精度诊断专题](pitfalls-and-diagnosis.md) | 所有本场景实现 |

PLAN 先登记 DESIGN `dependency_skills` 中的根入口，再登记当前首选路线需要的叶子。`reading_manifest` 只登记当前首选路线的初始资料，不能把全目录一股脑列入。

## 3. 有序动作规则

Step 3 将下列稳定动作类别实例化为 PLAN 的 `ordered_actions`。每个初始动作必须含 DESIGN refs、source refs、计划目标文件、前置、产物、checkpoint 和 failure rollback。

### 3.1 来源与项目副本

1. 核对 DESIGN/Investigation 与当前 Blaze 源码版本的抽象一致性。
2. 对 DESIGN 选择的官方组件（Scheduler、A 侧搬运路径等 preserved 合同）登记 read-only 来源；满足合同时直接引用，不创建 custom 副本。
3. 只有 `customization_scope` 明确授权时，登记从具体来源复制到项目目标文件并适配的 action，记录首个修改点和必须保持的不变量。
4. 只有 DESIGN 明确授权结构起点时，才可复制以下 Asset 到项目文件；Asset 原文件只读，复制后必须按当前 ABI/公式/API 重写和验证：
   - `assets/blaze_custom/kernel/weight_quant_matmul_kernel.h`（场景核心：V+C mixed Kernel）
   - `assets/blaze_custom/block/weight_quant_matmul_block_mmad.h`（场景核心：B 从 L1 读取的特化 BlockMmad）
   - `assets/blaze_custom/policy/dispatch_policy.h`（场景核心：WeightQuantMatmulPolicy 定义）
   - `assets/op_tiling/weight_quant/weight_quant_tiling_data.h`
   - `assets/op_tiling/weight_quant/weight_quant_tiling.h`
5. 以下 Blaze 组件由 asset 代码直接 include 引用，不需复制到项目：
   - `blaze/gemm/utils/common_utils.h`（MNK_M/N/K/B、CeilDiv、CeilAlign、FINAL_ACCUMULATION 等）
   - `blaze/gemm/utils/layout_utils.h`（IsTrans、IsWeightNz）
   - `blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h`（SWAT Scheduler）
6. 不复制整个 Blaze，不通过 include 顺序覆盖官方 specialization，不创建项目根之外或 DESIGN forbidden scope 内的目录。

### 3.2 Block层动作

按 DESIGN 冻结的 Block delta 实例化：

- `WeightQuantMatmulPolicy<NO_FULL_LOAD_MODE>` 特化绑定和 SFINAE 约束；
- B 从 L1 读取（非 GM）的修改点：跳过 GM→L1 B 搬运，BlockMmad 从 L1Params 提供的 B_dequant 地址读取；
- L1 布局：lower half [B0|A0], upper half [B1|A1], bias at tail 64-byte aligned；
- BiasType 一致性核对（GM/L1/BT/Tiling 四处）；
- 分形轴对齐约束校验；
- 官方 Scheduler（`MatmulSwatScheduler`）的直接引用和绑定；
- 结构/ABI 静态检查及必要的单变量正负对照。

未授权 custom Block 时只登记官方组件绑定和核对 action，不生成修改动作。

### 3.3 Kernel层动作

按 concrete witness 实例化：

- mixed Kernel entry（`__global__ __aicore__ __mix__(1,2)`）、AIC/AIV 角色和 ratio；
- AIC 侧：BlockMmad + Scheduler 初始化、K 循环和 L0C→GM 输出；
- AIV 侧：Prologue 初始化、UB 管理、VF 反量化、L1 写入；
- CV 同步：flag 预置（首轮）、K-loop 内 wait/set 交替、final drain 消费剩余；
- Block/Prologue、Params/TilingData、Wrapper 的类型接线；
- DESIGN 已证明必要的同步事件配对及负向对照。

所有涉及新增 operand（B/scale/offset/bias）、B_dequant 中间值、Prologue Params、Wrapper、Kernel entry 或同步的动作必须引用由 `base_design_binding_ref` 选定的 `matmul_base_analysis.abi_bindings[]` 中 `kernel_abi_contract`/`abi_crosswalk` 与对应 `abi_crosswalk_delta` 行；Wrapper、Kernel entry 或 Launcher 启动绑定还必须引用同一 binding 的 `source_backed_signature_skeleton`。不通过融合动作重新解释基础 MatMul 的 A 侧 GM 参数。

不得从专题或历史实现补入固定 ratio、flag ID 或 bridge。实现发现调用点不一致时回 Step 2/3，不临时改 DESIGN。

### 3.4 Prologue层动作

只实例化 DESIGN 的唯一首选路线：

- B 输入 dtype → B_dequant 输出 dtype 的 Cast 链（按 BType 分支选择，见 [VF 反量化链路专题](prologue-vf-dequant-design.md) §3 已验证链路表和 §3.1 实践参考）；
- 每步 Cast 的 CastTrait、LoadDist、StoreDist 配置从 asc-devkit API 文档确认并经编译+精度验证；
- `static constexpr` 声明形式与源码一致；
- UB ping-pong 布局：[bIn|bOut|scale|offset] × 2 half 的字节范围和对齐；
- UB 空间约束公式验证；
- scale/offset 加载时机（首轮 K-L1 迭代加载，后续复用）；
- hasOffset 的 `if constexpr` 编译期分支；
- Prologue K-L1 循环与 AIC K 循环的对应关系；
- CV 同步的 AIV 侧 wait/set 交替。

Prologue 层不形成可切换的备选分支；备选只留在 DESIGN，PLAN 不自动切换。

建议将 Prologue 的 `operator()` K 循环体拆分为三个无返回值、不含同步的纯数据操作方法，同步逻辑全部写在 `operator()` 中显式排列：

- **`LoadGM2UB`** — GM→UB 搬运 B/scale/offset，通过出参返回 UB 地址
- **`DequantVF`** — 调用 `asc_vf_call` 执行 VF 反量化，通过出参返回 bOut 地址
- **`StoreUB2L1`** — UB→L1 搬运（使用 `copy_ubuf_to_cbuf` 显式指定 srcGap=1 解除 bank 冲突）

`operator()` 只负责在三个阶段之间排列 `WaitFlag`/`SetFlag`/`CrossCoreWait`/`CrossCoreSet` 同步，使数据流和同步逻辑一目了然。

### 3.5 Tiling、Params、Wrapper 和 Launcher

按以下单向接线形成 action：

```text
project host Tiling
  -> TilingData
  -> Scheduler/Block/Prologue Params
  -> concrete Kernel type and entry
  -> Wrapper/build target
  -> Launcher buffers and arguments
```

- 先逐字段验证复制后的 Blaze skill Tiling Engine 与当前 Scheduler/Block/Prologue Params 合同兼容；只有全部语义/单位/合法域一致才复用或最小适配。
- 将 Tiling Engine 的 `elemBytesB()`/`elemBytesA()`/`elemBytesDequantB()` 从 `static constexpr` 改为 Args 成员字段，由 `GetTilingData` 新增参数 `weightElemBytes` 和 `dequantBElemBytes` 传入。
- 将 `CalL1AndUbTiling()` 中的魔数 `6UL` 替换为 `NUM_TWO * (args_.weightElemBytes + args_.dequantBElemBytes)`；逐字段验证参数化后的 Tiling Engine 与各 dtype 的 UB 空间约束公式一致。
- 不能复用但 device 合同已闭合时，为对应 demand partition 实例化返回固定合法控制值的项目 Tiling action。
- 需要重划 partition、新 specialization 或未知合法域时回 Step 2/3；不能让 Step 4 猜值。
- Launcher 只执行设备 Kernel 和输出结果；逻辑输入、物理转换、CPU Golden 和比较由 PLAN 指定的 host/Python 产物负责。
- 不预设固定工程树、脚本数量或文件名；PLAN 的 `target_file_manifest` 提供初始项目文件，实施阶段可在 `<project-root>/operators/<operator_name>/` 内补充实现文件，更新 PLAN 第 2 章并在第 11 章记录实际路径。
- Tiling、Wrapper、Launcher 和 buffer action 必须把额外 operand（B/scale/offset/bias）/B_dequant 中间值的物理字节、offset 单位、Params、workspace、grid/usedCore、entry 绑定和设备消费者映射回 `abi_crosswalk_delta`；任一项缺失时回 Step 3，不启动 Kernel。

### 3.6 数据、诊断和验证入口

实例化：

1. 逻辑输入和 deterministic seed；
2. 每个 Tensor 的逻辑到物理转换、padding/packing/broadcast 映射；
3. 与 DESIGN 公式/dtype 顺序一致的 CPU Golden（bf16 精度运算）；
4. A-only-MMAD、Dequant-only、Full 三模式；
5. transB、hasOffset、hasBias 的实际激活子集；
6. int8 边界值（-128, 127）、alignment/tail 和重复运行；
7. 诊断开关、额外输出和 Dump 的清理 action。

## 4. 接线和项目文件合同

PLAN 的 `target_file_manifest` 应逐文件记录初始目标的 `create/copy_and_adapt/modify/delete/read_only/forbidden`。至少覆盖当前项目预期需要的：

- Kernel/Wrapper 和可选 custom Block/Kernel/Prologue；
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
  -> Block/Kernel/Prologue consumer
  -> output buffer and Golden comparison
```

基础 MatMul 行必须引用与其 `design_binding_ref` 对应的 `matmul_base_analysis.abi_bindings[].abi_crosswalk` 和 `crosswalk_row_id`；只有本场景新增/替换的行写入 `abi_crosswalk_delta`，并写明 `delta_crosswalk_row_id`、`base_design_binding_ref` 和 `base_crosswalk_row_refs`。每行都要映射到 action、checkpoint 和 source ref，不能用泛化接线图替代。

## 5. 验证、交付与清理

PLAN checkpoints 至少包括：

1. source/ABI/模板实例化静态核对，包括基础 ABI、场景 delta、物理字节/offset、TilingData、workspace、grid/usedCore、Wrapper/entry 和设备消费者；
2. 分层构建和最小功能运行；
3. 三模式（A-only-MMAD、Dequant-only、Full）按最早失败域定位；
4. 每个声明支持的 transB/hasOffset/hasBias 组合、dtype、shape/alignment/tail；
5. CV 同步的 flag 预置/交替/final drain 用例；
6. required 单变量负/正证据；
7. 非有限值和 DESIGN 阈值门禁；
8. 清理临时诊断后重新构建并执行 Full 回归；
9. 最终 Asset/官方源码只读、交付件和支持边界审计。

`deliverable_manifest` 只列当前项目必需产物。`cleanup_contract` 必须覆盖 Dequant-only/A-only 诊断注入、故障开关、额外 Kernel entry/Params、Dump、临时数据、日志和构建产物；清理后回归未通过时不得交付。

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

每个激活的 `abi_crosswalk_delta` 行都必须映射到基础 ABI ref、初始 action、checkpoint 和 source ref。未映射 required 设计要求、只写"参见场景指导"或留下待选分支时，PLAN 为 blocking；实现阶段新增的项目文件和修复步骤不要求回写这组映射。

| 发现的问题 | 回退目标 |
|---|---|
| Blaze 源码版本不一致 | Step 1 |
| Blaze 源码类型链、Params/API/物理事实被推翻或缺失 | Step 2，再重做 Step 3 |
| DESIGN 首选、范围、场景合同或设计边界不完整 | Step 3 DESIGN |
| 本文无法表达必要 PLAN 动作/门禁 | Blaze skill 场景维护；修订指导后重新编译 PLAN |
| 实现或验证推翻设计前提 | Step 3；需要新 Blaze 源码事实时先 Step 2 |
| 普通编码、构建或验证错误 | Step 4 在项目根内持续诊断、修复和重跑；只有推翻设计前提才回 Step 3 |

Step 4 的规范性输入只有当前项目的 DESIGN 和 PLAN。它以 PLAN 登记资料为起点，也可以读取本文或其他相关资料诊断和实现，但不能据此重新匹配场景、生成新的设计路线、切换备选或扩大支持范围。
