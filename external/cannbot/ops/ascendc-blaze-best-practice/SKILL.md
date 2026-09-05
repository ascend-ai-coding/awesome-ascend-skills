---
name: external-cannbot-ops-ascendc-blaze-best-practice
description: 在 Ascend 950 / DAV_3510 平台上，基于 Blaze/tensor_api 开发 Basic、Batch、Grouped、Quantized、MX
  等 MatMul 类算子或相关的融合算子时使用。不适用于纯 Vector 算子或 A2/A3 平台。
original-name: ascendc-blaze-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# Blaze MatMul 算子开发指南

## 路径与文档合同

将 `<project-root>` 解释为最高层项目根，将 `<operator_name>` 解释为 `operators/` 下的算子目录名：

```text
project_root: <project-root>
operators_root: <project-root>/operators/
operator_root: <project-root>/operators/<operator_name>/
blaze_source_root: <project-root>/ops-tensor/
investigation_report: operators/<operator_name>/docs/blaze/blaze-investigation-report.md
design_path: operators/<operator_name>/docs/DESIGN.md
plan_path: operators/<operator_name>/docs/PLAN.md
```

只把 `<project-root>/ops-tensor/` 作为 Blaze 源码事实根；它与 `<project-root>/operators/` 同级。不得回退读取算子目录内、其他项目或历史工程的 ops-tensor。`project_contract_id` 是逻辑合同 ID，不是路径变量。

统一使用以下模板；本文中的 DESIGN、PLAN 分别指其生成的 `DESIGN.md`、`PLAN.md`：

- [Blaze DESIGN 模板](references/kernel-design/blaze-design-template.md)
- [Blaze PLAN 模板](references/kernel-design/blaze-plan-template.md)

Blaze skill 不读取或依赖 `environment.md`、外部 manifest 或其他工作流产物。调用方可以直接传入已经确认的 `target_chip`、`npu_arch` 和可选 `cann_version`。

`assets/` 是本 Skill 的版本化可复用资产来源：普通算子 Step 4 将其视为只读
结构起点并复制到项目内适配；只有明确的 Skill 维护计划才可直接修复资产原文件，
且必须以当前 SDK/target 的编译回归闭合。不要把两种写入边界混为一谈。

## 按请求目的路由

| 请求目的 | 执行 |
|---|---|
| 明确要求“开发算子”或等价的从零完整开发 | Step 1 → Step 2 → Step 3 → Step 4 |
| 明确要求算子设计、方案分析并输出设计文档 | Step 2 → Step 3 |
| 咨询、解释、评审、排障、能力查询 | 只读取相关 references |

不要按调用者身份增加模式。不要因发现已有 DESIGN/PLAN 而单独进入 Step 4；direct invoke Developer 直接消费相同 DESIGN/PLAN，Blaze Step 4 只属于本 skill 的完整四步开发流程。

## 计算执行原则

本 skill 开发的算子，其全部计算步骤必须在 device 侧单一 Kernel 中完成。不得将算子需求中的任何计算步骤放到 host 侧执行。host 侧只负责数据准备、Tiling 计算、Kernel launch 和结果搬运。

此原则适用于全部路线（`blaze_native`、`blaze_custom`）和全部场景。Step 3 在判定 Blaze 官方方案覆盖性时必须以完整算子语义为对象，不得通过将非 matmul 计算步骤排除到 host 侧来缩小 `native_gaps` 的判定范围。

## 四步流程

### Step 1: Project Setup

创建 `operator_root`，在 `blaze_source_root` clone 或更新授权 ops-tensor，递归初始化 submodule，确认抽象版本一致性，并建立根源码、项目 Blaze 副本、项目 tensor_api 副本三个只读区。不要预先实现公式、Tiling、Golden、固定工程或场景 recipe。

→ [Step 1: Project Setup](references/workflow/step1-project-setup.md)

### Step 2: Blaze Investigation

从 `project_root` 和 `operator_name` 派生路径并只读自检当前 `blaze_source_root`；不要依赖 Step 1 文件产物，不要 clone、更新、切换源码或读取场景资料。按需求语义调查候选 Blaze 组装方案、物理数据和 ABI 事实，只生成 Investigation。

→ [Step 2: Blaze Investigation](references/workflow/step2-blaze-investigation.md)

### Step 3: Kernel Design

依据需求和 Investigation 完成逻辑接口、Blaze 官方方案、必要的唯一 custom 场景、最终 ABI/资源/验证合同，并用统一模板生成 DESIGN 和可执行路线的 PLAN。`unsupported` 只生成 DESIGN；一次补充调查后仍缺决定性事实时不生成最终 DESIGN/PLAN。

→ [Step 3: Kernel Design](references/workflow/step3-kernel-design.md)

### Step 4: Implementation

只在完整四步流程中执行当前 `operator_root` 的 DESIGN/PLAN。核对联合门禁，按冻结的第 9、10 章执行；持续更新 PLAN 第 2、4--8 章并只追加第 11 章。不得重新匹配场景、选择路线/候选、改变接口/ABI、切换备选或扩大支持域。

→ [Step 4: Implementation](references/workflow/step4-operator-development.md)

## 通用排障参考

通用构建、头文件解析、链接、项目边界和 Blaze 源码工作区问题集中在 [`references/troubleshooting/`](references/troubleshooting/)。Step 4 按各文档的适用范围和触发信号读取相关资料；这些资料只用于预检和诊断，不能替代冻结的 DESIGN/PLAN 或新增实现动作。

ASC 联合编译中的 RegBase mask 可变性、`__VEC_SCOPE__` induction 类型、设备地址
空间和 Host/device helper 上下文，统一按
[`compile-troubleshooting.md`](references/troubleshooting/compile-troubleshooting.md)
的对应小节核对；它们是可复用的编译边界门禁，不是某个算子的固定实现。

首次构建还必须先完成该文档的目标/宏/include/link/API preflight；真实启动绑定与
launch/verifier 证据隔离按
[`launcher-development.md`](references/launcher/launcher-development.md)执行。

GMM 的 cumsum/count、逐组 Golden 写回和 grouped `GetTilingData` owner 读取
[`group-matmul-delta.md`](references/kernel-design/group-matmul-delta.md)；GLU 的
act/gate、无 scale 参数和公式改写授权读取
[`glu-development.md`](references/kernel-design/glu-development.md)。Golden
初始化超时、backend autoload 和逻辑到物理输入的边界读取
[`blaze-matmul-layout.md`](references/fundamentals/blaze-matmul-layout.md)，不要
把某次环境变量、临时路径或具体命令写成 Skill ABI 常量。

## 问题、恢复与证据记录

每个开发问题都必须保留“原始问题”而不只保留最终经验。项目的
`WALKTHROUGH.md`、PLAN `execution_record` 或等价交付记录至少逐项写明：

```text
issue_id
original_problem              # 用户合同或首次可观察现象
observed_signal_and_command   # 错误、首错位置、返回码和执行上下文
first_failed_boundary         # 编译、ABI、布局、同步、设备精度或性能
classification: actual_failure | preflight_risk | environment_blocked
root_cause_and_evidence
fix_or_recovery
positive_regression
remaining_boundary
skill_promotion: reusable | operator_local | none
```

`preflight_risk` 不能写成已经发生的设备失败；编译/静态通过不能写成
`device_verified`。设备结论必须标注 `sandbox` 或 `device_visible`、设备节点、
架构、实际命令和返回码；没有同合同 baseline 时性能写 `NOT_EVALUATED`。
具体 tile、flag、局部命名、binary hash 和单机告警保留在项目记录，不上升为
Skill 常量。

记录顺序必须是“原始问题 → 证据 → 根因 → 修复 → 回归”：`original_problem`
保留用户合同或第一次可观察的原始症状，不要从修复后的状态倒推；
`observed_signal_and_command` 保留首个错误、命令、上下文和返回码；
`first_failed_boundary` 只指第一个失败层，后续连锁错误另列为观察结果。没有
实际触发的设计风险写 `preflight_risk`，环境/权限/网络导致尚未进入目标层写
`environment_blocked`，不能把二者冒充实现失败。`positive_regression` 必须用
同一复现命令并覆盖受影响的 required cases；若仍有未覆盖边界，明确写在
`remaining_boundary`。

## 可复用的精度与设备完成门禁

以下门禁适用于所有 MatMul 及其融合变体；它们是完成条件，不是某个算子的
tile、Scale、padding 或 workaround：

1. **以设备输入事实源计算 Golden。** 对冻结合同中把已编码/量化 value 或
   scale 作为设备输入的变体（包括 MX/FP8/E8M0），先读取最终送入设备的输入
   字节（通常由 `data/input/` 产生），按合同解码、还原有效范围，再执行 CPU
   公式。量化前的 FP32 只可用于生成测试输入，不能作为最终 Golden；scale/data
   的 tail padding 也不能参与结果。普通非序列化 MatMul 仍按逻辑输入计算
   Golden。
2. **冻结可执行 Golden dtype 链。** DESIGN/PLAN 的
   `cpu_golden_formula_and_dtype_order` 必须逐项写明输入 dtype、累加 dtype、
   转换顺序、公式顺序、舍入、clamp/saturation 和非有限值处理；需求正文、
   接口合同与 Golden 表达不一致时标记 `conflict` 或 `blocking`，先解决权威
   来源再实现，不能用“Golden 通过”覆盖合同冲突。执行该合同所需的 Golden
   后端或依赖不可用时也必须保持 `blocking`；不得静默退回累加顺序、dtype、
   舍入或公式不同的 serial/fallback 实现。只有已证明与冻结合同等价的后端
   才能作为 fallback，并且必须在记录中标明等价性证据。参见
   [`blaze-design-template.md`](references/kernel-design/blaze-design-template.md)。
   主机 Golden 生成器出现 overflow、underflow 或非有限值告警时，先记录告警
   对应的输入范围和公式语义，再决定是否只抑制已证明不影响结果的日志；不得
   以“消除 warning”为由改变冻结公式、dtype 顺序或饱和语义。无法证明时保持
   `blocking`，并把告警与设备精度失败分开归类。
   当设备 BlockMmad 的累加 dtype 与 Golden 的累加 dtype 不同（例如设备使用
   int32 dot-product、Golden 使用 FP32 reduction）时，必须把两者分别写入合同，
   并提供声明支持域内的等价性证据；量化后的输出恰好一致不能反向证明中间
   累加等价。若合同改为设备累加语义，必须重新生成 Golden、输入和证据。
3. **分开声明验证层级。** 编译、静态检查、Skill 校验和 CPU Golden 只能证明
   对应层级；`device_verified` 必须来自真实设备路径，并在记录中保留
   `sandbox/device_visible`、设备节点、SoC/架构、完整命令、返回码和输出。
4. **覆盖会改变失败边界的形状。** 在需求合同允许的支持域内，精度门禁至少
   包含单行与多行、M 奇偶、K/N 非对齐与 tail、跨 tile，以及连续重复 launch；
   首行、单 tile 或单次通过不能外推到完整支持域。每次失败先定位首个边界，再
   决定修复范围。验证器必须在合同中明确 `mismatch_count` 的语义：它是
   bitwise exact 差异计数，还是超过 MERE/绝对误差门的元素计数；两者不得使用
   同一个字段混淆。若正式合同是 MERE/MARE，bitwise 差异只能作为独立诊断字段，
   不能把 1 ULP 差异自动写成精度失败。
   重复性首先在同一个 ACL context 内连续 launch 并比较输出，再单独测试多进程
   生命周期；后者出现挂起或空日志时，不得回写为 Kernel slot/sync 失败。
5. **闭合任务类型、同步和生命周期。** DESIGN/PLAN 必须从同一 source-backed
   witness 冻结 entry/task type（AIC、AIV 或 MIX）及其 ABI。异步 GM→UB
   搬运在 Vector 读取前必须有生产者到消费者的等待；清零、填充和 VF store
   也属于 UB slot 的写者，必须纳入同一生命周期。循环复用还必须有消费者到
   下一次覆盖的反向依赖，并为尾轮完成 drain。BlockMmad 与有状态 Epilogue
   在 Kernel 编排层同级持有；对象析构只负责词法生命周期内的本地 HardEvent，
   不替代 Kernel 的跨核释放、阶段 SyncAll 或最终 drain。所有 device/workspace
   分配必须在 `aclrtResetDevice`/`aclFinalize` 前释放。
   生产 Kernel 不得用只调用 `BlockMmad` 的手写 probe 替代 source-backed
   `GemmUniversal`/dispatch/scheduler/ownership 组装；这种 probe 只能作为隔离
   诊断。MIX 的每个逻辑 tile/输出行还必须有可证明的唯一 writer；物理
   `GetBlockIdx()`、task ratio、`GetSubBlockIdx()` 和 `GetBlockNum()` 的语义必须
   由当前源码或最小 probe 冻结，不能按命名或历史实现猜测。
6. **有界执行并隔离设备挂起。** 矩阵 runner 必须为每个设备 case 设置有限
   timeout，并保留已完成 case 的输入、输出和验证记录；某个 case 超时或挂起时，
   记录首个挂起 case、设备可见命令、进程等待状态和返回码，先终止/清理子进程，
   再在隔离设备上下文中单独重跑。单 case 重跑通过不能抹掉原始挂起；在所有
   required cases 完成前，整体结果保持 `blocked` 或 `environment_blocked`，不能
   汇总为 `device_verified`/`PASS`。具体 timeout 数值由项目验证器决定，不写成
   Skill 常量。
7. **闭合逻辑到物理 buffer 的转换。** 当 Host/data producer 将逻辑 Tensor
   物化为另一种 physical representation（例如 paired、pack、transpose、
   alignment 或 padding）时，必须在同一 `abi_crosswalk` 中记录转换、物理 shape、
   byte span、offset/stride 和首个设备消费者。若 physical buffer 可以预先物化，
   数据生成器应从逻辑输入独立生成 physical witness；Launcher 再按冻结转换重新
   生成待 H2D buffer，并在启动 Kernel 前执行 shape、size 和逐字节一致性检查。任一
   不一致都在 Host ABI/I/O 边界阻断，不能把预生成的 physical 文件直接当作逻辑
   合同，也不能进入设备后才依赖精度失败发现布局错误。Golden 必须使用最终实际
   H2D 的 physical bytes；没有逻辑到物理转换时，逻辑 bytes 本身就是该事实源，
   不额外伪造 witness。具体 pack 公式、tile、padding 值和 hash 保留在项目记录。
8. **区分公开输出与诊断中间量。** Verifier 必须分别给出用户合同输出和
   workspace/中间阶段的状态；公开输出 PASS 不能抹掉中间量的 diagnostic FAIL，
   中间量 diagnostic FAIL 也不能在 DESIGN 明确其非公开且不属于验收合同的情况下
   被改写成公开输出 FAIL。两种状态都要保留原始输入、比较顺序和数值差异，不能用
   容差或删除诊断字段掩盖未闭合的内部边界。

Golden 的序列化输入例外、同步事件配对和 Launcher 资源顺序分别见
[`blaze-matmul-layout.md`](references/fundamentals/blaze-matmul-layout.md)、
[`blaze-sync-patterns.md`](references/fundamentals/blaze-sync-patterns.md) 和
[`launcher-development.md`](references/launcher/launcher-development.md)。
MatMul 后 elementwise/broadcast 的公式稳定性、`workspace -> yScale -> y` 分层
和非有限值诊断见
[`elementwise-broadcast-epilogue-fusion/precision-diagnosis.md`](references/scenarios/elementwise-broadcast-epilogue-fusion/precision-diagnosis.md)。
GLU/SwiGLU 的 C-direct/V-known 分层定位、指标门禁和局部精度补偿边界见
[`glu-precision-diagnosis.md`](references/kernel-design/glu-precision-diagnosis.md)。

## 路线模型

```text
implementation_route: blaze_native | blaze_custom | unsupported
selected_scenario: <仅 blaze_custom 填写>
unsupported_points: <仅 unsupported 填写>
```

只在 Step 3 决定项目路线。官方 Blaze 覆盖全部 required partitions 时选择 `blaze_native` 且不读取场景注册表；存在证据闭合的 native gap 且场景唯一命中时选择 `blaze_custom`；证据充分且场景零命中或多命中时选择 `unsupported`。

### 路线与 ABI 的负例边界

一个实际的 GMM 融合问题暴露了两种容易混淆的“缺失”：pinned source 可能分别有
GMM 的某种量化/AIC 组件和 QBMM 的 A8W8/AIV 组件，却没有把它们闭合为用户所需的
GMM A8W8 + unary + full-row quant。这个 native gap 不是 SIMT fallback 的授权，
也不是自动的 `unsupported`；应分别记录 native candidate 的 source rejection、
唯一命中的 Scenario、custom delta 的 ABI/同步补足和后续设备证据。只有“native
覆盖完整”才能选 `blaze_native`；“native gap + exactly-one Scenario + custom
ABI/资源/同步可闭合”才选 `blaze_custom`；零命中或多命中才保持 `unsupported`。

如果 native GMM/QGMM POD 无法表达目标所需的 fp32 scale、workspace、`yScale` 或
group metadata，不得修改只读 `ops-tensor`，也不得把 MX/GLU 字段改名后继续复用。
应在项目侧定义唯一 POD/loader，并在 `abi_crosswalk` 中逐字段写清地址单位、stride、
容量、首个设备消费者和同步 owner；字段消费或设备 witness 未闭合前，保持
`blocked`/`NOT_EVALUATED`，不能用编译通过或同 stream fallback 代替。

## 扩展场景适配指导

仅在维护本 skill 的扩展能力时读取[场景接入指导](references/scenarios/scenario-extension-guide.md)。在 `references/scenarios/<scenario-id>/` 内维护场景设计和开发资料，并完成索引注册、源码前提、DESIGN/PLAN 合同和唯一匹配约束；普通算子流程不自动读取其他场景。
