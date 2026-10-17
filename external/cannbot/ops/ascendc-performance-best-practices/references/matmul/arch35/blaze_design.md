# matmul Blaze 路线评估（实现范式路由）

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

本文是 `matmul` 族的**实现范式路由条目**：只定义何时评估 Blaze、如何判定路线、需要哪些证据；Blaze 的调查、设计与实施方法完整位于 [ascendc-blaze-best-practice](../../../../ascendc-blaze-best-practice/SKILL.md)，不在本文复述。命中 Blaze 路线后必须跳转该 Skill 执行，不得凭记忆或本文片段自行组装 Blaze 代码。

## 1. 适用条件

同时满足以下两项才进入 Blaze 路线评估：

| 条件 | 判定 |
|------|------|
| 目标架构 | 目标 SoC 含 Ascend950 / DAV_3510（A5）；A2/A3 平台不适用，直接记录后跳过 |
| 算子族 | `matmul` 族所有变体：`matmul`、`batch_matmul`、`group_matmul` 及 quantized / MX / FP8 / A8W8、matmul+epilogue 融合（如 GLU/SwiGLU、per-token-quant）等 |

**不在本路由范围**：纯 Vector、Reduction、Gather/Scatter 等非 matmul 族；conv 族（含含 pointwise/1x1 子计算的融合算子）当前无 Blaze 场景，走 conv 族自身路径。

## 2. 与 AscendC 策略的选型顺序

Blaze 路线评估**先于**上表 AscendC 策略（pingpong/swat/streamk/fullload/scale_coalescing/mte2_preload）选型完成，但**不替代**它们——两者并列作为候选方案：

1. 先完成 Blaze 路线判定（第 3 节），结论与证据落盘；
2. bound 证据显示 AscendC 手段仍有明确空间（如未开 pingpong、尾碎片可切分、假 MTE2 bound）时，优先局部 AscendC 优化，改动小、风险低；Blaze 方案作对照候选；
3. 手写 AscendC 已到上限（深层流水/MEM 瓶颈、局部手段收益有限有逐 case 证据），或量化/epilogue 融合形态（Blaze 有现成场景资产）时，Blaze 路线作为高价值候选；
4. 评估不可跳过：不得因源码未用 Blaze、无 ops-tensor 环境而直接跳过判定；环境不可得时按第 5 节降级记录。

## 3. 路线判定

```text
implementation_route: blaze_native     # 官方 Blaze 组件覆盖全部算子语义
                   | blaze_custom     # 存在 native gap 且场景唯一命中 blaze_custom 注册表
                   | blaze_migration  # 输入为 ops-nn/ops-transformer 注册仓的 AscendC/CMCT/CGMCT 核函数
                   | unsupported      # 场景零命中或多命中，或环境不可得
```

| 路线 | 判定依据 | 执行入口 |
|------|---------|---------|
| `blaze_native` | 官方 GemmUniversal/dispatch/scheduler 组件经源码调查覆盖全部 required partitions | [ascendc-blaze-best-practice](../../../../ascendc-blaze-best-practice/SKILL.md) Step 2/3 |
| `blaze_custom` | native gap 有证据闭合，且 [场景注册表](../../../../ascendc-blaze-best-practice/references/scenarios/index.md) 唯一命中；多命中或零命中不得强选 | 同上 + 对应场景 design/development 文档 |
| `blaze_migration` | 待优化核函数来自注册仓（AscendC/CMCT/CGMCT），等价迁移不改 host tiling/ABI/数学语义/支持域 | [ascendc-blaze-migration](../../../../ascendc-blaze-migration/SKILL.md) |
| `unsupported` | 场景零命中或多命中；或 ops-tensor 不可获取 | 留 `unsupported_points` 证据后回退 AscendC 策略选型 |

`unsupported` 必须逐项记录 `unsupported_points`（缺失的官方组件能力、未命中的场景、环境 blocker），保证判定可审计、可复查，不允许无证据静默跳过。

## 4. 事实源纪律

- `<project-root>/ops-tensor/` 是 Blaze 源码唯一事实根，与算子目录同级；不得回退读取算子目录内、其他项目或历史工程的 ops-tensor；
- ops-tensor **只读**：不得修改其任何文件；候选需要的公共组件按 blaze skill 的路线模型处理；
- ops-tensor 的 commit SHA（含 submodule 递归初始化状态）必须进入证据身份，与基线源码/二进制身份同级，保证调查结论可复现；
- 布局转换（如 NCHW/NHWC → GEMM operand、pack/transpose/padding）必须走 Blaze 门禁的 `abi_crosswalk` 证据：物理 shape、byte span、offset/stride、首个设备消费者和同步 owner。

## 5. 环境缺失降级

ops-tensor 无法获取（无网络/权限/授权仓不可达）时：

- 路线判定记 `unsupported`，缺失原因记 `environment_blocked`；
- 不阻断同轮 AscendC 候选方案的制定与实施；
- 后续轮次环境恢复时可重新评估，重新评估须基于当轮基线证据。

## 6. 证据与记录要求

Blaze 路线判定的结论进入逐 case 分析与方案记录：

- 每个 matmul case 记录：`implementation_route` 结论、判定依据（调查结论/场景命中/迁移边界）、ops-tensor commit；
- Blaze 候选方案的 required features 必须包含：路线约束（不得换路线）、ABI crosswalk（涉及布局转换时）、launcher 证据、Golden dtype 链（量化变体）；
- `blaze_custom` 依赖新增 ops-tensor 公共组件时，方案须标注该依赖在回集成阶段需要人工决策（单仓集成 or 双仓 PR），不得静默降级。

## 7. 执行入口汇总

| 需要 | 读取 |
|------|------|
| Blaze 框架/组件/同步/构建基础 | [fundamentals/](../../../../ascendc-blaze-best-practice/references/fundamentals/blaze-framework-overview.md) |
| 路线调查方法（native/custom 判定） | [step2-blaze-investigation](../../../../ascendc-blaze-best-practice/references/workflow/step2-blaze-investigation.md) |
| blaze_custom 场景注册表 | [scenarios/index.md](../../../../ascendc-blaze-best-practice/references/scenarios/index.md) |
| 注册仓核函数等价迁移 | [ascendc-blaze-migration/SKILL.md](../../../../ascendc-blaze-migration/SKILL.md) |
| 精度与设备完成门禁（golden/ABI/验证层级） | [ascendc-blaze-best-practice/SKILL.md](../../../../ascendc-blaze-best-practice/SKILL.md) 的「可复用的精度与设备完成门禁」节 |
