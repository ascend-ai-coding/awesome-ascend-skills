---
name: external-cannbot-ops-ascendc-blaze-migration
description: 将 ops-nn、ops-transformer 等仓中的 Ascend 950 / DAV_3510 Matmul、BatchMatMul、GroupMatmul
  AscendC、CMCT 或 CGMCT 核函数等价迁移到 ops-tensor Blaze/tensor_api。使用场景：迁移边界、tiling/ABI 冻结、GM_ADDR/ListTensor、Blaze
  规范事实源、Scheduler/Kernel/BlockMmad/Epilogue 复用或扩展、CMCT/CGMCT 清理、逐字节一致性、性能和双仓 PR 门禁。证据不全为待确认（unknown），冲突为受阻（blocked）。
original-name: ascendc-blaze-migration
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# AscendC Blaze 等价迁移

## 范围

迁移只改设备侧核函数的表达和组件组织。主机侧 tiling、任务划分、数学语义、核函数 ABI、工作空间和支持域保持不变。

- 平台：Ascend 950 / DAV_3510。
- 算子：Matmul、BatchMatMul、GroupMatmul 家族。
- 来源：`ops-nn`、`ops-transformer` 或具备原核函数与验收入口的算子仓。
- 实现：Blaze 公共组件进入 `ops-tensor`；原仓只保留接入代码。
- 执行：默认单 Agent、单写入者。
- 交付：`ops-tensor` 实现 PR 和原仓接入 PR。

执行步骤见 [迁移工作流](references/migration-workflow.md)，证据格式见 [RotateQuant 试点](references/rotate-quant-pilot.md)。

## 执行顺序

开始任务时先读取迁移工作流。每次决定或动作都按以下顺序：

1. 只用一个 `migration-record.md` 记录状态。Provider、Bundle、Receipt 不能成为文件、对象或第二套状态；用户要求创建时也不创建。日志和验证结果是证据，由记录引用。
2. 每次答复的前两行必须是“当前门禁”和“状态”，再给依据、立即缺失信息、允许动作、禁止动作和下一步。
3. 缺失、未知、不可信或仅在沙箱失败的证据一律为待确认（`unknown`），只分析、取证和更新记录，不能标为 `blocked`。只有已证实的冻结契约冲突或已执行的验收失败才是受阻（`blocked`），并停止向下一门禁推进。
4. 当前门禁达到已验证（`verified`）后才能进入下一门禁。PR 已创建、独立样例通过或用户要求继续，都不能替代门禁证据。
5. 未读取活动源码和任务固定的 `ops-tensor` 源码前，不写修复代码，不推测 API，不决定新增公共组件。
6. 用户已给出的错误现场和原实现、迁移实现行为可以用于当前判断；缺少文件路径只限制源码核验，不能用来回避处理结论。
7. G0 待确认时先探测当前工作区可见的仓库和路径，创建或更新唯一的 `migration-record.md`，并一次列全版本、环境、依赖、命令、输入输出和原实现结果等立即缺失信息；不能只追问其中一项后停止。

记录只属于当前 task-id。不要全局搜索 `/tmp`，不要读取或复用其他任务的 `migration-record.md`；用户给出的事实足以回答门禁问题时，直接判断，不先寻找历史记录。

答复使用这个骨架，省略没有内容的说明，不省略门禁和状态：

```markdown
当前门禁：G0
状态：待确认（unknown）
依据：...
立即缺失信息：...
允许动作：...
禁止动作：...
下一步：...
```

## 事实源

G1 按顺序读取：

1. `ops-tensor/master` 最新的 [`CODING_CONVENTIONS.md`](https://gitcode.com/cann/ops-tensor/blob/master/CODING_CONVENTIONS.md)。
2. [`ascendc-blaze-best-practice`](../ascendc-blaze-best-practice/SKILL.md) 中匹配的场景和组件文档；涉及 AIC/AIV 或流水同步时读取 [Blaze 同步模式](../ascendc-blaze-best-practice/references/fundamentals/blaze-sync-patterns.md)。
3. 任务固定 `ops-tensor` 提交的公共源码，用于确认接口和能力。

记录规范对应的提交、获取时间和 SHA256。G3 前刷新；版本变化则按新规范重新检查。

最新规范、最佳实践或固定源码任一缺失时，G1 为待确认（`unknown`），不能编码。用户或同事口述存在冲突仍是待核验信息，不能据此标为 `blocked`；读取并确认规范与固定源码冲突后，记录影响并将 G1 标为受阻（`blocked`）。

沙箱中的 ACL、HAL 或设备失败不能作为原实现结果，也不能据此判断环境或实现失败；必须转到设备可见环境重跑，并记录设备、命令和返回码。

## 当前任务

每次迁移在 `/tmp/<task-id>/` 维护一个 `migration-record.md`。日志、tiling、逐字节比对、性能和 PR 证据放在同目录，由记录使用相对路径引用。

状态只有三种：

- 待确认（`unknown`）：当前门禁还缺事实或决定。
- 已验证（`verified`）：关闭条件已有证据。
- 受阻（`blocked`）：冻结契约冲突或验收失败。

G0 关闭前只做源码分析和取证。G0-G4 必须依次关闭；全部为已验证（`verified`）后，任务状态才是完成（`complete`）。

## 冻结契约

| 契约 | 内容 |
|---|---|
| 冻结 Tiling | tiling 结构、key、字段和值、工作空间、`blockDim`、分块与调度配置、任务映射和尾块规则 |
| 冻结 Blaze 架构 | 固定 Blaze 的公共接口、组件职责、模板契约、资源所有权、依赖方向和同步协议 |
| 运行时 | 核函数 ABI、`GM_ADDR` 身份与解引用顺序、形状、数据类型、布局、转置、量化、分组编码和边界输入 |
| 数学 | 表达式、累加与转换数据类型、舍入、饱和、NaN/Inf 和同步顺序 |

冻结 Tiling 与冻结 Blaze 架构冲突时，任务进入受阻（`blocked`）。

GM ABI 问题沿活动调用图处理：先定位第一个 MTE 错误，确认 `GM_ADDR` 是数据地址还是 TensorList/ListTensor 等描述符，保持原实现的描述符解析和分组或批次偏移顺序。未调用代码不进入迁移契约，也不保留为迁移方案。

现有公共组件能表达需求就复用。公开接口缺少能力时，先按 Scheduler、Kernel、BlockMmad、Epilogue、Policy/Fusion/Utility 逐层列出公共候选、匹配项、能力缺口、适用范围和方案，再由用户决定是否扩展；未完成逐层分析或批准只覆盖部分层级时，G1 为待确认（`unknown`），不能标为 `blocked`。用户拒绝必要方案或确认冻结契约冲突后才是受阻（`blocked`）。获批扩展进入 `ops-tensor`。DAV_3510 迁移实现的活动依赖和改动文件清除 CMCT/CGMCT；包装层、别名和编译开关不算清除。

新增公共类型必须写明支持与拒绝范围、参数化轴、资源和同步约束；不支持组合由编译期拒绝，并有正负向实例化测试。缺少任一项时 G3 规范检查失败。

## 门禁

| 门禁 | 关闭条件 |
|---|---|
| G0 环境与原实现 | 固定两仓、CANN/tensor_api、设备、依赖、输入、命令和原调用链结果 |
| G1 契约与能力 | 完成活动调用图、tiling 特征快照、GM ABI、支持域、API 映射和逐层能力缺口；用户批准新增设计 |
| G2 构建与接入 | 完整支持域构建通过；全新检出环境取得精确依赖；原构建和核函数下发路径执行迁移实现 SHA；CMCT/CGMCT 为零 |
| G3 等价与性能 | 全部有效输出逐字节一致；重复运行、尾块和越界防护区检查通过；性能达标或取得例外验收；Blaze 规范检查通过 |
| G4 双仓 PR | 两个 PR 已创建，最新提交 SHA 与 G2-G3 证据一致，第三方可执行复现说明 |

CI 和评审只在任务启动时写入完成线后纳入 G4。

直接询问能否通过时，先根据用户给出的事实判定对应门禁，不因缺少历史记录回避结论。G3 等价判定只比较原实现与迁移实现的全部有效输出，CPU 参考结果只作交叉检查；性能逐个用例报告，例外验收记录决策人、比值、理由、范围和风险。

G4 只有在 G2、G3 均为已验证（`verified`）时才能关闭。依赖未固定或测试程序、输入只在 `/tmp` 时，G2-G4 均为待确认（`unknown`）；全新检出环境必须取得精确依赖，复现说明必须包含两仓 SHA、环境、构建、安装、运行、仓内测试程序与输入、逐字节比对和性能协议。
