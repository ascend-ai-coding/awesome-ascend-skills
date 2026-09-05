---
name: external-cannbot-ops-ascendc-blaze-migration
description: 评估并将 ops-nn、ops-transformer 等仓中 Ascend 950 / DAV_3510 的 AscendC Kernel
  分支等价迁移到 ops-tensor Blaze/Tensor API。适用于 CMCT、CGMCT、单体或自定义分层 Kernel，以及 Cube、Vector、MIX、高阶
  Matmul、手工 Buffer、数据搬运、自定义调度、VF 和同步路径；在隔离迁移工程中串行完成迁移合同冻结、Blaze 开发、验证资产生成、原始基线、Blaze
  验收和双仓本地提交交付。
original-name: ascendc-blaze-migration
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# AscendC Kernel 到 Blaze 等价迁移

## 核心约束

将迁移视为受约束的设备侧结构重构，不是重新实现相同数学功能。默认保持主机 Tiling、TilingKey、ABI、workspace、blockDim、支持域、逻辑任务覆盖、数学语义和性能合同不变。

源活动 Kernel 中可由 Tensor API 或 Blaze 表达的计算、数据搬运、Tensor、Buffer、layout 和资源生命周期都必须迁移。只允许把 ABI、核索引、AIC/AIV 分支、硬件同步和设备模式控制保留为运行时边界；不得用运行时边界豁免 Copy、Buffer、layout 或 Matmul。

平台限于 Ascend 950 / DAV_3510。所有 DAV_3510 迁移必须清除目标活动依赖闭包和改动实现文件中的 CMCT/CGMCT，包装层、别名和编译开关只可作为有证据的排除项。

状态只使用 `unknown`、`verified`、`blocked`。证据不完整、正在执行或执行条件暂不具备为 `unknown`；证据完整且通过为 `verified`；完成诊断后仍存在不可执行的合同冲突或验收失败为 `blocked`。

## 串行门禁

```text
G0 迁移准备
  -> G1 迁移合同与行为设计冻结
  -> G2 Blaze 迁移开发与静态验收
  -> G3 验证设计、runner、输入和正式制品准备
  -> G4 原始实现基线验证
  -> G5 Blaze 迁移验收
  -> G6 代码交付
```

任意时刻只有一个活动门禁。当前门禁未达到关闭条件时不得启动下一门禁；关闭后立即登记交接并自动继续，不等待用户批准或发起额外 Question。

| 门禁 | 关闭条件 |
|---|---|
| G0 迁移准备 | 隔离工程、仓库身份、入口、环境状态文件和执行记录完整；不编译、不安装、不运行 |
| G1 迁移合同与行为设计冻结 | 接口合同、源活动行为、迁移范围、owner、内部执行合同、覆盖义务、判定和 review 计划完整；不生成具体 case、runner 或正式制品 |
| G2 Blaze 迁移开发与静态验收 | Blaze 迁移、组件证明、反模式检查、CMCT/CGMCT 清理和开发反馈编译完成；不进行正式设备功能/性能验收 |
| G3 验证设计、runner、输入和正式制品准备 | 将 G1 覆盖义务实例化为 10~30 个正式用例，完成 ACLNN runner、输入资产及 original/Blaze 正式 OPP 构建并冻结 |
| G4 原始实现基线验证 | 用冻结 runner 和输入完成 original 全部功能、稳定性和 msprof 性能验证并固定二进制基线 |
| G5 Blaze 迁移验收 | 只运行 Blaze，完成功能逐字节比较、msprof 性能比较、最终 review 和问题闭环 |
| G6 代码交付 | 两仓 `master` 分支提交与 G5 证据一致，依赖和复现信息完整，过程资产不进入产品仓 |

## 迁移工程

在 Agent CLI 启动目录创建单层任务目录，所有开发只发生在隔离代码仓中：

```text
blaze-migration-<task-id>/
├── repo/{original,blaze}/
├── reports/{migration-record.md,environment-state.json,migration-design.md,migration-validation.md,migration-review.md}
├── packages/{original,blaze}/
├── validation/{runner,cases,inputs,results/{original,blaze},msprof/{original,blaze},logs}/
└── SHA256SUMS
```

不要使用 `/tmp` 保存任务资产，不要修改 `repo/original/` 的基线 `master`；迁移代码直接提交到 `repo/blaze/` 的 `master`，不要在 `validation/` 复制产品代码仓，也不要把过程资产放入产品仓。

## 强制执行规则

1. 只用 `migration-record.md` 维护当前门禁、阶段状态、交接和证据索引；其他文档提供被引用的事实。
2. G0 只调用 `ascendc-env-check` 一次，将事实规范化写入 `environment-state.json`。后续门禁只读该文件，不重复探测，不以 `npu-smi` 等单一工具覆盖结论。
3. 环境文件只记录事实，不决定门禁。各阶段按自身文档消费构建、设备运行和 profiling 能力；能力不足时保持当前阶段 `unknown`，开发可继续，只有需要设备的验证动作才可被阻塞。
4. G0 必须完成源码仓隔离、代码身份冻结和依赖身份记录；源码获取的具体规则统一遵循[串行工作流](references/serial-workflow.md)的 G0 章节。`repo/original/` 保持基线 `master` 不变，`repo/blaze/` 在 `master` 上直接开发并提交，后续阶段使用提交后的代码身份。
5. G1 的 `migration-design.md` 必须完整冻结接口合同、源执行模型、迁移范围、内部执行合同、覆盖义务、判定规则、性能合同、构建计划和 review 计划；它必须足以指导 Blaze 开发，但不提前生成具体 case、runner、输入 bin 或原始基线。
6. 目标平台和场景的对外接口文档是支持约束的权威来源。checker、binary config、Host Tiling、TPL_SEL 和 Kernel 只提供一致性与路由证据，不得静默缩小文档合同。
7. G1 冻结的合同、源行为、迁移范围和覆盖义务在 G2/G3/G4/G5 不得删除、缩小、替换或按 Blaze 当前实现改写。发现 G1 事实错误必须返回 G1。
8. G2 只在 `repo/blaze/` 开发；G2 可进行必要的增量/翻译单元编译反馈，但不得建立正式 original 基线或进行正式设备功能、性能验收。
9. G3 只能从 G1 覆盖义务和独立源证据实例化具体用例，正式用例总数为 10~30；不得因 Blaze 实现困难删除或重分类用例。G3 才开发 runner、生成输入 bin、构建两套正式 OPP。
10. runner 必须基于目标算子的 ACLNN example 和 ACLNN 参考文档生成直接 ACLNN 测试程序，支持读取/输出 bin，并让 original 与 Blaze 使用相同 runner、输入和执行参数；不得使用 TTK、ST、ATK 或其他替代验证后端。
11. G4 只运行 original，建立稳定的二进制输出基线；G5 只运行 Blaze 并与 G4 基线逐字节比较。G4/G5 的正式性能测试调用 `ops-profiling` 并使用 msprof；本 skill 只维护性能 case、身份、可比性和门限。
12. 门禁关闭必须依赖逐 case 结果、环境 revision/hash、package/runner/input 身份、实际路由、精度、guard、执行次数和 profiling 原始数据索引。
13. runner 必须自动校验 design 的验证义务、具体 case 表、runner 注册、case 资产、执行清单与实际结果集合；任何 `FAIL`、`NOT_RUN`、缺失、零执行次数、错误路由、输出/动态元数据缺失或身份不一致都按 fail-closed 传播。
14. G5 功能验收的唯一通过标准是 original 与 Blaze 有效输出及动态元数据逐字节一致；close、rtol、atol、平均误差和外部 golden 只能作为诊断字段。
15. 功能、性能或审查问题在冻结合同内自动回退、修复、重新构建并复测，不设置普通 finding 审批点。
16. 证据只对绑定的代码、依赖、工具链、环境、制品、runner、用例和输入有效。变化后按 owner 回退，不得改报告哈希或复制旧结果恢复有效性。
17. 三份正式报告、migration record、环境状态、runner、用例、制品和 profiling 数据始终留在迁移工程，不属于代码交付件。

答复使用本 skill 的中文门禁和阶段名称，并给出状态、依据和下一步：

```markdown
当前门禁：G2 Blaze 迁移开发与静态验收
当前阶段：G2.2 迁移实现
状态：unknown
依据：...
下一步：...
```

## 按需读取

- 开始任务或推进门禁时，读取[串行工作流](references/serial-workflow.md)。
- 创建工程、固化环境、维护记录或判断证据所有权时，读取[工作区与证据](references/workspace-and-evidence.md)。
- 执行 G1 或设计迁移合同时，读取[合同设计与验证看护](references/design-and-harness.md)。
- 在 G3 具体化用例、生成输入或编写 runner 时，读取[用例覆盖设计](references/case-coverage-design.md)和[ACLNN Runner 开发](references/aclnn-runner-generation.md)。
- 执行 G2 时，读取[Blaze 迁移开发](references/blaze-migration.md)。
- 执行 G3/G4 原始制品和基线时，读取[原始实现基线](references/original-baseline.md)。
- 编译 Blaze/ops-nn OPP、确认本地 ops-tensor include staging 或执行 `asc_opc` 任务时，读取[Blaze OPP 编译指导](references/blaze-opp-build.md)。
- 执行 G5 功能、性能和最终审查时，读取[Blaze 迁移验收](references/blaze-acceptance.md)。
- 处理环境变化、证据失效、阶段回退或 G6 时，读取[恢复与交付](references/recovery-and-delivery.md)。

按需使用 `ascendc-blaze-best-practice` 调查现有 Blaze concrete witness，使用 `ascendc-st-design` 做约束与覆盖分析，使用 `ops-profiling` 负责 msprof 性能采集与分析。只复用相关能力，不运行与本流程冲突的大规模默认产物流程。

## 完成定义

只有 G5 为 `verified`，且 G6 的两仓代码身份、依赖和复现信息与最终证据一致时，才能声明迁移完成。编译、单例、局部测试、报告生成、PR 创建或远端推送均不能单独代替完成条件；G6 默认只创建本地 git commit。
