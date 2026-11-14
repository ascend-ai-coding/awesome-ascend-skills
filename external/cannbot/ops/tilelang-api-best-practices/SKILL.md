---
name: external-cannbot-ops-tilelang-api-best-practices
description: TileLang NPU API 查询入口。适用于用户需要API 查询时，根据已选 Ascend910 或 Ascend950 平台读取独立技能；两套规则不混用。
original-name: tilelang-api-best-practices
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-14'
synced-commit: 750045df251d031bf708de9dc1614c2202eed6dd
license: UNKNOWN
---

# TileLang API 查询

## 平台路由

本文件仅定位技能，不定义公共业务步骤。工作流已确定平台时使用调用方传入的 `Ascend910` 或 `Ascend950`，完整读取下表唯一对应正文，并复用调用方传入的完整硬件探测证据。独立调用且平台未知时，先按名称加载 `ascendc-env-check` Skill（仅使用其探测脚本），将该 Skill 的真实目录设为 `ASCENDC_ENV_CHECK_DIR`，运行 `python3 "$ASCENDC_ENV_CHECK_DIR/scripts/get_npu_arch.py" --json`；仅在退出码为 0 时将完整 JSON 作为 `evidence`，并使用其 `full_soc` 与 `npu_arch` 组合选择平台。Ascend910B1～B4（含 B2C、B4-1）/ Ascend910_93 系列且 NpuArch=2201 时选择 `Ascend910`；Ascend950PR / Ascend950DT 系列且 NpuArch=3510 时选择 `Ascend950`。不得从当前 Skill 目录拼接跨 Skill 相对路径，也不得用静态映射反推本机型号。平台确定后，按名称加载 `npu-arch` Skill 并传入该平台查询架构与硬件能力；脚本失败、Skill 缺失、证据冲突或平台不受支持时停止并报告。

| 平台 | 技能正文 |
|---|---|
| Ascend910 | [API 查询](Ascend910/instructions.md) |
| Ascend950 | 未提供此分支的独立实现。独立问答场景按「降级回答」执行（见下文）；算子开发工作流场景返回该平台工作流。 |

## 执行

正文中的技能引用、脚本和模板都以所选分支为准。只执行该正文规定的步骤，不读取另一分支、不共享 API 或测试规则；唯一的例外是下文「Ascend950 降级回答」。技能根目录的资源链接指向 Ascend910 分支；Ascend950 使用其分支目录内的资源。

### Ascend950 降级回答（分支未实现时）

独立问答场景（非算子开发工作流）路由到 Ascend950、但本技能无该分支独立实现时，**不得裸拒答**，按以下规则降级执行：

1. 读取 Ascend910 分支正文作答，但回复开头必须显式声明：「Ascend950 分支暂无独立实现，以下规则源自 Ascend910 分支，在 Ascend950 平台上需以官方文档或 tilelang 源码核实」。声明不得省略。
2. 不得将 Ascend910 的硬件数值约束（对齐字节、容量上限、pass_config 默认值等）当作 Ascend950 的权威数值直接给出；涉及具体数值约束时必须连带提示「该数值为 910 约束，950 需核实」。
3. API 语义、编程模式、同步策略等平台无关的知识正常作答，不受第 2 条限制。
4. 算子开发工作流（tilelang-op-design / tilelang-op-develop 等链路）内部调用本技能时，仍按各自技能的路由规则执行，不适用本降级规则。
