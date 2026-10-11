---
name: external-cannbot-ops-tilelang-op-test-design
description: TileLang NPU 算子测试设计与验收入口。适用于用户需要算子测试设计与验收时，根据已选 Ascend910 或 Ascend950
  平台读取独立技能；两套规则不混用。
original-name: tilelang-op-test-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-11'
synced-commit: 34920ba29ef88355bf9190b526e59970d95f696c
license: UNKNOWN
---

# TileLang 算子测试设计与验收

## 平台路由

本文件仅定位技能，不定义公共业务步骤。工作流已确定平台时使用调用方传入的 `Ascend910` 或 `Ascend950`，完整读取下表唯一对应正文，并复用调用方传入的完整硬件探测证据。独立调用且平台未知时，先按名称加载 `ascendc-env-check` Skill（仅使用其探测脚本），将该 Skill 的真实目录设为 `ASCENDC_ENV_CHECK_DIR`，运行 `python3 "$ASCENDC_ENV_CHECK_DIR/scripts/get_npu_arch.py" --json`；仅在退出码为 0 时将完整 JSON 作为 `evidence`，并使用其 `full_soc` 与 `npu_arch` 组合选择平台。Ascend910B1～B4（含 B2C、B4-1）/ Ascend910_93 系列且 NpuArch=2201 时选择 `Ascend910`；Ascend950PR / Ascend950DT 系列且 NpuArch=3510 时选择 `Ascend950`。不得从当前 Skill 目录拼接跨 Skill 相对路径，也不得用静态映射反推本机型号。平台确定后，按名称加载 `npu-arch` Skill 并传入该平台查询架构与硬件能力；脚本失败、Skill 缺失、证据冲突或平台不受支持时停止并报告。

| 平台 | 技能正文 |
|---|---|
| Ascend910 | [算子测试设计与验收](Ascend910/instructions.md) |
| Ascend950 | [算子测试设计与验收](Ascend950/instructions.md) |

## 执行

正文中的技能引用、脚本和模板都以所选分支为准。只执行该正文规定的步骤，不读取另一分支、不共享 API 或测试规则。技能根目录的资源链接指向 Ascend910 分支；Ascend950 使用其分支目录内的资源。
