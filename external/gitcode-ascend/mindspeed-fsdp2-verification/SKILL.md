---
name: external-gitcode-ascend-mindspeed-fsdp2-verification
description: 用于执行 MindSpeed-MM FSDP2 迁移的功能与可靠性验收门禁。适用于模型/数据/配置改动后，验证一次分布式端到端成功并留存证据时。
original-name: mindspeed-fsdp2-verification
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed FSDP2 验证

## 目标

基于功能与可靠性门禁验证迁移产物，并产出有证据支撑的验收结果。

## 适用场景

- 模型、数据、配置迁移产物已准备好，待验证。
- 用户要求对 MindSpeed-MM FSDP2 迁移做端到端验收。
- 迁移任务宣称已就绪，需要证据证明。

## 不适用场景

- 任务是实现模型/数据/配置代码。
- 任务仅做迁移设计而不执行。
- 任务聚焦性能基准测试。

## 输入

- 已迁移的模型/数据/配置产物
- 真实运行资产（模型路径、数据路径、可选图像根路径）
- 验收规则：
  - 启用功能门禁
  - 启用可靠性门禁
  - 要求一次分布式 E2E 跑通
  - 不启用性能门禁
  - 不启用 checkpoint 恢复门禁

## 输出

- `verification_report.md`
- `evidence.json`
- `failed_cases.md`（仅失败时产出）

## 必要门禁

1. 功能门禁：
   - 模型注册/构建链路成功
   - 数据注册/构建链路成功
   - 配置关联一致
   - 至少一次分布式端到端训练成功
2. 可靠性门禁：
   - 常见错误配置有清晰错误行为
   - 关键兼容性检查通过（schema、签名、路径类型、注册）

## 防偏差执行协议

1. 验证前必须完成：
   - 确认验证使用真实运行资产，不得使用占位路径
   - 确认失败回流模板已就绪
2. 验证中必须执行：
   - 功能门禁、可靠性门禁分开记录命令与退出码
   - 分布式 E2E 至少执行一次成功运行
3. 验证后必须产出：
   - `verification_report.md`
   - `evidence.json`
   - `failed_cases.md`（如失败）

## 禁止行为

- 禁止用 `print("check ...")` 代替真实检查
- 禁止只记录“通过/失败”而不附命令证据
- 禁止在 E2E 未执行成功时宣称迁移完成
- 禁止失败后不给责任技能映射与修复动作

## 明确排除项

- 本版本不做性能门禁。
- 本版本不做 checkpoint 保存/恢复门禁。

## 执行规则

- 运行验证命令并记录退出码。
- 没有命令证据不得宣称通过/失败。
- 任一门禁失败时，必须返回可执行的失败分析并映射责任技能：
  - model -> `mindspeed-fsdp2-model-migration`
  - data -> `mindspeed-fsdp2-data-migration`
  - config -> `mindspeed-fsdp2-config-migration`

## 退出标准

- 所有启用门禁通过且证据记录完整。
- 最终状态可追溯到明确的命令输出。

## 渐进式资源

- `checks/verification-gates.md`
- `checks/verification-hard-gates.md`
- `templates/evidence-json-template.md`
- `templates/command-templates.md`
- `templates/failure-case-template.md`
- `examples/minimal-io.md`
- `examples/handoff-context.md`

仅在门禁评估或证据格式化需要详细判据时读取这些文件。
