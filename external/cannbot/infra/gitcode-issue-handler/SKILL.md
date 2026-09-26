---
name: external-cannbot-infra-gitcode-issue-handler
description: 处理 GitCode 单个或批量 Issue：分诊、首响答疑、算子责任人转交、再次回复跟踪、 环境核对、复现修复、PR 交付和结果报告。支持仅回复、不改代码，以及已答复咨询的自动闭环。
  触发：用户要求处理、回复、跟进仓库 Issue，从 Issue 修复并创建 PR，或初始化、调整本 Skill 的项目配置。
license: CANN-2.0
original-name: gitcode-issue-handler
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
---

# GitCode Issue Handler

只执行当前请求和当前阶段明确要求的动作；按脚本返回的下一步推进。

## 选择入口

| 请求 | 入口 |
| --- | --- |
| 只询问规则（policy_query） | 读取相关规则回答；不建运行树、不检查 Token/Git/CANN、不访问 API |
| 初始化、调整配置 | [configuration-setup.md](references/configuration-setup.md)；复用已明确的选择，只做配置 |
| 仓库批量处理、续跑 | [pipeline.md](references/pipeline.md)，新一轮执行 `resume --new-run`，续跑执行 `resume` |
| 显式单个 Issue URL、特殊时间或编号子集 | [issue-intake.md](references/issue-intake.md)；初始化见 [runtime-setup.md](references/runtime-setup.md) |
| 明确要求咨询自动闭环 | [maintenance-stale-close.md](references/maintenance-stale-close.md)，默认 dry-run |

`ISSUE_HANDLER_SKILL_ROOT` 是当前安装的本 Skill 目录（链接安装使用实际目标），从本入口路径确定。已配置 batch 启动命令：

```bash
export ISSUE_HANDLER_SKILL_ROOT="<本 SKILL.md 所在目录的绝对路径>"
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" resume --new-run --repository-root .
```

每次用户新发起的批量处理加 `--new-run`；继续已有轮次（包括新会话恢复）用普通 `resume`。脚本校验配置、恢复队列、选择扫描范围和分类，无需手动重建状态。返回 `fix_configuration` 时按字段错误修复或等待用户补值，再重跑原命令。

## 共同边界

- “全部 Issue”仍遵守责任配置；正常 single/batch 只处理核心 open，pending 先核查范围。忽略、观察、等待和转交均不等于解决。
- 仅回复不包含代码修复、指派或状态变更。首次文字响应、自提豁免、已有 PR 分配依分类和当前阶段规则处理。
- 外部动作按 [authorization-contract.md](references/authorization-contract.md) 和 [automation.md](references/automation.md) 的准确授权执行；配置开关与会话指令共同决定权限。领取、审核、生成草稿不产生发布授权。写入后 GET 回查，未知结果先核验再推进。
- 每阶段结论和证据写入唯一运行状态；外部等待保留恢复点。回复不披露内部环境、权限故障或敏感信息。规则有未明确冲突时先澄清，独立事项继续。
- 代码修复前读取 [code-worktree.md](references/code-worktree.md)，遵守独立工作区、复现和验证门禁；提交和发布按精确授权执行。

## 按需读取

batch 以命令的 `required_references` 为当前阶段清单；同会话已读且内容未变化可复用，新会话重新读取。任务文件提供结果契约。普通执行调用脚本，参数查 `--help`；源码、测试、安装文档仅在开发或排错时加载。

| 条件 | 参考 |
| --- | --- |
| 复杂且独立的调查需要委派 | [batch-analysis.md](references/batch-analysis.md)；简单范围任务由主会话处理 |
| 文字诊断、选择处理路径 | [issue-routing.md](references/issue-routing.md) |
| 草拟或审核回复 | [response-writing.md](references/response-writing.md) |
| 已确认算子维护动作、需要责任人 | [operator-handoff.md](references/operator-handoff.md)、[operator-owner-candidates.md](references/operator-owner-candidates.md) |
| 发送、跟进回复 | [issue-comment-workflow.md](references/issue-comment-workflow.md)、[issue-followup.md](references/issue-followup.md) |
| 复现根因、实施验证 | [code-root-cause.md](references/code-root-cause.md)、[code-validation.md](references/code-validation.md) |
| 交付确认、提交/PR/CI | [delivery-confirmation.md](references/delivery-confirmation.md)、[delivery-publish.md](references/delivery-publish.md)；pre-commit 钩子失败按 [gitcode-toolkit/references/pre-commit.md](../gitcode-toolkit/references/pre-commit.md) 处置 |
| 发布或代码交付后补报告 | [delivery-reporting.md](references/delivery-reporting.md) |
| 恢复异常、能力或阶段故障 | [runtime-state.md](references/runtime-state.md)、[runtime-capability-checks.md](references/runtime-capability-checks.md)、[policy-error-handling.md](references/policy-error-handling.md) 对应条目 |
| 知识检索；显式知识维护 | [runtime-knowledge.md](references/runtime-knowledge.md)；[knowledge-maintenance.md](references/knowledge-maintenance.md) |

GitCode API/Token/通用接口复用同级 gitcode-toolkit。首次安装或依赖异常见 [安装指南](docs/installation-guide.md)。状态字段仅在手动维护对应状态时查 [runtime-state-schema.md](references/runtime-state-schema.md)。

## 完成条件

逐项有真实结果或明确等待对象、下一步及适用草稿；分析和报告有已核实代码链接，未定位或缺稿说明原因。batch 回读返回的 `delivery_report`，报告失败用 `report` 恢复；无需重复手写状态或生成同一报告。最终提供可点击报告链接，区分处理、列举与观察。

## 修改验证

在 Skill 源仓运行 `python3 -m pytest -q infra/gitcode-issue-handler/tests infra/gitcode-toolkit/tests` 和 `python3 tests/lib/skill_validator.py validate-skill infra/gitcode-issue-handler/SKILL.md`。环境门禁变更额外覆盖延迟预检、缺 Token 恢复及 toolkit 兼容；无 NPU 不声称上板通过。
