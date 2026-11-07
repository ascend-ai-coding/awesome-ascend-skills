---
name: external-cannbot-infra-gitcode-issue-handler
description: 处理 GitCode 单个或批量 Issue：分诊、首响答疑、算子责任人转交、再次回复跟踪、 环境核对、复现修复、PR 交付和结果报告。支持仅回复、不改代码，以及已答复咨询的自动闭环。
  触发：用户要求处理、回复、跟进仓库 Issue，从 Issue 修复并创建 PR，初始化、调整本 Skill 的项目配置， 或解释、模拟本 Skill 的分类、调查编排、PR
  覆盖判断及授权规则。
license: CANN-2.0
original-name: gitcode-issue-handler
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-07'
synced-commit: 2374941495c1549eb4a1a5a82987b567d61855f5
---

# GitCode Issue Handler

只执行当前请求和当前阶段明确要求的动作；按脚本返回的下一步推进。

## 选择入口

| 请求 | 入口 |
| --- | --- |
| 只询问规则（policy_query） | 读取相关规则回答；不建运行树、不检查 Token/Git/CANN、不访问 API |
| 回复内部自检、评分或档位说明 | [response-quality-rubric.md](references/response-quality-rubric.md)；按 Issue 类型给档位、理由与缺项，说明 0.25 不换算总分或通过线，自评仅内部留存 |
| 只补缺失配置，不改参数、不问答 | [configuration-setup.md](references/configuration-setup.md#仅补缺失文件)；需 Python 3.10+、PyYAML 和可写目标目录，不需要 Token/CANN |
| 初始化、调整配置 | [configuration-setup.md](references/configuration-setup.md)；复用已明确的选择，只做配置 |
| 范围待核查、调查分组或编排说明 | 先读 [batch-analysis.md](references/batch-analysis.md) 判断复杂度；简单范围由主会话核查，复杂独立调查才委派；实际队列仍走 pipeline |
| 分工记录、模型不可选或无子 agent 的降级说明 | [分工记录与能力降级](references/batch-analysis.md#分工记录与能力降级)；交代任务范围、执行者、状态、结果路径及串行/升级原因，以 `run_state.artifacts` 引用 |
| 多 PR 覆盖调查、首响方案与作者选择（含模拟） | 先读 [PR 覆盖调查](references/batch-analysis.md#pr-覆盖调查) 和 [自动化规则](references/automation.md#有效修复-pr-的自动关联与作者临时指派)；说明轻量/升级边界、覆盖证据与过滤选择 |
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
- 非自提 Issue 须具备可核验的首次有效回复和平台负责人，缺一项不能标记处理完成。自提免首次响应：提出者与负责人为同一账号，或提出者与有效关联 PR 作者为同一账号；无负责人仍补指派，新追问仍跟进。纯咨询和 direct 不另行豁免。仅回复请求仍不授权指派或状态变更，缺项须保留待办。
- 外部动作按 [authorization-contract.md](references/authorization-contract.md) 和 [automation.md](references/automation.md) 的准确授权执行；配置开关与会话指令共同决定权限。领取、审核、生成草稿不产生发布授权。写入后 GET 回查，未知结果先核验再推进。
- 本次处理为非自提 Issue 成功指定责任人后，添加 `pending-confirmation` 标签并回查，保留其他标签；自提 Issue 和仅找到候选、尚未指派的项不添加。执行与恢复见 [automation.md](references/automation.md#指派后的确认标签)。
- 首响与指派分别核验：规则已明确的接单人不等待 kernel 根因确认；每条实际处理项记录回复和指派状态，缺一项保留具体待办，收尾按 [双项核对](references/delivery-reporting.md#首响与指派的逐项核对) 校验。
- 每阶段结论和证据写入唯一运行状态；外部等待保留恢复点。回复不披露内部环境、权限故障或敏感信息。规则有未明确冲突时先澄清，独立事项继续。
- 每条 `need_attention` 先保存已知 `analysis.md` 和可确定的完整回复/分配稿；就绪项先落盘再发布，受阻项记录缺稿原因与恢复点。共享状态只由协调者更新，收尾按 [逐项响应材料](references/delivery-reporting.md#逐项响应材料) 核对最终路径，不等其他独立项。
- 代码修复前读取 [code-worktree.md](references/code-worktree.md)，遵守独立工作区、复现和验证门禁；提交和发布按精确授权执行。

## 按需读取

batch 以命令的 `required_references` 为当前阶段清单；同会话已读且内容未变化可复用，新会话重新读取。任务文件提供结果契约。普通执行调用脚本，参数查 `--help`；源码、测试、安装文档仅在开发或排错时加载。

仅模拟或解释流程时也按适用参考说明分流、依赖与收尾。用户禁止实际读取时，说明执行时应先读取的文档及适用步骤，不声称已经读取、执行或验证；不从当前 Skill 仓库推定目标仓库的版本或责任结论。

仅说明范围核查编排时，也须说明执行前先读 `batch-analysis.md`：按共享源码/文档证据分组，简单核查由主会话批量完成，复杂独立调查才委派；默认最多三个运行任务、每执行者一个，受宿主名额限制，逐项 `claim → submit → accept`，空闲立即补位。子任务接收准确 URL、原文、版本、源码线索、责任配置、共享证据引用和独立结果目录，只写自己的结果；主会话逐项审核并推进就绪项，不等全批。子集重分类复用同批 `--pr-snapshot` 并加 `--no-update-last-check`，只替换受影响项、保留其他结果，不推进全批游标或宣称全批完成。

| 条件 | 参考 |
| --- | --- |
| 复杂且独立的调查需要委派 | [batch-analysis.md](references/batch-analysis.md)；简单范围任务由主会话处理 |
| 文字诊断、选择处理路径 | [issue-routing.md](references/issue-routing.md) |
| 草拟、改写或审核回复（含模拟） | [response-writing.md](references/response-writing.md)；咨询另按 [评分表](references/response-quality-rubric.md) 记录咨询档位、已满足事实、缺项与理由，交付完整答案 |
| 已确认算子维护动作、需要责任人 | [operator-handoff.md](references/operator-handoff.md)、[operator-owner-candidates.md](references/operator-owner-candidates.md) |
| 发送、跟进回复 | [issue-comment-workflow.md](references/issue-comment-workflow.md)、[issue-followup.md](references/issue-followup.md) |
| 复现根因、实施验证 | [code-root-cause.md](references/code-root-cause.md)、[code-validation.md](references/code-validation.md) |
| 交付确认、提交/PR/CI | [delivery-confirmation.md](references/delivery-confirmation.md)、[delivery-publish.md](references/delivery-publish.md)；pre-commit 钩子失败按 [gitcode-toolkit/references/pre-commit.md](../gitcode-toolkit/references/pre-commit.md) 处置 |
| 发布或代码交付后补报告 | [delivery-reporting.md](references/delivery-reporting.md) |
| 恢复异常、能力或阶段故障 | [runtime-state.md](references/runtime-state.md)、[runtime-capability-checks.md](references/runtime-capability-checks.md)、[policy-error-handling.md](references/policy-error-handling.md) 对应条目 |
| 知识检索；显式知识维护 | [runtime-knowledge.md](references/runtime-knowledge.md)；[knowledge-maintenance.md](references/knowledge-maintenance.md) |

GitCode API/Token/通用接口复用同级 gitcode-toolkit。首次安装或依赖异常见 [安装指南](docs/installation-guide.md)。状态字段仅在手动维护对应状态时查 [runtime-state-schema.md](references/runtime-state-schema.md)。

## 完成条件

逐项 GET 核对有效首响评论（自提项核对豁免依据）和非空 assignee；已有有效结果直接复用，不重复评论或覆盖负责人。有效首响或经核验的自提豁免、以及负责人齐备才可标记本项处理完成；全批还须对照完整输入清单核对无漏项。待授权、未知 owner、写入失败或回查未知时保持等待，报告直接显示缺项、等待对象及恢复步骤；代码根因未确认不阻断已匹配接单规则的指派。处理完成不等于问题已解决。

报告顶部以汇总表展示每条 Issue 的最终结果、平台状态、首响 SLA 和下一步，详细处理过程放在下方默认折叠的详情中。逐项有真实结果或明确等待对象、下一步及适用草稿；分析和报告详情有已核实代码链接，未定位或缺稿说明原因。batch 回读返回的 `delivery_report`，报告失败用 `report` 恢复；无需重复手写状态或生成同一报告。

处理结束或停在等待状态时，在当前对话中直接给出简短结果汇总列表，每条实际处理的 Issue 一行：编号链接、最终结果、平台状态、负责人和首响 SLA；未完成项补充下一步或等待对象。末尾附可点击报告链接，详细过程留在报告中。不要只回复完成通知或文件链接；区分处理、列举与观察，无实际处理时明确说明。

## 修改验证

在 Skill 源仓运行 `python3 -m pytest -q infra/gitcode-issue-handler/tests infra/gitcode-toolkit/tests` 和 `python3 tests/lib/skill_validator.py validate-skill infra/gitcode-issue-handler/SKILL.md`。环境门禁变更额外覆盖延迟预检、缺 Token 恢复及 toolkit 兼容；无 NPU 不声称上板通过。
