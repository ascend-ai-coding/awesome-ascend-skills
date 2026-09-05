# 交付：报告与完成条件

## 目录

1. [读取时机](#读取时机)
2. [报告产物与生成顺序](#报告产物与生成顺序)
3. [逐 Issue 完整性契约](#逐-issue-完整性契约)
4. [报告结构](#报告结构)
5. [指标口径](#指标口径)
6. [内部卡点](#内部卡点)
7. [清理结果](#清理结果)
8. [完成条件](#完成条件)

## 读取时机

所有 Issue 和分组到达终止或等待状态后，执行步骤 9 前完整读取本文件。

## 报告产物与生成顺序

每次主 Issue 处理运行必须生成两份内容相同的 Markdown 报告：

- 历史报告：`.cannbot/gitcode-issue-handler/reports/<run_id>/summary.md`
- 最新报告：`.cannbot/gitcode-issue-handler/reports/latest.md`

历史报告不可覆盖其他 `run_id`。即使没有匹配 Issue、启动/同步失败、只有
`no_attention`、全部进入等待状态或发布失败，也必须生成报告，明确说明为何结束。

严格按以下顺序执行：

1. 汇总本轮实际处理的 Issue 和关联 group；纯获取/分类项只保留聚合计数。
2. 回查步骤 6f 已批准并执行的外部评论/指派，记录统一确认、写入和 GET 结果；未批准项
   只保留预览，不在报告阶段补写。
3. 执行安全清理，将清理和保留项写回运行状态。
4. 检查责任人方案输入队列应已在统一执行预览前清空。若后续新证据产生未知 owner，返回
   下方检查点补齐方案、更新完整预览并确认，不能在报告阶段静默指派。
5. 设置 `completed_at` 和 `overall_status`，把完整运行状态写到
   `.cannbot/gitcode-issue-handler/reports/<run_id>/run_state.json`。
6. 在目标仓库根目录执行：

   ```bash
   python "$ISSUE_HANDLER_SKILL_ROOT/scripts/generate_summary_report.py" \
     --state ".cannbot/gitcode-issue-handler/reports/<run_id>/run_state.json" \
     --strict
   ```

7. 回读生成的 `summary.md`，确认只出现本轮实际处理的 Issue，数量与 `issues_total`
   一致，且不含 `self_assigned` 或纯 `/assign` 项。
8. 更新 `report_generated: true`、`report_path`；最终回复只需给简短结论，但必须提供报告
   的可点击路径。

生成失败时先修正缺失状态并重试；不得用临时对话摘要代替文件报告，也不得因为某个
Issue 失败就跳过整个报告。

## 逐 Issue 完整性契约

`issues` 只保存和展示本轮**实际处理**的 Issue。实际处理是指至少满足一项：

- 进入 `need_attention` 后完成诊断、答疑、索要上下文、转交或等待状态落盘；
- 检测并处理提出者在维护侧回复后的新增评论，或完成`挂起`/`进行中`状态迁移与回查；
- 执行复现、根因确认、代码修改、测试、提交、推送、PR 或 CI 处置；
- 本轮发送了非 `/assign` 的有效工作流评论并回查成功。

以下均不是实际处理，不进入 `issues` 和逐 Issue 报告：

- 仅获取或固定规则分类；
- 发现已有负责人、已有 PR、已有评论或已有人跟进；
- `self_assigned`；
- 自动或已有的纯 `/assign`，尤其是 Issue 作者与关联 PR 作者相同的自提 Issue。

扫描规模写入 `run.issues_scanned_total`；`run.issues_total` 只统计实际处理 Issue，并等于
`issues` 长度。无需处理项最多保留 category 级聚合计数，不保存标题、作者、assign 内容、
PR 链接等逐项明细。

每个实际处理 Issue 只要求保留对结论有用的内容：

| 类别 | 必需内容 |
| --- | --- |
| 身份 | iid、标题、URL、作者 |
| 结果 | `handling_status`、`resolution_status`、`result_summary`、`next_action` |
| 过程 | 本轮实际动作及必要证据；不要为未执行阶段填充大段“不适用” |
| 对话跟踪 | 实际进入等待或再回复处理时，记录 `conversation_state`、最近双方 comment ID/时间和 follow-up SLA |
| 条件项 | 有复现/变更/测试/发布/卡点时才记录对应字段 |

`process_log` 每条使用以下结构；时间拿不到时写 `unknown`，不得删除记录：

```json
{
  "time": "2026-08-11T12:00:00+08:00",
  "stage": "triage | diagnose | reproduce | implement | validate | authorize | deliver | comment",
  "action": "执行了什么",
  "result": "结果与状态变化",
  "evidence": ["命令、相对路径、commit/PR/Issue URL 或关键观测"]
}
```

状态必须写真实值。创建但未合入 PR、等待用户上下文、环境不匹配、转交责任人或 CI
失败时，分别保持 `resolution_pending`/`unresolved`，不能为让报告好看而标记 resolved。

## 报告结构

`generate_summary_report.py` 生成精简报告：

1. **运行概览**：仓库、时间范围、总体状态、扫描数、实际处理数。
2. **本次实际处理的 Issue**：每项只展示结果、状态、下一步和链接；根因、评论、变更、
   测试、PR 仅在真实存在时追加。
3. **变更与交付**：仅存在代码修复组时生成。
4. **卡点与验证边界**：仅运行被阻塞/部分完成，或与实际处理 Issue 直接相关时生成；
   已恢复的限流、无影响的扫描预算等内部过程不进入 Markdown。

不要生成空章节，不展示例行分类轨迹、空环境、空测试、空清理、`unknown` 表格或
逐项 `no_attention` 说明。完整审计数据保留在 `run_state.json`，Markdown 面向快速阅读。

报告是给 Skill 使用者的内部交付物，可以包含维护侧环境和内部 blocker，但禁止写入
Token、Authorization、密码或其他 secret。命令和本地产物优先使用目标仓库相对路径；
脚本会按敏感键和常见认证格式再次脱敏。

## 指标口径

- 有效响应：维护者、责任人或授权助手给出受理、判断、最小信息请求、明确转交或
  解决结论；系统消息和仅 `/assign` 不计。
- 解决时长：从 `created_at` 到可核验的 `resolved_at`，按自然日。
- 解决率：统计周期内已解决的本仓职责 Issue / 本仓职责 Issue 总数。经证据确认的
  重复、非本仓和无效 Issue 可排除并注明理由。
- 创建但未合入 PR、`delegated`、`waiting_*`、`intermittent_waiting`、
  `ci_blocked`、`comment_failed` 均不算解决。
- 缺失数据时使用 `unknown` 或 `resolution_pending`，禁止推测。

## Issue 报告

终端和最终回复只给结论、实际处理 Issue、变更/PR、卡点与报告链接。若实际处理数为
0，直接写“本次未实际处理任何 Issue”，不要展开扫描到的自提、已有负责人或已有 PR
条目。指标只针对实际处理集合；扫描范围不足时不得冒充仓库整体指标。

## 内部卡点

内部报告必须保留：

- `waiting_environment`：目标和当前 CANN、源码 revision、SoC、架构的逐项对比。
- `ci_blocked`：PR、失败阶段、分类、重试/修复轮次和建议。
- 权限、基础设施、工具缺失和验证边界。

这些信息不得写入外部 Issue 评论。

## 清理结果

先按 `code-worktree.md` 串行执行受管 worktree 清理。报告必须列出：

- 已清理的 `group_id` 和原 worktree 路径。
- 因 `active`、`blocked`、工作区不干净或 manifest 校验失败而保留的 worktree。
- 保留项的下一步；不得用 `--force` 或直接删除目录规避拒绝。

清理 worktree 只删除临时工作目录及 Git 的 worktree 注册，不删除本地分支、远程分支、
commit、PR、复现证据或 manifest。manifest 作为本次运行的清理审计记录保留。

任务完成且附件分析结束后，只删除本流程生成的临时附件目录：

```bash
rm -rf .cannbot/gitcode-issue-handler/tmp/downloaded-attachments
```

复现证据、分类报告和需要交付的记录按报告保留；不得扩大删除范围。

## 完成条件

每个 `need_attention` Issue 都有真实状态、责任人或下一步；所有发布结果可核验；
本批次指标已按统一口径汇报；不得把等待、转交或未合入 PR 虚报为解决。此外必须同时
满足：

- 历史 `summary.md`、`run_state.json` 和 `latest.md` 都存在且可读。
- 报告 Issue 数量与 `issues_total` 一致，覆盖所有实际处理 Issue，且不包含
  `self_assigned`、纯 `/assign` 或其他仅观察到的 `no_attention` 明细。
- `--strict` 生成成功，报告已回读检查，最终回复提供历史报告路径。
