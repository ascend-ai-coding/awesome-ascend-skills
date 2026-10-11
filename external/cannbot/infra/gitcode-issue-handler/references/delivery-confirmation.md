# 交付：分析后统一执行确认

## 读取时机与回复检查点

完成获取、分类和文字诊断后先做回复检查点；稳定复现、最小实现和验证后再做代码交付检查点。任何 GitCode 写入、暂存、提交、推送、PR 或 CI 前完整读取本文。能力输入和算子责任人请求是方案输入，不等于执行确认。首响规则以 `issue-intake.md` 和 [issue-comment-workflow.md](issue-comment-workflow.md) 为准；已有 PR/指派不能代替实质回复，自提豁免须同账号，已有响应不重发、不因补首响重新指派，`not_applicable_existing` 要有依据。

回复检查点保存准确 URL、适用的完整正文或 `/assign` 稿、operation ID、依赖和摘要，逐项文件路径按 batch-analysis；批次可另建 `response-preview.md` 索引。它只需文字诊断证据，不等未知 owner、修复或测试；与交付预览共用授权契约但分别留证。按 [automation.md](automation.md) 开启自动首响，或当前会话已明确授权时，保存草稿后直接执行，不再询问；缺失时一次展示当前就绪回复的完整预览并取得授权，可继续独立只读分析。批准后 POST/GET，成功才开放后续操作；失败或未批准停止依赖该回复的指派、关联 PR 或 worktree。有效自提 PR 的仅分配路径按 automation 免首响，预览准确 Issue/login 后由 auto-response 或当前明确分配授权执行。明示不评论按门禁豁免记录。纯答疑完成即结束本轮，无空的代码确认。

使用 `response_confirmation_status`、`response_preview_path/digest` 和每项的 `authorization_evidence`（兼容 `post_analysis_execution_confirmation`，并标 `phase: response`）。

## 统一执行范围

`single`、`batch` 均从 `interactive` 开始。除 [automation.md](automation.md) 开关授权的首响和临时指派外，宽泛的“处理/自动处理”只授权只读分析，以及在受管隔离 worktree 准备未提交、可丢弃的修改和验证证据；不授权写入。确认前可做能力/凭据检查、fetch、GET、知识检索、代码分析、环境检查、稳定复现、最小修改和本地验证，但不得暂存、commit，且修改不得进入原工作区。禁止未经对应授权的 Issue/评论/指派/标签/状态 POST/PUT/PATCH/DELETE，禁止暂存、commit、push、PR 和 CI；Token、工具审批、宽泛请求和旧会话批准不构成当前准确授权。

若只有只读结论且无待执行操作，设 `execution_confirmation_status: not_required`。预览含 commit 时先对相关 worktree 做 `author` 检查（纯回评、无改动、只读不检查）。

方案输入必须先确定：未知算子 owner 按 [operator-owner-candidates.md](operator-owner-candidates.md) 逐算子列候选账号和贡献，保留 `awaiting_offline_confirmation`；候选不是已指派/已解决。配置/会话授权的候选临时指派走 `automation.md`；正常转交仍需用户确认 login 或对确切 Issue 选 `direct` 后才能补预览；不能把未知 owner 纳入批准范围，不能默认 direct。其他改变操作清单的用户选择同样先收齐，方案与执行批准分开。

## 预览与批准

将完整预览持久化为 `.cannbot/gitcode-issue-handler/reports/<run_id>/execution-preview.md`，摘要写入运行状态。先列仓库、Issue、交付模式、批准边界，再列实际操作：

| 操作 | 必须展示 |
|---|---|
| 评论 | IID/URL、完整正文；多条逐条列出 |
| 指派 | IID/URL、登录名、`/assign` |
| 状态 | IID/URL、是否 reopen、当前/目标自定义状态 |
| 源码/验证 | 修复组、根因、策略、changed files/diff、风险；已运行及待运行验证和边界 |
| commit/push | 精确文件、message；remote、源分支、目标功能分支 |
| PR/CI | head/base、标题、完整正文；目标、触发方式/命令 |
| direct push | remote/目标分支，并标明 commit 后需独立确认 |

每项按 `runtime-state.md` 的 `external_operations` 写稳定 `operation_id`；`kind` 为 `issue_comment | issue_assignment | issue_state_change | prepared_source_change | commit | branch_push | pr_create | first_ci | direct_push`。 `body` 仅评论/PR 要完整正文。禁止 Token、环境变量、绝对路径、敏感信息和 `<PR URL>` 等未知占位符。需要新 PR URL 的回评须在 PR 创建后新增 operation、更新预览并重新确认；首次 CI 可依赖获批的 `pr_create` operation ID。

依赖顺序必须写 `depends_on`：回复回查→指派/状态/watch；责任人或提出者再回复先恢复`进行中`（必要时 reopen）并回查；commit→push→分支回查→PR→首次 CI。共享修复组须满足所有成员回复门禁；失败的后续项标 `skipped`，不改走未预览替代动作。

展示预览后复核已有会话授权，只对未覆盖项确认。批准全部记 `execution_confirmation_status: approved`；`batch` 才转 `approved_batch`，`single` 保持 `interactive` 并将证据传给各操作。批准子集删除/标记其余项并移除依赖项；要求调整则更新摘要并重新展示；拒绝/未回复为 `rejected | pending`，不执行待确认写入。批准至少记录：

```yaml
execution_confirmation_status: not_required | pending | approved | rejected | invalidated
execution_preview_path:
execution_preview_digest:
execution_approved_at:
authorization_source: default | explicit_user_approval
execution_confirmation_source: post_analysis_user_approval
authorization_scope:
  repository:
  issue_iids: []
  operation_ids: []
  delivery_mode: pr | direct-push | none
authorization_evidence:
  checkpoint: post_analysis_execution_confirmation
  preview_digest:
```

工具包从同一批准派生 operation 证据；已逐字展示且未变化的操作不重复询问，POST/GET、push/远端回查同理。

## 失效与 direct push

仓库/Issue 集合、评论或 PR 正文、owner、根因/策略、文件范围/commit、remote/分支、交付模式、PR head/base/标题/首次 CI 方式发生实质变化，或 CI 修复新增源码/commit/push/正文/触发时，设 `invalidated`，停止未执行写入，更新完整预览并重新确认。缩小范围或独立失败不扩大授权、不重问未变化项。

统一预览可披露 direct push，但不授权实际 push。commit 后按 [delivery-publish.md](delivery-publish.md) 展示 exact remote URL/名称、目标分支、commit SHA、共享/保护提示和 fetch 后非快进结果，再取得第二次明确确认；该确认不能被统一批准吞并。
