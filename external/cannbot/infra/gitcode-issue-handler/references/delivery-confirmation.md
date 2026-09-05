# 交付：分析后统一执行确认

## 读取时机

完成目标 Issue 或批次的获取、分类和文字诊断后执行本检查点；代码路径还要先完成受管
worktree 稳定复现、最小本地实现和验证。执行任何 GitCode 写入或发布动作前，必须完整
读取本文件。能力输入请求、算子责任人等方案输入请求不属于执行确认，不能替代本检查点。

## 核心规则

`single` 和 `batch` 都从 `interactive` 开始。用户最初要求“处理 Issue”“自动处理”或
“auto apply”之类宽泛目标，只授权 Agent 进行只读分析，并在 manifest 管理的隔离
worktree 中准备可丢弃的未提交修复与验证证据，**不构成执行批准**。必须让用户看到基于
实际 Issue、最终 diff 和验证结果的完整操作清单并在当前会话明确确认后，才能产生任何
GitCode/发布写入。

确认前允许：

- 在对应真实操作前执行最小能力检查、凭据验证、fetch、GitCode GET、知识检索和代码只读分析；
- 生成运行状态、缓存、报告草稿、冲突计划和执行预览；
- 在受管 worktree 中执行环境检查、稳定复现、最小源码修改和本地验证，以形成准确根因、
  changed files、diff 和测试结果；这些改动不得暂存或 commit，也不得出现在原始工作区。

确认前禁止：

- POST/PUT/PATCH/DELETE Issue、评论、指派、标签、状态或其他远端资源；
- 暂存、commit、push、创建 PR 或触发 CI；
- 把 Token、初始请求、工具权限审批或旧会话批准当作本检查点证据。

如果本轮只有只读结论且没有待执行操作，设置
`execution_confirmation_status: not_required`，不向用户制造空确认。

若预览将包含 commit，在生成预览前对每个相关 worktree 执行 `author` 能力检查；纯回评、
无修改或只读结论不检查 git author。该检查只证明提交身份可用，不授权暂存或 commit。

## 先补齐方案输入

操作目标必须确定后才能确认。若明确算子缺少责任人，在生成统一预览前，把本轮缺失映射
合并成一次方案输入请求，让用户提供 `<算子名>: <GitCode 登录名>` 或为确切 Issue 选择
`#<IID>: direct`。这次请求只决定拟执行方案，不授权评论、指派或发布动作。用户未回复时
保持 `pending_operator_owner`，不能生成含未知指派目标的执行批准，也不能默认 `direct`。

其他只能由用户确定且会改变操作清单的选择也先合并收齐。不要把方案输入与执行批准混成
一句含糊的“是否继续”；方案确定后仍需展示下述完整预览。

## 统一执行预览

把预览持久化到
`.cannbot/gitcode-issue-handler/reports/<run_id>/execution-preview.md`，计算内容摘要并写入
运行状态。预览先列仓库、Issue 清单、交付模式和批准边界，然后只列本轮实际适用的操作，
不使用空占位项。

至少按以下契约展示：

| 操作类型 | 必须展示的内容 |
| --- | --- |
| Issue 评论 | Issue IID/URL 和**完整正文**；多条评论逐条列出 |
| 指派 | Issue IID/URL、目标 GitCode 登录名，以及将发送的 `/assign` |
| Issue 状态 | Issue IID/URL、核心 state 是否 reopen、自定义状态的当前名称与目标名称；状态 ID 运行时解析 |
| 本地源码改动 | Issue/修复组、最终根因、变更策略、实际 changed files/diff 摘要、兼容风险 |
| 验证 | 已运行的相关测试/构建、结果与降级边界；另列批准后仍需执行的验证（如适用） |
| commit | 修复组、精确拟暂存文件和 commit message |
| 功能分支 push | remote、源分支和目标功能分支 |
| PR | head/base、标题和完整正文 |
| 首次 CI | 目标 PR、触发方式或命令；不把后续未知重试冒充为已批准 |
| direct push | remote 与目标分支，并标明“本次不授权；commit 形成后凭 SHA 独立确认” |

每项操作按 `runtime-state.md` 的 `external_operations` 契约写入稳定 `operation_id`。`kind`
使用 `issue_comment | issue_assignment | issue_state_change | prepared_source_change | commit |
branch_push | pr_create | first_ci | direct_push`；`body` 只用于评论或 PR 等必须展示完整正文的
动作，其他操作省略。预览和状态均不得包含 Token、环境变量、维护侧绝对路径或其他敏感
信息。

评论和 PR 正文禁止保留 `<PR URL>` 等发布后才能确定的占位符。为了保持正常路径只有一次
确认，优先把正文写成不依赖未知 ID 的最终版本，并用 `depends_on` 关联后续操作；若业务
确实要求把新 PR URL 回评到 Issue，则 PR 创建后把该评论作为新增 operation 更新预览，
明确确认后再发送。首次 CI 可以用被批准的 `pr_create` operation ID 表示目标。

用 `depends_on` 保留执行顺序，例如有效首响回查成功后才能指派；索要信息的评论回查成功后
才能切`挂起`并写 reporter watch；责任人指派和 assignee 回查成功后才能切`挂起`并写
assignee watch。提出者或责任人再回复后，恢复`进行中`（提出者在关闭 Issue 回复时还需
reopen）成功后再发布本轮跟进评论。commit 成功后才能 push，
功能分支回查成功后才能创建 PR，PR 创建成功后才能首次触发 CI。依赖失败时把后续项标记
`skipped`，不得擅自改走未预览的替代动作。

## 取得批准

展示预览后发起本轮一次统一执行确认，问题明确说明：批准只覆盖列出的仓库、Issue、正文、
owner、源码范围、分支和交付动作；未列出的操作不执行。只接受用户基于当前预览的明确
“批准执行”或清楚的批准子集：

- 批准全部：记录 `execution_confirmation_status: approved`。`batch` 随后才切换为
  `approved_batch`；`single` 保持 `interactive`，但把同一检查点证据传给每个已列操作。
- 批准子集：删除或标记未批准项，重新计算批准 scope；有依赖的后续动作一并移除，例如
  不批准 push 时不能创建依赖该分支的 PR。
- 要求调整：修改预览并重新展示；旧摘要失效。
- 拒绝或未回复：设置 `rejected | pending`，保留分析、未提交 worktree 和预览，不执行
  任何待确认写操作。用户明确要求丢弃时再按 worktree 安全清理契约处理。

批准状态至少记录：

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

工具包子流程需要单评论或交付证据时，从这一次批准为对应 `operation_id` 派生证据；不得
再次询问已逐字展示且未变化的评论、commit、功能分支 push、PR 或首次 CI。POST 后 GET
回查、push 后远端回查也不再询问。

## 失效与重新确认

发生以下任一实质变化时，把状态设为 `invalidated`，停止尚未执行的外部/发布动作，更新
完整预览并重新确认：

- 仓库或 Issue 集合扩大；评论或 PR 正文实质变化；owner 改变；
- 根因/策略改变，修改文件超出展示路径，或新增 commit；
- remote、分支、交付模式、PR head/base、标题或首次 CI 触发方式改变；
- CI 修复需要新的源码修改、push 或再次触发 CI。

缩小动作范围或某项执行失败不自动扩大授权，也不要求重问已批准且未变化的独立操作。
每个操作执行后记录结果和回查证据。

## direct push 例外

统一预览必须提前披露 direct push 目标，但本检查点不授权实际 push。commit 形成后，按
`delivery-publish.md` 独立展示 exact remote URL/名称、目标分支、commit SHA、共享/保护分支提示
和非快进检查结果，再取得第二次明确确认。该强确认不能被 `approved_batch` 或统一执行
批准吞并。
