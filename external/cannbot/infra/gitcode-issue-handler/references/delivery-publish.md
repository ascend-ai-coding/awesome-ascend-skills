# 交付：提交与发布

## 读取时机

步骤 6 的质量门禁通过或形成允许发布的 `degraded_validation`，且
`delivery-confirmation.md` 的统一执行预览已获得批准后，执行步骤 7–8 前完整读取本文件。

## 步骤 7：提交

以下命令全部在当前组 manifest 记录的 `worktree_path` 执行。

提交前按本 Skill 的 [authorization-contract.md](authorization-contract.md) 校验分析后统一
执行授权：

- `single + pr`：精确暂存、commit、功能分支 push、PR 创建和首次 CI 的每项 operation ID
  都必须属于 `execution_confirmation_status: approved` 的当前预览摘要。把同一次确认派生
  的证据传给 toolkit，不在这里再次询问。
- `single + direct-push`：统一确认只覆盖精确暂存和 commit；实际 direct push 即使已在
  预览中披露，仍按步骤 8B 独立确认。
- `approved_batch`：它只能由分析后统一确认产生。校验本组 Issue、精确文件、operation
  IDs 和交付模式均在批准作用域内后继续；超出时把确认标记为 `invalidated`，更新完整预览，
  不得把旧批准扩展到新文件或新 Issue。
- `batch/interactive` 或 `execution_confirmation_status != approved`：不得暂存或提交，保留
  受管 worktree 中的未提交改动和预览。

只暂存本组明确修改的文件，禁止 `git add -A` 和 `git add .`：

```bash
git add <具体文件>
```

使用目标仓库约定；没有额外约定时使用 Conventional Commits：

```text
<type>(<scope>): <description>

Closes #<IID1>, #<IID2>
```

`type` 可为 `feat`、`fix`、`refactor`、`test`、`docs`、`chore`。多 Issue 组引用全部成员。
git author 应已在生成包含 commit 的统一预览前通过 `author` 能力检查；此处缺失时返回
该 worktree 的 author 检查点，用户补充后从提交继续，不得把前面的分析、修改和测试作废。
仅绑定邮箱不一致时记录警告并继续。

## 步骤 8A：PR 模式

1. 推送功能分支：

   ```bash
   git push -u "${fork_remote}" HEAD:"${branch_name}"
   git ls-remote --heads "${fork_remote}" "${branch_name}"
   ```

2. 按 `gitcode-toolkit` 的 `references/pr-creation-workflow.md` 创建 PR。Issue 处理请求
   本身不构成授权；传入 `authorization_mode`、批准作用域和分析后统一检查点证据。已有
   合法证据时 toolkit 不重复询问。沿用目标仓库模板，并把步骤 0 的目标分支作为 base。
   fork 创建 PR 时使用 toolkit 规定的 head 格式。
3. PR body 关联组内全部 Issue，写明修改摘要、测试结果和降级验证边界。
4. 创建前按源分支和目标分支查询并复用已有 PR；创建后记录 PR iid、URL 和响应校验
   结果。Token 应已在首次认证 API 操作前通过 `api` 能力检查；这里拿不到 Token 时先从
   主编排恢复同一会话凭据，仍未就绪才返回 API 检查点，不能汇报为“PR 创建失败”。
5. `interactive` / `approved_batch` 的业务校验和证据记录由本 Skill 完成后，使用
   `gitcode-toolkit/scripts/trigger_pr_pipeline.sh --repo <owner/repo> --pr <N>` 提交
   `compile` 评论触发 CI。toolkit 不解释或保存 handler 的授权上下文。

CI 失败时：

- 疑似偶发或基础设施失败：准备一次重试；如果原统一预览没有逐项列出该再次触发操作，
  先更新完整预览并重新确认。
- 明确代码失败：最多执行 2 轮“修改 → 本地门禁 → 推送 → 重触发”。
- CI 修复产生新的源码改动、commit、push、PR 正文变化或 CI 触发时，旧批准不覆盖这些
  新操作；`single` 和 `approved_batch` 都停止尚未执行写入，更新完整预览后重新确认。
- 预算耗尽：标记 `ci_blocked`，保留 PR，只在内部报告失败摘要，继续其他 PR。

禁止自动合并 PR。

## 步骤 8B：直接推送

目标必须与步骤 0 的 `target_remote_branch` 完全一致，并可拆成 remote 和 branch：

commit 形成后，必须独立展示并确认：remote 名称及 URL、目标分支、将推送的 commit
SHA、共享/保护分支提示，以及 `git fetch` 后的非快进检查结果。该确认不能被 single
统一执行确认或 `approved_batch` 替代。用户拒绝或未回复时保留本地 commit，
设置 `direct_push_confirmation_status: pending | rejected` 和
`delivery_waiting_confirmation`，不自动改走 PR。

```bash
git push <remote> HEAD:<branch>
git ls-remote --heads <remote> <branch>
```

仅在质量门禁满足后推送。共享或保护分支也按既定模式执行；若保护规则拒绝，停止并
报告，不要回退为 PR，也不要改推其他分支。

### worktree 生命周期

- CI 仍可能要求本地修改时保持 `active`。
- PR/直接推送已经可核验，且本地所有改动均已提交并推送后，标记 `published`。即使
  CI 最终为 `ci_blocked`，只要远程分支和 PR 已保存全部提交，也可标记 `published`，
  同时单独保留 `ci_status: blocked`。
- 推送或 PR 创建失败时标记 `blocked` 并保留 worktree，不得清理后丢失待发布提交。
- 本组确认无需变更时标记 `no_changes`。

状态写入和清理命令以 `code-worktree.md` 为准。

## 输出

```yaml
commit_sha:
delivery_mode: pr | direct-push
published_branch:
pr_url:
ci_status:
```

完成后进入 `delivery-reporting.md`。
