# 交付：提交与发布

## 读取时机与提交

步骤 6 质量门禁通过（或允许 `degraded_validation`），且 [delivery-confirmation.md](delivery-confirmation.md) 统一预览已批准后，执行步骤 7–8 前完整读取。命令均在当前组 manifest 的 `worktree_path` 执行。

按授权契约校验：`single + pr` 的精确暂存、commit、功能分支 push、PR、首次 CI 均须在当前批准摘要；`single + direct-push` 统一确认只覆盖暂存/commit；`approved_batch` 只覆盖精确 Issue、文件、operation 和交付模式；超出即 `invalidated` 并重做预览。`batch/interactive` 或未 approved 不得暂存/提交，保留未提交 worktree。author 检查缺失则返回该检查点，不能把分析/测试作废；邮箱仅不一致可警告并继续。

只暂存明确文件，禁止 `git add -A`/`git add .`：
```bash
git add <具体文件>
```
若本仓配置了 pre-commit（如 `.pre-commit-config.yaml`），按仓库约定启用当前 worktree 的提交钩子，确保本次 commit 实际触发并通过 pre-commit 检查，记录执行结果。检查失败先修复再提交；自动修改文件后核对 diff、重新暂存明确文件并重试。不得用 `--no-verify`、`SKIP` 或禁用钩子绕过检查。

无仓库约定时使用 Conventional Commit：
```text
<type>(<scope>): <description>

Closes #<IID1>, #<IID2>
```
`type` 为 `feat|fix|refactor|test|docs|chore`，多 Issue 组引用全部成员。

## PR 模式

1. 推送并回查功能分支：
   ```bash
   git push -u "${fork_remote}" HEAD:"${branch_name}"
   git ls-remote --heads "${fork_remote}" "${branch_name}"
   ```
2. 按 toolkit 的 PR 工作流创建 PR，传入授权模式、批准范围和分析后检查点证据；沿用模板，步骤 0 的目标分支为 base，fork head 遵循 toolkit 格式。Issue 请求本身不构成授权。
3. PR body 关联组内全部 Issue，写摘要、测试和降级边界。
4. 创建前查询并复用源/目标分支已有合法 PR；创建后记录 iid、URL、响应校验。Token 在首次 API 操作前由 `api` 能力检查；无 Token 时从同一会话恢复，仍无则返回 API 检查点，不报“PR 创建失败”。
5. 业务校验/证据记录完成后，用 `trigger_pr_pipeline.sh --repo <owner/repo> --pr <N>` 触发 compile 评论 CI。

禁止自动合并。偶发/基础设施失败准备一次重试；若未在原预览列出重试，更新预览并重新确认。明确代码失败最多两轮“修改→本地门禁→推送→重触发”。任何新增源码、commit、push、PR 正文或 CI 触发都超出旧批准，停止未执行写入、更新预览、重新确认。预算耗尽记 `ci_blocked`，保留 PR，继续独立 PR。

## 直接推送与生命周期

direct push 必须与步骤 0 的 `target_remote_branch` 完全一致。commit 后独立展示并确认 remote 名称/URL、目标分支、commit SHA、共享/保护提示和 fetch 后非快进结果：
```bash
git push <remote> HEAD:<branch>
git ls-remote --heads <remote> <branch>
```
拒绝/未回复时保留本地 commit，写 `direct_push_confirmation_status: pending | rejected` 和 `delivery_waiting_confirmation`，不改走 PR。保护规则拒绝则停止并报告，不换分支或模式。仅质量门禁通过后推送。

CI 仍可能需要本地修改时 worktree 为 `active`；PR/推送可核验且本地改动已提交并推送后为 `published`。即使 CI `ci_blocked`，远程分支和 PR 已保存全部提交仍可 `published`，另记 `ci_status: blocked`。推送/PR 失败为 `blocked` 并保留 worktree；无需变更为 `no_changes`。清理遵循 `code-worktree.md`。

完成后写入并进入 `delivery-reporting.md`：
```yaml
commit_sha:
delivery_mode: pr | direct-push
published_branch:
pr_url:
ci_status:
```
