# 代码处理：修复组并行执行与 worktree 生命周期

目录：

- [读取时机](#读取时机)
- [基本模型](#基本模型)
- [运行标识与路径](#运行标识与路径)
- [冲突计划](#冲突计划)
- [创建-worktree](#创建-worktree)
- [并行执行](#并行执行)
- [生命周期标记](#生命周期标记)
- [安全清理](#安全清理)

## 读取时机

步骤 2e 完成代码修复分组后完整读取本文件。步骤 4 最终修改路径发生变化时重新执行
冲突计划；步骤 9 清理前再次读取“安全清理”。

## 基本模型

- 并行单位是修复组，不是单个 Issue。一个组可以包含同根因、同一修改方案的多个 Issue。
- 每组使用唯一分支和独立 worktree；原始目标仓库只用于 fetch、Issue 分析和管理
  worktree，不承载修复代码或构建产物。
- `single` 和 `batch/interactive` 都可创建受管 worktree，用于确认前的环境检查、稳定复现、
  未提交最小实现和本地验证。统一执行确认前禁止暂存、commit 和任何远端写入；
  `approved_batch` 只能由验证后的统一预览批准产生。
- Git worktree 只隔离文件和分支，不隔离 NPU、端口、进程、环境变量和仓库外缓存。
- 创建和清理 worktree 串行执行；同一执行波次的组内步骤 3–8 才并行。

## 运行标识与路径

生成不含敏感信息且只含字母、数字、点、下划线或连字符的唯一 `run_id`，例如
`20260810T103000Z-4821`。从 `tmp` 能力检查选出的、位于目标仓库之外且不是其祖先的可写临时目录
建立 worktree 根：

```text
<selected_tmp>/gitcode-issue-handler-worktrees/<run_id>/<group_id>
```

manifest 保存在目标仓库：

```text
.cannbot/gitcode-issue-handler/worktrees/<run_id>.json
```

不得把 worktree 创建在目标仓库原始工作目录之内。manifest 是所有权边界：未记录在
manifest 中的目录一律不清理。如果 `tmp` 能力检查最后回退到仓库内的
`.cannbot/gitcode-issue-handler/tmp`，该路径只能存放普通临时文件，不能作为 worktree 根；没有任何
仓库外安全候选时标记环境卡点，继续处理不需要代码 worktree 的其他 Issue。

组分支名必须包含完整 `run_id`。安全清理默认保留本地分支作为恢复和审计入口，运行标识
可防止后续运行发生分支重名。

## 冲突计划

为每组写入：

```yaml
group_id: issue-101
planned_paths:
  - src/module_a.py
exclusive_resources:
  - npu:0
```

准备包含 `groups` 数组的 JSON 后运行：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/manage_worktrees.py" plan \
  --groups-json .cannbot/gitcode-issue-handler/data/groups.json
```

调度器按以下规则生成 `waves`：

1. 两组修改同一路径，或一组目录包含另一组路径时冲突。
2. 两组声明同一 `exclusive_resources` 时冲突。
3. 任一组 `planned_paths` 为空时范围未知，保守地与所有组冲突。
4. 无冲突组进入同一 wave 并行；冲突组进入不同 wave 串行。

`direct-push` 模式下，所有推送到同一目标分支的组还必须声明相同的
`delivery:<remote>/<branch>` 独占资源。由于本工作流禁止 rebase、merge 和 force push，
这些组必须串行，并在前一组推送成功后重新 fetch，再从更新后的远程目标 commit 创建下一
组 worktree。不得预先从同一旧 commit 创建多个 direct-push worktree。默认 `pr` 模式下
不同功能分支不需要该资源锁。

将输出的冲突原因、`execution_wave` 写回运行状态。步骤 4 确认最终根因后用最终
`change_locations` 更新 `planned_paths` 并重新计划。若新计划改变波次，只调整尚未开始
修改的组；不得让两个路径重叠的组继续并行写代码。

## 创建 worktree

PR 模式从步骤 0 固定的同一 `base_ref`/`base_commit` 创建全部组 worktree。为避免共享
Git 元数据锁竞争，按组串行执行：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/manage_worktrees.py" create \
  --repo-root "$TARGET_REPO_ROOT" \
  --manifest "$TARGET_REPO_ROOT/.cannbot/gitcode-issue-handler/worktrees/<run_id>.json" \
  --worktree-root "<selected_tmp>/gitcode-issue-handler-worktrees" \
  --run-id "<run_id>" \
  --group-id "<group_id>" \
  --branch "<proposed_branch>" \
  --base-ref "<base_commit>" \
  --wave "<execution_wave>" \
  --planned-path "<repo-relative-path>" \
  --exclusive-resource "<resource-id>"
```

`--planned-path` 和 `--exclusive-resource` 可重复。脚本拒绝绝对路径、`..`、`.git`、重复
组、已有分支和非空目标目录。创建完成后，从 manifest 读取确切 `worktree_path`。

direct-push 多组模式不预创建全部 worktree：按 wave 串行，在每组开始前 fetch 并验证
远程目标，把最新目标 commit 作为该组 `--base-ref`。前一组未成功推送时不得创建或执行
下一组。

## 并行执行

按 wave 从小到大执行；只有当前 wave 全部组达到终止或等待状态后才启动下一 wave。
运行平台支持并行 worker/agent 时，每个组分配一个独立 worker，并只传递：

- 当前组 Issue、根因证据和方案。
- manifest 中的 `worktree_path`、分支和发布目标。
- 当前组 `planned_paths`、独占资源和质量门禁。

worker 的所有 shell 命令都必须以本组 `worktree_path` 为工作目录，不得修改其他组或原始
工作区。并发度不得超过可用 worker 数量和资源容量；并行能力不足时在同一 wave 内有界
排队，不改变冲突判定。

确认前 worker 只能推进到步骤 6：保留未暂存 diff 和验证结果，然后退出或等待主流程的
统一执行检查点。获得对应 operation IDs 的证据后才恢复暂存、commit 和远端发布。用户
拒绝或尚未回复时保持 `active` 并保留 worktree；不得把未提交修复标为
`cancelled_clean` 以便清理。

## 生命周期标记

发布和验证结束后记录 worktree 生命周期：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/manage_worktrees.py" mark \
  --manifest "$TARGET_REPO_ROOT/.cannbot/gitcode-issue-handler/worktrees/<run_id>.json" \
  --group-id "<group_id>" \
  --status published \
  --commit-sha "<sha>" \
  --published-ref "<remote>/<branch>" \
  --pr-url "<url>"
```

状态含义：

| 状态 | 含义 | 自动清理 |
| --- | --- | :---: |
| `active` | 仍在复现、修改、验证或等待 CI 修复 | 否 |
| `blocked` | 存在未发布提交或需要继续调查 | 否 |
| `published` | 所有本地提交已推送，发布结果可核验 | 是，前提是工作区干净 |
| `no_changes` | 已确认无需代码修改 | 是，前提是工作区干净 |
| `cancelled_clean` | 本组已终止且没有需要保留的本地改动 | 是，前提是工作区干净 |
| `cleaned` | worktree 已安全移除 | 已完成 |

不得为了清理把有未提交修改的组标为 `cancelled_clean`。`handling_status`、`ci_status` 与
worktree 生命周期独立，例如远程 PR 已保存全部提交但 CI 预算耗尽时，可记录
`lifecycle_status: published` 和 `ci_status: blocked`。

## 安全清理

全部 worker 退出后，由主流程串行执行一次：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/manage_worktrees.py" cleanup \
  --manifest "$TARGET_REPO_ROOT/.cannbot/gitcode-issue-handler/worktrees/<run_id>.json"
```

也可通过重复的 `--group-id` 只清理指定组。脚本只有同时满足以下条件才执行不带
`--force` 的 `git worktree remove`：

1. worktree 属于 manifest 记录的仓库和本次 `worktree_root`。
2. 生命周期为 `published`、`no_changes` 或 `cancelled_clean`。
3. worktree 仍由 Git 注册。
4. `git status --porcelain` 为空。

否则返回 `skipped` 并保留目录。禁止直接 `rm -rf`、`git worktree remove --force` 或扩大
清理范围。清理不会删除本地分支、远程分支、commit、PR 和 manifest；本地分支可供恢复
或审计，是否删除属于本工作流之外的显式维护动作。

查看保留项：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/manage_worktrees.py" inspect \
  --manifest "$TARGET_REPO_ROOT/.cannbot/gitcode-issue-handler/worktrees/<run_id>.json"
git -C "<retained-worktree-path>" status --short
```

保留项由后续运行继续处理。只有确认本地内容已提交/发布或不再需要后，才更新终态并再次
执行 cleanup。
