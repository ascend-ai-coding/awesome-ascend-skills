# 运行能力检查：按操作延迟门禁

`gitcode-issue-handler` 不在启动时统一检查全部环境。先区分 `policy_query` 与真实执行；
`policy_query` 直接回答，不运行本脚本。真实执行也只在下一步确实需要 API、Git、临时目录
或提交身份时检查该操作的直接依赖。

> 脚本属于本 Skill：`bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" --checks <groups>`。
> 输出结构化路由 JSON；未选择的检查组不会被探测。不要调用 toolkit 的全量预检脚本代替。

## 门禁选择

| 下一步真实操作 | 命令 | 实际检查项 |
|----------------|------|------------|
| 调用认证 GitCode API | `--checks api` | Token、curl、python3 |
| clone、fetch、diff、log、push 等 Git 操作 | `--checks git` | git |
| 创建或选择临时工作目录 | `--checks tmp` | 可写临时目录 |
| 创建或 amend commit | `--checks author` | git、git author |

连续的同一执行阶段马上需要多个能力时组合检查，例如：

```bash
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" \
  --checks api,git,tmp \
  --work-dir "$WORK_DIR" \
  --token-available
```

遵循以下边界：

- `policy_query`、纯规则说明和完全离线输入：不运行预检；`batch/no_attention` 的分类本身不扩大检查组。
- 下一步只访问 API：只检查 `api`，不询问 git author、不测试临时目录。
- 下一步只分析已有本地仓库：只检查 `git`，不询问 Token。
- 只有准备创建临时目录时才检查 `tmp`。若直接使用已确认可写的用户目录，可不检查。
- 只有准备创建或 amend commit 时才检查 `author`。仅查看已有 commit 或推送已有分支
  不需要检查当前 git author；作者信息已经写入 commit，事后检查配置不能改变它。
- 不得因为后续“可能”会评论、改代码或提交 PR 而提前扩大检查组。

一般能力门禁失败只阻断依赖它的操作，可继续不依赖失败项的分析；但已确定认证写操作后
缺 Token 是下文规定的整轮暂停例外。恢复操作前先解决报告中的 `needs_user` 和 `blockers`。

## 检查组规则

### `api`：认证 API

在首次发送需要认证的 GitCode API 请求之前检查。Token 按以下顺序获取：

1. 用户当前消息直接提供的 Token；
2. 环境变量 `GITCODE_TOKEN`；
3. 两者都没有时询问用户。

若 Token 已保存在当前会话而非环境变量，传 `--token-available`。Token 只在当前会话
使用，不写入文件或日志；报告只记录来源，不输出明文。401/403 时重新进入 `api`
门禁并验证目标仓库所需权限，不盲目重试。

#### 缺 Token：一次询问并暂停

当流程已经确定后续存在评论、指派、状态修改、PR 或 CI 等认证 GitCode 写操作，且用户
当前消息与 `GITCODE_TOKEN` 均没有 Token 时，`--checks api` 返回的
`needs_user: ["token"]` 是本轮终止信号：

1. 在运行状态中写入 `overall_status: waiting_for_input`、
   `capability_checks.api: waiting_for_input`，并创建唯一的
   `input_id: gitcode_token`；记录 `request_count: 1` 和准确的 `resume_from`，不保存 Token。
2. 只汇总询问一次 Token，然后立即停止本轮。此断点之后不得继续任何 API 探测、真实
   Issue 拉取、知识刷新、代码诊断或测试框架读取，也不得以“先做不依赖 Token 的分析”
   为由继续推进。
3. 未解决的 `input_id: gitcode_token` 已存在时不得重复询问，也不得重复判断是否需要
   Token；保持 `waiting_for_input` 并结束。
4. 用户后续提供 Token 时，在当前会话用 `--token-available` 重跑 `api` 检查；通过后把
   输入标为 `resolved`，状态恢复为 `running` / `ready`，从保存的 `resume_from` 继续。
   不重新初始化运行、不重新拉取已保存的 Issue，也不重复询问。

若尚未确定任何认证 API 操作，则保持 `not_started`，不得为了预判未来路径而索取 Token。
Token 就绪仍不代表写操作已获业务授权。

### `git`：Git 操作

在第一个 Git 命令之前检查 `git`。该组不检查 Token、临时目录或作者身份。push 等
写操作还必须单独完成 remote 目标确认和写操作授权；工具存在不代表用户已授权。

### `tmp`：临时目录

在创建临时工作目录之前检查。传 `--work-dir` 后按以下顺序选择可写目录：

1. `ISSUE_HANDLER_TMP_DIR`；
2. `TMPDIR`；
3. `/tmp`；
4. `<work-dir>/.cannbot/gitcode-issue-handler/tmp`。

无参数模式用于本 Skill 的完整自检；正常编排必须传入精确的 `--checks`。

### `author`：创建 commit

紧邻 `git commit` 或 `git commit --amend` 前检查。`author` 自动包含 `git`，无需再组合
`git,author`。

读取顺序为 local（项目级）→ global → 询问用户：

```bash
NAME=$(git -C "$WORK_DIR" config --local user.name 2>/dev/null)
EMAIL=$(git -C "$WORK_DIR" config --local user.email 2>/dev/null)
if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  NAME=$(git -C "$WORK_DIR" config --global user.name 2>/dev/null)
  EMAIL=$(git -C "$WORK_DIR" config --global user.email 2>/dev/null)
fi
```

用户补充身份时只写工作目录 local 配置，禁止修改 `~/.gitconfig`：

```bash
git -C "$WORK_DIR" config user.name "$NAME"
git -C "$WORK_DIR" config user.email "$EMAIL"
```

不得从 fork URL、用户名等信息猜测身份，也不得用 `--author`、
`-c user.name=...` 等 inline 参数绕过门禁。

## 输出目录检查

输出目录不属于本脚本的通用预检组。仅在即将落盘前确认目标路径的父目录存在且可写；可以安全
创建时使用 `mkdir -p`，无权限时再询问替代路径。纯分析或无需落盘的流程跳过此项。

## 报告与恢复

报告中的 `results` 和 `summary.total` 只统计选中的实际检查项。例如：

- `--checks api`：3 项（token、curl、python3）；
- `--checks git,tmp`：2 项（git、tmp）；
- `--checks author`：2 项（git、git_author）。

按 `action` 路由：

| `action` | 处理 |
|----------|------|
| `continue` | 执行紧随其后的目标操作 |
| `request_inputs` | 一次汇总询问本次门禁缺少的用户输入 |
| `report_blockers` | 报告需修复的本地环境项 |
| `request_inputs_and_report_blockers` | 同时展示用户输入和本地 blocker |

恢复后只重跑失败的检查组，不重新执行已完成的业务分析。同一会话可复用已验证且未变化
的 Token 或环境检查结果；目标、工作目录或凭据发生变化时重新检查对应组。

## 反模式

- 启动 handler 就运行无参数完整预检；或让 `policy_query` 进入预检。
- 回答规则问题或完成 `no_attention` 路由前询问 Token、git author。
- 为 API-only 操作检查 Git、临时目录和提交身份。
- 为只读本地分析检查 Token。
- 把 Token 明文写入日志、命令输出或持久化配置。
- 修改全局 git author，或根据 fork owner 猜测 author。
- 把环境就绪等同于外部写操作已获授权。
