# 运行能力检查：按操作延迟门禁

`gitcode-issue-handler` 先区分 `policy_query` 与真实执行；不在启动时统一检查环境。真实执行只在下一步确实需要 API、Git、临时目录或提交身份时检查该操作的直接依赖。脚本属于本 Skill：

```bash
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" --checks <groups>
```

输出结构化路由 JSON；未选择的检查组不会被探测，不用 toolkit 的全量预检替代。

| 下一步操作 | 检查命令 | 检查项 |
| --- | --- | --- |
| 认证 GitCode API | `--checks api` | Token、curl、python3 |
| clone/fetch/diff/log/push 等 Git | `--checks git` | git |
| 创建或选择临时目录 | `--checks tmp` | 可写临时目录 |
| 创建或 amend commit | `--checks author` | git、git author |

连续操作马上需要多个能力时可组合检查，例如 `--checks api,git,tmp --work-dir "$WORK_DIR" --token-available`。仅离线规则说明、分类或本地分析不扩大检查组；只有创建临时目录才检查 `tmp`，只有 commit/amend 才检查 `author`。已验证且目标、工作目录、凭据未变的检查结果可复用。

`tmp` 按 `ISSUE_HANDLER_TMP_DIR` → `TMPDIR` → `/tmp` → `<work-dir>/.cannbot/gitcode-issue-handler/tmp` 选择可写目录；正常编排必须传精确 `--checks`；无参数模式仅用于完整自检。直接使用已确认可写的用户目录可跳过 `tmp`。仅查看已有 commit 或推送已有分支不检查当前 author。

## API Token

按当前消息 Token、`GITCODE_TOKEN`、目标仓库的 Git credential helper、用户输入的顺序获取。API 预检传目标仓库的 `--work-dir`；共享解析器非交互复用已保存的 HTTPS 凭据，报告只记 `source: git_credential` 等元数据，不把凭据写入文件或日志。只有 SSH 密钥且未存 HTTPS Token 时仍需提供 API Token。只在会话中提供而未设环境变量时传 `--token-available`，报告只记来源。凭据存在不等于目标 API 权限已通过，401/403 时核对目标权限并按 [Token 规则](../../gitcode-toolkit/references/token-config.md) 处理；不自动删除原 Git 凭据。若所有来源均不可用且后续需要认证写操作，输出 `needs_user: ["token"]` 后按 [runtime-state.md](runtime-state.md) 保存唯一输入并暂停整轮，不做 API 探测、Issue 拉取、知识刷新、代码诊断或测试读取；恢复时只重跑失败的 `api` 组。

## 其他失败路由

`git`、`tmp`、`author` 失败只阻断依赖它们的操作，可继续独立分析，并报告明确 `blockers`。按返回的 `action` 路由：`continue` 执行目标操作，`request_inputs` 一次汇总所需输入，`report_blockers` 报告本地阻断项，`request_inputs_and_report_blockers` 同时处理两者。恢复前先解决报告中的输入和 blocker。

报告的 `results`/`summary.total` 只统计实际选中的检查项：`api` 为 3 项，`git,tmp` 为 2 项，`author` 为 2 项。输出目录不是通用检查组，仅在即将落盘时确认父目录存在且可写。

`git` 检查不检查 Token、临时目录或 author；`author` 按项目 local → global → 用户输入读取身份，用户补充时只写工作目录 local 配置，不能猜测身份、修改全局配置或用 inline 参数绕过检查。

## 阶段约束

确定需要认证写操作而缺 Token 时，只汇总询问一次并停止本轮，保存恢复点；等待期间不继续 API 探测、真实 Issue 拉取或测试框架读取。其他能力在首个依赖操作前检查，失败只阻断依赖它的路径；详见 `runtime-capability-checks.md`。
