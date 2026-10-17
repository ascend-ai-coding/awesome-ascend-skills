# GitCode Issue Handler 快速上手

让 Agent 帮你分析和回复单个 Issue，或批量找出当前仓库需要处理的问题。

## 1. 安装

准备好本地代码仓和 Python 3.10+；使用 `npx` 安装还需要 Node.js 20+。先进入要处理 Issue 的仓库：

```bash
cd /path/to/target-repository
```

选择你使用的客户端，安装一次即可。

### Claude Code

在当前仓库启动 Claude Code，依次执行，安装范围选 `local` 即可：

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install infra-skills@cannbot
/reload-plugins
```

### OpenCode

```bash
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit --tool opencode --level project
```

### Codex

```bash
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit --tool codex --level project
```

### 也可以让 Agent 安装

本地已有 `cannbot-skills` 源码仓时，在该仓库启动 Agent，发给它：

```text
请用本仓源码，把 gitcode-issue-handler 和 gitcode-toolkit 以 Codex 的 project 级安装到 /path/to/target-repository，保留已有配置，并引导我完成配置。
```

把目录和客户端名称换成自己的即可。

## 2. 配置

安装 Python 依赖：

```bash
python3 -m pip install "requests>=2.28.0" "PyYAML>=6.0"
```

在 GitCode 的“个人设置 → 访问令牌”中创建 Token，勾选 Issue 和 Pull Request 相关权限。在启动 Agent 的终端设置：

```bash
export GITCODE_TOKEN="你的 Token"
```

Token 不要写入配置文件。只做配置时可以先不设置 Token。

在目标仓库中启动 Agent，输入：

```text
请为当前仓库初始化并配置 gitcode-issue-handler，帮我确认处理范围和自动响应方式，配置完成后先不要处理 Issue。
```

助手会自动补齐配置，保留已有设置。引导会分别询问仓库、范围和响应方式。默认处理所有非 Roadmap 等纯规划类的 open Issue，响应方式默认“先审稿，确认后发送”。自定义范围分为需要处理、仅统计列出、忽略三段；自定义响应先询问自动响应，开启后才询问自动指派。

## 3. 开始使用

### 单个 Issue

把 Issue 地址发给 Agent：

```text
使用 gitcode-issue-handler 处理 https://gitcode.com/cann/ops-math/issues/1511
```

只想分析和准备回复时，在任务后加上“先拟稿，不发送，不改代码”。

### 批量拉取并处理

从目标仓库启动 Agent，输入：

```text
使用 gitcode-issue-handler 批量拉取并处理当前仓库需要关注的 Issue
```

助手会筛出责任范围内需要关注的问题，准备分析、回复稿和待办。保持默认设置时，先看草稿，再决定是否发送回复或指派负责人。

如果这次只想看处理计划，可以说：

```text
使用 gitcode-issue-handler 批量拉取当前仓库需要关注的 Issue，先列出处理计划，不要发送回复
```

以后继续处理时，仍用第一条批量提示词即可。处理报告在 `.cannbot/gitcode-issue-handler/reports/latest.md`。

全局安装、手工配置、更新卸载和常见问题见 [安装与配置指南](docs/installation-guide.md)。
