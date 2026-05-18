# GitCode Code Reviewer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kali20gakki/gitcode-code-reviewer/blob/main/LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-kali20gakki%2Fgitcode--code--reviewer-181717?logo=github)](https://github.com/kali20gakki/gitcode-code-reviewer)
[![Skills](https://img.shields.io/badge/Works%20with-Codex%20%7C%20OpenCode%20%7C%20Claude%20Code-0A7B83)](https://github.com/kali20gakki/gitcode-code-reviewer)

一个面向 Codex、OpenCode、Claude Code 的 GitCode PR 审查 Skill。

给一个 GitCode Pull Request 链接，它就会拉取 PR 标题、描述、变更文件和 diff，结合仓库上下文生成更可靠的代码审查摘要，并可按需发布逐行评论或 Review。

适合希望把 GitCode PR 审查流程接入 Agent 的个人开发者和团队。

## 快速上手

### 1. 安装 Skill

#### Codex

```bash
npx skills add kali20gakki/gitcode-code-reviewer --skill gitcode-code-reviewer -a codex -g -y
```

#### OpenCode

```bash
npx skills add kali20gakki/gitcode-code-reviewer --skill gitcode-code-reviewer -a opencode -g -y
```

#### Claude Code

```bash
npx skills add kali20gakki/gitcode-code-reviewer --skill gitcode-code-reviewer -a claude-code -g -y
```

### 2. 配置 GitCode Token

推荐直接运行：

```bash
python3 scripts/setup_token.py
```

或者手动配置：

```bash
git config --global gitcode.token <your-token>
```

也可以使用环境变量：

```bash
export GITCODE_TOKEN=<your-token>
```

### 3. 直接开始用

在 Codex、OpenCode 或 Claude Code 中输入：

```text
请 review 这个 GitCode PR，并先给我展示审查摘要：
https://gitcode.com/owner/repo/pull/123
```

或者：

```text
帮我检查这个 GitCode PR 的回归风险，重点关注兼容性、边界条件和测试覆盖：
https://gitcode.com/owner/repo/pull/123
```

如果你希望最终把评论发回 PR，可以继续对 agent 说：

```text
把这些发现整理成逐行评论并发布到 PR。
```

## 这个 Skill 能做什么

- 拉取 GitCode PR 的元数据、文件列表和 diff
- 结合本地仓库上下文审查改动，而不是只看 patch
- 输出结构化审查摘要
- 按需发布普通 Review 或逐行评论到 GitCode PR

## 安装说明

### 推荐方式：使用 skills CLI

先检查仓库是否能被识别：

```bash
npx skills add kali20gakki/gitcode-code-reviewer --list
```

如果输出里能看到 `gitcode-code-reviewer`，说明可以正常安装。

### 手动安装

如果你不使用 `skills` CLI，也可以把仓库复制或软链接到对应的 skills 目录。

先拉取仓库：

```bash
git clone https://github.com/kali20gakki/gitcode-code-reviewer.git
cd gitcode-code-reviewer
```

#### Codex

```bash
mkdir -p ~/.codex/skills
ln -s "$(pwd)" ~/.codex/skills/gitcode-code-reviewer
```

#### OpenCode

```bash
mkdir -p ~/.config/opencode/skills
ln -s "$(pwd)" ~/.config/opencode/skills/gitcode-code-reviewer
```

#### Claude Code

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)" ~/.claude/skills/gitcode-code-reviewer
```

## 更新 Skill

### 使用 skills CLI 更新

先检查是否有可用更新：

```bash
npx skills check
```

更新已安装的 Skills：

```bash
npx skills update
```

如果你只想重新安装这个仓库里的最新版，也可以再次执行安装命令：

```bash
npx skills add kali20gakki/gitcode-code-reviewer --skill gitcode-code-reviewer -a codex -g -y
```

把上面的 `-a codex` 替换成你实际使用的 Agent 即可，例如 `opencode` 或 `claude-code`。

### 手动安装时如何更新

如果你是通过 `git clone` 加软链接的方式安装，进入仓库后执行：

```bash
git pull
```

如果你不是软链接，而是直接复制文件到 skills 目录，更新后需要重新复制一遍最新文件。

## Token 权限

你的 GitCode Token 至少需要这些权限：

- `pull_requests`
- `issues`
- `projects`

## 命令行脚本

如果你想单独调试，也可以直接使用仓库内脚本：

```bash
python3 scripts/fetch_pr_info.py --help
python3 scripts/post_pr_comment.py --help
python3 scripts/setup_token.py --help
```

## 常见用法

```text
review 这个 GitCode PR：
https://gitcode.com/owner/repo/pull/123
```

```text
检查这个 PR 有没有明显的回归风险：
https://gitcode.com/owner/repo/pull/123
```

```text
先给我审查摘要，确认后再发评论到 PR：
https://gitcode.com/owner/repo/pull/123
```

## 注意事项

- 不要把 GitCode Token 提交到仓库里
- 发布逐行评论前，建议先让 agent 展示审查摘要
- 如果 PR 更新过，建议重新抓取后再发布评论

## License

MIT
