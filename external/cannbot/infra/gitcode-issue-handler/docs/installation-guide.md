# GitCode Issue Handler 安装与配置指南

本文档说明 `gitcode-issue-handler` 的安装依赖、常用触发方式、可选批量配置、更新和卸载，
并提供可直接复制的使用示例。仓库级安装方式总览见
[CANNBot Skills 安装指南](../../../docs/installation-guide.md)，所有 Skill 的使用总览见
[CANNBot Skills 使用样例](../../../docs/skills-usage.md)。

## 按客户端安装

`gitcode-issue-handler` 依赖 `gitcode-toolkit`，两项必须安装到同一个客户端和同一级别。
Claude Code 的 `infra-skills` 包已包含两项；OpenCode 和 Codex 命令则显式安装两项。

本文的“当前项目”或 `project` 均指要处理 Issue 的**目标代码仓**，不是
`cannbot-skills` 源码仓。执行项目级命令前必须先 `cd` 到目标仓根目录；
`global` / `user` 级命令可在任意目录执行。Claude Code 的 `local` 与 `project`
都绑定当前目标仓。

| 客户端 | 首选方式 | 首选安装命令 |
|---|---|---|
| Claude Code | Plugin Marketplace | `/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git`，再执行 `/plugin install infra-skills@cannbot` |
| OpenCode | install-helper | `npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit --tool opencode --level project` |
| Codex | skills CLI | `npx skills add https://gitcode.com/cann/cannbot-skills.git --skill gitcode-issue-handler --skill gitcode-toolkit --agent codex` |

### Claude Code

Marketplace 是 Claude Code 的插件市场机制。`infra-skills@cannbot` 会一次安装 handler 和
toolkit；在 Claude Code 中执行：

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install infra-skills@cannbot
/reload-plugins
```

`/plugin install` 默认使用 Claude Code 的 user scope；如需 project/local scope，可通过
`/plugin` 界面选择。选择 project/local 前，需先从目标仓根目录启动 Claude Code：

```bash
cd /path/to/target-repository
claude
```

备选方案是用 install-helper 安装独立 Skills：

```bash
# 目标仓项目级（写入 <target-repository>/.claude/）
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool claude --level project

# 当前用户全局（写入 ~/.claude/，可在任意目录执行）
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool claude --level global
```

该备选要求 Node.js 18+，安装后可使用自然语言触发。

### OpenCode

推荐通过 `npx` 直接运行 install-helper，无需预先全局安装，但要求 Node.js 18+。以下命令
会把 handler 和 toolkit 安装为两个独立 Skill：

```bash
# 目标仓项目级（写入 <target-repository>/.opencode/）
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level project

# 当前用户全局（写入 ~/.config/opencode/，可在任意目录执行）
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level global
```

只在当前仓库使用时选 `project`；需要跨仓使用时选 `global`。OpenCode 是否自动把已安装
Skill 暴露为 slash 入口取决于版本：支持该行为的版本可使用 `/gitcode-issue-handler`；
其他版本使用自然语言 `使用 gitcode-issue-handler <任务描述>`。

### Codex

推荐使用 [skills CLI](https://github.com/vercel-labs/skills)；默认安装到当前项目，增加
`--global` 后可供该用户的所有项目使用：

```bash
# 目标仓项目级
cd /path/to/target-repository
npx skills add https://gitcode.com/cann/cannbot-skills.git \
  --skill gitcode-issue-handler --skill gitcode-toolkit --agent codex

# 当前用户全局（可在任意目录执行）
npx skills add https://gitcode.com/cann/cannbot-skills.git \
  --skill gitcode-issue-handler --skill gitcode-toolkit --agent codex --global
```

安装后在 Codex 中使用自然语言触发。

## Python 依赖

脚本需要 Python 3、`requests` 和 PyYAML。无法导入 `requests` 或 `yaml` 时，使用完整仓库
checkout 执行：

```bash
python3 -m pip install -r \
  /path/to/cannbot-skills/infra/gitcode-issue-handler/requirements.txt
```

安装机制不会保存 `GITCODE_TOKEN`，也不会替换目标仓库的 `AGENTS.md` / `CLAUDE.md`。
运行时不在启动阶段统一检查全部环境，而是在相关操作前按需检查：首次调用 GitCode API 前
检查 API 客户端和 Token；首次同步仓库、读取 Git 历史或创建 worktree 前检查 Git、目标仓库、
remote 和所需工作目录；仅在准备 commit 前检查 git author；仅当代码任务确实需要编译、运行、
复现或测试时，才在这些操作前检查 CANN 版本与环境一致性。纯规则咨询不做环境预检；某项
缺失只阻塞依赖它的操作，不阻塞无关分析。Token 只在当前会话使用。

## 使用示例

先从目标代码仓根目录启动客户端；下文的“当前仓库”都是该目标仓。不要在
`cannbot-skills` 源码仓中启动这些任务，除非它本身就是待处理仓库。

将下表中的“任务描述”代入当前客户端对应的调用格式：

- Claude Code 通过 Marketplace 安装后，Skill 命令带有 Plugin namespace：

```text
/infra-skills:gitcode-issue-handler
<任务描述>
```

- 支持自动暴露 Skill slash 入口的 OpenCode 版本可使用：

```text
/gitcode-issue-handler
<任务描述>
```

- Codex、未自动暴露 slash 入口的 OpenCode，以及 Claude Code 的 install-helper 备选安装，
  均使用通用自然语言格式：

```text
使用 gitcode-issue-handler <任务描述>
```

| 场景 | 任务描述 |
|---|---|
| 显式单 Issue 完整处理 | `完整处理 https://gitcode.com/cann/ops-math/issues/1511` |
| 只回复或答疑，不修改代码 | `只回复 https://gitcode.com/cann/ops-math/issues/456，不改代码` |
| 当前仓库批量分诊和处理 | `分诊并处理当前仓库需要关注的 Issue` |
| 继续处理挂起后的新回复 | `继续处理当前仓库中已挂起且提出者或责任人有新回复的 Issue` |
| 咨询 Issue 自动闭环 dry-run | `预览当前仓库已答复且长期无响应的咨询 Issue，不要实际关闭` |

当前仓库批量、挂起 follow-up 和自动闭环场景应在目标仓库目录中触发。

## 单 Issue、批量与自动闭环配置

显式单 Issue 模式不需要 YAML，也不要求预建
`.cannbot/gitcode-issue-handler/`。

批量模式以当前启动目录为目标仓库。流程先从 remote 推导仓库；分类器也可通过
`classify_issues.py --repo owner/repo` 显式指定，获取器则接收由仓库标识派生出的
`fetch_issues.py --url https://gitcode.com/owner/repo`。因此 `classify_config.yaml` 不是安装
前置条件。希望固定批量策略时，可从已加载 Skill 的 `assets/` 复制模板到统一配置目录：

```bash
# 配置和运行数据应写入目标仓
cd /path/to/target-repository
HANDLER_ROOT=/path/to/cannbot-skills/infra/gitcode-issue-handler
mkdir -p .cannbot/gitcode-issue-handler/config
cp -n "$HANDLER_ROOT/assets/classify_config.yaml.example" \
  .cannbot/gitcode-issue-handler/config/classify_config.yaml
cp -n "$HANDLER_ROOT/assets/operator_owners.yaml.example" \
  .cannbot/gitcode-issue-handler/config/operator_owners.yaml
```

- `classify_config.yaml`：可固定 `repo`、增量状态、follow-up watch、缓存和自动闭环参数；
- `operator_owners.yaml`：可选的算子到 GitCode 登录名映射；缺失时流程按责任人门禁请求
  补充，不会静默改为 Agent 自行处理；
- `.cannbot/gitcode-issue-handler/`：统一存放 `config/`、`data/`、`reports/`、
  `logs/`、`cache/`、`images/`、`repro/`、`worktrees/` 和 `tmp/`；流程会在当前
  仓库的 `.git/info/exclude` 中只追加 `/.cannbot/gitcode-issue-handler/`，不忽略
  其他 `.cannbot/` 内容，不改写 `.gitignore`。

新配置优先于仓根旧配置。仅当新配置不存在时，脚本才只读回退到仓根的
`classify_config.yaml` 或 `operator_owners.yaml`；新生成的默认数据、缓存和报告始终写入
统一目录。旧 `issue_analysis_data/` 不会自动移动或删除，自定义 CLI 路径也不会被改写。

`auto-close-stale` 使用 `classify_config.yaml` 中的 `auto_close` 策略；只有实际启用该维护
路径时才需要从模板准备并复核这些设置。它默认 dry-run，仍须显式 `--apply` 才写入。
日常批量获取默认维护 `data/followup-watch.json`：首次回看近期全部状态的更新，之后用游标
增量扫描，并定点刷新等待提出者或责任人的 Issue。该文件按仓库隔离，不应跨仓库复用。

配置和运行数据始终属于目标仓库，不写入全局工具配置根。多仓场景由各仓分别维护配置和
运行证据。

## 更新与卸载

使用与安装时相同的客户端和级别。

### Claude Code

Marketplace 安装：

```text
/plugin marketplace update cannbot
/plugin update infra-skills@cannbot
/reload-plugins

/plugin uninstall infra-skills@cannbot
```

上述命令对应默认的 user scope。如果原安装使用 project/local scope，先在目标仓根
目录启动 Claude Code，再在 `/plugin` 的 Installed 页签更新或卸载对应 scope 的实例。

如果使用 install-helper 备选方案，则按 OpenCode 下方命令操作，把 `--tool opencode` 改为
`--tool claude`。

### OpenCode

```bash
# 更新：重复安装以刷新 Skill 链接
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level project

# 卸载
npx @cannbot-ai/install-helper uninstall gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level project
```

全局安装时把两条命令的 `project` 同时改为 `global`，可在任意目录执行。卸载只删除
对应的 Skill。

### Codex

```bash
# 目标仓项目级（remove 不加 --global 时默认为项目级）
cd /path/to/target-repository
npx skills update gitcode-issue-handler gitcode-toolkit --project
npx skills remove gitcode-issue-handler gitcode-toolkit --agent codex

# 当前用户全局（可在任意目录执行）
npx skills update gitcode-issue-handler gitcode-toolkit --global
npx skills remove gitcode-issue-handler gitcode-toolkit --agent codex --global
```

卸载 Skill 默认不应删除目标仓库的 YAML、报告或复现证据。只有确认不再需要历史配置和
运行产物后，才手工清理 `.cannbot/gitcode-issue-handler/`。旧版的仓根 YAML 和
`issue_analysis_data/` 只在已归档且验证无需回退后手工清理。

## 常见问题

| 现象 | 处理 |
|---|---|
| 启动报告 `gitcode-toolkit` 缺失 | 用同一安装机制、同一工具和同一级别补装 toolkit |
| OpenCode `/` 菜单没有 Handler | 确认 `.opencode/skills/` 中已安装 handler 和 toolkit；当前版本未自动暴露 Skill slash 入口时，改用自然语言触发 |
| `requests` / `yaml` 无法导入 | 按 `requirements.txt` 安装 Python 依赖 |
| 批量模式无法确定仓库 | 在目标仓库启动，检查 remote；或配置 `repo` / 使用 `--repo owner/repo` |
| 需要查看最近处理报告 | 打开目标仓库的 `.cannbot/gitcode-issue-handler/reports/latest.md` |

运行期的环境、复现、授权、交付和清理门禁以 [SKILL.md](../SKILL.md) 为准。
