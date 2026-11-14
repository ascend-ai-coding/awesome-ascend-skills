# GitCode Issue Handler 安装与配置指南

本文档说明 `gitcode-issue-handler` 的安装依赖、常用触发方式、可选批量配置、更新和卸载，并提供可直接复制的使用示例。初次使用可先看 [快速上手](../README.md)；两份文档使用相同的首选安装命令，本文补充全局安装、备选方式和完整配置。仓库级安装方式总览见 [CANNBot Skills 安装指南](../../../docs/installation-guide.md)，所有 Skill 的使用总览见 [CANNBot Skills 使用样例](../../../docs/skills-usage.md)。

## 按客户端安装

`gitcode-issue-handler` 依赖 `gitcode-toolkit`，两项必须安装到同一个客户端和同一级别。
Claude Code 的 `infra-skills` 包已包含两项；OpenCode 和 Codex 命令则显式安装两项。

本文的“当前项目”或 `project` 均指要处理 Issue 的**目标代码仓**，不是 `cannbot-skills` 源码仓。执行项目级命令前必须先 `cd` 到目标仓根目录； `global` / `user` 级命令可在任意目录执行。Claude Code 的 `local` 与 `project` 都绑定当前目标仓。

| 场景/客户端 | 首选方式 | 首选安装命令 |
|---|---|---|
| Claude Code | Plugin Marketplace | `/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git`，再执行 `/plugin install infra-skills@cannbot` 和 `/reload-plugins` |
| OpenCode | install-helper | `npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit --tool opencode --level project` |
| Codex | install-helper | `npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit --tool codex --level project` |
| 本地已有源码仓 | 让 Agent 安装 | 在 `cannbot-skills` 仓库中告诉 Agent 目标目录、客户端和 project/global 级别 |

安装后可在目标仓库中让 Agent 初始化并引导配置，也可以在首次处理 Issue 时完成。安装器是否预先创建了 `.cannbot/gitcode-issue-handler/config/` 不影响这一步；已有自定义配置会保留。详见[配置初始化方式](#配置初始化方式)。

### Claude Code

Marketplace 是 Claude Code 的插件市场机制。`infra-skills@cannbot` 会一次安装 handler 和 toolkit；在 Claude Code 中执行：

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install infra-skills@cannbot
/reload-plugins
```

执行 `/plugin install` 后，Claude Code 会打开详情页供用户选择 user、project 或 local scope。选择 project/local 前，需先从目标仓根目录启动 Claude Code：

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

该备选要求 Node.js 20+，安装后可使用自然语言触发。

### OpenCode

推荐通过 `npx` 直接运行 install-helper，无需预先全局安装，但要求 Node.js 20+。以下命令会把 handler 和 toolkit 安装为两个独立 Skill：

```bash
# 目标仓项目级（写入 <target-repository>/.opencode/）
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level project

# 当前用户全局（写入 ~/.config/opencode/，可在任意目录执行）
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level global
```

只在当前仓库使用时选 `project`；需要跨仓使用时选 `global`。OpenCode 是否自动把已安装 Skill 暴露为 slash 入口取决于版本：支持该行为的版本可使用 `/gitcode-issue-handler`；其他版本使用自然语言 `使用 gitcode-issue-handler <任务描述>`。

### Codex

与 OpenCode 一样，推荐使用 install-helper，要求 Node.js 20+：

```bash
# 目标仓项目级（写入 <target-repository>/.agents/skills/）
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool codex --level project

# 当前用户全局（写入 ~/.agents/skills/，可在任意目录执行）
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool codex --level global
```

也可以使用 [skills CLI](https://github.com/vercel-labs/skills) 作为备选；默认安装到当前项目，增加 `--global` 后可供该用户的所有项目使用：

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

### 让 Agent 从本地源码安装

本地已经有 `cannbot-skills` 源码仓时，可以在该仓库中启动 Agent，让它安装当前工作区里的版本。
例如：

```text
请把当前 cannbot-skills 仓库中的 gitcode-issue-handler 和 gitcode-toolkit，以 project 级安装到 /path/to/target-repository，目标客户端是 Codex。使用本仓最新源码，保留已有配置，并引导我确认处理范围和自动响应方式；本次不处理 Issue。
```

将目标目录、客户端名称和安装级别替换为实际值。Agent 应使用当前仓库内容完成安装；项目级安装还应补齐目标仓库配置，并按 [首次配置引导](../references/configuration-setup.md) 询问常用设置。全局安装只安装 Skill，配置在各目标仓库中分别初始化。

## Python 依赖

脚本需要 Python 3.10+、`requests` 和 PyYAML。无法导入 `requests` 或 `yaml` 时，可按快速上手中的命令直接安装：

```bash
python3 -m pip install "requests>=2.28.0" "PyYAML>=6.0"
```

本地已有完整 `cannbot-skills` 源码仓时，也可以从依赖文件安装同样的依赖：

```bash
python3 -m pip install -r \
  /path/to/cannbot-skills/infra/gitcode-issue-handler/requirements.txt
```

安装机制不会保存 `GITCODE_TOKEN`，也不会替换目标仓库的 `AGENTS.md` / `CLAUDE.md`。
仅初始化或调整本地配置时需要 Python 3.10+ 和 PyYAML，不要求 Token 或 CANN 环境，也不访问 GitCode。`requests` 和 Token 在实际访问 API 前检查。
运行时不在启动阶段统一检查全部环境，而是在相关操作前按需检查：首次调用 GitCode API 前检查 API 客户端和 Token；首次同步仓库、读取 Git 历史或创建 worktree 前检查 Git、目标仓库、 remote 和所需工作目录；仅在准备 commit 前检查 git author；仅当代码任务确实需要编译、运行、复现或测试时，才在这些操作前检查 CANN 版本与环境一致性。纯规则咨询不做环境预检；某项缺失只阻塞依赖它的操作，不阻塞无关分析。Token 只在当前会话使用。

## 使用示例

先从目标代码仓根目录启动客户端；下文的“当前仓库”都是该目标仓。不要在 `cannbot-skills` 源码仓中启动这些任务，除非它本身就是待处理仓库。

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

- Codex、未自动暴露 slash 入口的 OpenCode，以及 Claude Code 的 install-helper 备选安装，均使用通用自然语言格式：

```text
使用 gitcode-issue-handler <任务描述>
```

| 场景 | 任务描述 |
|---|---|
| 只初始化并配置，不处理 Issue | `为当前仓库初始化并配置 gitcode-issue-handler，帮我确认处理范围和自动响应方式，配置完成后先不要处理 Issue` |
| 显式单 Issue 完整处理 | `完整处理 https://gitcode.com/cann/ops-math/issues/1511` |
| 只回复或答疑，不修改代码 | `只回复 https://gitcode.com/cann/ops-math/issues/456，不改代码` |
| 当前仓库批量分诊和处理 | `分诊并处理当前仓库需要关注的 Issue` |
| 继续处理挂起后的新回复 | `继续处理当前仓库中已挂起且提出者或责任人有新回复的 Issue` |
| 咨询 Issue 自动闭环 dry-run | `预览当前仓库已答复且长期无响应的咨询 Issue，不要实际关闭` |

当前仓库批量、挂起 follow-up 和自动闭环场景应在目标仓库目录中触发。

## 单 Issue、批量与自动闭环配置

配置会在运行时按需初始化，无需预先手写 YAML。模板 `repo` 留空；首次运行从显式目标或唯一 GitCode remote 仓库推导并保存。有多个不同候选或与已保存目标冲突时，助手通过问卷确认后保存并继续。之后直接复用非空 `repo`，不依赖 remote 名称。

### 配置初始化方式

- 支持安装后初始化的 install-helper 在**项目级安装**包含 `gitcode-issue-handler` 时，会自动准备两个配置文件，保留已有配置和仓根旧配置；不支持该功能的安装器由 Skill 在首次使用时补齐。
- Claude Marketplace 和第三方 `npx skills` 没有 handler 的项目配置初始化钩子；进入目标仓库后由 Skill 补齐缺失配置。
- **全局安装**没有绑定的目标仓库，配置在各仓库首次使用时分别初始化，不写入全局安装目录。

无论配置文件是在安装时还是首次运行时创建，Skill 都会区分“只有模板”和“已有用户设置”。只有模板时，Agent 分别询问仓库、范围和响应方式，可提供默认选项；已有自定义或仓根旧配置直接沿用，不强制重新配置。用户明确完成配置或选择保留当前设置后，记录在目标仓库的 `.cannbot/gitcode-issue-handler/config/setup-state.json`，以后不重复询问。未回答不记作确认；用户仍可随时要求修改配置。

**让 Agent 初始化并配置：**

在目标仓库中启动 Agent，输入：

```text
请为当前仓库初始化并配置 gitcode-issue-handler，帮我确认处理范围和自动响应方式，配置完成后先不要处理 Issue。
```

常用项为 `repo`、`responsibility`、`auto-response` 和 `auto-assign`。初始化引导必须分别询问仓库、范围和响应；唯一推导的仓库也作为候选确认，当前会话已明确的答案不重复询问。模板默认处理所有非 Roadmap 等纯规划类的 open Issue，纯规划类忽略，仅列举范围为空。自定义范围依次分三段询问需要处理、仅统计列出和忽略的范围。响应默认两个自动开关关闭；自定义时先询问 `auto-response`（默认 false），仅在其为 true 时再询问 `auto-assign`（默认 false），含义与授权边界见 [自动响应](../references/automation.md)。仅配置不拉取或处理 Issue，真实处理中的缺 Token 等待点仍按原规则优先执行。

**手工补齐文件（可选）：**

```bash
cd /path/to/target-repository
python3 /path/to/gitcode-issue-handler/scripts/init_config.py \
  --repository-root "$PWD"
```

将 `/path/to/gitcode-issue-handler` 替换为已安装 Skill 的实际目录；完整源码仓中的路径是 `/path/to/cannbot-skills/infra/gitcode-issue-handler`。目标始终是要处理 Issue 的仓库，不能因为脚本位于源码仓或全局安装目录，就把配置写到那里。

初始化脚本只创建缺失配置，重复执行不会覆盖已有文件；发现仓根旧配置时会沿用其设置并保留原文件。它不记录用户的配置选择；手工生成模板后，首次使用时仍会提供引导。若用户只要求“补齐文件”，Agent 执行此命令即可，不自行调整参数。交互引导的脚本入口和状态规则见 [首次配置引导](../references/configuration-setup.md)。

批量模式以当前启动目录为工作仓库。流程优先读取已配置的 `repo`，为空时再推导；分类器可通过 `classify_issues.py --repo owner/repo` 显式指定，获取器则接收由仓库标识派生出的 `fetch_issues.py --url https://gitcode.com/owner/repo`。因此 `classify_config.yaml` 不是安装前置条件。希望固定批量策略时，先用上述方式初始化，再编辑对应配置：

- `classify_config.yaml`：可固定 `repo`、增量状态、follow-up watch、缓存和自动闭环参数；
- `classify_config.yaml` 的 `responsibility`：`handle` 正常处理、`list-only` 仅列举一行、`ignore` 忽略；Agent 核查源码证据后交给分类脚本路由。旧 `responsibility_scope.md` 不再读取。
- `operator_owners.yaml`：可选的算子到 GitCode 登录名映射；缺失时流程按责任人门禁请求补充，不会静默改为 Agent 自行处理；
- `.cannbot/gitcode-issue-handler/`：统一存放 `config/`、`data/`、`reports/`、 `logs/`、`cache/`、`images/`、`repro/`、`worktrees/` 和 `tmp/`；流程会在当前仓库的 `.git/info/exclude` 中只追加 `/.cannbot/gitcode-issue-handler/`，不忽略其他 `.cannbot/` 内容，不改写 `.gitignore`。

新配置优先于仓根旧配置。仅当新配置不存在时，脚本才只读回退到仓根的 `classify_config.yaml` 或 `operator_owners.yaml`；新生成的默认数据、缓存和报告始终写入统一目录。旧 `issue_analysis_data/` 不会自动移动或删除，自定义 CLI 路径也不会被改写。

常规批量处理先复核责任范围，再按 Issue 类型准备实质首响；回复获授权且 GET 回查成功后才执行分派、PR 关联或代码修复。已有 PR、指派或响应时沿用原策略；分类器只读；首响和已明确负责人转交由 `auto-response` 控制，不明确时的候选临时指派由 `auto-assign` 控制，默认均关闭。详见 [自动响应](../references/automation.md)。
纯答疑直接融合在一次回复中；首响评分和样例见 `references/issue-comment-workflow.md`。

`auto-close-stale` 使用项目 `classify_config.yaml` 下部的 `auto_close` 策略；只有实际启用该维护路径时才需要复核这些设置。完整模板为 `assets/classify_config.yaml.template`，省略字段时从该模板补齐。它默认 dry-run，仍须显式 `--apply` 才写入。
日常批量获取默认维护 `data/followup-watch.json`：首次回看近期开启状态的更新，之后用游标增量扫描，并定点刷新等待提出者或责任人的 Issue；核心 closed 按 [intake 入口规则](../references/issue-intake.md) 跳过。该文件按仓库隔离，不应跨仓库复用。

配置和运行数据始终属于目标仓库，不写入全局工具配置根。多仓场景由各仓分别维护配置和运行证据。

## 更新与卸载

使用与安装时相同的安装工具、客户端和级别；已有安装继续用原工具维护，不需要为了采用本文的首选方式重新安装。

### Claude Code

Marketplace 安装：

```text
/plugin marketplace update cannbot
/plugin update infra-skills@cannbot
/reload-plugins

/plugin uninstall infra-skills@cannbot
```

上述命令对应默认的 user scope。如果原安装使用 project/local scope，先在目标仓根目录启动 Claude Code，再在 `/plugin` 的 Installed 页签更新或卸载对应 scope 的实例。

如果使用 install-helper 备选方案，则按 OpenCode 下方命令操作，把 `--tool opencode` 改为 `--tool claude`。

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

全局安装时把两条命令的 `project` 同时改为 `global`，可在任意目录执行。卸载只删除对应的 Skill。

### Codex

通过 install-helper 安装时：

```bash
# 更新
cd /path/to/target-repository
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool codex --level project

# 卸载
npx @cannbot-ai/install-helper uninstall gitcode-issue-handler gitcode-toolkit \
  --tool codex --level project
```

全局安装时把 `project` 改为 `global`。

如果原来使用的是 [skills CLI](https://github.com/vercel-labs/skills#skills-update)，则继续用它更新和卸载：

```bash
# 目标仓项目级（remove 不加 --global 时默认为项目级）
cd /path/to/target-repository
npx skills update gitcode-issue-handler gitcode-toolkit --project
npx skills remove gitcode-issue-handler gitcode-toolkit --agent codex

# 当前用户全局（可在任意目录执行）
npx skills update gitcode-issue-handler gitcode-toolkit --global
npx skills remove gitcode-issue-handler gitcode-toolkit --agent codex --global
```

卸载 Skill 默认不应删除目标仓库的 YAML、报告或复现证据。只有确认不再需要历史配置和运行产物后，才手工清理 `.cannbot/gitcode-issue-handler/`。旧版的仓根 YAML 和 `issue_analysis_data/` 只在已归档且验证无需回退后手工清理。

## 常见问题

| 现象 | 处理 |
|---|---|
| 启动报告 `gitcode-toolkit` 缺失 | 用同一安装机制、同一工具和同一级别补装 toolkit |
| OpenCode `/` 菜单没有 Handler | 确认 `.opencode/skills/` 中已安装 handler 和 toolkit；当前版本未自动暴露 Skill slash 入口时，改用自然语言触发 |
| `requests` / `yaml` 无法导入 | 按 `requirements.txt` 安装 Python 依赖 |
| 批量模式无法确定仓库 | 在目标仓库启动，检查 remote；或配置 `repo` / 使用 `--repo owner/repo` |
| 需要查看最近处理报告 | 打开目标仓库的 `.cannbot/gitcode-issue-handler/reports/latest.md` |

运行期的环境、复现、授权、交付和清理门禁以 [SKILL.md](../SKILL.md) 为准。
