# 批量并行代码检视系统

基于 AI Agent 的批量并行代码检视系统，支持在单台服务器上并行启动多个 Agent 进程，对多个代码目标或 PR 执行自动化代码检视。支持 `opencode` 和 `claude` 两种执行引擎。

## 核心特性

- **双引擎支持**：`opencode`（默认）和 `claude` 两种引擎，配置切换
- **并行执行**：多任务并发运行，可配置最大并发数
- **双模式支持**：手动 targets 配置模式 + PR 列表文件自动生成模式
- **Key Pool 轮转**：API Key 池自动 round-robin 分配，避免单 Key 限流
- **多 Provider 支持**：Anthropic / OpenAI / DeepSeek / Google / Azure，自动推断或显式指定 provider key
- **双重超时保障**：固定超时 + 空闲超时（日志无增量自动终止）

## 引擎切换

通过配置文件中的 `engine` 字段切换底层执行引擎，支持 `opencode`（默认）和 `claude` 两种。对调用者完全透明 — CLI 参数、PR 列表格式、运行命令均不变。

### Claude Code 引擎

使用 Claude Code 的 headless 模式（`claude -p`），通过 `--settings` 文件实现 Key 隔离。

配置示例（`review-config-claude.json`）：

```json
{
  "engine": "claude",
  "key_pool": [
    {
      "env": "ANTHROPIC_API_KEY_1",
      "base_url": "https://your-gateway.com"
    }
  ],
  "review": {
    "model": "claude-sonnet-4-20250514"
  }
}
```

Claude 引擎特有配置：

| 字段 | 位置 | 说明 |
|------|------|------|
| `base_url` | `review` 或 `key_pool` 条目 | 第三方网关地址（可选，不配则直连 Anthropic 官方） |

### OpenCode 引擎（默认）

使用 `opencode run` 非交互模式，通过环境变量注入实现 Key 隔离。

配置示例（`review-config.json`）：

```json
{
  "engine": "opencode",
  "key_pool": [
    {
      "env": "DEEPSEEK_API_KEY_1",
      "provider_key": "DEEPSEEK_API_KEY"
    }
  ]
}
```

OpenCode 引擎特有配置：

| 字段 | 位置 | 说明 |
|------|------|------|
| `provider_key` | `key_pool` 条目 | 标准 Provider Key 名（如 `ANTHROPIC_API_KEY`），字符串格式可自动推断 |

## 快速开始

### 读取 PR 列表文件

适合批量检视多个 PR 的场景，配合 Key Pool 自动轮转分配 API Key。

**1. 准备 PR 列表文件**

```bash
# pr_list.txt，每行一个 PR URL，支持 # 注释和空行
cat > pr_list.txt << 'EOF'
# 仓库 A 的 PR
https://gitcode.com/cann/ops-transformer/pull/3604
# 仓库 B 的 PR
https://gitcode.com/cann/ops-math/pull/1234
EOF
```

**2. 准备配置文件**

```json
{
  "engine": "opencode",
  "key_pool": [
    { "env": "ANTHROPIC_API_KEY_1", "provider_key": "ANTHROPIC_API_KEY" },
    { "env": "ANTHROPIC_API_KEY_2", "provider_key": "ANTHROPIC_API_KEY" },
    { "env": "ANTHROPIC_API_KEY_3", "provider_key": "ANTHROPIC_API_KEY" }
  ],
  "review": {
    "skill_prompt": "使用 ascendc-code-review skill 检视 {pr_url}。PR 检视模式。",
    "model": "anthropic/claude-sonnet-4-20250514",
    "timeout_sec": 1800,
    "idle_timeout_sec": 300
  },
  "execution": {
    "max_parallel": 3,
    "retry_on_failure": 1,
    "output_dir": "./reports"
  }
}
```

**3. 设置环境变量**

```bash
export ANTHROPIC_API_KEY_1="sk-ant-xxx"
export ANTHROPIC_API_KEY_2="sk-ant-yyy"
export ANTHROPIC_API_KEY_3="sk-ant-zzz"
```

**4. 运行**

```bash
# dry-run 预览 Key 分配
PYTHONPATH=src python -m batch_review run --config src/batch_review/review-config.json --pr-file src/batch_review/pr_list.txt --dry-run

# 正式运行
PYTHONPATH=src python -m batch_review run --config src/batch_review/review-config.json --pr-file src/batch_review/pr_list.txt
```

系统会自动从 URL 提取仓库名和 PR 编号生成 target 名称，并按 round-robin 分配 Key：

```
ops-transformer-pr-3604    → ANTHROPIC_API_KEY_1
ops-math-pr-1234           → ANTHROPIC_API_KEY_2
```

## CLI 参数

```bash
python -m batch_review run --config <配置文件> [选项]
```

| 参数 | 说明 |
|------|------|
| `--config <path>` | **必填**，JSON 配置文件路径 |
| `--pr-file <path>` | PR 列表文件路径（每行一个 PR URL，支持 `#` 注释） |
| `--output <path>` | 覆盖配置中的 `output_dir` |
| `--max-parallel <n>` | 覆盖配置中的 `max_parallel` |
| `--run-id <id>` | 自定义运行 ID（默认自动生成 `run_YYYYMMDD_HHMMSS_xxxxxx`） |
| `--dry-run` | 仅打印命令和 Key 分配，不实际执行 |
| `--foreground` | 前台执行（默认） |
| `--background` | 后台执行 |

## 配置说明

### engine（可选）

引擎选择，顶层字段，与 `execution` 同级。缺省时默认 `"opencode"`。

| 值 | 说明 |
|------|------|
| `"opencode"` | 使用 `opencode run` 非交互模式（默认） |
| `"claude"` | 使用 Claude Code headless 模式（`claude -p`） |

### key_pool section（可选）

API Key 池，用于 PR 列表文件模式下自动轮转分配 Key。

支持两种格式：

**格式一：对象列表（推荐）**

```json
"key_pool": [
  {
    "env": "ANTHROPIC_API_KEY_1",
    "provider_key": "ANTHROPIC_API_KEY",
    "base_url": "https://your-gateway.com"
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `env` | string | 是 | API Key 环境变量名 |
| `provider_key` | string | opencode 引擎必填 | 标准 Provider Key 名（如 `ANTHROPIC_API_KEY`），claude 引擎忽略，字符串格式可自动推断 |
| `base_url` | string | 否 | 第三方网关地址（仅 claude 引擎使用，支持 `review.base_url` 全局默认 + `key_pool` 级别覆盖） |

**格式二：字符串列表（简写，自动推断 provider_key）**

```json
"key_pool": [
  "ANTHROPIC_API_KEY_1",
  "ANTHROPIC_API_KEY_2"
]
```

> **provider_key 推断规则**：环境变量名包含 `ANTHROPIC` → `ANTHROPIC_API_KEY`，包含 `OPENAI` → `OPENAI_API_KEY`，包含 `DEEPSEEK` → `DEEPSEEK_API_KEY`，包含 `GOOGLE`/`GEMINI` → `GOOGLE_API_KEY`，包含 `AZURE` → `AZURE_API_KEY`。

### review section

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | - | 默认模型（可被 target 覆盖） |
| `agent` | string | null | 默认 agent（可选） |
| `base_url` | string | "" | 全局第三方网关地址（仅 claude 引擎，`key_pool` 条目可覆盖） |
| `timeout_sec` | int | 1800 | 单任务固定超时（秒） |
| `idle_timeout_sec` | int | 300 | 空闲超时（日志无增量秒数） |
| `format` | string | "text" | 输出格式 |
| `skill_prompt` | string | "" | PR 检视的 prompt 模板，使用 `{pr_url}` 占位符 |

### execution section

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_parallel` | int | 3 | 最大并发数 |
| `retry_on_failure` | int | 0 | 失败/超时重试次数 |
| `retry_delay_sec` | int | 30 | 重试前等待时间（秒） |
| `output_dir` | string | "./review-reports" | 报告输出根目录 |

## 输出结构

```
reports/
└── run_20250115_143022_a1b2c3/
    ├── run_state.json
    ├── batch_result.json
    └── <target-name>/
        ├── settings.json          # Claude 引擎新增：Key 隔离配置
        ├── PROMPT.md
        ├── claude.log             # Claude 引擎：JSON 输出
        ├── claude_stderr.log      # Claude 引擎：stderr 日志
        ├── opencode.log           # OpenCode 引擎：合并日志
        ├── review_report.md
        └── task_state.json
```

## 超时与重试机制

### 双重超时保障

1. **固定超时**（`timeout_sec`）：任务总时间超过此值则终止进程组
2. **空闲超时**（`idle_timeout_sec`）：日志文件长时间无增量（Agent 卡死）则终止进程组

> 超时后先发送 `SIGTERM`，等待 10 秒 grace period，若仍存活则发送 `SIGKILL`。

### 失败重试

配置 `retry_on_failure` 后，任务失败或超时会自动重试：

- 重试前清理上次的部分报告，日志重命名为 `opencode_attempt_N.log` / `claude_attempt_N.log` 保留
- 重试间隔 5 秒
- 最终状态记录实际尝试次数

## Skill 指定方式

在 `prompt` 字段中用自然语言告诉 Agent 使用哪个 skill：

```json
{
  "prompt": "使用 ascendc-code-review skill 检视这个项目的代码"
}
```

PR 列表文件模式下，通过 `review.skill_prompt` 配置模板，系统自动替换 `{pr_url}` 占位符：

```json
{
  "review": {
    "skill_prompt": "使用 ascendc-code-review skill 检视 {pr_url}。PR 检视模式。"
  }
}
```

## 环境变量隔离

每个子进程独立运行，Key 隔离机制根据引擎不同：

### OpenCode 引擎

- 将 target 指定的 Key 注入为标准 Provider Key（如 `ANTHROPIC_API_KEY`）
- 清空其他 Provider 的 Key，防止串用
- 注入 `REPORT_FILE`、`BENCHMARK_*` 等辅助环境变量

### Claude 引擎

- 为每个任务动态生成独立的 `settings.json`（含 `ANTHROPIC_AUTH_TOKEN` + `base_url`）
- 通过 `--settings` 参数指定，实现 Key 隔离
- 环境变量中不注入 API Key（由 settings.json 负责）
- 注入 `REPORT_FILE`、`BENCHMARK_*` 等辅助环境变量

## 架构概览

```
__main__.py     CLI 入口 — 参数解析、子命令路由
config.py       配置加载 — JSON 解析、环境变量校验、PR 列表生成 targets、engine 字段
scheduler.py    调度器 — 任务队列管理、并发槽位控制、汇总报告（含 cost 信息）
process.py      进程管理 — 子进程生命周期（启动 → 监控 → 清理）、重试、JSON 输出解析
prompt.py       Prompt 构建 — 生成 PROMPT.md、构建双引擎命令、环境变量隔离、settings.json 生成
timeout.py      超时控制 — 固定超时计时器 + 空闲超时监控线程
state.py        状态持久化 — 原子 JSON 写入、运行/任务状态追踪（含 engine/cost 字段）
```
```

---

### 4. `.gitignore` — 新增 `**/settings.json`

请在 `E:\reviewer_bench\.gitignore` 末尾追加以下内容：

```gitignore
# 防止 settings.json（含明文 API Key）意外提交
**/settings.json