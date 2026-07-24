---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-knowledge-issue-report
description: 生成并校验 cannbot-knowledge 知识库 GitCode Issue 提交材料。触发：当用户要提交 Issue、反馈知识内容错误/缺失、检索错误、图谱错误、知识编译错误、治理
  lint 错误、debug 下游 agent 使用问题，或要求打包复现与测试材料时使用。
original-name: knowledge-issue-report
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# knowledge-issue-report — 知识库 Issue 材料整理

本 skill 用于把用户反馈整理成可受理的 cannbot-knowledge Issue。目标是收集完整事实、标明缺失材料、生成可复现附件包，而不是直接裁决知识是否正确。

## 工作原则

- 不凭记忆补事实；缺材料时列入 `needs-info`。
- 原始 Prompt / Query、触发命令、终端输出必须逐字保留，不要转述或节选。
- 敏感信息可以脱敏，但不要删除配置结构。
- 内容错误必须要求 source、fixed commit、日志、trace 或可复现实验证据。
- 版本差异、平台差异、结论冲突进入知识治理流程，不直接覆盖已有知识。
- 涉及模型、agent、rerank 或 judge 时，必须记录模型完整标识和运行记录。

## Issue 类型

让用户选择一个或多个：

- 知识内容错误
- 知识缺失
- 检索结果错误
- 图谱关联错误
- 知识编译错误
- 知识治理 / lint 错误
- 下游 agent 使用知识库出错
- 其他

## 必备材料

整理 Issue 前检查以下材料。缺少关键材料时，先输出 `needs-info` 清单。

| # | 材料 | 要求 |
|---|---|---|
| 1 | 问题类型 | 从 Issue 类型中选择，可多选 |
| 2 | 知识库版本 | 仓库、分支、commit、tag/release、是否有本地修改 |
| 3 | CANNBot skills 版本 | `cannbot-skills` 仓库、分支、commit、插件路径 |
| 4 | CLI / 执行器及版本 | 执行器名称、版本、Python 版本、OS |
| 5 | 模型 | 使用 agent/LLM/rerank/judge 时提供完整模型标识；未使用写“未使用” |
| 6 | 完整原始 Prompt / Query | 逐字粘贴完整输入 |
| 7 | 完整触发命令 | 实际执行的完整命令行，包含所有参数 |
| 8 | 配置文件 | 本次运行实际生效配置；敏感信息可脱敏但保留结构 |
| 9 | 运行记录 | session、model request/response、tool call、生成文件 diff |
| 10 | 终端完整输出 | 从命令开始到问题发生为止的完整 stdout/stderr |
| 11 | 复现率 | 工具、检索、agent 类问题至少运行 3 次，说明“运行 y 次，出现 x 次” |
| 12 | 受影响知识 | doc-id/path、section/anchor、status、confidence、resource/sources |
| 13 | 期望行为与实际行为 | 分别说明正确行为和实际发生了什么 |
| 14 | 证据 / 建议修复 | 官方文档、fixed commit、源码路径、trace、日志、错误信息或建议 patch |

## 收集流程

1. 先确定 Issue 类型和最小复现路径。
2. 收集两个版本：
   - 知识库本体：`git remote -v`、`git branch --show-current`、`git rev-parse HEAD`、`git status --short`、必要时 `git diff`。
   - 插件仓：`cannbot-skills` 或当前插件仓的 remote、branch、commit、插件路径 `plugins-community/cannbot-knowledge/`。
3. 收集执行环境：CLI/执行器版本、Python 版本、OS。
4. 保存完整 Prompt/Query、完整命令、完整终端输出。
5. 如果涉及 agent 或模型，保存 session、模型 request/response、tool call、生成文件 diff。
6. 如果是工具、检索或 agent 问题，要求至少 3 次复现记录。
7. 如果是稳定知识内容错误，可说明“不需要重复运行”，但必须给出 affected card 和证据。
8. 输出 Issue 正文和附件包目录；材料不足时先输出 `needs-info`。

## 附件包结构

推荐包名：

```text
knowledge-issue-<YYYYMMDD>-<short-title>.tar.gz
```

推荐结构：

```text
knowledge-issue-20260702-query-error.tar.gz
├── README.md
├── environment.txt
├── command.txt
├── prompt.txt
├── terminal.log
├── configs/
├── sessions/
│   ├── opencode.db
│   ├── model-calls.jsonl
│   └── tool-calls.jsonl
├── knowledge/
│   ├── git-status.txt
│   ├── git-diff.patch
│   └── affected-files.txt
└── reproduction/
    ├── run-1/
    ├── run-2/
    └── run-3/
```

知识内容错误可简化为：

```text
README.md
affected-card.md
evidence.md
suggested-fix.patch
```

## Issue 模板

````markdown
## 1. 问题类型

- [ ] 知识内容错误
- [ ] 知识缺失
- [ ] 检索结果错误
- [ ] 图谱关联错误
- [ ] 知识编译错误
- [ ] 知识治理 / lint 错误
- [ ] 下游 agent 使用知识库出错
- [ ] 其他：

## 2. 问题摘要

请用 1-3 句话说明问题。

## 3. 知识库版本

- 知识库仓库：
- 分支：
- commit：
- tag / release：
- 是否有本地修改：
- `git status` / `git diff`：见附件 / 无

## 4. CANNBot skills 版本

- `cannbot-skills` 仓库：
- 分支：
- commit：
- 插件路径：`plugins-community/cannbot-knowledge/`

## 5. CLI / 执行器及版本

- 执行器：
- 执行器版本：
- Python 版本：
- OS：

## 6. 模型

- 模型完整标识：
- 用途：
- 如未使用模型，请写：未使用

## 7. 完整原始 Prompt / Query

请逐字粘贴完整输入。

未使用 Prompt 时填写：

> 未使用 Prompt，仅执行命令。

## 8. 完整触发命令

请粘贴完整命令，包含所有参数。

示例：

```bash
python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root /path/to/cannbot-knowledge search --query "A5 LayerNorm UB reuse"
```

## 9. 配置文件

请上传本次运行实际生效的配置文件。

包括但不限于：

- `.opencode/`
- `.cannbot/`
- 插件配置
- 检索配置
- `metadata/*.yaml`

敏感信息可以脱敏，但不要删除配置结构。

## 10. 运行记录

如使用 agent / LLM / 多 agent 工作流，请提供：

- session / `opencode.db`
- model request / response JSON
- tool call 记录
- 生成文件 diff

如未使用 agent，请填写：

> 未使用 agent，仅提供命令、配置和终端输出。

## 11. 终端完整输出

请上传完整 stdout / stderr。

要求：

- 不得只贴最后几行；
- 不得删减中间日志；
- 敏感信息可脱敏。

## 12. 复现率

工具类、检索类、agent 类问题请至少运行 3 次。

- Run 1：成功 / 失败，附件：
- Run 2：成功 / 失败，附件：
- Run 3：成功 / 失败，附件：
- 复现率：运行 y 次，出现 x 次

如果是确定性知识内容错误，可以填写：

> 内容错误稳定存在，不需要重复运行。

## 13. 受影响知识

- doc-id / path：
- section / anchor：
- status：
- confidence：
- resource / sources：

## 14. 期望行为

请说明正确行为应该是什么。

## 15. 实际行为

请说明实际发生了什么。

## 16. 证据 / 建议修复

请提供官方文档、fixed commit、源码路径、trace、日志、错误信息或建议 patch。
````

## 受理判断

- 材料完整的问题优先处理。
- 可复现的问题优先处理。
- 内容错误必须有 source、fixed commit、日志等证据。
- 缺少关键材料的 Issue 标记为 `needs-info`。
- 涉及敏感信息的材料必须先脱敏。
- 结论冲突、版本差异、平台差异进入知识治理流程，不直接覆盖已有知识。
