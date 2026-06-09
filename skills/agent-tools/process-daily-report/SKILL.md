---
name: process-daily-report
description: "Process daily work reports with LLM: extract progress, risks, and action items, preview highlighted Markdown, then send to a DingTalk group (with optional Yuque source link). Use when user wants to summarize 日报, send daily report to DingTalk, or extract 进展/风险/Action from a local markdown file or Yuque URL."
keywords:
  - daily-report
  - 日报
  - dingtalk
  - 钉钉
  - yuque
  - 语雀
  - progress
  - risk
  - action
  - team-collaboration
---

# Process Daily Report

使用 LLM 智能分析日报内容，提取关键进展、风险与 Action 项，生成适合钉钉 Markdown 的预览，确认后发送到钉钉群。

## When to Use

- 用户要处理本地日报 Markdown 文件或语雀链接
- 用户要把日报摘要发送到钉钉群
- 用户要从日报中提取进展、风险、Action 并结构化展示
- 触发词：`处理日报`、`发送日报`、`日报摘要`、`process daily report`、`DingTalk 日报`

## Prerequisites

在运行前配置钉钉机器人 Webhook：

```bash
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=xxx"
```

获取方式：钉钉群 → 设置 → 智能助手 → 添加机器人 → 自定义机器人 → 设置安全策略获取 Webhook

## Quick Start

输入为本地文件路径或语雀链接，通过 Skill args 传递：

```text
/path/to/daily-report.md
```

或

```text
https://www.yuque.com/team/repo/doc-id
```

## Workflow

1. **检查环境变量** — 确认 `DINGTALK_WEBHOOK` 已配置
2. **读取日报** — 从 args 指定的本地文件或语雀链接获取内容
3. **LLM 智能提取** — 按 [llm-extraction-prompt.md](references/llm-extraction-prompt.md) 提取结构化 JSON
4. **生成 Markdown 预览** — 按 [dingtalk-markdown-format.md](references/dingtalk-markdown-format.md) 转换为钉钉 Markdown
5. **预览确认** — 向用户展示预览并询问是否发送
6. **发送到钉钉** — 用户确认后调用钉钉 Webhook API 发送消息
7. **返回结果** — 报告发送成功或失败信息

## Step 1: Check Environment

检查 `DINGTALK_WEBHOOK` 是否已配置。如果未配置，提示用户设置：

```text
请先设置 DINGTALK_WEBHOOK 环境变量：
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=xxx"

获取方式：钉钉群设置 → 智能助手 → 添加机器人 → 自定义机器人 → 设置安全策略获取 Webhook
```

## Step 2: Read Report

读取 Skill args 传递的文件路径或语雀链接，获取日报内容。

如果文件或链接不存在，提示：

```text
文件不存在: {file_path} 或 yuque链接不存在: {yuque_url}
请检查文件/yuque链接路径是否正确
```

如果内容为空，提示：

```text
文件/yuque链接内容为空，请检查 {file_path} 或 {yuque_url}
```

## Step 3: LLM Extraction

使用 LLM 分析日报内容并返回结构化 JSON。完整 prompt 与字段规则见 [references/llm-extraction-prompt.md](references/llm-extraction-prompt.md)。

## Step 4: Generate Markdown Preview

将 LLM 返回的 JSON 转换为钉钉 Markdown。颜色、emoji 与排版规则见 [references/dingtalk-markdown-format.md](references/dingtalk-markdown-format.md)。

## Step 5: Preview Confirmation

向用户展示 Markdown 预览，然后询问：

```text
是否确认发送到钉钉群？
```

选项：

- 确认发送
- 取消

## Step 6: Send to DingTalk

用户确认后，发送一条 Markdown 消息到钉钉。消息末尾在提供语雀链接时附加原文链接。

```bash
DATE=$(date +%Y.%m.%d)

curl -X POST "${DINGTALK_WEBHOOK}" \
  -H "Content-Type: application/json" \
  -d "{
    \"msgtype\": \"markdown\",
    \"markdown\": {
      \"title\": \"日报摘要 - ${DATE}\",
      \"text\": \"${MARKDOWN_TEXT}\"
    }
  }"
```

`MARKDOWN_TEXT` 按 [dingtalk-markdown-format.md](references/dingtalk-markdown-format.md) 生成。如果输入是语雀链接，在末尾附加：

```markdown
---
📎 [查看原文](${YUQUE_URL})
*本消息由 AI Agent 自动生成*
```

如果输入是本地文件，不附加原文链接行。

## Step 7: Return Result

- 成功：`「日报摘要」已发送到钉钉群`
- 失败：`发送失败，错误信息：{error}`

## Notes

1. 文件路径或语雀链接通过 Skill args 传递
2. 确保 `DINGTALK_WEBHOOK` 环境变量已配置
3. LLM 提取时限制每类不超过 5 条
4. 发送到钉钉时优先使用 `<font color="...">...</font>` 着色，并保留 `**加粗**`、emoji、反引号作为兜底
5. 分析文本语义并突出真正的结果词、风险词、动作词，不要机械整句加粗
6. 只有高风险或阻塞项才使用红色 `🔴` 风格，普通待定位问题保持橙色 `🟠`
7. 发送钉钉消息时不要 `@all`，默认只发送消息正文
8. 每个风险必须包含应对措施、责任人、完成时间，使用 `|` 分隔符在一行内显示
9. 如果风险项已在「风险」部分列出应对措施，不要在「Action」部分重复列出

## References

- [LLM 提取 Prompt](references/llm-extraction-prompt.md)
- [钉钉 Markdown 格式](references/dingtalk-markdown-format.md)
