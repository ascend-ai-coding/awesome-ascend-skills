---
name: vllm-daily-pr-issue-tracker
description: "Track daily PRs and Issues from vllm-project/vllm and vllm-project/vllm-ascend, filter by model (DeepSeek/Qwen/GLM/MiniMax/Kimi) and tech topics (PD disaggregation, MTP, quantization, graph mode, performance), analyze with LLM, and generate a Markdown report. Use when user wants vllm daily tracker, PR/Issue digest, or Ascend inference ecosystem monitoring."
keywords:
  - vllm
  - vllm-ascend
  - daily-report
  - github
  - pr
  - issue
  - deepseek
  - qwen
  - quantization
  - torchair
  - 日报
  - 追踪
---

# vllm Daily PR & Issue Tracker

每日自动获取 `vllm-project/vllm` 和 `vllm-project/vllm-ascend` 当天更新的 PR 与 Issue，按关注场景筛选、深度分析分类，并生成 Markdown 报告。

## When to Use

- 用户要追踪 vllm / vllm-ascend 每日 PR 和 Issue 动态
- 用户关注 DeepSeek、Qwen、GLM 等模型在 vllm 生态的更新
- 用户关注 PD 分离、MTP、量化、图模式、性能优化等技术方向
- 触发词：`vllm 日报`、`PR Issue 追踪`、`vllm daily tracker`、`vllm-ascend 动态`

## Prerequisites

需要设置 `GITHUB_TOKEN` 环境变量。配置方式见 [env-setup.md](references/env-setup.md)。

## Quick Start

```bash
# Step 1: 拉取并筛选当天数据
python scripts/fetch_daily_data.py

# Step 2: AI 分析后生成报告（需先填充 ai_summaries）
python scripts/generate_report.py
```

输出目录：`daily-reports/`

| 文件 | 说明 |
|------|------|
| `daily-reports/daily-data-YYYY-MM-DD.json` | Step 1 原始筛选数据 |
| `daily-reports/daily-report-YYYY-MM-DD.md` | 最终 Markdown 报告 |

## Workflow

1. **初始化** — 运行 `scripts/fetch_daily_data.py` 拉取当天数据
2. **AI 深度分析** — 读取 JSON，逐条撰写中文摘要与影响分析
3. **生成报告** — 构建 `ai_summaries` 后运行 `scripts/generate_report.py`
4. **返回结果** — 告知报告路径与关键发现摘要

## Step 1: Fetch Data

运行数据采集脚本：

```bash
python <skill-path>/scripts/fetch_daily_data.py
```

脚本会：

- 通过 GitHub API 拉取两个仓库当天 merged/opened PR 与 opened/updated Issue
- 按 [keywords.md](references/keywords.md) 中的模型与技术关键词筛选
- 按 [priority-rules.md](references/priority-rules.md) 标注 P0–P3 优先级
- 保存 JSON 到 `daily-reports/daily-data-YYYY-MM-DD.json`

## Step 2: AI Analysis

读取 JSON 文件，对每条 PR/Issue 进行分析：

- 阅读 `title`、`body_preview`、`labels`
- 用中文撰写 2–3 句摘要
- 分析技术影响与用户影响
- 确认或修正自动分类结果

构建 `ai_summaries` 字典：

```python
ai_summaries = {
    "https://github.com/vllm-project/vllm/pull/12345": {
        "summary": "该 PR 修复了 DeepSeek-R1 在 FP8 量化模式下的数值精度问题...",
        "impact": "影响所有使用 FP8 量化运行 DeepSeek-R1 的用户，建议尽快升级。",
    }
}
```

## Step 3: Generate Report

将 `ai_summaries` 传入报告生成脚本，或在 AI 分析后直接调用 `generate_report()`：

```bash
python <skill-path>/scripts/generate_report.py
```

报告格式见 [report-template.md](references/report-template.md)。

## Step 4: Return Result

告知用户：

- 报告路径：`daily-reports/daily-report-YYYY-MM-DD.md`
- P0/P1 高优先级条目摘要
- 模型与技术方向分布统计

## Notes

1. 未设置 `GITHUB_TOKEN` 时 API 限流为每小时 60 次，可能拉取不完整
2. 关键词与优先级规则可在 `scripts/fetch_daily_data.py` 中调整
3. `ai_summaries` 的 key 为条目的 `url` 字段
4. 报告按优先级、技术方向、模型三个维度组织索引

## References

- [环境配置](references/env-setup.md)
- [关键词配置表](references/keywords.md)
- [优先级规则](references/priority-rules.md)
- [报告格式模板](references/report-template.md)
