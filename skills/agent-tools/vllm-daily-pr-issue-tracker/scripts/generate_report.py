#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Markdown daily report from fetched JSON data and AI summaries."""

import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


def _status_badge(status):
    badges = {
        "merged": "✅ merged",
        "open": "🟢 open",
        "closed": "🔴 closed",
    }
    return badges.get(status, status)


def generate_report(json_path: str, ai_summaries: dict | None = None) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    today_str = data["date"]
    items = data["items"]

    priority_groups = defaultdict(list)
    tech_groups = defaultdict(list)
    model_groups = defaultdict(list)

    for item in items:
        priority_groups[item["priority"]].append(item)
        for cat in item["tech_categories"]:
            tech_groups[cat].append(item)
        for cat in item["model_categories"]:
            model_groups[cat].append(item)

    total = len(items)
    vllm_count = sum(1 for i in items if "vllm-ascend" not in i["repo"])
    ascend_count = sum(1 for i in items if "vllm-ascend" in i["repo"])
    p0_count = len(priority_groups.get("P0", []))
    p1_count = len(priority_groups.get("P1", []))

    lines = []
    lines.append("# vllm 每日 PR & Issue 追踪报告")
    lines.append("")
    lines.append(f"> **日期**：{today_str}  ")
    lines.append(
        f"> **生成时间**：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
    )
    lines.append("> **数据来源**：`vllm-project/vllm` | `vllm-project/vllm-ascend`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📊 执行摘要")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 相关条目总数 | {total} |")
    lines.append(f"| vllm 主仓库 | {vllm_count} 条 |")
    lines.append(f"| vllm-ascend | {ascend_count} 条 |")
    lines.append(f"| P0 高优先级 | {p0_count} 条 |")
    lines.append(f"| P1 重要条目 | {p1_count} 条 |")
    lines.append("")

    if model_groups:
        lines.append("**模型关注分布**：")
        for model, model_items in sorted(model_groups.items(), key=lambda x: -len(x[1])):
            lines.append(f"- {model}：{len(model_items)} 条")
        lines.append("")

    if tech_groups:
        lines.append("**技术方向分布**：")
        for tech, tech_items in sorted(tech_groups.items(), key=lambda x: -len(x[1])):
            lines.append(f"- {tech}：{len(tech_items)} 条")
        lines.append("")

    lines.append("---")
    lines.append("")

    priority_config = [
        ("P0", "🔴 P0 — 紧急关注", "涉及多个关注模型 或 关注模型的 Bug Fix"),
        ("P1", "🟠 P1 — 重要关注", "关注模型 + 核心技术方向"),
        ("P2", "🟡 P2 — 一般关注", "核心技术方向 或 模型新特性/性能优化"),
        ("P3", "🟢 P3 — 低优先级", "其他相关条目"),
    ]

    for priority, title, desc in priority_config:
        group_items = priority_groups.get(priority, [])
        if not group_items:
            continue

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"> {desc}（共 {len(group_items)} 条）")
        lines.append("")

        for item in group_items:
            repo_short = item["repo"].split("/")[-1]
            type_emoji = "🔀" if item["type"] == "PR" else "🐛"
            status_badge = _status_badge(item["status"])

            lines.append(f"### {type_emoji} [{repo_short}] #{item['number']} {item['title']}")
            lines.append("")
            lines.append("| 字段 | 内容 |")
            lines.append("|------|------|")
            lines.append(f"| 类型 | {item['type']} |")
            lines.append(f"| 状态 | {status_badge} |")
            lines.append(f"| 仓库 | `{item['repo']}` |")
            lines.append(
                f"| 作者 | [@{item['author']}](https://github.com/{item['author']}) |"
            )
            lines.append(f"| 链接 | [{item['repo']}#{item['number']}]({item['url']}) |")
            lines.append(f"| 创建时间 | {item['created_at'][:10]} |")
            lines.append(f"| 更新时间 | {item['updated_at'][:10]} |")

            if item["labels"]:
                labels_str = " ".join(f"`{label}`" for label in item["labels"])
                lines.append(f"| 标签 | {labels_str} |")

            lines.append(f"| 模型分类 | {' / '.join(item['model_categories']) or '—'} |")
            lines.append(f"| 技术方向 | {' / '.join(item['tech_categories']) or '—'} |")
            lines.append(f"| 优先级 | **{item['priority']}** |")
            lines.append("")

            ai_info = (ai_summaries or {}).get(item["url"], {})
            summary = ai_info.get("summary", "")
            impact = ai_info.get("impact", "")

            if summary:
                lines.append("**📝 中文摘要**")
                lines.append("")
                lines.append(summary)
                lines.append("")

            if impact:
                lines.append("**💡 影响分析**")
                lines.append("")
                lines.append(impact)
                lines.append("")

            if item["body_preview"]:
                lines.append("<details>")
                lines.append("<summary>原文预览（点击展开）</summary>")
                lines.append("")
                lines.append("```")
                lines.append(item["body_preview"][:300])
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")

            lines.append("---")
            lines.append("")

    lines.append("## 📑 技术方向索引")
    lines.append("")

    tech_order = [
        "PD分离", "MTP/投机解码", "量化", "图模式/TorchAir",
        "模型Bug Fix", "性能优化", "模型新特性",
    ]
    for tech in tech_order:
        tech_items = tech_groups.get(tech, [])
        if not tech_items:
            continue
        lines.append(f"### {tech}（{len(tech_items)} 条）")
        lines.append("")
        for item in tech_items:
            repo_short = item["repo"].split("/")[-1]
            lines.append(f"- [{repo_short} #{item['number']}]({item['url']}) — {item['title']}")
        lines.append("")

    lines.append("## 🤖 模型关注索引")
    lines.append("")

    for model in ["DeepSeek", "Qwen", "GLM", "MiniMax", "Kimi/Moonshot"]:
        model_items = model_groups.get(model, [])
        if not model_items:
            continue
        lines.append(f"### {model}（{len(model_items)} 条）")
        lines.append("")
        for item in model_items:
            repo_short = item["repo"].split("/")[-1]
            lines.append(f"- [{repo_short} #{item['number']}]({item['url']}) — {item['title']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*本报告由 vllm-daily-pr-issue-tracker skill 自动生成*  ")
    lines.append(f"*关键词匹配 + AI 深度分析 | {today_str}*")
    lines.append("")

    report_content = "\n".join(lines)

    output_dir = os.path.join(os.getcwd(), "daily-reports")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"daily-report-{today_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[完成] 报告已生成: {report_path}")
    return report_path


if __name__ == "__main__":
    output_dir = os.path.join(os.getcwd(), "daily-reports")
    json_files = sorted(glob.glob(os.path.join(output_dir, "daily-data-*.json")))

    if not json_files:
        print("[错误] 未找到数据文件，请先运行 fetch_daily_data.py")
        sys.exit(1)

    latest_json = json_files[-1]
    print(f"[INFO] 使用数据文件: {latest_json}")

    # ai_summaries 由 AI 在分析后填充，格式见 SKILL.md
    ai_summaries = {}
    report_path = generate_report(latest_json, ai_summaries)
    print(f"报告路径: {report_path}")
