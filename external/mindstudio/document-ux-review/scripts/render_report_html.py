#!/usr/bin/env python3

import argparse
import html
import json
import re
from pathlib import Path


REF_RE = re.compile(r"((?:[\w.-]+(?:[\\/][\w.-]+)*)\.md:(\d+)(?:-(\d+))?)")
CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\n([\s\S]*?)```")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def clean_value(text: str) -> str:
    cleaned = str(text or "").strip()
    if (
        cleaned.startswith("`")
        and cleaned.endswith("`")
        and len(cleaned) >= 2
        and cleaned.count("`") == 2
    ):
        return cleaned[1:-1].strip()
    return cleaned


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def split_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^##\s+(.+)$", markdown))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        sections[title] = markdown[start:end].strip()
    return sections


def load_json_file(path: Path, fallback):
    text = read_text(path)
    if not text.strip():
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def normalize_section_title(title: str) -> str:
    title = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", title.strip())
    return re.sub(r"\s+", "", title)


def get_section_text(sections: dict[str, str], *aliases: str) -> str:
    normalized_sections = {
        normalize_section_title(key): value for key, value in sections.items()
    }
    for alias in aliases:
        if alias in sections:
            return sections[alias]
        normalized_alias = normalize_section_title(alias)
        if normalized_alias in normalized_sections:
            return normalized_sections[normalized_alias]
    return ""


def normalize_table_rows(headers: list[str], rows: list[list[str]]) -> list[list[str]]:
    if not headers:
        return []
    target_len = len(headers)
    normalized_rows = []
    for row in rows:
        normalized = [cell.strip() for cell in row]
        if not any(normalized):
            continue
        if len(normalized) < target_len:
            normalized += [""] * (target_len - len(normalized))
        elif len(normalized) > target_len:
            normalized = normalized[:target_len]
        normalized_rows.append(normalized)
    return normalized_rows


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned:
            return cleaned
    return ""


def parse_key_values(section_text: str) -> tuple[dict[str, str], list[str]]:
    data: dict[str, str] = {}
    ordered: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        if "：" in content:
            key, value = content.split("：", 1)
        elif ":" in content:
            key, value = content.split(":", 1)
        else:
            ordered.append(content)
            continue
        key = key.strip()
        value = value.strip()
        data[key] = value
        ordered.append(content)
    return data, ordered


def parse_list(section_text: str) -> list[str]:
    items = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
        elif re.match(r"\d+\.\s+", stripped):
            items.append(re.sub(r"^\d+\.\s+", "", stripped))
    return items


def extract_first_table(section_text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.rstrip() for line in section_text.splitlines()]
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
            in_table = True
        elif in_table:
            break
    if len(table_lines) < 2:
        return [], []
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return header, rows


def parse_detailed_issues(section_text: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    matches = list(re.finditer(r"(?m)^###\s+(.+)$", section_text))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section_text)
        body = section_text[start:end].strip()
        fields: dict[str, str] = {"标题": title}
        current_key = None
        for raw in body.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            bullet = re.match(r"^-\s*([^：:]+)[：:](.*)$", stripped)
            if bullet:
                current_key = bullet.group(1).strip()
                fields[current_key] = bullet.group(2).strip()
                continue
            if current_key and stripped:
                fields[current_key] = (
                    fields.get(current_key, "") + "\n" + stripped
                ).strip()
        issues.append(fields)
    return issues


def parse_commit(text: str) -> str:
    for candidate in re.findall(r"\b([0-9a-f]{7,40})\b", text, re.IGNORECASE):
        if len(candidate) == 40 or re.search(r"[a-f]", candidate, re.IGNORECASE):
            return candidate
    return ""


def parse_branch(text: str) -> str:
    cleaned = clean_value(text)
    if not cleaned:
        return ""
    commit = parse_commit(cleaned)
    if commit:
        cleaned = re.sub(rf"\b{re.escape(commit)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?i)\b(commit(?:\s+id)?|sha|revision)\b", "", cleaned)
    cleaned = re.sub(r"[()@,，:/|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.lower() in {"unknown", "n/a", "na"}:
        return ""
    return cleaned


def extract_score(text: str) -> str:
    match = re.search(r"总体评分[:：]\s*`?([^`\n]+/100)`?", text)
    return match.group(1).strip() if match else ""


def parse_risks(section_text: str) -> list[str]:
    risks = []
    in_risks = False
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 主要风险"):
            in_risks = True
            continue
        if in_risks and re.match(r"\d+\.\s+", stripped):
            risks.append(re.sub(r"^\d+\.\s+", "", stripped))
        elif in_risks and stripped.startswith("- "):
            break
    return risks


def inline_format(text: str) -> str:
    escaped = html.escape(text)
    if "`" in escaped:
        parts = escaped.split("`")
        rebuilt = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                rebuilt.append(f"<code>{part}</code>")
            else:
                rebuilt.append(part)
        escaped = "".join(rebuilt)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def render_simple_markdown(text: str) -> str:
    if not text.strip():
        return ""

    placeholders = {}

    def stash_code(match: re.Match[str]) -> str:
        key = f"__CODE_BLOCK_{len(placeholders)}__"
        placeholders[key] = (
            f"<pre><code>{html.escape(match.group(1).strip())}</code></pre>"
        )
        return key

    text = CODE_BLOCK_RE.sub(stash_code, text)
    blocks = re.split(r"\n\s*\n", text.strip())
    rendered = []
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if stripped in placeholders:
            rendered.append(placeholders[stripped])
            continue
        lines = stripped.splitlines()
        if all(line.strip().startswith("- ") for line in lines):
            items = "".join(
                f"<li>{inline_format(line.strip()[2:].strip())}</li>" for line in lines
            )
            rendered.append(f"<ul>{items}</ul>")
            continue
        if all(re.match(r"\d+\.\s+", line.strip()) for line in lines):
            items = "".join(
                f"<li>{inline_format(re.sub(r'^\d+\.\s+', '', line.strip()))}</li>"
                for line in lines
            )
            rendered.append(f"<ol>{items}</ol>")
            continue
        rendered.append(
            f"<p>{inline_format(' '.join(line.strip() for line in lines))}</p>"
        )
    return "".join(rendered)


def severity_class(text: str) -> str:
    mapping = {
        "阻塞": "sev-blocker",
        "高": "sev-high",
        "中": "sev-medium",
        "低": "sev-low",
        "缺失": "sev-high",
        "不完整": "sev-high",
        "不清晰": "sev-medium",
        "部分具备": "sev-medium",
        "偏弱": "sev-medium",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return "sev-neutral"


def status_class(text: str) -> str:
    mapping = {
        "OK": "status-ok",
        "完全走通": "status-ok",
        "阻塞": "status-blocked",
        "未走通": "status-blocked",
        "偏差继续": "status-deviation",
        "部分走通": "status-deviation",
        "未执行": "status-skipped",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return "status-neutral"


def build_ref_map(texts: list[str], repo_root: Path) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for text in texts:
        for match in REF_RE.finditer(text):
            full = match.group(1)
            if full in refs:
                continue
            file_part, start_str, end_str = (
                match.group(1).split(":", 1)[0].replace("\\", "/"),
                match.group(2),
                match.group(3),
            )
            start = int(start_str)
            end = int(end_str) if end_str else start
            path = repo_root / file_part
            excerpt_lines = []
            if path.exists():
                all_lines = read_text(path).splitlines()
                start_idx = max(1, start)
                end_idx = min(len(all_lines), end)
                if end_idx - start_idx > 24:
                    excerpt_range = list(range(start_idx, start_idx + 12)) + list(
                        range(max(start_idx + 12, end_idx - 6), end_idx + 1)
                    )
                    last = None
                    for line_no in excerpt_range:
                        if last and line_no - last > 1:
                            excerpt_lines.append("... (中间省略若干行) ...")
                        excerpt_lines.append(f"{line_no}: {all_lines[line_no - 1]}")
                        last = line_no
                else:
                    for line_no in range(start_idx, end_idx + 1):
                        excerpt_lines.append(f"{line_no}: {all_lines[line_no - 1]}")
            refs[full] = {
                "id": f"ref-{slugify(full)}",
                "label": full,
                "file": file_part,
                "excerpt": "\n".join(excerpt_lines)
                if excerpt_lines
                else "未找到本地源码摘录。",
            }
    return refs


def render_ref_links(text: str, ref_map: dict[str, dict]) -> str:
    parts = []
    for match in REF_RE.finditer(text):
        ref = match.group(1)
        ref_info = ref_map.get(ref)
        if ref_info:
            parts.append(
                f'<a class="doc-link" href="#{ref_info["id"]}">{html.escape(ref)}</a>'
            )
    if parts:
        return '<div class="doc-links">' + " ".join(parts) + "</div>"
    return f"<div>{inline_format(text)}</div>"


def issue_summary_from_row(row: list[str]) -> str:
    if len(row) > 4:
        return row[4].strip()
    return ""


def scenario_top_issue(scenario: dict) -> str:
    for row in scenario.get("issue_rows", []):
        summary = issue_summary_from_row(row)
        if summary:
            return summary
    for issue in scenario.get("issues", []):
        summary = first_nonempty(
            issue.get("标题", ""),
            issue.get("实际现象", ""),
            issue.get("影响分析", ""),
        )
        if summary:
            return summary
    return "未识别关键问题"


def scenario_from_run_dir(run_dir: Path, repo_root: Path) -> dict:
    outputs = run_dir / "outputs"
    report_text = read_text(outputs / "report.md")
    transcript_text = read_text(outputs / "transcript.md")
    grading = load_json_file(run_dir / "grading.json", {})
    metadata = load_json_file(run_dir.parent.parent / "eval_metadata.json", {})
    sections = split_sections(report_text)
    overview, _ = parse_key_values(get_section_text(sections, "1. 审查对象", "审查对象"))
    conclusion_text = get_section_text(
        sections, "2. 总体评分与结论", "总体评分与结论"
    )
    conclusion, _ = parse_key_values(conclusion_text)
    flow_header, flow_rows = extract_first_table(
        get_section_text(sections, "3. 体验流程图", "体验流程图")
    )
    issue_header, issue_rows = extract_first_table(
        get_section_text(sections, "5. 关键问题概览", "关键问题概览")
    )
    industry_header, industry_rows = extract_first_table(
        get_section_text(
            sections,
            "8. 开源项目关键章节与业界实践检查",
            "开源项目关键章节与业界实践检查",
        )
    )
    issues = parse_detailed_issues(get_section_text(sections, "6. 详细问题", "详细问题"))
    risks = parse_risks(conclusion_text)
    overview = {key: clean_value(value) for key, value in overview.items()}
    conclusion = {key: clean_value(value) for key, value in conclusion.items()}
    flow_rows = normalize_table_rows(flow_header, flow_rows)
    issue_rows = normalize_table_rows(issue_header, issue_rows)
    industry_rows = normalize_table_rows(industry_header, industry_rows)
    baseline_text = first_nonempty(
        overview.get("评审基线", ""),
        overview.get("审查基线", ""),
        overview.get("评审分支和commit id", ""),
        overview.get("审查分支和commit id", ""),
        conclusion.get("结论基线", ""),
    )
    branch = clean_value(
        first_nonempty(
            overview.get("评审分支", ""),
            overview.get("审查分支", ""),
            parse_branch(baseline_text),
        )
    )
    commit = clean_value(
        first_nonempty(
            overview.get("评审提交", ""),
            overview.get("评审 commit id", ""),
            overview.get("审查提交", ""),
            parse_commit(baseline_text),
            parse_commit(transcript_text),
        )
    )
    scenario_name = clean_value(
        metadata.get("eval_name")
        or run_dir.name
        or run_dir.parent.parent.name.replace("eval-", " ").replace("-", " ").strip()
    ).replace("-", " ")
    if not scenario_name:
        scenario_name = "未命名场景"
    score = clean_value(extract_score(report_text) or conclusion.get("总体评分", ""))
    status = clean_value(conclusion.get("是否按文档走通", ""))
    refs = build_ref_map([report_text, transcript_text], repo_root)

    return {
        "scenario_name": scenario_name,
        "report_text": report_text,
        "overview": overview,
        "conclusion": conclusion,
        "score": score,
        "status": status,
        "risks": risks,
        "flow_header": flow_header,
        "flow_rows": flow_rows,
        "issue_header": issue_header,
        "issue_rows": issue_rows,
        "industry_header": industry_header,
        "industry_rows": industry_rows,
        "issues": issues,
        "newbie_notes": parse_list(
            get_section_text(sections, "7. 新手友好度观察", "新手友好度观察")
        ),
        "positive_notes": parse_list(
            get_section_text(sections, "9. 正向观察", "8. 正向观察", "正向观察")
        ),
        "priority_fixes": parse_list(
            get_section_text(
                sections, "10. 优先修复建议", "9. 优先修复建议", "优先修复建议"
            )
        ),
        "appendix": get_section_text(sections, "11. 附录", "10. 附录", "附录"),
        "grading": grading,
        "refs": refs,
        "branch": branch,
        "commit": commit,
    }


def render_table(
    headers: list[str], rows: list[list[str]], table_type: str, ref_map: dict[str, dict]
) -> str:
    rows = normalize_table_rows(headers, rows)
    if not headers or not rows:
        return ""
    parts = [f'<table class="report-table {table_type}"><thead><tr>']
    for header in headers:
        parts.append(f"<th>{inline_format(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for idx, cell in enumerate(row):
            classes = []
            if table_type == "flow" and idx == 3:
                classes.append(status_class(cell))
                classes.append("cell-center")
            if idx == len(row) - 1 and table_type in {"flow", "issues", "industry"}:
                classes.append(severity_class(cell))
                classes.append("cell-center")
            if table_type == "issues" and idx == 1:
                classes.append("cell-center")
            if table_type == "industry" and idx == 1:
                classes.append("cell-center")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            if table_type == "flow" and idx == 3:
                content = f"<span class='pill table-pill {status_class(cell)}'>{inline_format(cell)}</span>"
            elif table_type == "flow" and idx == len(row) - 1:
                content = f"<span class='pill table-pill {severity_class(cell)}'>{inline_format(cell)}</span>"
            elif table_type == "issues" and idx == 1:
                content = f"<span class='pill table-pill {severity_class(cell)}'>{inline_format(cell)}</span>"
            elif table_type == "industry" and idx == 1:
                content = f"<span class='pill table-pill {severity_class(cell)}'>{inline_format(cell)}</span>"
            else:
                content = (
                    render_ref_links(cell, ref_map)
                    if (".md:" in cell or "原文：`" in cell)
                    else inline_format(cell)
                )
            parts.append(f"<td{class_attr}>{content}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_issue_cards(issues: list[dict[str, str]], ref_map: dict[str, dict]) -> str:
    if not issues:
        return ""
    parts = ['<div class="issue-grid">']
    for issue in issues:
        severity = issue.get("严重程度", "")
        parts.append(f'<section class="issue-card {severity_class(severity)}">')
        parts.append(
            f'<div class="issue-card-header"><h4>{inline_format(issue.get("标题", "未命名问题"))}</h4>'
        )
        if severity:
            parts.append(
                f'<span class="pill {severity_class(severity)}">{inline_format(severity)}</span>'
            )
        parts.append("</div>")
        category = issue.get("分类", "")
        if category:
            parts.append(
                f'<div class="meta-line"><strong>分类：</strong>{inline_format(category)}</div>'
            )
        location = issue.get("文档位置", "")
        if location:
            parts.append(
                '<div class="meta-line"><strong>文档位置：</strong>'
                + render_ref_links(location, ref_map)
                + "</div>"
            )
        excerpt = issue.get("文档原文 / 摘录", "")
        if excerpt:
            parts.append(
                f'<div class="quote-box"><strong>原文摘录</strong>{render_simple_markdown(excerpt)}</div>'
            )
        for key in ["复现上下文", "实际现象", "影响分析", "修改建议"]:
            value = issue.get(key, "")
            if value:
                parts.append(
                    f'<div class="issue-block"><div class="issue-label">{inline_format(key)}</div>{render_simple_markdown(value)}</div>'
                )
        parts.append("</section>")
    parts.append("</div>")
    return "".join(parts)


def render_source_refs(ref_map: dict[str, dict]) -> str:
    if not ref_map:
        return ""
    parts = [
        '<section class="source-section"><h3>文档依据摘录</h3><p class="section-note">点击报告中的文档位置可直接跳转到这里，查看本地源码中的对应行内容。</p>'
    ]
    for ref in sorted(ref_map.values(), key=lambda item: item["label"]):
        parts.append(f'<article class="source-card" id="{ref["id"]}">')
        parts.append(
            f'<div class="source-label">{html.escape(ref["label"])}<span>{html.escape(ref["file"])}</span></div>'
        )
        parts.append(f"<pre><code>{html.escape(ref['excerpt'])}</code></pre>")
        parts.append("</article>")
    parts.append("</section>")
    return "".join(parts)


def build_html(scenarios: list[dict], report_title: str) -> str:
    scores = []
    for scenario in scenarios:
        score = scenario.get("score", "")
        match = re.search(r"(\d+)", score)
        if match:
            scores.append(int(match.group(1)))
    avg_score = round(sum(scores) / len(scores)) if scores else None
    repo = scenarios[0]["overview"].get("仓库", "") if scenarios else ""

    html_parts = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(report_title)}</title>",
        "<style>",
        "body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f6f4ee;color:#161512;line-height:1.65}",
        ".page{max-width:1220px;margin:0 auto;padding:32px 24px 72px}",
        ".hero{background:linear-gradient(135deg,#1a1b18,#32352e);color:#fff;border-radius:20px;padding:28px 30px 24px;margin-bottom:22px;box-shadow:0 18px 40px rgba(0,0,0,.18)}",
        ".hero h1{margin:0 0 8px;font-size:2rem;line-height:1.2}",
        ".hero p{margin:0;color:rgba(255,255,255,.82)}",
        ".hero-meta{display:flex;flex-wrap:wrap;gap:12px 18px;margin-top:18px;font-size:.92rem}",
        ".hero-meta a{color:#ffd2c6}",
        ".score-big{display:inline-flex;align-items:center;justify-content:center;min-width:100px;padding:10px 16px;border-radius:999px;background:#f1d7b0;color:#4a2c12;font-weight:700}",
        ".summary-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:20px 0 28px}",
        ".card{background:#fff;border:1px solid #e5dfd2;border-radius:16px;padding:16px 18px;box-shadow:0 8px 18px rgba(24,24,20,.05)}",
        ".card h3{margin:0 0 8px;font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;color:#7b7467}",
        ".card .value{font-size:1.28rem;font-weight:700;color:#181714}",
        ".scenario{margin-top:28px;background:#fff;border:1px solid #e5dfd2;border-radius:22px;padding:26px 24px;box-shadow:0 14px 30px rgba(24,24,20,.06)}",
        ".scenario h2{margin:0 0 12px;font-size:1.5rem}",
        ".meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:18px}",
        ".meta-item{background:#faf8f2;border:1px solid #ece6d8;border-radius:12px;padding:12px 14px}",
        ".meta-item .label{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;color:#8a826f;margin-bottom:6px}",
        ".meta-item .content{font-size:.96rem}",
        ".meta-item a{color:#a24a2a}",
        ".pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;font-size:.74rem;font-weight:700;letter-spacing:.02em}",
        ".sev-blocker{background:#f8d6d6;color:#8f2020}.sev-high{background:#fde4cf;color:#9a531e}.sev-medium{background:#f4ebc8;color:#7b6615}.sev-low{background:#dff0db;color:#2f6b34}.sev-neutral{background:#ece7dc;color:#645f55}",
        ".status-ok{background:#dff0db;color:#2f6b34}.status-blocked{background:#f8d6d6;color:#8f2020}.status-deviation{background:#fde4cf;color:#9a531e}.status-skipped{background:#ece7dc;color:#645f55}.status-neutral{background:#ece7dc;color:#645f55}",
        ".section-title{margin:28px 0 12px;font-size:1.08rem;font-weight:800}",
        ".risk-list,.note-list{margin:10px 0 0;padding-left:18px}",
        ".timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin:16px 0 20px}",
        ".timeline-card{border:1px solid #e8e1d4;border-radius:16px;padding:14px 15px;background:#fcfbf7}",
        ".timeline-card h4{margin:0 0 8px;font-size:1rem;line-height:1.35}",
        ".timeline-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}",
        ".timeline-block{margin-top:8px}.timeline-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;color:#8a826f;margin-bottom:3px}",
        ".report-table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5dfd2;border-radius:14px;overflow:hidden}",
        ".report-table th,.report-table td{border:1px solid #e8e1d4;padding:10px 12px;vertical-align:top;text-align:left;font-size:.9rem}",
        ".report-table th{background:#f3efe5;color:#5b5548;font-size:.76rem;text-transform:uppercase;letter-spacing:.04em}",
        ".issue-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}",
        ".issue-card{border:1px solid #e5dfd2;border-radius:18px;padding:18px 18px 16px;background:#fff;box-shadow:0 10px 22px rgba(24,24,20,.05)}",
        ".issue-card-header{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px}",
        ".issue-card h4{margin:0;font-size:1.06rem;line-height:1.35}",
        ".meta-line{font-size:.9rem;margin:8px 0}.meta-line strong{color:#5f584a}",
        ".quote-box{background:#f8f4ea;border:1px solid #e6decd;border-radius:12px;padding:12px 13px;margin:12px 0}",
        ".quote-box strong{display:block;margin-bottom:6px}",
        ".issue-block{margin-top:12px}.issue-label{font-size:.75rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:#857d6f;margin-bottom:4px}",
        ".doc-links{display:flex;flex-wrap:wrap;gap:8px}.doc-link{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;background:#f0ebe1;color:#8c4b2e;text-decoration:none;font-size:.82rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.doc-link:hover{background:#e7ddcb}",
        ".source-section{margin-top:18px}.source-card{border:1px solid #e6decd;border-radius:14px;background:#faf8f2;padding:12px 14px;margin-bottom:12px}",
        ".source-label{display:flex;justify-content:space-between;gap:12px;align-items:center;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.84rem;margin-bottom:10px;color:#5c5449}",
        ".source-label span{color:#918978;font-size:.78rem}",
        "pre{background:#151618;color:#f4f2eb;border-radius:12px;padding:14px;overflow:auto;font-size:.84rem;line-height:1.55} code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}",
        "table code,p code,li code,div code{background:#f0ebe1;border:1px solid #e3dac7;border-radius:4px;padding:1px 5px;color:#4d463a}",
        ".section-note{color:#756d5f;font-size:.92rem}",
        "@media (max-width: 720px){.page{padding:20px 14px 52px}.hero{padding:22px 18px}.scenario{padding:20px 16px}}",
        "</style></head><body><div class='page'>",
        f"<section class='hero'><h1>{html.escape(report_title)}</h1><p>这是一份聚焦真实上手体验的最终报告页，只展示按照 skill 对仓库文档进行逐步体验后得到的审查结论，不展示 benchmark、反馈控件或基线翻页界面。</p>",
    ]
    html_parts.append("<div class='hero-meta'>")
    if repo:
        html_parts.append(
            f"<div><strong>仓库：</strong><a href='{html.escape(repo)}'>{html.escape(repo)}</a></div>"
        )
    html_parts.append(f"<div><strong>场景数：</strong>{len(scenarios)}</div>")
    if avg_score is not None:
        html_parts.append(f"<div class='score-big'>平均评分 {avg_score}/100</div>")
    html_parts.append("</div></section>")
    html_parts.append("<section class='summary-bar'>")
    html_parts.append(
        f"<div class='card'><h3>审查方式</h3><div class='value'>真实执行 + 新手视角</div></div>"
    )
    html_parts.append(
        f"<div class='card'><h3>输出形式</h3><div class='value'>单页 HTML 最终报告</div></div>"
    )
    html_parts.append(
        f"<div class='card'><h3>关注重点</h3><div class='value'>走通性 / 易用性 / 行业实践</div></div>"
    )
    html_parts.append("</section>")

    for scenario in scenarios:
        refs = scenario["refs"]
        overview = scenario["overview"]
        conclusion = scenario["conclusion"]
        html_parts.append("<section class='scenario'>")
        html_parts.append(f"<h2>{html.escape(scenario['scenario_name'])}</h2>")
        html_parts.append("<div class='meta-grid'>")
        meta_items = [
            ("仓库链接", overview.get("仓库", "")),
            ("评审提交", scenario.get("commit", "") or overview.get("评审提交", "")),
            ("审查时间", overview.get("审查时间", "")),
            ("体验环境", overview.get("体验环境", "")),
            ("已具备环境", overview.get("用户声明的已具备环境", "")),
            ("隔离策略", overview.get("采用的隔离策略", "")),
            ("总体评分", scenario.get("score", "")),
            ("走通状态", scenario.get("status", "")),
        ]
        for label, value in meta_items:
            if not value:
                continue
            content = (
                f"<a href='{html.escape(value)}'>{html.escape(value)}</a>"
                if label == "仓库链接"
                else inline_format(value)
            )
            html_parts.append(
                f"<div class='meta-item'><div class='label'>{html.escape(label)}</div><div class='content'>{content}</div></div>"
            )
        html_parts.append("</div>")

        if scenario["risks"]:
            html_parts.append(
                "<h3 class='section-title'>关键风险</h3><ul class='risk-list'>"
            )
            for risk in scenario["risks"]:
                html_parts.append(f"<li>{inline_format(risk)}</li>")
            html_parts.append("</ul>")

        if scenario["flow_rows"]:
            html_parts.append(
                "<h3 class='section-title'>体验流程</h3><div class='timeline'>"
            )
            for row in scenario["flow_rows"]:
                step = row[0] if len(row) > 0 else ""
                basis = row[1] if len(row) > 1 else ""
                expected = row[2] if len(row) > 2 else ""
                status = row[3] if len(row) > 3 else ""
                phenomenon = row[4] if len(row) > 4 else ""
                cause = row[5] if len(row) > 5 else ""
                severity = row[6] if len(row) > 6 else ""
                html_parts.append("<article class='timeline-card'>")
                html_parts.append(f"<h4>步骤 {inline_format(step)}</h4>")
                html_parts.append("<div class='timeline-meta'>")
                if status:
                    html_parts.append(
                        f"<span class='pill {status_class(status)}'>{inline_format(status)}</span>"
                    )
                if severity:
                    html_parts.append(
                        f"<span class='pill {severity_class(severity)}'>{inline_format(severity)}</span>"
                    )
                html_parts.append("</div>")
                if basis:
                    html_parts.append(
                        f"<div class='timeline-block'><div class='timeline-label'>文档依据</div>{render_ref_links(basis, refs)}</div>"
                    )
                if expected:
                    html_parts.append(
                        f"<div class='timeline-block'><div class='timeline-label'>预期动作</div><div>{inline_format(expected)}</div></div>"
                    )
                if phenomenon:
                    html_parts.append(
                        f"<div class='timeline-block'><div class='timeline-label'>现象 / 结果</div><div>{inline_format(phenomenon)}</div></div>"
                    )
                if cause:
                    html_parts.append(
                        f"<div class='timeline-block'><div class='timeline-label'>原因 / 成功依据</div><div>{inline_format(cause)}</div></div>"
                    )
                html_parts.append("</article>")
            html_parts.append("</div>")
            html_parts.append(
                render_table(
                    scenario["flow_header"], scenario["flow_rows"], "flow", refs
                )
            )

        if scenario["issue_rows"]:
            html_parts.append("<h3 class='section-title'>问题概览</h3>")
            html_parts.append(
                render_table(
                    scenario["issue_header"], scenario["issue_rows"], "issues", refs
                )
            )

        if scenario["issues"]:
            html_parts.append("<h3 class='section-title'>详细问题</h3>")
            html_parts.append(render_issue_cards(scenario["issues"], refs))

        if scenario["newbie_notes"]:
            html_parts.append(
                "<h3 class='section-title'>小白用户视角</h3><ul class='note-list'>"
            )
            for note in scenario["newbie_notes"]:
                html_parts.append(f"<li>{inline_format(note)}</li>")
            html_parts.append("</ul>")

        if scenario["industry_rows"]:
            html_parts.append(
                "<h3 class='section-title'>开源项目关键章节与业界实践检查</h3>"
            )
            html_parts.append(
                render_table(
                    scenario["industry_header"],
                    scenario["industry_rows"],
                    "industry",
                    refs,
                )
            )

        if scenario["positive_notes"]:
            html_parts.append(
                "<h3 class='section-title'>做得好的地方</h3><ul class='note-list'>"
            )
            for note in scenario["positive_notes"]:
                html_parts.append(f"<li>{inline_format(note)}</li>")
            html_parts.append("</ul>")

        if scenario["priority_fixes"]:
            html_parts.append(
                "<h3 class='section-title'>优先修复建议</h3><ol class='note-list'>"
            )
            for note in scenario["priority_fixes"]:
                html_parts.append(f"<li>{inline_format(note)}</li>")
            html_parts.append("</ol>")

        html_parts.append(render_source_refs(refs))
        html_parts.append("</section>")

    html_parts.append("</div></body></html>")
    return "".join(html_parts)


def scenario_display_name(name: str) -> str:
    key = name.lower().replace("_", "-")
    if "macos" in key and "source" in key:
        return "场景 A：macOS 源码编译路径"
    if "skip" in key and "toolkit" in key:
        return "场景 B：已有基础环境，跳过 toolkit 安装"
    if "linux" in key and "install" in key:
        return "场景：Linux 隔离安装路径"
    return name.replace("-", " ")


def scenario_meaning(name: str) -> str:
    key = name.lower().replace("_", "-")
    if "macos" in key and "source" in key:
        return "代表没有现成 macOS 安装包时，用户是否能仅凭 README 和安装文档走通源码构建路径。"
    if "skip" in key and "toolkit" in key:
        return "代表用户已具备部分基础环境，希望跳过重复安装，只验证剩余文档流程是否能继续走通。"
    if "linux" in key and "install" in key:
        return "代表在隔离 Linux 环境中，用户从零开始按文档完成安装与运行验证。"
    return "代表一个真实的文档体验场景。"


def step_phase_name(expected: str) -> str:
    text = expected.lower()
    if "源码" in expected or "clone" in text or "仓库" in expected:
        return "准备源码"
    if "软件包" in expected or "发布包" in expected or "校验" in expected:
        return "获取安装包"
    if "判断" in expected or "是否可走" in expected or "路径" in expected:
        return "选择路径"
    if "编译" in expected:
        return "源码构建"
    if "安装" in expected or "whl" in text or "plugin" in text:
        return "安装组件"
    if "启动" in expected and ("daemon" in text or "dynolog" in text):
        return "启动服务"
    if "验证" in expected or "status" in text:
        return "验证结果"
    if "训练" in expected or "推理" in expected or "task" in text:
        return "运行样例"
    if "monitor" in text or "trace" in text:
        return "功能验证"
    return "执行步骤"


def step_purpose(expected: str, status: str, severity: str) -> str:
    base = expected or "执行当前文档步骤"
    if "OK" in status or "完全走通" in status:
        tail = "这一环已按文档完成，可以作为后续步骤继续推进的可靠前提。"
    elif "阻塞" in status or "未走通" in status:
        tail = "这一环已经成为主流程中断点，如果不补齐前置条件或修正文档，后续步骤无法正常继续。"
    elif "偏差继续" in status or "部分走通" in status:
        tail = "这一环需要带着执行偏差继续，说明文档对环境或路径存在隐含假设。"
    elif "未执行" in status:
        tail = "这一环没有进入执行，通常是因为前序步骤已经阻塞。"
    else:
        tail = "这一环用于确认当前文档步骤是否能稳定成立。"

    if "阻塞" in severity:
        tail += " 从审查优先级看，这也是一个需要优先修复的关键节点。"
    elif severity == "高":
        tail += " 这一步虽然不一定立刻卡死，但会显著增加新手试错成本。"
    return f"目标：{base}。{tail}"


def common_styles() -> str:
    return """
body{margin:0;background:#f5f2ea;color:#171612;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.62}
*{box-sizing:border-box}
a,td,th,div,p,li,span{overflow-wrap:anywhere;word-break:break-word}
.page{max-width:1160px;margin:0 auto;padding:28px 18px 64px}
.topbar{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;align-items:center;margin-bottom:18px}
.topbar a{color:#8a4b2d;text-decoration:none;font-weight:600}
.hero{background:#1d1f1b;color:#fff;border-radius:18px;padding:22px 22px 18px;box-shadow:0 12px 28px rgba(0,0,0,.16)}
.hero h1{margin:0;font-size:1.68rem;line-height:1.22;font-weight:800;letter-spacing:-.01em}
.hero-meta{display:flex;flex-wrap:wrap;gap:10px 16px;margin-top:14px;color:rgba(255,255,255,.84);font-size:.9rem}
.hero-meta a{color:#ffd7ca}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:18px 0 24px}
.mini-card{background:#fff;border:1px solid #e6ded0;border-radius:14px;padding:14px 15px;box-shadow:0 8px 18px rgba(20,20,18,.05)}
.mini-card .label{font-size:.73rem;text-transform:uppercase;letter-spacing:.05em;color:#817869;margin-bottom:6px}
.mini-card .value{font-size:1rem;font-weight:650;color:#171612;line-height:1.45}
.section{background:#fff;border:1px solid #e6ded0;border-radius:18px;padding:22px 20px;margin-top:18px;box-shadow:0 10px 24px rgba(20,20,18,.05)}
.section h2{margin:0 0 14px;font-size:1.28rem;line-height:1.28;font-weight:800}
.section h3{margin:0 0 11px;font-size:1.06rem;line-height:1.36;font-weight:760}
.section p{font-size:.97rem;line-height:1.7;color:#1d1b17}
.summary-list,.note-list{margin:10px 0 0;padding-left:18px}
.summary-list li,.note-list li{margin-top:7px;font-size:.96rem}
.scenario-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.scenario-card{background:#fcfbf7;border:1px solid #e8dfd0;border-radius:16px;padding:18px 16px;text-decoration:none;color:inherit;display:block;box-shadow:0 8px 18px rgba(20,20,18,.04)}
.scenario-card:hover{border-color:#d0b79f;transform:translateY(-1px)}
.scenario-card h3{margin:0 0 8px;font-size:1.08rem;line-height:1.35}
.scenario-card p{margin:0 0 10px;color:#5f584d;font-size:.94rem;line-height:1.62}
.meta-row{display:flex;flex-wrap:wrap;gap:8px 10px;margin-top:12px}
.pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;font-size:.74rem;font-weight:750;line-height:1.2}
.table-pill{justify-content:center;min-width:72px;max-width:132px;white-space:normal;text-align:center;padding:6px 12px;font-size:.78rem;line-height:1.3}
.sev-blocker{background:#f8d6d6;color:#8f2020}.sev-high{background:#fde4cf;color:#9a531e}.sev-medium{background:#f6edca;color:#7b6615}.sev-low{background:#dff0db;color:#2f6b34}.sev-neutral{background:#ece7dc;color:#645f55}
.status-ok{background:#dff0db;color:#2f6b34}.status-blocked{background:#f8d6d6;color:#8f2020}.status-deviation{background:#fde4cf;color:#9a531e}.status-skipped{background:#ece7dc;color:#645f55}.status-neutral{background:#ece7dc;color:#645f55}
.report-table{width:100%;border-collapse:collapse;border:1px solid #e8dfd0;border-radius:14px;overflow:hidden;background:#fff}
.report-table th,.report-table td{border:1px solid #e8dfd0;padding:11px 12px;vertical-align:top;text-align:left;font-size:.92rem;line-height:1.58}
.report-table th{background:#f4efe4;color:#645c4f;font-size:.83rem;text-transform:uppercase;letter-spacing:.04em;font-weight:800;text-align:center;vertical-align:middle}
.report-table tbody tr:nth-child(even) td{background:#fcfaf5}
.report-table td.cell-center{text-align:center;vertical-align:middle}
.timeline{display:grid;grid-template-columns:1fr;gap:16px;margin:12px 0 18px}
.timeline-card{border:1px solid #e6ddd0;border-left:4px solid #d3ba9b;border-radius:18px;background:linear-gradient(180deg,#fcfbf7 0%,#fff 100%);padding:16px 16px 15px;box-shadow:0 10px 20px rgba(20,20,18,.05)}
.timeline-card h4{margin:0 0 6px;font-size:1.1rem;line-height:1.4;font-weight:820;letter-spacing:-.01em}
.timeline-step-tag{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;background:#efe5d6;color:#7c5936;font-size:.73rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin-bottom:9px}
.timeline-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.timeline-block{margin-top:10px;padding:11px 12px;border-radius:12px;background:#fff;border:1px solid #ece3d6}
.timeline-label{display:inline-flex;align-items:center;padding:4px 8px;border-radius:999px;background:#f2ebe0;color:#6b5a43;font-size:.76rem;letter-spacing:.03em;font-weight:800;margin-bottom:8px}
.timeline-content{font-size:.97rem;line-height:1.66;color:#1d1b17}
.timeline-block.intro{background:#f8f3eb;border-color:#e7dccb}
.issue-grid{display:grid;grid-template-columns:1fr;gap:18px}
.issue-card{border:1px solid #e8dfd0;border-radius:16px;background:#fff;padding:16px 15px;box-shadow:0 8px 18px rgba(20,20,18,.04)}
.issue-card h4{margin:0;font-size:1.08rem;line-height:1.42;font-weight:800}
.issue-card-header{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:10px}
.meta-line{margin:8px 0;font-size:.96rem;line-height:1.62}
.quote-box{background:#f8f4ea;border:1px solid #e7decd;border-radius:12px;padding:12px 13px;margin:13px 0}
.issue-label{display:inline-flex;align-items:center;padding:4px 8px;border-radius:999px;background:#f2ebe0;font-size:.73rem;text-transform:uppercase;letter-spacing:.04em;font-weight:800;color:#6f614d;margin-bottom:7px}
.issue-block{margin-top:13px}
.issue-block p,.issue-block li,.quote-box p,.quote-box li{font-size:.96rem;line-height:1.68}
.doc-links{display:flex;flex-wrap:wrap;gap:8px}.doc-link{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;background:#f0ebe1;color:#8c4b2e;text-decoration:none;font-size:.81rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.doc-link:hover{background:#e6ddcb}
.details-wrap{margin-top:18px}.details-wrap details{border:1px solid #e7decd;border-radius:14px;background:#faf8f2;padding:12px 14px}.details-wrap summary{cursor:pointer;font-weight:700;color:#5b5447}
.source-card{border:1px solid #e6ddcd;border-radius:12px;background:#fff;padding:12px 12px;margin-top:12px}.source-label{display:flex;justify-content:space-between;gap:10px;align-items:center;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.83rem;color:#5e564a;margin-bottom:8px}.source-label span{color:#8b8477;font-size:.77rem}
pre{background:#17191c;color:#f6f3ec;border-radius:12px;padding:13px;overflow:auto;font-size:.83rem;line-height:1.55}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table code,p code,li code,div code{background:#f0ebe1;border:1px solid #e3dac7;border-radius:4px;padding:1px 5px;color:#4d463a}
.backlink{display:inline-flex;align-items:center;gap:6px;color:#8a4b2d;text-decoration:none;font-weight:700}
@media (max-width:720px){.page{padding:18px 12px 48px}.hero{padding:18px 16px}.section{padding:16px 14px}.hero h1{font-size:1.42rem}.section h2{font-size:1.16rem}.timeline-card h4{font-size:1rem}.mini-card .value{font-size:.96rem}}
"""


def render_source_refs_collapsible(ref_map: dict[str, dict]) -> str:
    if not ref_map:
        return ""
    return (
        '<div class="details-wrap"><details><summary>查看文档依据摘录</summary>'
        + render_source_refs(ref_map)
        + "</details></div>"
    )


def build_overall_findings(scenarios: list[dict]) -> list[str]:
    findings = []
    if not scenarios:
        return findings
    blocked = sum(1 for s in scenarios if "未走通" in s.get("status", ""))
    partial = sum(1 for s in scenarios if "部分" in s.get("status", ""))
    findings.append(
        f"本轮共覆盖 {len(scenarios)} 个代表性场景，其中未走通 {blocked} 个，部分走通 {partial} 个。"
    )
    top_issue_texts = []
    seen = set()
    for scenario in scenarios:
        summary = scenario_top_issue(scenario)
        if summary != "未识别关键问题" and summary not in seen:
            seen.add(summary)
            top_issue_texts.append(summary)
    for text in top_issue_texts[:3]:
        findings.append(text)
    if not top_issue_texts:
        findings.append("当前未从问题概览或详细问题中识别出明确的关键问题，报告可能缺少该章节或仅输出了空表头。")
    return findings


def build_index_html(
    scenarios: list[dict], report_title: str, page_links: dict[str, str]
) -> str:
    scores = []
    for scenario in scenarios:
        match = re.search(r"(\d+)", scenario.get("score", ""))
        if match:
            scores.append(int(match.group(1)))
    avg_score = round(sum(scores) / len(scores)) if scores else None
    repo = scenarios[0]["overview"].get("仓库", "") if scenarios else ""
    branches = sorted({s.get("branch", "") for s in scenarios if s.get("branch", "")})
    commits = sorted({s.get("commit", "") for s in scenarios if s.get("commit", "")})
    branch_text = (
        branches[0] if len(branches) == 1 else f"{len(branches)} 个评审分支" if branches else ""
    )
    commit_text = (
        commits[0] if len(commits) == 1 else f"{len(commits)} 个提交快照" if commits else ""
    )

    parts = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(report_title)}</title><style>{common_styles()}</style></head><body><div class='page'>",
        f"<section class='hero'><h1>{html.escape(report_title)}</h1><div class='hero-meta'>",
    ]
    if repo:
        parts.append(
            f"<div><strong>仓库：</strong><a href='{html.escape(repo)}'>{html.escape(repo)}</a></div>"
        )
    if branch_text:
        parts.append(
            f"<div><strong>评审分支：</strong>{html.escape(branch_text)}</div>"
        )
    if commit_text:
        parts.append(
            f"<div><strong>评审提交：</strong>{html.escape(commit_text)}</div>"
        )
    parts.append(f"<div><strong>场景数：</strong>{len(scenarios)}</div>")
    if avg_score is not None:
        parts.append(f"<div><strong>平均评分：</strong>{avg_score}/100</div>")
    parts.append("</div></section>")

    parts.append("<section class='section'><h2>总体结论</h2><ul class='summary-list'>")
    for finding in build_overall_findings(scenarios):
        parts.append(f"<li>{inline_format(finding)}</li>")
    parts.append("</ul></section>")

    parts.append(
        "<section class='section'><h2>场景目录</h2><div class='scenario-grid'>"
    )
    for scenario in scenarios:
        filename = scenario.get("page_filename") or page_links.get(
            scenario["scenario_name"], "#"
        )
        display_name = scenario_display_name(scenario["scenario_name"])
        meaning = scenario_meaning(scenario["scenario_name"])
        status = scenario.get("status", "")
        score = scenario.get("score", "")
        top_issue = scenario_top_issue(scenario)
        parts.append(
            f"<a class='scenario-card' href='{html.escape(filename)}'><h3>{html.escape(display_name)}</h3>"
        )
        parts.append(f"<p>{inline_format(meaning)}</p>")
        parts.append(f"<p><strong>代表问题：</strong>{inline_format(top_issue)}</p>")
        parts.append("<div class='meta-row'>")
        if score:
            parts.append(
                f"<span class='pill sev-neutral'>评分 {inline_format(score)}</span>"
            )
        if status:
            parts.append(
                f"<span class='pill {status_class(status)}'>{inline_format(status)}</span>"
            )
        blocker_count = sum(
            1
            for row in scenario.get("issue_rows", [])
            if len(row) > 1 and "阻塞" in row[1]
        )
        if blocker_count:
            parts.append(
                f"<span class='pill sev-blocker'>阻塞问题 {blocker_count}</span>"
            )
        parts.append("</div></a>")
    parts.append("</div></section></div></body></html>")
    return "".join(parts)


def build_scenario_html(
    scenario: dict, report_title: str, page_links: dict[str, str], index_name: str
) -> str:
    refs = scenario["refs"]
    overview = scenario["overview"]
    display_name = scenario_display_name(scenario["scenario_name"])
    meaning = scenario_meaning(scenario["scenario_name"])

    parts = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(display_name)} - {html.escape(report_title)}</title>",
        f"<style>{common_styles()}</style></head><body><div class='page'>",
        f"<div class='topbar'><a class='backlink' href='{html.escape(index_name)}'>← 返回总览</a></div>",
        f"<section class='hero'><h1>{html.escape(display_name)}</h1><div class='hero-meta'><div>{inline_format(meaning)}</div></div></section>",
    ]

    parts.append("<section class='section'><h2>总体结论</h2><div class='card-grid'>")
    meta_items = [
        ("总体评分", scenario.get("score", "")),
        ("走通状态", scenario.get("status", "")),
        ("仓库链接", overview.get("仓库", "")),
        ("评审分支", scenario.get("branch", "") or overview.get("评审分支", "")),
        ("评审提交", scenario.get("commit", "") or overview.get("评审提交", "")),
        ("审查时间", overview.get("审查时间", "")),
        ("体验环境", overview.get("体验环境", "")),
        ("已具备环境", overview.get("用户声明的已具备环境", "")),
        ("隔离策略", overview.get("采用的隔离策略", "")),
    ]
    for label, value in meta_items:
        if not value:
            continue
        content = (
            f"<a href='{html.escape(value)}'>{html.escape(value)}</a>"
            if label == "仓库链接"
            else inline_format(value)
        )
        parts.append(
            f"<div class='mini-card'><div class='label'>{html.escape(label)}</div><div class='value'>{content}</div></div>"
        )
    parts.append("</div>")
    conclusion = scenario.get("conclusion", {})
    if conclusion.get("总体评价"):
        parts.append(f"<p>{inline_format(conclusion['总体评价'])}</p>")
    if scenario["risks"]:
        parts.append("<h3>关键风险</h3><ul class='summary-list'>")
        for risk in scenario["risks"]:
            parts.append(f"<li>{inline_format(risk)}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    if scenario["flow_rows"]:
        parts.append("<section class='section'><h2>体验流程</h2><div class='timeline'>")
        for row in scenario["flow_rows"]:
            step = row[0] if len(row) > 0 else ""
            basis = row[1] if len(row) > 1 else ""
            expected = row[2] if len(row) > 2 else ""
            status = row[3] if len(row) > 3 else ""
            phenomenon = row[4] if len(row) > 4 else ""
            cause = row[5] if len(row) > 5 else ""
            severity = row[6] if len(row) > 6 else ""
            phase = step_phase_name(expected)
            purpose = step_purpose(expected, status, severity)
            step_title = f"步骤 {step}：{expected}" if expected else f"步骤 {step}"
            parts.append("<article class='timeline-card'>")
            parts.append(
                f"<div class='timeline-step-tag'>{inline_format(phase)}</div><h4>{inline_format(step_title)}</h4><div class='timeline-meta'>"
            )
            if status:
                parts.append(
                    f"<span class='pill {status_class(status)}'>{inline_format(status)}</span>"
                )
            if severity:
                parts.append(
                    f"<span class='pill {severity_class(severity)}'>{inline_format(severity)}</span>"
                )
            parts.append("</div>")
            parts.append(
                f"<div class='timeline-block intro'><div class='timeline-label'>本步说明</div><div class='timeline-content'>{inline_format(purpose)}</div></div>"
            )
            if basis:
                parts.append(
                    f"<div class='timeline-block'><div class='timeline-label'>文档依据</div><div class='timeline-content'>{render_ref_links(basis, refs)}</div></div>"
                )
            if phenomenon:
                parts.append(
                    f"<div class='timeline-block'><div class='timeline-label'>结果</div><div class='timeline-content'>{inline_format(phenomenon)}</div></div>"
                )
            if cause:
                parts.append(
                    f"<div class='timeline-block'><div class='timeline-label'>判断依据</div><div class='timeline-content'>{inline_format(cause)}</div></div>"
                )
            parts.append("</article>")
        parts.append("</div></section>")

    if scenario["issue_rows"]:
        parts.append("<section class='section'><h2>问题概览</h2>")
        parts.append(
            render_table(
                scenario["issue_header"], scenario["issue_rows"], "issues", refs
            )
        )
        parts.append("</section>")

    if scenario["issues"]:
        parts.append("<section class='section'><h2>详细问题</h2>")
        parts.append(render_issue_cards(scenario["issues"], refs))
        parts.append("</section>")

    if scenario["newbie_notes"]:
        parts.append(
            "<section class='section'><h2>小白用户视角</h2><ul class='note-list'>"
        )
        for note in scenario["newbie_notes"]:
            parts.append(f"<li>{inline_format(note)}</li>")
        parts.append("</ul></section>")

    if scenario["industry_rows"]:
        parts.append("<section class='section'><h2>开源项目关键章节与业界实践检查</h2>")
        parts.append(
            render_table(
                scenario["industry_header"], scenario["industry_rows"], "industry", refs
            )
        )
        parts.append("</section>")

    if scenario["priority_fixes"]:
        parts.append(
            "<section class='section'><h2>优先修复建议</h2><ol class='note-list'>"
        )
        for note in scenario["priority_fixes"]:
            parts.append(f"<li>{inline_format(note)}</li>")
        parts.append("</ol></section>")

    parts.append(
        f"<section class='section'><h2>文档依据</h2><p class='section-note'>点击正文中的文档位置会跳到下方对应摘录。为避免正文过长，这部分默认折叠。</p>{render_source_refs_collapsible(refs)}</section>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a document UX review HTML report set (overview page plus scenario detail pages)"
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing outputs/report.md",
    )
    parser.add_argument(
        "--repo-root", required=True, help="Local repository root for source excerpts"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Overview HTML path inside the report output folder; scenario pages will be written beside it",
    )
    parser.add_argument("--title", default="文档体验审查最终报告", help="Page title")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    scenarios = [
        scenario_from_run_dir(Path(run_dir), repo_root) for run_dir in args.run_dir
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_links: dict[str, str] = {}
    for idx, scenario in enumerate(scenarios, start=1):
        filename = f"scenario-{idx}-{slugify(scenario['scenario_name'])}.html"
        page_links[scenario["scenario_name"]] = filename
        scenario["page_filename"] = filename

    index_html = build_index_html(scenarios, args.title, page_links)
    output_path.write_text(index_html, encoding="utf-8")

    for scenario in scenarios:
        scenario_path = output_path.parent / scenario["page_filename"]
        scenario_html = build_scenario_html(
            scenario, args.title, page_links, output_path.name
        )
        scenario_path.write_text(scenario_html, encoding="utf-8")


if __name__ == "__main__":
    main()
