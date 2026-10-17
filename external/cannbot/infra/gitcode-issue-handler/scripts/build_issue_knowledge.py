#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Build an evidence-first corpus from historical GitCode issues.

The script is intentionally conservative: it records observable signals from
issue text, comments, state, and linked PR descriptions.  It does not claim a
root cause or a successful fix unless the public discussion contains evidence.

The detailed JSON is runtime input for curation.  The Markdown report is a
compact audit artifact and candidate list; neither is a replacement for a
human-reviewed knowledge card.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from runtime_paths import (  # noqa: E402
    KNOWLEDGE_CACHE,
    KNOWLEDGE_CORPUS,
    KNOWLEDGE_REPORT,
    path_text,
    rate_limit_path,
)
from cli_output import write_stdout  # noqa: E402
import fetch_issues as issue_api  # noqa: E402
from fetch_cache import load_comments, save_comments  # noqa: E402

ISSUE_TYPE_RULES = (
    ("documentation", ("[documentation|", "文档反馈", "文档", "readme")),
    ("question", ("[question|", "问题咨询", "咨询", "请问", "如何")),
    ("performance", ("性能优化", "性能问题", "performance", "perf")),
    ("bug", ("[bug-report|", "缺陷反馈", "bug", "报错", "错误", "崩溃", "精度问题")),
    ("requirement", ("[requirement|", "需求建议", "新增", "支持")),
    ("test", ("[test]", "[test|", "单元测试", "系统测试", "测试用例")),
)

SIGNAL_RULES = {
    "change_claimed": (
        "已修复",
        "已修改",
        "已合入",
        "已解决",
        "修复完成",
        "修改完成",
        "fixed",
        "merged",
        "resolved by",
        "提交pr",
        "提交 pr",
    ),
    "answer_or_clarification": (
        "不支持",
        "支持",
        "可以",
        "原因是",
        "这是因为",
        "请参考",
        "用法",
        "预期行为",
        "设计如此",
        "结论",
        "目前",
        "当前版本",
    ),
    "more_information_requested": (
        "请提供",
        "请补充",
        "补充一下",
        "复现步骤",
        "复现用例",
        "报错日志",
        "环境信息",
        "版本信息",
        "能否提供",
        "麻烦提供",
    ),
    "duplicate_or_redirected": (
        "重复issue",
        "重复 issue",
        "duplicate",
        "已有issue",
        "已有 issue",
        "请转到",
        "转至",
        "不属于本仓",
        "其他仓库",
    ),
}

_ASSIGN_RE = re.compile(r"(?im)^\s*/assign\s+\[?@[^\s\]]+")
_PR_URL_RE = re.compile(
    r"https?://gitcode\.com/[^\s)]+/(?:pull|pulls|merge_requests)/(\d+)",
    re.IGNORECASE,
)
_ISSUE_REF_RE = re.compile(r"(?<!#)#(\d+)\b")
_ISSUE_URL_RE = re.compile(r"/issues/(\d+)\b")
LOGGER = logging.getLogger(__name__)


def _write_stdout(text):
    """Write the JSON result protocol to stdout."""
    write_stdout(text)


def classify_issue_type(issue: dict) -> str:
    """Classify only from explicit title/body markers."""
    text = f"{issue.get('title', '')}\n{issue.get('description', '')}".lower()
    labels = " ".join(issue.get("labels") or []).lower()
    searchable = f"{text}\n{labels}"
    for kind, markers in ISSUE_TYPE_RULES:
        if any(marker in searchable for marker in markers):
            return kind
    return "other"


def extract_issue_references(text: str) -> set[str]:
    """Return Issue numbers referenced by shorthand or GitCode Issue URLs."""
    return set(_ISSUE_REF_RE.findall(text)) | set(_ISSUE_URL_RE.findall(text))


def evidence_signals(issue: dict) -> list[str]:
    comments = issue.get("comments") or []
    comment_text = "\n".join(comment.get("body") or "" for comment in comments)
    lowered = comment_text.lower()
    signals = []
    if issue.get("linked_prs") or _PR_URL_RE.search(comment_text):
        signals.append("linked_change")
    if _ASSIGN_RE.search(comment_text):
        signals.append("assignment")
    for signal, markers in SIGNAL_RULES.items():
        if any(marker in lowered for marker in markers):
            signals.append(signal)
    if comments and not signals:
        signals.append("other_reply")
    if issue.get("state") == "closed" and not comments and not issue.get("linked_prs"):
        signals.append("closed_without_textual_evidence")
    if issue.get("state") == "open" and not comments and not issue.get("linked_prs"):
        signals.append("unresolved_without_reply")
    return signals


def handling_outcome(issue: dict) -> str:
    """Select a coarse outcome from evidence signals without inferring success."""
    signals = set(issue.get("evidence_signals") or [])
    if "linked_change" in signals:
        return "linked_change"
    if "change_claimed" in signals:
        return "change_claimed_in_comment"
    if "duplicate_or_redirected" in signals:
        return "duplicate_or_redirected"
    if issue.get("issue_type") == "question" and "answer_or_clarification" in signals:
        return "answered_or_clarified"
    if "more_information_requested" in signals:
        return "more_information_requested"
    if "assignment" in signals and len(signals) == 1:
        return "assignment_only"
    if "closed_without_textual_evidence" in signals:
        return "closed_without_textual_evidence"
    if "unresolved_without_reply" in signals:
        return "unresolved_without_reply"
    if issue.get("comments"):
        return "replied_without_strong_outcome_evidence"
    return "no_outcome_evidence"


class RequestLimiter:
    """Serialize request starts so GitCode's per-minute limit is respected."""

    def __init__(self, interval_seconds):
        self.interval = max(0.0, interval_seconds)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.interval
        if delay:
            time.sleep(delay)


def rate_limited_get(limiter, api, url, params=None):
    """Pace request starts; gitcode_client is the sole retry owner."""
    limiter.wait()
    return issue_api.api_get(api.session, url, api.token, params=params)


def fetch_comments(
    issues,
    api,
    workers,
    limiter,
    cache_dir=None,
    *,
    refresh=False,
):
    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    def _one(issue):
        number = issue.get("iid")
        if not (issue.get("comments_count") or 0):
            return number, []
        repo_key = f"{api.owner}/{api.repo}"
        cached = (
            None
            if refresh or cache_root is None
            else load_comments(str(cache_root), repo_key, issue)
        )
        if cached is not None:
            return number, cached

        thread_api = issue_api.RepoApiContext(
            issue_api.make_session(rate_limit_dir=rate_limit_path(cache_dir)),
            api.api_base,
            api.owner,
            api.repo,
            api.token,
        )
        url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues/{number}/comments"
        raw_comments = []
        page = 1
        while True:
            data = rate_limited_get(
                limiter,
                thread_api,
                url,
                params={"page": page, "per_page": 100, "sort": "asc"},
            )
            batch = data if isinstance(data, list) else []
            if not batch:
                break
            raw_comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        comments = [
            {
                "id": comment.get("id"),
                "author": (comment.get("user") or {}).get("login")
                or (comment.get("author") or {}).get("login")
                or "unknown",
                "body": comment.get("body", ""),
                "created_at": comment.get("created_at", ""),
                "updated_at": comment.get("updated_at", ""),
            }
            for comment in raw_comments
        ]
        if cache_root:
            save_comments(str(cache_root), repo_key, issue, comments)
        return number, comments

    comments_by_number = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, issue): issue.get("iid") for issue in issues}
        for future in as_completed(futures):
            number, comments = future.result()
            comments_by_number[str(number)] = comments
    for issue in issues:
        issue["comments"] = comments_by_number.get(str(issue.get("iid")), [])


def fetch_all_prs(api, limiter):
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/pulls"
    result = []
    page = 1
    while True:
        data = rate_limited_get(
            limiter,
            api,
            url,
            params={
                "state": "all",
                "page": page,
                "per_page": 100,
                "sort": "created",
                "direction": "desc",
            },
        )
        batch = data if isinstance(data, list) else []
        if not batch:
            break
        result.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return result


def build_pr_map(prs, issue_numbers):
    issue_numbers = {str(number) for number in issue_numbers}
    mapping = defaultdict(list)
    for pr in prs:
        text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
        refs = extract_issue_references(text)
        info = {
            "number": pr.get("number"),
            "state": pr.get("state"),
            "title": pr.get("title") or "",
            "url": pr.get("html_url") or "",
        }
        for ref in refs & issue_numbers:
            mapping[ref].append(info)
    return mapping


def compact_issue(issue):
    """Keep the public evidence needed by later curation."""
    return {
        "number": issue.get("iid"),
        "url": issue.get("url"),
        "title": issue.get("title"),
        "description": issue.get("description"),
        "labels": issue.get("labels"),
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "author": issue.get("author"),
        "assignee": issue.get("assignee"),
        "issue_type": issue.get("issue_type"),
        "linked_prs": issue.get("linked_prs"),
        "comments": issue.get("comments"),
        "evidence_signals": issue.get("evidence_signals"),
        "handling_outcome": issue.get("handling_outcome"),
    }


def build_summary(issues, source_total, self_assigned):
    type_outcomes = defaultdict(Counter)
    for issue in issues:
        type_outcomes[issue.get("issue_type")][issue.get("handling_outcome")] += 1
    return {
        "source_total": source_total,
        "excluded_self_assigned": self_assigned,
        "analyzed_non_self": len(issues),
        "with_comments": sum(bool(i.get("comments")) for i in issues),
        "with_linked_prs": sum(bool(i.get("linked_prs")) for i in issues),
        "state_counts": dict(Counter(i.get("state") for i in issues)),
        "type_counts": dict(Counter(i.get("issue_type") for i in issues)),
        "outcome_counts": dict(Counter(i.get("handling_outcome") for i in issues)),
        "type_outcome_counts": {
            kind: dict(counts) for kind, counts in sorted(type_outcomes.items())
        },
        "signal_counts": dict(
            Counter(
                signal
                for issue in issues
                for signal in issue.get("evidence_signals") or []
            )
        ),
    }


def candidate_score(issue):
    score = 0
    signals = set(issue.get("evidence_signals") or [])
    score += 8 if "linked_change" in signals else 0
    score += 5 if "change_claimed" in signals else 0
    score += 4 if "answer_or_clarification" in signals else 0
    score += 3 if "more_information_requested" in signals else 0
    score += min(len(issue.get("comments") or []), 4)
    score += 1 if issue.get("description") else 0
    return score


def _report_overview(metadata, summary):
    repository = metadata.get("repository") or "unknown-repository"
    repository_name = repository.rsplit("/", 1)[-1]
    scope = (
        "非自提 Issue 历史分析"
        if str(metadata.get("filter") or "").startswith("author != assignee")
        else "Issue 历史证据分析"
    )
    return [
        f"# {repository_name} {scope}",
        "",
        f"- 生成时间：{metadata['generated_at']}",
        f"- 数据源：{metadata['source_url']}",
        f"- 过滤口径：{metadata.get('filter') or 'unknown'}",
        "- 证据边界：统计只反映公开 Issue、评论及 PR 描述中的显式信号；不把关闭状态等同于已修复。",
        "",
        "## 全量概览",
        "",
        f"- Issue 总数：{summary['source_total']}",
        f"- 排除自提：{summary['excluded_self_assigned']}",
        f"- 纳入分析：{summary['analyzed_non_self']}",
        f"- 有评论：{summary['with_comments']}",
        f"- PR 描述显式关联：{summary['with_linked_prs']}",
        "",
        "### 类型分布",
        "",
    ]


def _append_distribution(lines, counts):
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`：{value}")


def _append_candidates(lines, issues, per_type):
    grouped = defaultdict(list)
    for issue in issues:
        grouped[issue["issue_type"]].append(issue)
    for kind in sorted(grouped):
        lines.extend([f"### {kind}", ""])
        chosen = sorted(
            grouped[kind], key=lambda i: (-candidate_score(i), -int(i["number"]))
        )[:per_type]
        for issue in chosen:
            signals = ", ".join(issue.get("evidence_signals") or ["none"])
            lines.append(
                f"- [#{issue['number']} {issue['title']}]({issue['url']}) — "
                f"`{issue['handling_outcome']}`；证据：`{signals}`"
            )
        lines.append("")


def render_report(metadata, summary, issues, per_type=5):
    lines = _report_overview(metadata, summary)
    _append_distribution(lines, summary["type_counts"])
    lines.extend(["", "### 处理证据分布（互斥结果）", ""])
    _append_distribution(lines, summary["outcome_counts"])
    lines.extend(["", "## 案例候选", ""])
    _append_candidates(lines, issues, per_type)
    return "\n".join(lines).rstrip() + "\n"


def write_text(path, content):
    """Atomically replace one generated artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_obj:
            temp_path = Path(file_obj.name)
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="GitCode repository URL")
    parser.add_argument(
        "--token", default=None, help="GitCode token; defaults to GITCODE_TOKEN"
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.35,
        help="Minimum seconds between GitCode request starts (default: 1.35)",
    )
    parser.add_argument(
        "--cache-dir",
        default=path_text(KNOWLEDGE_CACHE),
        help="Resume cache for per-issue comments; set to empty string to disable",
    )
    parser.add_argument("--include-self-assigned", action="store_true")
    parser.add_argument(
        "--skip-prs", action="store_true", help="Skip PR-description linkage"
    )
    parser.add_argument(
        "--output",
        default=path_text(KNOWLEDGE_CORPUS),
        help=f"Detailed JSON corpus path (default: {path_text(KNOWLEDGE_CORPUS)})",
    )
    parser.add_argument(
        "--report",
        default=path_text(KNOWLEDGE_REPORT),
        help=f"Markdown summary path (default: {path_text(KNOWLEDGE_REPORT)})",
    )
    parser.add_argument("--candidates-per-type", type=int, default=5)
    return parser.parse_args()


class CorpusContext(NamedTuple):
    api: object
    limiter: RequestLimiter
    owner: str
    repo: str
    all_issues: list
    issues: list


def load_corpus_context(args):
    token = issue_api.resolve_token(args.token)
    api_base = issue_api.resolve_api_base(args.api_base, args.url)
    owner, repo = issue_api.parse_repo_path(args.url)
    api = issue_api.RepoApiContext(
        issue_api.make_session(rate_limit_dir=rate_limit_path(args.cache_dir)),
        api_base,
        owner,
        repo,
        token,
    )
    limiter = RequestLimiter(args.request_interval)

    raw = issue_api.get_issues(api, state="all")
    all_issues = [issue_api.normalize_issue(item) for item in raw]
    issues = list(all_issues)
    if not args.include_self_assigned:
        issues = issue_api.filter_issues_by_self_assigned(issues)
    return CorpusContext(api, limiter, owner, repo, all_issues, issues)


def _annotate_issues(context, args):
    fetch_comments(
        context.issues,
        context.api,
        max(1, args.workers),
        context.limiter,
        cache_dir=args.cache_dir or None,
    )
    pr_map = {}
    if not args.skip_prs:
        prs = fetch_all_prs(context.api, context.limiter)
        pr_map = build_pr_map(prs, (issue.get("iid") for issue in context.issues))

    for issue in context.issues:
        issue["linked_prs"] = pr_map.get(str(issue.get("iid")), [])
        issue["issue_type"] = classify_issue_type(issue)
        issue["evidence_signals"] = evidence_signals(issue)
        issue["handling_outcome"] = handling_outcome(issue)


def _write_corpus_artifacts(context, args):
    metadata = {
        "schema_version": "issue-history.v1",
        "generated_at": datetime.now(issue_api.TZ_CHINA).isoformat(timespec="seconds"),
        "source_url": args.url,
        "repository": f"{context.owner}/{context.repo}",
        "filter": (
            "all issues"
            if args.include_self_assigned
            else "author != assignee, including missing assignee"
        ),
    }
    summary = build_summary(
        context.issues,
        source_total=len(context.all_issues),
        self_assigned=len(context.all_issues) - len(context.issues),
    )
    corpus = {
        "metadata": metadata,
        "summary": summary,
        "issues": [compact_issue(issue) for issue in context.issues],
    }
    write_text(args.output, json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    write_text(
        args.report,
        render_report(
            metadata,
            summary,
            context.issues,
            per_type=args.candidates_per_type,
        ),
    )
    return summary


def build_corpus(args):
    """Fetch evidence, write corpus artifacts, and return their summary."""
    context = load_corpus_context(args)
    _annotate_issues(context, args)
    return _write_corpus_artifacts(context, args)


def main():
    args = parse_args()
    try:
        summary = build_corpus(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1
    except requests.RequestException as exc:
        LOGGER.error(
            "Error: GitCode request failed — %s",
            issue_api.redact_token(str(exc)),
        )
        return 1
    _write_stdout(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
