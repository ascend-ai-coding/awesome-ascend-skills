#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""
Classify open GitCode issues into action buckets using a deterministic
decision tree.

Shared HTTP / token / URL utilities are imported from gitcode-toolkit's
gitcode_client module; this script only contains classification-specific
business logic (PR fetching, decision tree, report formatting).

Pipeline:
    # Standard usage: comments are fetched on demand after PR association
    python scripts/fetch_issues.py --since 2026-08-04 \\
      > .cannbot/gitcode-issue-handler/data/issues.json
    python scripts/classify_issues.py \\
      --input .cannbot/gitcode-issue-handler/data/issues.json

    # Default is interactive/dry-run: do not POST /assign comments
    python scripts/classify_issues.py \\
      --config .cannbot/gitcode-issue-handler/config/classify_config.yaml

    # approved_batch remains accepted for compatibility; classification stays read-only
    python scripts/classify_issues.py \\
      --config .cannbot/gitcode-issue-handler/config/classify_config.yaml \\
      --authorization-mode approved_batch

Config (YAML, see assets/classify_config.yaml.template):
    repo: cann/ops-math
    gitcode_api: https://api.gitcode.com/api/v5
    last_check_file: .cannbot/gitcode-issue-handler/data/last_check.json
    report_file: .cannbot/gitcode-issue-handler/reports/classification.txt

Token:
    GITCODE_TOKEN env var is not used for classification writes. The legacy
    --no-auto-assign option remains accepted for CLI compatibility.

Decision tree and reason strings mirror the reference plugin script.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import requests

_HERE = Path(__file__).resolve().parent
_TOOLKIT_SCRIPTS = _HERE.parent.parent / "gitcode-toolkit" / "scripts"
sys.path.insert(0, str(_TOOLKIT_SCRIPTS))
from gitcode_client import (  # noqa: E402
    SharedRateLimiter,
    make_session,
    api_get,
    parse_iso,
    TZ_CHINA,
    DEFAULT_GITCODE_API_BASE,
)

sys.path.insert(0, str(_HERE))
from pr_response_policy import pr_active, pr_author, pr_inactive, pr_merged, self_authored  # noqa: E402
from classification_cache import (
    ClassificationCacheEntry,
    load_settled,
    save_classification,
)
from fetch_cache import _atomic_write_json, _read_json, load_pr_links, save_pr_links  # noqa: E402
from fetch_issues import RepoApiContext, enrich_issues_with_comments  # noqa: E402
from issue_pr_evidence import collect_issue_prs  # noqa: E402
from cli_output import write_stdout  # noqa: E402
from runtime_paths import (  # noqa: E402
    CLASSIFICATION_REPORT,
    CLASSIFY_CONFIG,
    FETCH_CACHE,
    FOLLOWUP_WATCH_STATE,
    LAST_CHECK_STATE,
    LEGACY_CLASSIFY_CONFIG,
    compatible_read_path,
    migrate_legacy_runtime_defaults,
    path_text,
    rate_limit_path,
)

from handler_config import get_automation_policy, load_handler_config, load_template
from responsibility import LEVELS, policy_digest, review_responsibility
from resolve_repository import resolve_repository

_DEFAULT_CONFIG = load_template("classify_config")

DEFAULT_CONFIG_FILE = path_text(CLASSIFY_CONFIG)
DEFAULT_LAST_CHECK_FILE = path_text(LAST_CHECK_STATE)
DEFAULT_REPORT_FILE = path_text(CLASSIFICATION_REPORT)
DEFAULT_CACHE_DIR = path_text(FETCH_CACHE)
DEFAULT_LOOKBACK_DAYS = _DEFAULT_CONFIG["lookback_days"]
PR_FETCH_PAGES = _DEFAULT_CONFIG["pr_fetch_pages"]
PR_LINKAGE_API_BUDGET = _DEFAULT_CONFIG["pr_linkage_api_budget"]
_ASSIGN_PATTERN = re.compile(
    r"\A\s*/assign\s+(?:@\S+|\[@[^\]\s]+\]\([^)]+\))\s*\Z",
    re.IGNORECASE,
)
_MENTION_ONLY_PATTERN = re.compile(
    r"\A\s*(?:\[\s*)?@[^\s\],，。.!！?？)]+"
    r"(?:\s*\]\([^)]+\))?\s*[,，。.!！?？]?\s*\Z"
)
_SYSTEM_BOT_LOGINS = frozenset({"cann-robot"})
_SYSTEM_COMMENT_PATTERNS = (
    re.compile(
        r"\A\s*(?:#{1,6}\s*notice\s*)?"
        r"this\s+issue\s+can\s+not\s+be\s+assigned\s+to\s+\**yourself\**\.\s*"
        r"please\s+try\s+to\s+assign\s+to\s+the\s+repository\s+members\.\s*\Z",
        re.IGNORECASE,
    ),
)
_ASSIGNEE_WAIT_PATTERNS = (
    re.compile(
        r"(?:已|已经|正在)?(?:联系|转交|转给|反馈给|同步给).{0,24}"
        r"(?:责任人|负责人|owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:等待|请).{0,24}(?:责任人|负责人|owner|maintainer|assignee)"
        r".{0,24}(?:处理|回复|确认|反馈|分析|跟进)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:已|已经)?(?:转交|指派|分配)(?:给|至)?.{0,32}(?:处理|跟进)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:assigned|forwarded|escalated).{0,24}" r"(?:owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:waiting|pending).{0,24}(?:owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
)
LOGGER = logging.getLogger(__name__)
_CACHE_REVISION = hashlib.sha256(
    Path(__file__).read_bytes() + (_HERE / "pr_response_policy.py").read_bytes()
    + (_HERE / "issue_pr_evidence.py").read_bytes()
).hexdigest()


class LinkageOptions(NamedTuple):
    """Repository and scan settings for native PR-to-Issue linkage."""

    api_base: str
    repo: str
    token: str
    target_issue_numbers: object = None
    api_budget: int = PR_LINKAGE_API_BUDGET
    scan_mode: str = "ambiguous"
    cache_dir: object = DEFAULT_CACHE_DIR
    rate_limiter: object = None


class PRFetchOptions(NamedTuple):
    """Repository and paging settings for one recent-PR scan."""

    api_base: str
    repo: str
    token: str
    since_iso: str | None = None
    max_pages: int = PR_FETCH_PAGES
    rate_limiter: object = None


class ClassificationOptions(NamedTuple):
    """External evidence and side-effect policy for one classification."""

    issue_pr_map: dict
    post_fn: object
    dry_run: bool
    association_scan_complete: bool = True
    comment_scan_complete: bool = True
    automation_policy: dict | None = None


class CommentTimeline(NamedTuple):
    """Normalized participant comments in chronological order."""

    reporter: str
    assignee: str
    reporter_comments: list[dict]
    maintainer_comments: list[dict]
    latest_reporter: dict | None
    latest_maintainer: dict | None


class WatchSignals(NamedTuple):
    """Signals derived from an explicit persisted follow-up watch."""

    watch: dict
    state: str
    latest_assignee: dict | None
    reporter_followup: bool
    assignee_followup: bool


class InferredSignals(NamedTuple):
    """Signals derived from explicit hand-off wording in public comments."""

    latest_wait: dict | None
    waiting: bool
    assignee_followup: bool
    reporter_followup: bool


class ConversationAnalysis(NamedTuple):
    """Inputs needed to render the final conversation decision."""

    timeline: CommentTimeline
    watch: WatchSignals
    inferred: InferredSignals
    state: str
    pending_reporter: bool
    core_closed: bool
    custom_state: str


class IssueEvidence(NamedTuple):
    """Normalized evidence consumed by the classification decision tree."""

    number: object
    author: str | None
    assignee: str | None
    effective_comments: list[dict]
    active_prs: list[dict]
    followup_selected: bool
    linked_prs: list[dict]


def _write_stdout(text):
    """Write the classifier JSON output protocol to stdout."""
    write_stdout(text)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def resolve_config_path(path=None, *, allow_legacy=True):
    """Prefer canonical config and read the former root file only as fallback."""
    requested = CLASSIFY_CONFIG if path is None else Path(path)
    if not allow_legacy:
        return requested
    return compatible_read_path(
        requested,
        canonical=CLASSIFY_CONFIG,
        legacy=LEGACY_CLASSIFY_CONFIG,
    )


def load_config(path=None, repo_override=None, *, allow_legacy=True):
    cfg = load_handler_config(path, allow_legacy=allow_legacy)
    # Shared loading supplies template defaults; preserve the historical
    # migration of exact legacy runtime paths only for default config lookup.
    if path is None and allow_legacy:
        source = (
            CLASSIFY_CONFIG
            if CLASSIFY_CONFIG.exists()
            else LEGACY_CLASSIFY_CONFIG
        )
        cfg = migrate_legacy_runtime_defaults(cfg, source)
    if repo_override:
        if cfg.get("repo") and cfg["repo"] != repo_override:
            raise ValueError("Repository conflicts with configured repo; confirm via resolve_repository.py")
        cfg["repo"] = repo_override
    if not cfg.get("repo"):
        raise ValueError(
            "Error: repository is unknown. Pass --repo owner/repo, use input "
            "from fetch_issues.py, or set repo in the config file."
        )

    return cfg


# --------------------------------------------------------------------------- #
# last_check.json state
# --------------------------------------------------------------------------- #
def load_last_check(path, repo):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("repo") == repo:
            return data.get("last_all_clear_time")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_last_check(path, repo):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now_iso = datetime.now(TZ_CHINA).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"last_all_clear_time": now_iso, "repo": repo},
            f,
            ensure_ascii=False,
            indent=2,
        )


def get_since(path, repo, lookback_days=DEFAULT_LOOKBACK_DAYS):
    last = load_last_check(path, repo)
    if last:
        return last
    return (datetime.now(TZ_CHINA) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT00:00:00+08:00"
    )


def _normalize_since(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=TZ_CHINA).isoformat()
    except ValueError:
        parsed = parse_iso(value)
        return parsed.isoformat() if parsed is not None else value


def resolve_time_scope(raw, ignore_last_check, last_check_file, repo, lookback_days=DEFAULT_LOOKBACK_DAYS):
    """Resolve one authoritative input scope and whether to filter by update."""
    filters = raw.get("filters", {}) if isinstance(raw, dict) else {}
    if ignore_last_check:
        return None, False, "full_input"
    if filters.get("mode") == "single":
        return None, False, "single_issue"
    if filters.get("since"):
        return _normalize_since(filters["since"]), False, "fetch_since"
    return get_since(last_check_file, repo, lookback_days), True, "last_check"


# --------------------------------------------------------------------------- #
# PR fetching + issue→PR mapping
# --------------------------------------------------------------------------- #
def fetch_pr_linked_issues(api_base, repo, pr_number, token, rate_limiter=None):
    url = f"{api_base}/repos/{repo}/pulls/{pr_number}/issues"
    session = make_session(rate_limiter=rate_limiter)
    return api_get(session, url, token)


def fetch_recent_prs(options: PRFetchOptions):
    """Fetch recent PRs and stop once the updated-time window is exhausted."""
    url = f"{options.api_base}/repos/{options.repo}/pulls"
    since_dt = parse_iso(options.since_iso) if options.since_iso else None
    session = make_session(rate_limiter=options.rate_limiter)
    diagnostics = {
        "complete": True,
        "pages_requested": 0,
        "warnings": [],
    }
    all_prs = []
    for page in range(1, options.max_pages + 1):
        params = {
            "state": "all",
            "page": page,
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        }
        try:
            batch = api_get(session, url, options.token, params=params)
        except Exception as exc:
            diagnostics["complete"] = False
            diagnostics["warnings"].append(
                f"PR list page {page} failed: {type(exc).__name__}"
            )
            break
        diagnostics["pages_requested"] += 1
        batch = batch if isinstance(batch, list) else []
        if not batch:
            break
        all_prs.extend(batch)
        if len(batch) < 100:
            break
        if since_dt:
            updated_values = [parse_iso(pr.get("updated_at", "")) for pr in batch]
            if any(
                updated is not None and updated < since_dt for updated in updated_values
            ):
                break
    else:
        diagnostics["complete"] = False
        diagnostics["warnings"].append(
            f"PR scan reached configured page limit ({options.max_pages})"
        )
    return all_prs, diagnostics


class LinkageScanState:
    """Mutable state for one bounded native-linkage scan."""

    def __init__(self, options):
        self.options = options
        self.targets = (
            {
                str(number)
                for number in options.target_issue_numbers
                if number is not None
            }
            if options.target_issue_numbers is not None
            else None
        )
        self.parsed_refs = {}
        self.api_budget_used = 0
        self.incomplete_numbers = set()
        self.diagnostics = {
            "scan_mode": options.scan_mode,
            "candidates": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "api_errors": 0,
            "budget_exhausted": False,
            "incomplete_issue_numbers": [],
        }


def _text_refs(pr, targets):
    text = f"{pr.get('body') or ''} {pr.get('title') or ''}"
    refs = set(re.findall(r"#(\d+)", text) + re.findall(r"/issues/(\d+)", text))
    return text, refs & targets if targets is not None else refs


def _linkage_candidates(prs, state):
    candidates = []
    for pr in prs:
        pr_num = pr.get("number")
        if pr_num is None:
            continue
        text, refs = _text_refs(pr, state.targets)
        state.parsed_refs[pr_num] = refs
        if state.options.scan_mode == "all":
            if not refs:
                candidates.append((pr, set(state.targets or [])))
            continue
        if not state.targets:
            continue
        head_ref = (pr.get("head") or {}).get("ref") or ""
        ambiguous_refs = set()
        for ref in state.targets - refs:
            pattern = rf"(?<!\d){re.escape(ref)}(?!\d)"
            if re.search(pattern, f"{text} {head_ref}"):
                ambiguous_refs.add(ref)
        if ambiguous_refs:
            candidates.append((pr, ambiguous_refs))
    state.diagnostics["candidates"] = len(candidates)
    return candidates


def _linked_numbers(pr, ambiguous_refs, state):
    options = state.options
    cached = load_pr_links(options.cache_dir, options.repo, pr)
    if cached is not None:
        state.diagnostics["cache_hits"] += 1
        return cached
    if state.api_budget_used >= max(0, options.api_budget):
        state.diagnostics["budget_exhausted"] = True
        state.incomplete_numbers.update(ambiguous_refs or (state.targets or []))
        return None

    state.api_budget_used += 1
    state.diagnostics["api_calls"] += 1
    try:
        linked = fetch_pr_linked_issues(
            options.api_base,
            options.repo,
            pr["number"],
            options.token,
            options.rate_limiter,
        )
    except requests.RequestException:
        state.diagnostics["api_errors"] += 1
        state.incomplete_numbers.update(ambiguous_refs or (state.targets or []))
        return None
    linked_items = linked if isinstance(linked, list) else []
    linked_numbers = []
    for item in linked_items:
        if isinstance(item, dict) and item.get("number") is not None:
            linked_numbers.append(item.get("number"))
    save_pr_links(options.cache_dir, options.repo, pr, linked_numbers)
    return linked_numbers


def _resolve_linkage_candidates(candidates, state):
    for pr, ambiguous_refs in candidates:
        linked_numbers = _linked_numbers(pr, ambiguous_refs, state)
        if linked_numbers is None:
            continue
        for number in linked_numbers:
            ref = str(number)
            if state.targets is None or ref in state.targets:
                state.parsed_refs[pr["number"]].add(ref)


def _invert_pr_refs(prs, parsed_refs):
    issue_pr_map = {}
    for pr in prs:
        pr_num = pr.get("number")
        if pr_num is None:
            continue
        pr_info = {
            "pr_number": pr_num,
            "pr_state": pr.get("state"),
            "pr_merged": pr_merged(pr),
            "pr_url": pr.get("html_url") or pr.get("web_url"),
            "pr_expired": pr.get("expired") is True,
            "expiration_evidence": pr.get("expiration_evidence"),
            "pr_title": (pr.get("title") or "")[:60],
            "pr_author": pr_author(pr),
        }
        for ref in parsed_refs.get(pr_num, set()):
            issue_pr_map.setdefault(ref, []).append(pr_info)
    return issue_pr_map


def build_issue_pr_map(prs, options: LinkageOptions):
    """Build Issue-to-PR evidence using text refs and bounded native linkage."""
    if options.scan_mode not in {"ambiguous", "all"}:
        raise ValueError("scan_mode must be 'ambiguous' or 'all'")
    state = LinkageScanState(options)
    candidates = _linkage_candidates(prs, state)
    _resolve_linkage_candidates(candidates, state)
    state.diagnostics["incomplete_issue_numbers"] = sorted(
        state.incomplete_numbers, key=lambda value: int(value)
    )
    state.diagnostics["complete"] = (
        state.diagnostics["api_errors"] == 0
        and not state.diagnostics["budget_exhausted"]
    )
    return _invert_pr_refs(prs, state.parsed_refs), state.diagnostics


# --------------------------------------------------------------------------- #
# Comment shape helpers
# --------------------------------------------------------------------------- #
def is_only_assign_comments(comments):
    """Return whether every effective comment is an assignment command."""
    if not comments:
        return False
    for c in comments:
        body = (c.get("body") or "").strip()
        if not _ASSIGN_PATTERN.fullmatch(body):
            return False
    return True


def get_comment_author(comment):
    """Return a comment author's login for normalized or raw API shapes."""
    author = comment.get("author")
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        return author.get("login")
    user = comment.get("user")
    if isinstance(user, dict):
        return user.get("login")
    return None


def _is_system_comment(comment) -> bool:
    """Recognize explicit system events or known bot-generated templates."""
    if comment.get("system") is True:
        return True
    author = get_comment_author(comment)
    if not author or author.strip().casefold() not in _SYSTEM_BOT_LOGINS:
        return False
    body = str(comment.get("body") or "")
    return any(pattern.fullmatch(body) for pattern in _SYSTEM_COMMENT_PATTERNS)


def get_effective_comments(comments, issue_author):
    """Keep comments with a known author other than the issue reporter."""
    normalized_issue_author = issue_author.strip().casefold() if issue_author else None
    effective_comments = []
    for comment in comments:
        if _is_system_comment(comment):
            continue
        body = str(comment.get("body") or "")
        if not body.strip() or _MENTION_ONLY_PATTERN.fullmatch(body):
            continue
        comment_author = get_comment_author(comment)
        if not comment_author:
            continue
        normalized_comment_author = comment_author.strip().casefold()
        if normalized_comment_author == "unknown":
            continue
        if (
            normalized_issue_author
            and normalized_comment_author == normalized_issue_author
        ):
            continue
        effective_comments.append(comment)
    return effective_comments


def _comment_time(comment):
    """Return a normalized comment time without inventing missing evidence."""
    return parse_iso(comment.get("created_at", ""))


def _comment_id(comment):
    value = comment.get("id")
    return str(value) if value is not None else None


def _indicates_assignee_wait(body: str) -> bool:
    """Recognize explicit hand-off text without treating every reply as pending."""
    return any(pattern.search(body) for pattern in _ASSIGNEE_WAIT_PATTERNS)


def _comment_after_watch(entry, watch, watch_time) -> bool:
    """Return whether a comment is newer than the watch baseline."""
    if not entry or not watch:
        return False
    baseline_id = watch.get("last_maintainer_comment_id")
    if baseline_id is not None and entry["id"] == str(baseline_id):
        return False
    if watch_time is not None and entry["parsed_at"] is not None:
        return entry["parsed_at"] > watch_time
    return True


def _issue_type_signals(issue):
    """Read explicit issue types without treating arbitrary prose as a label."""
    values = list(issue.get("labels") or [])
    values.extend(issue.get(key) for key in ("issue_type", "type", "category"))
    match = re.match(r"\s*\[([^\]]+)\]", str(issue.get("title") or ""))
    if match:
        values.extend(match.group(1).split("|"))
    signals = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("title")
        if isinstance(value, str):
            signals.add(value.strip().casefold())
    return signals


def _planning_body_has_request(body):
    """Distinguish reported problems/questions from future work in a roadmap."""
    planned = re.compile(
        r"^(?:计划|规划|拟|目标|后续|下一步|预计|待办|将|准备|Goal\s*[:：]|Plan\b)", re.IGNORECASE,
    )
    current_problem = re.compile(
        r"(?:当前|目前|实际|实测|复现|运行时|使用时).*(?:报错|崩溃|失败|异常|无法|不支持|越界)"
        r"|\b(?:currently|observed|reproduced)\b.*\b(?:crash|error|failure|fails)\b", re.IGNORECASE,
    )
    question = re.compile(
        r"请问|能否|请(?:帮忙|协助|确认|解释)|(?:是否|如何|为什么|怎么).*?[？?]"
        r"|\b(?:how (?:do|can)|why does|could you|please help)\b", re.IGNORECASE,
    )
    for clause in re.split(r"(?<=[。；;！？!?])|\n", body):
        text = re.sub(r"^\s*(?:[-*+]\s+(?:\[[ xX]\]\s*)?|\d+[.)]\s+)", "", clause).strip()
        if planned.search(text):
            continue
        if current_problem.search(text) or question.search(text):
            return True
    return False


def _response_exemption(issue):
    """Recognize planning records from concrete content.

    A Roadmap mention alone never makes a reported defect a planning record.
    Account/PR-based self-authorship is handled separately by classify_one.
    """
    title = str(issue.get("title") or "").strip()
    body = str(issue.get("description") or issue.get("body") or "").strip()
    signals = _issue_type_signals(issue)
    defect_types = {"bug", "bug-report", "缺陷", "缺陷反馈", "问题反馈", "反馈", "feedback"}
    failure_title = re.search(
        r"\b(?:bug|crash(?:es)?|failure|exception|incorrect|regression)\b"
        r"|崩溃|报错|错误|不达标|异常|缺陷|失败", title, re.IGNORECASE,
    )
    failure_sections = re.search(
        r"^\s*#{1,6}\s*(?:问题描述|问题定位|实测复现|复现步骤|重现步骤|错误日志|实际结果)"
        r"[^\n]*\n\s*\S", body, re.MULTILINE,
    )
    if signals & defect_types or failure_title or failure_sections:
        return None
    planning_title = re.fullmatch(
        r"(?:(?:Development|Project)\s+)?Roadmap(?:\s*(?:\([^)]*\)|\d{4}\s*Q[1-4]))?"
        r"|\d{4}\s*Q[1-4]\s+Roadmap"
        r"|(?:项目|开发|技术)?(?:路线图|规划汇总)(?:[（(][^）)]*[）)])?",
        title, re.IGNORECASE,
    )
    planning_type = bool(signals & {"roadmap", "路线图", "规划汇总", "planning"})
    planning_structure = bool(
        re.search(r"^\s*#{1,6}\s*(?:总体方向|详细计划|规划汇总|Roadmap)\b", body, re.MULTILINE | re.IGNORECASE)
        or (len(re.findall(r"\bGoal\s*[:：]", body, re.IGNORECASE)) >= 2
            and len(re.findall(r"\bOwner\s*[:：]", body, re.IGNORECASE)) >= 2)
    )
    is_planning = bool((planning_title or planning_type) and planning_structure)
    if is_planning and not _planning_body_has_request(body):
        return _classification_result(
            "no_attention", "planning_record",
            "纯路线图或规划汇总，不属于问题、缺陷或反馈，不进入批量响应",
        )
    return None


def _comment_timeline(issue) -> CommentTimeline:
    """Normalize public comments into reporter and maintainer timelines."""
    reporter = str(issue.get("author") or "").strip()
    normalized_reporter = reporter.casefold()
    assignee = str(issue.get("assignee") or "").strip()
    comments = list(issue.get("comments") or [])
    if comments and all(_comment_time(comment) is not None for comment in comments):
        comments.sort(key=_comment_time)
    reporter_comments = []
    maintainer_comments = []
    for index, comment in enumerate(comments):
        if _is_system_comment(comment):
            continue
        author = get_comment_author(comment)
        body = str(comment.get("body") or "").strip()
        if not author or author.casefold() == "unknown" or not body:
            continue
        if _ASSIGN_PATTERN.fullmatch(body) or _MENTION_ONLY_PATTERN.fullmatch(body):
            continue
        entry = {
            "index": index,
            "id": _comment_id(comment),
            "created_at": comment.get("created_at", ""),
            "parsed_at": _comment_time(comment),
            "author": author,
            "body": body,
        }
        if normalized_reporter and author.casefold() == normalized_reporter:
            reporter_comments.append(entry)
        else:
            maintainer_comments.append(entry)
    return CommentTimeline(
        reporter,
        assignee,
        reporter_comments,
        maintainer_comments,
        reporter_comments[-1] if reporter_comments else None,
        maintainer_comments[-1] if maintainer_comments else None,
    )


def _latest_author_comment(comments, author):
    if not author:
        return None
    normalized = author.casefold()
    for entry in reversed(comments):
        if entry.get("author", "").casefold() == normalized:
            return entry
    return None


def _latest_wait_comment(comments):
    for entry in reversed(comments):
        if _indicates_assignee_wait(entry.get("body", "")):
            return entry
    return None


def _watch_signals(issue, timeline: CommentTimeline) -> WatchSignals:
    watch = issue.get("followup_watch") or {}
    watch_state = str(watch.get("conversation_state") or "").strip().casefold()
    watch_time = parse_iso(watch.get("last_maintainer_comment_at", ""))
    reporter_unanswered = bool(
        timeline.latest_reporter
        and (not timeline.latest_maintainer
             or timeline.latest_reporter["index"] > timeline.latest_maintainer["index"])
    )
    reporter_after_watch = bool(
        reporter_unanswered
        and watch
        and _comment_after_watch(timeline.latest_reporter, watch, watch_time)
    )
    awaited_assignee = str(watch.get("assignee") or timeline.assignee).strip()
    latest_assignee = _latest_author_comment(
        timeline.maintainer_comments, awaited_assignee
    )
    assignee_after_watch = bool(
        watch_state == "awaiting_assignee"
        and _comment_after_watch(latest_assignee, watch, watch_time)
        and latest_assignee == timeline.latest_maintainer
    )
    baseline_known = watch_time is not None or any(
        entry["id"] == str(watch.get("last_maintainer_comment_id"))
        for entry in timeline.maintainer_comments
    )
    superseded = baseline_known and _comment_after_watch(timeline.latest_maintainer, watch, watch_time)
    if superseded and not reporter_after_watch and not assignee_after_watch:
        # Ignore a superseded baseline for classification; persist/resolve the
        # original watch only through the authorized follow-up workflow.
        watch, watch_state = {}, ""
    return WatchSignals(
        watch,
        watch_state,
        latest_assignee,
        reporter_after_watch,
        assignee_after_watch,
    )


def _inferred_signals(timeline: CommentTimeline, watch: WatchSignals):
    latest_wait = _latest_wait_comment(timeline.maintainer_comments)
    normalized_reporter = timeline.reporter.casefold()
    inferred_assignee_wait = bool(
        not watch.watch
        and timeline.assignee
        and timeline.assignee.casefold() != normalized_reporter
        and latest_wait
    )
    inferred_assignee_followup = bool(
        inferred_assignee_wait
        and watch.latest_assignee
        and watch.latest_assignee["index"] > latest_wait["index"]
        and watch.latest_assignee == timeline.latest_maintainer
    )
    reporter_after_inferred_wait = bool(
        inferred_assignee_wait
        and timeline.latest_reporter
        and timeline.latest_reporter["index"] > timeline.latest_maintainer["index"]
    )
    return InferredSignals(
        latest_wait,
        inferred_assignee_wait,
        inferred_assignee_followup,
        reporter_after_inferred_wait,
    )


def _conversation_state(issue, timeline, watch, inferred, pending_reporter):
    core_closed = str(issue.get("state") or "").casefold() == "closed"
    custom_state = str(issue.get("issue_state") or "").strip()
    terminal_custom = custom_state.casefold() in {
        "已完成",
        "已解决",
        "已拒绝",
        "已取消",
        "已验收",
    }
    if pending_reporter:
        return (
            "reopened_followup"
            if core_closed or terminal_custom
            else "reporter_followup"
        )
    if watch.assignee_followup or inferred.assignee_followup:
        return "assignee_followup"
    if watch.state == "awaiting_assignee" or inferred.waiting:
        return "awaiting_assignee"
    if watch.watch:
        return "awaiting_reporter"
    if timeline.latest_maintainer:
        return "maintainer_replied"
    return "awaiting_maintainer"


def _conversation_output(analysis: ConversationAnalysis) -> dict:
    timeline, watch, inferred = analysis.timeline, analysis.watch, analysis.inferred
    if analysis.pending_reporter:
        pending_since = timeline.latest_reporter["created_at"]
    elif watch.assignee_followup or inferred.assignee_followup:
        pending_since = watch.latest_assignee["created_at"]
    elif inferred.waiting:
        pending_since = inferred.latest_wait["created_at"]
    else:
        pending_since = None
    waiting_on = {
        "awaiting_reporter": "reporter",
        "awaiting_assignee": "assignee",
    }.get(analysis.state, "maintainer")
    waiting_since = watch.watch.get("waiting_since") if watch.watch else None
    if not watch.watch and inferred.waiting:
        waiting_since = inferred.latest_wait["created_at"]
    return {
        "state": analysis.state,
        "waiting_on": waiting_on,
        "latest_reporter_comment_id": (
            timeline.latest_reporter["id"] if timeline.latest_reporter else None
        ),
        "latest_reporter_comment_at": (
            timeline.latest_reporter["created_at"] if timeline.latest_reporter else None
        ),
        "latest_maintainer_comment_id": (
            timeline.latest_maintainer["id"] if timeline.latest_maintainer else None
        ),
        "latest_maintainer_comment_at": (
            timeline.latest_maintainer["created_at"]
            if timeline.latest_maintainer
            else None
        ),
        "pending_since": pending_since,
        "waiting_since": waiting_since,
        "reopen_required": bool(analysis.pending_reporter and analysis.core_closed),
        "activate_required": bool(
            analysis.state
            in {"reporter_followup", "reopened_followup", "assignee_followup"}
            and analysis.custom_state.casefold() != "进行中"
        ),
        "waiting_status_reconcile_required": bool(
            analysis.state == "awaiting_reporter"
            and analysis.custom_state.casefold() != "挂起"
        ),
        # Historical replies are not a new response obligation. Establishing a
        # watch belongs to the current handoff workflow, not batch backfill.
        "waiting_watch_required": False,
    }


def analyze_conversation(issue):
    """Determine whose turn is actionable from ordered Issue comments."""
    timeline = _comment_timeline(issue)
    watch = _watch_signals(issue, timeline)
    inferred = _inferred_signals(timeline, watch)
    reporter_after_maintainer = bool(
        timeline.latest_reporter
        and timeline.latest_maintainer
        and timeline.latest_reporter["index"] > timeline.latest_maintainer["index"]
    )
    pending_reporter = bool(
        watch.reporter_followup
        or reporter_after_maintainer
        or inferred.reporter_followup
    )
    state = _conversation_state(issue, timeline, watch, inferred, pending_reporter)

    core_closed = str(issue.get("state") or "").casefold() == "closed"
    custom_state = str(issue.get("issue_state") or "").strip()
    analysis = ConversationAnalysis(
        timeline, watch, inferred, state, pending_reporter, core_closed, custom_state
    )
    return _conversation_output(analysis)


def followup_sla(pending_since, now=None):
    """Return a one-business-day follow-up response clock."""
    started = parse_iso(pending_since)
    if started is None:
        return "unknown"
    due = started + timedelta(days=1)
    while due.weekday() >= 5:
        due += timedelta(days=1)
    current = now or datetime.now(TZ_CHINA)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TZ_CHINA)
    if current > due:
        return "breached"
    if current >= due - timedelta(hours=4):
        return "at_risk"
    return "pending"


def should_fetch_comments(issue, issue_pr_map):
    """Fetch only when comments can still change the classification result."""
    sources = set(issue.get("fetch_sources") or [])
    if issue.get("followup_watch") or sources & {"updated", "watchlist"}:
        return True, "followup_detection_required"
    number = issue.get("number") or issue.get("iid")
    active_prs = [pr for pr in issue_pr_map.get(str(number), []) if pr_active(pr)]
    if active_prs:
        # Self-authorship exempts first response, not later reporter follow-ups.
        return True, "first_response_required"
    return True, "classification_required"


def filter_by_updated_since(issues, since_iso):
    """Keep Issues updated since the boundary, including unparseable dates."""
    if not since_iso:
        return issues
    since_dt = parse_iso(since_iso)
    if since_dt is None:
        return issues
    filtered = []
    for issue in issues:
        sources = set(issue.get("fetch_sources") or [])
        if sources & {"updated", "watchlist"}:
            filtered.append(issue)
            continue
        updated = parse_iso(issue.get("updated_at", ""))
        if updated is None or updated >= since_dt:
            filtered.append(issue)
    return filtered


# --------------------------------------------------------------------------- #
# Decision tree
# --------------------------------------------------------------------------- #
def _classification_result(bucket, category, reason, auto_action=None):
    return {
        "bucket": bucket,
        "category": category,
        "reason": reason,
        "auto_action": auto_action,
    }


def _pr_assignment_action(evidence):
    """Describe response-stage work; classification never performs assignment."""
    if evidence.assignee or not evidence.active_prs:
        return None
    authors = sorted({pr["pr_author"] for pr in evidence.active_prs if pr.get("pr_author")}, key=str.casefold)
    own = self_authored(evidence.author, evidence.active_prs)
    has_reply = bool(evidence.effective_comments and not is_only_assign_comments(evidence.effective_comments))
    return {
        "type": "assign_candidate",
        "strategy": "linked_pr_author_during_response",
        "candidate": evidence.author if own else (authors[0] if len(authors) == 1 else None),
        "candidates": authors,
        "selection": "self_authored" if own else "issue_coverage",
        "response_requirement": "exempt_self_authored_pr" if own else ("satisfied" if has_reply else "required"),
        "requires_verified_first_response": not own,
        "policy": "auto-response",
        "read_only": True,
    }


def _classify_unassigned_pr(evidence):
    action = _pr_assignment_action(evidence)
    if not action["candidates"]:
        return _classification_result(
            "need_attention", "needs_manual_no_pr_author",
            "无负责人，已有关联PR，但无法获取PR作者，需补查后响应和分配", action,
        )
    return _classification_result(
        "need_attention", "needs_pr_owner_handoff",
        "无负责人，response 阶段须分配关联PR作者；自提优先，否则按问题覆盖面选择",
        action,
    )


def _has_self_authored_active_pr(author_login, active_pr_refs):
    return self_authored(author_login, active_pr_refs)


def _needs_first_response_with_pr(evidence: IssueEvidence):
    if not evidence.active_prs or self_authored(evidence.author, evidence.active_prs):
        return None
    if evidence.effective_comments and not is_only_assign_comments(evidence.effective_comments):
        return None
    return _classification_result(
        "need_attention", "needs_first_response_with_pr",
        "已有关联PR，尚无他人实质回复；首响简述PR方案，无负责人时在response阶段分配PR作者",
        _pr_assignment_action(evidence),
    )


def _classify_unassigned(effective_comments):
    if not effective_comments:
        return _classification_result(
            "need_attention",
            "needs_first_look",
            "无负责人、无关联PR、无非提出者评论回复，需要关注并分配",
        )
    if is_only_assign_comments(effective_comments):
        return _classification_result(
            "need_attention",
            "needs_only_assign_cmd",
            "无负责人，评论仅为指派命令，需要关注并分配",
        )
    return _classification_result(
        "no_attention",
        "replied_no_owner",
        "无负责人，已有非提出者评论回复，不需要关注",
    )


def _classify_assigned(assignee_login, active_pr_refs, effective_comments):
    if active_pr_refs:
        return _classification_result(
            "no_attention",
            "our_team_done_with_pr",
            f"负责人 {assignee_login}，已有关联PR，已处理",
        )
    if not effective_comments:
        return _classification_result(
            "need_attention",
            "our_team_needs_work",
            f"负责人 {assignee_login}，无关联PR且无非提出者评论回复，需要处理",
        )
    if is_only_assign_comments(effective_comments):
        return _classification_result(
            "need_attention",
            "our_team_only_assign_cmd",
            f"负责人 {assignee_login}，评论仅为指派命令，需要处理",
        )
    return _classification_result(
        "no_attention",
        "our_team_replied",
        f"负责人 {assignee_login}，已有非提出者评论回复，可能已处理",
    )


def _issue_evidence(issue, options: ClassificationOptions) -> IssueEvidence:
    number = issue.get("number") or issue.get("iid")
    author_login = issue.get("author")
    assignee_login = issue.get("assignee")
    effective_comments = get_effective_comments(
        issue.get("comments", []) or [], author_login
    )
    pr_refs = options.issue_pr_map.get(str(number), [])
    active_pr_refs = [pr for pr in pr_refs if pr_active(pr)]
    followup_selected = bool(
        issue.get("followup_watch")
        or set(issue.get("fetch_sources") or []) & {"updated", "watchlist"}
    )
    return IssueEvidence(
        number,
        author_login,
        assignee_login,
        effective_comments,
        active_pr_refs,
        followup_selected,
        pr_refs,
    )


def _classify_conversation(conversation):
    state = conversation.get("state")
    if state in {"reporter_followup", "reopened_followup"}:
        reason = "提出者在维护者回复后新增评论，需要优先跟进"
        if state == "reopened_followup":
            reason += "；Issue 当前为关闭或终态，回复前应恢复为进行中"
        return _classification_result("need_attention", state, reason)
    if state == "assignee_followup":
        return _classification_result(
            "need_attention",
            "assignee_followup",
            "责任人在转交后新增实质回复，需要跟进当前进展",
        )
    if state == "awaiting_assignee":
        return _classification_result(
            "no_attention",
            "awaiting_assignee",
            "已有实质响应和责任人，无待跟进的新回复；不为历史状态或缺失watch重复纳入响应",
        )
    if state == "awaiting_reporter":
        if conversation.get("waiting_status_reconcile_required"):
            return _classification_result(
                "need_attention",
                "awaiting_reporter_setup",
                "Issue 正在等待提出者补充，需将自定义状态恢复为挂起",
            )
        return _classification_result(
            "no_attention",
            "awaiting_reporter",
            "维护者已请求提出者补充，Issue 保持挂起并由 watchlist 持续跟踪",
        )
    return None


def _self_assignee_result(evidence: IssueEvidence):
    """Account identity is sufficient self-authorship, independent of PRs."""
    author = str(evidence.author or "").strip().casefold()
    assignee = str(evidence.assignee or "").strip().casefold()
    if author and assignee and author == assignee:
        return _classification_result(
            "no_attention", "self_assigned",
            "Issue提出者与负责人是同一账号，按自提处理，免首响；无需重复指派，继续未闭环跟踪",
        )
    return None


def _classify_remaining(evidence: IssueEvidence):
    own = self_authored(evidence.author, evidence.active_prs)
    historical_self = (
        evidence.assignee and not evidence.active_prs
        and self_authored(evidence.author, [pr for pr in evidence.linked_prs if pr_inactive(pr)])
    )
    if evidence.assignee and (own or historical_self):
        reason = "自提Issue已有负责人，免首响"
        if historical_self:
            reason += "；关联PR已失效，保留历史自提豁免，继续未闭环跟踪"
        return _classification_result("no_attention", "self_assigned", reason)
    if own:
        return _classify_unassigned_pr(evidence)
    if evidence.active_prs and any(not pr.get("pr_author") for pr in evidence.active_prs):
        return _classification_result(
            "need_attention", "needs_manual_no_pr_author",
            "关联PR作者信息不完整，补查后再判断首响豁免和分配对象",
        )
    first_response_result = _needs_first_response_with_pr(evidence)
    if first_response_result:
        return first_response_result
    if evidence.assignee:
        result = _classify_assigned(evidence.assignee, evidence.active_prs, evidence.effective_comments)
        if not evidence.active_prs and evidence.linked_prs and result["bucket"] == "no_attention":
            result["reason"] += "；历史PR已失效，按无有效PR继续未闭环跟踪"
        return result
    if evidence.active_prs:
        return _classify_unassigned_pr(evidence)
    return _classify_unassigned(evidence.effective_comments)


def classify_one(issue, options: ClassificationOptions):
    """Combine response obligations with existing conversation tracking."""
    exemption = _response_exemption(issue)
    if exemption:
        return exemption
    evidence = _issue_evidence(issue, options)
    if not options.comment_scan_complete:
        return _classification_result(
            "need_attention", "comment_scan_incomplete",
            "评论获取未完成，保留待自动续跑，不执行外部动作",
        )
    conversation_result = _classify_conversation(analyze_conversation(issue))
    self_assignee = _self_assignee_result(evidence)
    if self_assignee:
        if conversation_result and conversation_result["category"] in {
            "reporter_followup", "reopened_followup", "assignee_followup",
        }:
            return conversation_result
        return self_assignee
    if not options.association_scan_complete or any(
        not pr_active(pr) and not pr_inactive(pr) for pr in evidence.linked_prs
    ):
        return _classification_result(
            "need_attention", "association_scan_incomplete",
            "PR关联扫描不完整，补查后再判断自提和选择负责人，不执行外部动作",
        )
    result = _classify_remaining(evidence)
    if conversation_result:
        # A new question keeps its category; a waiting state cannot hide a
        # missing PR-author assignment. Its separate state remains available.
        if conversation_result["bucket"] == "need_attention":
            conversation_result["auto_action"] = result["auto_action"]
            return conversation_result
        if result["bucket"] != "need_attention":
            return conversation_result
    return result


# --------------------------------------------------------------------------- #
# Report (Chinese, matches reference script format)
# --------------------------------------------------------------------------- #
def format_report(need_attention, no_attention, since_iso, all_clear):
    now_str = datetime.now(TZ_CHINA).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"===== 待处理 Issue ({now_str}) =====", ""]

    lines.append(f"【进入处理流程 ({len(need_attention)} 个）】")
    if need_attention:
        for item in need_attention:
            lines.append(f'  #{item["number"]} {item["title"]}')
            lines.append(f'    负责人: {item["assignee"] or "无"}')
            lines.append(f'    评论数: {item["comments_count"]}')
            lines.append(f'    原因: {item["reason"]}')
            lines.append(f'    链接: {item["url"]}')
            lines.append("")
    else:
        lines.append("  无")
        lines.append("")

    return "\n".join(lines)


def apply_processing_mode(result, single_mode):
    """Make an explicit single Issue actionable without losing triage evidence."""
    routed = dict(result)
    routed["classification_bucket"] = result["bucket"]
    routed["must_handle"] = bool(single_mode)
    routed["single_issue_override"] = bool(
        single_mode and result["bucket"] != "need_attention"
    )
    if single_mode:
        routed["bucket"] = "need_attention"
    return routed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def read_input(args):
    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                return json.load(f)
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError(
                "no JSON input received; the upstream fetch command may " "have failed"
            )
        return json.loads(text)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Error: cannot read classifier input — {exc}") from exc


def extract_issues(raw):
    if isinstance(raw, dict):
        return raw.get("issues", []) or []
    if isinstance(raw, list):
        return raw
    raise ValueError("Error: input JSON must be an object with 'issues' or an array")


def write_report(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class ClassifierRuntime(NamedTuple):
    """Validated values shared across one classifier run."""

    args: object
    cfg: dict
    issues: list
    repo: str
    api_base: str
    single_mode: bool
    since_iso: object
    time_scope_source: str
    token: str
    dry_run: bool
    post_fn: object
    rate_limiter: object
    automation_policy: dict


def _add_input_args(parser):
    parser.add_argument(
        "--config",
        default=None,
        help=f"Optional YAML config (default: {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Explicit target as owner/repo; conflicts require user selection",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Read issues JSON from file instead of stdin",
    )


def _add_run_policy_args(parser):
    parser.add_argument("--include-observations", action="store_true",
                        help="Include internal non-handled routing records for durable queue reconciliation")
    parser.add_argument(
        "--authorization-mode",
        choices=("interactive", "approved_batch"),
        default="interactive",
        help=(
            "Compatibility mode for the response/assignment workflow; "
            "classification itself never writes. Default: interactive."
        ),
    )
    parser.add_argument(
        "--no-auto-assign",
        action="store_true",
        help=(
            "Compatibility no-op: classification is always read-only and never "
            "POSTs /assign comments."
        ),
    )
    parser.add_argument(
        "--ignore-last-check",
        action="store_true",
        help="Classify the complete supplied input. An input filters.since value "
        "already takes precedence over last_check automatically.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable durable comment and PR-link cache reads and writes",
    )
    parser.add_argument(
        "--refresh-comments",
        action="store_true",
        help="Ignore valid cached comments and fetch required comments again",
    )
    parser.add_argument(
        "--no-update-last-check", action="store_true",
        help="Do not advance the repository cursor when classifying a subset of a batch",
    )
    parser.add_argument(
        "--pr-snapshot",
        help="Reuse a PR list snapshot within this batch; use a new path for each run. "
             "Repository/options must match and the snapshot must cover the time scope. --no-cache bypasses it.",
    )
    parser.add_argument(
        "--full-pr-linkage-scan",
        action="store_true",
        help="Use the native linkage API for every PR without a text link. "
        "Default: only verify ambiguous target-number matches.",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Classify open GitCode issues using a deterministic decision tree. "
            "Reads issues JSON from stdin (or --input), fetches related PRs, "
            "applies the tree, emits read-only automation strategies, and "
            "writes JSON to stdout plus a Chinese report to report_file."
        )
    )
    _add_input_args(parser)
    _add_run_policy_args(parser)
    return parser.parse_args(argv)


class RepositorySelectionRequired(ValueError):
    def __init__(self, result):
        super().__init__("Target repository requires user selection")
        self.result = result


def _runtime_input(args):
    raw = read_input(args)
    filters = raw.get("filters", {}) if isinstance(raw, dict) else {}
    input_repo = filters.get("repository")
    if args.repo and input_repo and args.repo != input_repo:
        raise RepositorySelectionRequired({
            "status": "needs_selection", "reason": "explicit_target_conflicts_with_input",
            "candidates": list(dict.fromkeys([args.repo, input_repo])),
        })
    selection = resolve_repository(
        Path.cwd(), target=args.repo or input_repo, config_path=args.config
    )
    if selection["status"] != "resolved":
        raise RepositorySelectionRequired(selection)
    cfg = load_config(selection["config_path"], allow_legacy=False)
    issues = [issue for issue in extract_issues(raw)
              if str(issue.get("state", "")).casefold() in {"open", "opened"}]
    return raw, filters, cfg, issues


def _classification_never_assigns(number, body, expected_assignee):
    """Compatibility callback; classification never performs assignment."""
    return False


def load_runtime(args):
    raw, filters, cfg, issues = _runtime_input(args)
    automation_policy = get_automation_policy(cfg)
    repo = cfg["repo"]
    api_base = cfg["gitcode_api"]
    single_mode = filters.get("mode") == "single"
    if single_mode and args.authorization_mode == "approved_batch":
        raise ValueError(
            "Error: approved_batch authorization is valid only for batch input"
        )
    since_iso, apply_updated_filter, time_scope_source = resolve_time_scope(
        raw,
        args.ignore_last_check,
        cfg["last_check_file"],
        repo,
        cfg["lookback_days"],
    )
    if apply_updated_filter:
        issues = filter_by_updated_since(issues, since_iso)
    token = os.environ.get("GITCODE_TOKEN", "")
    rate_limiter = SharedRateLimiter(rate_limit_path(cfg["cache_dir"]))
    # Classification is read-only.  Keep --no-auto-assign for CLI
    # compatibility, but never let authorization mode turn the classifier
    # into an assigner.
    dry_run = True
    return ClassifierRuntime(
        args,
        cfg,
        issues,
        repo,
        api_base,
        single_mode,
        since_iso,
        time_scope_source,
        token,
        dry_run,
        _classification_never_assigns,
        rate_limiter,
        automation_policy,
    )


def _finish_empty_run(runtime):
    if not runtime.single_mode and not runtime.args.no_update_last_check:
        save_last_check(runtime.cfg["last_check_file"], runtime.repo)
    output = {
        "total": 0,
        "responsibility_policy": runtime.cfg["responsibility"],
        "responsibility_policy_digest": policy_digest(runtime.cfg["responsibility"]),
        "by_responsibility": {level: 0 for level in (*LEVELS, "pending")},
        "listed_issues": [],
        "ignored_count": 0,
        "mode": "single" if runtime.single_mode else "batch",
        "authorization_mode": runtime.args.authorization_mode,
        "dry_run": True,
        "automation": {
            **runtime.automation_policy,
            "classification_read_only": True,
            "linked_pr_author_exception": (
                "PR-author assignment belongs to response; self-author first, "
                "otherwise select by Issue coverage; verified/reused response or self-authored exemption"
            ),
        },
        "transport": runtime.rate_limiter.snapshot(),
        "by_bucket": {"need_attention": 0, "no_attention": 0},
        "since": runtime.since_iso,
        "time_scope": {
            "source": runtime.time_scope_source,
            "since": runtime.since_iso,
        },
        "all_clear": True,
        "issues": [],
    }
    _write_stdout(json.dumps(output, indent=2, ensure_ascii=False))
    write_report(
        runtime.cfg["report_file"],
        format_report([], [], runtime.since_iso, all_clear=True),
    )


def _batch_prs(runtime, options):
    """Explicit run-local snapshot: subsequent scope reviews reuse one PR scan."""
    path = runtime.args.pr_snapshot if not runtime.args.no_cache else None
    identity = {"repo": runtime.repo, "api_base": runtime.api_base,
                "since": runtime.since_iso, "max_pages": options.max_pages}
    if path:
        cached = _read_json(path)
        previous = cached.get("identity", {}) if isinstance(cached, dict) else {}
        previous = previous if isinstance(previous, dict) else {}
        same_scan = all(
            previous.get(key) == identity.get(key)
            for key in ("repo", "api_base", "max_pages")
        )
        old_since = parse_iso(previous.get("since") or "") if isinstance(previous, dict) else None
        new_since = parse_iso(identity.get("since") or "")
        covers_window = ("since" in previous and previous["since"] is None) or (
            old_since is not None and new_since is not None and old_since <= new_since
        )
        valid_cache_shape = (
            isinstance(cached, dict)
            and isinstance(cached.get("prs"), list)
            and isinstance(cached.get("diagnostics"), dict)
        )
        complete_cache = valid_cache_shape and cached["diagnostics"].get("complete") is True
        reusable_cache = all((valid_cache_shape, same_scan, covers_window, complete_cache))
        if reusable_cache:
            diagnostics = dict(cached["diagnostics"], snapshot_hits=1, pages_requested=0)
            return cached["prs"], diagnostics
    prs, diagnostics = fetch_recent_prs(options)
    if path:
        _atomic_write_json(path, {"identity": identity, "prs": prs, "diagnostics": diagnostics})
    return prs, diagnostics


def _active_issues(runtime):
    active = []
    cache_hits = 0
    for issue in runtime.issues:
        issue.pop("_cached_classification", None)
        if _explicitly_ignored(issue, runtime.cfg):
            continue
        if not runtime.single_mode and _response_exemption(issue):
            continue
        if not runtime.single_mode and not runtime.args.no_cache and not runtime.args.refresh_comments:
            cached = load_settled(runtime.cfg["cache_dir"], runtime.repo, issue,
                                  runtime.cfg, _CACHE_REVISION)
            if cached is not None:
                issue["_cached_classification"] = cached
                cache_hits += 1
                continue
        if review_responsibility(issue, runtime.cfg["responsibility"])["level"] in {
            "handle", "list-only"
        }:
            active.append(issue)
    return active, cache_hits


def _comment_evidence(runtime, active, issue_pr_map, cache_hits):
    owner, repo_name = runtime.repo.split("/", 1)
    comment_api = RepoApiContext(
        make_session(rate_limiter=runtime.rate_limiter),
        runtime.api_base,
        owner,
        repo_name,
        runtime.token,
    )
    diagnostics = enrich_issues_with_comments(
        comment_api,
        active,
        cache_dir=None if runtime.args.no_cache else runtime.cfg["cache_dir"],
        refresh=runtime.args.refresh_comments,
        should_fetch=lambda issue: (
            should_fetch_comments(issue, issue_pr_map)
            if review_responsibility(issue, runtime.cfg["responsibility"])["level"] in {
                "handle", "list-only"
            }
            else (False, "responsibility_gate")
        ),
    )
    diagnostics["classification_cache_hits"] = cache_hits
    return diagnostics


def _collect_evidence(runtime):
    active, cache_hits = _active_issues(runtime)
    if not active:
        return {}, {"complete": True, "skipped": "responsibility_gate"}, {
            "complete": True, "incomplete_issue_numbers": [], "skipped": "responsibility_gate"
        }, {"skipped": "no_active_issues", "classification_cache_hits": cache_hits}
    pr_fetch_options = PRFetchOptions(
        api_base=runtime.api_base,
        repo=runtime.repo,
        token=runtime.token,
        since_iso=runtime.since_iso,
        max_pages=int(runtime.cfg["pr_fetch_pages"]),
        rate_limiter=runtime.rate_limiter,
    )
    prs, pr_fetch_diagnostics = _batch_prs(runtime, pr_fetch_options)
    issue_numbers = [
        issue.get("number") or issue.get("iid") for issue in active
    ]
    linkage_options = LinkageOptions(
        runtime.api_base,
        runtime.repo,
        runtime.token,
        target_issue_numbers=issue_numbers,
        api_budget=int(runtime.cfg["pr_linkage_api_budget"]),
        scan_mode=(
            "all"
            if runtime.args.full_pr_linkage_scan
            else runtime.cfg["pr_linkage_scan_mode"]
        ),
        cache_dir=None if runtime.args.no_cache else runtime.cfg["cache_dir"],
        rate_limiter=runtime.rate_limiter,
    )
    issue_pr_map, linkage_diagnostics = build_issue_pr_map(prs, linkage_options)
    comment_diagnostics = _comment_evidence(runtime, active, issue_pr_map, cache_hits)
    direct_prs, direct_diagnostics = collect_issue_prs(active, pr_fetch_options, prs)
    _merge_direct_prs(issue_pr_map, direct_prs, linkage_diagnostics, direct_diagnostics)
    return issue_pr_map, pr_fetch_diagnostics, linkage_diagnostics, comment_diagnostics


def _merge_direct_prs(issue_pr_map, direct_prs, linkage_diagnostics, direct_diagnostics):
    for number, prs in direct_prs.items():
        normalized = _invert_pr_refs(prs, {pr["number"]: {number} for pr in prs})
        combined = {pr["pr_number"]: pr for pr in issue_pr_map.get(number, [])}
        combined.update({pr["pr_number"]: pr for pr in normalized.get(number, [])})
        if combined:
            issue_pr_map[number] = list(combined.values())
    incomplete = set(linkage_diagnostics["incomplete_issue_numbers"])
    incomplete.update(direct_diagnostics["incomplete_issue_numbers"])
    linkage_diagnostics["incomplete_issue_numbers"] = sorted(incomplete, key=int)
    linkage_diagnostics["complete"] = linkage_diagnostics["complete"] and direct_diagnostics["complete"]
    linkage_diagnostics["issue_pr_scan"] = direct_diagnostics


def _classified_item(issue, routed, issue_pr_map):
    number = issue.get("number") or issue.get("iid")
    conversation = analyze_conversation(issue)
    return {
        "number": number,
        "title": issue.get("title", ""),
        "author": issue.get("author"),
        "url": issue.get("url", ""),
        "assignee": issue.get("assignee"),
        "comments_count": issue.get("comments_count", 0) or 0,
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "issue_age_days": issue.get("issue_age_days"),
        "first_response_sla": issue.get("first_response_sla", "unknown"),
        "state": issue.get("state", ""),
        "issue_state": issue.get("issue_state", ""),
        "fetch_sources": issue.get("fetch_sources", []),
        "conversation_state": conversation["state"],
        "waiting_on": conversation["waiting_on"],
        "latest_reporter_comment_id": conversation["latest_reporter_comment_id"],
        "latest_reporter_comment_at": conversation["latest_reporter_comment_at"],
        "latest_maintainer_comment_id": conversation["latest_maintainer_comment_id"],
        "latest_maintainer_comment_at": conversation["latest_maintainer_comment_at"],
        "followup_pending_since": conversation["pending_since"],
        "waiting_since": conversation["waiting_since"],
        "followup_sla": followup_sla(conversation["pending_since"]),
        "reopen_required": conversation["reopen_required"],
        "activate_required": conversation["activate_required"],
        "waiting_status_reconcile_required": conversation[
            "waiting_status_reconcile_required"
        ],
        "waiting_watch_required": conversation["waiting_watch_required"],
        "linked_prs": issue_pr_map.get(str(number), []),
        "bucket": routed["bucket"],
        "classification_bucket": routed["classification_bucket"],
        "category": routed["category"],
        "reason": routed["reason"],
        "auto_action": routed["auto_action"],
        "automation": routed.get("automation", routed["auto_action"]),
        "must_handle": routed["must_handle"],
        "single_issue_override": routed["single_issue_override"],
        "responsibility": routed.get("responsibility", "pending"),
        "responsibility_summary": routed.get("responsibility_summary", ""),
        "responsibility_evidence": routed.get("responsibility_evidence", []),
    }


def attention_sort_key(item):
    """Prioritize new reporter turns before ordinary first-look work."""
    category = item.get("category")
    followup = category in {
        "reporter_followup",
        "reopened_followup",
        "assignee_followup",
        "awaiting_assignee_setup",
        "awaiting_reporter_setup",
    }
    followup_urgent = followup and item.get("followup_sla") in {
        "breached",
        "at_risk",
    }
    first_response_urgent = item.get("first_response_sla") in {
        "breached",
        "at_risk",
    }
    age = item.get("issue_age_days")
    age = float(age) if isinstance(age, (int, float)) else -1
    if followup_urgent:
        rank = 0
    elif followup:
        rank = 1
    elif first_response_urgent:
        rank = 2
    elif age >= 7:
        rank = 3
    elif age >= 5:
        rank = 4
    else:
        rank = 5
    created = parse_iso(item.get("created_at", ""))
    return rank, created or datetime.max.replace(tzinfo=TZ_CHINA)


def _explicitly_ignored(issue, config):
    return str(issue.get("iid") or issue.get("number")) in {
        str(i) for i in config.get("ignored_issue_ids", [])
    }


def classify_with_responsibility(issue, options, policy, single_mode=False):
    review = review_responsibility(issue, policy)
    level = review["level"]
    exemption = _response_exemption(issue)
    if exemption and not single_mode:
        result = apply_processing_mode(exemption, False)
    elif level in {"handle", "list-only"}:
        result = apply_processing_mode(
            classify_one(issue, options), single_mode if level == "handle" else False
        )
        if level == "list-only":
            # Keep out-of-scope items non-actionable, but list only those that
            # would still need attention under the normal decision tree.
            result["bucket"] = "no_attention"
    else:
        category = "responsibility_review_required" if level == "pending" else f"responsibility_{level}"
        result = apply_processing_mode(
            _classification_result(
                "need_attention" if level == "pending" else "no_attention",
                category, review["summary"],
            ), False,
        )
    result.update(responsibility=level, responsibility_summary=review["summary"],
                  responsibility_evidence=review["evidence"])
    return result


def _classify_all(runtime, issue_pr_map, pr_diagnostics, linkage_diagnostics):
    need_attention = []
    no_attention = []
    incomplete_linkage = set(linkage_diagnostics["incomplete_issue_numbers"])
    for issue in runtime.issues:
        cached = issue.get("_cached_classification")
        if cached is not None:
            no_attention.append(dict(
                cached, fetch_sources=issue.get("fetch_sources", []),
                issue_age_days=issue.get("issue_age_days"),
                first_response_sla=issue.get("first_response_sla", "unknown"),
                followup_sla=followup_sla(cached.get("followup_pending_since")),
            ))
            continue
        number = issue.get("number") or issue.get("iid")
        comments_status = (issue.get("comments_fetch") or {}).get("status")
        options = ClassificationOptions(
            issue_pr_map,
            runtime.post_fn,
            runtime.dry_run,
            association_scan_complete=(
                pr_diagnostics["complete"] and str(number) not in incomplete_linkage
            ),
            comment_scan_complete=comments_status != "error",
            automation_policy=runtime.automation_policy,
        )
        if _explicitly_ignored(issue, runtime.cfg):
            routed = apply_processing_mode(_classification_result(
                "no_attention", "responsibility_ignore", "明确编号忽略规则"
            ), False)
            routed.update(responsibility="ignore", responsibility_summary="明确编号忽略规则",
                          responsibility_evidence=["配置 ignored_issue_ids 精确匹配"])
        else:
            routed = classify_with_responsibility(
                issue, options, runtime.cfg["responsibility"], runtime.single_mode
            )
        target = (
            need_attention if routed["bucket"] == "need_attention" else no_attention
        )
        item = _classified_item(issue, routed, issue_pr_map)
        target.append(item)
        if not runtime.args.no_cache and not runtime.single_mode:
            save_classification(ClassificationCacheEntry(
                runtime.cfg["cache_dir"], runtime.repo, issue,
                runtime.cfg, _CACHE_REVISION, item,
            ))
    need_attention.sort(key=attention_sort_key)
    return need_attention, no_attention


def _run_output(runtime, classified, diagnostics):
    need_attention, no_attention = classified
    pr_diagnostics, linkage_diagnostics, comment_diagnostics = diagnostics
    listed = []
    for item in no_attention:
        if item["responsibility"] == "list-only" and item["classification_bucket"] == "need_attention":
            listed.append(item)
    ignored = [item for item in no_attention if item["responsibility"] == "ignore"]
    visible = [item for item in no_attention if item["responsibility"] == "handle"]
    output = {
        "total": len(runtime.issues),
        "mode": "single" if runtime.single_mode else "batch",
        "authorization_mode": runtime.args.authorization_mode,
        "by_bucket": {
            "need_attention": len(need_attention),
            "no_attention": len(no_attention),
        },
        "since": runtime.since_iso,
        "time_scope": {
            "source": runtime.time_scope_source,
            "since": runtime.since_iso,
        },
        "all_clear": not need_attention,
        "dry_run": runtime.dry_run,
        "automation": {
            **runtime.automation_policy,
            "classification_read_only": True,
            "linked_pr_author_exception": (
                "PR-author assignment belongs to response; self-author first, "
                "otherwise select by Issue coverage; verified/reused response or self-authored exemption"
            ),
        },
        "transport": runtime.rate_limiter.snapshot(),
        "comment_fetch": comment_diagnostics,
        "association_scan": {
            "pr_fetch": pr_diagnostics,
            "linkage_fallback": linkage_diagnostics,
        },
        "responsibility_policy": runtime.cfg["responsibility"],
        "responsibility_policy_digest": policy_digest(runtime.cfg["responsibility"]),
        "by_responsibility": {
            level: sum(i["responsibility"] == level for i in need_attention + no_attention)
            for level in (*LEVELS, "pending")
        },
        "listed_issues": listed,
        "ignored_count": len(ignored),
        "issues": need_attention + visible,
    }
    if getattr(runtime.args, "include_observations", False):
        output["observations"] = no_attention
    return output, listed, visible


def _finish_run(runtime, classified, diagnostics):
    need_attention, _ = classified
    output, listed, visible_no_attention = _run_output(runtime, classified, diagnostics)
    all_clear = not need_attention
    if all_clear and not runtime.single_mode and not runtime.args.no_update_last_check:
        save_last_check(runtime.cfg["last_check_file"], runtime.repo)
    _write_stdout(json.dumps(output, indent=2, ensure_ascii=False))
    write_report(
        runtime.cfg["report_file"],
        format_report(need_attention, visible_no_attention, runtime.since_iso, all_clear)
        + ("\n【仅列举】\n" + "\n".join(
            f"- [#{i['number']}]({i['url']})：{i['responsibility_summary']}" for i in listed
        ) if listed else ""),
    )


def main(argv=None):
    args = parse_args(argv)

    try:
        runtime = load_runtime(args)
    except RepositorySelectionRequired as exc:
        _write_stdout(json.dumps(exc.result, ensure_ascii=False))
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    if not runtime.issues:
        _finish_empty_run(runtime)
        return 0
    evidence = _collect_evidence(runtime)
    issue_pr_map, pr_diagnostics, linkage_diagnostics, comment_diagnostics = evidence
    classified = _classify_all(
        runtime, issue_pr_map, pr_diagnostics, linkage_diagnostics
    )
    _finish_run(
        runtime,
        classified,
        (pr_diagnostics, linkage_diagnostics, comment_diagnostics),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
