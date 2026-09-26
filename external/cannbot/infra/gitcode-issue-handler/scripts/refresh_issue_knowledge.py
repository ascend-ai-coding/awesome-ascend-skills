#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Refresh the runtime Issue evidence corpus without rewriting curated cards.

The bundled ``knowledge/reference`` and ``knowledge/runbooks`` trees are
reviewed, versioned content.  This script only updates the target repository's
runtime corpus under ``.cannbot/gitcode-issue-handler``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import build_issue_knowledge as corpus_builder  # noqa: E402
import fetch_issues as issue_api  # noqa: E402
from cli_output import write_stdout  # noqa: E402
from runtime_paths import (  # noqa: E402
    KNOWLEDGE_CACHE,
    KNOWLEDGE_CORPUS,
    KNOWLEDGE_LOCK,
    KNOWLEDGE_REPORT,
    KNOWLEDGE_STATE,
    path_text,
    rate_limit_path,
)

CORPUS_SCHEMA = "issue-history.v2"
STATE_SCHEMA = "issue-knowledge-refresh.v1"
DEFAULT_TTL_SECONDS = 15 * 60
DEFAULT_FULL_INTERVAL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_OVERLAP_SECONDS = 5 * 60
LOGGER = logging.getLogger(__name__)


class SnapshotError(ValueError):
    """The persisted corpus and its commit metadata do not agree."""


class IncrementalOrderError(RuntimeError):
    """The remote list did not honor updated-time ordering."""


class RefreshContext(NamedTuple):
    """Shared identity and policy for one corpus refresh."""

    args: argparse.Namespace
    repository: str
    mode: str
    scan_started: datetime
    previous_state: dict
    previous_cursor: str | None = None


class CorpusInventory(NamedTuple):
    """Issue and PR records used to render one consistent snapshot."""

    issues: list[dict]
    source_total: int
    excluded_numbers: set[str]
    pr_records: dict[str, dict]


class IncrementalContext(NamedTuple):
    """API and snapshot inputs shared by incremental merge helpers."""

    args: argparse.Namespace
    api: object
    limiter: object
    boundary: datetime
    old_state: dict


class RefreshLock:
    """Small cross-process lock around one repository's refresh artifacts."""

    def __init__(self, path: str | Path, timeout: float):
        self.path = Path(path)
        self.timeout = max(0.0, timeout)
        self.file_obj = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file_obj = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    self.file_obj.close()
                    self.file_obj = None
                    raise TimeoutError(
                        f"knowledge refresh lock is busy: {self.path}"
                    ) from exc
                time.sleep(0.05)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_obj is not None:
            fcntl.flock(self.file_obj.fileno(), fcntl.LOCK_UN)
            self.file_obj.close()
        return False


def _now() -> datetime:
    return datetime.now(issue_api.TZ_CHINA)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(f"expected object in {path}")
    return payload


def load_valid_snapshot(corpus_path: str | Path, state_path: str | Path) -> tuple[dict, dict]:
    """Load only a snapshot committed by this lifecycle manager."""
    corpus_path = Path(corpus_path)
    state = _read_json(state_path)
    corpus = _read_json(corpus_path)
    if state.get("schema_version") != STATE_SCHEMA:
        raise SnapshotError("refresh state schema changed")
    if (corpus.get("metadata") or {}).get("schema_version") != CORPUS_SCHEMA:
        raise SnapshotError("corpus schema changed")
    if not isinstance(corpus.get("issues"), list):
        raise SnapshotError("corpus issues must be a list")
    actual_digest = _digest(corpus_path.read_bytes())
    if state.get("corpus_sha256") != actual_digest:
        raise SnapshotError("corpus digest does not match refresh state")
    if state.get("repository") != (corpus.get("metadata") or {}).get("repository"):
        raise SnapshotError("corpus repository does not match refresh state")
    return corpus, state


def _seconds_since(timestamp: str | None, now: datetime) -> float | None:
    parsed = issue_api.parse_iso(timestamp or "")
    return None if parsed is None else max(0.0, (now - parsed).total_seconds())


def choose_refresh_mode(args, snapshot, now: datetime) -> str:
    """Return ``full``, ``incremental``, or ``skip_fresh``."""
    if args.force_full or snapshot is None:
        return "full"
    _, state = snapshot
    owner, repo = issue_api.parse_repo_path(args.url)
    if state.get("repository") != f"{owner}/{repo}":
        return "full"
    policy = state.get("policy") or {}
    if bool(policy.get("include_self_assigned")) != args.include_self_assigned:
        return "full"
    if bool(policy.get("skip_prs")) != args.skip_prs:
        return "full"
    last_success_age = _seconds_since(state.get("last_success_at"), now)
    if last_success_age is not None and last_success_age < args.ttl_seconds:
        return "skip_fresh"
    full_age = _seconds_since(state.get("last_full_refresh_at"), now)
    if full_age is None or full_age >= args.full_interval_seconds:
        return "full"
    return "incremental"


def _is_self_assigned(issue: dict) -> bool:
    author = issue.get("author")
    assignee = issue.get("assignee")
    return bool(author and assignee and author == assignee)


def _pr_record(pr: dict) -> dict:
    text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
    refs = sorted(
        corpus_builder.extract_issue_references(text),
        key=lambda value: int(value),
    )
    return {
        "number": pr.get("number"),
        "state": pr.get("state"),
        "title": pr.get("title") or "",
        "url": pr.get("html_url") or "",
        "updated_at": pr.get("updated_at") or "",
        "head_sha": (pr.get("head") or {}).get("sha") or "",
        "issue_refs": refs,
    }


def _pr_map(records: dict[str, dict], issue_numbers) -> dict[str, list[dict]]:
    targets = {str(number) for number in issue_numbers}
    mapping = {number: [] for number in targets}
    for record in records.values():
        info = {
            "number": record.get("number"),
            "state": record.get("state"),
            "title": record.get("title") or "",
            "url": record.get("url") or "",
        }
        for number in set(record.get("issue_refs") or []) & targets:
            mapping[number].append(info)
    return mapping


def _annotate(issues: list[dict], pr_records: dict[str, dict]) -> None:
    mapping = _pr_map(pr_records, (issue.get("iid") for issue in issues))
    for issue in issues:
        issue["linked_prs"] = mapping.get(str(issue.get("iid")), [])
        issue["issue_type"] = corpus_builder.classify_issue_type(issue)
        issue["evidence_signals"] = corpus_builder.evidence_signals(issue)
        issue["handling_outcome"] = corpus_builder.handling_outcome(issue)


def _metadata(context: RefreshContext):
    args = context.args
    return {
        "schema_version": CORPUS_SCHEMA,
        "generated_at": _iso(_now()),
        "source_url": args.url,
        "repository": context.repository,
        "filter": (
            "all issues"
            if args.include_self_assigned
            else "author != assignee, including missing assignee"
        ),
        "trust": "runtime_evidence_only",
        "refresh": {
            "mode": context.mode,
            "scan_started_at": _iso(context.scan_started),
            "previous_cursor_at": context.previous_cursor,
            "overlap_seconds": args.overlap_seconds,
        },
    }


def _compose_corpus(
    context: RefreshContext,
    inventory: CorpusInventory,
) -> dict:
    issues = sorted(inventory.issues, key=lambda issue: int(issue.get("iid") or 0))
    metadata = _metadata(context)
    summary = corpus_builder.build_summary(
        issues, inventory.source_total, len(inventory.excluded_numbers)
    )
    return {
        "metadata": metadata,
        "summary": summary,
        "issues": [corpus_builder.compact_issue(issue) for issue in issues],
    }


def _fetch_updated_items(api, limiter, endpoint: str, boundary: datetime) -> list[dict]:
    """Fetch an overlap window, requiring updated-desc ordering to stop safely."""
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/{endpoint}"
    result = []
    page = 1
    previous_updated = None
    while True:
        data = corpus_builder.rate_limited_get(
            limiter,
            api,
            url,
            params={
                "state": "all",
                "page": page,
                "per_page": 100,
                "sort": "updated",
                "direction": "desc",
            },
        )
        batch = data if isinstance(data, list) else []
        if not batch:
            break
        parsed = [issue_api.parse_iso(item.get("updated_at") or "") for item in batch]
        ordered = [value for value in parsed if value is not None]
        if any(left < right for left, right in zip(ordered, ordered[1:])):
            raise IncrementalOrderError(f"{endpoint} list is not updated-desc ordered")
        if previous_updated is not None and ordered and previous_updated < ordered[0]:
            raise IncrementalOrderError(f"{endpoint} pagination order changed")
        if ordered:
            previous_updated = ordered[-1]
        result.extend(
            item
            for item, updated in zip(batch, parsed)
            if updated is None or updated >= boundary
        )
        all_older = bool(parsed) and all(
            updated is not None and updated < boundary for updated in parsed
        )
        if len(batch) < 100 or all_older:
            break
        page += 1
    return result


def _state_payload(
    context: RefreshContext,
    inventory: CorpusInventory,
    corpus_data: bytes,
) -> dict:
    args = context.args
    return {
        "schema_version": STATE_SCHEMA,
        "corpus_schema_version": CORPUS_SCHEMA,
        "repository": context.repository,
        "source_url": args.url,
        "last_attempt_at": _iso(_now()),
        "last_success_at": _iso(_now()),
        "last_full_refresh_at": (
            _iso(_now())
            if context.mode == "full"
            else context.previous_state.get("last_full_refresh_at")
        ),
        "cursor_at": _iso(context.scan_started),
        "last_refresh_mode": context.mode,
        "last_refresh_status": "success",
        "last_error": None,
        "corpus_sha256": _digest(corpus_data),
        "source_total": inventory.source_total,
        "excluded_issue_numbers": sorted(inventory.excluded_numbers, key=int),
        "pr_records": inventory.pr_records,
        "policy": {
            "ttl_seconds": args.ttl_seconds,
            "full_interval_seconds": args.full_interval_seconds,
            "overlap_seconds": args.overlap_seconds,
            "include_self_assigned": args.include_self_assigned,
            "skip_prs": args.skip_prs,
        },
    }


def _full_refresh(args, scan_started: datetime) -> tuple[dict, dict]:
    corpus_context = corpus_builder.load_corpus_context(args)
    repository = f"{corpus_context.owner}/{corpus_context.repo}"
    corpus_builder.fetch_comments(
        corpus_context.issues,
        corpus_context.api,
        max(1, args.workers),
        corpus_context.limiter,
        cache_dir=args.cache_dir or None,
        refresh=True,
    )
    pr_records = {}
    if not args.skip_prs:
        for pr in corpus_builder.fetch_all_prs(
            corpus_context.api, corpus_context.limiter
        ):
            number = pr.get("number")
            if number is not None:
                pr_records[str(number)] = _pr_record(pr)
    _annotate(corpus_context.issues, pr_records)
    excluded = {
        str(issue.get("iid"))
        for issue in corpus_context.all_issues
        if _is_self_assigned(issue)
    }
    refresh_context = RefreshContext(args, repository, "full", scan_started, {})
    inventory = CorpusInventory(
        corpus_context.issues,
        len(corpus_context.all_issues),
        excluded,
        pr_records,
    )
    corpus = _compose_corpus(refresh_context, inventory)
    corpus_data = _json_bytes(corpus)
    state = _state_payload(refresh_context, inventory, corpus_data)
    return corpus, state


def _incremental_api(args, old_state: dict):
    """Build the API context and overlap boundary for one incremental scan."""
    token = issue_api.resolve_token(args.token)
    api_base = issue_api.resolve_api_base(args.api_base, args.url)
    owner, repo = issue_api.parse_repo_path(args.url)
    repository = f"{owner}/{repo}"
    api = issue_api.RepoApiContext(
        issue_api.make_session(rate_limit_dir=rate_limit_path(args.cache_dir)),
        api_base,
        owner,
        repo,
        token,
    )
    limiter = corpus_builder.RequestLimiter(args.request_interval)
    cursor = issue_api.parse_iso(old_state.get("cursor_at") or "")
    if cursor is None:
        raise SnapshotError("incremental cursor is missing")
    boundary = cursor - timedelta(seconds=args.overlap_seconds)
    return repository, api, limiter, boundary


def _restore_issues(old_corpus: dict) -> dict[str, dict]:
    """Restore compact public records to the builder's normalized shape."""
    current = {}
    for compact in old_corpus.get("issues") or []:
        number = compact.get("number")
        if number is None:
            continue
        restored = dict(compact)
        restored["iid"] = number
        current[str(number)] = restored
    return current


def _merge_changed_issues(context: IncrementalContext, current):
    """Merge changed Issues and return totals plus records needing comments."""
    args, api, limiter, boundary, old_state = context
    excluded = {str(number) for number in old_state.get("excluded_issue_numbers") or []}
    known_numbers = set(current) | excluded
    source_total = int(old_state.get("source_total", len(known_numbers)))
    changed_included = []
    for raw in _fetch_updated_items(api, limiter, "issues", boundary):
        issue = issue_api.normalize_issue(raw)
        number = str(issue.get("iid"))
        if number == "None":
            continue
        if number not in known_numbers:
            source_total += 1
            known_numbers.add(number)
        if not args.include_self_assigned and _is_self_assigned(issue):
            current.pop(number, None)
            excluded.add(number)
            continue
        excluded.discard(number)
        current[number] = issue
        changed_included.append(issue)
    corpus_builder.fetch_comments(
        changed_included,
        api,
        max(1, args.workers),
        limiter,
        cache_dir=args.cache_dir or None,
    )
    for issue in changed_included:
        current[str(issue.get("iid"))] = issue
    return source_total, excluded


def _merge_pr_records(context: IncrementalContext):
    """Merge PR linkage records from the same overlap window."""
    args, api, limiter, boundary, old_state = context
    pr_records = dict(old_state.get("pr_records") or {})
    if args.skip_prs:
        return {}
    for pr in _fetch_updated_items(api, limiter, "pulls", boundary):
        if pr.get("number") is not None:
            pr_records[str(pr.get("number"))] = _pr_record(pr)
    return pr_records


def _incremental_refresh(args, snapshot, scan_started: datetime) -> tuple[dict, dict]:
    old_corpus, old_state = snapshot
    repository, api, limiter, boundary = _incremental_api(args, old_state)
    incremental = IncrementalContext(args, api, limiter, boundary, old_state)
    current = _restore_issues(old_corpus)
    source_total, excluded = _merge_changed_issues(incremental, current)
    pr_records = _merge_pr_records(incremental)

    issues = list(current.values())
    _annotate(issues, pr_records)
    context = RefreshContext(
        args,
        repository,
        "incremental",
        scan_started,
        old_state,
        previous_cursor=old_state.get("cursor_at"),
    )
    inventory = CorpusInventory(
        issues,
        source_total,
        excluded,
        pr_records,
    )
    corpus = _compose_corpus(context, inventory)
    corpus_data = _json_bytes(corpus)
    state = _state_payload(context, inventory, corpus_data)
    return corpus, state


def perform_refresh(args, mode: str, snapshot, scan_started: datetime) -> tuple[dict, dict]:
    if mode == "full":
        return _full_refresh(args, scan_started)
    try:
        return _incremental_refresh(args, snapshot, scan_started)
    except IncrementalOrderError:
        LOGGER.warning("updated ordering was not reliable; rebuilding full corpus")
        return _full_refresh(args, scan_started)


def _stage(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file_obj:
        file_obj.write(data)
        file_obj.flush()
        os.fsync(file_obj.fileno())
        return Path(file_obj.name)


def commit_snapshot(args, corpus: dict, state: dict) -> None:
    """Commit corpus/report/state, rolling back if a replacement fails."""
    corpus_path = Path(args.output)
    report_path = Path(args.report)
    state_path = Path(args.state_file)
    corpus_data = _json_bytes(corpus)
    if state.get("corpus_sha256") != _digest(corpus_data):
        raise ValueError("state digest was not computed from the corpus")
    report = corpus_builder.render_report(
        corpus["metadata"],
        corpus["summary"],
        corpus["issues"],
        per_type=args.candidates_per_type,
    ).encode("utf-8")
    state_data = _json_bytes(state)
    targets = ((corpus_path, corpus_data), (report_path, report), (state_path, state_data))
    old_data = {path: path.read_bytes() if path.exists() else None for path, _ in targets}
    staged = {path: _stage(path, data) for path, data in targets}
    replaced = []
    try:
        for path, _ in targets:
            os.replace(staged[path], path)
            replaced.append(path)
    except OSError:
        for path in reversed(replaced):
            previous = old_data[path]
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(_stage(path, previous), path)
        raise
    finally:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)


def _record_failure(state_path, state, error: str) -> None:
    if state is None:
        return
    updated = dict(state)
    updated["last_attempt_at"] = _iso(_now())
    updated["last_refresh_status"] = "stale_fallback"
    updated["last_error"] = error
    corpus_builder.write_text(state_path, _json_bytes(updated).decode("utf-8"))


def _result(status: str, **extra) -> dict:
    return {"status": status, **extra}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="GitCode repository URL")
    parser.add_argument("--token", default=None, help="defaults to GITCODE_TOKEN")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=1.35)
    parser.add_argument("--cache-dir", default=path_text(KNOWLEDGE_CACHE))
    parser.add_argument("--include-self-assigned", action="store_true")
    parser.add_argument("--skip-prs", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument(
        "--ttl-seconds", type=_nonnegative_int, default=DEFAULT_TTL_SECONDS
    )
    parser.add_argument(
        "--full-interval-seconds",
        type=_nonnegative_int,
        default=DEFAULT_FULL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--overlap-seconds",
        type=_nonnegative_int,
        default=DEFAULT_OVERLAP_SECONDS,
    )
    parser.add_argument("--lock-timeout", type=float, default=2.0)
    parser.add_argument("--output", default=path_text(KNOWLEDGE_CORPUS))
    parser.add_argument("--report", default=path_text(KNOWLEDGE_REPORT))
    parser.add_argument("--state-file", default=path_text(KNOWLEDGE_STATE))
    parser.add_argument("--lock-file", default=path_text(KNOWLEDGE_LOCK))
    parser.add_argument("--candidates-per-type", type=int, default=5)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero when refresh fails even if a prior snapshot is usable",
    )
    return parser.parse_args(argv)


def _load_optional_snapshot(args):
    try:
        return load_valid_snapshot(args.output, args.state_file)
    except SnapshotError:
        return None


def _run_locked_refresh(args):
    """Refresh while holding the artifact lock and return the public result."""
    snapshot = _load_optional_snapshot(args)
    scan_started = _now()
    mode = choose_refresh_mode(args, snapshot, scan_started)
    if mode == "skip_fresh":
        return _result(
            "fresh", mode="skip", snapshot_usable=True, corpus=args.output
        )
    corpus, state = perform_refresh(args, mode, snapshot, scan_started)
    commit_snapshot(args, corpus, state)
    return _result(
        "refreshed",
        mode=state.get("last_refresh_mode"),
        snapshot_usable=True,
        corpus=args.output,
        issues=len(corpus.get("issues") or []),
    )


def _fallback_result(args, snapshot, exc):
    safe_error = issue_api.redact_token(f"{type(exc).__name__}: {exc}")
    return _result(
        "stale_fallback" if snapshot else "unavailable",
        mode="none",
        snapshot_usable=bool(snapshot),
        corpus=args.output if snapshot else None,
        error=safe_error,
    )


def _handle_refresh_error(args, snapshot, exc) -> int:
    payload = _fallback_result(args, snapshot, exc)
    write_stdout(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if args.strict else 0


def main(argv=None):
    args = parse_args(argv)
    snapshot = _load_optional_snapshot(args)
    try:
        with RefreshLock(args.lock_file, args.lock_timeout):
            payload = _run_locked_refresh(args)
        write_stdout(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except TimeoutError as exc:
        snapshot = _load_optional_snapshot(args)
        return _handle_refresh_error(args, snapshot, exc)
    except (ValueError, requests.RequestException, OSError) as exc:
        safe_error = issue_api.redact_token(f"{type(exc).__name__}: {exc}")
        prior_state = snapshot[1] if snapshot else None
        try:
            _record_failure(args.state_file, prior_state, safe_error)
        except OSError:
            pass
        return _handle_refresh_error(args, snapshot, exc)


if __name__ == "__main__":
    raise SystemExit(main())
