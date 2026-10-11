#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Durable, read-only Issue intake and agent-task orchestration.

No command in this module publishes a comment, assigns, closes, commits or
pushes. Existing mutation executors and their authorization gates remain the
only publication path. See --help and references/pipeline.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from handler_config import ConfigError, load_handler_config
from fetch_cache import _atomic_write_json, load_comments
from pipeline_store import PipelineStore
from report_runs import ensure_run, publish_materials
from scope_cache import activity_digest, content_digest, digest, make_entry, restore, task_digest
from responsibility import policy_digest
import pipeline_tasks as queue
from pipeline_evidence import inspect_sources, verify_identities, content_hash
from pipeline_delivery import (
    assignment_only,
    assignment_action,
    MaterialError,
    response_enums,
    response_template,
    response_instruction,
    validate_response,
    materialize,
    report_state,
)
from protocol_output import write_json
from call_options import MISSING, bind_extra

RUNTIME = Path(".cannbot/gitcode-issue-handler")
TERMINAL = {"observe", "closed", "ignored"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(issue, repo):
    if not isinstance(issue, dict):
        raise ValueError("invalid_issue")
    iid = str(issue.get("iid") or issue.get("number") or "")
    if not iid.isdigit() or int(iid) < 1:
        raise ValueError("invalid_issue_number")
    expected = f"https://gitcode.com/{repo}/issues/{iid}"
    if issue.get("url") and issue["url"].rstrip("/").casefold() != expected.casefold():
        raise ValueError("issue_repository_mismatch")
    value = copy.deepcopy(issue)
    value.update(iid=iid, number=iid, url=expected)
    return value


def ignored_ids(cfg):
    return {str(i) for i in cfg.get("ignored_issue_ids", [])}


def scope_task_digest(entry, state):
    if entry.get("scope", {}).get("source_mode") == "moving":
        return digest([entry["input_digest"], state["scan"].get("id")])
    return entry["input_digest"]


def response_digest(entry, item):
    keys = (
        "bucket",
        "category",
        "assignee",
        "linked_prs",
        "responsibility",
        "latest_maintainer_comment_id",
        "latest_reporter_comment_id",
        "conversation_state",
        "auto_action",
    )
    return digest([entry["input_digest"], {k: item.get(k) for k in keys}, entry.get("response_revision", 0)])


def read_command(script, args, root):
    started = time.monotonic()
    result = subprocess.run([sys.executable, str(HERE / script), *args], cwd=root, capture_output=True, text=True)
    if result.returncode:
        # Child errors may contain remote/private body text: do not echo them.
        raise ValueError(f"{script}_failed_exit_{result.returncode}")
    try:
        value = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError(f"{script}_invalid_output") from exc
    return value, round(time.monotonic() - started, 3)


def _legacy_scope(issue, cfg, path):
    review = issue.get("responsibility_review")
    if not review:
        return None
    try:
        return make_entry(issue, cfg["responsibility"], review, provenance=str(path))
    except ValueError:
        return None


def _import_legacy_issues(state, old, repo, cfg, path):
    imported = 0
    for raw in old.get("issues", []):
        issue = normalize(raw, repo)
        # Reports lack full body/metadata; keep as refresh requests,
        # never overwrite an actual historical Issue snapshot.
        entry = state["issues"].setdefault(issue["iid"], {"issue": issue, "phase": "refresh"})
        if "description" in issue or "body" in issue:
            entry["issue"] = issue
            scope = _legacy_scope(issue, cfg, path)
            if scope is not None:
                entry["scope"] = scope
                imported += 1
        entry["phase"] = "refresh"
    return imported


def _import_legacy_operations(state, old, path):
    for op in old.get("external_operations", []):
        if op.get("status") in {"executed", "verified", "reused", "cancelled", "skipped"}:
            continue
        if op.get("kind") not in {
            "issue_comment",
            "issue_assignment",
            "issue_state_change",
            "pr_create",
            "first_ci",
            "branch_push",
            "direct_push",
        }:
            continue
        key = f"legacy:{path.parent.name}:{op.get('operation_id', 'unknown')}"
        # An explicit server rejection is a known failed attempt, not
        # an unknown write. Preserve its target/outcome without retrying.
        rejected = (
            op.get("status") == "failed" and op.get("execution_evidence", {}).get("remote_http_result") == "rejected"
        )
        state["operations"].setdefault(
            key,
            {
                "status": "verified" if rejected else "needs_review",
                "source": str(path),
                "operation_id": op.get("operation_id"),
                "kind": op.get("kind"),
                "issue_iids": op.get("issue_iids", []),
                "target": op.get("target"),
                "outcome": "confirmed_rejected" if rejected else "unknown",
                "execution_evidence": op.get("execution_evidence", {}),
            },
        )


def migrate(state, runtime, cfg):
    """Import scope evidence and outstanding operations, never old completion."""
    if state["migration_done"]:
        return
    paths = list((runtime / "data").glob("*reviewed*.json"))
    paths += list((runtime / "reports").glob("*/run_state.json"))
    imported = 0
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        try:
            old = read_json(path)
            repo = old.get("filters", {}).get("repository") or old.get("run", {}).get("repository")
            if repo != state["repository"]:
                continue
            imported += _import_legacy_issues(state, old, repo, cfg, path)
            _import_legacy_operations(state, old, path)
        except (ValueError, TypeError, KeyError, OSError):
            state.setdefault("migration_warnings", []).append(str(path))
    state["migration_done"] = True
    state["needs_full_scan"] = True
    state["migration_scope_imports"] = imported


def scan_complete(snapshot):
    primary = snapshot.get("primary_scan", {})
    follow = snapshot.get("filters", {}).get("follow_up", {})
    return primary.get("complete") is True and follow.get("complete") is True


def _ingest_issues(state, incoming, scan_id, filters, full):
    ids = [issue["iid"] for issue in incoming]
    for issue in incoming:
        iid = issue["iid"]
        entry = state["issues"].setdefault(iid, {})
        changed = content_digest(entry.get("issue", {})) != content_digest(issue) or activity_digest(
            entry.get("issue", {})
        ) != activity_digest(issue)
        entry.update(issue=issue, fresh_scan_id=scan_id)
        if changed or entry.get("phase") not in {"attention", "waiting"}:
            entry["phase"] = "scope"
        # Current scan evidence is needed even when PR state did not change
        # the Issue updated_at. Never reuse a final PR-dependent classification.
        entry.pop("classification_scan_id", None)
    closed = set(filters.get("follow_up", {}).get("updated_scan", {}).get("closed", []))
    closed.update(filters.get("follow_up", {}).get("watchlist_refresh", {}).get("closed", []))
    for iid, entry in state["issues"].items():
        if iid in {str(n) for n in closed}:
            entry["issue"]["state"] = "closed"
            entry["phase"] = "closed"
        else:
            needs_refresh = (
                full or entry.get("phase") not in TERMINAL or entry.get("classification", {}).get("linked_prs")
            )
            if iid in ids or not needs_refresh:
                continue
            # Absence from an open list isn't evidence of closure.
            entry["phase"] = "refresh"


def ingest(state, snapshot, cfg, now):
    filters = snapshot.get("filters", {})
    if filters.get("repository") != state["repository"]:
        raise ValueError("snapshot_repository_mismatch")
    if not isinstance(snapshot.get("issues"), list):
        raise ValueError("invalid_snapshot")
    incoming = [normalize(i, state["repository"]) for i in snapshot["issues"]]
    ids = [i["iid"] for i in incoming]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_issue_in_snapshot")
    scan_id = uuid.uuid4().hex
    complete = scan_complete(snapshot)
    full = (
        complete
        and not snapshot["primary_scan"].get("skipped")
        and not filters.get("since")
        and not filters.get("until")
    )
    state.pop("delivery_report", None)
    state["scan"].update(id=scan_id, attempted_at=now, complete=complete)
    if state.get("report_run"):
        state["report_run"]["scan_ids"].append(scan_id)
    if complete:
        state["scan"]["completed_at"] = now
        cursor = filters.get("follow_up", {}).get("cursor_proposed")
        if cursor:
            parsed = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.timestamp() > now + 60:
                raise ValueError("invalid_scan_cursor")
            old = state["scan"].get("cursor")
            if old is None or parsed > datetime.fromisoformat(old.replace("Z", "+00:00")):
                state["scan"]["cursor"] = cursor
    if full:
        state["scan"]["full_completed_at"] = now
        state["needs_full_scan"] = False
    _ingest_issues(state, incoming, scan_id, filters, full)
    return {"complete": complete, "full": full, "received": len(incoming)}


def invalidate_response(state, entry, reason, now):
    result = entry.pop("prepared_response", None)
    if result:
        entry.setdefault("response_history", []).append(
            {"reason": reason, "at": now, "result": result, "artifacts": entry.get("response_file_map", {})}
        )
        entry["response_revision"] = entry.get("response_revision", 0) + 1
    for field in (
        "response_artifacts",
        "response_file_map",
        "related_code",
        "prepared_source_records",
        "prepared_digest",
        "waiting_on",
    ):
        entry.pop(field, None)
    state.pop("delivery_report", None)


def _apply_scope_review(state, cfg, iid, entry, now):
    policy = cfg["responsibility"]
    issue = entry["issue"]
    review, reason = restore(entry.get("scope"), issue, policy)
    if reason == "moving_source_requires_review" and entry.get("scope", {}).get("accepted_scan_id") == state[
        "scan"
    ].get("id"):
        review = entry["scope"]["review"]
    entry["scope_cache_reason"] = reason
    if review:
        issue["responsibility_review"] = review
        if review["level"] == "ignore":
            entry["phase"] = "ignored"
        elif entry.get("classification_scan_id") != state["scan"].get("id"):
            entry["phase"] = "classify"
    else:
        issue.pop("responsibility_review", None)
        entry["phase"] = "scope"
        comments = load_comments(cfg["cache_dir"], state["repository"], issue)
        payload = {
            "repository": state["repository"],
            "issue": issue,
            "policy": policy,
            "policy_digest": policy_digest(policy),
            "previous_review": entry.get("scope"),
            "reason": reason,
            "comments": comments,
            "comments_complete": comments is not None,
            "instruction": (
                "按策略核查责任范围，依据充分即填写结果模板并返回；新活动只需核对增量。"
                "证据不足返回 needs_evidence，技术诊断由后续 response 任务处理。"
            ),
            "result_contract": {
                "decision": "handle|list-only|ignore|needs_evidence|needs_escalation",
                "summary": "非空依据说明",
                "evidence": ["实际核查的证据"],
                "source_mode": "fixed|moving",
            },
        }
        queue.ensure_task(
            state,
            iid,
            "scope",
            scope_task_digest(entry, state),
            payload,
            now,
            effort=1 if reason in {"activity_changed", "missing_review"} else 2,
        )


def _supersede_tasks(state, iid, protected):
    for task in state["tasks"].values():
        if task["iid"] == iid and task["status"] not in protected:
            task["status"] = "superseded"


def synchronize(state, cfg, now):
    policy, ignored = cfg["responsibility"], ignored_ids(cfg)
    for iid, entry in state["issues"].items():
        issue = entry["issue"]
        previous_fingerprint = entry.get("input_digest")
        fingerprint = task_digest(issue, policy, ignored)
        entry["input_digest"] = fingerprint
        if previous_fingerprint != fingerprint:
            invalidate_response(state, entry, "issue_or_policy_changed", now)
            _supersede_tasks(state, iid, {"superseded"})
            entry.pop("classification_scan_id", None)
            if entry.get("phase") != "refresh":
                entry["phase"] = "scope"
        if issue.get("state") in {"closed", "close"}:
            entry["phase"] = "closed"
        elif issue.get("state") not in {"open", "opened"}:
            entry["phase"] = "refresh"
            continue
        elif iid in ignored:
            entry["phase"] = "ignored"
        elif entry.get("phase") == "refresh":
            continue
        else:
            _apply_scope_review(state, cfg, iid, entry, now)
        if entry["phase"] in {"closed", "ignored"}:
            _supersede_tasks(state, iid, {"accepted", "superseded"})


def classify_ready(state, cfg, config_path, root, *args, **kwargs):
    store, now = bind_extra(args, kwargs, ("store", "now"), (MISSING, MISSING))
    entries = [e for e in state["issues"].values() if e["phase"] == "classify"]
    if not entries:
        return
    scan_id = state["scan"]["id"]
    batch = uuid.uuid4().hex
    folder = ensure_run(state, store, now) / "_internal" / "scans" / scan_id
    input_path = store.artifact(
        folder / f"input-{batch}.json",
        {"filters": {"mode": "batch", "repository": state["repository"]}, "issues": [e["issue"] for e in entries]},
    )
    result, seconds = read_command(
        "classify_issues.py",
        [
            "--input",
            input_path,
            "--config",
            str(config_path),
            "--ignore-last-check",
            "--no-update-last-check",
            "--include-observations",
            "--pr-snapshot",
            str(store.root / folder / "pr-snapshot.json"),
        ],
        root,
    )
    path = store.artifact(folder / f"classification-{batch}.json", result)
    for entry in entries:
        comments = load_comments(cfg["cache_dir"], state["repository"], entry["issue"])
        if comments is not None:
            entry["issue"]["comments"] = comments
            entry["issue"]["comments_fetch"] = {"status": "cached"}
    apply_classification(state, result, entries, path, now)
    state["metrics"].append({"stage": "classify", "seconds": seconds, "items": len(entries), "at": now})


def apply_classification(state, result, entries, path, now):
    items = {
        str(i["number"]): i
        for i in result.get("observations", []) + result.get("issues", []) + result.get("listed_issues", [])
    }
    for entry in entries:
        issue, iid = entry["issue"], entry["issue"]["iid"]
        item = items.get(iid)
        if not item:
            # Missing classifier output is not a completed task.
            entry["classification_error"] = "missing_classification"
            continue
        if entry.get("prepared_response") and entry.get("prepared_digest") != response_digest(entry, item):
            invalidate_response(state, entry, "classification_changed", now)
        entry.update(classification=item, classification_path=path, classification_scan_id=state["scan"]["id"])
        entry.pop("classification_error", None)
        if item["bucket"] == "no_attention" or item.get("responsibility") in {"ignore", "list-only"}:
            entry["phase"] = "observe"
            _supersede_tasks(state, iid, {"accepted", "superseded"})
            continue
        entry["phase"] = "attention"
        category = item.get("category", "")
        priority = 0 if "followup" in category else 1 if item.get("first_response_sla") == "breached" else 2
        # Child is evidence/draft-only. Scope completion does not authorize any
        # of the downstream mutation stages.
        queue.ensure_task(
            state,
            iid,
            "response",
            response_digest(entry, item),
            {
                "issue": issue,
                "classification": item,
                "classification_path": path,
                "quality_version": 2,
                "instruction": response_instruction(item),
                "result_contract": response_template(item),
            },
            now,
            priority=priority,
            effort=2,
        )
        if entry.get("prepared_response"):
            entry["phase"] = "waiting"


def validate_result_header(result, stage):
    if not isinstance(result, dict):
        raise ValueError("invalid_result_object")
    if not isinstance(result.get("summary"), str) or not result["summary"].strip():
        raise ValueError("missing_summary")
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(e, str) or not e.strip() for e in evidence):
        raise ValueError("missing_evidence")
    decisions = {"needs_evidence", "needs_escalation"} | (
        {"handle", "list-only", "ignore"} if stage == "scope" else {"prepared"}
    )
    if not isinstance(result.get("decision"), str) or result["decision"] not in decisions:
        raise ValueError("invalid_result_decision")
    if stage == "scope" and result.get("source_mode", "fixed") not in ("fixed", "moving"):
        raise ValueError("invalid_source_mode")


def _accept_scope(state, task, entry, result, cfg):
    if task["input_digest"] != scope_task_digest(entry, state):
        raise ValueError("stale_input")
    review = {
        "level": result["decision"],
        "summary": result["summary"],
        "evidence": result["evidence"],
        "policy_digest": policy_digest(cfg["responsibility"]),
    }
    entry["scope"] = make_entry(
        entry["issue"], cfg["responsibility"], review, source_mode=result.get("source_mode", "fixed")
    )
    entry["scope"]["accepted_scan_id"] = state["scan"].get("id")
    entry["phase"] = "classify"


def _accept_response(state, task, entry, result, root):
    if entry.get("classification_scan_id") != state["scan"].get("id") or task["input_digest"] != response_digest(
        entry, entry.get("classification", {})
    ):
        raise ValueError("stale_classification")
    if result["decision"] != "prepared":
        raise ValueError("invalid_response_decision")
    if task["payload"].get("quality_version") == 2:
        # Rendering happens in run() after this pure validation, before save.
        records = validate_response(state, task, result, root)
        entry["prepared_response"] = copy.deepcopy(result)
        entry["prepared_digest"] = task["input_digest"]
        entry["prepared_source_records"] = records
        entry["phase"] = "waiting"
        entry["waiting_on"] = "authorized_response_workflow"
        return
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("missing_response_artifacts")
    for artifact in artifacts:
        path = (root / artifact).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid_artifact_path")
    entry["phase"] = "waiting"
    entry["waiting_on"] = "authorized_response_workflow"
    entry["response_artifacts"] = artifacts


def accept_result(state, task_id, cfg, root, now):
    task = state["tasks"].get(task_id)
    if not task or task["status"] != "submitted":
        raise ValueError("task_not_submitted")
    result = task["result"]
    validate_result_header(result, task["stage"])
    entry = state["issues"][task["iid"]]
    if result["decision"] in {"needs_evidence", "needs_escalation"}:
        queue.hold(state, task_id, result["summary"], now)
        entry["phase"] = "waiting"
        entry["waiting_on"] = result["summary"]
        return
    if task["stage"] == "scope":
        _accept_scope(state, task, entry, result, cfg)
    else:
        _accept_response(state, task, entry, result, root)
    queue.accept(state, task_id, now)


def export_delivery(state, root, store, now):
    # Project only reviewed materials. Existing executors retain all publication
    # state; this report records preparation, never infers a remote outcome.
    relative = ensure_run(state, store, now)
    value = publish_materials(state, store, now, report_state(state, root, now))
    if not value["issues"] and not value["listed_issues"]:
        state.pop("delivery_report", None)
        return None
    path = store.artifact(relative / "_internal/run_state.json", value)
    result, seconds = read_command(
        "generate_summary_report.py",
        ["--state", path, "--output", str(store.root / relative / "summary.md"), "--strict", "--no-latest"],
        root,
    )
    state["delivery_report"] = result["report_path"]
    state["metrics"].append({"stage": "report", "seconds": seconds, "at": now})
    return result


def open_claim(args, state, root):
    if not args.claim_file:
        return None
    claim = read_json(args.claim_file)
    if not isinstance(claim, dict):
        raise ValueError("invalid_claim_object")
    task = state["tasks"].get(claim.get("task_id"))
    if not task:
        raise ValueError("stale_claim_file")
    claim_mismatch = claim.get("repository") != state["repository"] or task["attempt_id"] != claim.get("attempt_id")
    if claim_mismatch or task["input_digest"] != claim.get("input_digest"):
        raise ValueError("stale_claim_file")
    if task["status"] == "superseded":
        raise ValueError("stale_claim_file")
    if args.task and args.task != task["task_id"]:
        raise ValueError("claim_task_mismatch")
    args.task, args.attempt = task["task_id"], task["attempt_id"]
    if not args.result_file:
        args.result_file = claim["result_file"]
    return claim


def claim_bundle(task, state, root, store):
    relative = (
        ensure_run(state, store, time.time())
        / "_internal/work"
        / f"{task['iid']}-{task['stage']}-{uuid.uuid4().hex[:8]}"
    )
    result_file = str(store.root / relative / "result.json")
    claim = {k: task[k] for k in ("task_id", "attempt_id", "input_digest", "iid", "stage")}
    claim.update(repository=state["repository"], result_file=result_file)
    claim_file = store.artifact(relative / "claim.json", claim)
    payload = dict(task["payload"])
    if task["stage"] == "scope":
        payload["source_mode_help"] = {
            "fixed": (
                "根据 Issue/评论判定本仓文档、缺陷或需求属于责任范围，或证据来自固定提交；"
                "后续回复需要查源码也用 fixed。"
            ),
            "moving": "责任范围结论本身依赖当前分支源码内容；下次扫描必须重新核查。",
        }
    task_file = store.artifact(
        relative / "task.json",
        {
            **payload,
            "required_references": task_references(task),
            "allowed_values": response_enums()
            if task["stage"] == "response"
            else {
                "decision": ["handle", "list-only", "ignore", "needs_evidence", "needs_escalation"],
                "source_mode": ["fixed", "moving"],
            },
        },
    )
    template = (
        response_template(task["payload"].get("classification", {}))
        if task["stage"] == "response"
        else {"decision": "", "summary": "", "evidence": [], "source_mode": "fixed"}
    )
    store.artifact(relative / "result.json", template)
    return {
        "iid": task["iid"],
        "stage": task["stage"],
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "claim_file": claim_file,
        "task_file": task_file,
        "result_file": result_file,
        "required_references": task_references(task),
    }


def refresh_remote(state, cfg, config_path, root, *args, **kwargs):
    store, now, force = bind_extra(args, kwargs, ("store", "now", "force"), (MISSING, MISSING, False))
    full = state["needs_full_scan"] or now - state["scan"].get("full_completed_at", 0) >= 86400
    session_id = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
    new_session = (
        session_id != state["scan"].get("session_id")
        if session_id
        else (now - state["scan"].get("completed_at", 0) >= 300)
    )
    due = force or full or new_session or not state["scan"].get("complete")
    if due:
        args = [
            "--url",
            f"https://gitcode.com/{state['repository']}",
            "--config",
            str(config_path),
            "--defer-cursor",
            "--output",
            str(store.root / "data/pipeline-fetch.json"),
        ]
        if not full and state["scan"].get("cursor"):
            start = datetime.fromisoformat(state["scan"]["cursor"].replace("Z", "+00:00")) - timedelta(minutes=5)
            args += ["--updated-since", start.isoformat()]
        snapshot, seconds = read_command("fetch_issues.py", args, root)
        ingest(state, snapshot, cfg, time.time())
        if session_id:
            state["scan"]["session_id"] = session_id
        state["metrics"].append({"stage": "fetch", "seconds": seconds, "at": now})
        # Cursor and queue are in this SAME atomic checkpoint, before any
        # classification or agent dispatch. A crash cannot strand input.
        store.save(state)
    context = SimpleNamespace(state=state, config_path=config_path, root=root, store=store, now=now)
    for iid, entry in state["issues"].items():
        _refresh_issue(context, iid, entry)


def _refresh_issue(context, iid, entry):
    state, config_path, root, store, now = (
        context.state,
        context.config_path,
        context.root,
        context.store,
        context.now,
    )
    if (
        entry.get("scope")
        and entry["phase"] == "scope"
        and entry["scope"].get("activity_digest") != activity_digest(entry["issue"])
    ):
        # A delta scope task needs the new conversation, not only a count.
        entry["phase"] = "refresh"
    if entry["phase"] != "refresh":
        return
    try:
        snapshot, seconds = read_command(
            "fetch_issues.py",
            ["--issue", entry["issue"]["url"], "--config", str(config_path), "--defer-cursor"],
            root,
        )
        raw = snapshot.get("issues", [])
        if len(raw) != 1 or str(raw[0].get("iid") or raw[0].get("number")) != iid:
            raise ValueError("missing_refreshed_issue")
        entry.update(issue=normalize(raw[0], state["repository"]), phase="scope", fresh_scan_id=state["scan"].get("id"))
        entry.pop("classification_scan_id", None)
        entry.pop("refresh_error", None)
        state["metrics"].append({"stage": "refresh", "iid": iid, "seconds": seconds, "at": now})
        store.save(state)
    except ValueError as exc:
        entry["refresh_error"] = str(exc)


def _next_action(state, counts, pending, ready, operations):
    recovery = state.get("recovery", {}).get("requires_operation_audit")
    if recovery or operations:
        action = "verify_operations"
    elif pending:
        action = "review_results"
    elif ready:
        action = "claim_task"
    elif counts["classify"]:
        action = "classify_ready"
    elif counts["refresh"] or state["needs_full_scan"] or not state["scan"].get("complete"):
        action = "refresh_evidence"
    elif any(t["status"] == "running" for t in state["tasks"].values()):
        action = "await_task_results"
    elif counts["attention"] or counts["waiting"] or counts["scope"]:
        action = "resume_waiting_work"
    else:
        action = "complete"
    return action


def _status_text(state, summary, counts, ready, pending):
    action = summary["next_action"]
    summary["next_step"] = {
        "verify_operations": (
            "只读operation_files中的精确目标和证据，不遍历旧报告；结果未知先回查，不重发POST。"
            "已确认失败与替代成功操作分开记录。"
        ),
        "review_results": "逐项审核 submitted 任务的证据；accept --task TASK_ID。不等其他任务。",
        "claim_task": (
            "claim --worker NAME，读取task_file并编辑result_file；submit/accept使用返回的--claim-file，不手抄任务ID。"
        ),
        "classify_ready": "resume（在线模式），仅处理已就绪项，不等其余范围任务。",
        "refresh_evidence": "resume --refresh；未取得完整扫描前不报告全批完成。",
        "await_task_results": "只等待已确认运行中的agent；有单项结果即submit，不等全部。长任务可renew租约。",
        "resume_waiting_work": (
            "检查waiting_issues与tasks.waiting中的reason；按等待对象恢复，缺新证据不自动重派。"
            "准备好的回复沿原授权和发布流程执行。"
        ),
        "complete": "本轮没有待推进动作；observe不等于问题解决。回读delivery_report；报告失败用report恢复。",
    }[action]
    text = (
        f"# Issue 流水线检查点\n\n仓库：{state['repository']}\n\n"
        f"下一步：{action}\n\n扫描完整：{summary['scan_complete']}\n\n"
        "| 队列状态 | 数量 |\n| --- | ---: |\n"
        + "\n".join(f"| {name} | {count} |" for name, count in sorted(counts.items()))
        + f"\n\n{summary['next_step']}\n\n观察、转交与等待均不代表已解决。\n"
    )
    running_count = sum(t["status"] == "running" for t in state["tasks"].values())
    public_text = (
        f"# Issue 处理结果\n\n仓库：{state['repository']}\n\n"
        f"扫描记录：{len(state['issues'])}；忽略：{counts['ignored']}；观察：{counts['observe']}。\n\n"
        f"扫描完整：{summary['scan_complete']}。本轮尚无已审核的响应材料。\n\n"
        f"待调查或审核任务：{len(ready) + len(pending) + running_count}。"
        "观察与等待不代表问题解决。\n"
    )
    return text, public_text


def _populate_status_artifacts(state, store, report_dir, summary, operations):
    summary["delivery_report"] = state.get("delivery_report")
    summary["operation_files"] = []
    for key in operations:
        operation = {"operation_id": key, **state["operations"][key]}
        operation["operation_id"] = key
        summary["operation_files"].append(
            store.artifact(report_dir / "_internal/operations" / f"{digest(key)[:12]}.json", operation)
        )
    task_dir = report_dir / "_internal/tasks"
    for task in state["tasks"].values():
        if task["status"] in {"ready", "running", "submitted", "waiting"}:
            store.artifact(task_dir / f"{task['task_id']}.json", task)
    summary["task_directory"] = str(store.root / task_dir)
    summary["waiting_issues"] = [
        {
            "iid": iid,
            "phase": e["phase"],
            "waiting_on": e.get("waiting_on"),
            "artifacts": e.get("response_artifacts", []),
        }
        for iid, e in state["issues"].items()
        if e["phase"] in {"waiting", "attention", "refresh"}
    ]
    summary["prepared_items"] = [
        {
            "iid": iid,
            "files": sorted(name for name, path in e.get("response_file_map", {}).items() if Path(path).is_file()),
        }
        for iid, e in state["issues"].items()
        if e.get("prepared_response") and e["phase"] == "waiting"
    ]


def _compact_task_status(task_status):
    task_fields = (
        "task_id",
        "iid",
        "stage",
        "priority",
        "effort",
        "worker",
        "attempt_id",
        "lease_expires_at",
        "reason",
        "held",
    )
    compact_tasks = {}
    for key, tasks in task_status.items():
        if isinstance(tasks, list) and key not in {"accepted", "superseded"}:
            compact_tasks[key] = [{field: task.get(field) for field in task_fields} for task in tasks]
    compact_tasks["counts"] = task_status["counts"]
    compact_tasks["timings"] = task_status.get("timings", {})
    return compact_tasks


def output_status(state, store, now):
    report_dir = ensure_run(state, store, now)
    publish_materials(state, store, now, report_state(state, store.root.parent.parent, now))
    queue.expire(state, now)
    counts = Counter(e.get("phase", "refresh") for e in state["issues"].values())
    pending = [t for t in state["tasks"].values() if t["status"] == "submitted"]
    ready = [t for t in state["tasks"].values() if t["status"] == "ready"]
    operations = [k for k, v in state["operations"].items() if v.get("status") != "verified"]
    action = _next_action(state, counts, pending, ready, operations)
    task_status = queue.status(state, now)
    compact_tasks = _compact_task_status(task_status)
    summary = {
        "repository": state["repository"],
        "next_action": action,
        "scan_complete": bool(state["scan"].get("complete")) and not state["needs_full_scan"],
        "counts": dict(counts),
        "tasks": compact_tasks,
        "operations_to_verify": operations,
        "recovery": state.get("recovery"),
        "state_file": str(store.path),
    }
    if action == "review_results":
        references = set()
        for task in pending:
            references.update(task_references(task))
        summary["required_references"] = sorted(references)
    _populate_status_artifacts(state, store, report_dir, summary, operations)
    text, public_text = _status_text(state, summary, counts, ready, pending)
    summary["report_path"] = state.get("delivery_report") or store.text_artifact(report_dir / "summary.md", public_text)
    store.text_artifact(report_dir / "_internal/queue-summary.md", text)
    summary["report_directory"] = str(store.root / report_dir)
    summary["run_file"] = str(store.root / report_dir / "run.json")
    summary["status_file"] = str(store.root / report_dir / "_internal/status.json")
    store.artifact(report_dir / "_internal/status.json", summary)
    return summary


def inspection_output(evidence, relative, root):
    return {
        "evidence_file": relative,
        "revision": evidence["revision"],
        "records": evidence["records"],
        "reported_documents": evidence["reported_documents"],
        "detail_command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "inspect",
            "--repository-root",
            str(root),
            "--evidence-file",
            relative,
        ],
        "next_step": "翻页复用detail_command；新行范围用原claim、--revision所示固定提交和--lines重新inspect。",
    }


def command_argv(command, root, args):
    value = [sys.executable, str(Path(__file__).resolve()), command, "--repository-root", str(root)]
    if args.config:
        value += ["--config", str(Path(args.config).resolve())]
    if args.claim_file:
        value += ["--claim-file", str(Path(args.claim_file).resolve())]
    else:
        value += ["--task", args.task]
        if command == "submit":
            value += ["--attempt", args.attempt, "--result-file", str(Path(args.result_file).resolve())]
    if command == "submit" and args.claim_file and args.result_file:
        value += ["--result-file", str(Path(args.result_file).resolve())]
    if args.offline:
        value.append("--offline")
    return value


def reference_paths(*names):
    return [str(Path(__file__).resolve().parents[1] / "references" / name) for name in names]


def task_references(task):
    if task["stage"] == "scope":
        return reference_paths("responsibility-scope.md")
    classification = task.get("payload", {}).get("classification", {})
    names = [] if assignment_only(classification) else ["response-writing.md"]
    if assignment_action(classification):
        names.append("automation.md")
    return reference_paths(*names)


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def page_limit(value):
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return number


def line_range(value):
    if not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("use START:END, inclusive")
    start, end = map(int, value.split(":"))
    if end < start or end - start >= 160:
        raise argparse.ArgumentTypeError("range must contain 1..160 lines")
    return start, end


def _list_sections(view):
    collections = {key: value for key, value in view.items() if isinstance(value, list)}
    collections.update(
        {"tasks." + key: value for key, value in view.get("tasks", {}).items() if isinstance(value, list)}
    )
    for record in view.get("records", []):
        collections.update(
            {"records." + record["path"] + "." + key: value for key, value in record.items() if isinstance(value, list)}
        )
    return collections


def _section_view(view, args):
    if not args.section:
        return None
    collections = _list_sections(view)
    if args.section not in collections:
        raise ValueError("unknown_output_section")
    items = collections[args.section]
    if args.iid and args.section.startswith(("waiting_issues", "prepared_items", "tasks.")):
        items = [item for item in items if str(item.get("iid")) == str(args.iid)]
    end = args.offset + args.limit
    return {
        "protocol_version": 2,
        "next_action": view.get("next_action"),
        "required_references": view["required_references"],
        "status_file": view.get("status_file"),
        "evidence_file": view.get("evidence_file"),
        "detail_command": view.get("detail_command"),
        "section": args.section,
        "items": items[slice(args.offset, end)],
        "total": len(items),
        "offset": args.offset,
        "next_offset": end if end < len(items) else None,
    }


def _compact_issue_detail(view, args):
    if isinstance(view.get("issue_detail"), dict):
        entry = view["issue_detail"]
        view["issue_detail"] = {
            key: entry.get(key) for key in ("phase", "waiting_on", "response_file_map", "response_revision")
        }
        view["issue_detail"]["iid"] = str(args.iid)
        view["issue_detail"]["scope"] = (entry.get("scope") or {}).get("review")
        view["issue_detail"]["classification"] = {
            key: (entry.get("classification") or {}).get(key)
            for key in ("bucket", "category", "responsibility", "reason")
        }


def _page_items(items, name, args, pages, nested=False):
    selected = items
    if args.iid and name in {"waiting_issues", "tasks.ready", "tasks.running", "tasks.submitted", "tasks.waiting"}:
        selected = [item for item in items if str(item.get("iid")) == str(args.iid)]
    offset = 0 if nested else args.offset
    end = offset + args.limit
    pages[name] = {
        "total": len(selected),
        "offset": offset,
        "returned": len(selected[slice(offset, end)]),
        "next_offset": end if end < len(selected) else None,
    }
    return selected[slice(offset, end)]


def _page_records(view, args, pages):
    view["records"] = _page_items(view["records"], "records", args, pages)
    for record in view["records"]:
        name = record["path"]
        excerpt = record.get("excerpt", "")
        # Explicit range requests retain the selected excerpt; ordinary inspection
        # gives a preview with an honest flag and the full evidence_file path.
        if not args.lines and len(excerpt) > 2400:
            record["excerpt"] = excerpt[:2400]
            record["excerpt_truncated"] = True
        for key in ("links", "history", "files"):
            if isinstance(record.get(key), list):
                record[key] = _page_items(record[key], "records." + name + "." + key, args, pages, nested=True)
    view["reported_documents"] = _page_items(view.get("reported_documents", []), "reported_documents", args, pages)


def _paged_view(view, args):
    pages = {}

    tasks = view.get("tasks", {})
    if args.command in {"submit", "accept"}:
        view["tasks"] = {"counts": tasks.get("counts", {})}
        view.pop("waiting_issues", None)
        view.pop("prepared_items", None)
        # Operation details are available through status; count preserves the blocker.
        view["operations_pending"] = len(view.pop("operations_to_verify", []))
        view.pop("operation_files", None)
    else:
        for name, items in list(tasks.items()):
            if isinstance(items, list):
                tasks[name] = _page_items(items, "tasks." + name, args, pages)
        for name in ("waiting_issues", "prepared_items", "operations_to_verify", "operation_files"):
            if isinstance(view.get(name), list):
                view[name] = _page_items(view[name], name, args, pages)
        for item in view.get("waiting_issues", []):
            item.pop("artifacts", None)  # Full paths remain in status_file.
    if "records" in view:
        _page_records(view, args, pages)
    if pages:
        view["pages"] = pages
        view["detail_hint"] = (
            "完整数据见 status_file/evidence_file；--section NAME --offset N --limit N 翻页，--details 展开。"
        )
    return view


def protocol_view(result, args):
    """Project stdout only; durable state and Python run() retain complete data."""
    view = copy.deepcopy(result)
    view["protocol_version"] = 2
    refs = {
        "configure_repository": ["configuration-setup.md"],
        "fix_configuration": ["configuration-setup.md"],
        "verify_operations": ["authorization-contract.md", "runtime-state.md"],
        "provide_token": ["runtime-capability-checks.md"],
    }
    view.setdefault("required_references", reference_paths(*refs.get(view.get("next_action"), [])))
    if getattr(args, "identities", False):
        view["required_references"] = reference_paths("operator-owner-candidates.md")
    if args.iid:
        for name, items in view.get("tasks", {}).items():
            if isinstance(items, list):
                view["tasks"][name] = [item for item in items if str(item.get("iid")) == str(args.iid)]
        for name in ("waiting_issues", "prepared_items"):
            if name in view:
                view[name] = [item for item in view[name] if str(item.get("iid")) == str(args.iid)]
    if args.section:
        return _section_view(view, args)
    if args.details:
        return view
    _compact_issue_detail(view, args)
    return _paged_view(view, args)


def _parser_options(p):
    p.add_argument(
        "command",
        choices=[
            "resume",
            "status",
            "claim",
            "submit",
            "accept",
            "requeue",
            "renew",
            "hold",
            "inspect",
            "check",
            "revise",
            "report",
            "record-operation",
            "verify-operation",
            "audit-recovery",
        ],
    )
    p.add_argument("--repository-root", default=".")
    p.add_argument("--config")
    p.add_argument("--input", help="Ingest a captured fetch JSON; no live fetch for this invocation")
    p.add_argument("--offline", action="store_true", help="Never access remote APIs; preserve unknowns")
    p.add_argument(
        "--new-run",
        action="store_true",
        help="resume: start a new user-requested report round; ordinary resume keeps the existing round",
    )
    p.add_argument("--refresh", action="store_true", help="Refresh remote changes even within the 5-minute scan window")
    p.add_argument("--worker", default="coordinator")
    p.add_argument("--task")
    p.add_argument("--attempt")
    p.add_argument("--result-file")
    p.add_argument("--claim-file", help="Use generated claim.json; task/attempt/result path are loaded automatically")


def _parser_detail_options(p):
    p.add_argument("--path", action="append", help="Repository-relative source file/directory to inspect; repeatable")
    p.add_argument("--revision", default="HEAD")
    p.add_argument("--evidence-file", help="inspect: read a sealed existing snapshot without collecting again")
    p.add_argument(
        "--details", action="store_true", help="Return full legacy output; complete status is always saved locally"
    )
    p.add_argument("--section", help="Page one list named in pages, e.g. tasks.submitted or records.PATH.links")
    p.add_argument("--offset", type=nonnegative_int, default=0, help="Offset for each returned list")
    p.add_argument("--limit", type=page_limit, default=5, help="Items per list, 1..100 (default 5)")
    p.add_argument("--lines", type=line_range, help="inspect one file at START:END, inclusive, at most 160 lines")
    p.add_argument(
        "--identities", action="store_true", help="Read GitCode commit author identities for owner-candidate evidence"
    )
    p.add_argument("--reason")
    p.add_argument("--operation-id")
    p.add_argument("--iid")
    p.add_argument("--lease-seconds", type=int, default=300)
    p.add_argument("--max-running", type=int, default=3)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    _parser_options(p)
    _parser_detail_options(p)
    return p


def _command_claim(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    root = ctx.root
    store = ctx.store
    if state.get("recovery", {}).get("requires_operation_audit"):
        raise ValueError("recovery_audit_required")
    value = queue.claim(state, args.worker, now, lease_seconds=args.lease_seconds, max_running=args.max_running)
    if value:
        value = claim_bundle(value, state, root, store)
    return value


def _command_inspect(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    root = ctx.root
    repo = ctx.repo
    cfg = ctx.cfg
    store = ctx.store
    claim = ctx.claim
    queue.expire(state, now)
    if not claim or state["tasks"][args.task]["status"] != "running":
        raise ValueError("running_claim_file_required")
    issue = state["issues"][state["tasks"][args.task]["iid"]]["issue"]
    evidence = inspect_sources(
        root,
        repo,
        args.path,
        args.revision,
        reported_text=issue.get("description", ""),
        lines=getattr(args, "lines", None),
    )
    if args.identities and not args.offline:
        verify_identities(evidence, cfg)
    folder = Path(args.claim_file).resolve().parent.relative_to(store.root)
    name = store.artifact(folder / f"inspection-{uuid.uuid4().hex[:8]}.json", evidence)
    relative = str(Path(name).relative_to(root))
    state.setdefault("source_evidence", {})[relative] = {
        "task_id": args.task,
        "attempt_id": args.attempt,
        "digest": content_hash(evidence),
    }
    result = read_json(args.result_file)
    result.setdefault("inspection_files", []).append(relative)
    store.artifact(Path(args.result_file).resolve().relative_to(store.root), result)
    value = inspection_output(evidence, relative, root)
    return value


def _command_check(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    root = ctx.root
    claim = ctx.claim
    queue.expire(state, now)
    task = state["tasks"].get(args.task)
    response_claim = claim and task and task["stage"] == "response" and task["status"] == "running"
    if not response_claim:
        raise ValueError("response_claim_required")
    result = read_json(args.result_file)
    validate_result_header(result, task["stage"])
    validate_response(state, task, result, root)
    value = {
        "valid": True,
        "next_step": "核对回复中的事实与inspection一致后submit，再accept；结构校验不是语义审核。",
    }
    return value


def _command_submit(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    root = ctx.root
    result = read_json(args.result_file)
    if not isinstance(result, dict):
        raise ValueError("invalid_result_object")
    task = state["tasks"].get(args.task)
    if task:
        validate_result_header(result, task["stage"])
    prepared_response = task and task["stage"] == "response" and task["payload"].get("quality_version") == 2
    if prepared_response and result.get("decision") == "prepared":
        validate_response(state, task, result, root)
    value = queue.submit(state, args.task, args.attempt, result, now)
    value = {
        "iid": value["iid"],
        "stage": value["stage"],
        "task_id": value["task_id"],
        "status": value["status"],
    }
    return value


def _command_accept(ctx):
    args = ctx.args
    state = ctx.state
    cfg = ctx.cfg
    root = ctx.root
    now = ctx.now
    store = ctx.store
    accept_result(state, args.task, cfg, root, now)
    task = state["tasks"][args.task]
    entry = state["issues"][task["iid"]]
    value = {k: task[k] for k in ("iid", "stage", "task_id", "status")}
    if task["stage"] == "response" and entry.get("prepared_response"):
        relative = (
            ensure_run(state, store, now)
            / "_internal/materials"
            / f"issue-{task['iid']}"
            / f"revision-{entry.get('response_revision', 0)}"
        )
        files, related = materialize(entry["prepared_response"], entry["prepared_source_records"], store, relative)
        entry.update(response_artifacts=list(files.values()), response_file_map=files, related_code=related)
    return value


def _command_revise(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    entry = state["issues"].get(args.iid or state["tasks"].get(args.task, {}).get("iid"))
    if not entry or not args.reason or not entry.get("prepared_response"):
        raise ValueError("prepared_issue_and_revision_reason_required")
    invalidate_response(state, entry, args.reason, now)
    entry["phase"] = "attention"
    state.pop("delivery_report", None)
    apply_classification(state, {"issues": [entry["classification"]]}, [entry], entry.get("classification_path"), now)
    return None


def _command_report(ctx):
    state = ctx.state
    root = ctx.root
    store = ctx.store
    now = ctx.now
    value = export_delivery(state, root, store, now)
    return value


def _command_requeue(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    if not args.reason:
        raise ValueError("requeue_reason_required")
    value = queue.requeue(state, args.task, args.reason, now)
    return value


def _command_renew(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    value = queue.renew(state, args.task, args.attempt, now, args.lease_seconds)
    return value


def _command_hold(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    if not args.reason:
        raise ValueError("waiting_reason_required")
    value = queue.hold(state, args.task, args.reason, now)
    return value


def _command_record_operation(ctx):
    args = ctx.args
    state = ctx.state
    now = ctx.now
    if not args.operation_id or args.iid not in state["issues"]:
        raise ValueError("operation_identity_required")
    state["operations"].setdefault(args.operation_id, {"status": "unknown", "iid": args.iid, "recorded_at": now})
    return None


def _command_verify_operation(ctx):
    args = ctx.args
    state = ctx.state
    receipt = read_json(args.result_file)
    if receipt.get("verified") is not True or not receipt.get("evidence"):
        raise ValueError("verification_evidence_required")
    if args.command == "audit-recovery":
        state.setdefault("recovery", {}).update(requires_operation_audit=False, receipt=receipt)
    else:
        op = state["operations"].get(args.operation_id)
        if not op or receipt.get("operation_id") != args.operation_id:
            raise ValueError("operation_mismatch")
        op.update(status="verified", receipt=receipt)
    return None


COMMAND_HANDLERS = {
    "claim": _command_claim,
    "inspect": _command_inspect,
    "check": _command_check,
    "submit": _command_submit,
    "accept": _command_accept,
    "revise": _command_revise,
    "report": _command_report,
    "requeue": _command_requeue,
    "renew": _command_renew,
    "hold": _command_hold,
    "record-operation": _command_record_operation,
    "verify-operation": _command_verify_operation,
    "audit-recovery": _command_verify_operation,
}


def _existing_evidence(ctx):
    args, state, root = ctx.args, ctx.state, ctx.root
    if args.command == "inspect" and getattr(args, "evidence_file", None):
        path = Path(args.evidence_file).resolve()
        relative = str(path.relative_to(root))
        seal = state.get("source_evidence", {}).get(relative)
        if not seal:
            raise ValueError("unknown_evidence_file")
        evidence = read_json(path)
        if content_hash(evidence) != seal["digest"]:
            raise ValueError("modified_evidence_file")
        return inspection_output(evidence, relative, root)
    return None


def _prepare_run(ctx):
    args, state, cfg = ctx.args, ctx.state, ctx.cfg
    root, store, now = ctx.root, ctx.store, ctx.now
    ensure_run(state, store, now, new=getattr(args, "new_run", False))
    store.save(state)
    migrate(state, store.root, cfg)
    synchronize(state, cfg, now)
    # Persist policy invalidation and recovery before any potentially
    # failing network/child process or acceptance step.
    store.save(state)
    claim = open_claim(args, state, root)
    ctx.claim = claim
    if claim and args.command in {"inspect", "check"}:
        task = state["tasks"][args.task]
        entry = state["issues"][task["iid"]]
        if task["stage"] == "response" and (
            entry.get("classification_scan_id") != state["scan"].get("id")
            or task["input_digest"] != response_digest(entry, entry.get("classification", {}))
        ):
            raise ValueError("stale_classification")


def _command_resume(ctx):
    args, state, cfg = ctx.args, ctx.state, ctx.cfg
    config_path, root, store, now = ctx.config_path, ctx.root, ctx.store, ctx.now
    if args.input:
        ingest(state, read_json(args.input), cfg, now)
        store.save(state)
    elif not args.offline:
        if not os.environ.get("GITCODE_TOKEN"):
            return {"next_action": "provide_token", "state_file": str(store.path)}
        refresh_remote(state, cfg, config_path, root, store, now, args.refresh or getattr(args, "new_run", False))
    synchronize(state, cfg, now)
    return None


def _finish_run(ctx, value):
    args, state, cfg = ctx.args, ctx.state, ctx.cfg
    config_path, root, store, now = ctx.config_path, ctx.root, ctx.store, ctx.now
    if args.command == "accept":
        synchronize(state, cfg, now)
    # A successful acceptance/submission must survive a later classifier
    # or network failure. Model results and external receipts are durable
    # BEFORE the next dependent stage starts.
    store.save(state)
    if args.command in {"resume", "accept"} and not args.offline:
        if os.environ.get("GITCODE_TOKEN"):
            classify_ready(state, cfg, config_path, root, store, now)
    store.save(state)
    if args.command in {"resume", "accept"}:
        export_delivery(state, root, store, now)
    result = output_status(state, store, now)
    if args.command == "status" and args.iid:
        result["issue_detail"] = state["issues"].get(str(args.iid))
    store.save(state)
    if value is not None:
        result["task"] = value
    if args.command == "claim" and value:
        return {
            "next_action": "work_task",
            "task": value,
            "required_references": value.get("required_references"),
            "next_step": (
                "按task_file及该阶段必要文档填写result_file，然后submit --claim-file CLAIM，"
                "再accept --claim-file CLAIM。"
            ),
        }
    if args.command == "check":
        value["next_command"] = command_argv("submit", root, args)
    if args.command in {"inspect", "check"}:
        return value
    if args.command == "submit":
        result["next_command"] = command_argv("accept", root, args)
    if args.command in {"submit", "accept"}:
        # The complete queue remains in status.json; progress acknowledgments
        # only need counts and actionable review/waiting entries.
        result["tasks"] = {k: v for k, v in result["tasks"].items() if k in {"counts", "submitted", "waiting"}}
    return result


def run(args):
    root = Path(args.repository_root).resolve()
    os.chdir(root)
    config_path = Path(args.config).resolve() if args.config else root / RUNTIME / "config/classify_config.yaml"
    if not config_path.exists():
        return {
            "next_action": "configure_repository",
            "instruction": "按 configuration-setup.md 完成或复用项目配置后再次 resume。",
        }
    cfg = load_handler_config(config_path)
    repo = cfg.get("repo", "")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
        raise ValueError("configure_repository_first")
    now = time.time()
    with PipelineStore(root / RUNTIME, repo) as store:
        state = store.load()
        ctx = SimpleNamespace(
            args=args,
            state=state,
            cfg=cfg,
            config_path=config_path,
            root=root,
            repo=repo,
            now=now,
            store=store,
            claim=None,
        )
        existing = _existing_evidence(ctx)
        if existing is not None:
            return existing
        _prepare_run(ctx)
        if args.command == "resume":
            early = _command_resume(ctx)
            if early is not None:
                return early
            value = None
        else:
            handler = COMMAND_HANDLERS.get(args.command)
            value = handler(ctx) if handler else None
        return _finish_run(ctx, value)


def _validate_cli_args(command_parser, args):
    if args.new_run and args.command != "resume":
        command_parser.error("--new-run is only supported by resume")
    if args.section and args.command not in {"status", "inspect"}:
        command_parser.error("--section is only supported by status and inspect")
    conflicting_evidence_options = args.command != "inspect" or args.path or args.lines or args.identities
    if args.evidence_file and conflicting_evidence_options:
        command_parser.error(
            "--evidence-file reads an existing snapshot; use inspect without --path/--lines/--identities"
        )
    if args.lines and args.command != "inspect":
        command_parser.error("--lines is only supported by inspect")


def _pipeline_error(exc, args):
    # No raw exception from network paths or Issue text in error protocol.
    code = str(exc) if re.fullmatch(r"[A-Za-z0-9_.-]+", str(exc)) else type(exc).__name__
    action = {
        "pipeline_busy": "retry_when_current_command_finishes",
        "stale_input": "reload_task",
        "stale_attempt": "reload_task",
        "stale_lease": "reload_task",
        "stale_classification": "refresh_evidence",
        "missing_evidence": "requeue_corrected_result",
        "missing_summary": "requeue_corrected_result",
        "invalid_result_object": "requeue_corrected_result",
        "missing_response_artifacts": "requeue_corrected_result",
        "invalid_artifact_path": "requeue_corrected_result",
        "invalid_response_decision": "requeue_corrected_result",
        "recovery_audit_required": "verify_operations",
    }.get(code, "inspect_error_and_resume")
    if args.command in {"check", "submit"} and code in {
        "missing_summary",
        "missing_evidence",
        "invalid_result_object",
        "invalid_result_decision",
        "invalid_source_mode",
    }:
        action = "correct_result_file"
    result = {"error": code, "next_action": action, "state_preserved": True}
    if action == "correct_result_file":
        result.update(
            result_file=args.result_file, next_step="修正 result_file 中的字段后重试原命令；当前任务尚未提交。"
        )
    if action == "requeue_corrected_result":
        result["next_step"] = (
            "requeue --task TASK_ID --reason 审核修正；claim --worker NAME 领取新attempt；"
            "使用新attempt提交修正结果，再accept。不得覆盖旧提交。"
        )
    return result


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    _validate_cli_args(command_parser, args)
    try:
        result = run(args)
        write_json(protocol_view(result, args), ensure_ascii=False, separators=(",", ":"))
        return 0
    except ConfigError as exc:
        write_json(
            {
                "error": "invalid_configuration",
                "errors": exc.errors,
                "next_action": "fix_configuration",
                "state_preserved": True,
                "required_references": reference_paths("configuration-setup.md"),
                "next_step": "按 errors 修改配置后重新执行原命令；无需人工逐项复核，脚本会自动重验。",
            },
            ensure_ascii=False,
        )
        return 2
    except MaterialError as exc:
        write_json(
            {
                "error": str(exc),
                "next_action": "complete_response_materials",
                "errors": exc.errors,
                "state_preserved": True,
                "next_step": (
                    "按errors补证据或修正result_file，再check/submit；证据不足使用needs_evidence，不绕过校验。"
                ),
            },
            ensure_ascii=False,
        )
        return 2
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        write_json(_pipeline_error(exc, args), ensure_ascii=False)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
