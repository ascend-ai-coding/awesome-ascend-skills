#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Post a classified first response or new follow-up through the toolkit.

Result reports and explicitly requested additional comments use the general
toolkit executor. This gate consumes classifier output, not reply self-scores;
it does not replace authorization or refresh stale evidence automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli_output import write_stderr  # noqa: E402
from handler_config import get_automation_policy, load_handler_config  # noqa: E402

TOOLKIT = Path(__file__).resolve().parents[2] / "gitcode-toolkit" / "scripts"
sys.path.insert(0, str(TOOLKIT))
import post_issue_comment as executor  # noqa: E402
from gitcode_client import api_get, make_session, parse_issue_url, resolve_api_base, resolve_token  # noqa: E402
from response_artifacts import save_response_artifacts  # noqa: E402

REPLY_CATEGORIES = frozenset({
    "needs_first_look", "needs_only_assign_cmd", "our_team_needs_work",
    "our_team_only_assign_cmd", "needs_first_response_with_pr",
    "reporter_followup", "reopened_followup", "assignee_followup",
})
FOLLOWUP_CATEGORIES = frozenset({
    "reporter_followup", "reopened_followup", "assignee_followup",
})


def issue_identity(url):
    """Keep the full repository path and origin, unlike a number-only match."""
    if not isinstance(url, str):
        raise ValueError("classification must contain an Issue URL")
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    invalid_origin = parsed.scheme != "https" or not parsed.netloc
    has_credentials = bool(parsed.username or parsed.password)
    has_extras = bool(parsed.query or parsed.fragment)
    invalid_path = (
        len(parts) < 4
        or parts[-2] != "issues"
        or not parts[-1].isdigit()
        or any(not part for part in parts)
    )
    invalid_url = any((invalid_origin, has_credentials, has_extras, invalid_path))
    if invalid_url:
        raise ValueError("expected an exact HTTPS Issue URL")
    return parsed.netloc.lower(), tuple(parts[:-2]), parts[-1]


def _matching_item(classification, target):
    if not isinstance(classification, dict) or classification.get("mode") not in {"single", "batch"}:
        raise ValueError("classification must be a single or batch classifier result")
    issues = classification.get("issues")
    if not isinstance(issues, list):
        raise ValueError("classification issues are missing")
    matches = []
    for item in issues:
        if isinstance(item, dict) and issue_identity(item.get("url")) == target:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError("classification must contain exactly one matching repository and Issue")
    return matches[0]


def _scan_affected_issues(name, scan):
    if name == "linkage":
        return list(map(str, scan.get("incomplete_issue_numbers", [])))
    if name != "comment":
        return []
    errors = scan.get("errors", [])
    if any(not isinstance(error, dict) or not error.get("issue_number") for error in errors):
        raise ValueError("comment scan errors lack Issue identity")
    return [str(error["issue_number"]) for error in errors]


def _validate_scans(classification, target):
    association = classification.get("association_scan") or {}
    scans = (
        ("PR", association.get("pr_fetch")),
        ("linkage", association.get("linkage_fallback")),
        ("comment", classification.get("comment_fetch")),
    )
    for name, scan in scans:
        if not isinstance(scan, dict):
            raise ValueError(f"{name} scan must be complete before publishing")
        # Numeric skip counts differ from the scope gate's string marker.
        if isinstance(scan.get("skipped"), str) and scan["skipped"]:
            raise ValueError(f"{name} scan was skipped; rerun classification after scope review")
        affected = _scan_affected_issues(name, scan)
        if target[-1] in affected:
            raise ValueError(f"Issue {name} scan is incomplete")
        # A known failure on another Issue must not stall this one.
        if scan.get("complete") is not True and (scan.get("complete") is not False or not affected):
            raise ValueError(f"{name} scan must be complete before publishing")


def validate_classification(classification, issue_url):
    target = issue_identity(issue_url)
    item = _matching_item(classification, target)
    if str(item.get("number")) != target[-1]:
        raise ValueError("classification Issue number conflicts with its URL")
    if item.get("responsibility") != "handle":
        raise ValueError("responsibility review must resolve to handle; return to issue-intake, "
                         "record responsibility_review and rerun classify_issues.py")
    if item.get("bucket") != "need_attention" or item.get("category") not in REPLY_CATEGORIES:
        raise ValueError("classification does not require a new first response or follow-up")
    _validate_scans(classification, target)
    return item


def _check_config_target(config, issue_url):
    if issue_identity(issue_url)[-1] in {str(i) for i in config.get("ignored_issue_ids", [])}:
        raise ValueError("Issue is explicitly ignored by the current configuration")
    configured_repo = str(config.get("repo") or "").strip().casefold()
    if not configured_repo:
        return
    target = issue_identity(issue_url)
    target_repo = "/".join(target[1]).casefold()
    if configured_repo != target_repo:
        raise ValueError(
            f"配置 repo={config.get('repo')} 与目标 Issue 仓库不一致，禁止发送"
        )


def _check_apply_authorization(item, config, issue_url, reply_approved):
    _check_config_target(config, issue_url)
    policy = get_automation_policy(config)
    needs_approval = (
        not policy["auto_response"]
        or item.get("category") in FOLLOWUP_CATEGORIES
    )
    if needs_approval and not reply_approved:
        raise ValueError(
            "--apply 需要显式 --reply-approved：auto-response=false 时必须批准，"
            "follow-up 回复始终需要批准"
        )


def _check_live_open(issue_url, session=None):
    owner, repo, number = parse_issue_url(issue_url)
    endpoint = f"{resolve_api_base(repo_url=issue_url)}/repos/{owner}/{repo}/issues/{number}"
    session = session or make_session()
    try:
        payload = api_get(session, endpoint, resolve_token())
    except Exception as exc:
        raise RuntimeError("live Issue state could not be verified") from exc
    if not isinstance(payload, dict) or str(payload.get("state") or "").strip().casefold() not in {"open", "opened"}:
        raise ValueError("Issue must still be open before publishing a triage reply")
    return session


def execute(
    classification, issue_url, body, result_file, *, apply=False, session=None,
    config=None, reply_approved=False, analysis=None,
):
    item = validate_classification(classification, issue_url)
    save_response_artifacts(result_file, issue_url, analysis or item.get("reason", ""), reply=body)
    if apply:
        _check_apply_authorization(
            item,
            config if config is not None else load_handler_config(),
            issue_url,
            reply_approved,
        )
        session = _check_live_open(issue_url, session)
    return executor.execute(issue_url, body, result_file, apply=apply, session=session)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--issue-url", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--analysis-file", help="Per-Issue analysis to persist with the response draft")
    parser.add_argument("--config", default=None,
                        help="Handler config used for auto-response policy")
    parser.add_argument("--reply-approved", action="store_true",
                        help="Explicitly approve a reply when policy requires it")
    args = parser.parse_args(argv)
    try:
        classification = json.loads(Path(args.classification).read_text(encoding="utf-8"))
        item = validate_classification(classification, args.issue_url)
        analysis = (
            Path(args.analysis_file).read_text(encoding="utf-8")
            if args.analysis_file else item.get("reason", "")
        )
        save_response_artifacts(
            args.result_file, args.issue_url, analysis,
            reply=Path(args.body_file).read_text(encoding="utf-8"),
        )
        config = load_handler_config(args.config) if args.apply else None
        if args.apply:
            _check_apply_authorization(item, config, args.issue_url, args.reply_approved)
            try:
                _check_live_open(args.issue_url)
            except ValueError:
                raise
            except Exception:
                write_stderr("Error: live Issue state could not be verified")
                return 2
    except (ValueError, OSError, TypeError, AttributeError) as exc:
        write_stderr(f"Error: triage publication blocked: {exc}")
        return 2
    # Preserve the toolkit CLI's credential-safe errors and readback semantics.
    forwarded = ["--issue", args.issue_url, "--body-file", args.body_file,
                 "--result-file", args.result_file]
    if args.apply:
        forwarded.append("--apply")
    return executor.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
