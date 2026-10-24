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
"""Preview or idempotently associate one verified covering PR with an Issue.

The caller must provide a reviewed evidence file proving that the PR covers the
Issue on the intended base and that no unresolved additional risk was found.
The executor revalidates the live Issue, PR author/state/base/head and existing
association, writes at most one POST, and confirms the relationship from both
the PR and Issue directions before producing evidence usable by assign_issue.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
_TOOLKIT = SCRIPT_DIR.parent.parent / "gitcode-toolkit" / "scripts"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from gitcode_client import (  # noqa: E402
    api_get,
    api_post_json,
    make_session,
    parse_issue_url,
    resolve_api_base,
    resolve_token,
    safe_error_text,
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from cli_output import write_stderr, write_stdout  # noqa: E402
from handler_config import get_automation_policy, load_handler_config  # noqa: E402
from mutation_result import ResultLock as _ResultLock, save_json as _save_json  # noqa: E402
from runtime_paths import rate_limit_path  # noqa: E402


GOOD_RESPONSE_STATUSES = {"verified", "reused"}
GOOD_PR_STATES = {"open", "opened", "merged"}


class AssociationError(ValueError):
    """Raised when association policy or evidence is invalid."""


def _read_json(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssociationError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AssociationError(f"{label} is not valid JSON: {path}") from exc


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _login(value: Any) -> str:
    if isinstance(value, dict):
        return _as_text(value.get("login") or value.get("username"))
    return _as_text(value)


def _issue_target(issue_url: str) -> dict[str, str]:
    try:
        owner, repo, number = parse_issue_url(issue_url)
    except Exception as exc:
        raise AssociationError(f"invalid Issue URL: {issue_url}") from exc
    if not str(number).isdigit():
        raise AssociationError("Issue URL must contain a numeric IID")
    return {
        "owner": owner,
        "repo": repo,
        "issue_number": str(number),
        "url": issue_url,
    }


def _pr_target(pr_url: str) -> dict[str, str]:
    parsed = urlparse(pr_url)
    if parsed.scheme != "https" or parsed.netloc.casefold() not in {"gitcode.com", "www.gitcode.com"}:
        raise AssociationError("PR URL must be an HTTPS GitCode URL")
    parts = [part for part in parsed.path.split("/") if part]
    markers = {"pull", "pulls", "merge_requests"}
    try:
        marker = next(index for index, part in enumerate(parts) if part.casefold() in markers)
        owner, repo, number = parts[marker - 2], parts[marker - 1], parts[marker + 1]
    except (StopIteration, IndexError) as exc:
        raise AssociationError("PR URL has no valid PR identity") from exc
    if marker < 2 or not number.isdigit():
        raise AssociationError("PR URL has no valid PR identity")
    return {
        "owner": owner,
        "repo": repo,
        "pr_number": number,
        "url": pr_url,
    }


def _same_repo(issue: dict[str, str], pr: dict[str, str]) -> bool:
    return (
        issue["owner"].casefold(), issue["repo"].casefold()
    ) == (
        pr["owner"].casefold(), pr["repo"].casefold()
    )


def _identity_from_issue(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str):
        return None
    try:
        owner, repo, number = parse_issue_url(value)
    except Exception:
        return None
    return owner.casefold(), repo.casefold(), str(number)


def _response_target(response: dict[str, Any], target: dict[str, str]) -> None:
    if response.get("status") not in GOOD_RESPONSE_STATUSES:
        raise AssociationError("response result must be verified or reused")
    actual = response.get("target")
    identity = None
    if isinstance(actual, str):
        identity = _identity_from_issue(actual)
    elif isinstance(actual, dict):
        identity = _identity_from_issue(actual.get("url"))
        if identity is None and all(actual.get(key) is not None for key in ("owner", "repo", "issue_number")):
            identity = (
                str(actual["owner"]).casefold(),
                str(actual["repo"]).casefold(),
                str(actual["issue_number"]),
            )
    if identity is None:
        identity = _identity_from_issue(response.get("issue_url"))
    expected = (target["owner"].casefold(), target["repo"].casefold(), target["issue_number"])
    if identity != expected:
        raise AssociationError("response result target does not match the Issue URL")


def _check_config_target(config: dict[str, Any], target: dict[str, str]) -> None:
    if target["issue_number"] in {str(i) for i in config.get("ignored_issue_ids", [])}:
        raise AssociationError("Issue is explicitly ignored by the current configuration")
    configured = config.get("repo")
    if configured in (None, ""):
        return
    value = str(configured).strip().removesuffix(".git").strip("/").casefold()
    expected = f"{target['owner']}/{target['repo']}".casefold()
    if value != expected:
        raise AssociationError("config repo does not match the Issue URL")


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _duplicate_evidence(document, expected_issue):
    duplicate_issue_url = _as_text(document.get("duplicate_issue_url"))
    if not duplicate_issue_url:
        return None, None
    duplicate_identity = _identity_from_issue(duplicate_issue_url)
    if duplicate_identity is None:
        raise AssociationError("duplicate Issue URL is invalid")
    if duplicate_identity[:2] != expected_issue[:2] or duplicate_identity[2] == expected_issue[2]:
        raise AssociationError("duplicate Issue must be a different Issue in the same repository")
    if document.get("duplicate_relationship_verified") is not True:
        raise AssociationError("duplicate Issue relationship is not verified")
    return duplicate_issue_url, duplicate_identity[2]


def _validate_evidence(document: Any, issue: dict[str, str], pr: dict[str, str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise AssociationError("association evidence must be a JSON object")
    issue_identity = _identity_from_issue(document.get("target_issue_url"))
    expected_issue = (issue["owner"].casefold(), issue["repo"].casefold(), issue["issue_number"])
    if issue_identity != expected_issue:
        raise AssociationError("association evidence target Issue does not match")
    evidence_pr = _pr_target(document.get("pr_url", ""))
    if (
        evidence_pr["owner"].casefold(), evidence_pr["repo"].casefold(), evidence_pr["pr_number"]
    ) != (
        pr["owner"].casefold(), pr["repo"].casefold(), pr["pr_number"]
    ):
        raise AssociationError("association evidence PR does not match")
    required_true = (
        "coverage_verified",
        "base_compatibility_verified",
        "additional_risk_reviewed",
        "safe_to_associate",
    )
    missing = [key for key in required_true if document.get(key) is not True]
    if missing:
        raise AssociationError("association evidence lacks required safety checks: " + ", ".join(missing))
    if document.get("unresolved_risks") != []:
        raise AssociationError("association evidence has unresolved risks")
    if not _nonempty_list(document.get("changed_files")):
        raise AssociationError("association evidence must list reviewed changed files")
    if not _nonempty_list(document.get("verification")):
        raise AssociationError("association evidence must list verification evidence")
    expected_author = _as_text(document.get("pr_author"))
    expected_head = _as_text(document.get("head_sha"))
    expected_base = _as_text(document.get("base_ref"))
    if not all((expected_author, expected_head, expected_base)):
        raise AssociationError("association evidence requires pr_author, head_sha, and base_ref")
    duplicate_issue_url, duplicate_issue_number = _duplicate_evidence(document, expected_issue)
    return {
        "pr_author": expected_author,
        "head_sha": expected_head,
        "base_ref": expected_base,
        "changed_files": document["changed_files"],
        "verification": document["verification"],
        "duplicate_issue_url": duplicate_issue_url or None,
        "duplicate_issue_number": duplicate_issue_number,
    }


def _numbers(payload: Any) -> set[str]:
    if not isinstance(payload, list):
        return set()
    return {
        str(item.get("number"))
        for item in payload
        if isinstance(item, dict) and item.get("number") is not None
    }


def _live_context(issue_payload: Any, pr_payload: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(issue_payload, dict):
        raise AssociationError("live Issue GET returned invalid data")
    if not isinstance(pr_payload, dict):
        raise AssociationError("live PR GET returned invalid data")
    issue_author = _login(issue_payload.get("user") or issue_payload.get("author") or issue_payload.get("reporter"))
    pr_author = _login(pr_payload.get("user") or pr_payload.get("author"))
    state = _as_text(pr_payload.get("state") or pr_payload.get("pr_state")).casefold()
    merged = pr_payload.get("merged") is True or pr_payload.get("pr_merged") is True or state == "merged"
    base_ref = _as_text(
        (pr_payload.get("base") or {}).get("ref")
        if isinstance(pr_payload.get("base"), dict)
        else pr_payload.get("base_ref")
    )
    head_sha = _as_text(
        (pr_payload.get("head") or {}).get("sha")
        if isinstance(pr_payload.get("head"), dict)
        else pr_payload.get("head_sha")
    )
    if not issue_author:
        raise AssociationError("live Issue author could not be verified")
    if not pr_author or pr_author.casefold() != evidence["pr_author"].casefold():
        raise AssociationError("live PR author differs from reviewed evidence")
    if state not in GOOD_PR_STATES and not merged:
        raise AssociationError("PR is not open or merged")
    if state in {"closed", "close"} and not merged:
        raise AssociationError("PR is closed without merge")
    if base_ref != evidence["base_ref"]:
        raise AssociationError("live PR base differs from reviewed evidence")
    if head_sha != evidence["head_sha"]:
        raise AssociationError("live PR head changed after review")
    return {
        "issue_author": issue_author,
        "pr_author": pr_author,
        "state": state or ("merged" if merged else "open"),
        "merged": merged,
        "base_ref": base_ref,
        "head_sha": head_sha,
    }


def _base_result(issue: dict[str, str], pr: dict[str, str], evidence_file: str | Path,
                 response_result: str | Path) -> dict[str, Any]:
    return {
        "target": issue,
        "pr": pr,
        "status": "preview",
        "association_status": "not_started",
        "association_source": None,
        "post_started": False,
        "post_attempted": False,
        "post_http_status": None,
        "response_result": str(response_result),
        "evidence_file": str(evidence_file),
    }


class LinkedPrRecord(NamedTuple):
    path: Path
    issue: dict[str, str]
    pr: dict[str, str]
    live: dict[str, Any]
    result_file: Path
    source: str
    native_linked_issue: str | None = None


def _write_linked_pr_file(record: LinkedPrRecord) -> None:
    path, issue, pr, live, result_file, source, native_linked_issue = record
    native_association = source != "duplicate_cross_reference"
    document = {
        "target_issue_url": issue["url"],
        "issue_author": live["issue_author"],
        "repository": f"{issue['owner']}/{issue['repo']}",
        "association_result": str(result_file),
        "linked_prs": [{
            "pr_url": pr["url"],
            "pr_author": live["pr_author"],
            "state": live["state"],
            "merged": live["merged"],
            "association_verified": native_association,
            "association_source": source,
            "association_mode": source,
            "public_cross_reference_verified": not native_association,
            "native_linked_issue": native_linked_issue,
            "association_constraint": "single_issue_limit" if not native_association else None,
            "covers_issue": True,
            "base_ref": live["base_ref"],
            "head_sha": live["head_sha"],
        }],
    }
    _save_json(path, document)


class AssociationOperation(NamedTuple):
    issue: dict[str, str]
    pr: dict[str, str]
    evidence: dict[str, Any]
    response: dict[str, Any]
    result: dict[str, Any]
    result_path: Path
    linked_path: Path
    issue_endpoint: str
    pr_endpoint: str
    pr_issues_endpoint: str
    issue_prs_endpoint: str
    session: Any
    token: str


def _preview_association(operation):
    with _ResultLock(operation.result_path):
        if operation.result_path.exists():
            old = _read_json(operation.result_path, "association result")
            matches = (
                isinstance(old, dict)
                and old.get("target") == operation.issue
                and old.get("pr") == operation.pr
            )
            if matches and old.get("post_started"):
                return old
        _save_json(operation.result_path, operation.result)
    return operation.result


def _load_previous_association(operation):
    if not operation.result_path.exists():
        return None
    old = _read_json(operation.result_path, "association result")
    if not isinstance(old, dict) or old.get("target") != operation.issue or old.get("pr") != operation.pr:
        raise AssociationError("association result conflicts with target Issue or PR")
    operation.result["post_started"] = bool(old.get("post_started"))
    operation.result["post_attempted"] = bool(old.get("post_attempted"))
    return old


def _association_preflight(operation):
    try:
        issue_payload = api_get(operation.session, operation.issue_endpoint, operation.token)
        pr_payload = api_get(operation.session, operation.pr_endpoint, operation.token)
        live = _live_context(issue_payload, pr_payload, operation.evidence)
        links = api_get(
            operation.session, operation.pr_issues_endpoint, operation.token,
            params={"per_page": 100},
        )
        return issue_payload, live, links, None
    except AssociationError:
        raise
    except Exception as exc:
        operation.result.update(
            status="unknown", association_status="unknown", reason="live preflight GET failed",
        )
        operation.result["error_type"] = type(exc).__name__
        _save_json(operation.result_path, operation.result)
        return None, None, None, operation.result


def _reuse_preexisting_association(operation, live, links):
    if operation.issue["issue_number"] not in _numbers(links):
        return None
    try:
        reverse = api_get(
            operation.session, operation.issue_prs_endpoint, operation.token,
            params={"mode": 0, "per_page": 100},
        )
    except Exception as exc:
        operation.result.update(
            status="unknown", association_status="unknown",
            reason="reverse association readback failed",
        )
        operation.result["error_type"] = type(exc).__name__
    else:
        if operation.pr["pr_number"] not in _numbers(reverse):
            operation.result.update(
                status="unknown", association_status="unknown",
                reason="association is visible only from PR side",
            )
        else:
            operation.result.update(
                status="reused", association_status="verified",
                association_source="preexisting", pr_author=live["pr_author"],
                issue_author=live["issue_author"],
            )
            _write_linked_pr_file(LinkedPrRecord(
                operation.linked_path, operation.issue, operation.pr, live,
                operation.result_path, "preexisting",
            ))
    _save_json(operation.result_path, operation.result)
    return operation.result


def _duplicate_cross_reference(operation, live, links):
    other_links = sorted(_numbers(links) - {operation.issue["issue_number"]})
    if not other_links:
        return None
    if len(other_links) != 1 or operation.evidence.get("duplicate_issue_number") != other_links[0]:
        operation.result.update(
            status="blocked_existing_association", association_status="not_started",
            association_source="preexisting_other_issue", existing_linked_issues=other_links,
            reason="PR already has a different native Issue association",
        )
        _save_json(operation.result_path, operation.result)
        return operation.result
    comment_id = operation.response.get("comment_id")
    if comment_id is None:
        raise AssociationError("duplicate cross-reference requires a verified response comment ID")
    try:
        comments = api_get(
            operation.session, f"{operation.issue_endpoint}/comments", operation.token,
            params={"per_page": 100},
        )
    except Exception as exc:
        operation.result.update(
            status="unknown", association_status="unknown",
            reason="response cross-reference GET failed",
        )
        operation.result["error_type"] = type(exc).__name__
        _save_json(operation.result_path, operation.result)
        return operation.result
    comment = next((
        item for item in comments
        if isinstance(item, dict) and str(item.get("id")) == str(comment_id)
    ), None) if isinstance(comments, list) else None
    body = _as_text(comment.get("body")) if isinstance(comment, dict) else ""
    if operation.pr["url"] not in body or operation.evidence["duplicate_issue_url"] not in body:
        raise AssociationError("verified response does not publicly cross-reference the PR and duplicate Issue")
    operation.result.update(
        status="verified", association_status="verified_cross_reference",
        association_source="duplicate_cross_reference", existing_linked_issues=other_links,
        pr_author=live["pr_author"], issue_author=live["issue_author"],
        reason=("GitCode permits one native Issue per PR; preserved the duplicate "
                "Issue link and verified the public cross-reference"),
    )
    _save_json(operation.result_path, operation.result)
    _write_linked_pr_file(LinkedPrRecord(
        operation.linked_path, operation.issue, operation.pr, live, operation.result_path,
        "duplicate_cross_reference", native_linked_issue=other_links[0],
    ))
    return operation.result


def _send_association(operation):
    post_error = None
    try:
        response = api_post_json(
            operation.session, operation.pr_issues_endpoint, operation.token,
            json_data=[int(operation.issue["issue_number"])],
        )
        operation.result["post_http_status"] = getattr(response, "status_code", None)
        if operation.result["post_http_status"] is not None and operation.result["post_http_status"] >= 400:
            operation.result["post_error"] = safe_error_text(response)
    except Exception as exc:
        post_error = exc
        operation.result["post_http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
    return post_error


def _association_readback(operation):
    try:
        pr_links = api_get(
            operation.session, operation.pr_issues_endpoint, operation.token,
            params={"per_page": 100},
        )
        issue_links = api_get(
            operation.session, operation.issue_prs_endpoint, operation.token,
            params={"mode": 0, "per_page": 100},
        )
    except Exception as exc:
        operation.result.update(
            status="unknown", association_status="unknown",
            reason="association POST could not be verified",
        )
        operation.result["error_type"] = type(exc).__name__
        _save_json(operation.result_path, operation.result)
        return None
    return (
        operation.issue["issue_number"] in _numbers(pr_links)
        and operation.pr["pr_number"] in _numbers(issue_links)
    )


def _finish_post_association(operation, live, verified, post_error):
    if verified:
        operation.result.update(
            status="verified", association_status="verified",
            pr_author=live["pr_author"], issue_author=live["issue_author"],
        )
        _save_json(operation.result_path, operation.result)
        _write_linked_pr_file(LinkedPrRecord(
            operation.linked_path, operation.issue, operation.pr, live,
            operation.result_path, "current_run",
        ))
    elif post_error is not None:
        operation.result.update(
            status="unknown", association_status="unknown",
            reason="POST outcome unknown and association is absent",
        )
    else:
        operation.result.update(
            status="failed", association_status="failed",
            reason="POST completed but bidirectional GET did not verify association",
        )
    if not verified:
        _save_json(operation.result_path, operation.result)
    return operation.result


def _post_association(operation, live):
    operation.result.update(
        post_started=True, post_attempted=True, association_source="current_run",
    )
    _save_json(operation.result_path, operation.result)
    post_error = _send_association(operation)
    verified = _association_readback(operation)
    if verified is None:
        return operation.result
    return _finish_post_association(operation, live, verified, post_error)


def _apply_association(operation):
    with _ResultLock(operation.result_path):
        old = _load_previous_association(operation)
        issue_payload, live, links, terminal = _association_preflight(operation)
        if terminal is not None:
            return terminal
        terminal = _reuse_preexisting_association(operation, live, links)
        if terminal is not None:
            return terminal
        terminal = _duplicate_cross_reference(operation, live, links)
        if terminal is not None:
            return terminal
        if old and old.get("post_started"):
            operation.result.update(
                status="unknown", association_status="unknown",
                association_source="current_run",
                reason="previous association POST result is not visible; no retry was sent",
            )
            _save_json(operation.result_path, operation.result)
            return operation.result
        if _as_text(issue_payload.get("state")).casefold() not in {"open", "opened"}:
            raise AssociationError("Issue must still be open before PR association")
        return _post_association(operation, live)


def execute(*, config_path: str | Path, issue_url: str, pr_url: str,
            evidence_file: str | Path, response_result: str | Path,
            result_file: str | Path, linked_pr_file: str | Path,
            apply: bool = False, association_approved: bool = False,
            session: Any = None) -> dict[str, Any]:
    issue = _issue_target(issue_url)
    pr = _pr_target(pr_url)
    if not _same_repo(issue, pr):
        raise AssociationError("Issue and PR must belong to the same repository")
    config = load_handler_config(config_path)
    try:
        policy = get_automation_policy(config)
    except Exception as exc:
        raise AssociationError(str(exc)) from exc
    _check_config_target(config, issue)
    response = _read_json(response_result, "response result")
    if not isinstance(response, dict):
        raise AssociationError("response result must be a JSON object")
    _response_target(response, issue)
    evidence = _validate_evidence(_read_json(evidence_file, "association evidence"), issue, pr)
    result_path = Path(result_file)
    linked_path = Path(linked_pr_file)
    result = _base_result(issue, pr, evidence_file, response_result)
    result["response_status"] = response["status"]
    api_base = resolve_api_base(config.get("gitcode_api"), repo_url=issue_url)
    repo_endpoint = f"{api_base}/repos/{issue['owner']}/{issue['repo']}"
    issue_endpoint = f"{repo_endpoint}/issues/{issue['issue_number']}"
    pr_endpoint = f"{repo_endpoint}/pulls/{pr['pr_number']}"
    operation = AssociationOperation(
        issue, pr, evidence, response, result, result_path, linked_path,
        issue_endpoint, pr_endpoint, f"{pr_endpoint}/issues",
        f"{issue_endpoint}/pull_requests", session, "",
    )
    if not apply:
        return _preview_association(operation)
    if not (policy["auto_response"] or association_approved):
        raise AssociationError(
            "PR association is disabled; enable auto-response or pass --association-approved"
        )
    token = resolve_token()
    session = session or make_session(rate_limit_dir=rate_limit_path(config.get("cache_dir")))
    operation = operation._replace(session=session, token=token)
    return _apply_association(operation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--issue-url", required=True)
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--evidence-file", required=True)
    parser.add_argument("--response-result", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--linked-pr-file", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--association-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(
            config_path=args.config,
            issue_url=args.issue_url,
            pr_url=args.pr_url,
            evidence_file=args.evidence_file,
            response_result=args.response_result,
            result_file=args.result_file,
            linked_pr_file=args.linked_pr_file,
            apply=args.apply,
            association_approved=args.association_approved,
        )
    except (AssociationError, OSError, RuntimeError, ValueError) as exc:
        write_stderr(f"Error: PR association blocked: {exc}")
        return 2
    write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"preview", "verified", "reused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
