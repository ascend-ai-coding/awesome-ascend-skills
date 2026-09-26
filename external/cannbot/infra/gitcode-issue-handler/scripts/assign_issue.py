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
"""Preview or idempotently assign one GitCode Issue.

Assignment is deliberately a separate operation from publishing a response.
The response result and the selected owner evidence are checked locally before
the Issue endpoint is touched.  The default mode only writes a local result
file; ``--apply`` is gated by the configured automatic-response policy or an
explicit assignment approval.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

_HERE = Path(__file__).resolve().parent
_TOOLKIT = _HERE.parent.parent / "gitcode-toolkit" / "scripts"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from gitcode_client import (  # noqa: E402
    api_get,
    api_patch,
    make_session,
    parse_issue_url,
    resolve_api_base,
    resolve_token,
)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from cli_output import write_stderr, write_stdout  # noqa: E402
from handler_config import get_automation_policy, load_handler_config  # noqa: E402
from mutation_result import ResultLock as _ResultLock, save_json as _save_json  # noqa: E402
from pr_response_policy import pr_active, select_response_pr  # noqa: E402
from response_artifacts import save_response_artifacts  # noqa: E402
from operator_owner_config import lookup_owner  # noqa: E402
from runtime_paths import OPERATOR_OWNERS_CONFIG, rate_limit_path  # noqa: E402


OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GOOD_RESPONSE_STATUSES = {"verified", "reused"}


class AssignmentError(ValueError):
    """Raised when local evidence or assignment policy is invalid."""


def _read_json(path: str | Path, label: str) -> Any:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssignmentError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AssignmentError(f"{label} is not valid JSON: {path}") from exc
    return value


def _normalize_owner(value: str) -> str:
    owner = str(value or "").strip().removeprefix("@")
    if not OWNER_RE.fullmatch(owner) or owner.casefold() == "direct":
        raise AssignmentError("owner must be one GitCode login")
    return owner


def _target(issue_url: str) -> dict[str, str]:
    try:
        owner, repo, number = parse_issue_url(issue_url)
    except Exception as exc:  # toolkit exposes a ValueError subclass
        raise AssignmentError(f"invalid Issue URL: {issue_url}") from exc
    if not str(number).isdigit():
        raise AssignmentError("Issue URL must contain a numeric IID")
    return {
        "owner": owner,
        "repo": repo,
        "issue_number": str(number),
        "url": issue_url,
    }


def _identity_from_url(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str):
        return None
    try:
        owner, repo, number = parse_issue_url(value)
    except Exception:
        return None
    return owner.casefold(), repo.casefold(), str(number)


def _same_identity(left: tuple[str, str, str], right: dict[str, str]) -> bool:
    return left == (
        right["owner"].casefold(), right["repo"].casefold(), right["issue_number"],
    )


def _response_target(response: dict[str, Any], target: dict[str, str]) -> None:
    if response.get("status") not in GOOD_RESPONSE_STATUSES:
        raise AssignmentError("response result must be verified or reused")
    actual = response.get("target")
    if isinstance(actual, str):
        identity = _identity_from_url(actual)
    elif isinstance(actual, dict):
        identity = None
        if actual.get("url"):
            identity = _identity_from_url(actual.get("url"))
        if identity is None and all(actual.get(key) is not None for key in ("owner", "repo", "issue_number")):
            identity = (
                str(actual["owner"]).casefold(),
                str(actual["repo"]).casefold(),
                str(actual["issue_number"]),
            )
    else:
        identity = _identity_from_url(response.get("issue_url"))
    if identity is None or not _same_identity(identity, target):
        raise AssignmentError("response result target does not match the Issue URL")


def _check_config_target(config: dict[str, Any], target: dict[str, str]) -> None:
    if target["issue_number"] in {str(i) for i in config.get("ignored_issue_ids", [])}:
        raise AssignmentError("Issue is explicitly ignored by the current configuration")
    configured = config.get("repo")
    if configured in (None, ""):
        return
    value = str(configured).strip().removesuffix(".git").strip("/").casefold()
    expected = f"{target['owner']}/{target['repo']}".casefold()
    if value != expected:
        raise AssignmentError("config repo does not match the Issue URL")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _operator_names(entry: dict[str, Any], document: dict[str, Any]) -> list[str]:
    values = entry.get("operators", entry.get("operator"))
    if values is None:
        values = document.get("operators", document.get("operator"))
    if isinstance(values, str):
        return [values] if values.strip() else []
    if isinstance(values, list):
        return [value.strip() for value in values if isinstance(value, str) and value.strip()]
    return []


def _candidate_entries(document: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)], {}
    if not isinstance(document, dict):
        raise AssignmentError("candidate file must contain a JSON object or list")
    entries = document.get("candidates")
    if entries is None:
        entries = document.get("operator_candidates")
    if not isinstance(entries, list):
        raise AssignmentError("candidate file has no candidates list")
    return [item for item in entries if isinstance(item, dict)], document


def _core_evidence(entry: dict[str, Any]) -> bool:
    if entry.get("login_verified") is not True:
        return False
    identity = entry.get("identity_evidence")
    if not (_as_text(identity) or (isinstance(identity, list) and identity)):
        return False
    evidence_keys = (
        "actual_core_contribution",
        "direct_core_fix_verified",
        "core_contribution_verified",
    )
    explicit = [entry.get(key) for key in evidence_keys if key in entry]
    if explicit and not any(value is True for value in explicit):
        return False
    contribution = _as_text(entry.get("contribution")) or _as_text(entry.get("summary_description"))
    evidence = entry.get("evidence")
    has_evidence = bool(_as_text(evidence)) or (
        isinstance(evidence, list) and any(_as_text(item) or isinstance(item, dict) for item in evidence)
    )
    return bool(contribution and has_evidence)


def _declared_target(document: dict[str, Any], target: dict[str, str]) -> None:
    # Candidate reports often describe the source Issue, so only enforce fields
    # explicitly named as the target.  This prevents a source IID from being
    # mistaken for the fork IID while still rejecting an explicit mismatch.
    for key in ("target_issue_url", "issue_url"):
        if key in document:
            identity = _identity_from_url(document[key])
            if identity is None or not _same_identity(identity, target):
                raise AssignmentError("candidate file target does not match the Issue URL")
    for key in ("target_issue_iid", "target_issue_number"):
        if key in document and str(document[key]) != target["issue_number"]:
            raise AssignmentError("candidate file target IID does not match the Issue URL")


def _owner_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.name == "classify_config.yaml":
        return path.with_name(OPERATOR_OWNERS_CONFIG.name)
    return path.with_name("operator_owners.yaml")


def _configured_owner(config_path: str | Path, operators: list[str], owner: str) -> str | None:
    path = _owner_config_path(config_path)
    for operator in operators:
        try:
            _, configured = lookup_owner(path, operator)
        except Exception as exc:
            raise AssignmentError(f"cannot read operator owner mapping: {exc}") from exc
        if configured and configured.casefold() == owner.casefold():
            return configured
    return None


def _validate_candidate(path: str | Path, target: dict[str, str], owner: str,
                       config_path: str | Path) -> dict[str, Any]:
    document = _read_json(path, "candidate file")
    entries, root = _candidate_entries(document)
    _declared_target(root, target)
    matches = [item for item in entries if _as_text(item.get("login")).casefold() == owner.casefold()]
    if len(matches) != 1:
        raise AssignmentError("candidate file must contain exactly one selected owner login")
    entry = matches[0]
    operators = _operator_names(entry, root)
    configured = _configured_owner(config_path, operators, owner) if operators else None
    if not configured and not _core_evidence(entry):
        raise AssignmentError("selected candidate lacks core contribution evidence")
    return {
        "source": "operator_owner" if configured else "core_candidate",
        "provisional": configured is None,
        "owner_confirmation": "configured" if configured else "pending",
        "operators": operators,
        "candidate_login": _as_text(entry.get("login")),
        "evidence_count": len(entry.get("evidence")) if isinstance(entry.get("evidence"), list) else 1,
    }


def _fallback_context(path: str | Path, target: dict[str, str],
                      config_path: str | Path) -> dict[str, Any] | None:
    document = _read_json(path, "candidate file")
    # Require an explicit empty result, not missing/malformed candidates or
    # rejected candidate evidence. An unfinished investigation is not empty.
    if not isinstance(document, dict) or document.get("candidates") != []:
        return None
    _declared_target(document, target)
    if not document.get("target_issue_url") and not document.get("issue_url"):
        raise AssignmentError("empty candidate result requires target_issue_url")
    if document.get("status") not in {"prepared", "insufficient_evidence"}:
        raise AssignmentError("empty candidate result requires completed investigation status")
    if document.get("operator_candidates"):
        raise AssignmentError("fallback cannot be used while operator candidates exist")
    operators = _operator_names({}, document)
    for operator in operators:
        _, configured = lookup_owner(_owner_config_path(config_path), operator)
        if configured:
            raise AssignmentError("configured operator owner takes precedence over fallback")
    return {
        "source": "fallback_user", "provisional": True,
        "owner_confirmation": "pending", "operators": operators,
        "reason": "no candidates after investigation; configured fallback recipient",
    }


def _verify_fallback_user(owner: str, config: dict[str, Any], session: Any, token: str) -> dict[str, str]:
    base = resolve_api_base(config.get("gitcode_api"), repo_url="https://gitcode.com")
    endpoint = f"{base}/users/{owner}"
    try:
        user = api_get(session, endpoint, token)
    except Exception as exc:
        raise AssignmentError("fallback user verification failed; no assignment sent") from exc
    login = _login_from_payload(user)
    if not isinstance(user, dict) or not login or login.casefold() != owner.casefold():
        raise AssignmentError("fallback user is invalid or returned a different login; no assignment sent")
    return {"endpoint": endpoint, "login": login}


def _pr_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if "/merge_requests/" in value or "/pull/" in value:
        return value
    return None


def _linked_records(document: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)], {}
    if not isinstance(document, dict):
        raise AssignmentError("linked PR file must contain a JSON object or list")
    values = document.get("linked_prs", document.get("prs", document.get("pulls")))
    if values is None:
        values = [document]
    if not isinstance(values, list):
        raise AssignmentError("linked PR file PR entries must be a list")
    records = []
    for item in values:
        if isinstance(item, dict):
            merged = dict(document)
            merged.update(item)
            records.append(merged)
    return records, document


def _linked_author(record: dict[str, Any]) -> str:
    for key in ("pr_author", "author_login", "login"):
        if _as_text(record.get(key)):
            return _as_text(record[key])
    for key in ("user", "author"):
        value = record.get(key)
        if isinstance(value, dict) and _as_text(value.get("login")):
            return _as_text(value["login"])
    return ""


def _linked_valid(record: dict[str, Any], target: dict[str, str]) -> bool:
    url = _pr_url(record.get("pr_url") or record.get("merge_request_url") or record.get("url"))
    if not url:
        return False
    issue_url = record.get("target_issue_url") or record.get("issue_url")
    if issue_url is not None:
        identity = _identity_from_url(issue_url)
        if identity is None or not _same_identity(identity, target):
            return False
    elif record.get("target_issue_iid") is not None:
        if str(record["target_issue_iid"]) != target["issue_number"]:
            return False
    elif record.get("target_issue_number") is not None:
        if str(record["target_issue_number"]) != target["issue_number"]:
            return False
    elif record.get("issue_iid") is not None:
        if str(record["issue_iid"]) != target["issue_number"]:
            return False
    else:
        return False
    # A PR number and an IID alone are insufficient: the relationship must
    # either predate this run or have been created and verified by the
    # association executor, and actual coverage must be explicit.

    def first_present(keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in record:
                return record[key]
        return None

    preexisting = first_present(
        ("preexisting_association", "association_preexisting", "linked_before")
    )
    association_verified = first_present(("association_verified", "linkage_verified"))
    cross_reference_verified = record.get("public_cross_reference_verified") is True
    covers = first_present((
        "covers_issue", "covers_target_issue", "issue_coverage",
        "coverage_verified", "issue_coverage_verified", "direct_issue_fix_verified",
    ))
    association_confirmed = (
        preexisting is not True
        and association_verified is not True
        and not cross_reference_verified
    )
    if association_confirmed or covers is not True:
        return False
    repository = (
        record.get("target_repository")
        or record.get("issue_repository")
        or record.get("repository")
    )
    if repository is not None:
        expected = f"{target['owner']}/{target['repo']}".casefold()
        if str(repository).strip().removesuffix(".git").strip("/").casefold() != expected:
            return False
    return bool(_linked_author(record) and pr_active(record))


def _validate_linked(path: str | Path, target: dict[str, str], owner: str | None) -> dict[str, Any]:
    document = _read_json(path, "linked PR file")
    records, root = _linked_records(document)
    issue_author_value = root.get("issue_author") or root.get("issue_author_login") or root.get("reporter")
    if not issue_author_value and isinstance(root.get("issue"), dict):
        issue_author_value = root["issue"].get("author") or root["issue"].get("reporter")
    if isinstance(issue_author_value, dict):
        issue_author_value = issue_author_value.get("login")
    issue_author = _as_text(issue_author_value)
    if not issue_author:
        raise AssignmentError("linked PR evidence lacks a verified Issue author")
    valid = [
        dict(record, pr_author=_linked_author(record),
             pr_url=_pr_url(record.get("pr_url") or record.get("merge_request_url") or record.get("url")))
        for record in records if _linked_valid(record, target)
    ]
    try:
        record, selection = select_response_pr(valid, issue_author)
    except ValueError as exc:
        raise AssignmentError(str(exc)) from exc
    selected_owner = _normalize_owner(_linked_author(record))
    if owner and _normalize_owner(owner).casefold() != selected_owner.casefold():
        raise AssignmentError("selected owner conflicts with self-author priority or PR coverage selection")
    return {
        "source": "linked_pr", "provisional": True, "owner_confirmation": "linked_pr",
        "owner": selected_owner, "issue_author": issue_author,
        "response_exempt": selection == "self_authored",
        "selection": selection, "coverage_review": record.get("coverage_review", {}),
        "pr_url": _pr_url(record.get("pr_url") or record.get("merge_request_url") or record.get("url")),
        "pr_state": _as_text(record.get("state") or record.get("pr_state")),
        "association_mode": _as_text(record.get("association_mode") or record.get("association_source")),
        "native_linked_issue": _as_text(record.get("native_linked_issue")),
    }


def _live_pr_endpoint(pr_url: str, config: dict[str, Any]) -> str:
    parsed = urlparse(pr_url)
    if parsed.scheme != "https" or parsed.netloc.casefold() not in {"gitcode.com", "www.gitcode.com"}:
        raise AssignmentError("linked PR URL must be an HTTPS GitCode URL")
    parts = [part for part in parsed.path.split("/") if part]
    try:
        marker = next(
            index
            for index, part in enumerate(parts)
            if part.casefold() in {"pull", "pulls", "merge_requests"}
        )
        pr_owner, pr_repo, pr_number = parts[marker - 2], parts[marker - 1], parts[marker + 1]
    except (StopIteration, IndexError) as exc:
        raise AssignmentError("linked PR URL has no valid PR identity") from exc
    if marker < 2 or not pr_number.isdigit():
        raise AssignmentError("linked PR URL has no valid PR identity")
    base = resolve_api_base(config.get("gitcode_api"), repo_url="https://gitcode.com")
    return f"{base}/repos/{pr_owner}/{pr_repo}/pulls/{pr_number}"


def _login_from_payload(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("login", "username"):
            if _as_text(value.get(key)):
                return _as_text(value[key])
    return _as_text(value)


def _live_issue_author(issue: Any) -> str:
    if not isinstance(issue, dict):
        return ""
    for key in ("user", "author", "reporter"):
        login = _login_from_payload(issue.get(key))
        if login:
            return login
    return ""


class LinkedPrVerification(NamedTuple):
    context: dict[str, Any]
    target: dict[str, str]
    owner: str
    config: dict[str, Any]
    session: Any
    token: str
    issue_payload: Any


def _linked_pr_covers_target(linked_issues, target, context):
    if not isinstance(linked_issues, list):
        return False
    linked_numbers = {
        str(item.get("number"))
        for item in linked_issues
        if isinstance(item, dict) and item.get("number") is not None
    }
    if target["issue_number"] in linked_numbers:
        return True
    duplicate_fallback = context.get("association_mode") == "duplicate_cross_reference"
    native_issue = _as_text(context.get("native_linked_issue"))
    return duplicate_fallback and bool(native_issue) and native_issue in linked_numbers


def _verify_linked_pr_live(request: LinkedPrVerification) -> dict[str, Any]:
    context, target, owner, config, session, token, issue_payload = request
    issue_author = _live_issue_author(issue_payload)
    if not issue_author:
        raise AssignmentError("Issue author could not be verified from live GET")
    if issue_author.casefold() != context["issue_author"].casefold():
        raise AssignmentError("live Issue author differs from reviewed evidence")
    if context["response_exempt"] and issue_author.casefold() != owner.casefold():
        raise AssignmentError("self-authored response exemption does not match live Issue")
    endpoint = _live_pr_endpoint(context["pr_url"], config)
    try:
        live = api_get(session, endpoint, token)
    except Exception as exc:
        raise AssignmentError("linked PR live verification GET failed") from exc
    if not isinstance(live, dict):
        raise AssignmentError("linked PR live verification returned invalid data")
    live_author = _login_from_payload(live.get("user") or live.get("author"))
    if not live_author or live_author.casefold() != owner.casefold():
        raise AssignmentError("linked PR live author does not match selected owner")
    state = _as_text(live.get("state") or live.get("pr_state")).casefold()
    if not pr_active(live):
        raise AssignmentError("linked PR is not open or merged, or has expired")
    reviewed_sha = context.get("coverage_review", {}).get("head_sha")
    if reviewed_sha and reviewed_sha != (live.get("head") or {}).get("sha"):
        raise AssignmentError("linked PR changed after coverage review")
    try:
        linked_issues = api_get(session, f"{endpoint}/issues", token, params={"per_page": 100})
    except Exception as exc:
        raise AssignmentError("linked PR relationship verification GET failed") from exc
    if not _linked_pr_covers_target(linked_issues, target, context):
        raise AssignmentError("linked PR is not associated with the target Issue")
    return {
        "endpoint": endpoint,
        "author": live_author,
        "state": state,
        "association_verified": True,
        "association_mode": context.get("association_mode") or "native",
    }


def _assignee(issue: Any) -> str | None:
    if not isinstance(issue, dict):
        return None
    values = issue.get("assignee")
    if isinstance(values, dict):
        return _as_text(values.get("login")) or _as_text(values.get("username")) or None
    if isinstance(values, str) and values.strip():
        return values.strip().removeprefix("@")
    values = issue.get("assignees")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict) and _as_text(item.get("login")):
                return _as_text(item["login"])
            if isinstance(item, str) and item.strip():
                return item.strip().removeprefix("@")
    return None


def _issue_endpoint(target: dict[str, str], config: dict[str, Any]) -> str:
    base = resolve_api_base(config.get("gitcode_api"), repo_url=target["url"])
    return f"{base}/repos/{target['owner']}/{target['repo']}/issues/{target['issue_number']}"


def _base_result(target: dict[str, str], owner: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": target,
        "owner": owner,
        "status": "preview",
        "assignment_status": "not_started",
        "assignment_source": context["source"],
        "provisional": context["provisional"],
        "owner_confirmation": context["owner_confirmation"],
        "assignment_provisional": context["provisional"],
        "assigned_candidate": owner if context["provisional"] else None,
        "patch_started": False,
        "patch_attempted": False,
        "patch_http_status": None,
        "readback_assignee": None,
    }


class AssignmentOperation(NamedTuple):
    config: dict[str, Any]
    context: dict[str, Any]
    target: dict[str, str]
    owner: str
    result_path: Path
    result: dict[str, Any]
    session: Any
    token: str
    endpoint: str


def _load_previous_assignment(operation):
    if not operation.result_path.exists():
        return None
    old = _read_json(operation.result_path, "assignment result")
    if not isinstance(old, dict):
        raise AssignmentError("assignment result must be a JSON object")
    same_target = old.get("target") == operation.target
    same_owner = str(old.get("owner", "")).casefold() == operation.owner.casefold()
    if not same_target or not same_owner:
        raise AssignmentError("assignment result conflicts with target or owner")
    if old.get("assignment_source") != operation.context["source"]:
        raise AssignmentError("assignment result conflicts with assignment source")
    operation.result["patch_started"] = bool(old.get("patch_started"))
    operation.result["patch_attempted"] = bool(old.get("patch_attempted"))
    return old


def _initial_assignment_readback(operation, old):
    result = operation.result
    try:
        before = api_get(operation.session, operation.endpoint, operation.token)
    except Exception as exc:
        result.update(status="unknown", assignment_status="unknown", error="Issue GET failed")
        result["error_type"] = type(exc).__name__
        _save_json(operation.result_path, result)
        return None, result
    current = _assignee(before)
    result["readback_assignee"] = current
    if current:
        if current.casefold() == operation.owner.casefold():
            result.update(status="reused", assignment_status="verified")
            result["reuse_reason"] = "existing assignee matches selected owner; no PATCH sent"
            if operation.context["source"] == "linked_pr":
                result["linked_pr_live_readback"] = "skipped_existing_assignee"
        else:
            result.update(status="blocked_existing_assignee", assignment_status="not_started")
            result["reason"] = "Issue already has a different assignee"
        _save_json(operation.result_path, result)
        return None, result
    recoverable = old and old.get("patch_started") and old.get("status") in {
        "preview", "unknown", "verified", "reused", "failed",
    }
    if recoverable:
        result["patch_started"] = True
        result["patch_attempted"] = bool(old.get("patch_attempted", True))
        if old.get("status") in {"verified", "reused"}:
            result.update(status="failed", assignment_status="failed")
            result["reason"] = "previous assignment was verified but the assignee is now absent; no overwrite"
        else:
            result.update(status="unknown", assignment_status="unknown")
            result["reason"] = "previous PATCH result was unknown; no retry was sent"
        _save_json(operation.result_path, result)
        return None, result
    return before, None


def _verify_assignment_evidence(operation, before):
    if not isinstance(before, dict) or _as_text(before.get("state")).casefold() not in {"open", "opened"}:
        raise AssignmentError("Issue must still be open before assignment")
    if operation.context["source"] == "linked_pr":
        request = LinkedPrVerification(
            operation.context, operation.target, operation.owner, operation.config,
            operation.session, operation.token, before,
        )
        operation.result["linked_pr_live_readback"] = _verify_linked_pr_live(request)
    if operation.context["source"] == "fallback_user":
        operation.result["fallback_user_live_readback"] = _verify_fallback_user(
            operation.owner, operation.config, operation.session, operation.token
        )


def _patch_assignment(operation):
    result = operation.result
    result["patch_started"] = True
    result["patch_attempted"] = True
    _save_json(operation.result_path, result)
    patch_error = None
    try:
        response = api_patch(
            operation.session, operation.endpoint, operation.token,
            json_data={"assignee": operation.owner},
        )
        result["patch_http_status"] = getattr(response, "status_code", None)
    except Exception as exc:
        patch_error = exc
        result["patch_http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        after = api_get(operation.session, operation.endpoint, operation.token)
        current = _assignee(after)
        result["readback_assignee"] = current
    except Exception as exc:
        result.update(status="unknown", assignment_status="unknown")
        result["reason"] = "PATCH result could not be verified by GET"
        result["error_type"] = type(exc).__name__
        _save_json(operation.result_path, result)
        return result
    if current and current.casefold() == operation.owner.casefold():
        result.update(status="verified", assignment_status="verified")
    elif patch_error is not None:
        result.update(status="unknown", assignment_status="unknown")
        result["reason"] = "PATCH outcome unknown and assignee is still absent"
    else:
        result.update(status="failed", assignment_status="failed")
        result["reason"] = "PATCH completed but GET assignee did not match owner"
    _save_json(operation.result_path, result)
    return result


def _apply_assignment(operation):
    with _ResultLock(operation.result_path):
        old = _load_previous_assignment(operation)
        before, terminal = _initial_assignment_readback(operation, old)
        if terminal is not None:
            return terminal
        _verify_assignment_evidence(operation, before)
        return _patch_assignment(operation)


class AssignmentSelection(NamedTuple):
    config: dict[str, Any]
    target: dict[str, str]
    config_path: str | Path
    candidate_file: str | Path | None
    linked_pr_file: str | Path | None
    owner: str | None
    auto_response: bool
    auto_assign: bool
    assignment_approved: bool
    result_file: str | Path


def _skipped_fallback(request):
    result = {
        "target": request.target, "status": "skipped",
        "assignment_status": "not_started", "assignment_source": "fallback_user",
        "patch_started": False, "patch_attempted": False,
        "reason": "auto-assign disabled or fallback user empty",
    }
    result_path = Path(request.result_file)
    with _ResultLock(result_path):
        if result_path.exists():
            previous = _read_json(result_path, "assignment result")
            if isinstance(previous, dict) and previous.get("patch_started"):
                raise AssignmentError("existing assignment must be reconciled before skipping")
        _save_json(result_path, result)
    return result


def _select_assignment(request):
    fallback = (
        _fallback_context(request.candidate_file, request.target, request.config_path)
        if request.candidate_file else None
    )
    if fallback is not None:
        fallback_owner = request.config.get("auto-assign-fallback-user") or ""
        if not request.auto_assign or not fallback_owner:
            return None, None, None, _skipped_fallback(request)
        selected_owner = _normalize_owner(fallback_owner)
        if request.owner is not None and _normalize_owner(request.owner).casefold() != selected_owner.casefold():
            raise AssignmentError("selected owner does not match configured fallback user")
        return selected_owner, fallback, request.auto_assign, None
    if request.linked_pr_file:
        context = _validate_linked(request.linked_pr_file, request.target, request.owner)
        allowed = request.auto_response or request.assignment_approved
        return context["owner"], context, allowed, None
    selected_owner = _normalize_owner(request.owner)
    context = _validate_candidate(
        request.candidate_file, request.target, selected_owner, request.config_path,
    )
    allowed = request.auto_assign or request.assignment_approved or (
        request.auto_response and context["source"] == "operator_owner"
    )
    return selected_owner, context, allowed, None


def _preview_assignment(result_path, result, target, selected_owner, context):
    result["status"] = "preview"
    with _ResultLock(result_path):
        old = _read_json(result_path, "assignment result") if result_path.exists() else None
        if _matches_inflight_preview(old, target, selected_owner, context):
            return old
        _save_json(result_path, result)
    return result


def _matches_inflight_preview(old, target, selected_owner, context):
    if not isinstance(old, dict) or not old.get("patch_started"):
        return False
    matches = (
        old.get("target") == target,
        str(old.get("owner", "")).casefold() == selected_owner.casefold(),
        old.get("assignment_source") == context["source"],
    )
    return all(matches)


def _assignment_response(response_result, target, context, apply):
    if context.get("response_exempt"):
        return "exempt_self_authored_pr"
    if response_result is None:
        if apply:
            raise AssignmentError("response result must be verified or reused before assignment")
        return "pending"
    response = _read_json(response_result, "response result")
    if not isinstance(response, dict):
        raise AssignmentError("response result must be a JSON object")
    _response_target(response, target)
    return response["status"]


def execute(*, config_path: str | Path, issue_url: str, owner: str | None = None,
            response_result: str | Path | None = None, result_file: str | Path,
            candidate_file: str | Path | None = None,
            linked_pr_file: str | Path | None = None,
            apply: bool = False, assignment_approved: bool = False,
            session: Any = None) -> dict[str, Any]:
    if bool(candidate_file) == bool(linked_pr_file):
        raise AssignmentError("provide exactly one of candidate-file or linked-pr-file")
    target = _target(issue_url)
    config = load_handler_config(config_path)
    try:
        policy = get_automation_policy(config)
    except Exception as exc:
        raise AssignmentError(str(exc)) from exc
    auto_response = policy["auto_response"]
    auto_assign = policy["auto_assign"]
    _check_config_target(config, target)
    selection = AssignmentSelection(
        config, target, config_path, candidate_file, linked_pr_file, owner,
        auto_response, auto_assign, assignment_approved, result_file,
    )
    selected_owner, context, allowed, terminal = _select_assignment(selection)
    if terminal is not None:
        return terminal
    result_path = Path(result_file)
    result = _base_result(target, selected_owner, context)
    result["response_status"] = _assignment_response(response_result, target, context, apply)
    if response_result:
        result["response_result"] = str(response_result)
    result["selection"] = context.get("selection")
    result["artifacts"] = save_response_artifacts(
        result_file, issue_url,
        f"分配依据：{context['source']}；选择方式：{context.get('selection', 'owner/candidate')}。\n"
        f"首响依据：{result['response_status']}。\n"
        f"关联 PR：{context.get('pr_url', '无')}。\n"
        f"覆盖审阅：{json.dumps(context.get('coverage_review', {}), ensure_ascii=False)}",
        owner=selected_owner,
    )
    if context.get("pr_url"):
        result["linked_pr_url"] = context["pr_url"]
    if context.get("operators"):
        result["operators"] = context["operators"]
    if not apply:
        return _preview_assignment(result_path, result, target, selected_owner, context)
    if not allowed:
        raise AssignmentError("assignment is disabled; enable auto-assign or pass --assignment-approved")
    token = resolve_token()
    endpoint = _issue_endpoint(target, config)
    session = session or make_session(rate_limit_dir=rate_limit_path(config.get("cache_dir")))
    operation = AssignmentOperation(
        config, context, target, selected_owner, result_path, result,
        session, token, endpoint,
    )
    return _apply_assignment(operation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--issue-url", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate-file")
    group.add_argument("--linked-pr-file")
    parser.add_argument(
        "--owner",
        help="Selected login; omit to select a linked PR author or use an empty-candidate fallback",
    )
    parser.add_argument("--response-result", help="Verified/reused response; omit for self-authored PR or preview")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--assignment-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(
            config_path=args.config,
            issue_url=args.issue_url,
            candidate_file=args.candidate_file,
            linked_pr_file=args.linked_pr_file,
            owner=args.owner,
            response_result=args.response_result,
            result_file=args.result_file,
            apply=args.apply,
            assignment_approved=args.assignment_approved,
        )
    except (AssignmentError, OSError, RuntimeError, ValueError) as exc:
        write_stderr(f"Error: assignment blocked: {exc}")
        return 2
    write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"preview", "verified", "reused", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
