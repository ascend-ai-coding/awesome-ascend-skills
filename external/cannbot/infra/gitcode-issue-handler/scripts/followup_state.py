#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Manage Issue follow-up watches and verified GitCode custom-state changes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import requests

_HERE = Path(__file__).resolve().parent
_TOOLKIT_SCRIPTS = _HERE.parent.parent / "gitcode-toolkit" / "scripts"
sys.path.insert(0, str(_TOOLKIT_SCRIPTS))
from gitcode_client import (  # noqa: E402
    api_get,
    api_patch,
    api_put,
    make_session,
    parse_issue_url,
    parse_iso,
    resolve_api_base,
    resolve_token,
)

sys.path.insert(0, str(_HERE))
from cli_output import write_stdout  # noqa: E402
from runtime_paths import (  # noqa: E402
    CLASSIFY_CONFIG,
    FETCH_CACHE,
    FOLLOWUP_WATCH_STATE,
    path_text,
    rate_limit_path,
)

SCHEMA_VERSION = "issue-followup.v1"
DEFAULT_STATE_FILE = path_text(FOLLOWUP_WATCH_STATE)
DEFAULT_WAITING_STATUS = "挂起"
DEFAULT_ACTIVE_STATUS = "进行中"
DEFAULT_POLL_HOURS = 24
DEFAULT_STALE_HOURS = 48
WAITING_TARGETS = {"reporter", "assignee"}
LOGGER = logging.getLogger(__name__)


class StatusTransition(NamedTuple):
    """Resolved endpoints and states for one custom-status operation."""

    session: requests.Session
    token: str
    web_api: str
    canonical_url: str
    issue_iid: str
    project_id: str
    current_status: str
    target_status: str
    target_status_id: object


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_timestamp(value: str | None, label: str) -> datetime:
    parsed = parse_iso(value) if isinstance(value, str) else None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError(f"Error: {label} must be an ISO-8601 timestamp with timezone")
    return parsed


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Error: cannot read follow-up state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Error: follow-up state must be a JSON object")
    return payload


def load_followup_config(path: str | Path | None = None) -> dict:
    """Load only the shared follow_up settings without importing classifier logic."""
    config_path = Path(path) if path else CLASSIFY_CONFIG
    if not config_path.exists():
        if path:
            raise ValueError(f"Error: config file does not exist: {config_path}")
        return {}
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except ImportError as exc:
        raise ValueError("Error: PyYAML is required to read follow-up config") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Error: cannot read follow-up config — {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Error: follow-up config must contain a YAML object")
    followup = raw.get("follow_up") or {}
    if not isinstance(followup, dict):
        raise ValueError("Error: follow_up config must be an object")
    return followup


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".followup-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_followup_state(path: str | Path, repository: str | None = None) -> dict:
    """Load and validate a repository-scoped follow-up state document."""
    state_path = Path(path)
    payload = _read_json(state_path)
    if not payload:
        return {
            "schema_version": SCHEMA_VERSION,
            "repository": repository,
            "updated_cursor": None,
            "issues": {},
        }
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Error: unsupported follow-up state schema")
    stored_repository = payload.get("repository")
    if repository and stored_repository and repository != stored_repository:
        raise ValueError(
            "Error: follow-up state belongs to a different repository: "
            f"{stored_repository}"
        )
    issues = payload.get("issues")
    if not isinstance(issues, dict):
        raise ValueError("Error: follow-up state 'issues' must be an object")
    payload["repository"] = repository or stored_repository
    return payload


def advance_updated_cursor(path: str | Path, repository: str, cursor: str) -> dict:
    """Atomically advance the all-state updated-at scan cursor."""
    proposed = _required_timestamp(cursor, "follow-up cursor")
    state = load_followup_state(path, repository)
    current_value = state.get("updated_cursor")
    current = (
        _required_timestamp(current_value, "stored follow-up cursor")
        if current_value
        else None
    )
    if current is None or proposed > current:
        state["updated_cursor"] = cursor
    _atomic_write(Path(path), state)
    return state


def watch_issue(
    path: str | Path,
    repository: str,
    issue_iid: str,
    *,
    reporter: str,
    assignee: str = "",
    waiting_on: str = "reporter",
    issue_url: str = "",
    maintainer_comment_id: str | int | None = None,
    maintainer_comment_at: str,
    waiting_since: str | None = None,
    poll_hours: int = DEFAULT_POLL_HOURS,
    stale_hours: int = DEFAULT_STALE_HOURS,
) -> dict:
    """Record which participant owns the next actionable Issue turn."""
    baseline = _required_timestamp(maintainer_comment_at, "maintainer comment time")
    waiting = (
        _required_timestamp(waiting_since, "waiting-since")
        if waiting_since
        else baseline
    )
    if not str(issue_iid).isdigit():
        raise ValueError("Error: Issue IID must be numeric")
    if not reporter.strip():
        raise ValueError("Error: reporter must not be empty")
    waiting_on = waiting_on.strip().casefold()
    if waiting_on not in WAITING_TARGETS:
        raise ValueError("Error: waiting-on must be reporter or assignee")
    normalized_assignee = assignee.strip().removeprefix("@")
    if waiting_on == "assignee" and not normalized_assignee:
        raise ValueError("Error: assignee must not be empty when waiting on assignee")
    if poll_hours <= 0 or stale_hours <= 0:
        raise ValueError("Error: poll and stale hours must be greater than zero")

    state = load_followup_state(path, repository)
    item = {
        "iid": str(issue_iid),
        "url": issue_url,
        "reporter": reporter.strip(),
        "assignee": normalized_assignee or None,
        "waiting_on": waiting_on,
        "conversation_state": f"awaiting_{waiting_on}",
        "last_maintainer_comment_id": maintainer_comment_id,
        "last_maintainer_comment_at": maintainer_comment_at,
        "waiting_since": waiting_since or maintainer_comment_at,
        "next_check_at": _iso(waiting + timedelta(hours=poll_hours)),
        "expires_at": (
            _iso(waiting + timedelta(hours=stale_hours))
            if waiting_on == "reporter"
            else None
        ),
        "updated_at": _iso(_now()),
    }
    state["issues"][str(issue_iid)] = item
    _atomic_write(Path(path), state)
    return item


def resolve_issue(path: str | Path, repository: str, issue_iid: str) -> bool:
    """Remove a watch only after the follow-up turn reached a real terminal state."""
    state = load_followup_state(path, repository)
    removed = state["issues"].pop(str(issue_iid), None) is not None
    if removed:
        _atomic_write(Path(path), state)
    return removed


def _web_api_base(issue_url: str, override: str | None) -> str:
    if override:
        return override.rstrip("/")
    parsed = urlparse(issue_url)
    if parsed.netloc.lower() in {"gitcode.com", "www.gitcode.com"}:
        return "https://web-api.gitcode.com"
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _issue_project_id(issue: dict) -> str:
    repository = issue.get("repository") or {}
    project_id = repository.get("id") if isinstance(repository, dict) else None
    if project_id is None:
        raise ValueError("Error: Issue response does not contain repository.id")
    return str(project_id)


def _status_catalog(session, web_api_base: str, owner: str, token: str) -> list[dict]:
    url = f"{web_api_base}/api/v2/groups/{owner}/issue-extend/issue_extend_status_list"
    payload = api_get(session, url, token)
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            return content
    raise ValueError("Error: GitCode custom Issue status list has an invalid shape")


def _extended_issue(
    session, web_api_base: str, project_id: str, issue_iid: str, token: str
) -> dict:
    url = (
        f"{web_api_base}/issuepr/api/v1/issue/{project_id}/"
        f"issue-extend/info/{issue_iid}"
    )
    payload = api_get(session, url, token)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("status"), str)
        or not payload["status"].strip()
    ):
        raise ValueError("Error: GitCode custom Issue state response is incomplete")
    return payload


def _find_status(catalog: list[dict], status_name: str) -> dict:
    """Return the single enabled status whose name matches the request."""
    matches = []
    for item in catalog:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        if str(item.get("name") or "").casefold() == status_name.casefold():
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(
            f"Error: custom Issue status '{status_name}' is missing or ambiguous"
        )
    return matches[0]


def _resolve_transition(
    issue_url: str,
    status_name: str,
    token: str,
    api_base: str | None,
    web_api_base: str | None,
) -> tuple[StatusTransition, dict]:
    """Read the Issue and custom-status catalog needed for a transition."""
    owner, repo, issue_iid = parse_issue_url(issue_url)
    session = make_session(rate_limit_dir=rate_limit_path(FETCH_CACHE))
    resolved_api = resolve_api_base(api_base, issue_url)
    resolved_web_api = _web_api_base(issue_url, web_api_base)
    canonical_url = f"{resolved_api}/repos/{owner}/{repo}/issues/{issue_iid}"
    issue = api_get(session, canonical_url, token)
    if not isinstance(issue, dict):
        raise ValueError("Error: GitCode Issue response is incomplete")
    project_id = _issue_project_id(issue)
    current = _extended_issue(session, resolved_web_api, project_id, issue_iid, token)
    target = _find_status(
        _status_catalog(session, resolved_web_api, owner, token), status_name
    )
    transition = StatusTransition(
        session,
        token,
        resolved_web_api,
        canonical_url,
        issue_iid,
        project_id,
        current.get("status", ""),
        target.get("name", ""),
        target.get("id"),
    )
    return transition, issue


def _transition_plan(
    issue_url: str,
    issue: dict,
    transition: StatusTransition,
    reopen: bool,
    apply: bool,
) -> dict:
    core_state = str(issue.get("state") or "")
    return {
        "issue": issue_url,
        "project_id": transition.project_id,
        "current_status": transition.current_status,
        "target_status": transition.target_status,
        "target_status_id": transition.target_status_id,
        "core_state": core_state,
        "reopen": bool(reopen and core_state.casefold() == "closed"),
        "mode": "apply" if apply else "dry_run",
    }


def _apply_custom_status(transition: StatusTransition, plan: dict) -> dict | None:
    """Write and verify the custom status, returning a failure when present."""
    if transition.current_status.casefold() == transition.target_status.casefold():
        return None
    transition_url = (
        f"{transition.web_api}/issuepr/api/v1/issue/{transition.project_id}/"
        f"issue-extend/status-flow/{transition.issue_iid}"
    )
    response = api_put(
        transition.session,
        transition_url,
        transition.token,
        json_data={
            "status_before": transition.current_status,
            "status_current": transition.target_status,
        },
    )
    if response.status_code >= 400:
        return {
            **plan,
            "status": "transition_failed",
            "http_status": response.status_code,
        }
    verified = _extended_issue(
        transition.session,
        transition.web_api,
        transition.project_id,
        transition.issue_iid,
        transition.token,
    )
    if (
        str(verified.get("status") or "").casefold()
        != transition.target_status.casefold()
    ):
        return {
            **plan,
            "status": "transition_failed",
            "reason": "status_readback_mismatch",
        }
    return None


def _reopen_core_issue(transition: StatusTransition, plan: dict) -> dict | None:
    """Reopen and verify the core Issue when the transition requires it."""
    if not plan.get("reopen"):
        return None
    response = api_patch(
        transition.session,
        transition.canonical_url,
        transition.token,
        json_data={"state": "open"},
    )
    if response.status_code >= 400:
        return {**plan, "status": "reopen_failed", "http_status": response.status_code}
    reopened = api_get(transition.session, transition.canonical_url, transition.token)
    if str(reopened.get("state") or "").casefold() not in {"open", "opened"}:
        return {
            **plan,
            "status": "reopen_failed",
            "reason": "state_readback_mismatch",
        }
    return None


def transition_issue_status(
    issue_url: str,
    status_name: str,
    *,
    token: str,
    api_base: str | None = None,
    web_api_base: str | None = None,
    reopen: bool = False,
    apply: bool = False,
    authorization_evidence: str | None = None,
) -> dict:
    """Plan or apply a custom Issue status transition and verify all writes."""
    status_name = status_name.strip()
    if not status_name:
        raise ValueError("Error: target custom Issue status must not be empty")
    transition, issue = _resolve_transition(
        issue_url, status_name, token, api_base, web_api_base
    )
    plan = _transition_plan(issue_url, issue, transition, reopen, apply)
    if not apply:
        return {**plan, "status": "would_transition"}
    if not authorization_evidence:
        raise ValueError("Error: --authorization-evidence is required with --apply")
    failure = _apply_custom_status(transition, plan) or _reopen_core_issue(
        transition, plan
    )
    if failure:
        return failure
    return {
        **plan,
        "status": "transitioned",
        "authorization_evidence": authorization_evidence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Optional classify config; the canonical config is used when present",
    )
    parser.add_argument(
        "--state-file",
        help=f"Override follow-up state path (default: {DEFAULT_STATE_FILE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="track an Issue awaiting its next actor")
    watch.add_argument("--repo", required=True)
    watch.add_argument("--issue", required=True)
    watch.add_argument("--reporter", required=True)
    watch.add_argument("--assignee", default="")
    watch.add_argument(
        "--waiting-on",
        choices=sorted(WAITING_TARGETS),
        default="reporter",
    )
    watch.add_argument("--issue-url", default="")
    watch.add_argument("--maintainer-comment-id")
    watch.add_argument("--maintainer-comment-at", required=True)
    watch.add_argument("--waiting-since")
    watch.add_argument("--poll-hours", type=int)
    watch.add_argument("--stale-hours", type=int)

    resolve = sub.add_parser("resolve", help="remove a terminal follow-up watch")
    resolve.add_argument("--repo", required=True)
    resolve.add_argument("--issue", required=True)

    inspect = sub.add_parser("inspect", help="print the follow-up state")
    inspect.add_argument("--repo")

    transition = sub.add_parser(
        "transition", help="plan or apply a verified GitCode custom-state change"
    )
    transition.add_argument("--issue", required=True, metavar="URL")
    transition.add_argument("--status-name", required=True)
    transition.add_argument("--reopen", action="store_true")
    transition.add_argument("--token", default=None)
    transition.add_argument("--api-base")
    transition.add_argument("--web-api-base")
    transition.add_argument("--apply", action="store_true")
    transition.add_argument("--authorization-evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        followup_cfg = load_followup_config(args.config)
        state_file = (
            args.state_file or followup_cfg.get("state_file") or DEFAULT_STATE_FILE
        )
        if args.command == "watch":
            result = watch_issue(
                state_file,
                args.repo,
                args.issue,
                reporter=args.reporter,
                assignee=args.assignee,
                waiting_on=args.waiting_on,
                issue_url=args.issue_url,
                maintainer_comment_id=args.maintainer_comment_id,
                maintainer_comment_at=args.maintainer_comment_at,
                waiting_since=args.waiting_since,
                poll_hours=int(
                    args.poll_hours
                    if args.poll_hours is not None
                    else followup_cfg.get("poll_hours", DEFAULT_POLL_HOURS)
                ),
                stale_hours=int(
                    args.stale_hours
                    if args.stale_hours is not None
                    else followup_cfg.get("stale_hours", DEFAULT_STALE_HOURS)
                ),
            )
        elif args.command == "resolve":
            result = {"removed": resolve_issue(state_file, args.repo, args.issue)}
        elif args.command == "inspect":
            result = load_followup_state(state_file, args.repo)
        else:
            result = transition_issue_status(
                args.issue,
                args.status_name,
                token=resolve_token(args.token),
                api_base=args.api_base,
                web_api_base=args.web_api_base,
                reopen=args.reopen,
                apply=args.apply,
                authorization_evidence=args.authorization_evidence,
            )
        write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, requests.RequestException) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
