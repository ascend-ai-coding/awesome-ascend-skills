#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check whether the active CANN environment matches an Issue.

The script is intentionally read-only. It extracts expected CANN versions from
Issue text (or explicit CLI values), fingerprints the active environment, and
returns exit code 2 when reproduction would use a mismatched or mixed version.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from cli_output import write_stdout  # noqa: E402

VERSION = r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:\.[A-Za-z0-9]+)?"
CANN_TEXT_RE = re.compile(rf"(?i)\bcann\b[^\n\r]{{0,50}}?\b[vV]?({VERSION})")
CANN_PATH_RE = re.compile(rf"(?i)(?:^|[/\\])cann[-_]?({VERSION})(?:[/\\]|$)")
VERSION_LINE_RE = re.compile(rf"(?im)^\s*version\s*=\s*[vV]?({VERSION})\s*$")

ENV_PATH_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
)
TOOL_NAMES = ("asc_opc", "ccec", "atc")
LOGGER = logging.getLogger(__name__)


def _write_stdout(text: str) -> None:
    """Write command output without treating the output protocol as a log."""
    write_stdout(text)


def extract_expected_cann_versions(text: str) -> list[str]:
    versions = set(CANN_TEXT_RE.findall(text or ""))
    versions.update(CANN_PATH_RE.findall(text or ""))
    return sorted(versions)


def _issue_text(payload: object, issue_id: str | None) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("issues"), list):
        issues = payload["issues"]
    elif isinstance(payload, list):
        issues = payload
    elif isinstance(payload, dict):
        issues = [payload]
    else:
        raise ValueError("Issue JSON must be an object or an array")

    if issue_id is not None:
        issues = [
            item
            for item in issues
            if str(item.get("iid", item.get("number", ""))) == issue_id
        ]
        if not issues:
            raise ValueError(f"Issue {issue_id} was not found in the JSON input")
    elif len(issues) != 1:
        raise ValueError("JSON contains multiple Issues; pass --issue <IID>")

    issue = issues[0]
    parts = [str(issue.get(key, "")) for key in ("title", "description", "body")]
    for comment in issue.get("comments", []) or []:
        if isinstance(comment, dict):
            parts.append(str(comment.get("body", "")))
    return "\n".join(parts)


def _version_from_path(path: Path) -> str | None:
    match = CANN_PATH_RE.search(str(path))
    return match.group(1) if match else None


def _candidate_roots(path: Path) -> list[Path]:
    path = path.resolve(strict=False)
    roots = [path]
    roots.extend(list(path.parents)[:3])
    return roots


def _read_install_version(root: Path) -> list[tuple[Path, str]]:
    candidates = [root / "ascend_toolkit_install.info"]
    try:
        candidates.extend(root.glob("*-linux/ascend_toolkit_install.info"))
    except OSError:
        pass

    found = []
    for info_file in candidates:
        if not info_file.is_file():
            continue
        try:
            match = VERSION_LINE_RE.search(
                info_file.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        if match:
            found.append((info_file, match.group(1)))
    return found


def _collect_env_path_evidence(
    environ: dict[str, str],
    key: str,
    inspected_roots: set[Path],
) -> list[dict[str, str]]:
    """Collect evidence from one configured CANN environment variable."""
    evidence: list[dict[str, str]] = []
    value = environ.get(key)
    if not value:
        return evidence
    for raw_path in filter(None, value.split(os.pathsep)):
        path = Path(raw_path).expanduser().resolve(strict=False)
        path_version = _version_from_path(path)
        if path_version:
            evidence.append({"source": key, "path": str(path), "version": path_version})
        for root in _candidate_roots(path):
            if root in inspected_roots:
                continue
            inspected_roots.add(root)
            evidence.extend(
                {
                    "source": f"{key}:install_info",
                    "path": str(info_file),
                    "version": version,
                }
                for info_file, version in _read_install_version(root)
            )
    return evidence


def _collect_tool_evidence(
    environ: dict[str, str], tool_names: tuple[str, ...]
) -> list[dict[str, str]]:
    """Collect evidence from CANN executables found on PATH."""
    evidence = []
    for tool in tool_names:
        executable = shutil.which(tool, path=environ.get("PATH"))
        if not executable:
            continue
        path = Path(executable).resolve(strict=False)
        evidence.append(
            {
                "source": f"tool:{tool}",
                "path": str(path),
                "version": _version_from_path(path) or "unknown",
            }
        )
    return evidence


def collect_active_cann(
    environ: dict[str, str] | None = None,
    tool_names: tuple[str, ...] = TOOL_NAMES,
) -> list[dict[str, str]]:
    environ = dict(os.environ if environ is None else environ)
    inspected_roots: set[Path] = set()
    evidence = []
    for key in ENV_PATH_KEYS:
        evidence.extend(_collect_env_path_evidence(environ, key, inspected_roots))
    evidence.extend(_collect_tool_evidence(environ, tool_names))
    unique = {
        (item["source"], item["path"], item["version"]): item for item in evidence
    }
    return sorted(
        unique.values(),
        key=lambda item: (item["source"], item["path"], item["version"]),
    )


def _version_parts(version: str) -> tuple[int, ...] | None:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version):
        return None
    return tuple(int(part) for part in version.split("."))


def versions_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    expected_parts = _version_parts(expected)
    actual_parts = _version_parts(actual)
    if expected_parts is None or actual_parts is None:
        return False
    shorter = min(len(expected_parts), len(actual_parts))
    return expected_parts[:shorter] == actual_parts[:shorter]


def evaluate(expected: list[str], evidence: list[dict[str, str]]) -> dict[str, object]:
    expected = sorted(set(expected))
    actual = sorted(
        {item["version"] for item in evidence if item["version"] != "unknown"}
    )

    if not expected:
        status = "expected_unknown"
        action = "Extract the required CANN version before running an environment-sensitive reproduction."
    elif not actual:
        status = "environment_unavailable"
        action = (
            "Activate or provide the required CANN environment before reproduction."
        )
    elif len(actual) > 1:
        status = "mixed_environment"
        action = "Clean the shell environment so all CANN variables and tools resolve to one version."
    elif any(versions_match(wanted, found) for wanted in expected for found in actual):
        status = "match"
        action = "Environment version matches; reproduction may proceed."
    else:
        status = "mismatch"
        action = (
            "Stop before reproduction and report the expected and active CANN versions."
        )

    return {
        "status": status,
        "expected_cann_versions": expected,
        "active_cann_versions": actual,
        "evidence": evidence,
        "action": action,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issue-json",
        type=Path,
        help="fetch_issues.py JSON output or one Issue object",
    )
    parser.add_argument(
        "--issue", help="Issue IID to select when --issue-json contains multiple Issues"
    )
    parser.add_argument(
        "--issue-text", default="", help="Issue title/body/comments as plain text"
    )
    parser.add_argument(
        "--expected-cann",
        action="append",
        default=[],
        help="Expected CANN version; repeatable",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.issue_text
    if args.issue_json:
        try:
            payload = json.loads(args.issue_json.read_text(encoding="utf-8"))
            text = "\n".join((text, _issue_text(payload, args.issue)))
        except (OSError, ValueError) as exc:
            LOGGER.error("error: %s", exc)
            return 2

    expected = sorted(set(args.expected_cann + extract_expected_cann_versions(text)))
    result = evaluate(expected, collect_active_cann())
    if args.format == "json":
        _write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        lines = (
            f"status: {result['status']}",
            f"expected CANN: {', '.join(result['expected_cann_versions']) or 'unknown'}",
            f"active CANN: {', '.join(result['active_cann_versions']) or 'unavailable'}",
            f"action: {result['action']}",
        )
        _write_stdout("\n".join(lines))

    return 0 if result["status"] == "match" else 2


if __name__ == "__main__":
    raise SystemExit(main())
