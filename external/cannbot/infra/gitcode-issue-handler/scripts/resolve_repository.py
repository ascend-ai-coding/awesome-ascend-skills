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
"""Resolve and persist the GitCode repository used by the issue handler."""

from __future__ import annotations

import argparse
import json
import shutil
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from cli_output import write_stdout
    from handler_config import initialize_config
    from runtime_paths import CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG
except ModuleNotFoundError:  # pragma: no cover - direct import from a test
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cli_output import write_stdout  # type: ignore
    from handler_config import initialize_config  # type: ignore
    from runtime_paths import CLASSIFY_CONFIG, LEGACY_CLASSIFY_CONFIG  # type: ignore


_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HOSTS = {"gitcode.com", "www.gitcode.com"}


def _normalize_repo(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.endswith(".git"):
        value = value[:-4]
    if not _REPO.fullmatch(value) or any(part in {".", ".."} for part in value.split("/")):
        return None
    return value


def _repo_from_url(value: str) -> str | None:
    value = value.strip().rstrip("\r")
    match = re.match(r"^git@([^:]+):([^\s]+)$", value)
    if match:
        if match.group(1).lower() not in _HOSTS:
            return None
        return _normalize_repo(match.group(2).split("/issues/", 1)[0])
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "ssh", "git"}:
        return None
    if (parsed.hostname or "").lower() not in _HOSTS:
        return None
    path = parsed.path.strip("/")
    path = path.split("/issues/", 1)[0].split("/pull/", 1)[0]
    return _normalize_repo(path)


def _parse_target(target: str | None) -> str | None:
    if not target:
        return None
    return _normalize_repo(target) or _repo_from_url(target)


def _remote_candidates(root: Path) -> list[str]:
    git_executable = shutil.which("git")
    if not git_executable:
        return []
    try:
        proc = subprocess.run(
            [git_executable, "-C", str(root), "remote", "-v"],
            text=True, capture_output=True, check=False,
        )
    except OSError:
        return []
    found: list[str] = []
    for line in proc.stdout.splitlines():
        # remote -v has: name<TAB>url<TAB>(fetch|push).  Only parse the URL.
        parts = line.split()
        if len(parts) < 2:
            continue
        repo = _repo_from_url(parts[1])
        if repo and repo not in found:
            found.append(repo)
    return found


def _yaml_repo(path: Path) -> tuple[str, bool]:
    if yaml is None:
        raise ValueError("PyYAML is required")
    text = path.read_text(encoding="utf-8")
    try:
        node = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {exc}") from exc
    if node is None:
        return "", False
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("YAML config must contain a mapping")
    repo_nodes = []
    for key, value in node.value:
        if isinstance(key, yaml.ScalarNode) and key.value == "repo":
            repo_nodes.append(value)
    if len(repo_nodes) > 1:
        raise ValueError("config contains duplicate repo keys")
    if not repo_nodes:
        return "", False
    value = repo_nodes[0]
    if not isinstance(value, yaml.ScalarNode):
        raise ValueError("repo must be a scalar")
    if value.tag.endswith(":null") or value.value.strip().lower() in {"null", "~"}:
        return "", True
    return value.value.strip(), True


def _replace_repo(path: Path, repo: str) -> None:
    text = path.read_text(encoding="utf-8")
    node = yaml.compose(text)
    encoded = json.dumps(repo, ensure_ascii=False)
    if node is None:
        updated = text + ("" if not text or text.endswith("\n") else "\n") + f"repo: {encoded}\n"
    else:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("YAML config must contain a mapping")
        matches = [v for k, v in node.value if isinstance(k, yaml.ScalarNode) and k.value == "repo"]
        if matches:
            value = matches[0]
            start, end = value.start_mark.index, value.end_mark.index
            # An empty scalar can start directly after the colon (repo:).
            prefix = " " if start and text[start - 1] == ":" else ""
            updated = text[:start] + prefix + encoded + text[end:]
        elif node.flow_style:
            position = node.end_mark.index - 1
            prefix = ", " if node.value else ""
            updated = text[:position] + prefix + f"repo: {encoded}" + text[position:]
        else:
            position = node.end_mark.index
            prefix = "" if not position or text[position - 1] == "\n" else "\n"
            updated = text[:position] + prefix + f"repo: {encoded}\n" + text[position:]
    expected = yaml.safe_load(text) or {}
    expected["repo"] = repo
    if yaml.safe_load(updated) != expected:
        raise ValueError("Cannot update repo while preserving other YAML values")
    _atomic_text_replace(path, updated)


def _atomic_text_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _config_path(root: Path, config_path) -> Path:
    if config_path is not None:
        path = Path(config_path)
        path = path if path.is_absolute() else root / path
        if not path.exists():
            raise ValueError(f"config file does not exist: {path}")
        return path
    canonical = root / CLASSIFY_CONFIG
    legacy = root / LEGACY_CLASSIFY_CONFIG
    return canonical if canonical.exists() else (legacy if legacy.exists() else canonical)


def _selection_result(path, repo, candidates, reason):
    return {
        "status": "needs_selection",
        "repo": repo or "",
        "config_path": str(path),
        "candidates": candidates,
        "reason": reason,
    }


def _choose_repository(root, path, configured_repo, target, selection):
    requested = _parse_target(target)
    selected = _parse_target(selection)
    invalid_values = (target and not requested, selection and not selected)
    if any(invalid_values):
        return None, _selection_result(path, configured_repo, [], "invalid repository target")
    conflicting_target = all((configured_repo, requested, requested != configured_repo))
    if conflicting_target and not selected:
        candidates = [configured_repo, requested]
        return None, _selection_result(
            path, configured_repo, candidates, "explicit target conflicts with configured repo",
        )
    chosen = selected or requested or configured_repo
    if chosen:
        return chosen, None
    candidates = _remote_candidates(root)
    if len(candidates) == 1:
        return candidates[0], None
    reason = "multiple repository candidates" if candidates else "no valid GitCode repository candidate"
    return None, _selection_result(path, "", candidates, reason)


def _ensure_canonical_config(root, path):
    if not path.exists():
        initialize_config(root)
        return path
    if path != root / LEGACY_CLASSIFY_CONFIG or (root / CLASSIFY_CONFIG).exists():
        return path
    canonical = root / CLASSIFY_CONFIG
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=canonical.parent, delete=False) as stream:
        stream.write(path.read_bytes())
        temporary = Path(stream.name)
    temporary.replace(canonical)
    return canonical


def resolve_repository(root, target=None, selection=None, config_path=None) -> dict[str, Any]:
    root = Path(root).resolve()
    path = _config_path(root, config_path)
    configured = _yaml_repo(path)[0] if path.exists() else ""
    configured_repo = _normalize_repo(configured) if configured else None
    if configured and not configured_repo:
        raise ValueError("configured repo must be owner/repo")
    chosen, unresolved = _choose_repository(root, path, configured_repo, target, selection)
    if unresolved:
        return unresolved
    path = _ensure_canonical_config(root, path)
    if chosen != configured_repo:
        _replace_repo(path, chosen)
    return {"status": "resolved", "repo": chosen, "config_path": str(path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--target")
    parser.add_argument("--select")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        result = resolve_repository(args.repository_root, args.target, args.select, args.config)
    except (OSError, ValueError) as exc:
        result = {"status": "error", "reason": str(exc)}
        write_stdout(json.dumps(result, ensure_ascii=False))
        return 2
    write_stdout(json.dumps(result, ensure_ascii=False))
    return 2 if result.get("status") == "needs_selection" else 0


if __name__ == "__main__":
    raise SystemExit(main())
