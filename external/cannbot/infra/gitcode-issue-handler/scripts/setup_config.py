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
"""Prepare repository configuration and remember an explicit setup decision.

Questions and selective YAML edits belong to the calling agent. This script
never enables automation, accesses GitCode, or infers consent from silence.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from cli_output import write_stdout
from handler_config import (
    ConfigError,
    initialize_config,
    load_named_config,
    load_template,
)
from runtime_paths import (
    CLASSIFY_CONFIG,
    LEGACY_CLASSIFY_CONFIG,
    LEGACY_OPERATOR_OWNERS_CONFIG,
    OPERATOR_OWNERS_CONFIG,
    SETUP_STATE,
)

_CONFIG_PATHS = {
    "classify_config": CLASSIFY_CONFIG,
    "operator_owners": OPERATOR_OWNERS_CONFIG,
}
_SUMMARY_KEYS = ("repo", "responsibility", "auto-response", "auto-assign")
_DECISIONS = {"pending", "configured", "kept"}


def _read_state(root: Path) -> dict:
    path = root / SETUP_STATE
    if path.is_symlink():
        raise ConfigError("setup-state.json must not be a symbolic link")
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ConfigError("unsupported setup state; existing file was preserved")
    status = state.get("status")
    if not isinstance(status, str) or status not in _DECISIONS:
        raise ConfigError("invalid setup status; existing file was preserved")
    return state


def _save_decision(root: Path, status: str) -> None:
    path = root / SETUP_STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".setup-", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"version": 1, "status": status}, stream)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_configs(root: Path, *, allow_missing: bool = False) -> dict:
    documents = {}
    for name, relative in _CONFIG_PATHS.items():
        path = root / relative
        if not path.is_file():
            if allow_missing and not path.exists() and not path.is_symlink():
                continue
            raise ConfigError(f"configuration is missing or not a file: {path}")
        documents[name] = load_named_config(name, path, allow_legacy=False)
    return documents


def _has_existing_settings(root: Path, documents: dict) -> bool:
    for relative in (LEGACY_CLASSIFY_CONFIG, LEGACY_OPERATOR_OWNERS_CONFIG):
        if (root / relative).is_file():
            return True
    for name, document in documents.items():
        current = dict(document)
        defaults = dict(load_template(name))
        # Repository resolution may already have filled repo in a fresh template.
        if name == "classify_config":
            current.pop("repo", None)
            defaults.pop("repo", None)
        if current != defaults:
            return True
    return False


def _result(root: Path, status: str, documents: dict, created: dict) -> dict:
    config = documents["classify_config"]
    return {
        "status": status,
        "needs_configuration": status == "pending",
        "created": created,
        "config_path": str(root / CLASSIFY_CONFIG),
        "state_path": str(root / SETUP_STATE),
        "settings": {key: config.get(key) for key in _SUMMARY_KEYS},
    }


def prepare_setup(repository_root: str | Path, *, reconfigure: bool = False) -> dict:
    """Fill missing files and distinguish templates from existing user settings."""
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ConfigError("repository root must be an existing directory")
    state = _read_state(root)
    documents = _read_configs(root, allow_missing=True)
    existing = _has_existing_settings(root, documents)
    created = initialize_config(root)
    documents = _read_configs(root)
    status = state.get("status", "existing" if existing else "pending")
    restored_template = "classify_config" in created and not (root / LEGACY_CLASSIFY_CONFIG).is_file()
    if reconfigure or restored_template:
        status = "pending"
    if status == "pending" and state.get("status") != "pending":
        _save_decision(root, status)
    return _result(root, status, documents, created)


def finish_setup(repository_root: str | Path, decision: str) -> dict:
    """Record completion only after the caller receives the user's decision."""
    if decision not in {"configured", "kept"}:
        raise ConfigError("setup decision must be configured or kept")
    root = Path(repository_root).resolve()
    state = _read_state(root)
    documents = _read_configs(root)
    if state.get("status") != "pending":
        raise ConfigError("prepare or reopen the configuration guide before finishing")
    _save_decision(root, decision)
    return _result(root, decision, documents, {})


def main(argv: list[str] | None = None) -> int:
    """Run a local setup action and emit its result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--reconfigure", action="store_true", help="Reopen setup on user request")
    action.add_argument("--complete", action="store_true", help="Record user-confirmed settings")
    action.add_argument("--keep", action="store_true", help="Record the user's choice to keep settings")
    args = parser.parse_args(argv)
    try:
        if args.complete or args.keep:
            decision = "configured" if args.complete else "kept"
            result = finish_setup(args.repository_root, decision)
        else:
            result = prepare_setup(args.repository_root, reconfigure=args.reconfigure)
    except (OSError, ValueError) as exc:
        write_stdout(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 2
    write_stdout(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
