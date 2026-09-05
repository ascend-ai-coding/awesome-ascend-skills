#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read and update the operator-to-GitCode-owner mapping safely.

The GitCode Issue Handler uses this script after an operator owner is supplied by
the user.  It keeps operator matching case-insensitive, preserves unrelated
top-level YAML fields and mappings, and replaces the config atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from runtime_paths import (  # noqa: E402
    LEGACY_OPERATOR_OWNERS_CONFIG,
    OPERATOR_OWNERS_CONFIG,
    compatible_read_path,
    path_text,
)
from cli_output import write_stdout  # noqa: E402

OWNER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _write_stdout(text: str) -> None:
    """Write the JSON command protocol to stdout."""
    write_stdout(text)


class ConfigError(ValueError):
    """Raised when operator owner input or config is invalid."""


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConfigError(
            "PyYAML is required; install it with: pip install pyyaml"
        ) from exc
    return yaml


def normalize_operator(value: str) -> str:
    operator = value.strip()
    if not operator or any(char in operator for char in "\r\n\0"):
        raise ConfigError("operator name must be a non-empty single-line value")
    return operator


def normalize_owner(value: str) -> str:
    owner = value.strip()
    if owner.startswith("@"):
        owner = owner[1:]
    if owner.casefold() == "direct":
        raise ConfigError(
            "'direct' is a handling decision and must not be saved as an owner"
        )
    if not OWNER_PATTERN.fullmatch(owner):
        raise ConfigError(
            "owner must be one GitCode login using letters, digits, '.', '_' or '-'"
        )
    return owner


def load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"operators": {}}
    yaml = _yaml_module()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("operator owner config must be a YAML mapping")
    operators = raw.setdefault("operators", {})
    if operators is None:
        raw["operators"] = {}
    elif not isinstance(operators, dict):
        raise ConfigError("'operators' must be a YAML mapping")
    return raw


def lookup_owner(path: Path, operator: str) -> tuple[str | None, str | None]:
    requested = normalize_operator(operator)
    operators = load_document(path)["operators"]
    for key, value in operators.items():
        if not isinstance(key, str):
            raise ConfigError("all operator names must be strings")
        if key.strip().casefold() != requested.casefold():
            continue
        if value in (None, ""):
            return key, None
        if not isinstance(value, str):
            raise ConfigError(f"owner for operator '{key}' must be a string")
        return key, normalize_owner(value)
    return None, None


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml_module()
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            yaml.safe_dump(
                document,
                temp_file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        try:
            os.chmod(temp_name, mode)
            os.replace(temp_name, path)
        except OSError as exc:
            raise ConfigError(f"cannot update {path}: {exc}") from exc
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def set_owner(
    path: Path,
    operator: str,
    owner: str,
    *,
    source_path: Path | None = None,
) -> dict[str, str]:
    requested = normalize_operator(operator)
    normalized_owner = normalize_owner(owner)
    document = load_document(source_path or path)
    operators = document["operators"]
    matched_key = next(
        (
            key
            for key in operators
            if isinstance(key, str) and key.strip().casefold() == requested.casefold()
        ),
        requested,
    )
    operators[matched_key] = normalized_owner
    _atomic_write(path, document)
    return {"operator": matched_key, "owner": normalized_owner}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read or update operator_owners.yaml")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Config path (default: {path_text(OPERATOR_OWNERS_CONFIG)})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lookup = subparsers.add_parser("lookup", help="Look up an operator owner")
    lookup.add_argument("--operator", required=True)

    update = subparsers.add_parser("set", help="Persist an operator owner")
    update.add_argument("--operator", required=True)
    update.add_argument("--owner", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = OPERATOR_OWNERS_CONFIG if args.config is None else Path(args.config)
    read_path = (
        compatible_read_path(
            path,
            canonical=OPERATOR_OWNERS_CONFIG,
            legacy=LEGACY_OPERATOR_OWNERS_CONFIG,
        )
        if args.config is None
        else path
    )
    try:
        if args.command == "lookup":
            matched_key, owner = lookup_owner(read_path, args.operator)
            payload = {
                "status": "found" if owner else "missing",
                "operator": matched_key or normalize_operator(args.operator),
                "owner": owner,
                "config": str(read_path),
            }
        else:
            updated = set_owner(path, args.operator, args.owner, source_path=read_path)
            payload = {"status": "saved", "config": str(path), **updated}
    except ConfigError as exc:
        _write_stdout(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
        )
        return 2
    _write_stdout(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
