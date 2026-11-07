#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Check DESIGN.md and eval/module_interfaces.yaml against their output format.

Checks document structure and module input/output references. It does not
evaluate formulas, choose patterns, or judge algorithmic correctness.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
FORMAT_PATH = Path(__file__).resolve().parent.parent / "references" / "artifacts.yaml"

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_format() -> dict:
    data = yaml.safe_load(FORMAT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("artifacts"), dict):
        raise ValueError("output format must contain an artifacts mapping")
    for name in ("DESIGN", "module_interfaces"):
        entry = data["artifacts"].get(name)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
        ):
            raise ValueError(f"output format must define {name}.path")
    return data


def _split_document(text: str) -> tuple[str, str] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    body_start = end + 5
    return text[4:end], text[body_start:]


def _frontmatter(text: str) -> dict | None:
    parts = _split_document(text)
    if parts is None:
        return None
    try:
        data = yaml.safe_load(parts[0])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def headings(body: str) -> set[str]:
    found = set()
    fence = None
    for line in body.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            marker_end = marker.end()
            if fence is None:
                fence = token
            elif (
                token[0] == fence[0]
                and len(token) >= len(fence)
                and not line[marker_end:].strip()
            ):
                fence = None
            continue
        if fence is None:
            match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*$", line)
            if match:
                found.add(match.group(1).strip().lower())
    return found


def validate_design(path: Path, contract: dict) -> list[str]:
    entry = contract.get("artifacts", {}).get("DESIGN")
    if not isinstance(entry, dict):
        return ["contract has no artifacts.DESIGN entry"]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"DESIGN.md not readable: {path}: {exc}"]

    parts = _split_document(text)
    if parts is None:
        return ["DESIGN.md must start with a YAML front matter block"]
    meta = _frontmatter(text)
    if meta is None:
        return ["DESIGN.md front matter must be a YAML mapping"]

    failures = [
        f"frontmatter missing required key: {key}"
        for key in entry.get("frontmatter", {}).get("required", [])
        if key not in meta
    ]
    if "op_name" in meta and (
        not isinstance(meta["op_name"], str) or not meta["op_name"].strip()
    ):
        failures.append("op_name must be a non-empty string")
    if "dynamic_axes" in meta and (
        not isinstance(meta["dynamic_axes"], list) or not meta["dynamic_axes"]
    ):
        failures.append("dynamic_axes must be a non-empty list")
    required_headings = entry.get("required_headings", [])
    hset = headings(parts[1])
    failures.extend(
        f"required heading missing: {heading}"
        for heading in required_headings
        if str(heading).lower() not in hset
    )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op-dir",
        type=Path,
        required=True,
        help="operator directory containing the design outputs",
    )
    args = parser.parse_args(argv)

    try:
        contract = load_format()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logging.error("FAIL: %s", exc)
        return 1

    entry = contract["artifacts"]["DESIGN"]
    path = args.op_dir / entry["path"]
    failures = validate_design(path, contract)
    failures.extend(validate_interfaces(args.op_dir, contract))
    if failures:
        logging.error("FAIL")
        for failure in failures:
            logging.error("- %s", failure)
        return 1
    logging.info("PASS")
    return 0


def validate_interfaces(op_dir: Path, contract: dict) -> list[str]:
    from validate_yaml import validate

    entry = contract.get("artifacts", {}).get("module_interfaces")
    if not isinstance(entry, dict):
        return ["contract has no module_interfaces entry"]
    path = op_dir / entry["path"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"module interfaces not readable: {exc}"]
    if not isinstance(data, dict):
        return ["module interfaces must be a YAML mapping"]
    missing = [key for key in entry.get("required_keys", []) if key not in data]
    if missing:
        return [f"module interfaces missing keys: {missing}"]
    return validate(data)


if __name__ == "__main__":
    raise SystemExit(main())
