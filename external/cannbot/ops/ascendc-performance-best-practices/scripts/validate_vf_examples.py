#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate the Reg VF example catalog without requiring a CANN toolchain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = SKILL_ROOT / "reference" / "regbase" / "catalog.json"
VALID_FORMS = {"reference_template", "integration_skeleton", "annotated_excerpt"}
VALID_READINESS = {"integration_skeleton", "annotated_excerpt", "reference_only"}
VALID_BUILD_CONTRACTS = {"unverified", "device_only", "xasc_combined"}
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def load_catalog(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("catalog root must be an object")
    return data


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        catalog = load_catalog(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot load catalog: {exc}"]

    if catalog.get("schema_version") != 5:
        errors.append("schema_version must be 5")
    if catalog.get("architecture") != "dav-3510":
        errors.append("architecture must be dav-3510")
    if catalog.get("api_namespace") != "AscendC::Reg":
        errors.append("api_namespace must be AscendC::Reg")

    examples = catalog.get("examples")
    if not isinstance(examples, list) or not examples:
        return errors + ["examples must be a non-empty array"]

    seen_ids: set[str] = set()
    required = {
        "id",
        "pattern",
        "match_hints",
        "form",
        "readiness",
        "guide_path",
        "entry_points",
        "preconditions",
        "example_build_contract",
    }
    for index, example in enumerate(examples):
        label = f"examples[{index}]"
        if not isinstance(example, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = sorted(required - example.keys())
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
            continue

        example_id = example["id"]
        label = str(example_id)
        if not isinstance(example_id, str) or not ID_PATTERN.fullmatch(example_id):
            errors.append(f"{label}: id must be a lowercase kebab-case string")
        elif example_id in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(example_id)

        if example["form"] not in VALID_FORMS:
            errors.append(f"{label}: unsupported form {example['form']!r}")
        if example["readiness"] not in VALID_READINESS:
            errors.append(f"{label}: unsupported readiness {example['readiness']!r}")
        if "verification" in example:
            errors.append(f"{label}: catalog must not contain historical verification results")

        match_hints = example["match_hints"]
        if (
            not isinstance(match_hints, list)
            or not match_hints
            or not all(isinstance(item, str) and item.strip() for item in match_hints)
        ):
            errors.append(f"{label}: match_hints must be a non-empty string array")

        guide = SKILL_ROOT / str(example["guide_path"])
        for field, target in (("guide_path", guide),):
            try:
                target.relative_to(SKILL_ROOT)
            except ValueError:
                errors.append(f"{label}: {field} escapes the skill directory")
                continue
            if not target.is_file():
                errors.append(f"{label}: {field} does not exist: {target}")
            else:
                guide_text = target.read_text(encoding="utf-8", errors="replace")
                for heading in ("## 为什么这样写", "## 怎么使用"):
                    if heading not in guide_text:
                        errors.append(f"{label}: guide is missing required explanation section {heading!r}")

        entry_points = example["entry_points"]
        if (
            not isinstance(entry_points, list)
            or not entry_points
            or not all(isinstance(item, str) for item in entry_points)
        ):
            errors.append(f"{label}: entry_points must be a non-empty string array")

        missing_code_fields = [field for field in ("implementation_path", "sha256") if field not in example]
        if missing_code_fields:
            errors.append(f"{label}: missing code fields: {', '.join(missing_code_fields)}")
        else:
            implementation = SKILL_ROOT / str(example["implementation_path"])
            try:
                implementation.relative_to(SKILL_ROOT)
            except ValueError:
                errors.append(f"{label}: implementation_path escapes the skill directory")
            else:
                if not implementation.is_file():
                    errors.append(f"{label}: implementation_path does not exist: {implementation}")
                else:
                    digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
                    if digest != example["sha256"]:
                        errors.append(f"{label}: sha256 mismatch: expected {example['sha256']}, got {digest}")
                    implementation_text = implementation.read_text(encoding="utf-8", errors="replace")
                    if example["readiness"] != "reference_only":
                        legacy_tokens = {
                            "MicroAPI::": "legacy MicroAPI namespace",
                            "Reg::DataCopy": "legacy Reg::DataCopy spelling; use LoadAlign/StoreAlign",
                            "DataCopyUnAlign": "legacy unaligned copy spelling; use StoreUnAlign/StoreUnAlignPost",
                            "Reg::ReduceMax": "legacy reduction spelling; use Reg::Reduce<Reg::ReduceType::MAX>",
                            "Reg::ReduceMin": "legacy reduction spelling; use Reg::Reduce<Reg::ReduceType::MIN>",
                        }
                        for token, description in legacy_tokens.items():
                            if token in implementation_text:
                                errors.append(f"{label}: reusable example contains {description}")
                    if isinstance(entry_points, list):
                        for symbol in entry_points:
                            if isinstance(symbol, str) and symbol not in implementation_text:
                                errors.append(f"{label}: entry point not found in implementation: {symbol!r}")

        example_contract = example["example_build_contract"]
        if not isinstance(example_contract, dict) or example_contract.get("type") not in VALID_BUILD_CONTRACTS:
            errors.append(f"{label}: example_build_contract.type must be one of {sorted(VALID_BUILD_CONTRACTS)}")
        elif example_contract["type"] == "unverified":
            if not example_contract.get("note"):
                errors.append(f"{label}: unverified example_build_contract requires a note")
        else:
            contract_probe = example_contract.get("probe_path")
            if not contract_probe or not (SKILL_ROOT / str(contract_probe)).is_file():
                errors.append(f"{label}: example_build_contract requires an existing probe_path")

        readiness = example["readiness"]
        if readiness == "integration_skeleton":
            if example["form"] != "integration_skeleton":
                errors.append(f"{label}: integration_skeleton readiness requires integration_skeleton form")
        elif readiness == "annotated_excerpt":
            if example["form"] != "annotated_excerpt":
                errors.append(f"{label}: annotated_excerpt readiness requires annotated_excerpt form")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    catalog_path = args.catalog.resolve()
    errors = validate(catalog_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    catalog = load_catalog(catalog_path)
    count = len(catalog["examples"])
    print(f"VF example catalog OK: {count} self-contained examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
