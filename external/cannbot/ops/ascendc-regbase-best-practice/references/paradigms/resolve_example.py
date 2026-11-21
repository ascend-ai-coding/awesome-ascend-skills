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
"""Resolve example-routes.yaml — accept spec.yaml, extract paradigms, map to reference example packages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def load_example_routes(skill_dir: Path) -> dict:
    routes_file = skill_dir / "references" / "paradigms" / "example-routes.yaml"
    if not routes_file.exists():
        return {}
    with open(routes_file) as f:
        data = yaml.safe_load(f) or {}
    return data.get("routes", {})


def extract_paradigm_info(spec_text: str) -> tuple[str, list[str]]:
    category = ""
    paradigms: list[str] = []
    in_op = False
    for line in spec_text.splitlines():
        if line.startswith("op:"):
            in_op = True
            continue
        if in_op and line and not line.startswith((" ", "\t", "#")):
            break
        if in_op:
            if "category:" in line:
                category = line.split(":", 1)[1].strip()
            if "paradigms:" in line:
                items = re.findall(r"(\w+)", line.split(":", 1)[1])
                paradigms = items
    return category, paradigms


def resolve_examples(
    routes: dict,
    paradigms: list[str],
    skill_dir: Path,
) -> dict[str, list[str]]:
    para_routes = routes.get("paradigms", {})
    result: dict[str, list[str]] = {}
    seen: set[str] = set()

    for p in paradigms:
        for ref_path in para_routes.get(p, []):
            full = skill_dir / ref_path
            if full.exists():
                if str(full) not in seen:
                    result.setdefault(p, []).append(str(full))
                    seen.add(str(full))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve paradigms to example packages from example-routes.yaml"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="operators/{op}/docs/spec.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parents[2]
    spec_text = args.spec.read_text(encoding="utf-8")
    _category, paradigms = extract_paradigm_info(spec_text)

    routes = load_example_routes(skill_dir)
    resolved = resolve_examples(routes, paradigms, skill_dir)

    if args.json:
        import json

        print(json.dumps(resolved, indent=2, ensure_ascii=False))
    else:
        for para, paths in resolved.items():
            print(f"[{para}]")
            for p in paths:
                print(f"  {p}")


if __name__ == "__main__":
    main()
