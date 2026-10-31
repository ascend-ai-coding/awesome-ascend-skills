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

"""Materialize the bundled compiled warmup into a target operator project."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

DEFAULT_DESTINATION = "ops_profiling_warmup_ext"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    return parser.parse_args()


def asset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "compiled_warmup"


def source_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def tree_sha256(root: Path) -> str:
    """Deterministic digest over the asset files: sorted names plus their own SHA-256."""
    manifest = [
        [path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()]
        for path in source_files(root)
    ]
    payload = json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_destination(project_dir: Path, destination: str) -> Path:
    if not destination or Path(destination).is_absolute():
        raise ValueError("--destination must be a non-empty relative path")
    target = (project_dir / destination).resolve()
    if target == project_dir or project_dir not in target.parents:
        raise ValueError("--destination must stay inside --project-dir")
    return target


def identical_trees(source: Path, target: Path) -> bool:
    if not target.is_dir():
        return False
    source_names = [path.relative_to(source) for path in source_files(source)]
    return all(
        (target / name).is_file() and (source / name).read_bytes() == (target / name).read_bytes()
        for name in source_names
    )


def install(project_dir: Path, destination: str) -> dict:
    source = asset_root()
    if not source.is_dir():
        raise ValueError(f"bundled asset is missing: {source}")
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise ValueError(f"project directory does not exist: {project_dir}")
    target = validate_destination(project_dir, destination)

    status = "installed"
    if target.exists():
        if not identical_trees(source, target):
            raise ValueError(
                f"destination already exists with local changes: {target}; "
                "choose a new destination or reconcile it manually"
            )
        status = "already_current"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    return {
        "status": status,
        "destination": str(target),
        "asset_sha256": tree_sha256(source),
        "file_count": len(source_files(source)),
    }


def main() -> int:
    args = parse_args()
    try:
        result = install(Path(args.project_dir), args.destination)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
