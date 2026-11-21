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
"""Locate OPP kernel metadata/object candidates with optional TilingKey filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any


from cli_common import configure_logging, integer


LOGGER = logging.getLogger(__name__)
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.output")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def symbol_candidates(logical_name: str, core_type: str) -> list[str]:
    core_upper = core_type.upper()
    if core_upper.startswith("MIX"):
        return [f"{logical_name}_mix_aic", f"{logical_name}_mix_aiv"]
    if core_upper in {"AIC", "CUBE"}:
        return [logical_name, f"{logical_name}_mix_aic"]
    if core_upper in {"AIV", "VECTOR"}:
        return [logical_name, f"{logical_name}_mix_aiv"]
    return [logical_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find OPP kernel JSON/object pairs with optional TilingKey filtering."
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--opp-root",
        help="OPP root. Defaults to ASCEND_OPP_PATH, then $ASCEND_HOME_PATH/opp.",
    )
    location.add_argument(
        "--cann-root",
        help="CANN root containing opp/. Useful when ASCEND variables are not loaded.",
    )
    parser.add_argument("--operator", required=True, help="Operator name substring.")
    parser.add_argument(
        "--tiling-key",
        type=integer,
        help="Optional exact TilingKey filter (decimal or hex); omit to keep all entries.",
    )
    parser.add_argument("--soc", help="Optional path substring such as ascend910b.")
    return parser.parse_args()


def unique_existing_roots(entries: list[str]) -> list[Path]:
    roots: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_opp_roots(args: argparse.Namespace) -> list[Path]:
    if args.opp_root:
        return unique_existing_roots([args.opp_root])
    if args.cann_root:
        opp_root = Path(args.cann_root).expanduser() / "opp"
        return unique_existing_roots([str(opp_root)])

    opp_entries = os.environ.get("ASCEND_OPP_PATH", "").split(os.pathsep)
    roots = unique_existing_roots(opp_entries)
    if roots:
        return roots
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        return []
    return unique_existing_roots([str(Path(ascend_home).expanduser() / "opp")])


def parse_entry_key(entry: dict[str, Any]) -> int | None:
    value = entry.get("tilingKey")
    if value is None:
        return None
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def match_entries(
    metadata: dict[str, Any], metadata_path: Path, requested_key: int | None
) -> list[dict[str, Any]]:
    kernel_list = metadata.get("kernelList")
    entries = kernel_list if isinstance(kernel_list, list) else []
    matches: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parsed_key = parse_entry_key(entry)
        if requested_key is not None and parsed_key != requested_key:
            continue
        logical_name = str(
            entry.get("kernelName") or metadata.get("kernelName") or metadata_path.stem
        )
        core_type = str(entry.get("kernelType") or metadata.get("coreType") or "")
        matches.append(
            {
                "tilingKey": entry.get("tilingKey"),
                "kernelType": core_type,
                "logicalKernelName": logical_name,
                "symbolCandidates": symbol_candidates(logical_name, core_type),
            }
        )
    return matches


def load_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def object_path_from_metadata(
    metadata: dict[str, Any], metadata_path: Path
) -> Path | None:
    binary_name = metadata.get("binFileName") or metadata_path.stem
    binary_suffix = metadata.get("binFileSuffix") or ".o"
    if not isinstance(binary_name, str) or not isinstance(binary_suffix, str):
        return None
    return metadata_path.parent / f"{binary_name}{binary_suffix}"


def add_object_hashes(
    record: dict[str, Any], metadata: dict[str, Any], object_path: Path
) -> None:
    metadata_sha256 = metadata.get("sha256")
    if isinstance(metadata_sha256, str):
        record["metadataSha256"] = metadata_sha256
    if not object_path.is_file():
        return
    object_sha256 = file_sha256(object_path)
    record["objectSha256"] = object_sha256
    if isinstance(metadata_sha256, str):
        record["sha256MatchesMetadata"] = (
            object_sha256.lower() == metadata_sha256.lower()
        )


def build_record(
    opp_root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    object_path: Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    kernel_list = metadata.get("kernelList")
    kernel_count = len(kernel_list) if isinstance(kernel_list, list) else 0
    record: dict[str, Any] = {
        "oppRoot": str(opp_root),
        "metadata": str(metadata_path),
        "object": str(object_path),
        "objectExists": object_path.is_file(),
        "metadataKernelName": metadata.get("kernelName"),
        "coreType": metadata.get("coreType"),
        "blockDim": metadata.get("blockDim"),
        "opParaSize": metadata.get("opParaSize"),
        "workspace": metadata.get("workspace"),
        "kernelCount": kernel_count,
        "entries": entries,
    }
    add_object_hashes(record, metadata, object_path)
    return record


def path_matches(metadata_path: Path, operator: str, soc: str | None) -> bool:
    path_text = str(metadata_path).lower()
    if operator not in normalize_name(path_text):
        return False
    return soc is None or soc in path_text


def find_records(
    opp_roots: list[Path], operator: str, tiling_key: int | None, soc: str | None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for opp_root in opp_roots:
        for metadata_path in opp_root.rglob("*.json"):
            if not path_matches(metadata_path, operator, soc):
                continue
            metadata = load_metadata(metadata_path)
            if metadata is None:
                continue
            entries = match_entries(metadata, metadata_path, tiling_key)
            if tiling_key is not None and not entries:
                continue
            object_path = object_path_from_metadata(metadata, metadata_path)
            if object_path is None:
                continue
            record = build_record(
                opp_root, metadata_path, metadata, object_path, entries
            )
            results.append(record)
    return results


def main() -> int:
    configure_logging(OUTPUT_LOGGER)
    args = parse_args()
    opp_roots = resolve_opp_roots(args)
    if not opp_roots:
        LOGGER.error(
            "OPP root is missing; pass --opp-root/--cann-root or source the "
            "matching set_env.sh."
        )
        return 2
    operator = normalize_name(args.operator)
    if not operator:
        LOGGER.error("--operator must contain letters or digits")
        return 2
    soc = args.soc.lower() if args.soc else None
    results = find_records(opp_roots, operator, args.tiling_key, soc)
    if not results:
        LOGGER.error("No matching OPP kernel metadata found.")
        return 1
    OUTPUT_LOGGER.info("%s", json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
