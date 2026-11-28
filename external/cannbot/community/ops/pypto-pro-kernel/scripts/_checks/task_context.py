# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded controller-declared task inventory; never discover or execute artifacts."""

from __future__ import annotations

import hashlib
import json
import re

from . import actions, host_io

KINDS = {"user_statement", "synthetic_fixture", "verifier_observation"}
ROLES = {"source", "reference", "diagnostic", "measurement", "verification"}


def report(path, *, config, goal=None):
    path = host_io.checked_path(path, config)
    stamp = path.stat()
    raw = host_io._bytes(
        path,
        min(
            65536,
            config["limits"]["max_file_bytes"],
            config["limits"]["max_scan_bytes"],
        ),
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            "query --task requires a pypto-task-context/1 JSON inventory, not task"
            " prose; use query TEXT for knowledge discovery"
        ) from error
    if not isinstance(data, dict) or data.get("schema") != "pypto-task-context/1":
        raise ValueError(
            "Expected a pypto-task-context/1 JSON inventory; use query TEXT for knowledge discovery"
        )
    declared = actions.goal(data.get("goal"))
    if goal is not None and declared is not None and goal != declared:
        raise ValueError("Requested goal differs from the controller task goal")
    goal = actions.goal(goal if goal is not None else declared)
    artifacts, inline = data.get("artifacts", []), data.get("inline_evidence", [])
    unavailable = data.get("unavailable", [])
    if (
        not isinstance(artifacts, list)
        or len(artifacts) > 16
        or not isinstance(inline, list)
        or len(inline) > 16
        or not isinstance(unavailable, list)
        or len(unavailable) > 16
    ):
        raise ValueError(
            "Task inventory needs at most 16 artifacts, inline items and unavailable items each"
        )
    result = host_io._result("task_context", config)
    result.update(
        manifest={"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        inline_evidence=[],
        unavailable=unavailable,
        validation={
            "native": "not_established",
            "numerical": "not_established",
            "performance": "not_established",
        },
        evidence_limit=(
            "Controller declarations and checked file bytes only. Inline evidence is"
            " not a file to discover; labels do not authenticate verifier claims or establish kernel acceptance."
        ),
    )
    result["coverage"].update(files_read=1, scan_bytes=len(raw))
    for item in inline:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or item.get("kind") not in KINDS
            or "value" not in item
        ):
            raise ValueError("Inline evidence needs id, kind and value")
        result["inline_evidence"].append(
            {**item, "storage": "inline", "authenticated": False}
        )
    if any(
        not isinstance(item, str) or not item or len(item) > 1000
        for item in unavailable
    ):
        raise ValueError(
            "Unavailable items must be bounded descriptions, not inferred absences"
        )
    seen = set()
    for index, item in enumerate(artifacts):
        if result["coverage"]["files_read"] >= config["limits"]["max_files"]:
            host_io._partial(result, "file_scan_budget")
            result["remaining_artifact_index"] = index
            break
        if (
            not isinstance(item, dict)
            or item.get("role") not in ROLES
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError(
                "Each declared artifact needs a role, path and expected SHA256"
            )
        target = host_io.checked_path(
            path.parent / item["path"], config, must_exist=False
        )
        if target in seen:
            raise ValueError("Duplicate declared artifact")
        seen.add(target)
        entry = {
            "path": str(target),
            "role": item["role"],
            "expected_sha256": item["sha256"],
            "sha256": None,
        }
        if not target.exists():
            entry["status"] = "missing"
        else:
            remaining = (
                config["limits"]["max_scan_bytes"] - result["coverage"]["scan_bytes"]
            )
            try:
                payload = host_io._bytes(
                    target, min(config["limits"]["max_file_bytes"], remaining)
                )
            except ValueError:
                entry["status"] = "unverified_read_limit"
            else:
                result["coverage"]["files_read"] += 1
                result["coverage"]["scan_bytes"] += len(payload)
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
                entry["status"] = (
                    "verified_bytes"
                    if entry["sha256"] == entry["expected_sha256"]
                    else "changed"
                )
        result["items"].append(entry)
        if entry["status"] != "verified_bytes":
            result["status"] = "unverified"
            result["coverage"]["complete"] = False
        if not host_io._fits(result):
            result["items"].pop()
            host_io._partial(result, "task_context_output_budget")
            result["remaining_artifact_index"] = index
            break
    # Check stability without a second unaccounted read of the manifest bytes.
    final_stamp = path.stat()
    if any(
        getattr(final_stamp, name) != getattr(stamp, name)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("Task manifest changed during inspection")
    actions.for_inventory(result, goal)
    host_io.encode(result)
    return result
