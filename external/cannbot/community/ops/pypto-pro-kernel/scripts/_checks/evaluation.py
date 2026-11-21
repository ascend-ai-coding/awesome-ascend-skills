# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small development/held-out tool tasks and comparable external agent observations.

This suite measures checker behavior, not model ability. Model observations must
come from an independent runner; absent token, time or correctness data stay null.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from . import feedback
from .paths import PYPTO_ROOT, SKILL_ROOT


def native_launch_geometry(values, *, workers, tasks):
    """Compare caller-observed default-stream launch values, not source spellings.

    The caller owns the version-guarded observation capability and provenance.
    This pure check does not import a runtime, access a device or prove execution.
    None means observation unavailable; an empty sequence means zero submissions.
    """
    if type(workers) is not int or workers < 1 or type(tasks) is not int or tasks < 1:
        raise ValueError("workers and tasks must be positive integers")
    result = {
        "status": "unverified",
        "expected_workers": workers,
        "expected_tasks": tasks,
        "scope": "Caller-observed requested default-stream launch values only; execution is not authenticated.",
    }
    if values is None:
        return result
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(v, (tuple, list)) or len(v) != 2 for v in values
    ):
        raise ValueError("launch observations must be (stream, worker_count) pairs")
    result["observed_tasks"] = len(values)
    result["status"] = (
        "matched"
        if len(values) == tasks
        and all(
            stream is None
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == workers
            for stream, count in values
        )
        else "mismatch"
    )
    return result


def bundle_identity(root):
    files = {
        str(path.relative_to(root)): feedback.digest(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".pyi", ".md"}
        and not {"__pycache__", ".pytest_cache"} & set(path.parts)
    }
    return feedback.digest(json.dumps(files, sort_keys=True).encode())


def event_metrics(path, *, host_counters=None):
    """Read observed Codex JSONL events without estimating tokenizer counts."""
    items, completed, usages, unknown = {}, set(), [], set()
    turns_started, turns_failed = 0, 0
    known_tools = {"command_execution", "mcp_tool_call", "file_change"}
    incomplete_lines = 0
    sessions, hook_notices, feature_notices = set(), set(), set()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            if position != len(lines) - 1:
                raise ValueError("Invalid JSON before the final event line") from error
            incomplete_lines += 1
            continue
        if not isinstance(event, dict):
            raise ValueError("Each event must be an object")
        if event.get("type") == "thread.started":
            sessions.add(event.get("thread_id"))
        if event.get("type") == "turn.started":
            turns_started += 1
        if event.get("type") == "turn.failed":
            turns_failed += 1
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError("Turn usage must be an object or null")
            usages.append(usage or {})
        if event.get("type") not in {"item.started", "item.completed"}:
            continue
        item = event.get("item", {})
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("An item event needs a string identity")
        ident, kind = item["id"], item.get("type")
        if kind == "error" and item.get("message") == (
            "`--dangerously-bypass-hook-trust` is enabled. Enabled hooks may run without review for this invocation."
        ):
            hook_notices.add(ident)
            continue
        if (
            kind == "error"
            and not items
            and isinstance(item.get("message"), str)
            and re.fullmatch(
                r"Under-development features enabled: [a-z][a-z0-9_]*(?:, [a-z][a-z0-9_]*)*\. "
                r"Under-development features are incomplete and may behave unpredictably\. "
                r"To suppress this warning, set `suppress_unstable_features_warning = true` in [^\r\n]+config\.toml\.",
                item.get("message", ""),
            )
        ):
            feature_notices.add(ident)
            continue
        if kind == "agent_message":
            continue
        if kind not in known_tools:
            unknown.add(str(kind))
            continue
        if ident in items and items[ident].get("type") != kind:
            raise ValueError("An item identity changed type")
        if ident in completed:
            if event["type"] == "item.completed" and item != items[ident]:
                raise ValueError("Conflicting completions for one item identity")
            continue
        items[ident] = item
        if event["type"] == "item.completed":
            completed.add(ident)

    def total(field):
        values = [usage.get(field) for usage in usages]
        if any(v is not None and (type(v) is not int or v < 0) for v in values):
            raise ValueError(f"Invalid host usage field {field}")
        return (
            sum(values)
            if values
            and turns_started == len(usages)
            and not turns_failed
            and not incomplete_lines
            and all(v is not None for v in values)
            else None
        )

    metrics = {
        field: total(field)
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    inputs, cached = metrics["input_tokens"], metrics["cached_input_tokens"]
    if inputs is not None and cached is not None and cached > inputs:
        raise ValueError("Cached input tokens exceed total input tokens")
    metrics["uncached_input_tokens"] = (
        inputs - cached if inputs is not None and cached is not None else None
    )
    counts = Counter(item["type"] for item in items.values())
    commands = [
        item
        for ident, item in items.items()
        if ident in completed and item["type"] == "command_execution"
    ]
    outputs = [item.get("aggregated_output") for item in commands]
    metrics.update(
        {
            "visible_actions": len(items) if not unknown else None,
            "tool_calls": None,
            "tool_completions": len(completed),
            "pending_tools": len(items) - len(completed),
            "shell_calls": counts["command_execution"],
            "file_edits": counts["file_change"],
            "mcp_calls": counts["mcp_tool_call"],
            "command_output_bytes": sum(len(s.encode()) for s in outputs)
            if all(isinstance(s, str) for s in outputs)
            else None,
            "context_tokens": None,
        }
    )
    counter_identity = None
    host_coverage = "not_requested"
    if host_counters is not None:
        data = json.loads(Path(host_counters).read_text())
        if (
            not isinstance(data, dict)
            or data.get("schema") != "pypto-host-counters/1"
            or len(sessions) != 1
            or data.get("session_id") not in sessions
            or any(
                type(data.get(key)) is not int or data[key] < 0
                for key in ("calls", "denied", "rewritten")
            )
            or data["denied"] + data["rewritten"] > data["calls"]
            or data["calls"] - data["denied"] < counts["command_execution"]
        ):
            raise ValueError(
                "Host counters do not match this session and its visible commands"
            )
        counter_identity = feedback.file_identity(host_counters)
        metrics.update(
            host_shell_attempts=data["calls"],
            host_read_refusals=data["denied"],
            host_context_rewrites=data["rewritten"],
        )
        if metrics["visible_actions"] is not None:
            metrics["visible_actions"] += data["calls"] - counts["command_execution"]
        host_coverage = "observed"
    elif hook_notices:
        # This host can omit pre-tool refusals entirely from JSONL.
        metrics["visible_actions"] = None
        host_coverage = "unverified_missing_counters"
    metrics["host_notices"] = len(hook_notices) + len(feature_notices)
    return {
        "schema": "pypto-agent-events/1",
        "input": feedback.file_identity(path),
        "status": "complete"
        if usages
        and turns_started == len(usages)
        and not turns_failed
        and not unknown
        and not incomplete_lines
        and not metrics["pending_tools"]
        and host_coverage != "unverified_missing_counters"
        else "incomplete",
        "metrics": metrics,
        "unknown_item_types": sorted(unknown),
        "incomplete_event_lines": incomplete_lines,
        "host_counters": counter_identity,
        "host_attempt_coverage": host_coverage,
        "scope": (
            "Visible actions count command/edit/MCP identities once; matching host"
            " counters also include Bash attempts omitted before execution. Recognized"
            " hook-trust and pre-tool feature notices are not actions. Without those"
            " counters hook-enabled attempt coverage remains unknown. Codex JSONL does"
            " not expose every session-poll API call, so total tool_calls remain"
            " unknown. Host input tokens are cumulative across requests and include"
            " cached input; peak/unique context tokens remain unknown. Command output"
            " bytes measure logged visible text only. No numerical or direction correctness is inferred."
        ),
    }


def prepare_agent_task(directory):
    fixture = Path(__file__).parent / "tests/data/ub_l1_repair.py"
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "candidate.py"
    target.write_bytes(fixture.read_bytes())
    return {
        "task_id": "ub-l1-repair",
        "task_sha256": feedback.file_identity(fixture)["sha256"],
        "source": str(target.resolve()),
        "prompt": "Repair this native handover to compute ((x + 1) rounded to FP16) @ rhs into FP32 out. "
        "Preserve its public callable and ROWS/DIRECT tiling keys. ROWS is 8, 16 or 32; DIRECT is 0 or 1. "
        "x has shape [2*ROWS,128], rhs [128,16], out [2*ROWS,16]. Preserve disjoint subblock ownership "
        "and cross-core correctness. Edit only candidate.py; use the assigned isolated skill/examples bundle. "
        "A local source-check pass is not native verification; the independent grader checks every specialization.",
    }


def _source(
    *,
    rows=32,
    source_layout="NZ",
    destination="Mat",
    alias="pl",
    api="insert",
    unknown=False,
):
    source = f"""import pypto_pro.language as {alias}
@{alias}.jit(auto_mutex=False)
def candidate():
    src = {alias}.make_tile({alias}.TileType(shape=[{rows}, 128], dtype={alias}.DT_FP16, target_memory={alias}.MemorySpace.Vec, layout={alias}.{source_layout}), addr=0)
    dst = {alias}.make_tile({alias}.TileType(shape=[{rows}, 128], dtype={alias}.DT_FP16, target_memory={alias}.MemorySpace.{destination}, layout={alias}.NZ), addr=16384)
"""
    if unknown:
        source += "    src = object()\n"
    return (
        source
        + f"    {alias}.{api}(dst, src"
        + (", [0, 0])\n" if api == "insert" else ")\n")
    )


def _handover_tasks():
    return [
        {
            "id": "nd-handover",
            "split": "development",
            "source": _source(source_layout="ND"),
            "expected_rule": "ub-l1-layout-mismatch",
            "expected_exit": 0,
        },
        {
            "id": "odd-conversion",
            "split": "development",
            "source": _source(
                rows=33, source_layout="ND", destination="Vec", api="move"
            ),
            "expected_rule": "ub-nz-move-pitch",
            "expected_exit": 1,
        },
        {
            "id": "aligned-conversion",
            "split": "development",
            "source": _source(source_layout="ND", destination="Vec", api="move"),
            "expected_rule": None,
            "expected_exit": 0,
        },
        {
            "id": "compact-insert",
            "split": "development",
            "source": _source(),
            "expected_rule": None,
            "expected_exit": 0,
        },
        {
            "id": "odd-insert",
            "split": "development",
            "source": _source(rows=33),
            "expected_rule": None,
            "expected_exit": 0,
        },
        {
            "id": "aliased-nd-handover",
            "split": "heldout",
            "source": _source(alias="dsl", source_layout="ND", api="move"),
            "expected_rule": "ub-l1-layout-mismatch",
            "expected_exit": 0,
        },
        {
            "id": "different-odd-conversion",
            "split": "heldout",
            "source": _source(
                rows=49, alias="dsl", source_layout="ND", destination="Vec", api="move"
            ),
            "expected_rule": "ub-nz-move-pitch",
            "expected_exit": 1,
        },
        {
            "id": "different-aligned-conversion",
            "split": "heldout",
            "source": _source(
                rows=48, source_layout="ND", destination="Vec", api="move"
            ),
            "expected_rule": None,
            "expected_exit": 0,
        },
        {
            "id": "full-nz-move",
            "split": "heldout",
            "source": _source(api="move"),
            "expected_rule": None,
            "expected_exit": 0,
        },
        {
            "id": "unknown-producer",
            "split": "heldout",
            "source": _source(source_layout="ND", unknown=True),
            "expected_rule": None,
            "expected_exit": 0,
        },
    ]


def run(*, checker=None, split="all", pypto_root=PYPTO_ROOT):
    checker = (checker or SKILL_ROOT / "scripts/check.py").resolve()
    selected = [task for task in _handover_tasks() if split == "all" or task["split"] == split]
    result = feedback.envelope("tool_evaluation", pypto_root)
    result["evaluated_checker"] = feedback.file_identity(checker)
    result["evaluated_skill_sha256"] = bundle_identity(checker.parent.parent)
    result["task_set_sha256"] = feedback.digest(
        json.dumps(selected, sort_keys=True).encode()
    )
    result["results"] = []
    with tempfile.TemporaryDirectory(prefix="ub-l1-evaluation-") as directory:
        for task in selected:
            source = Path(directory) / (task["id"] + ".py")
            source.write_text(task["source"], encoding="utf-8")
            start = time.monotonic()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(checker),
                    "--pypto-root",
                    str(pypto_root),
                    "rules",
                    str(source),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            matched = set(
                re.findall(
                    r"\b(?:ub-l1-layout-mismatch|ub-nz-move-pitch)\b", proc.stdout
                )
            )
            expected = {task["expected_rule"]} if task["expected_rule"] else set()
            result["results"].append(
                {
                    "id": task["id"],
                    "split": task["split"],
                    "source_sha256": feedback.file_identity(source)["sha256"],
                    "passed": proc.returncode == task["expected_exit"]
                    and matched == expected,
                    "exit_code": proc.returncode,
                    "matched_rules": sorted(matched),
                    "wall_seconds": time.monotonic() - start,
                    "output_bytes": len((proc.stdout + proc.stderr).encode()),
                    "context_tokens": None,
                }
            )
    result["passed"] = sum(row["passed"] for row in result["results"])
    result["total"] = len(selected)
    result["status"] = "passed" if result["passed"] == result["total"] else "failed"
    result["agent_effectiveness"] = "not_established"
    return result


def _cost_eligibility(baseline, candidate, metrics):
    reasons, rejected = [], False
    for variant, row in (("baseline", baseline), ("candidate", candidate)):
        for field in ("correct", "budget_met", "valid_protocol"):
            if row.get(field) is not True:
                rejected |= row.get(field) is False
                reasons.append(
                    f"{variant}.{field}="
                    + ("false" if row.get(field) is False else "unknown")
                )
        if row.get("direction_correct") is False:
            rejected = True
            reasons.append(f"{variant}.direction_correct=false")
        if row.get("telemetry_status") != "complete":
            reasons.append(
                f"{variant}.telemetry_status="
                + (row.get("telemetry_status") or "unknown")
            )
    return {
        "status": "ineligible" if rejected else "unverified" if reasons else "eligible",
        "reasons": reasons,
        "available_metrics": [
            m
            for m in metrics
            if m not in {"correct", "direction_correct", "budget_met", "valid_protocol"}
            and baseline.get(m) is not None
            and candidate.get(m) is not None
        ],
    }


def compare_observations(path):
    """Summarize externally observed matched trials without imputing missing metrics."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Agent observations must be a nonempty JSON array")
    keys = (
        "task_id",
        "task_sha256",
        "model",
        "reasoning",
        "budget",
        "toolchain",
        "repeat",
    )
    metrics = (
        "correct",
        "direction_correct",
        "budget_met",
        "valid_protocol",
        "repeated_errors",
        "tool_calls",
        "visible_actions",
        "context_tokens",
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "command_output_bytes",
        "native_requests",
        "native_starts",
        "device_requests",
        "wall_seconds",
        "latency_us",
        "host_shell_attempts",
        "host_read_refusals",
        "host_context_rewrites",
    )
    groups = {}
    for row in data:
        if any(
            k not in row for k in keys + ("variant", "skill_sha256", "source_sha256")
        ) or row["variant"] not in {"baseline", "candidate"}:
            raise ValueError(
                "Each observation needs task/model/reasoning/budget/toolchain/repeat,"
                " variant and skill/source identities"
            )
        for key in ("task_sha256", "skill_sha256", "source_sha256"):
            if not isinstance(row[key], str) or not re.fullmatch(
                r"[0-9a-f]{64}", row[key]
            ):
                raise ValueError(f"Invalid {key}")
        for key in metrics:
            val = row.get(key)
            if key in {"correct", "direction_correct", "budget_met", "valid_protocol"}:
                valid = val is None or isinstance(val, bool)
            else:
                valid = (
                    val is None
                    or type(val) in (int, float)
                    and math.isfinite(val)
                    and val >= 0
                )
            if not valid:
                raise ValueError(f"Invalid observed metric {key}")
        if row.get("telemetry_status") not in (
            None,
            "complete",
            "incomplete",
            "invalid",
        ):
            raise ValueError("Invalid telemetry_status")
        group = groups.setdefault(
            json.dumps([row[k] for k in keys], sort_keys=True), {}
        )
        if row["variant"] in group:
            raise ValueError("Duplicate trial identity and variant")
        group[row["variant"]] = row
    pairs = []
    for key, group in groups.items():
        if set(group) != {"baseline", "candidate"}:
            pairs.append({"identity": json.loads(key), "matched": False})
            continue
        baseline, candidate = group["baseline"], group["candidate"]
        pairs.append(
            {
                "identity": json.loads(key),
                "matched": True,
                "cost_comparison": _cost_eligibility(baseline, candidate, metrics),
                "observations": {
                    m: {"baseline": baseline.get(m), "candidate": candidate.get(m)}
                    for m in metrics
                },
            }
        )
    return {
        "schema": "pypto-agent-comparison/1",
        "input": feedback.file_identity(path),
        "pairs": pairs,
        "scope": (
            "External observations, not inferred from checker success. Missing metrics"
            " remain null; a pilot does not establish general model effectiveness."
        ),
    }
