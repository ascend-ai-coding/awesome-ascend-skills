# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect and parse external Ascend profiler captures without importing PyPTO.

The module provides optional ``msprof`` (application level) and ``msprof op`` /
``msopprof`` (operator level) collection on top of the torch_npu path in
``profile_kernel.py``. Parsing is pure and offline: it accepts capture paths and
never imports ``torch_npu``. Missing evidence stays missing; ambiguous or
unattributable rows fail closed instead of being guessed.

Every capture writes ``collection.json`` next to the raw output. It records the
collector that actually ran, its version, the invoked argv, the target device and
the requested scope; ``summarize`` reads identity from that file rather than from
the parsing environment. A capture without it keeps its identity unknown.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_OUT = logging.getLogger("pypto.perf_collect.stdout")
_OUT.setLevel(logging.INFO)
_OUT.propagate = False
if not _OUT.handlers:
    _out_handler = logging.StreamHandler(sys.stdout)
    _out_handler.setFormatter(logging.Formatter("%(message)s"))
    _OUT.addHandler(_out_handler)

_ERR = logging.getLogger("pypto.perf_collect.stderr")
_ERR.setLevel(logging.ERROR)
_ERR.propagate = False
if not _ERR.handlers:
    _err_handler = logging.StreamHandler(sys.stderr)
    _err_handler.setFormatter(logging.Formatter("%(message)s"))
    _ERR.addHandler(_err_handler)


def _emit(text):
    _OUT.info("%s", text)


def _emit_error(text):
    _ERR.error("%s", text)


SCHEMA = "pypto-perf-evidence/1"
COLLECTION_SCHEMA = "pypto-perf-collection/1"
PROBE_SCHEMA = "pypto-perf-probe/1"
METRICS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)
OPPROF_METRICS = ("OpBasicInfo",) + METRICS
_OPPROF_CSV = {
    metric: re.compile(rf"^{re.escape(metric)}(_[0-9][0-9A-Za-z_.\-]*)?\.csv$")
    for metric in OPPROF_METRICS
}

_MISSING = {"", "NA", "N/A", "NULL", "NONE", "NAN", "INF", "+INF", "-INF", "-"}
_NAME_KEYS = ("Op Name", "Kernel Name", "Name")
_TYPE_KEYS = ("Task Type", "Type", "Op Type")
_DURATION_KEYS = ("Task Duration(us)", "Duration(us)", "Task Duration")


class PerfCollectError(RuntimeError):
    """A capture, attribution or parsing boundary failed."""


def _text(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text.upper() in _MISSING else text


def _number(value):
    text = _text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise PerfCollectError(f"unsupported numeric value: {value!r}") from exc
    return number if math.isfinite(number) else None


def _first(row, keys):
    for key in keys:
        if key in row and _text(row[key]) is not None:
            return row[key]
    return None


def _duration(row):
    raw = _first(row, _DURATION_KEYS)
    value = _number(raw)
    if value is None:
        raise PerfCollectError(f"row has no positive task duration: {sorted(row)}")
    if value <= 0:
        raise PerfCollectError(f"task duration must be positive, got {value}")
    return value


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        return list(csv.DictReader(stream))


def _one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        return None
    if len(hits) != 1:
        raise PerfCollectError(f"expected one capture for {pattern}, found {len(hits)}: {hits}")
    return hits[0]


def _ascend_home():
    for name in ("ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return Path("/usr/local/Ascend/ascend-toolkit/latest")


def _binary(name, relative):
    found = shutil.which(name)
    if found:
        return Path(found)
    home = _ascend_home()
    for base in (home, home / "tools"):
        candidate = base / relative
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _version(path):
    try:
        done = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [
        line.strip()
        for line in (done.stdout or done.stderr or "").strip().splitlines()
        if line.strip() and not line.strip().lower().startswith("usage")
    ]
    patterns = (r"^(?:msprof|msopprof)\s+\d", r"^version\b", r"^\d+\.\d+")
    return next(
        (line for line in lines if any(re.match(p, line, re.I) for p in patterns)),
        None,
    )


def _info_version(name):
    """Read an installed package version when the binary exposes no version flag."""
    path = _ascend_home() / "share" / "info" / name / "version.info"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "version" and value.strip():
            return f"{name} {value.strip()}"
    return None


def _binary_version(name, path):
    version = _version(path) if path else None
    return version if version else _info_version(name)


def probe(env=None):
    """Report collector availability and identity without touching a device.

    ``env`` is an optional overlay for ``PATH`` and ``ASCEND_*`` when probing an
    alternate installation. It applies only for the duration of the call and is
    restored afterwards, so probing never rewrites the process environment.
    """
    previous = {}
    for key, value in (env or {}).items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        home = _ascend_home()
        collectors = {}
        for name, relative in (
            ("msprof", Path("profiler/bin/msprof")),
            ("msopprof", Path("msopprof/bin/msopprof")),
        ):
            path = _binary(name, relative)
            collectors[name] = {
                "path": str(path) if path else None,
                "version": _binary_version(name, path),
                "available": bool(path),
            }
        return {
            "schema": PROBE_SCHEMA,
            "ascend_home": str(home),
            "arch": os.environ.get("NPU_ARCH") or os.environ.get("ASCEND_SOC_VERSION") or None,
            "collectors": collectors,
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scan_op_summaries(root):
    paths = glob.glob(str(Path(root, "**", "op_summary_*.csv")), recursive=True)
    return sorted(path for path in paths if f"{os.sep}PROF_Sample{os.sep}" not in path)


def _op_row(row, source):
    name = _text(_first(row, _NAME_KEYS))
    if not name:
        return None
    duration = _number(_first(row, _DURATION_KEYS))
    if duration is None or duration <= 0:
        return None
    return {
        "op_name": name,
        "op_type": _text(_first(row, _TYPE_KEYS)) or "",
        "task_duration_us": duration,
        "source": str(Path(source).resolve()),
    }


def _summary_rows(capture):
    rows = []
    for path in _scan_op_summaries(capture):
        for row in _rows(path):
            item = _op_row(row, path)
            if item:
                rows.append(item)
    return rows


def _operator_dir_name(directory):
    """Operator name for a capture without ``OpBasicInfo``: the launch index is skipped."""
    parent = directory.parent
    if directory.name.isdigit() and parent != directory:
        return parent.name
    return directory.name


def _opprof_dir_row(directory):
    """Describe an operator directory that has no ``OpBasicInfo`` CSV.

    The operator name is the mangled directory name; the duration comes from any
    other metric CSV and stays ``0.0`` when no row carries one, so ``discover``
    still lists the operator instead of reporting an empty capture.
    """
    for metric in METRICS:
        path = _metric_csv(directory, metric)
        if path is None:
            continue
        for row in _rows(path):
            duration = _number(_first(row, _DURATION_KEYS))
            if duration is not None and duration > 0:
                return {
                    "op_name": _operator_dir_name(directory),
                    "op_type": _text(_first(row, _TYPE_KEYS)) or "",
                    "task_duration_us": duration,
                    "source": str(Path(path).resolve()),
                }
    return {
        "op_name": _operator_dir_name(directory),
        "op_type": "",
        "task_duration_us": 0.0,
        "source": str(Path(directory).resolve()),
    }


def _opprof_op_rows(capture):
    rows = []
    for directory in _opprof_directories(capture, required=False):
        path = _metric_csv(directory, "OpBasicInfo")
        if path is None:
            rows.append(_opprof_dir_row(directory))
            continue
        for row in _rows(path):
            item = _op_row(row, path)
            if item:
                rows.append(item)
    return rows


def list_op_names(capture):
    """List exact Op Name values from one capture, application or operator level."""
    rows = _summary_rows(capture)
    if not rows and Path(capture).is_dir():
        rows = _opprof_op_rows(capture)
    if not rows:
        raise PerfCollectError(
            f"no op_summary_*.csv or OpBasicInfo rows found under {capture}"
        )
    grouped = {}
    for row in rows:
        key = (row["op_name"], row["op_type"])
        item = grouped.setdefault(
            key, {"op_name": key[0], "op_type": key[1], "task_duration_us": 0.0, "rows": 0}
        )
        item["task_duration_us"] = max(item["task_duration_us"], row["task_duration_us"])
        item["rows"] += 1
    return sorted(
        grouped.values(), key=lambda item: (-item["task_duration_us"], item["op_name"])
    )


def _pick(rows, op_name, source):
    matched = [row for row in rows if _text(_first(row, _NAME_KEYS)) == op_name]
    if not matched:
        names = sorted({_text(_first(row, _NAME_KEYS)) for row in rows} - {None})
        raise PerfCollectError(
            f"{source}: op {op_name!r} not found; candidates: {names[:20]}"
        )
    if len(matched) != 1:
        raise PerfCollectError(
            f"{source}: op {op_name!r} matched {len(matched)} rows; "
            "collect one measured launch or isolate the case"
        )
    return matched[0]


def _resolve_group(capture):
    root = Path(capture)
    if any((root / f"PROF_{metric}").is_dir() for metric in METRICS) or (root / "PROF_Sample").is_dir():
        return root
    hits = sorted(glob.glob(str(root / "PROF_GROUP_*")))
    if len(hits) == 1:
        return Path(hits[0])
    if len(hits) > 1:
        raise PerfCollectError(
            f"{root}: multiple PROF_GROUP_* captures; pass one explicitly: {hits}"
        )
    return root


def parse_msprof_group(capture, op_name):
    """Parse one PROF_GROUP_* capture for a single exact Op Name."""
    capture = _resolve_group(capture)
    groups, missing, merged = {}, [], {}
    for metric in METRICS:
        metric_dir = Path(capture, f"PROF_{metric}")
        if not metric_dir.is_dir():
            missing.append(metric)
            continue
        path = _one(str(metric_dir / "**" / "mindstudio_profiler_output" / "op_summary_*.csv"))
        if path is None:
            missing.append(metric)
            continue
        rows = _rows(path)
        row = _pick(rows, op_name, path)
        groups[metric] = {"source": str(Path(path).resolve()), "row": row}
        for key, value in row.items():
            if key in (None, ""):
                continue
            current = merged.get(key)
            if current is None or current in _MISSING:
                merged[key] = value
    if not groups:
        raise PerfCollectError(f"{capture}: no metric group matched op {op_name!r}")
    duration = _duration(merged)
    return {
        "op_name": op_name,
        "task_duration_us": duration,
        "groups": groups,
        "merged": merged,
        "missing": missing,
    }


def _cycle_rows(path, family):
    try:
        connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return None, f"unavailable:sqlite_error:{exc}"
    try:
        if family == "vector":
            queries = (
                "SELECT coreid, SUM(task_cyc) FROM EventCount "
                "WHERE task_cyc>0 GROUP BY coreid ORDER BY coreid",
                "SELECT coreid, SUM(pmucount) FROM task_cyc "
                "WHERE pmucount>0 GROUP BY coreid ORDER BY coreid",
            )
        else:
            queries = (
                "SELECT coreid, SUM(CAST(task_cyc AS INTEGER)) FROM AICoreOriginalData "
                "WHERE CAST(task_cyc AS INTEGER)>0 GROUP BY coreid ORDER BY coreid",
                "SELECT coreid, SUM(pmucount) FROM task_cyc "
                "WHERE pmucount>0 GROUP BY coreid ORDER BY coreid",
            )
        for query in queries:
            try:
                rows = list(connection.execute(query))
            except sqlite3.Error:
                continue
            cycles = [
                (int(core), int(value))
                for core, value in rows
                if core is not None and value is not None
            ]
            if cycles:
                return cycles, None
        return [], None
    finally:
        connection.close()


def parse_aicore_db(capture, core_time_us, family=None):
    """Convert sample-based per-core cycles to time; scope stays process-level.

    ``family`` is ``cube`` or ``vector`` and is resolved from the task type and the
    populated time field before this call, so cycles and reference time always come
    from the same execution family. Without it Cube is tried first, then Vector and
    the fallback can mix families; callers that know the family must pass it.
    """
    capture = _resolve_group(capture)
    patterns = {"cube": "aicore.db", "vector": "ai_vector_core.db"}
    order = (family,) if family in patterns else ("cube", "vector")
    reason = "unavailable:no_aicore_db"
    for candidate in order:
        pattern = patterns.get(candidate)
        if pattern is None:
            continue
        hits = sorted(
            glob.glob(
                str(Path(capture, "PROF_Sample", "PROF_*", "device_*", "sqlite", pattern))
            )
        )
        if not hits:
            if len(order) == 1:
                reason = f"unavailable:no_{candidate}_db"
            continue
        if len(hits) != 1:
            reason = f"unavailable:ambiguous_{candidate}_db:{len(hits)}"
            continue
        cycles, error = _cycle_rows(hits[0], candidate)
        if error:
            reason = error
            continue
        if not cycles:
            reason = f"unavailable:no_positive_{candidate}_cycles"
            continue
        if core_time_us is None or core_time_us <= 0:
            return None, "unavailable:no_attributable_core_time"
        return _core_time_result(candidate, cycles, core_time_us)
    return None, reason


def _core_time_result(family, cycles, core_time_us):
    max_cycles = max(value for _, value in cycles)
    ns_per_cycle = core_time_us * 1000.0 / max_cycles
    times = [(core, value * ns_per_cycle / 1000.0) for core, value in cycles]
    values = [time for _, time in times]
    maximum, minimum = max(values), min(values)
    return (
        {
            "scope": "process_scope_unattributed",
            "family": family,
            "ns_per_cycle": ns_per_cycle,
            "freq_ghz": 1.0 / ns_per_cycle,
            "valid_cores": len(times),
            "imbalance_ratio_pct": (maximum - minimum) / maximum * 100.0,
            "cores": [
                {"coreid": core, "cycles": value, "time_us": time}
                for (core, value), (_, time) in zip(cycles, times)
            ],
        },
        None,
    )


def _metric_csv(directory, metric):
    """Return the single CSV for one OPPROF metric, timestamp suffix accepted."""
    pattern = _OPPROF_CSV[metric]
    hits = sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and pattern.match(path.name)
    )
    if len(hits) > 1:
        raise PerfCollectError(
            f"{directory}: multiple {metric} CSVs; keep one capture: "
            f"{[path.name for path in hits]}"
        )
    return hits[0] if hits else None


def _opprof_csv_parents(root):
    """Directories under ``root`` that hold a recognized OPPROF metric CSV."""
    parents = set()
    for metric, pattern in _OPPROF_CSV.items():
        for path in root.glob(f"**/{metric}*.csv"):
            if pattern.match(path.name):
                parents.add(path.parent)
    return parents


def _opprof_directories(capture, *, required=True):
    """Return the per-operator directories that hold at least one OPPROF metric CSV.

    ``OpBasicInfo*.csv`` is optional: some collectors omit it. A directory whose
    other metric CSVs are present still parses, with the operator name taken from
    the directory and the task duration possibly unknown.
    """
    root = Path(capture)
    if not root.is_dir():
        raise PerfCollectError(f"{capture}: capture directory not found")
    directories = sorted(_opprof_csv_parents(root))
    if not directories and required:
        raise PerfCollectError(f"{capture}: no OPPROF metric CSV found")
    return directories


def _opprof_matches(directory, op_name):
    """Match the CSV ``Op Name`` or the operator directory (mangled symbol).

    Real captures nest ``<kernel>/<launch index>/OpBasicInfo*.csv``, so the mangled
    symbol is the parent directory of the CSV directory. Without ``OpBasicInfo``
    only the directory name can identify the operator.
    """
    if op_name in (directory.name, directory.parent.name):
        return True
    path = _metric_csv(directory, "OpBasicInfo")
    if path is None:
        return False
    return any(
        _text(_first(row, _NAME_KEYS)) == op_name for row in _rows(path)
    )


def _select_opprof_directory(capture, directories, op_name):
    if op_name is None:
        if len(directories) == 1:
            return directories[0]
        raise PerfCollectError(
            f"{capture}: {len(directories)} operator captures; pass --op-name or one "
            f"directory: {[str(item) for item in directories[:10]]}"
        )
    # Validate the requested name even for a single capture: without an
    # ``OpBasicInfo`` CSV the row cannot reject a mismatch, so the directory is the
    # only identity and a wrong ``--op-name`` must not return another operator.
    matched = [item for item in directories if _opprof_matches(item, op_name)]
    if len(matched) != 1:
        raise PerfCollectError(
            f"{capture}: op {op_name!r} matched {len(matched)} operator captures; "
            f"candidates: {[str(item) for item in directories[:10]]}"
        )
    return matched[0]


def _opprof_row(root, op_name):
    path = _metric_csv(root, "OpBasicInfo")
    if path is None:
        return {}
    basic = _rows(path)
    if not basic:
        raise PerfCollectError(f"{path}: no rows")
    if op_name is None:
        if len(basic) != 1:
            raise PerfCollectError(f"{path}: multiple ops; pass a kernel name")
        return basic[0]
    if op_name in (root.name, root.parent.name) and len(basic) == 1:
        return basic[0]
    return _pick(basic, op_name, str(path))


def _opprof_groups(root):
    groups, missing = {}, []
    for metric in OPPROF_METRICS:
        path = _metric_csv(root, metric)
        if path is None:
            if metric != "OpBasicInfo":
                missing.append(metric)
            continue
        groups[metric] = {"source": str(path.resolve()), "rows": _rows(path)}
    return groups, missing


def _opprof_duration(groups, notes):
    """First positive task duration from another metric CSV, or ``None``."""
    ordered = ("PipeUtilization",) + tuple(
        item for item in METRICS if item != "PipeUtilization"
    )
    for metric in ordered:
        for item in groups.get(metric, {}).get("rows", []):
            value = _number(_first(item, _DURATION_KEYS))
            if value is not None and value > 0:
                return value
    notes.append("no_task_duration")
    return None


def _opprof_per_core(pipe_rows):
    """Per-core rows from ``PipeUtilization.csv``; ``None`` when there are none."""
    per_core = []
    for item in pipe_rows:
        cycles = _number(item.get("aic_total_cycles")) or _number(item.get("aiv_total_cycles"))
        time_us = _number(item.get("aic_time(us)")) or _number(item.get("aiv_time(us)"))
        per_core.append(
            {
                "block_id": _text(item.get("block_id")),
                "sub_block_id": _text(item.get("sub_block_id")),
                "cycles": cycles,
                "time_us": time_us,
            }
        )
    return per_core or None


def parse_opprof(capture, op_name=None):
    """Parse one OPPROF capture into metric rows and per-core samples.

    Real collectors nest one directory per operator and suffix every CSV with a
    timestamp (``OPPROF_*/<kernel>/0/OpBasicInfo_<ts>.csv``). Flat
    ``OPPROF_*/OpBasicInfo.csv`` captures remain accepted. ``OpBasicInfo`` itself
    is optional: when it is absent the operator name comes from the directory, the
    duration is derived from another metric row when possible, and the gap is
    recorded as a note instead of rejecting the capture.
    """
    directories = _opprof_directories(capture)
    root = _select_opprof_directory(capture, directories, op_name)
    row = _opprof_row(root, op_name)
    groups, missing = _opprof_groups(root)
    notes = []
    if _metric_csv(root, "OpBasicInfo") is None:
        notes.append("no_op_basic_info")
    name = _text(_first(row, _NAME_KEYS)) if row else None
    if name is None:
        name = _text(_operator_dir_name(root))
    duration = _duration(row) if row else _opprof_duration(groups, notes)
    return {
        "op_name": name,
        "task_duration_us": duration,
        "basic": row,
        "groups": groups,
        "missing": missing,
        "notes": notes,
        "per_core": _opprof_per_core(groups.get("PipeUtilization", {}).get("rows", [])),
    }


class _ParsedParts(NamedTuple):
    parsed: dict
    groups: dict
    per_core: object
    reason: object
    note: object
    attribution: str


def _msprof_parts(root, op_name):
    parsed = parse_msprof_group(root, op_name)
    merged = parsed["merged"]
    task_type = _text(_first(merged, _TYPE_KEYS)) or ""
    cube_time = _number(merged.get("aicore_time(us)")) or 0.0
    vector_time = _number(merged.get("aiv_time(us)")) or 0.0
    if "VECTOR" in task_type.upper() or (vector_time > 0 and cube_time <= 0):
        family, core_time = "vector", vector_time
    elif cube_time > 0:
        family, core_time = "cube", cube_time
    else:
        # Neither time field nor the task type identifies the family. Keep the
        # documented cube-then-vector fallback and record that the family was
        # guessed, instead of forcing ``cube`` and dropping a usable vector db.
        family, core_time = None, 0.0
    notes = []
    if core_time <= 0:
        core_time = parsed["task_duration_us"]
        notes.append("per_core_uses_task_duration")
    if family is None:
        notes.append("family_unresolved")
    per_core, reason = parse_aicore_db(root, core_time, family)
    groups = {
        name: {"source": item["source"], "row": item["row"]}
        for name, item in parsed["groups"].items()
    }
    return _ParsedParts(parsed, groups, per_core, reason, notes, "single_target")


def _opprof_parts(root, op_name):
    parsed = parse_opprof(root, op_name)
    per_core = parsed["per_core"]
    attribution = "single_target" if op_name or parsed["op_name"] else "unverified"
    groups = {
        name: {"source": item["source"], "rows": item["rows"]}
        for name, item in parsed["groups"].items()
    }
    return _ParsedParts(parsed, groups, per_core, None, parsed["notes"], attribution)


def _parse_environment():
    return {
        "arch": os.environ.get("NPU_ARCH") or os.environ.get("ASCEND_SOC_VERSION") or None,
        "soc": os.environ.get("ASCEND_SOC_VERSION") or None,
        "cann": os.environ.get("ASCEND_HOME_PATH") or None,
        "ascend_home": str(_ascend_home()),
    }


def _coverage(parsed, groups, collection):
    """Separate requested-but-absent groups from groups that were never requested."""
    requested = [item for item in (collection or {}).get("metrics") or [] if item in METRICS]
    if not requested:
        return {
            "requested": None,
            "collected": sorted(groups),
            "missing": list(parsed["missing"]),
            "not_requested": None,
        }
    requested_set = set(requested)
    return {
        "requested": [item for item in METRICS if item in requested_set],
        "collected": sorted(groups),
        "missing": [
            item for item in METRICS if item in requested_set and item not in groups
        ],
        "not_requested": [item for item in METRICS if item not in requested_set],
    }


def _collection_status(collection):
    """Capture status: a recorded stage error is a failure, not a complete run."""
    if not collection:
        return "unknown"
    return "failed" if collection.get("error") else "complete"


def _evidence_notes(collection, note, reason):
    """Merge the parser notes, the per-core reason and the capture notes."""
    notes = []
    for item in (note, reason):
        if isinstance(item, (list, tuple)):
            notes.extend(entry for entry in item if entry)
        elif item:
            notes.append(item)
    notes.extend(collection.get("notes") or [])
    return notes


def _evidence_identity(parsed, collection, backend, requested_name):
    """Capture-time identity; the parsed name is used only as the kernel fallback."""
    identity = collection.get("identity") or {}
    kernel_name = collection.get("kernel_name")
    if kernel_name is None and backend != "msprof":
        kernel_name = requested_name
    return {
        "arch": identity.get("arch"),
        "soc": identity.get("soc"),
        "cann": identity.get("cann"),
        "toolkit_version": identity.get("toolkit_version"),
        "toolkit_innerversion": identity.get("toolkit_innerversion"),
        "op_name": parsed["op_name"],
        "kernel_name": kernel_name,
        "source": "collection" if collection else "unknown",
    }


def _evidence_envelope(
    backend, parsed, collection, warm_up, groups, per_core, reason, note,
    attribution, requested_name,
):
    collection = collection or {}
    collector = collection.get("collector") or {}
    coverage = _coverage(parsed, groups, collection)
    coverage["notes"] = _evidence_notes(collection, note, reason)
    return {
        "schema": SCHEMA,
        "collector": {
            "backend": backend,
            "path": collector.get("path"),
            "version": collector.get("version"),
        },
        "collection": {
            "status": _collection_status(collection),
            "error": collection.get("error"),
            "stages": collection.get("stages") or [],
        },
        "identity": _evidence_identity(parsed, collection, backend, requested_name),
        "parse_environment": _parse_environment(),
        "scope": {
            "metric_groups": sorted(groups),
            "warm_up": collection.get("warm_up", warm_up),
            "measured_runs": collection.get("launch_count"),
            "attribution": attribution,
        },
        "metrics": {
            "task_duration_us": {
                "value": parsed["task_duration_us"],
                "samples": 1 if parsed["task_duration_us"] is not None else 0,
            },
            "groups": groups,
            "per_core": per_core,
        },
        "coverage": coverage,
    }


def _load_collection(capture_dir):
    """Read the metadata owned by one capture; a sibling capture is never used."""
    path = Path(capture_dir) / "collection.json"
    return _read_collection(path) if path.is_file() else None


def _owning_capture(root, target=None):
    """Directory whose ``collection.json`` describes ``target``.

    The walk stops at the enclosing ``OPPROF_*`` / ``PROF_GROUP_*`` directory. A
    common output root that holds several captures therefore cannot lend its single
    metadata file to a capture that was written without one.
    """
    start = Path(target if target is not None else root).resolve()
    for candidate in (start, *start.parents):
        if candidate.name.startswith(("OPPROF_", "PROF_GROUP_")):
            return candidate
    return Path(root).resolve()


def _layout_backend(root):
    """Backend implied by the capture layout, independent of any metadata."""
    return "msprof" if glob.glob(str(Path(root) / "PROF_*")) else "msopprof-op"


def _read_collection(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfCollectError(f"{path}: unreadable collection metadata: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != COLLECTION_SCHEMA:
        raise PerfCollectError(f"{path}: unsupported collection metadata")
    return payload


def build_evidence(capture, *, op_name=None, backend="auto", collection=None, warm_up=None):
    root = Path(capture)
    layout = backend if backend != "auto" else _layout_backend(root)
    owner = None
    if layout == "msprof":
        owner = _resolve_group(root)
    elif layout in ("msopprof-op", "msprof-op"):
        directories = _opprof_directories(root)
        owner = _owning_capture(root, _select_opprof_directory(root, directories, op_name))
    if collection is None and owner is not None:
        collection = _load_collection(owner)
    recorded = (collection or {}).get("backend")
    if backend == "auto":
        backend = recorded or layout
    elif recorded and recorded != backend:
        raise PerfCollectError(
            f"{capture}: collected with backend {recorded}, not {backend}"
        )
    if backend == "msprof":
        parts = _msprof_parts(root, op_name)
    elif backend in ("msopprof-op", "msprof-op"):
        parts = _opprof_parts(root, op_name)
    else:
        raise PerfCollectError(f"unsupported backend: {backend}")
    return _evidence_envelope(
        backend, parts.parsed, collection, warm_up, parts.groups, parts.per_core,
        parts.reason, parts.note, parts.attribution, op_name,
    )


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _argv(command):
    return shlex.split(command) if isinstance(command, str) else list(command)


_APP_FAILURE = re.compile(
    r"An exception has occurred in process App, return code:\s*(?P<reason>[^\r\n]*)"
)
_APP_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")


def _tail(text, limit=20):
    return "\n".join((text or "").splitlines()[-limit:])


def _line_failure(line):
    """Return the failure reported on one log line, if any."""
    match = _APP_FAILURE.search(line)
    if match:
        return match.group("reason").strip() or "unknown application error"
    if _APP_TRACEBACK.search(line):
        return "application traceback"
    return None


def _text_failure(text):
    """Return the application failure reported in an in-memory log, if any."""
    for line in (text or "").splitlines():
        reason = _line_failure(line)
        if reason:
            return reason
    return None


def _log_failure(path, limit=20):
    """Scan a log file line by line: bounded tail plus the first failure marker."""
    tail = deque(maxlen=limit)
    reason = None
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            tail.append(line.rstrip("\n"))
            if reason is None:
                reason = _line_failure(line)
    return "\n".join(tail), reason


_TERMINATE_GRACE_S = 2.0


def _popen(argv, **kwargs):
    """Spawn one stage process in its own session so its group can be cleaned."""
    return subprocess.Popen(
        argv, start_new_session=os.name == "posix", **kwargs
    )


def _terminate_process_group(process, grace=_TERMINATE_GRACE_S):
    """Terminate the process group created for one stage, grandchildren included.

    A collector may exit while the application it launched keeps running on the
    device, and a timeout that only ends the collector would leave that NPU work
    behind. Only the group created for this stage is signalled.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def _stage_outcome(log_path, stdout):
    """Return the bounded tail and the application failure for one finished stage."""
    if log_path is not None:
        return _log_failure(Path(log_path))
    text = stdout or ""
    return _tail(text), _text_failure(text)


def _communicate_stage(process, stage, timeout):
    """Wait for one stage; on timeout clean the process group before failing."""
    try:
        return process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        try:
            process.communicate(timeout=_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
        raise PerfCollectError(f"{stage} timed out after {timeout}s") from exc


def run_stage(argv, *, stage, timeout, log_path=None):
    """Run one collector stage; a failed collector or application fails the stage.

    Output streams to ``log_path`` instead of being buffered, so a long capture
    cannot exhaust memory, and non-UTF-8 bytes are replaced instead of raising.
    Some collectors exit 0 even when the profiled application raised and only
    report the failure in their log; that marker or a traceback fails the stage so
    an incomplete capture is never reported as success. On timeout the whole
    process group is terminated before the error is raised.
    """
    stream = None
    if log_path is not None:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w", encoding="utf-8", errors="replace")
    try:
        process = _popen(
            argv,
            stdout=stream if stream is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, _ = _communicate_stage(process, stage, timeout)
    finally:
        if stream is not None:
            stream.close()
    tail, app_failure = _stage_outcome(log_path, stdout)
    if process.returncode:
        raise PerfCollectError(f"{stage} failed (rc={process.returncode}): {tail}")
    if app_failure:
        raise PerfCollectError(f"{stage}: application failed: {app_failure}")
    return process


def _l2_clean():
    """Best-effort L2 flush; a failed flush is reported as a note, not a pass."""
    raw = os.environ.get("TILE_FWK_DEVICE_ID", "")
    device = int(raw.strip()) if raw.strip().isdigit() else 0
    code = (
        "import torch, torch_npu;"
        f"d={device};torch.npu.set_device(d);"
        "b=torch.ones((96,1024,1024),dtype=torch.float16,device=f'npu:{d}');"
        "torch.max(b);torch.npu.synchronize()"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    return None if done.returncode == 0 else f"l2_clean_failed:rc={done.returncode}"


def _require(binary, source):
    if binary is None:
        raise PerfCollectError(
            f"{source} not found; set ASCEND_HOME_PATH or extend PATH"
        )
    return binary


def _select_metrics(metrics):
    selected = METRICS if metrics in (None, "", "all") else tuple(
        item.strip() for item in str(metrics).split(",") if item.strip()
    )
    unknown = [item for item in selected if item not in METRICS]
    if unknown:
        raise PerfCollectError(f"unsupported metrics: {unknown}")
    return selected


def _collector_identity(name, path):
    return {"name": name, "path": str(path), "version": _binary_version(name, path)}


def _toolkit_info():
    """Read the installed toolkit version without executing any collector."""
    home = _ascend_home()
    for path in (
        home / "x86_64-linux" / "ascend_toolkit_install.info",
        home / "ascend_toolkit_install.info",
    ):
        if not path.is_file():
            continue
        info = {}
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return {}
        for line in lines:
            key, separator, value = line.partition("=")
            if separator and key.strip() in ("version", "innerversion"):
                info[key.strip()] = value.strip() or None
        return info
    return {}


def _capture_identity():
    toolkit = _toolkit_info()
    return {
        "arch": os.environ.get("NPU_ARCH") or os.environ.get("ASCEND_SOC_VERSION") or None,
        "soc": os.environ.get("ASCEND_SOC_VERSION") or None,
        "cann": os.environ.get("ASCEND_HOME_PATH") or None,
        "ascend_home": str(_ascend_home()),
        "toolkit_version": toolkit.get("version"),
        "toolkit_innerversion": toolkit.get("innerversion"),
        "device": os.environ.get("TILE_FWK_DEVICE_ID") or None,
    }


def _collection_payload(backend, collector, argv, app, *, metrics, warm_up,
                        launch_count, l2_clean, stages, notes, extra=None):
    payload = {
        "schema": COLLECTION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "collector": collector,
        "argv": [str(item) for item in argv],
        "app": app,
        "metrics": [str(item) for item in metrics],
        "warm_up": int(warm_up),
        "launch_count": int(launch_count),
        "l2_clean": bool(l2_clean),
        "identity": _capture_identity(),
        "stages": stages,
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return payload


def _run_metric_capture(msprof, app, group, metric, *, timeout):
    sub = group / f"PROF_{metric}"
    sub.mkdir(parents=True, exist_ok=True)
    run_stage(
        [
            str(msprof),
            f"--output={sub}",
            "--ai-core=on",
            f"--aic-metrics={metric}",
            "--task-time=on",
            "--ascendcl=on",
            *_argv(app),
        ],
        stage=f"metric[{metric}]",
        timeout=timeout,
        log_path=sub / "msprof.log",
    )


def _run_sample_capture(msprof, app, group, *, timeout):
    sample = group / "PROF_Sample"
    sample.mkdir(parents=True, exist_ok=True)
    run_stage(
        [
            str(msprof),
            f"--output={sample}",
            "--ai-core=on",
            "--aic-metrics=PipeUtilization",
            "--aic-mode=sample-based",
            "--aic-freq=100",
            "--task-time=on",
            "--ascendcl=on",
            *_argv(app),
        ],
        stage="sample",
        timeout=timeout,
        log_path=sample / "msprof.log",
    )
    return sample


def _l2_clean_note(enabled, notes):
    if not enabled:
        return
    note = _l2_clean()
    if note:
        notes.append(note)


def _run_warm_ups(app, group, warm_up, timeout, stages):
    for index in range(int(warm_up)):
        run_stage(
            _argv(app),
            stage=f"warm_up[{index}]",
            timeout=timeout,
            log_path=group / f"warm_up_{index}.log",
        )
        stages.append({"stage": f"warm_up[{index}]", "returncode": 0})


def _run_metric_stages(msprof, app, group, selected, l2_clean, timeout, stages, notes):
    for metric in selected:
        _l2_clean_note(l2_clean, notes)
        _run_metric_capture(msprof, app, group, metric, timeout=timeout)
        stages.append({"stage": f"metric[{metric}]", "returncode": 0})


def _run_sample_stage(msprof, app, group, l2_clean, timeout, stages, notes):
    _l2_clean_note(l2_clean, notes)
    _run_sample_capture(msprof, app, group, timeout=timeout)
    stages.append({"stage": "sample", "returncode": 0})


def collect_msprof(app, *, output_dir, metrics="all", warm_up=3, l2_clean=True, timeout=3600):
    """Run the seven application-level metric groups plus one sample-based capture."""
    msprof = _require(_binary("msprof", Path("profiler/bin/msprof")), "msprof")
    selected = _select_metrics(metrics)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    group = Path(output_dir, f"PROF_GROUP_{stamp}_{os.getpid()}")
    group.mkdir(parents=True, exist_ok=False)
    stages, notes = [], []
    meta = _collection_payload(
        "msprof", _collector_identity("msprof", msprof), [msprof], app,
        metrics=selected, warm_up=warm_up, launch_count=1, l2_clean=l2_clean,
        stages=stages, notes=notes,
    )
    try:
        _run_warm_ups(app, group, warm_up, timeout, stages)
        _run_metric_stages(msprof, app, group, selected, l2_clean, timeout, stages, notes)
        _run_sample_stage(msprof, app, group, l2_clean, timeout, stages, notes)
    except PerfCollectError as exc:
        meta["error"] = str(exc)
        _write_json(group / "collection.json", meta)
        raise
    _write_json(group / "collection.json", meta)
    return group


def collect_msopprof_op(app, *, output_dir, kernel_name=None, aic_metrics="PipeUtilization",
                        warm_up=3, launch_count=1, timeout=3600):
    """Run one operator-level ``msprof op`` / ``msopprof`` capture."""
    msopprof = _binary("msopprof", Path("msopprof/bin/msopprof"))
    if msopprof is not None:
        backend, argv = "msopprof-op", [str(msopprof)]
        collector = _collector_identity("msopprof", msopprof)
    else:
        msprof = _require(_binary("msprof", Path("profiler/bin/msprof")), "msprof")
        backend, argv = "msprof-op", [str(msprof), "op"]
        collector = _collector_identity("msprof", msprof)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = set(glob.glob(str(out / "OPPROF_*")))
    argv += ["--output", str(out), "--warm-up", str(int(warm_up)),
             "--launch-count", str(int(launch_count))]
    if kernel_name:
        argv += ["--kernel-name", kernel_name]
    if aic_metrics:
        argv += ["--aic-metrics", aic_metrics]
    argv.extend(_argv(app))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = out / f"capture_{stamp}_{os.getpid()}.log"
    run_stage(argv, stage="capture", timeout=timeout, log_path=log_path)
    captures = sorted(set(glob.glob(str(out / "OPPROF_*"))) - before)
    if len(captures) != 1:
        raise PerfCollectError(
            f"expected one new OPPROF_* directory under {out}, found {len(captures)}"
        )
    capture = Path(captures[0])
    requested = [item.strip() for item in (aic_metrics or "").split(",") if item.strip()]
    payload = _collection_payload(
        backend, collector, argv, app,
        metrics=requested,
        warm_up=warm_up, launch_count=launch_count, l2_clean=False,
        stages=[{"stage": "capture", "returncode": 0}], notes=[],
        extra={"kernel_name": kernel_name},
    )
    if not _opprof_directories(capture, required=False):
        payload["error"] = (
            "capture has no OPPROF metric CSV; --kernel-name did not match or no "
            "launch was recorded"
        )
        _write_json(capture / "collection.json", payload)
        raise PerfCollectError(f"{capture}: {payload['error']}")
    _write_json(capture / "collection.json", payload)
    return capture


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="report collector availability")
    p_probe.add_argument("--format", choices=("text", "json"), default="text")

    p_discover = sub.add_parser("discover", help="list exact Op Name values")
    p_discover.add_argument("--capture", required=True)
    p_discover.add_argument("--format", choices=("text", "json"), default="text")

    p_run = sub.add_parser("run", help="collect a new capture")
    p_run.add_argument("--backend", required=True, choices=("msprof", "msopprof-op"))
    p_run.add_argument("--app", required=True)
    p_run.add_argument("--output", required=True)
    p_run.add_argument("--metrics", default="all")
    p_run.add_argument("--warm-up", type=int, default=3)
    p_run.add_argument("--timeout", type=int, default=3600)
    p_run.add_argument("--no-l2-clean", action="store_true")
    p_run.add_argument("--kernel-name")
    p_run.add_argument("--aic-metrics", default="PipeUtilization")
    p_run.add_argument("--launch-count", type=int, default=1)

    p_sum = sub.add_parser("summarize", help="parse a capture into evidence JSON")
    p_sum.add_argument("--capture", required=True)
    p_sum.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "msprof", "msopprof-op", "msprof-op"),
    )
    p_sum.add_argument("--op-name")
    p_sum.add_argument("--out", required=True)
    return parser


def _command_probe(args):
    payload = probe()
    if args.format == "json":
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for name, item in payload["collectors"].items():
        _emit(f"{name}: {item['path'] or 'not found'}")
    return 0


def _command_discover(args):
    payload = list_op_names(args.capture)
    if args.format == "json":
        _emit(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for item in payload:
        _emit(f"{item['task_duration_us']:.3f}us\t{item['op_name']}\t{item['op_type']}")
    return 0


def _command_run(args):
    if args.backend == "msprof":
        capture = collect_msprof(
            args.app,
            output_dir=args.output,
            metrics=args.metrics,
            warm_up=args.warm_up,
            l2_clean=not args.no_l2_clean,
            timeout=args.timeout,
        )
    else:
        capture = collect_msopprof_op(
            args.app,
            output_dir=args.output,
            kernel_name=args.kernel_name,
            aic_metrics=args.aic_metrics,
            warm_up=args.warm_up,
            launch_count=args.launch_count,
            timeout=args.timeout,
        )
    _emit(str(capture))
    return 0


def _command_summarize(args):
    evidence = build_evidence(
        args.capture, op_name=args.op_name, backend=args.backend
    )
    _emit(str(_write_json(args.out, evidence)))
    return 2 if evidence["collection"]["status"] == "failed" else 0


_COMMANDS = {
    "probe": _command_probe,
    "discover": _command_discover,
    "run": _command_run,
    "summarize": _command_summarize,
}


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except PerfCollectError as exc:
        _emit_error(f"error: {exc}")
        return 2
    except subprocess.TimeoutExpired as exc:
        _emit_error(f"error: collector timed out after {exc.timeout}s")
        return 2


if __name__ == "__main__":
    sys.exit(main())
