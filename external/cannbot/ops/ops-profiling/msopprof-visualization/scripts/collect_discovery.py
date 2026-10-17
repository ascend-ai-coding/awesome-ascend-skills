#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Executable discovery and build-tree inspection helpers for msOpProf collection.

Split out of collect.py to keep that module within static-check size limits.
Everything here is re-exported by collect.py, so the public collector surface
is unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import runtime_guard as rtguard

from collect_validate import STANDARD_CSV_NAMES, bin_contains_all, normalize_metric_token

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(x) for x in cmd])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint_key(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def ensure_msprof_output_permissions(path: Path) -> Optional[Dict[str, Any]]:
    """Strip group/other write bits so msprof accepts ``path`` as an output directory.

    msprof refuses to write into a directory that is writable by group or other
    users: it exits 0, logs ``writable by any other users or group users``, and
    produces no artifacts, which otherwise surfaces as a misleading empty-canary
    abort. Directories created under a permissive umask (e.g. 002) hit this.
    Returns a fix record for audit, or None when no change was needed.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    if not (st.st_mode & 0o022):
        return None
    new_mode = st.st_mode & ~0o022
    record: Dict[str, Any] = {"path": str(path), "old_mode": oct(st.st_mode & 0o777)}
    try:
        os.chmod(path, new_mode)
        record["new_mode"] = oct(new_mode & 0o777)
    except OSError as exc:
        record["error"] = repr(exc)
    return record


def sanitize_output_tree(root: Path) -> List[Dict[str, Any]]:
    """Remove group/other write bits from every directory under the output root.

    Covers both pre-existing directories from earlier runs and the fixed block
    directories created at startup, before any msprof command runs.
    """
    fixes: List[Dict[str, Any]] = []
    if not root.is_dir():
        return fixes
    candidates = [root]
    candidates.extend(p for p in root.rglob("*") if p.is_dir() and not p.is_symlink())
    for directory in candidates:
        record = ensure_msprof_output_permissions(directory)
        if record is not None:
            fixes.append(record)
    return fixes


def previous_run_allows_reuse(output_root: Path) -> Tuple[bool, str]:
    run_state = read_json(output_root / "_internal" / "run_state.json", {})
    state_files = (
        list((output_root / "_internal" / "state").glob("*.json"))
        if (output_root / "_internal" / "state").is_dir() else []
    )
    if not state_files:
        return True, "No prior command state exists."
    if run_state.get("status") == "completed":
        return True, "Prior run completed successfully."
    return False, (f"Prior output is not marked completed (status={run_state.get('status') or 'missing'}); "
        f"cached command state is disabled.")


def is_single_exact_kernel_filter(kernel_name: Optional[str]) -> bool:
    """Return True when one exact kernel prefix/name is requested, not a union/wildcard."""
    return bool(kernel_name and "|" not in kernel_name and "*" not in kernel_name)


# Cross-skill contract. A visualization Skill may depend on these names.
BLOCK_DIRS: Dict[str, str] = {
    "discovery": "00_discovery",
    "details": "01_details",
    "roofline": "02_roofline",
    "timeline": "03_timeline",
    "source": "04_source",
    "warp_stall": "05_warp_stall",
    "instruction_timeline": "06_instruction_timeline",
    "memory_detail": "07_memory_detail",
    "raw_data": "08_raw_data",
    "timeline_detail": "09_timeline_detail",
    "kernel_scale": "10_kernel_scale",
    "onchip_memory": "11_onchip_memory",
}


KNOWN_METRICS = [
    "ArithmeticUtilization", "L2Cache", "Memory", "MemoryL0", "MemoryUB",
    "PipeUtilization", "ResourceConflictRatio", "BasicInfo", "Roofline",
    "Occupancy", "KernelScale", "PipeTimeline", "pipeTimeLine",
    "TimelineDetail", "instrTimeLine", "Source", "PCSampling", "PcSampling", "Default",
    "MemoryDetail",
]


def run_capture(cmd: Sequence[str], timeout: int = 20, cwd: Optional[Path] = None) -> Dict[str, Any]:
    return rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        cmd,
        timeout=timeout,
        cwd=cwd,
        heartbeat_seconds=0,
    ))


def is_candidate_executable(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    if path.suffix.lower() in {".sh", ".py", ".pl", ".rb", ".so", ".o", ".a", ".json", ".csv"}:
        return False
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return bool(mode & stat.S_IXUSR)


def _candidate_score(operator_root: Path, path: Path) -> Tuple[int, List[str]]:
    rel_parts = [x.lower() for x in path.relative_to(operator_root).parts]
    name = path.name.lower()
    score = 0
    reasons: List[str] = []
    if name in {"demo", "main", "run"}:
        score += 100
        reasons.append(f"standard sample executable name {name}")
    if "build_npu" in rel_parts:
        score += 90
        reasons.append("located in build_npu")
    elif "build" in rel_parts:
        score += 60
        reasons.append("located in build")
    elif "out" in rel_parts:
        score += 50
        reasons.append("located in out")
    elif "bin" in rel_parts:
        score += 40
        reasons.append("located in bin")
    depth = len(rel_parts)
    score += max(0, 20 - depth)
    reasons.append(f"path depth {depth}")
    return score, reasons


def discover_executable_candidates(operator_root: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path in operator_root.rglob("*"):
        if not is_candidate_executable(path):
            continue
        if {"CMakeFiles", ".git", "_deps"} & set(path.parts):
            continue
        score, reasons = _candidate_score(operator_root, path)
        candidates.append({
            "path": str(path.resolve()),
            "relative_path": path.relative_to(operator_root).as_posix(),
            "score": score,
            "reasons": reasons,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        })
    candidates.sort(key=lambda x: (x["score"], x["mtime_ns"], x["size"], x["relative_path"]), reverse=True)
    return candidates


def detect_relative_resource_hints(app: Path, app_cwd: Path) -> Dict[str, Any]:
    hints: List[str] = []
    try:
        with app.open("rb") as handle:
            data = handle.read(64 * 1024 * 1024)
        pattern = rb"(?:\.\.?/)[A-Za-z0-9_.\-/]+(?:\.(?:bin|json|csv|txt|npy|data|cfg))"
        for raw in re.findall(pattern, data):
            value = raw.decode("utf-8", errors="replace")
            if value not in hints:
                hints.append(value)
            if len(hints) >= 50:
                break
    except OSError:
        logger.debug("failed to read executable for resource hints: %s", app, exc_info=True)
    resolved = []
    missing = []
    for hint in hints:
        target = (app_cwd / hint).resolve()
        item = {"hint": hint, "resolved": str(target), "exists": target.exists()}
        resolved.append(item)
        if not target.exists():
            missing.append(item)
    return {"hints": resolved, "missing_count": len(missing), "missing": missing[:20]}


def resolve_executable_details(operator_root: Path, explicit_app: Optional[str]) -> Dict[str, Any]:
    if explicit_app:
        path = Path(explicit_app).expanduser()
        if not path.is_absolute():
            path = operator_root / path
        path = path.resolve()
        if not is_candidate_executable(path):
            raise rtguard.UsageError(f"ERROR: executable not found or not executable: {path}")
        return {
            "selected": path,
            "selection_mode": "explicit",
            "selection_reason": "Selected by --app.",
            "ambiguous": False,
            "candidates": [{
                "path": str(path),
                "relative_path": (
                    path.relative_to(operator_root).as_posix()
                    if path.is_relative_to(operator_root) else None
                ),
                "score": None,
                "reasons": ["explicit --app"],
            }],
        }

    candidates = discover_executable_candidates(operator_root)
    if not candidates:
        raise rtguard.UsageError("ERROR: no executable found. Build the sample first or pass --app explicitly.")
    selected = Path(candidates[0]["path"])
    top_score = candidates[0]["score"]
    tied = [x for x in candidates if x["score"] == top_score]
    ambiguous = len(tied) > 1
    reason = "Selected highest-ranked executable: " + "; ".join(candidates[0]["reasons"])
    if ambiguous:
        reason += f". {len(tied)} candidates share the top score; pass --app to remove ambiguity."
    return {
        "selected": selected,
        "selection_mode": "auto",
        "selection_reason": reason,
        "ambiguous": ambiguous,
        "candidates": candidates[:25],
    }


def resolve_executable(operator_root: Path, explicit_app: Optional[str]) -> Path:
    return Path(resolve_executable_details(operator_root, explicit_app)["selected"])


TEST_LIKE_EXECUTABLE = re.compile(r"(?:test|gtest|unittest)", re.IGNORECASE)


def test_like_executable_warning(app: Path) -> Optional[str]:
    if TEST_LIKE_EXECUTABLE.search(app.name):
        return (
            f"Selected executable '{app.name}' looks like a test binary; "
            "pass --app explicitly if this is not the intended profiling target."
        )
    return None


def resolve_app_cwd(operator_root: Path, raw: Optional[str]) -> Path:
    if not raw:
        return operator_root.resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = operator_root / path
    path = path.resolve()
    if not path.is_dir():
        raise rtguard.UsageError(f"ERROR: application working directory does not exist: {path}")
    return path


def parse_supported_metrics(help_text: str) -> List[str]:
    """Return exact metric spellings exposed by the installed CLI.

    Detection is case-insensitive so version-specific spellings such as PCSampling and
    PcSampling resolve to the same capability, while the exact help token is retained
    and passed back to msprof.
    """
    by_normalized: Dict[str, Tuple[int, str]] = {}
    for metric in KNOWN_METRICS:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(metric)}(?![A-Za-z0-9_])"
        for match in re.finditer(pattern, help_text, flags=re.IGNORECASE):
            actual = match.group(0)
            key = normalize_metric_token(actual)
            current = by_normalized.get(key)
            if current is None or match.start() < current[0]:
                by_normalized[key] = (match.start(), actual)
    return [item[1] for item in sorted(by_normalized.values(), key=lambda x: x[0])]


def parse_supported_options(help_text: str) -> List[str]:
    """Extract exact long-option names from `msprof op --help`."""
    seen: Set[str] = set()
    out: List[str] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_-])(--[A-Za-z0-9][A-Za-z0-9-]*)", help_text):
        option = match.group(1)
        if option not in seen:
            seen.add(option)
            out.append(option)
    return out


def _find_cache_candidates(operator_root: Path) -> List[Path]:
    candidates = []
    for preferred in [operator_root / "build" / "CMakeCache.txt", operator_root / "CMakeCache.txt"]:
        if preferred.is_file():
            candidates.append(preferred)
    if not candidates:
        candidates = sorted(operator_root.rglob("CMakeCache.txt"))[:8]
    return candidates


def _read_cache_values(path: Path, keys: Set[str]) -> Optional[Dict[str, str]]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        is_blank_or_comment = not line or line.startswith(("#", "//"))
        is_key_value = "=" in line and ":" in line.split("=", 1)[0]
        if is_blank_or_comment or not is_key_value:
            continue
        lhs, value = line.split("=", 1)
        key = lhs.split(":", 1)[0]
        if key in keys:
            values[key] = value
    return values


def parse_cmake_cache(operator_root: Path) -> Dict[str, Any]:
    """Capture reproducibility-critical build metadata without enforcing one sample config."""
    keys = {
        "SCENARIO_NUM", "CMAKE_ASC_ARCHITECTURES", "CMAKE_BUILD_TYPE",
        "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_ASC_FLAGS",
    }
    caches: List[Dict[str, Any]] = []
    for path in _find_cache_candidates(operator_root):
        values = _read_cache_values(path, keys)
        if values is None:
            continue
        caches.append({"path": str(path), "values": values})
    return {"cmake_caches": caches}


def run_shell_capture(command: str, cwd: Path, timeout: int) -> Dict[str, Any]:
    result = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        ["bash", "-lc", command],
        timeout=timeout,
        cwd=cwd,
        heartbeat_seconds=30,
        heartbeat_label="debug rebuild",
    ))
    result["command"] = command
    return result


def _debug_scope_root(operator_root: Path, app: Path) -> Path:
    """Return the build tree that actually owns the selected executable.

    Source eligibility must never be inferred from an unrelated sibling build.
    Prefer the nearest ancestor containing CMakeCache.txt and otherwise stay
    inside the executable's parent directory.
    """
    app = app.resolve()
    operator_root = operator_root.resolve()
    for parent in [app.parent, *app.parents]:
        if (parent / "CMakeCache.txt").is_file():
            return parent
        if parent == operator_root:
            break
    return app.parent


def _collect_debug_check_candidates(scope_root: Path, app: Path, suffixes: Set[str]) -> List[Path]:
    checks: List[Path] = [app]
    if scope_root.exists():
        for candidate in scope_root.rglob("*"):
            if len(checks) >= 160:
                break
            if not candidate.is_file() or candidate == app:
                continue
            if candidate.suffix.lower() in suffixes or is_candidate_executable(candidate):
                checks.append(candidate)
    return checks


def _inspect_debug_candidate(candidate: Path, readelf: str, capture_fn: Any) -> Dict[str, Any]:
    result = capture_fn([readelf, "-SW", str(candidate)], timeout=8)
    text = ((result.get("stdout") or "") + (result.get("stderr") or "")).lower()
    return {
        "path": str(candidate),
        "debug_line": bool(re.search(r"\.debug_line(?:\b|\.dwo)", text)),
        "debug_info": bool(re.search(r"\.debug_info(?:\b|\.dwo)", text)),
        "return_code": result.get("return_code"),
    }


def detect_debug_symbols_details(operator_root: Path, app: Path, capture=None) -> Dict[str, Any]:
    """Detect usable line-table debug information for the selected executable.

    ``Source`` requires ``.debug_line`` (or ``.debug_line.*``) in the selected
    executable or an artifact from the same build tree. ``.debug_info`` alone
    is not sufficient, and debug sections in another scenario/build directory
    must not make the current executable appear eligible.
    """
    app = app.resolve()
    scope_root = _debug_scope_root(operator_root, app)
    suffixes = {".o", ".so", ".elf", ".out", ".bin", ".a"}
    checks = _collect_debug_check_candidates(scope_root, app, suffixes)
    readelf = shutil.which("readelf")
    if not readelf:
        return {
            "detected": False, "tool": None, "scope_root": str(scope_root),
            "inspected": [], "detected_in": [], "info_only_in": [],
            "reason": "readelf not found",
        }
    inspected: List[Dict[str, Any]] = []
    detected_in: List[str] = []
    info_only_in: List[str] = []
    capture_fn = capture or run_capture
    for candidate in checks:
        record = _inspect_debug_candidate(candidate, readelf, capture_fn)
        inspected.append(record)
        if record["debug_line"]:
            detected_in.append(record["path"])
        elif record["debug_info"]:
            info_only_in.append(record["path"])
    reason = None
    if not detected_in:
        reason = "No usable .debug_line section was found in the selected executable build tree."
        if info_only_in:
            reason += " .debug_info without line tables is insufficient for Source mapping."
    return {
        "detected": bool(detected_in),
        "tool": readelf,
        "scope_root": str(scope_root),
        "selected_executable": str(app),
        "selected_executable_has_debug_line": str(app) in detected_in,
        "inspected": inspected,
        "detected_in": detected_in,
        "info_only_in": info_only_in,
        "reason": reason,
    }


def detect_kernel_scale_instrumentation(operator_root: Path, app: Path) -> Dict[str, Any]:
    needles = ["MetricsProfStart", "MetricsProfStop"]
    matched: List[str] = []
    source_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".asc", ".py"}
    for path in operator_root.rglob("*"):
        if len(matched) >= 40:
            break
        if not path.is_file() or path.suffix.lower() not in source_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if all(n in text for n in needles):
            matched.append(str(path))
    binary_evidence = all(bin_contains_all(app, needles).values())
    return {"detected": bool(matched or binary_evidence), "source_files": matched, "binary_evidence": binary_evidence}


def find_result_dirs(run_dir: Path) -> List[Path]:
    return sorted(
        [p for p in run_dir.rglob("OPPROF_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime_ns,
    )


def find_latest_result_dir(run_dir: Path) -> Optional[Path]:
    dirs = find_result_dirs(run_dir)
    return dirs[-1] if dirs else None


def find_latest_file(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None:
        return None
    files = [p for p in root.rglob(name) if p.is_file() and p.stat().st_size > 0]
    return max(files, key=lambda p: p.stat().st_mtime_ns) if files else None


def _row_value_case_insensitive(row: Mapping[str, Any], names: Sequence[str]) -> Optional[str]:
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _parse_operator_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    name = _row_value_case_insensitive(row, ["Op Name", "OpName", "Kernel Name"]) or ""
    op_type = _row_value_case_insensitive(row, ["Op Type", "OpType", "Kernel Type"]) or ""
    if not name:
        return None
    duration_raw = _row_value_case_insensitive(row, ["Duration(us)", "Duration (us)", "Duration",
        "Task Duration(us)"])
    block_dim_raw = _row_value_case_insensitive(row, ["Block Dim", "BlockDim", "Block Dimension"])
    try:
        duration_us = float(duration_raw.replace(",", "")) if duration_raw is not None else None
    except ValueError:
        duration_us = None
    try:
        block_dim = int(float(block_dim_raw)) if block_dim_raw is not None else None
    except ValueError:
        block_dim = None
    return {
        "name": name,
        "type": op_type,
        "duration_us": duration_us,
        "block_dim": block_dim,
    }


def discover_operators(result_dir: Optional[Path]) -> List[Dict[str, Any]]:
    csv_path = find_latest_file(result_dir, "OpBasicInfo.csv")
    if not csv_path:
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    unique: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        parsed = _parse_operator_row(row)
        if parsed is None or (parsed["name"], parsed["type"]) in seen:
            continue
        seen.add((parsed["name"], parsed["type"]))
        unique.append(parsed)
    return unique


def parse_key_value(items: Sequence[str], option_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise rtguard.UsageError(f"ERROR: {option_name} expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise rtguard.UsageError(f"ERROR: {option_name} contains an empty key: {item}")
        out[key] = value.strip()
    return out


DEFAULT_BLOCK_TIMEOUTS: Dict[str, int] = {
    "discovery": 180,
    "details": 180,
    "roofline": 300,
    "timeline": 480,
    "source": 360,
    "warp_stall": 360,
    "instruction_timeline": 480,
    "memory_detail": 300,
    "raw_data": 240,
    "timeline_detail": 600,
    "kernel_scale": 600,
    "onchip_memory": 60,
}


def parse_block_timeouts(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    valid = set(BLOCK_DIRS)
    for item in items:
        if "=" not in item:
            raise rtguard.UsageError(f"ERROR: --block-timeout expects BLOCK=SECONDS, got: {item}")
        block, raw = item.split("=", 1)
        block = block.strip().lower().replace("-", "_")
        if block not in valid:
            raise rtguard.UsageError(f"ERROR: unknown block in --block-timeout: {block}")
        try:
            seconds = int(raw)
        except ValueError as exc:
            raise rtguard.UsageError(f"ERROR: invalid timeout seconds: {raw}") from exc
        if seconds <= 0:
            raise rtguard.UsageError("ERROR: block timeout must be positive.")
        out[block] = seconds
    return out


def derive_timeout_profile(
    global_timeout: int,
    overrides: Mapping[str, int],
    *,
    preflight_elapsed: Optional[float] = None,
    adaptive: bool = True,
) -> Dict[str, int]:
    profile: Dict[str, int] = {}
    observed = max(0.0, float(preflight_elapsed or 0.0))
    for block_id, base in DEFAULT_BLOCK_TIMEOUTS.items():
        timeout = int(base)
        if adaptive and observed > 0:
            # Profiler startup/device initialization dominates microsecond kernels.
            # Scale by the observed canary cost while retaining a feature-specific floor.
            multiplier = 12 if block_id in {"timeline", "instruction_timeline",
                "timeline_detail", "kernel_scale"} else 8
            timeout = max(timeout, int(observed * multiplier + 30))
        timeout = min(int(global_timeout), timeout)
        if block_id in overrides:
            timeout = int(overrides[block_id])
        profile[block_id] = max(1, timeout)
    return profile


# ---------------------------------------------------------------------------
# Collection contract catalog: feature/block/preset tables shared with collect.py
# (moved here to keep collect.py within static-check size limits; collect.py
# re-exports every name, so the public surface is unchanged).
# ---------------------------------------------------------------------------

CONTRACT_SCHEMA = "msopprof-collection/v2"


RAW_COUNTER_METRICS = [
    "ArithmeticUtilization", "L2Cache", "Memory", "MemoryL0",
    "MemoryUB", "PipeUtilization", "ResourceConflictRatio",
]

# A visual feature can require more than one collector block. In particular,
# the full Details page needs Occupancy for inter-core balance and
# MemoryDetail/Default for compute, memory, cache, and advice structures.
# FEATURE_TO_BLOCK is retained as the primary-provider compatibility map;
# FEATURE_TO_BLOCKS is authoritative for collection planning.
FEATURE_TO_BLOCKS: Dict[str, List[str]] = {
    "details": ["details", "memory_detail"],
    "occupancy": ["details"],
    "compute": ["memory_detail"],
    "memory": ["memory_detail"],
    "cache": ["memory_detail"],
    "advice": ["memory_detail"],
    "roofline": ["roofline"],
    "timeline": ["timeline"],
    "source": ["source"],
    "warp-stall": ["warp_stall"],
    "instruction-timeline": ["instruction_timeline"],
    "memory-detail": ["memory_detail"],
    "raw-data": ["raw_data"],
    "timeline-detail": ["timeline_detail"],
    "kernel-scale": ["kernel_scale"],
    "onchip-memory": ["onchip_memory"],
}

FEATURE_TO_BLOCK: Dict[str, str] = {
    feature: blocks[0] for feature, blocks in FEATURE_TO_BLOCKS.items()
}

BLOCKS: Dict[str, Dict[str, Any]] = {
    "discovery": {
        "display_name": "Discovery",
        "metric_candidates": ["BasicInfo"],
        "visual_features": [],
        "primary_artifact": "OpBasicInfo.csv",
        "data": ["OpBasicInfo.csv", "visualize_data.bin"],
        "purpose": "Discover operator names and the profiler-reported operator type.",
        "use_cases": ["Resolve a single operator before feature collection"],
        "cost": "low",
    },
    "details": {
        "display_name": "Details",
        "metric_candidates": ["Occupancy"],
        "visual_features": ["occupancy"],
        "primary_artifact": "visualize_data.bin",
        "data": [
            "visualize_data.bin: op_detail / subblock_detail / core_memory_map / table_per_block",
            "CSV files when the selected metric emits them",
        ],
        "purpose": ("Collect inter-core occupancy and load-balance evidence for the occupancy portion of the "
            "composite Details page."),
        "use_cases": [
            "Check inter-core load balance",
            "Compare physical-core duration, throughput, and cache-hit indicators",
        ],
        "cost": "medium",
    },
    "roofline": {
        "display_name": "Roofline",
        "metric_candidates": ["Roofline"],
        "visual_features": ["roofline"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin: multiple_rooflines and bound classification", "Standard profiling CSV suite"],
        "purpose": ("Collect Roofline-specific records and bottleneck classification without replacing "
            "canonical Details data."),
        "use_cases": ["Classify compute-vs-memory bound behavior", "Plot actual Roofline points when present"],
        "cost": "medium",
    },
    "timeline": {
        "display_name": "Pipe Timeline",
        "metric_candidates": ["PipeTimeline", "pipeTimeLine"],
        "visual_features": ["timeline"],
        "primary_artifact": "trace.json",
        "data": ["trace.json", "visualize_data.bin fallback"],
        "purpose": "Collect sampled per-pipe execution intervals for lane-based timing analysis.",
        "use_cases": ["Inspect Vector/Scalar/MTE overlap", "Compare sampled cores and subcores", ("Locate "
            "long execution intervals")],
        "cost": "high",
    },
    "source": {
        "display_name": "Source Hotspot",
        "metric_candidates": ["Source"],
        "visual_features": ["source"],
        "primary_artifact": "visualize_data.bin",
        "data": ["Source visualize_data.bin: source snapshots / line map / instruction map / GPR status", ("Optional "
            "PCSampling visualize_data.bin: per-line and per-instruction stall percentages and reason counts")],
        "purpose": ("Collect source-to-instruction mappings and optionally augment the Source explorer with "
            "compatible PCSampling stall evidence."),
        "use_cases": ["Locate hot or stalled source lines", "Link instructions and source code", ("Inspect "
            "GPR lifetimes"), "Inspect stall-reason composition"],
        "cost": "medium",
        "requires_debug_symbols": True,
    },
    "warp_stall": {
        "display_name": "Warp Stall",
        "metric_candidates": ["PCSampling", "PcSampling"],
        "visual_features": ["warp-stall"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin: PC sampling and SIMT stall information"],
        "purpose": "Collect SIMT stall samples and warp-level bottleneck evidence.",
        "use_cases": ["Classify warp stall reasons", "Find high-stall PCs"],
        "cost": "medium",
        "requires_simt": True,
    },
    "instruction_timeline": {
        "display_name": "Instruction Timeline",
        "metric_candidates": ["instrTimeLine", "TimelineDetail,Default", "TimelineDetail"],
        "visual_features": ["instruction-timeline"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin: instruction-level timing records"],
        "purpose": "Collect instruction-granularity timeline data when exposed by the installed CLI.",
        "use_cases": ["Inspect instruction ordering", "Locate instruction-level latency"],
        "cost": "high",
    },
    "memory_detail": {
        "display_name": "Memory Detail",
        "metric_candidates": ["MemoryDetail", "Default"],
        "visual_features": ["compute", "memory", "cache", "advice", "memory-detail"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin and any detailed memory/cache outputs emitted by the CLI"],
        "purpose": ("Canonical targeted Memory Workload replay: prefer MemoryDetail; fall back to Default "
            "without adding a duplicate replay."),
        "use_cases": ["Build authoritative per-block directed memory edge tables", ("Inspect detailed memory "
            "behavior and native tables")],
        "cost": "medium",
    },
    "raw_data": {
        "display_name": "Raw Data",
        "metric_candidates": ["Default"],
        "visual_features": ["raw-data"],
        "primary_artifact": "CSV",
        "data": STANDARD_CSV_NAMES,
        "purpose": "Collect the standard CSV suite only when no richer already-requested replay has produced it.",
        "use_cases": ["Inspect raw per-block counters", "Verify visualized values", ("Export data for "
            "external analysis")],
        "cost": "low",
    },
    "timeline_detail": {
        "display_name": "Timeline Detail",
        "metric_candidates": ["TimelineDetail,Default", "TimelineDetail"],
        "visual_features": ["timeline-detail"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin: simulation instruction timeline / code hotspot structures"],
        "purpose": ("Scenario-specific fallback for simulation instruction timeline and code-hotspot data "
            "when the installed CLI exposes TimelineDetail."),
        "use_cases": ["Framework operator instruction timeline", "Simulation code hotspot"],
        "cost": "high",
    },
    "onchip_memory": {
        "display_name": "On-Chip Memory",
        "metric_candidates": [],
        "visual_features": ["onchip-memory"],
        "primary_artifact": "memory_info.json",
        "data": [("memory_info.json: buffer/tensor lifetime, address range, allocation metadata, and "
            "bank/group distribution")],
        "purpose": ("Discover and preserve the compiler-generated memory_info.json sidecar; do not replace it "
            "with msprof MemoryDetail counters."),
        "use_cases": ["On-chip memory lifetime map", "Peak/fragmentation inspection", "UB bank distribution"],
        "cost": "low",
    },
    "kernel_scale": {
        "display_name": "Kernel Scale",
        "metric_candidates": ["KernelScale"],
        "visual_features": ["kernel-scale"],
        "primary_artifact": "visualize_data.bin",
        "data": ["visualize_data.bin and metric-scale records emitted between MetricsProfStart/MetricsProfStop"],
        "purpose": ("Collect explicitly instrumented kernel-scale regions when the operator source or binary "
            "contains MetricsProfStart/MetricsProfStop markers."),
        "use_cases": ["Measure a selected code region", "Compare instrumented kernel phases or loop bodies"],
        "cost": "high",
        "requires_kernel_scale_instrumentation": True,
    },
}

PRESETS: Dict[str, List[str]] = {
    # Fast keeps one unique load-balance replay plus one canonical metric bundle.
    "fast": ["details", "raw-data"],
    # Core adds Roofline and PipeTimeline; Raw Data aliases Roofline when possible.
    "core": ["details", "roofline", "timeline", "raw-data"],
    # Complete maximizes general on-board coverage while avoiding duplicate Default replays.
    "complete": [
        "details", "memory-detail", "roofline", "timeline", "source",
        "warp-stall", "instruction-timeline", "onchip-memory", "raw-data",
    ],
    # Deep also attempts scenario-specific TimelineDetail.
    "deep": [
        "details", "memory-detail", "roofline", "timeline", "source",
        "warp-stall", "instruction-timeline", "timeline-detail", "kernel-scale", "onchip-memory", "raw-data",
    ],
}


def normalize_feature_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def command_description(block_id: str, metric: str) -> Dict[str, Any]:
    spec = BLOCKS[block_id]
    return {
        "display_name": spec["display_name"],
        "metric": metric,
        "output_directory": BLOCK_DIRS[block_id],
        "produced_data": spec["data"],
        "visual_features": spec["visual_features"],
        "purpose": spec["purpose"],
        "use_cases": spec["use_cases"],
        "cost": spec["cost"],
    }


def feature_catalog() -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for feature, block_id in FEATURE_TO_BLOCK.items():
        spec = BLOCKS[block_id]
        block_ids = list(FEATURE_TO_BLOCKS[feature])
        features[feature] = {
            "block_id": block_id,
            "block_ids": block_ids,
            "display_name": spec["display_name"],
            "metric_candidates": spec["metric_candidates"],
            "metric_policies": {b: list(BLOCKS[b]["metric_candidates"]) for b in block_ids},
            "output_directory": BLOCK_DIRS[block_id],
            "output_directories": {b: BLOCK_DIRS[b] for b in block_ids},
            "primary_artifact": spec["primary_artifact"],
            "primary_artifacts": {b: BLOCKS[b]["primary_artifact"] for b in block_ids},
            "purpose": spec["purpose"],
            "use_cases": spec["use_cases"],
            "cost": spec["cost"],
        }
    return {
        "schema": CONTRACT_SCHEMA,
        "default_preset": "complete",
        "presets": PRESETS,
        "features": features,
    }


def print_feature_list() -> None:
    cli_logger.info("Available msOpProf collection features\n")
    cli_logger.info(f"{'FEATURE':24} {'BLOCK':22} {'METRIC POLICY':30} {'OUTPUT'}")
    cli_logger.info("-" * 105)
    for feature, _ in FEATURE_TO_BLOCK.items():
        block_ids = FEATURE_TO_BLOCKS[feature]
        metrics = " + ".join(
            f"{b}:{' > '.join(BLOCKS[b]['metric_candidates'])}" for b in block_ids
        )
        blocks = "+".join(block_ids)
        outputs = "+".join(BLOCK_DIRS[b] for b in block_ids)
        cli_logger.info(f"{feature:24} {blocks:22} {metrics:30} {outputs}")
    cli_logger.info("\nDefault: --preset complete")
    cli_logger.info("Targeted example: --feature cache --feature timeline")


def print_feature_explanation(feature: str) -> None:
    name = normalize_feature_name(feature)
    if name not in FEATURE_TO_BLOCK:
        raise rtguard.UsageError(f"ERROR: unknown feature: {feature}. Use --list-features.")
    block_ids = FEATURE_TO_BLOCKS[name]
    block_id = block_ids[0]
    spec = BLOCKS[block_id]
    payload = {
        "feature": name,
        "collector_block": block_id,
        "collector_blocks": block_ids,
        "display_name": spec["display_name"],
        "metric_policy": {b: BLOCKS[b]["metric_candidates"] for b in block_ids},
        "fixed_output": {b: BLOCK_DIRS[b] for b in block_ids},
        "produced_data": {b: BLOCKS[b]["data"] for b in block_ids},
        "visual_features_from_same_block": spec["visual_features"],
        "purpose": spec["purpose"],
        "use_cases": spec["use_cases"],
        "cost": spec["cost"],
        "targeted_command": (
            "python scripts/collect.py --operator-path /path/to/operator "
            f"--output /path/to/output --feature {name}"
        ),
    }
    cli_logger.info(json.dumps(payload, ensure_ascii=False, indent=2))


def visualization_contract() -> Dict[str, Any]:
    return {
        "rule": ("Read collection_manifest.json first. Resolve each visual feature only from its declared "
            "collector block(s); do not scan arbitrary directories."),
        "feature_to_block": FEATURE_TO_BLOCK,
        "feature_to_blocks": FEATURE_TO_BLOCKS,
        "features": {
            "occupancy": {"block": "details", "artifact": "visualize_data"},
            "compute": {"block": "memory_detail", "artifact": "visualize_data"},
            "memory": {"block": "memory_detail", "artifact": "visualize_data"},
            "cache": {"block": "memory_detail", "artifact": "visualize_data"},
            "advice": {"block": "memory_detail", "artifact": "visualize_data"},
            "roofline": {"block": "roofline", "artifact": "visualize_data"},
            "timeline": {"block": "timeline", "artifact": "trace", "fallback_artifact": "visualize_data"},
            "source": {"block": "source", "artifact": "visualize_data",
                "optional_supplemental_block": "warp_stall", ("s"
                "upplemental_artifact"): "visualize_data", "fusion_key": "validated path+line and instruction address"},
            "warp-stall": {"block": "warp_stall", "artifact": "visualize_data"},
            "instruction-timeline": {"block": "instruction_timeline", "artifact": "visualize_data"},
            "memory-detail": {"block": "memory_detail", "artifact": "visualize_data"},
            "raw-data": {"block": "raw_data", "artifact": "csv"},
            "timeline-detail": {"block": "timeline_detail", "artifact": "visualize_data"},
            "kernel-scale": {"block": "kernel_scale", "artifact": "visualize_data"},
            "onchip-memory": {"block": "onchip_memory", "artifact": "memory_info"},
        },
    }
