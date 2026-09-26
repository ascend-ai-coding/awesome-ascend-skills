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
"""Semantic validation and artifact-inspection helpers for msOpProf collection.

Split out of collect.py to keep that module within static-check size limits.
Everything here is re-exported by collect.py, so the public collector surface
is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import runtime_guard as rtguard


STANDARD_CSV_NAMES = [
    "OpBasicInfo.csv",
    "ArithmeticUtilization.csv",
    "L2Cache.csv",
    "Memory.csv",
    "MemoryL0.csv",
    "MemoryUB.csv",
    "PipeUtilization.csv",
    "ResourceConflictRatio.csv",
]

# Core raw tables required for the Default CSV suite; OpBasicInfo is optional
# because some runs omit it.
DEFAULT_CSV_SUITE_NAMES = frozenset(STANDARD_CSV_NAMES) - {"OpBasicInfo.csv"}


def rel(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    return path.resolve().relative_to(root.resolve()).as_posix()


def normalize_metric_token(token: str) -> str:
    """Normalize spelling/case variants while preserving the exact CLI token for execution."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def split_metric_expression(expression: str) -> List[str]:
    """Split a comma-separated --aic-metrics expression into non-empty tokens."""
    return [item.strip() for item in str(expression).split(",") if item.strip()]


def metric_expression_key(expression: str) -> Tuple[str, ...]:
    """Return an order-preserving normalized key for one metric expression."""
    return tuple(normalize_metric_token(item) for item in split_metric_expression(expression))


def metric_expression_contains(expression: str, token: str) -> bool:
    target = normalize_metric_token(token)
    return any(normalize_metric_token(item) == target for item in split_metric_expression(expression))


def csv_artifact_map(result_dir: Optional[Path], root: Path) -> Dict[str, str]:
    if not result_dir:
        return {}
    out: Dict[str, str] = {}
    for p in sorted(result_dir.rglob("*.csv")):
        if p.is_file() and p.stat().st_size > 0:
            out[p.name] = rel(p, root) or ""
    return out


def _scan_simt_evidence(handle: Any, needles: Sequence[bytes]) -> bool:
    overlap = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        data = (overlap + chunk).lower()
        if any(n.lower() in data for n in needles):
            return True
        overlap = data[-128:]
    return False


def detect_simt_evidence_from_bin(bin_path: Optional[Path]) -> bool:
    if not bin_path or not bin_path.is_file():
        return False
    needles = [b"simt_vf_instructions", b"simt", b"warp_stall", b"PcSampling"]
    try:
        with bin_path.open("rb") as f:
            return _scan_simt_evidence(f, needles)
    except OSError:
        logger.debug("SIMT evidence scan failed for %s", bin_path, exc_info=True)
        return False


def _scan_bin_for_needles(handle: Any, pending: Dict[str, bytes], results: Dict[str, bool]) -> None:
    overlap = b""
    while pending:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        data = (overlap + chunk).lower()
        for needle, raw in list(pending.items()):
            if raw in data:
                results[needle] = True
                del pending[needle]
        overlap = data[-512:]


def bin_contains_all(path: Optional[Path], needles: Sequence[str]) -> Dict[str, bool]:
    """Scan ``path`` once and report which needles are present.

    Multi-needle single-pass variant of ``bin_contains_any``: the binary is read
    at most once no matter how many needles are checked.
    """
    results = {needle: False for needle in needles}
    if not path or not path.is_file() or path.stat().st_size <= 0:
        return results
    pending = {needle: needle.lower().encode("utf-8") for needle in results}
    try:
        with path.open("rb") as f:
            _scan_bin_for_needles(f, pending, results)
    except OSError:
        logger.debug("binary needle scan failed for %s", path, exc_info=True)
    return results


def bin_contains_any(path: Optional[Path], needles: Sequence[str]) -> bool:
    return any(bin_contains_all(path, needles).values())


_TRACE_EVENT_PH = re.compile(rb'"ph"\s*:\s*"[XBE]"')


def _count_trace_events(handle: Any) -> int:
    count = 0
    overlap = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        data = overlap + chunk
        # Markers fully inside the overlap were already counted with the
        # previous chunk; only count matches crossing into new data.
        boundary = len(overlap)
        count += sum(1 for match in _TRACE_EVENT_PH.finditer(data) if match.end() > boundary)
        overlap = data[-64:]
    return count


def trace_event_count(path: Optional[Path]) -> int:
    """Count complete trace events without loading the whole trace.json.

    trace.json files can be hundreds of MB, so events are counted by streaming
    chunks and matching ``"ph": "X"|"B"|"E"`` markers. Each traceEvents object
    carries exactly one ``ph`` key, so this matches the previous json.loads
    count for both compact and pretty-printed layouts.
    """
    if not path or not path.is_file() or path.stat().st_size <= 0:
        return 0
    try:
        with path.open("rb") as f:
            return _count_trace_events(f)
    except OSError:
        logger.debug("trace event count failed for %s", path, exc_info=True)
        return 0


def _empty_source_payload() -> Dict[str, Any]:
    return {
        "native_framing": False,
        "source_snapshot_count": 0,
        "unique_source_files": 0,
        "unique_source_versions": 0,
        "duplicate_source_snapshots": 0,
        "conflicting_source_paths": 0,
        "source_line_map": False,
        "mapped_line_records": 0,
        "instruction_map": False,
        "instruction_count": 0,
        "instructions_with_gpr_status": 0,
        "user_source_available": False,
        "toolchain_source_count": 0,
    }


def _parse_json_section(payload: bytes) -> Any:
    try:
        return json.loads(payload.rstrip(b"\0").decode("utf-8"))
    except Exception:
        logger.debug("unparseable JSON section in source payload", exc_info=True)
        return None


def _parse_one_section(
    data: bytes,
    offset: int,
) -> Optional[Tuple[int, int, Optional[Tuple[str, str]], Any]]:
    """Parse one framed section; return (end, section, snapshot, json_obj) or None."""
    try:
        if offset + 12 > len(data):
            return None
        payload_length = struct.unpack_from("<Q", data, offset)[0]
        section = data[offset + 8]
        if section == 1:
            end = offset + 12 + 4096 + payload_length
            if end > len(data):
                return None
            slot = data[offset + 12:offset + 12 + 4096]
            source_path = slot.split(b"\0", 1)[0].decode("utf-8", "replace").replace("\\", "/")
            payload = data[offset + 12 + 4096:end]
            digest = hashlib.sha256(payload.rstrip(b"\0")).hexdigest()
            return end, section, (source_path, digest), None
        end = offset + 12 + payload_length
        if end > len(data):
            return None
        payload = data[offset + 12:end]
        obj = _parse_json_section(payload) if section in {3, 4} else None
        return end, section, None, obj
    except Exception:
        return None


def _parse_source_sections(
    data: bytes,
) -> Optional[Tuple[List[str], List[Tuple[str, str]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Walk the native framed sections; return None on malformed framing."""
    offset = 0
    paths: List[str] = []
    source_versions: List[Tuple[str, str]] = []
    line_obj: Optional[Dict[str, Any]] = None
    instruction_obj: Optional[Dict[str, Any]] = None
    while offset < len(data):
        frame = _parse_one_section(data, offset)
        if frame is None:
            return None
        end, section, snapshot, obj = frame
        if snapshot is not None:
            paths.append(snapshot[0])
            source_versions.append(snapshot)
        elif section == 3 and isinstance(obj, dict):
            line_obj = obj
        elif section == 4 and isinstance(obj, dict):
            instruction_obj = obj
        offset = end
    if offset != len(data):
        return None
    return paths, source_versions, line_obj, instruction_obj


def _summarize_source_paths(
    result: Dict[str, Any],
    paths: List[str],
    source_versions: List[Tuple[str, str]],
) -> None:
    result["source_snapshot_count"] = len(paths)
    result["unique_source_files"] = len(set(paths))
    unique_versions = set(source_versions)
    result["unique_source_versions"] = len(unique_versions)
    result["duplicate_source_snapshots"] = max(0, len(source_versions) - len(unique_versions))
    versions_by_path: Dict[str, Set[str]] = {}
    for source_path, digest in source_versions:
        versions_by_path.setdefault(source_path, set()).add(digest)
    result["conflicting_source_paths"] = sum(1 for digests in versions_by_path.values() if len(digests) > 1)
    for source_path in set(paths):
        lower = source_path.lower()
        if any(token in lower for token in ["/usr/local/ascend/", "/bisheng_compiler/", "/tikcpp/"]):
            result["toolchain_source_count"] += 1
        else:
            result["user_source_available"] = True


def _summarize_line_map(result: Dict[str, Any], line_obj: Optional[Dict[str, Any]]) -> None:
    if not line_obj:
        return
    files = line_obj.get("Files") if isinstance(line_obj.get("Files"), list) else []
    result["source_line_map"] = bool(files)
    result["mapped_line_records"] = sum(len(item.get("Lines") or []) for item in files if isinstance(item, dict))


def _summarize_instruction_map(result: Dict[str, Any], instruction_obj: Optional[Dict[str, Any]]) -> None:
    if not instruction_obj:
        return
    instructions = instruction_obj.get("Instructions") if isinstance(instruction_obj.get("Instructions"),
        list) else []
    result["instruction_map"] = bool(instructions)
    result["instruction_count"] = len(instructions)
    result["instructions_with_gpr_status"] = sum(
        1 for item in instructions if isinstance(item, dict) and item.get("GPR Status")
    )


def inspect_source_payload(path: Optional[Path]) -> Dict[str, Any]:
    result = _empty_source_payload()
    if not path or not path.is_file() or path.stat().st_size <= 0:
        return result
    try:
        data = path.read_bytes()
    except OSError:
        return result
    sections = _parse_source_sections(data)
    if sections is None:
        return result
    paths, source_versions, line_obj, instruction_obj = sections
    result["native_framing"] = True
    _summarize_source_paths(result, paths, source_versions)
    _summarize_line_map(result, line_obj)
    _summarize_instruction_map(result, instruction_obj)
    return result


def _block_log_text(block_id: str, output_root: Path) -> str:
    log_root = output_root / "_internal" / "logs"
    if not log_root.is_dir():
        return ""
    parts: List[str] = []
    for path in sorted(log_root.glob(f"{block_id}*.log")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _source_failure_reason(lower: str) -> str:
    if "missed debug_line information" in lower or "debug_line" in lower and "miss" in lower:
        return (
            "Source mapping is empty because the profiled kernel did not contain usable .debug_line "
            "information. Rebuild the exact selected executable/kernel with line-table debug data (-g), "
            "verify that the rebuilt artifact is the one being profiled, and rerun Source."
        )
    if "hot spot function calculate data failed" in lower or "generate hot spot function failed" in lower:
        return (
            "The profiler completed, but source-hotspot analysis did not produce snapshots, line maps, "
            "or instruction maps. Verify debug-line availability for the exact executable and rerun Source."
        )
    return (
        "The Source replay returned no source snapshots, line map, or instruction map. A zero exit code "
        "does not make the feature usable; verify exact-build debug-line data and profiler Source support."
    )


def infer_semantic_failure_reason(block_id: str, output_root: Path, coverage: Dict[str, Any]) -> str:
    """Return an actionable, feature-generic reason for empty/partial output."""
    logs = _block_log_text(block_id, output_root)
    lower = logs.lower()
    if block_id == "source":
        return _source_failure_reason(lower)
    if block_id == "timeline":
        return "The timeline replay produced no valid trace intervals after semantic validation."
    if not coverage.get("visualize_data") and not coverage.get("csv_count") and not coverage.get("trace_events"):
        return "The profiler command completed without a usable artifact for this feature."
    return "The profiler artifact exists, but required semantic structures are incomplete."


def _details_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    bin_hits = bin_contains_all(viz, ["op_detail", "subblock_detail", "table_per_block", "core_memory_map"])
    coverage["occupancy"] = bin_hits["op_detail"]
    coverage["compute_memory"] = any(bin_hits[n] for n in ["subblock_detail", "table_per_block", "core_memory_map"])
    if metric_expression_contains(str(entry.get("metric") or ""), "Occupancy") and not coverage["occupancy"]:
        # Generic metadata alone is not a usable Details payload. Mark it
        # empty so the Occupancy block is not presented from generic metadata.
        return "partial" if coverage["compute_memory"] else "empty"
    return status


def _roofline_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["roofline_structure"] = bin_contains_any(viz, ["multiple_rooflines", "latency bound",
        "compute caused", "memory caused"])
    coverage["default_csv_suite"] = DEFAULT_CSV_SUITE_NAMES.issubset(csvs)
    if not coverage["roofline_structure"] and not coverage["default_csv_suite"]:
        return "empty"
    if not coverage["roofline_structure"]:
        return "partial"
    return status


def _timeline_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["timeline_payload"] = coverage["trace_events"] > 0 or bin_contains_any(viz, ["traceEvents",
        "VECTOR", "SCALAR", "MTE2", "MTE3"])
    return status if coverage["timeline_payload"] else "empty"


def _source_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    source_coverage = inspect_source_payload(viz)
    coverage.update(source_coverage)
    coverage["source_mapping"] = bool(
        source_coverage["source_snapshot_count"]
        and source_coverage["source_line_map"]
        and source_coverage["instruction_map"]
    )
    if coverage["source_mapping"]:
        return status
    has_partial_mapping = (
        source_coverage["source_snapshot_count"]
        or source_coverage["source_line_map"]
        or source_coverage["instruction_map"]
    )
    return "partial" if has_partial_mapping else "empty"


def _warp_stall_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["stall_samples"] = bin_contains_any(viz, ["stall", "PCSampling", "PcSampling", "warp"])
    return status if coverage["stall_samples"] else "empty"


def _instruction_timeline_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["instruction_timeline"] = bin_contains_any(viz, ["traceEvents", "instruction", "opcode",
        "instrTimeLine"])
    return status if coverage["instruction_timeline"] else "empty"


def _memory_detail_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["default_csv_suite"] = DEFAULT_CSV_SUITE_NAMES.issubset(csvs)
    coverage["memory_detail"] = bin_contains_any(viz, ["MemoryDetail", "active_bw", "core_memory_map",
        "table_per_block"])
    if not coverage["default_csv_suite"] and not coverage["memory_detail"]:
        return "empty"
    return status


def _raw_data_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["default_csv_suite"] = DEFAULT_CSV_SUITE_NAMES.issubset(csvs)
    if coverage["default_csv_suite"]:
        return status
    return "partial" if csvs else "empty"


def _timeline_detail_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["timeline_detail"] = bin_contains_any(viz, ["TimelineDetail", "instruction", "source", "traceEvents"])
    return status if coverage["timeline_detail"] else "empty"


def _kernel_scale_status(status: str, entry: Dict[str, Any], viz: Optional[Path], csvs: Set[str],
    coverage: Dict[str, Any]) -> str:
    coverage["kernel_scale"] = bin_contains_any(viz, ["KernelScale",
        "MetricsProfStart", "MetricsProfStop", "kernel_scale"])
    return status if coverage["kernel_scale"] else "empty"


_BLOCK_STATUS_CHECKS = {
    "details": _details_status,
    "roofline": _roofline_status,
    "timeline": _timeline_status,
    "source": _source_status,
    "warp_stall": _warp_stall_status,
    "instruction_timeline": _instruction_timeline_status,
    "memory_detail": _memory_detail_status,
    "raw_data": _raw_data_status,
    "timeline_detail": _timeline_detail_status,
    "kernel_scale": _kernel_scale_status,
}


def _initial_coverage(viz: Optional[Path], trace: Optional[Path], csvs: Set[str]) -> Dict[str, Any]:
    return {
        "visualize_data": bool(viz and viz.is_file() and viz.stat().st_size > 0),
        "trace_events": trace_event_count(trace),
        "csv_count": len(csvs),
    }


def validate_block_entry(block_id: str, entry: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    artifacts = entry.get("artifacts") or {}
    viz_rel = artifacts.get("visualize_data")
    trace_rel = artifacts.get("trace")
    viz = output_root / viz_rel if viz_rel else None
    trace = output_root / trace_rel if trace_rel else None
    csvs = set((artifacts.get("csv") or {}).keys())
    coverage = _initial_coverage(viz, trace, csvs)

    if entry.get("status") not in {"ok", "reused", "aliased"}:
        entry["coverage"] = coverage
        return entry

    status = entry.get("status")
    check = _BLOCK_STATUS_CHECKS.get(block_id)
    if check is not None:
        status = check(status, entry, viz, csvs, coverage)

    entry["status"] = status
    entry["coverage"] = coverage
    if status in {"partial", "empty"} and not entry.get("reason"):
        entry["reason"] = infer_semantic_failure_reason(block_id, output_root, coverage)
    return entry


def _validate_mstx_include(mstx_include: Any, mstx: Any) -> None:
    if mstx_include and not re.fullmatch(r"[A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*", str(mstx_include)):
        raise rtguard.UsageError(
            "ERROR: --mstx-include accepts only A-Z, a-z, 0-9, underscore, "
            "and pipe-separated message names."
        )
    if mstx_include and mstx != "on":
        raise rtguard.UsageError("ERROR: --mstx-include requires --mstx=on.")


def _validate_core_id(core_id: Any) -> None:
    core_tokens = str(core_id).split("|")
    if not core_tokens or any(not token.isdigit() or not 0 <= int(token) <= 49 for token in core_tokens):
        raise rtguard.UsageError("ERROR: --core-id must contain pipe-separated logical core IDs in [0, 49].")


def _validate_range_mode(metric: str, mstx: Any, warm_up: Any, kill: Any) -> None:
    if mstx != "on":
        raise rtguard.UsageError("ERROR: --replay-mode=range requires --mstx=on.")
    if warm_up == 0:
        raise rtguard.UsageError("ERROR: --warm-up=0 is invalid with --replay-mode=range.")
    if kill == "on":
        raise rtguard.UsageError("ERROR: --kill=on is incompatible with --replay-mode=range.")
    incompatible: List[str] = []
    for name in ["MemoryDetail", "TimelineDetail", "Source"]:
        if metric_expression_contains(metric, name):
            incompatible.append(name)
    if incompatible:
        raise rtguard.UsageError(
            "ERROR: --replay-mode=range is incompatible with --aic-metrics=" + ",".join(incompatible) + "."
        )


def _validate_dump_scope(metric: str, dump: Any, core_id: Any) -> None:
    if (dump is not None or core_id is not None) and not metric_expression_contains(metric, "TimelineDetail"):
        raise rtguard.UsageError("ERROR: --dump/--core-id are valid only for TimelineDetail collection.")


def validate_profiler_options(metric: str, profiler_options: Mapping[str, Any]) -> None:
    """Validate documented msprof option/metric compatibility before execution.

    Raises ``rtguard.UsageError`` with the ERROR text; the CLI ``main`` entry
    points convert it into a logged error and return code 1.
    """
    replay_mode = profiler_options.get("replay_mode")
    mstx = profiler_options.get("mstx")
    mstx_include = profiler_options.get("mstx_include")
    warm_up = profiler_options.get("warm_up")
    kill = profiler_options.get("kill")
    dump = profiler_options.get("dump")
    core_id = profiler_options.get("core_id")

    _validate_mstx_include(mstx_include, mstx)
    if core_id is not None:
        _validate_core_id(core_id)
    if replay_mode == "range":
        _validate_range_mode(metric, mstx, warm_up, kill)
    if replay_mode == "application" and metric_expression_contains(metric, "TimelineDetail"):
        raise rtguard.UsageError("ERROR: TimelineDetail is incompatible with --replay-mode=application.")
    _validate_dump_scope(metric, dump, core_id)


def extract_log_diagnostics(path: Path, limit: int = 20) -> List[str]:
    """Return concise warning/error lines while preserving the full log on disk."""
    if not path.is_file():
        return []
    diagnostics: List[str] = []
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            lower = line.lower()
            if "warning" not in lower and "error" not in lower and "failed to analyze" not in lower:
                continue
            diagnostics.append(line[:1000])
            if len(diagnostics) >= limit:
                break
    except OSError:
        logger.debug("log diagnostics read failed for %s", path, exc_info=True)
        return []
    return diagnostics


def has_standard_csv_suite(entry: Dict[str, Any]) -> bool:
    csvs = set((entry.get("artifacts") or {}).get("csv", {}).keys())
    return DEFAULT_CSV_SUITE_NAMES.issubset(csvs)


def fallback_allowed(entry: Mapping[str, Any]) -> bool:
    """Only semantic emptiness after a successful command may trigger fallback."""
    return (
        entry.get("return_code") == 0
        and entry.get("status") == "empty"
        and not entry.get("timed_out")
    )
