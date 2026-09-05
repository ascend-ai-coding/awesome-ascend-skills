# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import logging
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple
from visualize_common import ArtifactSet, iter_dicts, positive, to_float
from visualize_parsers import parse_trace_events

logger = logging.getLogger(__name__)

SOURCE_PATH_SLOT_BYTES = 4096


SOURCE_FRAME_FILE = 1


SOURCE_FRAME_LINE_MAP = 3


SOURCE_FRAME_INSTRUCTION_MAP = 4


def _decode_native_frame(data: bytes, offset: int) -> Optional[Tuple[Dict[str, Any], int]]:
    """Decode one length-framed record; ``None`` marks malformed framing."""
    if offset + 12 > len(data):
        return None
    payload_length = struct.unpack_from("<Q", data, offset)[0]
    section = data[offset + 8]
    subtype = data[offset + 9]
    marker = data[offset + 10:offset + 12]
    if payload_length > len(data):
        return None
    if section == SOURCE_FRAME_FILE:
        end = offset + 12 + SOURCE_PATH_SLOT_BYTES + payload_length
        if end > len(data):
            return None
        path_slot = data[offset + 12:offset + 12 + SOURCE_PATH_SLOT_BYTES]
        source_path = path_slot.split(b"\0", 1)[0].decode("utf-8", "replace")
        payload = data[offset + 12 + SOURCE_PATH_SLOT_BYTES:end]
    else:
        end = offset + 12 + payload_length
        if end > len(data):
            return None
        source_path = None
        payload = data[offset + 12:end]
    frame = {
        "offset": offset,
        "payload_length": int(payload_length),
        "section": int(section),
        "subtype": int(subtype),
        "marker": marker.hex(),
        "source_path": source_path,
        "payload": payload,
    }
    return frame, end


def parse_native_visualize_frames(path: Optional[Path], data: Optional[bytes] = None) -> List[Dict[str, Any]]:
    """Parse the native length-framed visualize_data container.

    Source snapshots (section 1) contain a fixed 4096-byte path slot followed by
    ``payload_length`` bytes of file content. Other sections contain exactly
    ``payload_length`` bytes. Unknown sections are retained; malformed framing
    returns an empty list instead of scanning arbitrary bytes as source data.
    Pre-read ``data`` may be supplied so callers that need several views of the
    same artifact only read the file once.
    """
    if data is None:
        if not path or not path.is_file() or path.stat().st_size <= 0:
            return []
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.debug("Failed to read native visualize frames from %s: %s", path, exc)
            return []
    frames: List[Dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        decoded = _decode_native_frame(data, offset)
        if decoded is None:
            return []
        frame, offset = decoded
        frames.append(frame)
    return frames if offset == len(data) else []


def _frame_json(frames: Sequence[Mapping[str, Any]], section: int) -> Optional[Dict[str, Any]]:
    for frame in frames:
        if frame.get("section") != section:
            continue
        payload = frame.get("payload")
        if not isinstance(payload, (bytes, bytearray)):
            continue
        try:
            obj = json.loads(bytes(payload).rstrip(b"\0").decode("utf-8"))
        except ValueError:
            # Non-JSON frame payloads are expected; scan on for a JSON section.
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _source_path_key(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def _source_location(value: Any) -> Tuple[str, Optional[int]]:
    text = str(value or "")
    match = re.match(r"^(.*):(\d+)(?::\d+)?$", text)
    if not match:
        return _source_path_key(text), None
    return _source_path_key(match.group(1)), int(match.group(2))


def _address_int(value: Any) -> Optional[int]:
    try:
        text = str(value).strip().lower()
        return int(text, 16) if text.startswith("0x") else int(text)
    except (TypeError, ValueError):
        return None


def _scalar_sum(value: Any) -> float:
    if isinstance(value, list):
        total = 0.0
        for item in value:
            number = to_float(item)
            if number is not None:
                total += number
        return total
    return to_float(value) or 0.0


def _per_core_values(value: Any, cores: Sequence[str]) -> Dict[str, float]:
    names = [str(x) for x in cores] or ["ALL"]
    values = value if isinstance(value, list) else [value]
    numbers = [(to_float(v) or 0.0) for v in values]
    out: Dict[str, float] = {}
    if len(numbers) == len(names):
        out.update({name: numbers[i] for i, name in enumerate(names)})
    elif len(numbers) == 1:
        out[names[0] if len(names) == 1 else "ALL"] = numbers[0]
    else:
        out["ALL"] = sum(numbers)
    if "ALL" not in out:
        out["ALL"] = sum(out.values())
    return out


def _source_kind(path: str) -> Tuple[str, bool, int]:
    lower = path.lower()
    suffix = Path(path).suffix.lower()
    toolchain = any(token in lower for token in ["/usr/local/ascend/", "/bisheng_compiler/", "/tikcpp/"])
    user = not toolchain
    if user and suffix in {".asc", ".c", ".cc", ".cpp", ".cxx"}:
        return "User source", True, 0
    if user:
        return "User include", True, 1
    if suffix in {".h", ".hpp", ".inc"}:
        return "Toolchain header", False, 3
    return "Generated / toolchain source", False, 4


def _stall_metric(obj: Any) -> Dict[str, Any]:
    """Normalize one native stall cell.

    Native ``Percent`` values are normally fractions in [0, 1], while some
    versions may emit percentage points. The canonical model always stores
    percentage points and preserves every numeric reason count.
    """
    if not isinstance(obj, Mapping):
        return {"percent": 0.0, "ratio": 0.0, "total_samples": 0.0, "details": {}, "segments": []}
    raw_percent = to_float(obj.get("Percent")) or 0.0
    percent = raw_percent * 100.0 if abs(raw_percent) <= 1.0000001 else raw_percent
    details: Dict[str, float] = {}
    raw_details = obj.get("Details")
    if isinstance(raw_details, Mapping):
        for key, value in raw_details.items():
            number = to_float(value)
            if number is not None:
                details[str(key)] = max(0.0, number)
    total = sum(details.values())
    segments = [
        {
            "reason": reason,
            "count": count,
            "share_percent": (count / total * 100.0) if total > 0 else 0.0,
        }
        for reason, count in details.items()
        if count > 0
    ]
    return {
        "percent": max(0.0, percent),
        "ratio": max(0.0, percent) / 100.0,
        "total_samples": total,
        "details": details,
        "segments": segments,
    }


def _stall_percent(obj: Any) -> float:
    return _stall_metric(obj)["percent"]


def _stall_score(metric: Mapping[str, Any]) -> Tuple[float, float, int]:
    return (
        to_float(metric.get("total_samples")) or 0.0,
        to_float(metric.get("percent")) or 0.0,
        len(metric.get("segments") or []),
    )


def _preferred_stall(primary: Any, supplemental: Any) -> Tuple[Dict[str, Any], bool]:
    base = _stall_metric(primary)
    extra = _stall_metric(supplemental)
    if _stall_score(extra) > _stall_score(base):
        return extra, True
    return base, False


def _raw_source_line_lookup(source_map: Mapping[str, Any]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    out: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for file_entry in source_map.get("Files") or []:
        if not isinstance(file_entry, Mapping):
            continue
        path = _source_path_key(file_entry.get("Source"))
        for raw_line in file_entry.get("Lines") or []:
            if not isinstance(raw_line, Mapping):
                continue
            line_no = int(to_float(raw_line.get("Line")) or 0)
            if path and line_no > 0:
                out[(path, line_no)] = raw_line
    return out


def _raw_instruction_signature(raw: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(raw.get("Address") or "").strip().lower(),
        str(raw.get("Pipe") or "").strip().upper(),
        str(raw.get("Source") or "").strip(),
        str(raw.get("AscendC Inner Code") or "").strip(),
    )


def _instruction_semantic_signature(raw: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(raw.get("Pipe") or "").strip().upper(),
        str(raw.get("Source") or "").strip(),
        str(raw.get("AscendC Inner Code") or "").strip(),
    )


def _fusion_instruction_lists(
    primary_instruction_map: Mapping[str, Any],
    supplemental_instruction_map: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    primary = [x for x in (primary_instruction_map.get("Instructions") or []) if isinstance(x, Mapping)]
    supplemental = [x for x in (supplemental_instruction_map.get("Instructions") or []) if isinstance(x, Mapping)]
    return primary, supplemental


def _instruction_address_set(instructions: Sequence[Mapping[str, Any]]) -> set:
    addresses = {_address_int(x.get("Address")) for x in instructions}
    addresses.discard(None)
    return addresses


def _semantic_address_index(instructions: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], List[int]]:
    by_sig: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for item in instructions:
        addr = _address_int(item.get("Address"))
        if addr is not None:
            by_sig[_instruction_semantic_signature(item)].append(addr)
    return by_sig


def _relocation_delta(
    p_by_sig: Mapping[Tuple[str, str, str], List[int]],
    s_by_sig: Mapping[Tuple[str, str, str], List[int]],
    primary_addresses: set,
    supplemental_addresses: set,
) -> Tuple[int, int]:
    delta_votes: Dict[int, int] = defaultdict(int)
    for signature in set(p_by_sig) & set(s_by_sig):
        pvals, svals = p_by_sig[signature], s_by_sig[signature]
        if len(pvals) == 1 and len(svals) == 1:
            delta_votes[svals[0] - pvals[0]] += 1
    if delta_votes:
        relocation_delta = max(delta_votes, key=lambda d: (delta_votes[d], -abs(d)))
        return relocation_delta, delta_votes[relocation_delta]
    if primary_addresses and supplemental_addresses:
        return min(supplemental_addresses) - min(primary_addresses), 0
    return 0, 0


def _overlap_ratio(left: set, right: set) -> float:
    denom = min(len(left), len(right))
    return len(left & right) / denom if denom else 0.0


def _fusion_reason(accepted: bool, relocation_delta: int) -> str:
    if accepted:
        relocation_text = f" with uniform address relocation {relocation_delta:+#x}" if relocation_delta else ""
        return (f"accepted: compatible source-line, semantic-instruction and normalized-address "
            f"identities{relocation_text}")
    return ("rejected: supplemental capture did not meet source-line/semantic/normalized-address "
        "overlap thresholds")


def _source_fusion_compatibility(
    primary_source_map: Mapping[str, Any],
    primary_instruction_map: Mapping[str, Any],
    supplemental_source_map: Mapping[str, Any],
    supplemental_instruction_map: Mapping[str, Any],
) -> Dict[str, Any]:
    primary_lines = set(_raw_source_line_lookup(primary_source_map))
    supplemental_lines = set(_raw_source_line_lookup(supplemental_source_map))
    primary_instructions, supplemental_instructions = _fusion_instruction_lists(
        primary_instruction_map, supplemental_instruction_map)
    primary_addresses = _instruction_address_set(primary_instructions)
    supplemental_addresses = _instruction_address_set(supplemental_instructions)

    # Source and PCSampling are separate replays. The same instruction stream
    # may be relocated to another runtime base address. Infer a dominant
    # relocation delta from unique semantic signatures, then validate the full
    # normalized address sets. This avoids both absolute-address assumptions
    # and unsafe list-ordinal joins.
    p_by_sig = _semantic_address_index(primary_instructions)
    s_by_sig = _semantic_address_index(supplemental_instructions)
    relocation_delta, relocation_vote_count = _relocation_delta(
        p_by_sig, s_by_sig, primary_addresses, supplemental_addresses)

    normalized_supplemental_addresses = {addr - relocation_delta for addr in supplemental_addresses}
    line_overlap = _overlap_ratio(primary_lines, supplemental_lines)
    address_overlap = _overlap_ratio(primary_addresses, normalized_supplemental_addresses)
    semantic_primary = {_instruction_semantic_signature(x) for x in primary_instructions}
    semantic_supplemental = {_instruction_semantic_signature(x) for x in supplemental_instructions}
    semantic_overlap = _overlap_ratio(semantic_primary, semantic_supplemental)
    has_supplemental = bool(supplemental_lines and normalized_supplemental_addresses)
    line_overlap_ok = line_overlap >= 0.70
    address_overlap_ok = address_overlap >= 0.70
    semantic_overlap_ok = semantic_overlap >= 0.70
    accepted = bool(has_supplemental and line_overlap_ok and address_overlap_ok and semantic_overlap_ok)
    reason = _fusion_reason(accepted, relocation_delta)
    return {
        "available": bool(supplemental_lines or supplemental_addresses),
        "accepted": accepted,
        "line_overlap_ratio": line_overlap,
        "instruction_address_overlap_ratio": address_overlap,
        "instruction_semantic_overlap_ratio": semantic_overlap,
        "address_relocation_delta": relocation_delta,
        "relocation_vote_count": relocation_vote_count,
        "primary_line_count": len(primary_lines),
        "supplemental_line_count": len(supplemental_lines),
        "primary_instruction_address_count": len(primary_addresses),
        "supplemental_instruction_address_count": len(supplemental_addresses),
        "reason": reason,
        "merged_line_metrics": 0,
        "merged_instruction_metrics": 0,
    }


def _no_supplemental_fusion() -> Dict[str, Any]:
    return {
        "available": False,
        "accepted": False,
        "line_overlap_ratio": 0.0,
        "instruction_address_overlap_ratio": 0.0,
        "reason": "no supplemental PCSampling artifact",
        "merged_line_metrics": 0,
        "merged_instruction_metrics": 0,
    }


def _source_frame_maps(frames: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        _frame_json(frames, SOURCE_FRAME_LINE_MAP) or {},
        _frame_json(frames, SOURCE_FRAME_INSTRUCTION_MAP) or {},
    )


def _source_snapshots(frames: Sequence[Mapping[str, Any]]) -> Dict[str, List[bytes]]:
    snapshots: Dict[str, List[bytes]] = defaultdict(list)
    for frame in frames:
        if frame.get("section") == SOURCE_FRAME_FILE and frame.get("source_path"):
            snapshots[_source_path_key(frame["source_path"])].append(bytes(frame.get("payload") or b""))
    return snapshots


def _core_names(mapping: Mapping[str, Any]) -> List[str]:
    return [str(x) for x in (mapping.get("Cores") or [])]


def _empty_stall() -> Dict[str, Any]:
    return {"percent": 0.0, "ratio": 0.0, "total_samples": 0.0, "details": {}, "segments": []}


def _source_line_obj(number: int, text: str) -> Dict[str, Any]:
    return {
        "number": number,
        "text": text,
        "mapped": False,
        "instructions_executed_by_core": {"ALL": 0.0},
        "gpr_count": None,
        "address_ranges": [],
        "related_instruction_ids": [],
        "stall_all": _empty_stall(),
        "stall_not_issue": _empty_stall(),
        "stall_all_percent": 0.0,
        "stall_not_issue_percent": 0.0,
    }


def _source_file_obj(index: int, source_path: str, candidates: Sequence[bytes]) -> Dict[str, Any]:
    version_groups: Dict[str, List[bytes]] = defaultdict(list)
    for candidate in candidates:
        digest = hashlib.sha256(candidate.rstrip(b"\0")).hexdigest()
        version_groups[digest].append(candidate)
    ranked_versions = sorted(version_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    content_bytes = ranked_versions[0][1][0] if ranked_versions else b""
    content_versions = [
        {"content_hash": digest, "snapshot_count": len(items),
            "byte_size": len(items[0].rstrip(b"\0")) if items else 0}
        for digest, items in ranked_versions
    ]
    content = content_bytes.rstrip(b"\0").decode("utf-8", "replace")
    lines = content.split("\n") if content else []
    if lines and lines[-1] == "":
        lines = lines[:-1]
    kind, is_user, _ = _source_kind(source_path)
    return {
        "id": f"file-{index}",
        "path": source_path,
        "name": Path(source_path).name or source_path,
        "kind": kind,
        "is_user_source": is_user,
        "language": Path(source_path).suffix.lower().lstrip(".") or "text",
        "line_count": len(lines),
        "snapshot_count": len(candidates),
        "content_hash": hashlib.sha256(content_bytes.rstrip(b"\0")).hexdigest() if content_bytes else None,
        "content_versions": content_versions,
        "version_conflict": len(content_versions) > 1,
        "lines": [_source_line_obj(i + 1, text) for i, text in enumerate(lines)],
    }


def _source_files(
    snapshots: Mapping[str, List[bytes]],
    raw_files: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    mapped_paths = [_source_path_key(item.get("Source")) for item in raw_files if isinstance(item, Mapping)]
    all_paths: List[str] = []
    for candidate in list(snapshots) + mapped_paths:
        if candidate and candidate not in all_paths:
            all_paths.append(candidate)
    ranked_paths = sorted(all_paths, key=lambda p: (_source_kind(p)[2], p.lower()))
    files: List[Dict[str, Any]] = []
    file_by_path: Dict[str, Dict[str, Any]] = {}
    line_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for index, source_path in enumerate(ranked_paths):
        file_obj = _source_file_obj(index, source_path, snapshots.get(source_path) or [])
        files.append(file_obj)
        file_by_path[source_path] = file_obj
    for file_obj in files:
        for line in file_obj["lines"]:
            line_index[(file_obj["path"], int(line["number"]))] = line
    return files, file_by_path, line_index


@dataclass
class _SourceBuildState:
    fusion: Dict[str, Any]
    source_cores: List[str]
    instruction_cores: List[str]
    supplemental_lines: Mapping[Tuple[str, int], Mapping[str, Any]]
    supplemental_by_address: Dict[int, List[Mapping[str, Any]]]
    file_by_path: Dict[str, Dict[str, Any]]
    line_index: Dict[Tuple[str, int], Dict[str, Any]]
    stall_reasons: List[str]
    stall_reason_totals: Dict[str, float]


def _source_add_address_ranges(line: Dict[str, Any], pairs: Any) -> None:
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        start, end = _address_int(pair[0]), _address_int(pair[1])
        if start is None or end is None:
            continue
        item = {"start": str(pair[0]), "end": str(pair[1]), "start_value": start, "end_value": end}
        if item not in line["address_ranges"]:
            line["address_ranges"].append(item)


def _source_map_line(raw_line: Any, source_path: str, file_obj: Dict[str, Any], state: _SourceBuildState) -> bool:
    if not isinstance(raw_line, Mapping):
        return False
    line_no = int(to_float(raw_line.get("Line")) or 0)
    if line_no <= 0:
        return False
    while len(file_obj["lines"]) < line_no:
        n = len(file_obj["lines"]) + 1
        placeholder = _source_line_obj(n, "")
        file_obj["lines"].append(placeholder)
        state.line_index[(source_path, n)] = placeholder
    line = state.line_index[(source_path, line_no)]
    line["mapped"] = True
    values = _per_core_values(raw_line.get("Instructions Executed"), state.source_cores or ["ALL"])
    for core, value in values.items():
        line["instructions_executed_by_core"][core] = line["instructions_executed_by_core"].get(core,
            0.0) + value
    gpr = to_float(raw_line.get("GPR Count"))
    if gpr is not None:
        line["gpr_count"] = max(gpr, line["gpr_count"] or 0.0)
    _source_add_address_ranges(line, raw_line.get("Address Range") or [])
    supplement = state.supplemental_lines.get((source_path, line_no))
    all_metric, all_merged = _preferred_stall(
        raw_line.get("Stall Sampling(All Samples)"),
        supplement.get("Stall Sampling(All Samples)") if supplement else None,
    )
    issue_metric, issue_merged = _preferred_stall(
        raw_line.get("Stall Sampling(Not Issue)"),
        supplement.get("Stall Sampling(Not Issue)") if supplement else None,
    )
    line["stall_all"] = all_metric
    line["stall_not_issue"] = issue_metric
    line["stall_all_percent"] = all_metric["percent"]
    line["stall_not_issue_percent"] = issue_metric["percent"]
    if all_merged or issue_merged:
        state.fusion["merged_line_metrics"] += 1
    for metric in (all_metric, issue_metric):
        for reason, count in metric.get("details", {}).items():
            if reason not in state.stall_reasons:
                state.stall_reasons.append(reason)
            state.stall_reason_totals[reason] += count
    return True


def _source_map_lines(raw_files: Sequence[Any], state: _SourceBuildState) -> int:
    source_line_rows = 0
    for file_entry in raw_files:
        if not isinstance(file_entry, Mapping):
            continue
        source_path = _source_path_key(file_entry.get("Source"))
        file_obj = state.file_by_path.get(source_path)
        if not file_obj:
            continue
        for raw_line in file_entry.get("Lines") or []:
            if _source_map_line(raw_line, source_path, file_obj, state):
                source_line_rows += 1
    return source_line_rows


def _gpr_statuses(raw_inst: Mapping[str, Any]) -> List[Dict[str, Any]]:
    statuses = []
    for status in raw_inst.get("GPR Status") or []:
        if isinstance(status, Mapping):
            statuses.append({
                "register": str(status.get("regIndex") or ""),
                "status": int(to_float(status.get("regStatus")) or 0),
                "survival_time": int(to_float(status.get("survivalTime")) or 0),
            })
    return statuses


def _instruction_supplement(
    raw_inst: Mapping[str, Any],
    address: str,
    supplemental_by_address: Mapping[int, List[Mapping[str, Any]]],
) -> Optional[Mapping[str, Any]]:
    primary_address = _address_int(address)
    by_address = supplemental_by_address.get(primary_address) or [] if primary_address is not None else []
    if not by_address:
        return None
    semantic = _instruction_semantic_signature(raw_inst)
    semantic_matches = [x for x in by_address if _instruction_semantic_signature(x) == semantic]
    if len(semantic_matches) == 1:
        return semantic_matches[0]
    if len(by_address) == 1:
        return by_address[0]
    return None


def _source_instruction(index: int, raw_inst: Any, state: _SourceBuildState) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_inst, Mapping):
        return None
    source_path, source_line = _source_location(raw_inst.get("AscendC Inner Code"))
    address = str(raw_inst.get("Address") or "")
    supplement = _instruction_supplement(raw_inst, address, state.supplemental_by_address)
    all_metric, all_merged = _preferred_stall(
        raw_inst.get("Stall Sampling(All Samples)"),
        supplement.get("Stall Sampling(All Samples)") if supplement else None,
    )
    issue_metric, issue_merged = _preferred_stall(
        raw_inst.get("Stall Sampling(Not Issue)"),
        supplement.get("Stall Sampling(Not Issue)") if supplement else None,
    )
    if all_merged or issue_merged:
        state.fusion["merged_instruction_metrics"] += 1
    return {
        "id": f"inst-{index}",
        "index": index + 1,
        "address": address,
        "address_value": _address_int(address),
        "pipe": str(raw_inst.get("Pipe") or ""),
        "opcode": str(raw_inst.get("Source") or ""),
        "source_location": str(raw_inst.get("AscendC Inner Code") or ""),
        "source_path": source_path,
        "source_line": source_line,
        "instructions_executed_by_core": _per_core_values(raw_inst.get("Instructions Executed"),
            state.instruction_cores or [("A"
            "LL")]),
        "gpr_count": to_float(raw_inst.get("GPR Count")) or 0.0,
        "gpr_status": _gpr_statuses(raw_inst),
        "process_bytes": to_float(raw_inst.get("Process Bytes")),
        "stall_all": all_metric,
        "stall_not_issue": issue_metric,
        "stall_all_percent": all_metric["percent"],
        "stall_not_issue_percent": issue_metric["percent"],
        "related_lines": [],
    }


def _source_interval_index(state: _SourceBuildState) -> Dict[str, Tuple[List[int], List[Tuple[int, int, int]]]]:
    # Per-file sorted (start, end, line_no) interval lists, built once, so each
    # instruction bisects candidate address ranges instead of scanning every
    # mapped line of every file (previously O(instructions x lines)).
    intervals_by_path: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for (range_path, range_line_no), range_line in state.line_index.items():
        if not range_line.get("mapped"):
            continue
        for address_range in range_line.get("address_ranges") or []:
            intervals_by_path[range_path].append(
                (address_range["start_value"], address_range["end_value"], range_line_no)
            )
    interval_index: Dict[str, Tuple[List[int], List[Tuple[int, int, int]]]] = {}
    for range_path, entries in intervals_by_path.items():
        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        interval_index[range_path] = ([entry[0] for entry in entries], entries)
    return interval_index


def _instruction_relation_candidates(
    inst: Mapping[str, Any],
    interval_index: Mapping[str, Tuple[List[int], List[Tuple[int, int, int]]]],
    state: _SourceBuildState,
) -> Dict[Tuple[str, int], set]:
    candidates: Dict[Tuple[str, int], set] = defaultdict(set)
    if inst.get("source_line") and (inst["source_path"], int(inst["source_line"])) in state.line_index:
        candidates[(inst["source_path"], int(inst["source_line"]))].add("exact_location")
    address_value = inst.get("address_value")
    if address_value is None:
        return candidates
    for range_path, (starts, entries) in interval_index.items():
        # All intervals that can contain the address start at or before
        # it; bisect once and then only test the end bound.
        for start, end, line_no in entries[:bisect.bisect_right(starts, address_value)]:
            if start <= address_value <= end:
                candidates[(range_path, line_no)].add("address_range")
    return candidates


def _source_relations(instructions: Sequence[Dict[str, Any]], state: _SourceBuildState) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    relation_seen: set = set()
    interval_index = _source_interval_index(state)
    for inst in instructions:
        candidates = _instruction_relation_candidates(inst, interval_index, state)
        for (source_path, line_no), reasons in candidates.items():
            file_obj = state.file_by_path.get(source_path)
            if not file_obj:
                continue
            relation_key = (inst["id"], file_obj["id"], line_no)
            if relation_key in relation_seen:
                continue
            relation_seen.add(relation_key)
            rel = {"instruction_id": inst["id"], "file_id": file_obj["id"],
                "source_path": source_path, "line": line_no, ("r"
                "easons"): sorted(reasons)}
            relations.append(rel)
            inst["related_lines"].append({"file_id": file_obj["id"], "line": line_no, "reasons": sorted(reasons)})
            line = state.line_index[(source_path, line_no)]
            if inst["id"] not in line["related_instruction_ids"]:
                line["related_instruction_ids"].append(inst["id"])
    return relations


def _file_related_instruction_count(file_obj: Mapping[str, Any]) -> int:
    instruction_ids = set()
    for line in file_obj["lines"]:
        for iid in line.get("related_instruction_ids") or []:
            instruction_ids.add(iid)
    return len(instruction_ids)


def _source_default_focus(files: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    for file_obj in files:
        file_obj["mapped_line_count"] = sum(1 for line in file_obj["lines"] if line.get("mapped"))
        file_obj["related_instruction_count"] = _file_related_instruction_count(file_obj)
    default_file = next((f for f in files if f["is_user_source"] and f["mapped_line_count"]),
        None) or next((f for f in files if f["mapped_line_count"]), None) or (files[0] if files else None)
    default_line = None
    if default_file:
        mapped = [line for line in default_file["lines"] if line.get("mapped")]
        if mapped:
            default_line = max(mapped, key=lambda line: (
                line.get("stall_all_percent") or 0.0,
                line.get("stall_not_issue_percent") or 0.0,
                len(line.get("related_instruction_ids") or []),
                line["instructions_executed_by_core"].get("ALL", 0.0),
                -line["number"],
            ))["number"]
    return default_file, default_line


def _stall_has_content(metric: Any) -> bool:
    if not isinstance(metric, Mapping):
        return False
    if (metric.get("total_samples") or 0) > 0 or (metric.get("percent") or 0) > 0:
        return True
    if metric.get("details"):
        return True
    return any((segment.get("count") or 0) > 0 for segment in metric.get("segments") or [])


def _strip_empty_stall_metrics(files: Sequence[Dict[str, Any]], instructions: Sequence[Dict[str, Any]]) -> None:
    # Payload economy: the flat stall_all_percent/stall_not_issue_percent keys
    # duplicate the nested metric's .percent, and a completely empty stall
    # metric renders identically through the template's defaulting
    # (sourceStallMetric -> 0.00% with no segments). Strip both after every
    # internal consumer (default-line ranking, summary counters) has run.
    for file_obj in files:
        for line in file_obj["lines"]:
            line.pop("stall_all_percent", None)
            line.pop("stall_not_issue_percent", None)
            if not _stall_has_content(line.get("stall_all")):
                line.pop("stall_all", None)
            if not _stall_has_content(line.get("stall_not_issue")):
                line.pop("stall_not_issue", None)
    for inst in instructions:
        inst.pop("stall_all_percent", None)
        inst.pop("stall_not_issue_percent", None)
        if not _stall_has_content(inst.get("stall_all")):
            inst.pop("stall_all", None)
        if not _stall_has_content(inst.get("stall_not_issue")):
            inst.pop("stall_not_issue", None)


class _SourceTally(NamedTuple):
    line_rows: int
    stall_reason_totals: Mapping[str, float]


def _source_summary(
    snapshots: Mapping[str, List[bytes]],
    files: Sequence[Dict[str, Any]],
    instructions: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    tally: _SourceTally,
) -> Dict[str, Any]:
    pipe_counts: Dict[str, int] = defaultdict(int)
    for inst in instructions:
        pipe_counts[inst["pipe"] or "Unknown"] += 1
    unique_source_versions = 0
    conflicting_source_paths = 0
    for items in snapshots.values():
        version_hashes = {hashlib.sha256(item.rstrip(b"\0")).hexdigest() for item in items}
        unique_source_versions += len(version_hashes)
        if len(version_hashes) > 1:
            conflicting_source_paths += 1
    snapshot_count = sum(len(v) for v in snapshots.values())
    stall_sampled_instructions = 0
    for inst in instructions:
        total_samples = (inst.get("stall_all") or {}).get(("tota"
            "l_samples")) or 0
        if total_samples > 0:
            stall_sampled_instructions += 1
    return {
        "source_snapshot_count": snapshot_count,
        "unique_source_files": len(files),
        "unique_source_versions": unique_source_versions,
        "duplicate_source_snapshots": max(0, snapshot_count - unique_source_versions),
        "conflicting_source_paths": conflicting_source_paths,
        "mapped_line_records": tally.line_rows,
        "mapped_lines": sum(f["mapped_line_count"] for f in files),
        "instruction_count": len(instructions),
        "relation_count": len(relations),
        "user_source_files": sum(1 for f in files if f["is_user_source"]),
        "instructions_with_gpr_status": sum(1 for i in instructions if i["gpr_status"]),
        "instructions_with_stall_samples": stall_sampled_instructions,
        "stall_metrics_available": any(value > 0 for value in tally.stall_reason_totals.values()),
        "stall_reason_totals": dict(tally.stall_reason_totals),
        "pipe_counts": dict(sorted(pipe_counts.items())),
    }


def _source_contract() -> Dict[str, Any]:
    return {
        "file_deduplication": ("normalized path + content hash; repeated snapshots are retained as "
            "snapshot_count metadata; conflicting versions at one path remain auditable"),
        "line_binding": ("source path + line is primary; address-range membership is preserved as an "
            "additional relation"),
        "instruction_binding": ("instruction address and AscendC Inner Code location; ambiguous/inlined "
            "relations are not collapsed"),
        "gpr_status": ("raw regStatus and survivalTime values are visualized without assigning "
            "undocumented semantic names"),
        "stall_fusion": ("PCSampling is optional; fusion requires validated source-line and "
            "instruction-address overlap and never joins by list ordinal"),
        "stall_units": "canonical stall percentages are percentage points; native reason counts are preserved",
    }


def _supplemental_address_index(
    instruction_map: Mapping[str, Any],
    relocation_delta: int,
    accepted: bool,
) -> Dict[int, List[Mapping[str, Any]]]:
    by_address: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    if accepted:
        for raw in instruction_map.get("Instructions") or []:
            if not isinstance(raw, Mapping):
                continue
            address = _address_int(raw.get("Address"))
            if address is not None:
                by_address[address - relocation_delta].append(raw)
    return by_address


@dataclass
class _SourceBase:
    primary_frames: List[Dict[str, Any]]
    supplemental_frames: List[Dict[str, Any]]
    primary_source_map: Dict[str, Any]
    primary_instruction_map: Dict[str, Any]
    supplemental_source_map: Dict[str, Any]
    supplemental_instruction_map: Dict[str, Any]
    base_provider: str


def _source_base(
    primary_frames: List[Dict[str, Any]],
    supplemental_frames: List[Dict[str, Any]],
) -> _SourceBase:
    primary_source_map, primary_instruction_map = _source_frame_maps(primary_frames)
    supplemental_source_map, supplemental_instruction_map = _source_frame_maps(supplemental_frames)
    # PCSampling can still provide a usable source explorer if the dedicated
    # Source replay is missing, provided it exposes the same native structures.
    primary_has_maps = bool(primary_source_map.get("Files") and primary_instruction_map.get("Instructions"))
    if (not primary_has_maps and supplemental_source_map.get("Files")
            and supplemental_instruction_map.get("Instructions")):
        return _SourceBase(
            primary_frames=supplemental_frames,
            supplemental_frames=[],
            primary_source_map=supplemental_source_map,
            primary_instruction_map=supplemental_instruction_map,
            supplemental_source_map={},
            supplemental_instruction_map={},
            base_provider="PCSampling fallback",
        )
    return _SourceBase(
        primary_frames=primary_frames,
        supplemental_frames=supplemental_frames,
        primary_source_map=primary_source_map,
        primary_instruction_map=primary_instruction_map,
        supplemental_source_map=supplemental_source_map,
        supplemental_instruction_map=supplemental_instruction_map,
        base_provider="Source",
    )


def _merged_core_list(source_cores: List[str], instruction_cores: List[str]) -> List[str]:
    cores: List[str] = ["ALL"]
    for name in source_cores + instruction_cores:
        if name not in cores:
            cores.append(name)
    return cores


class _FileIndexes(NamedTuple):
    by_path: Dict[str, Dict[str, Any]]
    lines: Dict[Tuple[str, int], Dict[str, Any]]


def _source_build_state(
    base: _SourceBase,
    fusion: Dict[str, Any],
    source_cores: List[str],
    instruction_cores: List[str],
    indexes: _FileIndexes,
) -> _SourceBuildState:
    supplemental_lines = _raw_source_line_lookup(base.supplemental_source_map) if fusion.get("accepted") else {}
    relocation_delta = int(fusion.get("address_relocation_delta") or 0)
    supplemental_by_address = _supplemental_address_index(
        base.supplemental_instruction_map, relocation_delta, bool(fusion.get("accepted")))
    return _SourceBuildState(
        fusion=fusion, source_cores=source_cores, instruction_cores=instruction_cores,
        supplemental_lines=supplemental_lines, supplemental_by_address=supplemental_by_address,
        file_by_path=indexes.by_path, line_index=indexes.lines,
        stall_reasons=[], stall_reason_totals=defaultdict(float),
    )


def _source_explorer_model(base: _SourceBase) -> Dict[str, Any]:
    fusion = _source_fusion_compatibility(
        base.primary_source_map,
        base.primary_instruction_map,
        base.supplemental_source_map,
        base.supplemental_instruction_map,
    ) if base.supplemental_frames else _no_supplemental_fusion()

    raw_files = (
        base.primary_source_map.get("Files")
        if isinstance(base.primary_source_map.get("Files"), list) else []
    )
    raw_instructions = (
        base.primary_instruction_map.get("Instructions")
        if isinstance(base.primary_instruction_map.get("Instructions"), list) else []
    )
    source_cores = _core_names(base.primary_source_map)
    instruction_cores = _core_names(base.primary_instruction_map)
    cores = _merged_core_list(source_cores, instruction_cores)

    snapshots = _source_snapshots(base.primary_frames)
    files, file_by_path, line_index = _source_files(snapshots, raw_files)
    state = _source_build_state(base, fusion, source_cores, instruction_cores, _FileIndexes(file_by_path, line_index))
    source_line_rows = _source_map_lines(raw_files, state)

    instructions: List[Dict[str, Any]] = []
    for index, raw_inst in enumerate(raw_instructions):
        inst = _source_instruction(index, raw_inst, state)
        if inst is not None:
            instructions.append(inst)

    relations = _source_relations(instructions, state)
    default_file, default_line = _source_default_focus(files)
    _strip_empty_stall_metrics(files, instructions)

    return {
        "available": bool(files and instructions and source_line_rows),
        "schema": "msopprof-source-explorer/v2",
        "contract": _source_contract(),
        "base_provider": base.base_provider,
        "fusion": fusion,
        "cores": cores,
        "files": files,
        "instructions": instructions,
        "relations": relations,
        "stall_reasons": state.stall_reasons,
        "default_file_id": default_file["id"] if default_file else None,
        "default_line": default_line,
        "summary": _source_summary(snapshots, files, instructions, relations,
            _SourceTally(source_line_rows, state.stall_reason_totals)),
    }


def parse_source_artifact(
    path: Optional[Path],
    supplemental_path: Optional[Path] = None,
    *,
    primary_frames: Optional[List[Dict[str, Any]]] = None,
    supplemental_frames: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a canonical Source Explorer model from Source plus optional PCSampling.

    The Source replay is authoritative for snapshots, line mappings, execution
    counters and GPR status. A compatible PCSampling replay may augment stall
    metrics. Fusion is never based on source order: it requires validated
    source-path/line overlap and instruction-address overlap. Pre-parsed frames
    may be supplied so a shared artifact is only read from disk once.
    """
    if primary_frames is None:
        primary_frames = parse_native_visualize_frames(path)
    if supplemental_frames is None:
        supplemental_frames = parse_native_visualize_frames(supplemental_path)
    base = _source_base(primary_frames, supplemental_frames)
    if not base.primary_frames:
        return {"available": False, "files": [], "instructions": [], "relations": []}
    return _source_explorer_model(base)


_SOURCE_LINE_KEYS = {"line", "line_no", "line_number", "lineno", "source_line"}


_SOURCE_TEXT_KEYS = {"source", "code", "source_code", "text", "content"}


_SOURCE_PC_KEYS = {"pc", "address", "addr", "program_counter"}


_SOURCE_INST_KEYS = {"instruction", "instr", "opcode", "assembly", "asm"}


_SOURCE_METRIC_TOKENS = ("hot", "sample", "cycle", "ratio")


def _first_lower_value(obj: Mapping[str, Any], lower: Mapping[str, str], keys: set) -> Any:
    for key in keys:
        if key in lower:
            return obj[lower[key]]
    return None


def _generic_source_metric(obj: Mapping[str, Any]) -> Optional[float]:
    for key, value in obj.items():
        if not any(token in str(key).lower() for token in _SOURCE_METRIC_TOKENS):
            continue
        number = to_float(value)
        if number is not None:
            return number
    return None


def _normalize_source_obj(
    obj: Mapping[str, Any],
    source_lines: Dict[Tuple[Optional[int], str], Dict[str, Any]],
    instructions: List[Dict[str, Any]],
) -> None:
    lower = {str(k).lower(): k for k in obj.keys()}
    line_value = _first_lower_value(obj, lower, _SOURCE_LINE_KEYS)
    text_value = _first_lower_value(obj, lower, _SOURCE_TEXT_KEYS)
    pc_value = _first_lower_value(obj, lower, _SOURCE_PC_KEYS)
    inst_value = _first_lower_value(obj, lower, _SOURCE_INST_KEYS)
    line_no = int(to_float(line_value)) if to_float(line_value) is not None else None
    if isinstance(text_value, str) and text_value.strip():
        key = (line_no, text_value)
        source_lines.setdefault(key, {"line": line_no, "text": text_value, "pc": pc_value})
    if inst_value is not None or (pc_value is not None and any(k in lower for k in _SOURCE_INST_KEYS)):
        instructions.append({
            "pc": str(pc_value) if pc_value is not None else "",
            "instruction": str(inst_value) if inst_value is not None else "",
            "source_line": line_no,
            "metric": _generic_source_metric(obj),
        })


def _source_line_sort_key(item: Mapping[str, Any]) -> Tuple[bool, int, str]:
    line = item["line"]
    return (line is None, line if line is not None else 0, item["text"])


def normalize_source(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source_lines: Dict[Tuple[Optional[int], str], Dict[str, Any]] = {}
    instructions: List[Dict[str, Any]] = []

    for record in records:
        for obj in iter_dicts(record):
            _normalize_source_obj(obj, source_lines, instructions)
    lines = sorted(source_lines.values(), key=_source_line_sort_key)
    # Require meaningful source or instruction content; generic metadata must not create a false page.
    available = bool(lines and instructions)
    return {"available": available, "source_lines": lines[:5000] if available else [],
        "instructions": instructions[:10000] if available else []}


def _first_key_containing(lower: Mapping[str, str], tokens: Sequence[str]) -> Optional[str]:
    for key in lower:
        if any(token in key for token in tokens):
            return lower[key]
    return None


def _stall_obj_entry(obj: Mapping[str, Any]) -> Optional[Tuple[str, float, Optional[str]]]:
    lower = {str(k).lower(): k for k in obj.keys()}
    reason_key = _first_key_containing(lower, ["stall_reason", "reason", ("stall_typ"
        "e"), "category"])
    value_key = _first_key_containing(lower, ["ratio", "percent", "count", "samples", ("c"
        "ycles")])
    pc_key = next((lower[k] for k in lower if k in {"pc", "address", "addr", "program_counter"}), None)
    if reason_key is None or value_key is None:
        return None
    value = to_float(obj.get(value_key))
    if value is None:
        return None
    pc = str(obj.get(pc_key)) if pc_key is not None else None
    return str(obj.get(reason_key)), value, pc


def normalize_stall(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    categories: Dict[str, List[float]] = defaultdict(list)
    pcs: List[Dict[str, Any]] = []
    for record in records:
        for obj in iter_dicts(record):
            entry = _stall_obj_entry(obj)
            if entry is None:
                continue
            reason, value, pc = entry
            categories[reason].append(value)
            if pc is not None:
                pcs.append({"pc": pc, "reason": reason, "value": value})
    rows = [{"name": name, "value": sum(vals) / len(vals)} for name, vals in categories.items()]
    rows.sort(key=lambda x: -abs(x["value"]))
    return {"available": bool(rows), "categories": rows[:100], "pcs": pcs[:5000]}


def _stall_categories(totals: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
    categories: List[Dict[str, Any]] = []
    total_samples = sum(max(0.0, to_float(value) or 0.0) for value in totals.values())
    for name, raw_value in totals.items():
        value = to_float(raw_value)
        if value is None or value <= 0:
            continue
        categories.append({
            "name": str(name),
            "value": value,
            "share_percent": (value / total_samples * 100.0) if total_samples > 0 else None,
        })
    categories.sort(key=lambda item: (-item["value"], item["name"]))
    return categories, total_samples


def _stall_pc_entry(inst: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    metric = inst.get("stall_not_issue") or inst.get("stall_all") or {}
    details = metric.get("details") or {}
    positive_details = []
    for name, value in details.items():
        number = to_float(value) or 0.0
        if number > 0:
            positive_details.append((str(name), number))
    positive_details.sort(key=lambda item: (-item[1], item[0]))
    percent = to_float(metric.get("percent"))
    if not positive_details and (percent is None or percent <= 0):
        return None
    return {
        "pc": str(inst.get("address") or inst.get("pc") or ""),
        "reason": positive_details[0][0] if positive_details else "Sampled stall",
        "value": percent if percent is not None else 0.0,
        "samples": to_float(metric.get("total_samples")),
        "pipe": str(inst.get("pipe") or ""),
        "opcode": str(inst.get("opcode") or inst.get("instruction") or ""),
        "source_location": str(inst.get("source_location") or ""),
    }


def _stall_pcs(instructions: Any) -> List[Dict[str, Any]]:
    pcs: List[Dict[str, Any]] = []
    for inst in instructions:
        entry = _stall_pc_entry(inst)
        if entry is not None:
            pcs.append(entry)
    pcs.sort(key=lambda item: (-(item.get("value") or 0.0), -(item.get("samples") or 0.0), item.get("pc") or ""))
    return pcs


def normalize_stall_from_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a standalone stall summary from the native Source/PCSampling model.

    PCSampling is commonly encoded using the same Files/Instructions structures as
    Source. Treat it as a separate visual module without assuming one operator,
    opcode set, source suffix, or stall taxonomy.
    """
    summary = source.get("summary") or {}
    totals = summary.get("stall_reason_totals") or {}
    categories, total_samples = _stall_categories(totals)
    pcs = _stall_pcs(source.get("instructions") or [])
    return {
        "available": bool(categories or pcs),
        "categories": categories[:100],
        "pcs": pcs[:5000],
        "total_samples": total_samples,
        "source_schema": source.get("schema"),
    }


def _synthetic_trace_event(obj: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    lower = {str(k).lower(): k for k in obj.keys()}
    start_key = next((lower[k] for k in lower if k in {"start", "ts", "timestamp", "begin"}), None)
    duration_key = next((lower[k] for k in lower if k in {"duration", "dur", "elapsed"}), None)
    name_key = _first_key_containing(lower, ["instruction", "instr", "opcode", "name"])
    lane_key = _first_key_containing(lower, ["pipe", "lane", "tid", "unit"])
    start = to_float(obj.get(start_key)) if start_key else None
    dur = to_float(obj.get(duration_key)) if duration_key else None
    if start is None or dur is None or name_key is None:
        return None
    return {
        "ph": "X",
        "pid": "instruction",
        "tid": str(obj.get(lane_key, "lane")) if lane_key else "lane",
        "name": str(obj.get(name_key)),
        "ts": start,
        "dur": dur,
        "args": {},
    }


def normalize_instruction_timeline(records: Sequence[Mapping[str, Any]], max_events: int) -> Dict[str, Any]:
    for record in records:
        if isinstance(record.get("traceEvents"), list):
            return parse_trace_events(record["traceEvents"], max_events=max_events)

    synthetic: List[Dict[str, Any]] = []
    for record in records:
        for obj in iter_dicts(record):
            event = _synthetic_trace_event(obj)
            if event is not None:
                synthetic.append(event)
    return parse_trace_events(synthetic, max_events=max_events) if synthetic else {"available": False, "lanes": []}


def _csv_table_rows(reader: csv.DictReader, headers: Sequence[str], max_rows: int) -> Tuple[List[List[Any]], int]:
    rows: List[List[Any]] = []
    total = 0
    for raw in reader:
        total += 1
        if len(rows) < max_rows:
            # Rows are positional arrays aligned to `headers`; the
            # template falls back to header-keyed access for legacy
            # object rows.
            rows.append([raw.get(h, "") for h in headers])
    return rows, total


def read_csv_table(path: Path, max_rows: int) -> Optional[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = [h for h in (reader.fieldnames or []) if h is not None and str(h).strip()]
            rows, total = _csv_table_rows(reader, headers, max_rows)
            return {"headers": headers, "rows": rows, "total_rows": total, "embedded_rows": len(rows)}
    except Exception as exc:
        logger.debug("Failed to read CSV table %s: %s", path, exc)
        return None


def read_csv_headers(path: Path) -> List[str]:
    """Read only the header row of a CSV (cheap companion for reused rows)."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                return [h for h in row if h is not None and str(h).strip()]
    except Exception as exc:
        logger.debug("Failed to read CSV headers %s: %s", path, exc)
        return []
    return []


def parse_raw_data(
    artifacts: ArtifactSet,
    max_rows: int,
    csv_rows: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    tables: List[Dict[str, Any]] = []
    for name, path in sorted(artifacts.csv.items()):
        # Preserve every manifest-declared CSV, including OpBasicInfo metadata.
        # The Raw Data module is an evidence browser, not only a chart source.
        if csv_rows is not None:
            # Reuse the rows the SourceBundle already loaded instead of
            # re-parsing every CSV from disk; only the header line is re-read.
            full_rows = csv_rows.get(name)
            if full_rows is None or not path.is_file() or path.stat().st_size == 0:
                continue
            headers = read_csv_headers(path)
            if not headers:
                continue
            embedded = []
            for row in full_rows[:max_rows]:
                embedded.append([row.get(h, "") for h in headers])
            table: Optional[Dict[str, Any]] = {
                "headers": headers,
                "rows": embedded,
                "total_rows": len(full_rows),
                "embedded_rows": len(embedded),
            }
        else:
            table = read_csv_table(path, max_rows=max_rows)
        if table:
            tables.append({"name": name, **table})
    return {"available": bool(tables), "tables": tables}