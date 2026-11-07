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

import json
import logging
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple
from visualize_common import (
    ArtifactSet,
    flatten_numeric,
    humanize_key,
    iter_dicts,
    mean,
    metric_is_percent,
    percentile,
    to_float,
)

logger = logging.getLogger(__name__)


def _parse_embedded_blob(data: bytes, start: int, end: int) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Parse the balanced blob ``data[start:end + 1]``.

    Returns ``(next_index, record)``. An unparseable blob yields
    ``(start + 1, None)`` so the caller resumes its garbage-skipping rescan one
    byte later; valid JSON that is not a dict yields ``(end + 1, None)``.
    """
    blob = data[start:end + 1]
    try:
        obj = json.loads(blob.decode("utf-8"))
    except Exception:
        return start + 1, None
    record = obj if isinstance(obj, dict) else None
    return end + 1, record


def _string_scan_state(c: int, in_string: bool, escaped: bool) -> Tuple[bool, bool]:
    """Advance the string/escape state machine for one byte inside a string."""
    if escaped:
        return in_string, False
    if c == 92:
        return in_string, True
    if c == 34:
        return False, False
    return in_string, escaped


def _scan_embedded_candidate(data: bytes, start: int, n: int) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Scan one candidate JSON object beginning at ``start``.

    Returns ``(next_index, record)``; see ``_parse_embedded_blob`` for the
    rescan semantics of unparseable candidates.
    """
    depth = 0
    in_string = False
    escaped = False
    for j in range(start, n):
        c = data[j]
        if in_string:
            in_string, escaped = _string_scan_state(c, in_string, escaped)
            continue
        if c == 34:
            in_string = True
            continue
        if c == 123:
            depth += 1
            continue
        if c != 125:
            continue
        depth -= 1
        if depth == 0:
            return _parse_embedded_blob(data, start, j)
    return start + 1, None


def extract_embedded_json_records(data: bytes) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    i, n = 0, len(data)
    while i < n:
        start = data.find(b"{", i)
        if start < 0:
            break
        i, record = _scan_embedded_candidate(data, start, n)
        if record is not None:
            records.append(record)
    return records


def load_visualize_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return extract_embedded_json_records(path.read_bytes())
    except Exception as exc:
        logger.debug("Failed to load visualize records from %s: %s", path, exc)
        return []


def collect_advice(records: Sequence[Mapping[str, Any]]) -> List[str]:
    found: List[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            for line in value.splitlines():
                cleaned = re.sub(r"^[\s\t\-\d\)\.]+", "", line).strip()
                if cleaned and cleaned not in found:
                    found.append(cleaned)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                add(item)

    for record in records:
        if "advice" in record:
            add(record.get("advice"))
        for obj in iter_dicts(record):
            if obj is record:
                continue
            if "advice" in obj:
                add(obj.get("advice"))
    return found


def parse_block_detail(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    for record in records:
        detail = record.get("block_detail")
        if not isinstance(detail, Mapping):
            continue
        headers = [str(h) for h in (detail.get("head_name") or [])]
        rows: List[Dict[str, Any]] = []
        for item in detail.get("row") or []:
            values = list(item.get("value") or []) if isinstance(item, Mapping) else []
            row: Dict[str, Any] = {}
            for h, v in zip(headers, values):
                row[h] = v
            rows.append(row)
        return rows
    return []


def parse_basic(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    for record in records:
        detail = record.get("block_detail")
        if not isinstance(detail, Mapping):
            continue
        blocks: List[Dict[str, Any]] = []
        for row in parse_block_detail([record]):
            block_id = next((v for k, v in row.items() if "block" in k.lower() and "id" in k.lower()), None)
            core_type = next((v for k, v in row.items() if "core" in k.lower() and "type" in k.lower()), "")
            duration = next((v for k, v in row.items() if "duration" in k.lower()), None)
            duration_us = to_float(duration)
            if block_id is None or duration_us is None:
                continue
            blocks.append({"block_id": str(block_id), "core_type": str(core_type or ""), "duration_us": duration_us})
        current_freq = to_float(record.get("cur_freq"))
        rated_freq = to_float(record.get("rated_freq"))
        frequency_ratio = None
        if current_freq is not None and rated_freq not in (None, 0):
            frequency_ratio = current_freq / rated_freq * 100.0
        return {
            "available": True,
            "name": str(record.get("name", "")),
            "duration_us": to_float(record.get("duration")),
            "op_type": str(record.get("op_type", "")),
            "block_dim": str(record.get("block_dim", "")),
            "mix_block_dim": str(record.get("mix_block_dim", "")),
            "soc": str(record.get("soc", "")),
            "device_id": str(record.get("device_id", "")),
            "current_frequency_mhz": current_freq,
            "rated_frequency_mhz": rated_freq,
            "frequency_ratio_percent": frequency_ratio,
            "blocks": blocks,
        }
    return {"available": False, "blocks": []}


OCCUPANCY_PACKED_LAYOUTS: Dict[str, Dict[str, Any]] = {
    # Exact SoC string emitted by the supplied Ascend 950 profiler payload.
    "ASCEND950PR9579": {
        "lanes_per_core": 2,
        # Source-slot order verified against the authoritative Ascend 950 occupancy view.
        # Unlisted groups keep source order; raw IDs are always retained.
        "slot_order": {
            0: [1, 2, 3, 0],
            1: [0, 1, 2, 3],
            2: [0, 1, 2, 3],
            3: [3, 2, 1, 0],
            4: [0, 1, 2, 3],
            5: [0, 1, 2, 3],
            6: [0, 1, 2, 3],
            128: [0, 3, 1, 2],
            129: [0, 1, 2, 3],
            130: [0, 1, 2, 3],
            131: [3, 2, 1, 0],
            132: [0, 1, 2, 3],
            133: [0, 1, 2, 3],
            134: [0, 1, 2, 3],
        },
    },
}


OCCUPANCY_PACKED_LAYOUT_ALIASES: Dict[str, str] = {
    "910A5": "ASCEND950PR9579",
}


def normalize_soc_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()


def occupancy_packed_schema(raw_items: Sequence[Mapping[str, Any]], soc: str) -> Optional[Dict[str, Any]]:
    if not raw_items:
        return None
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in raw_items:
        grouped[str(item.get("raw_core_id", ""))].append(item)
    duplicate_groups = [rows for rows in grouped.values() if len(rows) > 1]
    if not duplicate_groups:
        return None
    # Packed schema signature: repeated raw core IDs, identical nested subcore IDs,
    # and one engine family. This avoids rewriting ordinary multi-subcore payloads.
    repeated_subcore_ids = all(len({str(x.get("raw_subcore_id", "")) for x in rows}) == 1 for rows in duplicate_groups)
    engine_types = {str(x.get("subcore_type", "")).lower() for x in raw_items if str(x.get("subcore_type", "")).strip()}
    if not repeated_subcore_ids or len(engine_types) != 1:
        return None
    normalized_soc = normalize_soc_name(soc)
    layout_key = OCCUPANCY_PACKED_LAYOUT_ALIASES.get(normalized_soc, normalized_soc)
    layout = OCCUPANCY_PACKED_LAYOUTS.get(layout_key)
    if layout:
        result = dict(layout)
        result["layout_key"] = layout_key
        return result
    # Conservative generic fallback for vector payloads: only expand when every
    # repeated group can form two-lane physical cores exactly.
    if engine_types == {"vector"} and all(len(rows) % 2 == 0 for rows in duplicate_groups):
        return {"lanes_per_core": 2, "slot_order": {}}
    return None


def _raw_core_sort_key(value: str) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def _assemble_packed_logical(
    grouped: Mapping[str, List[Dict[str, Any]]],
    slot_orders: Mapping[Any, Any],
    lanes_per_core: int,
) -> Tuple[List[Dict[str, Any]], int]:
    logical: List[Dict[str, Any]] = []
    logical_core_id = 0
    for raw_id in sorted(grouped, key=_raw_core_sort_key):
        rows = grouped[raw_id]
        try:
            numeric_raw = int(raw_id)
        except (TypeError, ValueError):
            numeric_raw = None
        order = slot_orders.get(numeric_raw)
        if isinstance(order, list) and len(order) == len(rows) and sorted(order) == list(range(len(rows))):
            rows = [rows[i] for i in order]
        for offset in range(0, len(rows), lanes_per_core):
            # Do not invent a missing lane. Keep the residual entry visible.
            chunk = rows[offset:offset + lanes_per_core]
            for lane_id, item in enumerate(chunk):
                item["core_id"] = logical_core_id
                item["subcore_id"] = lane_id
                item["logical_core_id"] = logical_core_id
                item["logical_subcore_id"] = lane_id
                logical.append(item)
            logical_core_id += 1
    return logical, logical_core_id


def normalize_packed_occupancy(raw_items: Sequence[Mapping[str,
    Any]], soc: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    schema = occupancy_packed_schema(raw_items, soc)
    if not schema:
        return [dict(x) for x in raw_items], {
            "adapter": "native",
            "soc": soc,
            "logical_core_count": len({str(x.get('core_id')) for x in raw_items}),
            "raw_core_count": len({str(x.get('raw_core_id')) for x in raw_items}),
        }

    lanes_per_core = max(1, int(schema.get("lanes_per_core", 2)))
    slot_orders = schema.get("slot_order") or {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in raw_items:
        grouped[str(item.get("raw_core_id", ""))].append(dict(item))
    logical, logical_core_id = _assemble_packed_logical(grouped, slot_orders, lanes_per_core)

    return logical, {
        "adapter": "packed-core-group",
        "soc": soc,
        "lanes_per_core": lanes_per_core,
        "logical_core_count": logical_core_id,
        "raw_core_count": len(grouped),
        "raw_core_ids": sorted(grouped, key=_raw_core_sort_key),
        "layout_key": schema.get("layout_key"),
    }


def _occupancy_op_entries(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], str]:
    op_entries: List[Mapping[str, Any]] = []
    soc = ""
    for record in records:
        if isinstance(record.get("op_detail"), list):
            op_entries.extend(x for x in record["op_detail"] if isinstance(x, Mapping))
            if record.get("soc") and not soc:
                soc = str(record.get("soc"))
    return op_entries, soc


def _occupancy_metrics_of(sub: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key, value in flatten_numeric(sub).items():
        if key in {"subcore_id", "core_id"}:
            continue
        if key.lower().endswith("_id") or key.lower().endswith(".id"):
            continue
        metrics[key] = value
    return metrics


def _occupancy_raw_item(
    core: Mapping[str, Any],
    entry_index: int,
    detail_index: int,
    sub: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    metrics = _occupancy_metrics_of(sub)
    if not metrics:
        return None
    core_id = core.get("core_id")
    return {
        "core_id": core_id,
        "subcore_id": sub.get("subcore_id"),
        "subcore_type": sub.get("subcore_type", ""),
        "raw_core_id": core_id,
        "raw_subcore_id": sub.get("subcore_id"),
        "source_entry_index": entry_index,
        "source_detail_index": detail_index,
        "metrics": metrics,
    }


def _occupancy_raw_items(op_entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    raw_items: List[Dict[str, Any]] = []
    for entry_index, core in enumerate(op_entries):
        details = core.get("core_detail") or []
        for detail_index, sub in enumerate(details):
            if not isinstance(sub, Mapping):
                continue
            item = _occupancy_raw_item(core, entry_index, detail_index, sub)
            if item is not None:
                raw_items.append(item)
    return raw_items


def _occupancy_metrics(cores: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_values: Dict[str, List[Optional[float]]] = defaultdict(list)
    for item in cores:
        for key, value in (item.get("metrics") or {}).items():
            metric_values[key].append(value)

    metrics: List[Dict[str, Any]] = []
    preferred_tokens = ["cycles", "throughput", "hit", "instructions", "instruction_per_cycle"]
    for key, values in metric_values.items():
        finite = [v for v in values if v is not None]
        if not finite:
            continue
        priority = min((preferred_tokens.index(t) for t in preferred_tokens if t in key.lower()), default=99)
        signal = max(abs(v) for v in finite) + (max(finite) - min(finite))
        med = statistics.median(finite)
        metrics.append({
            "key": key,
            "label": humanize_key(key),
            "format": "percent" if metric_is_percent(key, values) else "number",
            "min": min(finite),
            "max": max(finite),
            "median": med,
            "p10": percentile(finite, 0.10),
            "p90": percentile(finite, 0.90),
            "spread_pct": ((max(finite) - min(finite)) / med * 100.0) if med not in {0, None} else None,
            "priority": priority,
            "signal": signal,
        })
    # Do not default to an all-zero metric merely because its name is normally preferred.
    metrics.sort(key=lambda x: (0 if x["signal"] > 1e-12 else 1, x["priority"], x["label"]))
    for metric in metrics:
        metric.pop("priority", None)
        metric.pop("signal", None)
    return metrics


def parse_occupancy(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    op_entries, soc = _occupancy_op_entries(records)
    raw_items = _occupancy_raw_items(op_entries)
    cores, layout = normalize_packed_occupancy(raw_items, soc)
    metrics = _occupancy_metrics(cores)
    return {"available": bool(cores and metrics), "cores": cores, "metrics": metrics, "layout": layout}


def parse_compute(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    raw_items: List[Mapping[str, Any]] = []
    for record in records:
        if isinstance(record.get("subblock_detail"), list):
            raw_items.extend(x for x in record["subblock_detail"] if isinstance(x, Mapping))

    items: List[Dict[str, Any]] = []
    for item in raw_items:
        value = to_float(item.get("value"))
        if value is None:
            continue
        items.append({
            "block_id": str(item.get("block_id", "")),
            "block_type": str(item.get("block_type", "") or "Compute"),
            "name": str(item.get("name", "Metric") or "Metric"),
            "unit": item.get("unit"),
            "value": value,
            "origin_value": to_float(item.get("origin_value")),
        })

    items.sort(key=lambda x: (
        int(x["block_id"]) if str(x["block_id"]).isdigit() else 10**9,
        str(x["block_type"]),
        str(x["name"]),
        str(x.get("unit")),
    ))
    # Mixed numeric/non-numeric Block IDs cannot be compared directly; order
    # numeric IDs first (by value), then the rest lexicographically.
    block_ids = sorted({x["block_id"] for x in items}, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    block_types = sorted({x["block_type"] for x in items})

    # Keep a compact aggregate view for compatibility with earlier consumers and tests.
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[(item["name"], item["block_type"], str(item.get("unit")))].append(item)
    metrics: List[Dict[str, Any]] = []
    for (name, block_type, unit_key), group in grouped.items():
        vals = [x["value"] for x in group]
        is_percent = str(group[0].get("unit")) == "3" or metric_is_percent(name, vals)
        metrics.append({
            "key": f"{block_type}::{name}::unit={unit_key}",
            "name": name,
            "display_name": name + (" (%)" if is_percent else " (raw)"),
            "block_type": block_type,
            "format": "percent" if is_percent else "number",
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "blocks": [{"block_id": x["block_id"], "value": x["value"]} for x in group],
        })
    return {"available": bool(items), "items": items, "block_ids": block_ids,
        "block_types": block_types, "metrics": metrics}


def _tpb_row_item(row: Mapping[str, Any], block_id: str, headers: Sequence[str]) -> Dict[str, Any]:
    values = list(row.get("value") or [])
    metric_headers = headers[1:] if len(headers) == len(values) + 1 else headers[:len(values)]
    item: Dict[str, Any] = {"block_id": block_id, "name": str(row.get("name", ""))}
    for h, v in zip(metric_headers, values):
        item[str(h)] = v
    return item


def _tpb_table_item(
    table: Mapping[str, Any],
    block_id: str,
    grouped: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    table_name = str(table.get("table_name", "Unknown"))
    headers = [str(h) for h in (table.get("header_name") or [])]
    rows: List[Dict[str, Any]] = []
    for row in table.get("row") or []:
        if not isinstance(row, Mapping):
            continue
        item = _tpb_row_item(row, block_id, headers)
        rows.append(item)
        grouped[table_name].append(item)
    return {"name": table_name, "headers": headers, "rows": rows}


def _tpb_block_item(
    block: Mapping[str, Any],
    grouped: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    block_id = str(block.get("block_id", ""))
    normalized_tables: List[Dict[str, Any]] = []
    for table in block.get("table_detail") or []:
        if not isinstance(table, Mapping):
            continue
        normalized_tables.append(_tpb_table_item(table, block_id, grouped))
    return {"block_id": block_id, "tables": normalized_tables}


def normalize_table_per_block(records: Sequence[Mapping[str,
    Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    blocks: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        source = record.get("table_per_block")
        if not isinstance(source, list):
            continue
        for block in source:
            if not isinstance(block, Mapping):
                continue
            blocks.append(_tpb_block_item(block, grouped))
    return blocks, dict(grouped)


def _memory_table_item(table: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    headers = [str(h) for h in (table.get("headers") or [])]
    value_headers = headers[1:] if headers else []
    rows: List[Dict[str, Any]] = []
    for row in table.get("rows") or []:
        values: Dict[str, Any] = {}
        for key, value in row.items():
            if key in {"block_id", "name"}:
                continue
            values[str(key)] = value
        rows.append({"name": str(row.get("name", "")), "values": values})
    if not rows:
        return None
    return {"name": str(table.get("name", "Unknown")),
        "headers": headers or (["Name"] + value_headers), ("r"
        "ows"): rows}


def _memory_block_tables(raw_blocks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for block in raw_blocks:
        tables: List[Dict[str, Any]] = []
        for table in block.get("tables") or []:
            item = _memory_table_item(table)
            if item is not None:
                tables.append(item)
        if tables:
            blocks.append({"block_id": str(block.get("block_id", "")), "tables": tables})
    return blocks


def _memory_core_map_entry(core: Mapping[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    core_id = str(core.get("core_no", ""))
    paths: List[Dict[str, Any]] = []
    for unit in core.get("memory_unit") or []:
        if not isinstance(unit, Mapping):
            continue
        paths.append({
            "memory_path": unit.get("memory_path"),
            "request": to_float(unit.get("request")),
            "bandwidth": to_float(unit.get("bandwidth")),
            "peak_ratio": to_float(unit.get("peak_ratio")),
            "display": unit.get("display"),
        })
    summary: Dict[str, Any] = {}
    for key, value in core.items():
        if key in {"core_no", "memory_unit", "advice", "soc", "op_type"}:
            continue
        if not isinstance(value, (Mapping, list)):
            continue
        summary[key] = value
    return core_id, paths, summary


def _memory_core_maps(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    core_paths: Dict[str, List[Dict[str, Any]]] = {}
    core_summaries: Dict[str, Dict[str, Any]] = {}
    for record in records:
        source = record.get("core_memory_map")
        if not isinstance(source, list):
            continue
        for core in source:
            if not isinstance(core, Mapping):
                continue
            core_id, paths, summary = _memory_core_map_entry(core)
            core_paths[core_id] = paths
            core_summaries[core_id] = summary
    return core_paths, core_summaries


def _memory_row_keys(row_items: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: List[str] = []
    for row_item in row_items:
        for key in (row_item.get("values") or {}):
            if key in keys:
                continue
            keys.append(key)
    return keys


def _memory_group_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_name[str(row.get("name", ""))].append(row)
    normalized_rows: List[Dict[str, Any]] = []
    for row_name, row_items in by_name.items():
        metrics: Dict[str, Optional[float]] = {}
        for key in _memory_row_keys(row_items):
            metrics[key] = mean((ri.get("values") or {}).get(key) for ri in row_items)
        normalized_rows.append({
            "name": row_name,
            "metrics": metrics,
        })
    return normalized_rows


def _memory_groups(blocks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        for table in block.get("tables") or []:
            for row in table.get("rows") or []:
                by_table[table["name"]].append(row)
    for table_name, rows in by_table.items():
        field_names = set()
        for row in rows:
            for key in (row.get("values") or {}).keys():
                field_names.add(str(key).lower())
        cache_summary = (
            "cache" in table_name.lower()
            and any("hit" in f or "miss" in f for f in field_names)
            and not any("throughput" in f or "bandwidth" in f or "request" in f for f in field_names)
        )
        if cache_summary:
            continue
        normalized_rows = _memory_group_rows(rows)
        if normalized_rows:
            groups.append({"name": table_name, "rows": normalized_rows})
    return groups


def parse_memory(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    raw_blocks, _ = normalize_table_per_block(records)
    blocks = _memory_block_tables(raw_blocks)
    core_paths, core_summaries = _memory_core_maps(records)
    groups = _memory_groups(blocks)
    advice = collect_advice(records)
    return {
        "available": bool(blocks or core_paths),
        "blocks": blocks,
        "core_paths": core_paths,
        "core_summaries": core_summaries,
        "groups": groups,
        "advice": advice,
    }


def _canonical_cache_field(name: Any) -> Optional[str]:
    low = str(name).strip().lower().replace("_", " ")
    if "hit rate" in low or ("hit" in low and "rate" in low):
        return "hit_rate"
    if low == "hit" or low.endswith(" hit"):
        return "hit"
    if low == "miss" or low.endswith(" miss"):
        return "miss"
    if low == "total" or low.endswith(" total"):
        return "total"
    return None


class _CacheAccumulators(NamedTuple):
    entries: Dict[str, float]
    metric_keys: List[str]
    aggregate: Dict[str, List[float]]


def _cache_row_fields(
    row: Mapping[str, Any],
    row_name: str,
    family: Dict[str, Optional[float]],
    accumulators: _CacheAccumulators,
) -> None:
    for field, raw in row.items():
        if field in {"block_id", "name"}:
            continue
        value = to_float(raw)
        if value is None:
            continue
        key = f"{row_name} · {field}"
        accumulators.entries[key] = value
        accumulators.aggregate[key].append(value)
        if key not in accumulators.metric_keys:
            accumulators.metric_keys.append(key)
        canonical = _canonical_cache_field(field)
        if canonical:
            family[canonical] = value


def _cache_table_field_names(table: Mapping[str, Any]) -> set:
    fields = set()
    for row in table["rows"]:
        for field_name in row.keys():
            if field_name in {"block_id", "name"}:
                continue
            fields.add(str(field_name).lower())
    return fields


def _cache_block_entries(
    block: Mapping[str, Any],
    family_by_block: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    metric_keys: List[str],
    series_keys: List[str],
    aggregate: Dict[str, List[float]],
) -> Dict[str, float]:
    entries: Dict[str, float] = {}
    accumulators = _CacheAccumulators(entries, metric_keys, aggregate)
    block_id = str(block["block_id"])
    for table in block["tables"]:
        fields = _cache_table_field_names(table)
        is_cache_summary = "cache" in table["name"].lower() and any("hit" in f or "miss" in f for f in fields)
        if not is_cache_summary:
            continue
        for row in table["rows"]:
            row_name = str(row.get("name", "")).strip() or str(table["name"])
            family = family_by_block[block_id].setdefault(row_name, {"hit": None, "miss": None, "total": None, ("h"
                "it_rate"): None})
            _cache_row_fields(row, row_name, family, accumulators)
            if row_name not in series_keys:
                series_keys.append(row_name)
    return entries


def _cache_summary(aggregate: Mapping[str, List[float]]) -> List[Dict[str, Any]]:
    summary = []
    for key, vals in aggregate.items():
        summary.append({
            "key": key,
            "label": key,
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "format": "percent" if metric_is_percent(key, vals) else "number",
        })
    summary.sort(key=lambda x: (0 if "hit rate" in x["key"].lower() else 1, x["key"]))
    return summary


def _cache_family_cells(
    family_name: str,
    cache_blocks: Sequence[Mapping[str, Any]],
    family_by_block: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], float, float]:
    raw_cells: List[Dict[str, Any]] = []
    hit_sum = miss_sum = 0.0
    for block in cache_blocks:
        values = family_by_block.get(str(block["block_id"]), {}).get(family_name)
        if not values:
            continue
        hit = to_float(values.get("hit")) or 0.0
        miss = to_float(values.get("miss")) or 0.0
        total = to_float(values.get("total"))
        if total is None:
            total = hit + miss
        hit_rate = to_float(values.get("hit_rate"))
        if hit_rate is None and total > 0:
            hit_rate = hit / total * 100.0
        hit_sum += hit
        miss_sum += miss
        raw_cells.append({
            "index": str(block["block_id"]),
            "index_kind": "block_id",
            "block_id": str(block["block_id"]),
            "hit": hit,
            "miss": miss,
            "total": total,
            "hit_rate": hit_rate,
            "source_links": [],
        })
    for cell in raw_cells:
        cell["hit_share_percent"] = (cell["hit"] / hit_sum * 100.0) if hit_sum > 0 else 0.0
        cell["miss_share_percent"] = (cell["miss"] / miss_sum * 100.0) if miss_sum > 0 else 0.0
    return raw_cells, hit_sum, miss_sum


def _cache_family_models(
    series_keys: Sequence[str],
    cache_blocks: Sequence[Mapping[str, Any]],
    family_by_block: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    family_models: List[Dict[str, Any]] = []
    for family_name in series_keys:
        raw_cells, hit_sum, miss_sum = _cache_family_cells(family_name, cache_blocks, family_by_block)
        if raw_cells:
            cols = max(1, math.ceil(math.sqrt(len(raw_cells))))
            rows_count = math.ceil(len(raw_cells) / cols)
            family_models.append({
                "name": family_name,
                "index_kind": "block_id",
                "index_label": "Profiler Block ID",
                "granularity_note": ("This capture exposes per-block cache aggregates; native cacheline "
                    "indices and instruction linkage are not present in the payload."),
                "columns": cols,
                "rows": rows_count,
                "cells": raw_cells,
                "totals": {"hit": hit_sum, "miss": miss_sum, "events": hit_sum + miss_sum},
            })
    family_models.sort(key=lambda x: (0 if x["name"].lower() == "l2 cache total" else 1,
        0 if "l2 cache" in x["name"].lower() else 1, x["name"].lower()))
    return family_models


def parse_cache(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize cache summaries into paired Hit/Miss event maps.

    The profiler payload used by many on-board captures is block aggregated, not
    cacheline resolved.  The model therefore records the index semantics instead
    of labelling block IDs as cacheline indices.  A future native cacheline
    provider can populate the same family/cell contract with
    ``index_kind=cacheline`` without changing the renderer.
    """
    blocks, _ = normalize_table_per_block(records)
    cache_blocks: List[Dict[str, Any]] = []
    metric_keys: List[str] = []
    series_keys: List[str] = []
    aggregate: Dict[str, List[float]] = defaultdict(list)
    family_by_block: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(dict)

    for block in blocks:
        entries = _cache_block_entries(block, family_by_block, metric_keys, series_keys, aggregate)
        if entries:
            cache_blocks.append({"block_id": str(block["block_id"]), "values": entries})

    summary = _cache_summary(aggregate)
    metric_keys.sort(key=lambda x: (0 if "hit rate" in x.lower() else 1, x))
    family_models = _cache_family_models(series_keys, cache_blocks, family_by_block)
    supports_source_linkage = False
    for family in family_models:
        for cell in family.get("cells") or []:
            if cell.get("source_links"):
                supports_source_linkage = True
    return {
        "available": bool(cache_blocks),
        "schema": "msopprof-cache-heatmap/v2",
        "granularity": "block",
        "index_kind": "block_id",
        "blocks": cache_blocks,
        "metrics": metric_keys,
        "summary": summary,
        "families": family_models,
        "default_family": family_models[0]["name"] if family_models else None,
        "supports_cacheline_detail": False,
        "supports_source_linkage": supports_source_linkage,
    }


def _walk_json_nodes(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_json_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_nodes(child)


def _first_key(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    normalized = {str(k).strip().lower().replace("-", "_"): v for k, v in mapping.items()}
    for name in names:
        key = name.lower().replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def _numeric_pair(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return to_float(value[0]), to_float(value[1])
    if isinstance(value, Mapping):
        return to_float(_first_key(value, ["start", "begin", "min"])), to_float(_first_key(value, ["end", "finish", ("m"
            "ax")]))
    return None, None


def _onchip_missing_diagnostic() -> Dict[str, Any]:
    return {
        "available": False,
        "diagnostic": {
            "code": "memory_info_missing",
            "title": "On-Chip Memory unavailable",
            "explanation": ("The collection does not contain a manifest-declared memory_info.json file. "
                "MemoryDetail counters cannot be substituted for allocation lifetime/address data."),
            "evidence": ["Required artifact: memory_info.json", ("Required views: memory block map, "
                "selected allocation detail, bank/group distribution")],
            "requirements": ["Generate memory_info.json for the selected kernel/build.", ("Declare or "
                "pass the exact file to the visualization pipeline.")],
        },
    }


def _onchip_errors(raw: Any) -> List[str]:
    errors: List[str] = []
    if isinstance(raw, Mapping):
        for key in ["error", "errors", "error_message", "compile_error", "message"]:
            value = _first_key(raw, [key])
            if value and any(token in str(value).lower() for token in ["fail", "error", "失败"]):
                items = value if isinstance(value, list) else [value]
                errors.extend(str(x) for x in items)
    return errors


def _onchip_int_list(value: Any) -> List[int]:
    if isinstance(value, (int, float, str)):
        return [int(x) for x in re.findall(r"\d+", str(value))]
    if isinstance(value, list):
        numbers: List[int] = []
        for item in value:
            for token in re.findall(r"\d+", str(item)):
                numbers.append(int(token))
        return numbers
    return []


def _onchip_candidate(node: Mapping[str, Any], ordinal: int) -> Optional[Dict[str, Any]]:
    start = to_float(_first_key(node, ["start", "start_time",
        "life_start", "begin", "alloc_order", "start_line", ("a"
        "lloc_line"), "birth"]))
    end = to_float(_first_key(node, ["end", "end_time", "life_end", "finish", "free_order", "end_line", ("free_l"
        "ine"), "death"]))
    life_start, life_end = _numeric_pair(_first_key(node, ["lifetime", "life_time", "life", "interval"]))
    start = start if start is not None else life_start
    end = end if end is not None else life_end
    size = to_float(_first_key(node, ["size", "size_bytes", "buffer_size", "memory_size", "bytes", "tensor_size"]))
    addr_start = to_float(_first_key(node, ["address", "offset", "start_address", "address_start", "addr", ("ub_"
        "offset")]))
    addr_end = to_float(_first_key(node, ["end_address", "address_end", "addr_end"]))
    if addr_end is None and addr_start is not None and size is not None:
        addr_end = addr_start + size
    signals = sum(x is not None for x in [start, end, size, addr_start, addr_end])
    if signals < 3 or start is None or end is None:
        return None
    name = _first_key(node, ["name", "tensor_name", "buffer_name", "id", "block_id", "allocation", "var_name"])
    block_id = str(name if name not in {None, ""} else f"allocation-{ordinal}")
    banks_raw = _first_key(node, ["banks", "bank_ids", "bank_id", "bank", "physical_banks"])
    groups_raw = _first_key(node, ["groups", "group_ids", "group_id", "bank_group"])
    temporary_raw = _first_key(node, ["temporary", "is_temporary", "is_temp", "temp"])
    temporary = None if temporary_raw is None else str(temporary_raw).lower() in {"1", "true", "yes", "y", ("tem"
        "porary"), "temp"}
    return {
        "id": f"mem-{ordinal}",
        "name": block_id,
        "start": start,
        "end": end,
        "lifetime": max(0.0, end - start),
        "size_bytes": size,
        "address_start": addr_start,
        "address_end": addr_end,
        "allocation_location": str(_first_key(node, ["allocation_location", "apply_position", "alloc_position", ("s"
            "ource"), "source_location", "ir_line"]) or ""),
        "scope": str(_first_key(node, ["scope", "scope_type", "memory_scope"]) or ""),
        "temporary": temporary,
        "bank_ids": sorted(set(_onchip_int_list(banks_raw))),
        "group_ids": sorted(set(_onchip_int_list(groups_raw))),
        "_signature": (block_id, start, end, addr_start, size),
    }


def _onchip_candidates(raw: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for node in _walk_json_nodes(raw):
        candidate = _onchip_candidate(node, len(candidates))
        if candidate is None:
            continue
        signature = candidate.pop("_signature")
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)
    return candidates


def _onchip_bank_layout(raw: Any, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bank_count = group_count = banks_per_group = None
    for node in _walk_json_nodes(raw):
        bank_count = bank_count or to_float(_first_key(node, ["bank_count", "num_banks", "banks_count"]))
        group_count = group_count or to_float(_first_key(node, ["group_count", "num_groups", "bank_group_count"]))
        banks_per_group = banks_per_group or to_float(_first_key(node, ["banks_per_group", "bank_num_per_group"]))
    max_bank = max((max(x["bank_ids"]) for x in candidates if x["bank_ids"]), default=-1)
    if bank_count is None and max_bank >= 0:
        bank_count = max_bank + 1
    if group_count is None and candidates:
        max_group = max((max(x["group_ids"]) for x in candidates if x["group_ids"]), default=-1)
        if max_group >= 0:
            group_count = max_group + 1
    if banks_per_group is None and bank_count and group_count:
        banks_per_group = math.ceil(bank_count / group_count)

    bank_cells: List[Dict[str, Any]] = []
    if bank_count:
        for bank in range(int(bank_count)):
            group = int(bank // banks_per_group) if banks_per_group else None
            owners = [x["id"] for x in candidates if bank in x["bank_ids"]]
            bank_cells.append({"bank_id": bank, "group_id": group, "owner_ids": owners})
    return {"bank_count": int(bank_count) if bank_count else 0,
        "group_count": int(group_count) if group_count else 0, ("b"
        "anks_per_group"): int(banks_per_group) if banks_per_group else 0, "cells": bank_cells}


def parse_onchip_memory(path: Optional[Path]) -> Dict[str, Any]:
    """Parse a memory_info.json sidecar into the three-view On-Chip Memory contract."""
    if path is None or not path.is_file():
        return _onchip_missing_diagnostic()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "diagnostic": {"code": "memory_info_invalid_json", "title": ("On-Chip "
            "Memory payload is invalid"), "explanation": str(exc), "evidence": [path.name]}}

    errors = _onchip_errors(raw)
    candidates = _onchip_candidates(raw)
    if not candidates:
        return {
            "available": False,
            "diagnostic": {
                "code": "memory_info_no_usable_allocations",
                "title": "On-Chip Memory payload has no usable allocation records",
                "explanation": ("memory_info.json was found, but no records contained a lifetime plus "
                    "size/address information required by the memory block map."),
                "evidence": errors or [path.name],
            },
        }
    start_min = min(x["start"] for x in candidates)
    end_max = max(x["end"] for x in candidates)
    known_addr: List[float] = []
    for candidate in candidates:
        for addr_value in (candidate.get("address_start"), candidate.get("address_end")):
            if addr_value is not None:
                known_addr.append(addr_value)
    return {
        "available": True,
        "schema": "msopprof-onchip-memory/v1",
        "source_name": path.name,
        "status": "partial" if errors else "ok",
        "errors": errors,
        "blocks": candidates,
        "execution_range": [start_min, end_max],
        "address_range": [min(known_addr), max(known_addr)] if known_addr else None,
        "bank_layout": _onchip_bank_layout(raw, candidates),
        "summary": {"allocation_count": len(candidates),
            "peak_known_bytes": max((x.get("size_bytes") or 0 for x in candidates), default=0), ("t"
            "emporary_count"): sum(x.get("temporary") is True for x in candidates)},
    }


def recursive_find_key_values(obj: Any, key_name: str) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == key_name:
                out.append(value)
            out.extend(recursive_find_key_values(value, key_name))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(recursive_find_key_values(value, key_name))
    return out


X_KEY_PATTERNS = ["arithmetic_intensity", "arithmetic intensity", "intensity", "ai", "x"]


Y_KEY_PATTERNS = ["performance", "perf", "throughput", "flops", "gflops", "tflops", "y"]


def first_matching_numeric(mapping: Mapping[str, Any],
    patterns: Sequence[str]) -> Tuple[Optional[str], Optional[float]]:
    for key, value in mapping.items():
        low = str(key).lower()
        if any(low == p or p in low for p in patterns):
            x = to_float(value)
            if x is not None:
                return str(key), x
    return None, None


def _point_label(obj: Mapping[str, Any], path: str) -> str:
    label = path or ("P"
        "oint")
    for key in obj:
        if str(key).lower() in {"name", "label", "series", "type"}:
            label = str(obj[key])
            break
    return label


def collect_point_candidates(obj: Any, path: str = "") -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if isinstance(obj, Mapping):
        xk, x = first_matching_numeric(obj, X_KEY_PATTERNS)
        yk, y = first_matching_numeric(obj, Y_KEY_PATTERNS)
        if x is not None and y is not None and xk != yk:
            points.append({"x": x, "y": y, "label": _point_label(obj, path)})
        for key, value in obj.items():
            points.extend(collect_point_candidates(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            points.extend(collect_point_candidates(value, f"{path}[{i}]"))
    return points


def _advice_item_lines(item: Any) -> List[str]:
    cleaned_lines: List[str] = []
    for line in str(item).splitlines():
        cleaned = re.sub(r"^[\s\t\-\d\)\.]+", "", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    return cleaned_lines


def _collect_advice_item(found: List[str], item: Any) -> None:
    for cleaned in _advice_item_lines(item):
        if cleaned not in found:
            found.append(cleaned)


def collect_top_level_advice(records: Sequence[Mapping[str, Any]]) -> List[str]:
    found: List[str] = []
    for record in records:
        value = record.get("advice")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            _collect_advice_item(found, item)
    return found


def _normalized_ratio_fraction(value: Any, measured: Optional[float], limit: Optional[float]) -> Optional[float]:
    """Return a ratio as a fraction while preserving profiler variability.

    Native profiler versions may emit either a fraction (0.14) or percentage
    points (14.0). When the native value is absent, derive the ratio only from
    the canonical measured point and its active roofline limit.
    """
    ratio = to_float(value)
    if ratio is not None and ratio >= 0:
        if ratio > 1.5 and ratio <= 100:
            ratio /= 100.0
        return ratio
    if measured is not None and limit is not None and limit > 0:
        return measured / limit
    return None


def _roofline_bound(arithmetic_intensity: float, knee_intensity: Optional[float]) -> str:
    if knee_intensity is None or knee_intensity <= 0:
        return "unknown"
    tolerance = 0.02
    if arithmetic_intensity < knee_intensity * (1 - tolerance):
        return "bandwidth-bound region"
    if arithmetic_intensity > knee_intensity * (1 + tolerance):
        return "compute-bound region"
    return "ridge region"


def _roofline_chart_rank(title: str, source_index: int) -> Tuple[int, int]:
    """Order architecture-level memory roofs before execution-unit roofs.

    This is a semantic display rule, not an operator-specific title whitelist.
    Unknown profiler titles retain their source order after known categories.
    """
    text = title.casefold()
    hierarchy_tokens = ("gm", "l2", "dram", "global memory", "memory hierarchy")
    unit_tokens = ("memory unit", "vector", "cube", "simt", "simd", "tensor")
    if any(token in text for token in hierarchy_tokens):
        return 0, source_index
    if any(token in text for token in unit_tokens):
        return 1, source_index
    return 2, source_index


def _roofline_curve_metrics(
    raw: Mapping[str, Any],
    x: float,
    y: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    bandwidth = to_float(raw.get("bw"))
    compute_ceiling = to_float(raw.get("computility"))
    has_curve = bandwidth is not None and bandwidth > 0 and compute_ceiling is not None and compute_ceiling > 0
    knee_intensity = compute_ceiling / bandwidth if has_curve else None
    roofline_limit = min(bandwidth * x, compute_ceiling) if has_curve else None
    ratio_fraction = _normalized_ratio_fraction(raw.get("ratio"), y, roofline_limit)
    return bandwidth, compute_ceiling, knee_intensity, roofline_limit, ratio_fraction


def _roofline_row_item(
    raw: Mapping[str, Any],
    group_index: int,
    row_index: int,
    title: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    point = raw.get("point")
    x = y = None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x = to_float(point[0])
        y = to_float(point[1])
    if x is None or y is None:
        _, x = first_matching_numeric(raw, X_KEY_PATTERNS)
        _, y = first_matching_numeric(raw, Y_KEY_PATTERNS)
        if x is None or y is None:
            return None
    if x <= 0 or y <= 0:
        return None

    bandwidth, compute_ceiling, knee_intensity, roofline_limit, ratio_fraction = _roofline_curve_metrics(raw, x, y)
    bandwidth_name = str(raw.get("bw_name") or "Bandwidth")
    compute_name = str(raw.get("computility_name") or "Compute")
    series_id = f"{group_index}:{row_index}"
    item = {
        "series_id": series_id,
        "curve_id": f"{bandwidth_name}|{compute_name}|{bandwidth}|{compute_ceiling}",
        "bandwidth": bandwidth,
        "bandwidth_name": bandwidth_name,
        "compute_ceiling": compute_ceiling,
        "compute_name": compute_name,
        "point": {"x": x, "y": y},
        "arithmetic_intensity": x,
        "measured_performance": y,
        "knee_intensity": knee_intensity,
        "roofline_limit": roofline_limit,
        "ratio": ratio_fraction,
        "ratio_percent": ratio_fraction * 100 if ratio_fraction is not None else None,
        "bound_region": _roofline_bound(x, knee_intensity),
    }
    flat_point = {
        "series_id": series_id,
        "chart_title": title,
        "x": x,
        "y": y,
        "label": f"{title} · {bandwidth_name} · {compute_name}",
    }
    for shared_key in ("bandwidth", "compute_ceiling", "knee_intensity", "roofline_limit",
                       "ratio", "ratio_percent", "bound_region"):
        flat_point[shared_key] = item.get(shared_key)
    return item, flat_point


def _roofline_group_rows(
    group: Mapping[str, Any],
    group_index: int,
    title: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    flat_points: List[Dict[str, Any]] = []
    for row_index, raw in enumerate(group.get("rooflines") or []):
        if not isinstance(raw, Mapping):
            continue
        parsed = _roofline_row_item(raw, group_index, row_index, title)
        if parsed is None:
            continue
        item, flat_point = parsed
        rows.append(item)
        flat_points.append(flat_point)
    return rows, flat_points


def _roofline_native_charts(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    charts: List[Dict[str, Any]] = []
    flat_points: List[Dict[str, Any]] = []
    seen_chart_signatures: set[Tuple[Any, ...]] = set()

    for record in records:
        source = record.get("multiple_rooflines")
        if not isinstance(source, list):
            continue
        for group_index, group in enumerate(source):
            if not isinstance(group, Mapping):
                continue
            title = str(group.get("title") or f"Roofline {group_index + 1}")
            rows, group_flat_points = _roofline_group_rows(group, group_index, title)
            flat_points.extend(group_flat_points)
            if not rows:
                continue
            signature = (
                title,
                tuple((r["bandwidth_name"], r["compute_name"], r["point"]["x"], r["point"]["y"]) for r in rows),
            )
            if signature in seen_chart_signatures:
                continue
            seen_chart_signatures.add(signature)
            charts.append({
                "title": title,
                "source_index": group_index,
                "series": rows,
                "bandwidth_names": list(dict.fromkeys(r["bandwidth_name"] for r in rows)),
                "compute_names": list(dict.fromkeys(r["compute_name"] for r in rows)),
                "axis": {
                    "x_label": "Arithmetic Intensity",
                    "x_unit": "Ops/Byte",
                    "y_label": "Performance",
                    "y_unit": "TOPS/s",
                    "scale": "logarithmic",
                },
            })
    return charts, flat_points


def _roofline_fallback_charts(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Any] = []
    for record in records:
        if isinstance(record.get("multiple_rooflines"), list):
            items.extend(record["multiple_rooflines"])
    points = collect_point_candidates(items)
    seen = set()
    unique_points = []
    for point in points:
        key = (round(point["x"], 12), round(point["y"], 12), point["label"])
        if key not in seen and point["x"] > 0 and point["y"] > 0:
            seen.add(key)
            unique_points.append(point)
    if not unique_points:
        return [], unique_points
    series: List[Dict[str, Any]] = []
    for index, point in enumerate(unique_points):
        series.append({
            "series_id": f"fallback:{index}",
            "curve_id": f"fallback:{index}",
            "bandwidth": None,
            "bandwidth_name": "Measured point",
            "compute_ceiling": None,
            "compute_name": str(point.get("label") or "Point"),
            "point": {"x": point["x"], "y": point["y"]},
            "arithmetic_intensity": point["x"],
            "measured_performance": point["y"],
            "knee_intensity": None,
            "roofline_limit": None,
            "ratio": None,
            "ratio_percent": None,
            "bound_region": "unknown",
        })
    charts = [{
        "title": "Roofline",
        "source_index": 0,
        "series": series,
        "bandwidth_names": ["Measured point"],
        "compute_names": list(dict.fromkeys(str(point.get("label") or "Point") for point in unique_points)),
        "axis": {
            "x_label": "Arithmetic Intensity",
            "x_unit": "Ops/Byte",
            "y_label": "Performance",
            "y_unit": "TOPS/s",
            "scale": "logarithmic",
        },
    }]
    return charts, unique_points


def parse_roofline(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize native and fallback Roofline payloads into an interactive model.

    The model deliberately describes semantics rather than any specific
    operator: every native chart, bandwidth source, compute ceiling and point
    becomes a self-contained series that can be rendered alone or composed
    with other report modules.
    """
    charts, flat_points = _roofline_native_charts(records)

    # Backward-compatible fallback for earlier x/y-shaped payloads. It keeps
    # measured coordinates visible without inventing bandwidths or ceilings.
    if not charts:
        charts, flat_points = _roofline_fallback_charts(records)

    charts.sort(key=lambda chart: _roofline_chart_rank(str(chart.get("title", "")), int(chart.get("source_index", 0))))
    chart_order = {str(chart.get("title")): i for i, chart in enumerate(charts)}
    flat_points.sort(key=lambda point: (chart_order.get(str(point.get("chart_title",
        str(point.get("label", "")).split((" "
        "· "), 1)[0])), 999), str(point.get("label", ""))))

    advice = collect_top_level_advice(records)
    return {
        "available": bool(charts or advice),
        "plot_available": bool(charts),
        "charts": charts,
        "points": flat_points[:400],
        "advice": advice,
        "coordinate_contract": "point=[arithmetic_intensity_ops_per_byte, measured_performance_tops_per_s]",
        "interaction_contract": {
            "continuous_crosshair": True,
            "axis_coordinate_readout": True,
            "point_tooltip": True,
            "curve_tooltip": True,
            "curve_hover_emphasis": True,
        },
    }


class _TraceSink(NamedTuple):
    lanes: Dict[Tuple[str, str], List[Dict[str, Any]]]
    stacks: Dict[Tuple[str, str], List[Dict[str, Any]]]
    starts: List[float]
    ends: List[float]


class _TraceEvent(NamedTuple):
    name: str
    category: str
    start: float
    duration: float
    args: Any


def _trace_add_event(sink: _TraceSink, lane: Tuple[str, str], event: _TraceEvent) -> None:
    if event.duration < 0:
        return
    item = {
        "name": event.name,
        "category": event.category,
        "start": event.start,
        "duration": event.duration,
        "args": event.args if isinstance(event.args, Mapping) else {},
    }
    sink.lanes[lane].append(item)
    sink.starts.append(event.start)
    sink.ends.append(event.start + event.duration)


def _trace_consume(events: Sequence[Any], sink: _TraceSink) -> None:
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        ph = str(raw.get("ph", ""))
        pid = str(raw.get("pid", "process"))
        tid = str(raw.get("tid", "lane"))
        ts = to_float(raw.get("ts"))
        if ph == "X":
            dur = to_float(raw.get("dur"))
            if ts is None or dur is None:
                continue
            _trace_add_event(sink, (pid, tid), _TraceEvent(str(raw.get("name", "event")),
                str(raw.get("cat", raw.get("cname", ""))), ts, dur, raw.get(("a"
                "rgs"), {})))
        elif ph == "B" and ts is not None:
            sink.stacks[(pid, tid)].append({
                "name": str(raw.get("name", "event")),
                "category": str(raw.get("cat", raw.get("cname", ""))),
                "start": ts,
                "args": raw.get("args", {}),
            })
        elif ph == "E" and ts is not None and sink.stacks[(pid, tid)]:
            begin = sink.stacks[(pid, tid)].pop()
            _trace_add_event(sink, (pid, tid), _TraceEvent(begin["name"], begin["category"],
                begin["start"], ts - begin["start"], begin["args"]))


def _trace_out_lanes(
    lanes: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_events: int,
) -> Tuple[List[Dict[str, Any]], int]:
    out_lanes: List[Dict[str, Any]] = []
    embedded_count = 0
    for (pid, tid), items in sorted(lanes.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        items.sort(key=lambda x: (x["start"], x["duration"]))
        remaining = max_events - embedded_count
        if remaining <= 0:
            break
        kept = items[:remaining]
        embedded_count += len(kept)
        out_lanes.append({"pid": pid, "tid": tid, "label": f"{pid} / {tid}", "events": kept})
    return out_lanes, embedded_count


def _trace_timeline_kind(lanes: Mapping[Tuple[str, str], List[Dict[str, Any]]]) -> Tuple[str, str]:
    lane_tokens = " ".join(f"{pid} {tid}" for pid, tid in lanes).upper()
    event_tokens = " ".join(
        str(item.get("name", ""))
        for items in lanes.values()
        for item in items[:20]
    ).upper()
    combined_tokens = f"{lane_tokens} {event_tokens}"
    communication_markers = ("HCCL", "AI CPU", "AIC BLOCK", "AIV BLOCK", "ASCENDC API", "TURN")
    pipe_markers = ("SCALAR", "VECTOR", "MTE1", "MTE2", "MTE3", "CUBE", "FIXPIPE")
    if any(token in combined_tokens for token in communication_markers):
        return "communication", "Communication-Compute Timeline"
    if any(token in combined_tokens for token in pipe_markers):
        return "pipe", "Pipe Timeline"
    return "generic", "Timeline"


def parse_trace_events(events: Sequence[Any], max_events: int = 10000) -> Dict[str, Any]:
    sink = _TraceSink(defaultdict(list), defaultdict(list), [], [])
    _trace_consume(events, sink)
    lanes, starts, ends = sink.lanes, sink.starts, sink.ends

    if not starts:
        return {"available": False, "lanes": []}
    t0, t1 = min(starts), max(ends)
    out_lanes, embedded_count = _trace_out_lanes(lanes, max_events)
    timeline_type, timeline_title = _trace_timeline_kind(lanes)
    return {
        "available": True,
        "timeline_type": timeline_type,
        "title": timeline_title,
        "start": t0,
        "end": t1,
        "duration": t1 - t0,
        "event_count": sum(len(v) for v in lanes.values()),
        "embedded_event_count": embedded_count,
        "lanes": out_lanes,
    }


def load_trace_from_path(path: Optional[Path], max_events: int) -> Optional[Dict[str, Any]]:
    if not path or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        # The trace can be tens of MB while only ~max_events are embedded, but
        # event_count, B/E stack pairing and the global t0/t1 range all require
        # the full event list, so a partial/streaming parse would not be exact.
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load trace from %s: %s", path, exc)
        return None
    events = raw.get("traceEvents", raw if isinstance(raw, list) else []) if isinstance(raw, (dict, list)) else []
    if not isinstance(events, list):
        return None
    return parse_trace_events(events, max_events=max_events)


def parse_timeline(artifacts: ArtifactSet, max_events: int) -> Dict[str, Any]:
    parsed = load_trace_from_path(artifacts.trace, max_events)
    if parsed and parsed.get("available"):
        return parsed
    records = load_visualize_records(artifacts.visualize_data)
    for record in records:
        if isinstance(record.get("traceEvents"), list):
            return parse_trace_events(record["traceEvents"], max_events=max_events)
    return {"available": False, "lanes": []}
