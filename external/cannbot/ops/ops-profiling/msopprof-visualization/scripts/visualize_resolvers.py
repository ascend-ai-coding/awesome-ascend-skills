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

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from visualize_common import (
    ArtifactSet,
    NormalizedCollection,
    VisualizationError,
    humanize_key,
    iter_dicts,
    positive,
    ratio_pct,
    safe_relative_path,
    to_float,
)
from visualize_parsers import (
    collect_advice,
    load_visualize_records,
    normalize_table_per_block,
    parse_basic,
    parse_cache,
    parse_compute,
    parse_occupancy,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceBundle:
    block_id: str
    artifacts: ArtifactSet
    records: List[Dict[str, Any]]
    csv_rows: Dict[str, List[Dict[str, Any]]]


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as exc:
        logger.debug("Failed to read CSV rows %s: %s", path, exc)
        return []


def load_source_bundle(collection: NormalizedCollection, block_id: str) -> SourceBundle:
    artifacts = collection.blocks.get(block_id, ArtifactSet(None, None, {}))
    return SourceBundle(
        block_id=block_id,
        artifacts=artifacts,
        records=load_visualize_records(artifacts.visualize_data),
        csv_rows={name: read_csv_rows(path) for name, path in artifacts.csv.items()},
    )


def build_source_bundles(collection: NormalizedCollection) -> Dict[str, SourceBundle]:
    ids = ["details", "raw_data", "roofline", "memory_detail"]
    return {block_id: load_source_bundle(collection, block_id) for block_id in ids}


def csv_rows_named(bundle: SourceBundle, filename: str) -> List[Dict[str, Any]]:
    return list(bundle.csv_rows.get(filename) or [])


def row_valid_for_prefix(row: Mapping[str, Any], prefix: str) -> bool:
    time_key = f"{prefix}_time(us)"
    cycles_key = f"{prefix}_total_cycles"
    if positive(row.get(time_key)) or positive(row.get(cycles_key)):
        return True
    return any(k.startswith(prefix + "_") and positive(v) for k, v in row.items())


def _obj_has_simt_evidence(obj: Mapping[str, Any]) -> bool:
    for key, value in obj.items():
        low = str(key).lower()
        if "simt" not in low or not (positive(value) or isinstance(value, Mapping)):
            continue
        return True
    return False


def _records_simt_evidence(records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        for obj in iter_dicts(record):
            if _obj_has_simt_evidence(obj):
                return True
    return False


def _records_op_type(records: Sequence[Mapping[str, Any]]) -> str:
    for record in records:
        if record.get("op_type"):
            return str(record.get("op_type")).lower()
    return ""


def _records_table_names(records: Sequence[Mapping[str, Any]]) -> set:
    table_names: set[str] = set()
    for record in records:
        for block in record.get("table_per_block") or [] if isinstance(record.get("table_per_block"), list) else []:
            for table in block.get("table_detail") or [] if isinstance(block, Mapping) else []:
                table_names.add(str(table.get("table_name", "")).lower())
    return table_names


def _architecture_kind(has_aic: bool, has_aiv: bool, has_simd: bool, has_simt: bool) -> str:
    if has_aic and has_aiv:
        return "mix"
    if has_aic:
        return "cube"
    if has_aiv and has_simd and has_simt:
        return "vector-hybrid"
    if has_aiv and has_simt:
        return "vector-simt"
    if has_aiv and has_simd:
        return "vector-simd"
    if has_aiv:
        return "vector"
    return "unknown"


def detect_architecture(bundle: SourceBundle) -> Dict[str, Any]:
    arithmetic = csv_rows_named(bundle, "ArithmeticUtilization.csv")
    pipe = csv_rows_named(bundle, "PipeUtilization.csv")
    all_rows = arithmetic + pipe
    has_aic = any(row_valid_for_prefix(r, "aic") for r in all_rows)
    has_aiv = any(row_valid_for_prefix(r, "aiv") for r in all_rows)
    has_simd = any(positive(r.get("aiv_vec_vf_ratio")) for r in arithmetic)
    has_simt = any(positive(r.get("aiv_vec_simt_vf_ratio")) for r in arithmetic)
    # Occupancy payload can expose SIMT evidence even when BasicInfo still says vector.
    if _records_simt_evidence(bundle.records):
        has_simt = True
    op_type = _records_op_type(bundle.records)
    if not has_aic and "cube" in op_type:
        has_aic = True
    if not has_aiv and "vector" in op_type:
        has_aiv = True
    # Schema-driven fallback: table names reveal architecture even when CSVs are absent.
    table_names = _records_table_names(bundle.records)
    if table_names & {"cube", "l0a", "l0b", "l0c", "l1"}:
        has_aic = True
    if table_names & {"vector", "ub", "dcache"}:
        has_aiv = True

    return {
        "kind": _architecture_kind(has_aic, has_aiv, has_simd, has_simt),
        "has_aic": has_aic,
        "has_aiv": has_aiv,
        "has_simd": has_simd,
        "has_simt": has_simt,
    }


def source_priority(block_id: str) -> int:
    # Raw Default is the richest canonical metric suite, then Roofline, then Details.
    return {"raw_data": 4, "roofline": 3, "memory_detail": 2, "details": 1}.get(block_id, 0)


def resolve_basic(bundles: Mapping[str, SourceBundle]) -> Dict[str, Any]:
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for block_id in ["details", "raw_data", "roofline"]:
        bundle = bundles[block_id]
        model = parse_basic(bundle.records)
        if not model.get("available"):
            continue
        score = (
            len(model.get("blocks") or []) * 10
            + (5 if model.get("duration_us") is not None else 0)
            + source_priority(block_id)
        )
        if best is None or score > best[0]:
            best = (score, model)
    return best[1] if best else {"available": False, "blocks": []}


def resolve_occupancy(bundles: Mapping[str, SourceBundle]) -> Dict[str, Any]:
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for block_id in ["details", "raw_data", "roofline", "memory_detail"]:
        model = parse_occupancy(bundles[block_id].records)
        if not model.get("available"):
            continue
        score = (
            len(model.get("cores") or []) * 100 + len(model.get("metrics") or [])
            + (10 if block_id == "details" else 0)
        )
        if best is None or score > best[0]:
            best = (score, model)
    return best[1] if best else {"available": False, "cores": [], "metrics": []}


def index_csv_by_subblock(bundle: SourceBundle, filename: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in csv_rows_named(bundle, filename):
        key = (str(row.get("block_id", "")), str(row.get("sub_block_id", "")))
        out[key] = row
    return out


def percent_metric(name: str, value: Any, *, category: str = "pipeline") -> Optional[Dict[str, Any]]:
    pct = ratio_pct(value)
    if pct is None:
        return None
    return {"name": name, "value": pct, "category": category, "format": "percent"}


def number_metric(name: str, value: Any, *, category: str = "counter") -> Optional[Dict[str, Any]]:
    x = to_float(value)
    if x is None:
        return None
    return {"name": name, "value": x, "category": category, "format": "number"}


def compact_metrics(items: Iterable[Optional[Dict[str, Any]]], include_zero: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if not include_zero and abs(float(item["value"])) < 1e-12:
            continue
        key = item["name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class _MetricBuckets(NamedTuple):
    pipeline: List[Dict[str, Any]]
    breakdown: List[Dict[str, Any]]
    waits: List[Dict[str, Any]]
    counters: List[Dict[str, Any]]


def _generic_field_skippable(low: str, prefix: str, known_fields: set) -> bool:
    if not low.startswith(prefix + "_") or low in known_fields:
        return True
    if low.endswith("_time(us)") or low.endswith("_total_cycles"):
        return True
    return "icache_miss_rate" in low


def _generic_compute_metric(low: str, label: str, raw: Any, buckets: _MetricBuckets) -> bool:
    if "ratio" in low or "rate" in low:
        metric = percent_metric(label, raw,
            category="wait" if ("wait" in low or "cflt" in low or "conflict" in low) else ("p"
            "ipeline"))
        if not metric:
            return False
        if "wait" in low or "cflt" in low or "conflict" in low:
            buckets.waits.append(metric)
        elif any(token in low for token in ["_fp_", "_int_", "_vf_", "simt", "sfu"]):
            metric["category"] = "breakdown"
            buckets.breakdown.append(metric)
        else:
            buckets.pipeline.append(metric)
        return True
    if any(token in low for token in ["instr", "count", "number"]):
        metric = number_metric(label, raw)
        if metric:
            buckets.counters.append(metric)
            return True
    return False


def extend_generic_compute_metrics(prefix: str, rows: Sequence[Mapping[str, Any]],
    buckets: _MetricBuckets) -> None:
    existing = {m["name"] for m in buckets.pipeline + buckets.breakdown + buckets.waits + buckets.counters}
    known_fields = {
        "aic_cube_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_mte3_ratio", "aic_fixpipe_ratio",
        "aic_cube_fp_ratio", "aic_cube_int_ratio", "aic_cube_wait_ratio", "aic_mte1_wait_ratio", ("aic_mte2_wait_rat"
            "io"), "aic_mte3_wait_ratio",
        "aic_cube_total_instr_number", "aic_cube_fp_instr_number", "aic_cube_int_instr_number",
        "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_vec_vf_ratio", ("aiv_vec_simt_"
            "vf_ratio"), "aiv_vec_sfu_ratio",
        "aiv_vec_wait_ratio", "aiv_mte2_wait_ratio", "aiv_mte3_wait_ratio", "aiv_vec_stu_cflt_ratio", ("aiv_vec_ldu_"
            "cflt_ratio"), "aiv_vec_sfu_cflt_ratio",
    }
    for row in rows:
        for key, raw in row.items():
            low = str(key).lower()
            if _generic_field_skippable(low, prefix, known_fields):
                continue
            label = humanize_key(key)
            if label in existing:
                continue
            if _generic_compute_metric(low, label, raw, buckets):
                existing.add(label)


def _cube_engine(ar: Mapping[str, Any], pr: Mapping[str, Any], cr: Mapping[str, Any], sub_id: str) -> Dict[str, Any]:
    pipeline = compact_metrics([
        percent_metric("Cube", pr.get("aic_cube_ratio")),
        percent_metric("Scalar", pr.get("aic_scalar_ratio")),
        percent_metric("MTE1", pr.get("aic_mte1_ratio")),
        percent_metric("MTE2", pr.get("aic_mte2_ratio")),
        percent_metric("MTE3", pr.get("aic_mte3_ratio")),
        percent_metric("FixPipe", pr.get("aic_fixpipe_ratio")),
    ])
    breakdown = compact_metrics([
        percent_metric("Cube FP", ar.get("aic_cube_fp_ratio"), category="breakdown"),
        percent_metric("Cube INT", ar.get("aic_cube_int_ratio"), category="breakdown"),
    ])
    waits = compact_metrics([
        percent_metric("Cube Wait", cr.get("aic_cube_wait_ratio"), category="wait"),
        percent_metric("MTE1 Wait", cr.get("aic_mte1_wait_ratio"), category="wait"),
        percent_metric("MTE2 Wait", cr.get("aic_mte2_wait_ratio"), category="wait"),
        percent_metric("MTE3 Wait", cr.get("aic_mte3_wait_ratio"), category="wait"),
    ])
    counters = compact_metrics([
        number_metric("Cube Instructions", ar.get("aic_cube_total_instr_number")),
        number_metric("Cube FP Instructions", ar.get("aic_cube_fp_instr_number")),
        number_metric("Cube INT Instructions", ar.get("aic_cube_int_instr_number")),
    ])
    extend_generic_compute_metrics("aic", [ar, pr, cr], _MetricBuckets(pipeline, breakdown, waits, counters))
    waits = [m for m in waits if abs(float(m.get("value", 0))) > 1e-12]
    return {
        "engine_id": sub_id or "cube",
        "engine_type": "cube",
        "label": f"CUBE · {sub_id}" if sub_id else "CUBE",
        "time_us": to_float(pr.get("aic_time(us)")) or to_float(ar.get("aic_time(us)")),
        "cycles": to_float(pr.get("aic_total_cycles")) or to_float(ar.get("aic_total_cycles")),
        "pipeline": pipeline,
        "breakdown": breakdown,
        "waits": waits,
        "counters": counters,
    }


def _vector_engine(ar: Mapping[str, Any], pr: Mapping[str, Any], cr: Mapping[str, Any], sub_id: str) -> Dict[str, Any]:
    pipeline = compact_metrics([
        percent_metric("Vector", pr.get("aiv_vec_ratio")),
        percent_metric("Scalar", pr.get("aiv_scalar_ratio")),
        percent_metric("MTE2", pr.get("aiv_mte2_ratio")),
        percent_metric("MTE3", pr.get("aiv_mte3_ratio")),
    ])
    breakdown = compact_metrics([
        percent_metric("SIMD VF", ar.get("aiv_vec_vf_ratio"), category="breakdown"),
        percent_metric("SIMT VF", ar.get("aiv_vec_simt_vf_ratio"), category="breakdown"),
        percent_metric("SFU", ar.get("aiv_vec_sfu_ratio"), category="breakdown"),
    ])
    waits = compact_metrics([
        percent_metric("Vector Wait", cr.get("aiv_vec_wait_ratio"), category="wait"),
        percent_metric("MTE2 Wait", cr.get("aiv_mte2_wait_ratio"), category="wait"),
        percent_metric("MTE3 Wait", cr.get("aiv_mte3_wait_ratio"), category="wait"),
        percent_metric("STU Conflict", cr.get("aiv_vec_stu_cflt_ratio"), category="wait"),
        percent_metric("LDU Conflict", cr.get("aiv_vec_ldu_cflt_ratio"), category="wait"),
        percent_metric("SFU Conflict", cr.get("aiv_vec_sfu_cflt_ratio"), category="wait"),
    ])
    counters: List[Dict[str, Any]] = []
    # Preserve instruction counters when available in current/future schemas.
    for key, value in ar.items():
        low = str(key).lower()
        if low.startswith("aiv_") and "instr" in low and positive(value):
            metric = number_metric(humanize_key(key), value)
            if metric:
                counters.append(metric)
    extend_generic_compute_metrics("aiv", [ar, pr, cr], _MetricBuckets(pipeline, breakdown, waits, counters))
    waits = [m for m in waits if abs(float(m.get("value", 0))) > 1e-12]
    return {
        "engine_id": sub_id or "vector",
        "engine_type": "vector",
        "label": f"VECTOR · {sub_id}" if sub_id else "VECTOR",
        "time_us": to_float(pr.get("aiv_time(us)")) or to_float(ar.get("aiv_time(us)")),
        "cycles": to_float(pr.get("aiv_total_cycles")) or to_float(ar.get("aiv_total_cycles")),
        "pipeline": pipeline,
        "breakdown": breakdown,
        "waits": waits,
        "counters": counters,
    }


def _compute_score(block_models: Sequence[Mapping[str, Any]], arch: Mapping[str, Any]) -> float:
    score = 0.0
    for block in block_models:
        for engine in block["engines"]:
            score += 100
            score += 6 * sum(1 for m in engine["pipeline"] if abs(m["value"]) > 1e-9)
            score += 10 * sum(1 for m in engine["breakdown"] if abs(m["value"]) > 1e-9)
            score += 3 * sum(1 for m in engine["waits"] if abs(m["value"]) > 1e-9)
    if arch["kind"] in {"mix", "vector-hybrid"}:
        score += 40
    return score


def parse_compute_csv(bundle: SourceBundle) -> Dict[str, Any]:
    arithmetic = index_csv_by_subblock(bundle, "ArithmeticUtilization.csv")
    pipe = index_csv_by_subblock(bundle, "PipeUtilization.csv")
    conflict = index_csv_by_subblock(bundle, "ResourceConflictRatio.csv")
    keys = sorted(set(arithmetic) | set(pipe) | set(conflict),
        key=lambda x: (int(x[0]) if x[0].isdigit() else 10**9, x[1]))
    blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    arch = detect_architecture(bundle)

    for block_id, sub_id in keys:
        ar = arithmetic.get((block_id, sub_id), {})
        pr = pipe.get((block_id, sub_id), {})
        cr = conflict.get((block_id, sub_id), {})
        if row_valid_for_prefix({**ar, **pr}, "aic"):
            blocks[block_id].append(_cube_engine(ar, pr, cr, sub_id))
        if row_valid_for_prefix({**ar, **pr}, "aiv"):
            blocks[block_id].append(_vector_engine(ar, pr, cr, sub_id))

    block_models = [{"block_id": bid, "engines": engines} for bid, engines in blocks.items() if engines]
    block_models.sort(key=lambda x: int(x["block_id"]) if str(x["block_id"]).isdigit() else 10**9)
    return {
        "available": bool(block_models),
        "architecture": arch,
        "blocks": block_models,
        "block_ids": [b["block_id"] for b in block_models],
        "score": _compute_score(block_models, arch) + source_priority(bundle.block_id),
        "source_block": bundle.block_id,
    }


def _bin_fallback_engine(unit_type: str, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pipeline = []
    counters = []
    for item in items:
        if str(item.get("unit")) == "3":
            metric = percent_metric(str(item.get("name")), item.get("value"))
            if metric:
                pipeline.append(metric)
        else:
            metric = number_metric(str(item.get("name")), item.get("value"))
            if metric:
                counters.append(metric)
    return {
        "engine_id": unit_type,
        "engine_type": unit_type.lower(),
        "label": unit_type.upper(),
        "time_us": None,
        "cycles": None,
        "pipeline": compact_metrics(pipeline),
        "breakdown": [],
        "waits": [],
        "counters": compact_metrics(counters),
    }


def parse_compute_bin_fallback(bundle: SourceBundle) -> Dict[str, Any]:
    old = parse_compute(bundle.records)
    blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for block_id in old.get("block_ids") or []:
        rows = [x for x in old.get("items") or [] if str(x.get("block_id")) == str(block_id)]
        by_type: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[str(row.get("block_type") or "compute")].append(row)
        for unit_type, items in by_type.items():
            blocks[str(block_id)].append(_bin_fallback_engine(unit_type, items))
    block_models = [{"block_id": k, "engines": v} for k, v in blocks.items() if v]
    block_models.sort(key=lambda x: int(x["block_id"]) if x["block_id"].isdigit() else 10**9)
    return {
        "available": bool(block_models),
        "architecture": detect_architecture(bundle),
        "blocks": block_models,
        "block_ids": [x["block_id"] for x in block_models],
        "score": len(block_models) * 10 + source_priority(bundle.block_id),
        "source_block": bundle.block_id,
    }


def resolve_compute(bundles: Mapping[str, SourceBundle]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for block_id in ["raw_data", "roofline", "memory_detail", "details"]:
        bundle = bundles[block_id]
        model = parse_compute_csv(bundle)
        if model.get("available"):
            candidates.append(model)
        else:
            fallback = parse_compute_bin_fallback(bundle)
            if fallback.get("available"):
                candidates.append(fallback)
    if not candidates:
        return {"available": False, "blocks": [], "block_ids": [], "architecture": {"kind": "unknown"}}
    return max(candidates, key=lambda x: x.get("score", 0))


def table_lookup(block: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for table in block.get("tables") or []:
        rows = {}
        for row in table.get("rows") or []:
            rows[str(row.get("name", "")).lower()] = dict(row.get("values") or {})
        out[str(table.get("name", "")).lower()] = rows
    return out


def metric_from_row(tables: Mapping[str, Any], table: str, row: str, contains: str) -> Optional[float]:
    values = (tables.get(table.lower()) or {}).get(row.lower()) or {}
    for key, value in values.items():
        if contains.lower() in str(key).lower():
            x = to_float(value)
            if x is not None:
                return x
    return None


def edge_from_row(tables: Mapping[str, Any], endpoints: Tuple[str, str],
    table: str, row: str, label: str) -> Optional[Dict[str, Any]]:
    bw = metric_from_row(tables, table, row, "throughput")
    req = metric_from_row(tables, table, row, "request")
    peak = metric_from_row(tables, table, row, "peak")
    if bw is None and req is None and peak is None:
        return None
    src, dst = endpoints
    return {"src": src, "dst": dst, "label": label, "bandwidth": bw, "request": req, "peak_ratio": peak}


MEMORY_PATH_EDGE_MAP: Dict[int, Tuple[str, str]] = {
    # Common GM/L2 boundary.
    0: ("GM", "L2 Cache"),
    1: ("L2 Cache", "GM"),
    # Cube memory system.
    2: ("L2 Cache", "L1"),
    6: ("L1", "L0B"),
    7: ("L1", "L0A"),
    8: ("Cube", "L0C"),
    37: ("L0B", "Cube"),
    38: ("L0A", "Cube"),
    39: ("L0C", "FixPipe"),
    40: ("FixPipe", "L2 Cache"),
    41: ("FixPipe", "L1"),
    # Vector memory system.
    12: ("L2 Cache", "UB"),
    13: ("UB", "L2 Cache"),
    14: ("UB", "Vector"),
    15: ("Vector", "UB"),
    48: ("L2 Cache", "DCache"),
    52: ("Vector", "DCache"),
    54: ("DCache", "Vector"),
}


def core_memory_records(bundle: SourceBundle) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for record in bundle.records:
        source = record.get("core_memory_map")
        if isinstance(source, list):
            records.extend(x for x in source if isinstance(x, Mapping))
    return records


def bind_core_memory_records(
    raw_blocks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Bind each table_per_block entry to its exact core_memory_map record.

    `block_id` is an execution-block index. `core_no` is a physical-core
    assignment and may repeat across dispatch rounds. They must not be joined by
    equality. When core_memory_map omits block_id, source ordinal is the only
    lossless binding if the two arrays have equal length.
    """
    if not raw_blocks:
        return []

    totals: Dict[str, int] = defaultdict(int)
    for core in records:
        totals[str(core.get("core_no", ""))] += 1
    seen: Dict[str, int] = defaultdict(int)

    explicit: Dict[str, Mapping[str, Any]] = {}
    for core in records:
        if core.get("block_id") is not None:
            explicit[str(core.get("block_id"))] = core

    positional = len(records) == len(raw_blocks) and len(records) > 0
    unique_core_no = len(records) == len({str(x.get("core_no", "")) for x in records})
    bindings: List[Dict[str, Any]] = []
    for index, block in enumerate(raw_blocks):
        block_id = str(block.get("block_id", ""))
        core: Optional[Mapping[str, Any]] = explicit.get(block_id)
        binding = "explicit_block_id" if core is not None else "none"
        if core is None and positional:
            core = records[index]
            binding = "source_ordinal"
        elif core is None and unique_core_no:
            core = next((x for x in records if str(x.get("core_no", "")) == block_id), None)
            binding = "unique_core_no_fallback" if core is not None else "none"

        physical_core_id = str(core.get("core_no", "")) if core is not None else None
        dispatch_index = None
        dispatch_count = None
        if core is not None:
            dispatch_count = totals.get(str(core.get("core_no", "")), 1)
            dispatch_index = seen[str(core.get("core_no", ""))] + 1
            seen[str(core.get("core_no", ""))] += 1
        bindings.append({
            "block_id": block_id,
            "core_record": core,
            "binding": binding,
            "physical_core_id": physical_core_id,
            "dispatch_index": dispatch_index,
            "dispatch_count": dispatch_count,
            "source_ordinal": index,
        })
    return bindings


def authoritative_memory_core(
    bundle: SourceBundle,
    block_id: str,
    *,
    ordinal: Optional[int] = None,
) -> Optional[Mapping[str, Any]]:
    """Return an authoritative core_memory_map record.

    Prefer explicit source ordinal when available. The legacy `core_no ==
    block_id` fallback is retained only for old single-record/unit tests where
    core_no is unique.
    """
    records = core_memory_records(bundle)
    if ordinal is not None and 0 <= ordinal < len(records):
        return records[ordinal]
    for core in records:
        if core.get("block_id") is not None and str(core.get("block_id")) == str(block_id):
            return core
    matches = [core for core in records if str(core.get("core_no", "")) == str(block_id)]
    return matches[0] if len(matches) == 1 else None


def _authoritative_edges(core: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    edges: List[Dict[str, Any]] = []
    unmapped: List[Dict[str, Any]] = []
    for unit in core.get("memory_unit") or []:
        if not isinstance(unit, Mapping):
            continue
        try:
            path_id = int(unit.get("memory_path"))
        except (TypeError, ValueError):
            path_id = -1
        pair = MEMORY_PATH_EDGE_MAP.get(path_id)
        item = {
            "memory_path": path_id,
            "bandwidth": to_float(unit.get("bandwidth")),
            "request": to_float(unit.get("request")),
            "peak_ratio": to_float(unit.get("peak_ratio")),
            "display": unit.get("display"),
            "source_kind": "core_memory_map",
        }
        if not pair:
            unmapped.append(item)
            continue
        src, dst = pair
        edge_id = f"{src}->{dst}"
        edges.append({
            "edge_id": edge_id,
            "src": src,
            "dst": dst,
            "source": src,
            "target": dst,
            **item,
        })
    return edges, unmapped


def _authoritative_node_metrics(core: Mapping[str, Any]) -> Dict[str, str]:
    node_metrics: Dict[str, str] = {}
    l2 = core.get("L2cache")
    if isinstance(l2, Mapping):
        hit = to_float(l2.get("hit_ratio"))
        if hit is not None:
            node_metrics["L2 Cache"] = f"Hit Rate: {hit:.2f}%"
    for engine_name, node_name in (("Cube", "Cube"), ("Vector", "Vector")):
        engine = core.get(engine_name)
        if isinstance(engine, Mapping):
            ratio = to_float(engine.get("ratio"))
            if ratio is not None:
                node_metrics[node_name] = f"Active: {ratio:.2f}%"
    return node_metrics


def authoritative_memory_edge_table(
    bundle: SourceBundle,
    block_id: str,
    core_record: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    """Build directed edges 1:1 from one exact core_memory_map record.

    No value is sourced from CSV/native tables when a mapped authoritative edge
    exists. This prevents cross-block replacement, cross-replay replacement, and
    edge-label drift.
    """
    core = core_record or authoritative_memory_core(bundle, block_id)
    if not core:
        return [], [], {}
    edges, unmapped = _authoritative_edges(core)
    return edges, unmapped, _authoritative_node_metrics(core)


def _edge_present(edges: List[Dict[str, Any]], edge_id: str) -> bool:
    return any(e.get("edge_id") == edge_id for e in edges)


def _fallback_add_edge(
    edges: List[Dict[str, Any]],
    edge: Optional[Dict[str, Any]],
    source_kind: str = "native_table",
) -> None:
    if not edge:
        return
    edge_id = f"{edge['src']}->{edge['dst']}"
    if _edge_present(edges, edge_id):
        return
    edges.append({
        "edge_id": edge_id,
        "source": edge["src"],
        "target": edge["dst"],
        "source_kind": source_kind,
        **edge,
    })


_MEMORY_COMPONENTS = {"GM", "L2 Cache", "L1", "L0A", "L0B", "L0C", "UB", "DCache", "Vector", "Cube", "FixPipe"}


def _normalize_memory_component(raw: str) -> Optional[str]:
    text = raw.strip().replace("FIXP", "FixPipe")
    aliases = {"Main Memory": "GM", "L2": "L2 Cache", "Dcache": "DCache"}
    text = aliases.get(text, text)
    if "/" in text or text.upper() == "MTE":
        return None
    return text if text in _MEMORY_COMPONENTS else None


def _component_title(text: str) -> str:
    return text.title().replace("L0a", "L0A").replace("L0b", "L0B").replace("L0c", "L0C")


def _fallback_inferred_edge(row_name: str, values: Mapping[str, Any], edges: List[Dict[str, Any]]) -> None:
    m = re.match(r"^(.+?)\s+(Read|Write)\s+(.+)$", row_name, re.I)
    if not m:
        return
    left, verb, right = m.group(1), m.group(2).lower(), m.group(3)
    a = _normalize_memory_component(_component_title(left))
    b = _normalize_memory_component(_component_title(right))
    if not a or not b:
        return
    src, dst = (b, a) if verb == "read" else (a, b)
    edge_id = f"{src}->{dst}"
    if _edge_present(edges, edge_id):
        return
    bw = next((to_float(v) for k, v in values.items() if "throughput" in str(k).lower()), None)
    req = next((to_float(v) for k, v in values.items() if "request" in str(k).lower()), None)
    peak = next((to_float(v) for k, v in values.items() if "peak" in str(k).lower()), None)
    if bw is None and req is None and peak is None:
        return
    edges.append({
        "edge_id": edge_id,
        "src": src,
        "dst": dst,
        "source": src,
        "target": dst,
        "label": row_name,
        "bandwidth": bw,
        "request": req,
        "peak_ratio": peak,
        "source_kind": "native_inference",
    })


def _fallback_inferred_edges(tables: Mapping[str, Any], edges: List[Dict[str, Any]]) -> None:
    for rows in tables.values():
        for row_name, values in rows.items():
            _fallback_inferred_edge(row_name, values, edges)


def fallback_memory_edge_table(block: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Semantic fallback used only when core_memory_map has no mapped topology edges."""
    tables = table_lookup(block)
    edges: List[Dict[str, Any]] = []

    # Known exact semantic rows. These are never allowed to overwrite authoritative edges.
    _fallback_add_edge(edges, edge_from_row(tables, ("GM", "L2 Cache"), "GM", "Read Main Memory", "GM Read"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L2 Cache", "GM"), "GM", "Write Main Memory", "GM Write"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L2 Cache", "L1"), "L1", "L1 Read GM", "L2 → L1"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L1", "L0A"), "L0A", "L0A Read L1/GM", "L1 → L0A"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L1", "L0B"), "L0B", "L0B Read L1/GM", "L1 → L0B"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L0A", "Cube"), "Cube", "Cube Read L0A", "L0A → Cube"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L0B", "Cube"), "Cube", "Cube Read L0B", "L0B → Cube"))
    _fallback_add_edge(edges, edge_from_row(tables, ("Cube", "L0C"), "Cube", "Cube Write L0C", "Cube → L0C"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L0C", "FixPipe"), "L0C", "L0C Write FIXP", "L0C → FixPipe"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L2 Cache", "UB"), "UB", "UB Read GM", "L2 → UB"))
    _fallback_add_edge(edges, edge_from_row(tables, ("UB", "Vector"), "Vector", "Vector Read UB", "UB → Vector"))
    _fallback_add_edge(edges, edge_from_row(tables, ("Vector", "UB"), "Vector", "Vector Write UB", "Vector → UB"))
    _fallback_add_edge(edges, edge_from_row(tables, ("L2 Cache", "DCache"), "DCache", "DCache Read GM", "L2 → DCache"))
    _fallback_add_edge(edges, edge_from_row(tables, ("DCache", "Vector"), "DCache", "DCache Write Vector",
        "DCache → Vector"))
    _fallback_add_edge(edges, edge_from_row(tables, ("Vector", "DCache"), "DCache", "DCache Read Vector",
        "Vector → DCache"))

    _fallback_inferred_edges(tables, edges)
    return edges


def topology_adapter(architecture: Mapping[str, Any], nodes: Sequence[str]) -> str:
    if architecture.get("has_aic") and architecture.get("has_aiv"):
        return "mix"
    if architecture.get("has_aic"):
        return "cube"
    has_ub = "UB" in nodes
    has_dcache = "DCache" in nodes
    if has_ub and has_dcache:
        return "vector-hybrid"
    if has_dcache:
        return "vector-simt"
    return "vector-simd"


def _topology_nodes(edges: Sequence[Mapping[str, Any]], architecture: Mapping[str, Any]) -> List[str]:
    nodes: List[str] = []
    for edge in edges:
        for node in [edge.get("src"), edge.get("dst")]:
            if node and node not in nodes:
                nodes.append(str(node))

    # Preserve architecture nodes for stable layout, even when an edge is zero or missing.
    desired: List[str] = ["GM", "L2 Cache"]
    if architecture.get("has_aic"):
        desired += ["L1", "L0A", "L0B", "Cube", "L0C", "FixPipe"]
    if architecture.get("has_aiv"):
        if architecture.get("has_simd") or "UB" in nodes:
            desired.append("UB")
        if architecture.get("has_simt") or "DCache" in nodes:
            desired.append("DCache")
        desired.append("Vector")
    for node in desired:
        if node not in nodes:
            nodes.append(node)
    return nodes


def _fallback_node_metrics(
    tables: Mapping[str, Any],
    node_metrics: Dict[str, str],
    compute_block: Optional[Mapping[str, Any]],
) -> None:
    # Fallback node metrics only when authoritative core_memory_map did not provide them.
    if "L2 Cache" not in node_metrics:
        cache_rate = metric_from_row(tables, "Cache", "L2 Cache Total", "hit rate")
        if cache_rate is not None:
            node_metrics["L2 Cache"] = f"Hit Rate: {cache_rate:.2f}%"
    if compute_block:
        for engine in compute_block.get("engines") or []:
            primary = next((m for m in engine.get("pipeline") or [] if m.get("name") in {"Cube", "Vector"}), None)
            node = "Cube" if engine.get("engine_type") == "cube" else "Vector"
            if primary and node not in node_metrics:
                node_metrics[node] = f"Active: {primary['value']:.2f}%"


class _CoreBinding(NamedTuple):
    record: Optional[Mapping[str, Any]] = None
    meta: Optional[Mapping[str, Any]] = None


def build_memory_topology(
    bundle: SourceBundle,
    block: Mapping[str, Any],
    architecture: Mapping[str, Any],
    compute_block: Optional[Mapping[str, Any]],
    core_binding: Optional[_CoreBinding] = None,
) -> Dict[str, Any]:
    block_id = str(block.get("block_id", ""))
    core_binding = core_binding or _CoreBinding()
    edges, unmapped, node_metrics = authoritative_memory_edge_table(bundle, block_id,
        core_record=core_binding.record)
    source_kind = "core_memory_map" if edges else "native_fallback"
    if not edges:
        edges = fallback_memory_edge_table(block)

    nodes = _topology_nodes(edges, architecture)
    _fallback_node_metrics(table_lookup(block), node_metrics, compute_block)

    adapter = topology_adapter(architecture, nodes)
    binding = dict(core_binding.meta or {})
    binding.pop("core_record", None)
    # Payload economy: the renderer derives `edge_id` as `f"{src}->{dst}"`
    # client-side, and `source`/`target` duplicate `src`/`dst`. Emit one
    # naming pair only. `edges` is the single edge list (the historical
    # `edge_table` alias was byte-identical and is no longer duplicated; the
    # template still falls back to it for old payloads).
    compact_edges: List[Dict[str, Any]] = []
    for edge in edges:
        compact = {k: v for k, v in edge.items() if k not in {"edge_id", "source", "target"}}
        compact_edges.append(compact)
    return {
        "adapter": adapter,
        "nodes": nodes,
        "edges": compact_edges,
        "node_metrics": node_metrics,
        "source_kind": source_kind,
        "unmapped_paths": unmapped,
        "execution_binding": binding,
    }


class _MemoryCandidateCtx(NamedTuple):
    bundle: SourceBundle
    bindings: Sequence[Mapping[str, Any]]
    arch: Mapping[str, Any]
    compute_by_block: Mapping[str, Any]


def _candidate_block_tables(block: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int, set]:
    tables: List[Dict[str, Any]] = []
    nonzero_cells = 0
    groups: set[str] = set()
    for table in block.get("tables") or []:
        rows: List[Dict[str, Any]] = []
        for row in table.get("rows") or []:
            values = dict(row)
            values.pop("block_id", None)
            name = str(values.pop("name", ""))
            nonzero_cells += sum(1 for value in values.values() if positive(value))
            rows.append({"name": name, "values": values})
        if rows:
            groups.add(str(table.get("name", "Unknown")))
            tables.append({"name": str(table.get("name", "Unknown")),
                "headers": list(table.get("headers") or []), ("r"
                "ows"): rows})
    return tables, nonzero_cells, groups


def _candidate_occurrence(
    index: int,
    block: Mapping[str, Any],
    ctx: _MemoryCandidateCtx,
) -> Tuple[Dict[str, Any], int, int, set]:
    block_id = str(block.get("block_id", ""))
    tables, nonzero_cells, groups = _candidate_block_tables(block)
    binding = ctx.bindings[index] if index < len(ctx.bindings) else {
        "block_id": block_id, "core_record": None, "binding": "none", "source_ordinal": index,
        "physical_core_id": None, "dispatch_index": None, "dispatch_count": None,
    }
    block_model = {
        "block_id": block_id,
        "tables": tables,
        "execution_binding": {k: v for k, v in binding.items() if k != "core_record"},
    }
    topology = build_memory_topology(
        ctx.bundle,
        block_model,
        ctx.arch,
        ctx.compute_by_block.get(block_id),
        core_binding=_CoreBinding(record=binding.get("core_record"), meta=binding),
    )
    authoritative_count = sum(1 for e in topology["edges"] if e.get("source_kind") == "core_memory_map")
    edge_count = sum(1 for e in topology["edges"] if positive(e.get("bandwidth")) or positive(e.get("request")))
    if authoritative_count:
        nonzero_cells += authoritative_count * 8
    block_model["topology"] = topology
    return block_model, nonzero_cells, edge_count, groups


def _group_occurrence_blocks(
    occurrence_blocks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    grouped_blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    block_order: List[str] = []
    for occurrence in occurrence_blocks:
        key = str(occurrence.get("block_id", ""))
        if key not in grouped_blocks:
            block_order.append(key)
        grouped_blocks[key].append(occurrence)
    blocks: List[Dict[str, Any]] = []
    for block_id in block_order:
        occurrences = grouped_blocks[block_id]
        primary = dict(occurrences[0])
        primary["occurrences"] = occurrences
        primary["occurrence_count"] = len(occurrences)
        # tables/topology are byte-identical to occurrences[0] and the UI
        # always drills into occurrences, so keep only identity/binding
        # metadata at the block top level.
        primary.pop("tables", None)
        primary.pop("topology", None)
        blocks.append(primary)
    return blocks, grouped_blocks


def parse_memory_candidate(bundle: SourceBundle, compute_model: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    raw_blocks, _ = normalize_table_per_block(bundle.records)
    arch = detect_architecture(bundle)
    compute_by_block = {str(b.get("block_id")): b for b in (compute_model or {}).get("blocks") or []}
    memory_records = core_memory_records(bundle)
    bindings = bind_core_memory_records(raw_blocks, memory_records)
    ctx = _MemoryCandidateCtx(bundle, bindings, arch, compute_by_block)
    occurrence_blocks: List[Dict[str, Any]] = []
    nonzero_cells = 0
    edge_count = 0
    groups: set[str] = set()
    for index, block in enumerate(raw_blocks):
        block_model, cells, block_edges, table_groups = _candidate_occurrence(index, block, ctx)
        occurrence_blocks.append(block_model)
        nonzero_cells += cells
        edge_count += block_edges
        groups |= table_groups
    score = (
        len(occurrence_blocks) * 100 + nonzero_cells + len(groups) * 25
        + edge_count * 20 + source_priority(bundle.block_id)
    )
    # The profiler can emit several execution records with the same logical
    # block_id (for this payload: four dispatch rounds per Block ID). Keep every
    # record, but group them under one UI Block ID so the selector never repeats
    # 0..255 four times and the first dispatch matches the native Insight view.
    blocks, grouped_blocks = _group_occurrence_blocks(occurrence_blocks)

    binding_modes = sorted({str(x.get("binding")) for x in bindings if x.get("binding")})
    repeated_core_ids = {str(x.get("physical_core_id")) for x in bindings if (x.get("dispatch_count") or 0) > 1}
    repeated_block_ids = sum(1 for occurrences in grouped_blocks.values() if len(occurrences) > 1)
    return {
        "available": bool(blocks),
        "architecture": arch,
        "blocks": blocks,
        "groups": sorted(groups),
        "advice": collect_advice(bundle.records),
        "score": score,
        "source_block": bundle.block_id,
        "topology_contract": "directed-edge-table/v2",
        "block_binding_contract": ("table_per_block ↔ core_memory_map by explicit block_id or validated "
            "source ordinal; core_no is metadata only"),
        "binding_modes": binding_modes,
        "execution_record_count": len(occurrence_blocks),
        # Number of unique logical Block IDs after repeated dispatch
        # occurrences have been grouped (one entry per element of `blocks`).
        "unique_block_count": len(blocks),
        "repeated_block_ids": repeated_block_ids,
        "max_occurrences_per_block": max((len(x) for x in grouped_blocks.values()), default=1),
        "repeated_physical_core_assignments": len(repeated_core_ids),
        "occurrence_contract": "one logical Block ID with one or more source-ordered dispatch occurrences",
    }


def resolve_memory(bundles: Mapping[str, SourceBundle], compute_model: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = []
    for block_id in ["raw_data", "roofline", ("memory_detai"
        "l"), "details"]:
        candidates.append(parse_memory_candidate(bundles[block_id], compute_model))
    candidates = [x for x in candidates if x.get("available")]
    if not candidates:
        return {"available": False, "blocks": [], "architecture": {"kind": "unknown"}}
    return max(candidates, key=lambda x: x.get("score", 0))


def _cache_nonzero_cells(model: Mapping[str, Any]) -> int:
    nonzero = 0
    for cache_block in model.get("blocks") or []:
        for cell_value in (cache_block.get("values") or {}).values():
            if positive(cell_value):
                nonzero += 1
    return nonzero


def resolve_cache(bundles: Mapping[str, SourceBundle]) -> Dict[str, Any]:
    candidates = []
    for block_id in ["raw_data", "roofline", "memory_detail", "details"]:
        model = parse_cache(bundles[block_id].records)
        if model.get("available"):
            score = (
                len(model.get("blocks") or []) * 100 + _cache_nonzero_cells(model)
                + len(model.get("metrics") or []) * 5 + source_priority(block_id)
            )
            candidates.append((score, model))
    if not candidates:
        return {"available": False, "blocks": [], "metrics": []}
    best = max(candidates, key=lambda x: x[0])[1]
    if best.get("families"):
        # Every cell the heatmap renders lives in `families`; `blocks` is the
        # legacy no-families fallback shape and duplicates the family cells,
        # so shrink it. The legacy renderer path only reads `blocks` when no
        # families exist, in which case this branch never runs.
        best = dict(best, blocks=[])
    return best


def resolve_advice(bundles: Mapping[str, SourceBundle]) -> Dict[str, Any]:
    items: List[str] = []
    for block_id in ["details", "raw_data", "roofline", "memory_detail"]:
        for item in collect_advice(bundles[block_id].records):
            if item not in items:
                items.append(item)
    return {"available": bool(items), "items": items}


def _read_internal_block_logs(collection: NormalizedCollection, block_id: str) -> str:
    log_root = collection.root / "_internal" / "logs"
    if not log_root.is_dir():
        return ""
    parts: List[str] = []
    for path in sorted(log_root.glob(f"{block_id}*.log")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            logger.debug("Failed to read block log %s: %s", path, exc)
            continue
    return "\n".join(parts)


def _source_cause(entry: Mapping[str, Any], logs: str) -> Tuple[str, str]:
    lower = logs.lower()
    reason = str(entry.get("reason") or "").strip()
    explanation = reason or (
        ("The Source replay completed without the source snapshots, line map, and instruction map required by "
            "the Source explorer.")
    )
    if "missed debug_line information" in lower or ("debug_line" in lower and "miss" in lower):
        return "missing_debug_line", (
            "The profiler reported that the selected kernel is missing usable debug-line information. "
            "As a result, it could not build source-line or instruction mappings."
        )
    if "hot spot function calculate data failed" in lower or "generate hot spot function failed" in lower:
        return "source_hotspot_generation_failed", (
            ("The profiler ran successfully, but source-hotspot generation produced no source snapshots, line "
                "map, or instruction map.")
        )
    if entry.get("status") == "unavailable":
        return "source_collection_unavailable", explanation
    return "empty_source_payload", explanation


def _source_diagnostic(cause_code: str, explanation: str, entry: Mapping[str, Any],
    coverage: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = [
        {"label": "Collector status", "value": str(entry.get("status") or "unknown")},
        {"label": "Profiler return code",
            "value": str(entry.get("return_code") if entry.get("return_code") is not None else ("N"
            "A"))},
        {"label": "Source snapshots", "value": str(coverage.get("source_snapshot_count", 0))},
        {"label": "Mapped source lines", "value": str(coverage.get("mapped_line_records", 0))},
        {"label": "Instruction records", "value": str(coverage.get("instruction_count", 0))},
    ]
    return {
        "code": cause_code,
        "title": "Source mapping unavailable",
        "explanation": explanation,
        "evidence": evidence,
        "requirements": [
            "The exact executable/kernel selected for profiling must contain usable .debug_line tables.",
            "The Source payload must contain source snapshots, a source-line map, and an instruction map.",
            "Semantic coverage, not command exit code alone, determines whether the Source page is available.",
        ],
        "remediation": [
            "Rebuild the exact selected executable/kernel with line-table debug data (-g).",
            ("Verify that the profiler is using that rebuilt artifact rather than another build or "
                "scenario directory."),
            "Rerun targeted Source collection and accept the result only when source_mapping coverage is true.",
        ],
    }


def attach_source_diagnostic(collection: NormalizedCollection, model: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach an auditable unavailability diagnosis without inventing source data."""
    out = dict(model)
    blocks = collection.manifest.get("blocks") or {}
    entry = blocks.get("source") if isinstance(blocks, Mapping) else None
    entry = entry if isinstance(entry, Mapping) else {}
    coverage = entry.get("coverage") if isinstance(entry.get("coverage"), Mapping) else {}
    logs = _read_internal_block_logs(collection, "source")
    cause_code, explanation = _source_cause(entry, logs)
    out.update({
        "available": False,
        "files": out.get("files", []),
        "instructions": out.get("instructions", []),
        "relations": out.get("relations", []),
        "diagnostic": _source_diagnostic(cause_code, explanation, entry, coverage),
    })
    return out


def _onchip_memory_from_artifacts(artifacts: Mapping[str, Any], root: Path) -> Optional[Path]:
    for key in ["memory_info", "memory_info.json", "onchip_memory"]:
        rel_path = artifacts.get(key)
        if isinstance(rel_path, str) and rel_path:
            return safe_relative_path(root, rel_path)
    extra = artifacts.get("json")
    if isinstance(extra, Mapping):
        for filename, rel_path in extra.items():
            if str(filename).lower() == "memory_info.json" and isinstance(rel_path, str):
                return safe_relative_path(root, rel_path)
    return None


def resolve_onchip_memory_path(collection: NormalizedCollection,
    explicit_path: Optional[Path] = None) -> Optional[Path]:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.is_file():
            raise VisualizationError(f"--memory-info does not point to a file: {candidate}")
        return candidate
    blocks = collection.manifest.get("blocks") or {}
    for block in blocks.values():
        if not isinstance(block, Mapping):
            continue
        artifacts = block.get("artifacts") or {}
        if not isinstance(artifacts, Mapping):
            continue
        found = _onchip_memory_from_artifacts(artifacts, collection.root)
        if found is not None:
            return found
    root_artifacts = collection.manifest.get("artifacts") or {}
    if isinstance(root_artifacts, Mapping):
        rel_path = root_artifacts.get("memory_info") or root_artifacts.get("memory_info.json")
        if isinstance(rel_path, str) and rel_path:
            return safe_relative_path(collection.root, rel_path)
    return None
