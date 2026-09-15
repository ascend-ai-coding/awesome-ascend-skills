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

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from visualize_common import (  # noqa: F401
    ArtifactSet,
    NormalizedCollection,
    PAGE_ORDER,
    SUPPORTED_INPUT_SCHEMAS,
    VisualizationError,
    flatten_numeric,
    humanize_key,
    iter_dicts,
    load_collection,
    mean,
    median,
    metric_is_percent,
    normalize_artifacts,
    percentile,
    positive,
    ratio_pct,
    read_json,
    safe_relative_path,
    to_float,
)
from visualize_parsers import (  # noqa: F401
    OCCUPANCY_PACKED_LAYOUTS,
    OCCUPANCY_PACKED_LAYOUT_ALIASES,
    X_KEY_PATTERNS,
    Y_KEY_PATTERNS,
    _first_key,
    _normalized_ratio_fraction,
    _numeric_pair,
    _parse_embedded_blob,
    _roofline_bound,
    _roofline_chart_rank,
    _scan_embedded_candidate,
    _walk_json_nodes,
    collect_advice,
    collect_point_candidates,
    collect_top_level_advice,
    extract_embedded_json_records,
    first_matching_numeric,
    load_trace_from_path,
    load_visualize_records,
    normalize_packed_occupancy,
    normalize_soc_name,
    normalize_table_per_block,
    occupancy_packed_schema,
    parse_basic,
    parse_block_detail,
    parse_cache,
    parse_compute,
    parse_memory,
    parse_occupancy,
    parse_onchip_memory,
    parse_roofline,
    parse_timeline,
    parse_trace_events,
    recursive_find_key_values,
)
from visualize_source import (  # noqa: F401
    SOURCE_FRAME_FILE,
    SOURCE_FRAME_INSTRUCTION_MAP,
    SOURCE_FRAME_LINE_MAP,
    SOURCE_PATH_SLOT_BYTES,
    _address_int,
    _frame_json,
    _instruction_semantic_signature,
    _per_core_values,
    _preferred_stall,
    _raw_instruction_signature,
    _raw_source_line_lookup,
    _scalar_sum,
    _source_fusion_compatibility,
    _source_kind,
    _source_location,
    _source_path_key,
    _stall_metric,
    _stall_percent,
    _stall_score,
    normalize_instruction_timeline,
    normalize_source,
    normalize_stall,
    normalize_stall_from_source,
    parse_native_visualize_frames,
    parse_raw_data,
    parse_source_artifact,
    read_csv_headers,
    read_csv_table,
)
from visualize_resolvers import (  # noqa: F401
    MEMORY_PATH_EDGE_MAP,
    SourceBundle,
    _read_internal_block_logs,
    attach_source_diagnostic,
    authoritative_memory_core,
    authoritative_memory_edge_table,
    bind_core_memory_records,
    build_memory_topology,
    build_source_bundles,
    compact_metrics,
    core_memory_records,
    csv_rows_named,
    detect_architecture,
    edge_from_row,
    extend_generic_compute_metrics,
    fallback_memory_edge_table,
    index_csv_by_subblock,
    load_source_bundle,
    metric_from_row,
    number_metric,
    parse_compute_bin_fallback,
    parse_compute_csv,
    parse_memory_candidate,
    percent_metric,
    read_csv_rows,
    resolve_advice,
    resolve_basic,
    resolve_cache,
    resolve_compute,
    resolve_memory,
    resolve_occupancy,
    resolve_onchip_memory_path,
    row_valid_for_prefix,
    source_priority,
    table_lookup,
    topology_adapter,
)
from visualize_render import (  # noqa: F401
    PAYLOAD_TAG_COMPRESSED,
    PAYLOAD_TAG_PLAIN,
    compact_payload_for_delivery,
    export_memory_edge_tables,
    export_source_tables,
    render_report,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# The skill tree must stay free of __pycache__ artifacts (self_check rule).
sys.dont_write_bytecode = True

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)

VERSION = "4.8.0"


REPORT_SCHEMA = "msopprof-visualization/v3"


DETAIL_SECTIONS = ["occupancy", "compute", "memory", "advice"]


PRESETS: Dict[str, List[str]] = {
    "complete": list(PAGE_ORDER),
    "core": ["details", "roofline", "timeline", "cache", "raw-data"],
    "fast": ["details", "cache"],
}


FEATURE_ALIASES = {
    "stall": "warp-stall",
    "warp_stall": "warp-stall",
    "instruction_timeline": "instruction-timeline",
    "raw": "raw-data",
    "raw_data": "raw-data",
    "onchip_memory": "onchip-memory",
    "on-chip-memory": "onchip-memory",
    "onchip": "onchip-memory",
}


FEATURE_INFO: Dict[str, Dict[str, Any]] = {
    "details": {
        "page": "details",
        "collector_feature": "details",
        "collector_block": "details",
        "artifact": "visualize_data",
        "structures": ["op_detail", "subblock_detail", "core_memory_map", "table_per_block", "advice"],
        "visuals": ["Basic Information cards", "Block Duration strip chart",
            "Outlier-first Core Occupancy", ("Per-block "
            "compute-unit panels"), "Insight-style memory topology", "Profiler advice"],
        "purpose": "Inspect the main execution, compute, memory, and load-balance characteristics from one replay.",
        "use_cases": ["Core imbalance", "Pipeline utilization", "Memory-path pressure", "Profiler recommendations"],
    },
    "occupancy": {
        "page": "details",
        "detail_section": "occupancy",
        "collector_feature": "occupancy",
        "collector_block": "details",
        "artifact": "visualize_data",
        "structures": ["op_detail"],
        "visuals": ["Dynamic metric selector", "Physical-core/subcore matrix",
            "Robust high/low outlier flags", ("Selected-core "
            "detail")],
        "purpose": "Compare physical core/subcore execution metrics without assuming a fixed metric set.",
        "use_cases": ["Load imbalance", "Outlier cores", "Throughput spread", "Per-core cache differences"],
    },
    "compute": {
        "page": "details",
        "detail_section": "compute",
        "collector_feature": "compute",
        "collector_block": "memory_detail",
        "artifact": "visualize_data",
        "structures": ["subblock_detail"],
        "visuals": ["Block selector", "Compute-unit panels", "0–100% utilization bars", "Unit-aware detail tables"],
        "purpose": "Show which compute/pipeline metrics dominate and how they vary across blocks.",
        "use_cases": ["Vector/Cube activity", "Wait behavior", "Block-level variability"],
    },
    "memory": {
        "page": "details",
        "detail_section": "memory",
        "collector_feature": "memory",
        "collector_block": "memory_detail",
        "artifact": "visualize_data",
        "structures": ["core_memory_map.memory_unit -> directed edge table", "table_per_block"],
        "visuals": ["Block selector", "Bandwidth/Request switch",
            "Profiler-path topology with edge values", ("Per-group "
            "native tables")],
        "purpose": "Compare requests, throughput, bandwidth, and peak ratios across profiler-defined memory groups.",
        "use_cases": ["GM bandwidth", "UB pressure", "DCache traffic", "Memory-path bottlenecks"],
    },
    "advice": {
        "page": "details",
        "detail_section": "advice",
        "collector_feature": "advice",
        "collector_block": "memory_detail",
        "artifact": "visualize_data",
        "structures": ["advice"],
        "visuals": ["Deduplicated profiler advice"],
        "purpose": "Present profiler recommendations alongside the metrics that produced them.",
        "use_cases": ["Optimization guidance", "Fast triage"],
    },
    "roofline": {
        "page": "roofline",
        "collector_feature": "roofline",
        "collector_block": "roofline",
        "artifact": "visualize_data",
        "structures": ["multiple_rooflines", "advice"],
        "visuals": ["Interactive logarithmic Roofline", "Continuous coordinate crosshair", ("Point and curve "
            "tooltips"), "Bound classification"],
        "purpose": ("Normalize native Roofline semantics and expose them as a standalone or composable "
            "interactive module without inventing missing coordinates."),
        "use_cases": ["Compute-bound vs memory-bound", "Arithmetic-intensity positioning"],
    },
    "timeline": {
        "page": "timeline",
        "collector_feature": "timeline",
        "collector_block": "timeline",
        "artifact": "trace",
        "structures": ["traceEvents"],
        "visuals": ["Hierarchical lane timeline", "Strong Pipe colors", "Zoom/Fit/Reset", "Slice Detail", ("Range "
            "selection"), "Slice List statistics"],
        "purpose": "Inspect execution order and overlap across dynamic profiler lanes.",
        "use_cases": ["Pipeline overlap", "Long events", "Lane imbalance", "Serialization"],
    },
    "cache": {
        "page": "cache",
        "collector_feature": "cache",
        "collector_block": "memory_detail",
        "artifact": "visualize_data",
        "structures": ["table_per_block -> *Cache* tables", "optional cacheline event matrices and source linkage"],
        "visuals": ["Paired Hit/Miss event heatmaps", "Single-cell detail", "Event count/share modes", ("Full-screen "
            "heatmap enlargement"), "Optional Source linkage"],
        "purpose": ("Present L2/iCache hit and miss behavior as paired event maps. The index semantics are "
            "stated explicitly: cacheline when native cacheline data exist, otherwise profiler block "
            "aggregation."),
        "use_cases": ["Hit/miss concentration", "Cache outlier blocks or cachelines", ("Single-cell event "
            "inspection"), "Source linkage when debug mapping exists"],
    },
    "onchip-memory": {
        "page": "onchip-memory",
        "collector_feature": "onchip-memory",
        "collector_block": "onchip_memory",
        "artifact": "memory_info.json",
        "structures": ["memory_info.json buffer/tensor lifetimes", "address ranges", ("temporary-variable "
            "metadata"), "bank/group assignments"],
        "visuals": ["Execution-order × on-chip-address memory block map", "Selected allocation detail", ("Bank/group "
            "physical distribution")],
        "purpose": ("Visualize UB/on-chip allocation lifetimes and physical bank distribution from "
            "memory_info.json without substituting unrelated MemoryDetail counters."),
        "use_cases": ["Peak on-chip memory", "Fragmentation",
            "Long-lived buffers", "UB overflow risk", ("Bank-conflict "
            "layout inspection")],
    },
    "source": {
        "page": "source",
        "collector_feature": "source",
        "collector_block": "source",
        "artifact": "visualize_data",
        "structures": ["native framed source snapshots", "Files → Lines → address ranges", ("Instructions → "
            "source location / opcode / pipe / GPR status"), "optional PCSampling Files/Instructions stall payload"],
        "visuals": ["Viewport-filling source explorer",
            "Bidirectional source-line ↔ instruction navigation", ("Per-core "
            "execution counters"), "All Samples / Not Issue segmented stall bars",
                "Raw GPR-lifetime/status overlay", ("Instruction "
                "detail drill-down")],
        "purpose": ("Decode native Source payloads, fuse compatible PCSampling stall records by validated "
            "source-line and instruction-address identities, and preserve auditable relations without assuming "
            "one operator, path, source suffix, or fixed instruction count."),
        "use_cases": ["User source and included-header inspection",
            "Hot or highly stalled source lines", ("Instruction "
            "mapping"), "PC drill-down", "Register-pressure context", "Stall-reason composition"],
    },
    "warp-stall": {
        "page": "warp-stall",
        "collector_feature": "warp-stall",
        "collector_block": "warp_stall",
        "artifact": "visualize_data",
        "structures": ["stall/reason/ratio/count/PC signatures"],
        "visuals": ["Stall category bars", "PC detail"],
        "purpose": "Summarize stall reasons and high-stall PCs when sampling data are available.",
        "use_cases": ["Warp bottlenecks", "Stall-root-cause analysis"],
    },
    "instruction-timeline": {
        "page": "instruction-timeline",
        "collector_feature": "instruction-timeline",
        "collector_block": "instruction_timeline",
        "artifact": "visualize_data",
        "structures": ["traceEvents", "instruction/pipe/start/duration signatures"],
        "visuals": ["Instruction lane timeline"],
        "purpose": "Inspect instruction-level scheduling when a supported payload exists.",
        "use_cases": ["Pipe overlap", "Instruction serialization", "Long instruction spans"],
    },
    "timeline-detail": {
        "page": "instruction-timeline",
        "collector_feature": "timeline-detail",
        "collector_block": "timeline_detail",
        "artifact": "visualize_data",
        "structures": ["TimelineDetail instruction/source signatures"],
        "visuals": ["Scenario-specific instruction timeline / source-hotspot fallback"],
        "purpose": "Use TimelineDetail payloads when direct Source or instrTimeLine data are unavailable.",
        "use_cases": ["Framework operator instruction timing", "Simulation code hotspot"],
    },
    "raw-data": {
        "page": "raw-data",
        "collector_feature": "raw-data",
        "collector_block": "raw_data",
        "artifact": "csv",
        "structures": ["manifest-declared CSV files"],
        "visuals": ["Table selector", "Search", "Sort", "Row count"],
        "purpose": "Inspect native profiler tables without exposing filesystem paths or command lines.",
        "use_cases": ["Field validation", "Custom investigation", "Cross-checking visual summaries"],
    },
}


def normalize_feature_name(value: str) -> str:
    name = value.strip().lower().replace("_", "-")
    return FEATURE_ALIASES.get(name, name)


def _accumulate_detail_sections(page_sections: Dict[str, List[str]], info: Mapping[str, Any],
    feature: str) -> None:
    if feature == "details":
        page_sections["details"] = list(DETAIL_SECTIONS)
        return
    section = info.get("detail_section")
    if section and section not in page_sections["details"]:
        page_sections["details"].append(section)


def resolve_requested_features(preset: str, raw_features: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    if raw_features:
        selected: List[str] = []
        for raw in raw_features:
            name = normalize_feature_name(raw)
            if name not in FEATURE_INFO:
                raise VisualizationError(f"Unknown feature: {raw}. Use --list-features.")
            if name not in selected:
                selected.append(name)
    else:
        selected = list(PRESETS[preset])

    page_sections: Dict[str, List[str]] = defaultdict(list)
    pages: List[str] = []
    for feature in selected:
        info = FEATURE_INFO[feature]
        page = info["page"]
        if page not in pages:
            pages.append(page)
        if page == "details":
            _accumulate_detail_sections(page_sections, info, feature)

    # Preset/explicit page-level Details defaults to all sections.
    if "details" in pages and not page_sections["details"]:
        page_sections["details"] = list(DETAIL_SECTIONS)
    ordered_pages = [p for p in PAGE_ORDER if p in pages]
    return ordered_pages, dict(page_sections)


def print_feature_list() -> None:
    cli_logger.info("Available msOpProf visualization features\n")
    cli_logger.info(f"{'FEATURE':24} {'REPORT PAGE':24} {'COLLECTOR BLOCK':22} {'PRIMARY ARTIFACT'}")
    cli_logger.info("-" * 108)
    for name, info in FEATURE_INFO.items():
        cli_logger.info(f"{name:24} {info['page']:24} {info['collector_block']:22} {info['artifact']}")
    cli_logger.info("\nDefault: --preset complete")
    cli_logger.info("Targeted example: --feature occupancy --feature cache")


def print_feature_explanation(raw: str) -> None:
    name = normalize_feature_name(raw)
    if name not in FEATURE_INFO:
        raise VisualizationError(f"Unknown feature: {raw}. Use --list-features.")
    info = dict(FEATURE_INFO[name])
    payload = {
        "feature": name,
        "report_page": info["page"],
        "collector_requirement": f"--feature {info['collector_feature']}",
        "collector_block": info["collector_block"],
        "primary_artifact": info["artifact"],
        "input_structures": info["structures"],
        "visual_components": info["visuals"],
        "purpose": info["purpose"],
        "use_cases": info["use_cases"],
        "targeted_visualization_command": (
            "python scripts/visualize.py --input /path/to/collection "
            f"--output /path/to/report --feature {name}"
        ),
    }
    cli_logger.info(json.dumps(payload, ensure_ascii=False, indent=2))


def page_view_available(page: str, view: Mapping[str, Any]) -> bool:
    if page == "details":
        if view.get("show_basic") is not False and (view.get("basic") or {}).get("available"):
            return True
        for section_name in view.get("sections") or []:
            if (view.get(section_name) or {}).get("available"):
                return True
        return False
    return bool(view.get("available"))


def _evidence_details(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sections": [name for name in view.get("sections") or [] if (view.get(name) or {}).get("available")],
        "block_count": len(((view.get("basic") or {}).get("blocks") or [])),
    }


def _evidence_timeline(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {"event_count": view.get("event_count", 0), "lane_count": len(view.get("lanes") or [])}


def _evidence_roofline(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {"chart_count": len(view.get("charts") or []), "point_count": len(view.get("points") or [])}


def _evidence_cache(view: Mapping[str, Any]) -> Dict[str, Any]:
    # `blocks` is shrunk when families carry the cells; report the largest
    # family's cell count so the evidence metadata keeps reflecting the
    # number of per-block cache entries.
    block_count = len(view.get("blocks") or []) or max(
        (len(family.get("cells") or []) for family in view.get("families") or []), default=0
    )
    return {"block_count": block_count, "metric_count": len(view.get("metrics") or []),
        "family_count": len(view.get(("f"
        "amilies")) or [])}


def _evidence_onchip_memory(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {"allocation_count": len(view.get("blocks") or []),
        "bank_count": ((view.get("bank_layout") or {}).get("bank_count", 0)), ("s"
        "tatus"): view.get("status")}


def _evidence_source(view: Mapping[str, Any]) -> Dict[str, Any]:
    summary = view.get("summary") or {}
    return {"file_count": len(view.get("files") or []),
        "instruction_count": summary.get("instruction_count", 0), ("s"
        "tall_sampled_instructions"): summary.get("instructions_with_stall_samples", 0)}


def _evidence_warp_stall(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {"category_count": len(view.get("categories") or []), "pc_count": len(view.get("pcs") or []), ("total"
        "_samples"): view.get("total_samples", 0)}


def _evidence_raw_data(view: Mapping[str, Any]) -> Dict[str, Any]:
    return {"table_count": len(view.get("tables") or [])}


_PAGE_EVIDENCE_BUILDERS = {
    "details": _evidence_details,
    "timeline": _evidence_timeline,
    "instruction-timeline": _evidence_timeline,
    "roofline": _evidence_roofline,
    "cache": _evidence_cache,
    "onchip-memory": _evidence_onchip_memory,
    "source": _evidence_source,
    "warp-stall": _evidence_warp_stall,
    "raw-data": _evidence_raw_data,
}


def page_evidence_summary(page: str, view: Mapping[str, Any]) -> Dict[str, Any]:
    builder = _PAGE_EVIDENCE_BUILDERS.get(page)
    return builder(view) if builder else {}


def _details_models(
    bundles: Mapping[str, SourceBundle],
    selected_detail_sections: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    basic_model = resolve_basic(bundles)
    occupancy_model = resolve_occupancy(bundles) if "occupancy" in selected_detail_sections else None
    compute_model = (
        resolve_compute(bundles)
        if ("compute" in selected_detail_sections or "memory" in selected_detail_sections)
        else {"available": False, "blocks": [], "block_ids": [], "architecture": {"kind": "unknown"}}
    )
    memory_model = resolve_memory(bundles, compute_model) if "memory" in selected_detail_sections else None

    details_payload: Dict[str, Any] = {"sections": selected_detail_sections,
        "basic": basic_model, "show_basic": not (len(selected_detail_sections) == 1)}
    if occupancy_model is not None:
        details_payload["occupancy"] = occupancy_model
    if "compute" in selected_detail_sections:
        details_payload["compute"] = compute_model
    if memory_model is not None:
        details_payload["memory"] = memory_model
    if "advice" in selected_detail_sections:
        details_payload["advice"] = resolve_advice(bundles)
    return details_payload, compute_model, memory_model


def _shared_source_model(
    collection: NormalizedCollection,
    requested_pages: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], bytes]:
    source_model: Optional[Dict[str, Any]] = None
    warp_stall_frames: Optional[List[Dict[str, Any]]] = None
    warp_stall_bytes: bytes = b""
    if "source" in requested_pages or "warp-stall" in requested_pages:
        # The warp_stall visualize_data.bin can be tens of MB and feeds up to
        # three consumers (Source fusion, standalone stall explorer, legacy
        # stall records); read it once and share the bytes/frames between them.
        warp_stall_path = collection.blocks["warp_stall"].visualize_data
        try:
            warp_stall_bytes = warp_stall_path.read_bytes() if warp_stall_path and warp_stall_path.is_file() else b""
        except OSError as exc:
            logger.debug("Failed to read warp_stall artifact %s: %s", warp_stall_path, exc)
            warp_stall_bytes = b""
        warp_stall_frames = parse_native_visualize_frames(None, data=warp_stall_bytes)
        source_model = parse_source_artifact(
            collection.blocks["source"].visualize_data,
            supplemental_frames=warp_stall_frames,
        )
        if not source_model.get("available"):
            source_records = load_visualize_records(collection.blocks["source"].visualize_data)
            source_model = normalize_source(source_records)
        if not source_model.get("available"):
            source_model = normalize_source(load_visualize_records(collection.blocks["timeline_detail"].visualize_data))
    return source_model, warp_stall_frames, warp_stall_bytes


def _source_page_view(collection: NormalizedCollection, source_model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source_view = source_model or {"available": False}
    if not source_view.get("available"):
        source_view = attach_source_diagnostic(collection, source_view)
    source_block = (collection.manifest.get("blocks") or {}).get("source", {})
    collector_warnings = list(source_block.get("diagnostic_warnings") or [])
    if collector_warnings:
        source_view = dict(source_view)
        source_view["collector_diagnostics"] = {
            "warnings": collector_warnings,
            "stderr_log": ((source_block.get("logs") or {}).get("stderr")),
            "non_fatal": source_block.get("status") in {"ok", "reused", "partial", "aliased"},
        }
    return source_view


def _warp_stall_view(
    source_model: Optional[Dict[str, Any]],
    warp_stall_frames: Optional[List[Dict[str, Any]]],
    warp_stall_bytes: bytes,
) -> Dict[str, Any]:
    stall_model = normalize_stall_from_source(source_model or {})
    if not stall_model.get("available"):
        standalone_source = parse_source_artifact(None, primary_frames=warp_stall_frames)
        stall_model = normalize_stall_from_source(standalone_source)
    if not stall_model.get("available"):
        stall_model = normalize_stall(extract_embedded_json_records(warp_stall_bytes))
    return stall_model


def _instruction_timeline_view(collection: NormalizedCollection, max_events: int) -> Dict[str, Any]:
    instruction_model = normalize_instruction_timeline(load_visualize_records(collection.blocks[("instruction_ti"
        "meline")].visualize_data), max_events=max_events)
    if not instruction_model.get("available"):
        instruction_model = normalize_instruction_timeline(load_visualize_records(collection.blocks[("timeline_d"
            "etail")].visualize_data), max_events=max_events)
    return instruction_model


_PAGE_TO_BLOCK = {
    "details": "details",
    "roofline": "roofline",
    "timeline": "timeline",
    "cache": "details",
    "onchip-memory": "onchip_memory",
    "source": "source",
    "warp-stall": "warp_stall",
    "instruction-timeline": "instruction_timeline",
    "raw-data": "raw_data",
}


def _module_status_entry(page: str, view: Mapping[str, Any], manifest_blocks: Mapping[str, Any]) -> Dict[str, Any]:
    available = page_view_available(page, view)
    block_id = _PAGE_TO_BLOCK.get(page)
    block_present = bool(block_id and block_id in manifest_blocks)
    block = manifest_blocks.get(block_id, {}) if block_id else {}
    collector_status = block.get("status", "not-collected" if block_id else "unknown")
    reason = block.get("reason")
    if not available and not reason:
        if page == "onchip-memory" and not block_present:
            reason = ("The captured collection did not declare an On-Chip Memory block and contains no "
                "manifest-linked memory_info.json sidecar.")
        elif not block_present:
            reason = "The captured collection did not declare this module."
        else:
            reason = "No semantically usable payload was found for this module."
    return {
        "page": page,
        "available": available,
        "collector_block": block_id,
        "collector_status": collector_status,
        "metric": block.get("metric"),
        "reason": reason,
        "evidence": page_evidence_summary(page, view),
    }


def _finalize_pages(
    requested_pages: Sequence[str],
    views: Dict[str, Any],
    manifest_blocks: Mapping[str, Any],
    unavailable_policy: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    module_status: List[Dict[str, Any]] = []
    rendered_pages: List[str] = []
    omitted_modules: List[Dict[str, Any]] = []
    for page in requested_pages:
        view = views.get(page) or {"available": False}
        status = _module_status_entry(page, view, manifest_blocks)
        module_status.append(status)
        if status["available"] or unavailable_policy == "explain":
            if not status["available"]:
                diagnostic = dict(view.get("diagnostic") or {})
                diagnostic.setdefault("title", f"{page.replace('-', ' ').title()} unavailable")
                diagnostic.setdefault("explanation", status["reason"])
                diagnostic.setdefault("evidence", [f"collector_status={status['collector_status']}", (f"metric="
                    f"{status['metric'] or 'not selected'}")])
                view = dict(view)
                view["diagnostic"] = diagnostic
                views[page] = view
            rendered_pages.append(page)
        else:
            omitted_modules.append(status)
    return module_status, rendered_pages, omitted_modules


def _basic_page_views(
    collection: NormalizedCollection,
    bundles: Mapping[str, SourceBundle],
    requested_pages: Sequence[str],
    max_events: int,
    memory_info_path: Optional[Path],
) -> Dict[str, Any]:
    views: Dict[str, Any] = {}
    if "roofline" in requested_pages:
        views["roofline"] = parse_roofline(bundles["roofline"].records)
    if "timeline" in requested_pages:
        views["timeline"] = parse_timeline(collection.blocks["timeline"], max_events=max_events)
    if "cache" in requested_pages:
        views["cache"] = resolve_cache(bundles)
    if "onchip-memory" in requested_pages:
        views["onchip-memory"] = parse_onchip_memory(resolve_onchip_memory_path(collection, memory_info_path))
    return views


def _extended_page_views(
    collection: NormalizedCollection,
    bundles: Mapping[str, SourceBundle],
    requested_pages: Sequence[str],
    views: Dict[str, Any],
    options: "PayloadOptions",
) -> None:
    source_model, warp_stall_frames, warp_stall_bytes = _shared_source_model(collection, requested_pages)
    if "source" in requested_pages:
        views["source"] = _source_page_view(collection, source_model)
    if "warp-stall" in requested_pages:
        views["warp-stall"] = _warp_stall_view(source_model, warp_stall_frames, warp_stall_bytes)
    if "instruction-timeline" in requested_pages:
        views["instruction-timeline"] = _instruction_timeline_view(collection, options.max_events)
    if "raw-data" in requested_pages:
        views["raw-data"] = parse_raw_data(
            collection.blocks["raw_data"],
            max_rows=options.max_raw_rows,
            csv_rows=bundles["raw_data"].csv_rows,
        )


class PayloadOptions(NamedTuple):
    max_events: int = 10000
    max_raw_rows: int = 5000
    unavailable_policy: str = "omit"
    memory_info_path: Optional[Path] = None


def build_payload(
    collection: NormalizedCollection,
    pages: Sequence[str],
    detail_sections: Mapping[str, Sequence[str]],
    options: Optional[PayloadOptions] = None,
) -> Dict[str, Any]:
    if options is None:
        options = PayloadOptions()
    if options.unavailable_policy not in {"omit", "explain"}:
        raise VisualizationError("unavailable_policy must be 'omit' or 'explain'.")

    bundles = build_source_bundles(collection)
    requested_pages = list(pages)
    selected_detail_sections = list(detail_sections.get("details", []))
    details_payload, compute_model, memory_model = _details_models(bundles, selected_detail_sections)

    views: Dict[str, Any] = {}
    if "details" in requested_pages:
        views["details"] = details_payload
    views.update(_basic_page_views(collection, bundles, requested_pages, options.max_events, options.memory_info_path))
    _extended_page_views(collection, bundles, requested_pages, views, options)

    manifest_blocks = collection.manifest.get("blocks") or {}
    module_status, rendered_pages, omitted_modules = _finalize_pages(
        requested_pages, views, manifest_blocks, options.unavailable_policy)

    if not rendered_pages:
        rendered_pages = ["availability"]
        views["availability"] = {
            "available": True,
            "requested_modules": module_status,
            "message": "None of the requested modules produced semantically usable visualization data.",
        }

    architecture = (memory_model or {}).get("architecture") or compute_model.get("architecture") or {"kind": "unknown"}
    return {
        "schema": REPORT_SCHEMA,
        "renderer_version": VERSION,
        "input_schema": collection.schema,
        "architecture": architecture,
        "requested_pages": requested_pages,
        "pages": rendered_pages,
        "module_status": module_status,
        "omitted_modules": omitted_modules,
        "views": views,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Generate a universal standalone msOpProf performance "
        "report from a collection manifest."))
    parser.add_argument("--input", help="Collection root containing collection_manifest.json.")
    parser.add_argument("--output", help="Report output directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="complete")
    parser.add_argument("--feature", action="append", default=[], help="Repeatable targeted feature.")
    parser.add_argument("--list-features", action="store_true")
    parser.add_argument("--explain", metavar="FEATURE")
    parser.add_argument("--max-trace-events", type=int, default=10000)
    parser.add_argument("--max-raw-rows", type=int, default=5000)
    parser.add_argument("--payload-only", action="store_true", help=("Write report_payload.json but skip HTML "
        "rendering."))
    parser.add_argument("--report-name", default="report.html", help="Output HTML filename inside --output.")
    parser.add_argument("--memory-info", help="Optional explicit memory_info.json for the On-Chip Memory module.")
    parser.add_argument("--unavailable-policy", choices=["auto", "omit", "explain"], default="auto", help=("Omit "
        "unavailable modules in preset reports; explain them in explicit targeted reports by default."))
    parser.add_argument("--compact-payload", action=argparse.BooleanOptionalAction, default=True, help=("Omit "
        "duplicated top-level Source relations after exporting the full relation CSV."))
    parser.add_argument("--pretty-payload", action="store_true", help=("Pretty-print report_payload.json. "
        "Default is compact JSON for faster writes and smaller artifacts."))
    parser.add_argument("--compress-payload", choices=["on", "off"], default="off", help=("Embed the HTML "
        "payload as base64 gzip inflated via DecompressionStream (smaller HTML; requires a modern browser). "
        "Default off: plain JSON embed."))
    return parser


class _ReportPaths(NamedTuple):
    root: Path
    report: Path
    payload: Path


def _prepare_output_paths(args: argparse.Namespace) -> _ReportPaths:
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_name = Path(args.report_name).name
    if report_name != args.report_name or not report_name.lower().endswith(".html"):
        raise VisualizationError("--report-name must be a plain .html filename without directory components.")
    return _ReportPaths(output_root, output_root / report_name, output_root / "report_payload.json")


def _export_debug_tables(
    payload: Dict[str, Any],
    output_root: Path,
    phase_seconds: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    phase_started = time.perf_counter()
    edge_tables = export_memory_edge_tables(payload, output_root)
    source_tables = export_source_tables(payload, output_root)
    phase_seconds["export_debug_tables"] = time.perf_counter() - phase_started
    return edge_tables, source_tables


def _serialize_payload(
    payload: Dict[str, Any],
    payload_path: Path,
    pretty: bool,
    phase_seconds: Dict[str, float],
) -> str:
    phase_started = time.perf_counter()
    if pretty:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_path.write_text(payload_text, encoding="utf-8")
    phase_seconds["write_payload"] = time.perf_counter() - phase_started
    return payload_text


def _render_html(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    report_path: Path,
    payload_text: str,
    phase_seconds: Dict[str, float],
) -> None:
    if args.payload_only:
        return
    phase_started = time.perf_counter()
    template = Path(__file__).resolve().parent.parent / "templates" / "report_template.html"
    # Reuse the already-serialized payload text; render_report applies
    # the `</` escaping (or gzip compression) instead of dumping twice.
    render_report(
        payload,
        template,
        report_path,
        payload_text=payload_text,
        compress_payload=args.compress_payload == "on",
    )
    phase_seconds["render_html"] = time.perf_counter() - phase_started


def _write_timing(
    paths: _ReportPaths,
    wall_started: float,
    phase_seconds: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    timing_summary = {
        "schema": "msopprof-visualization-timing/v1",
        "renderer_version": VERSION,
        "wall_seconds": round(time.perf_counter() - wall_started, 6),
        "phases": [
            {"phase": name, "elapsed_seconds": round(seconds, 6)}
            for name, seconds in phase_seconds.items()
        ],
        "compact_payload": bool(args.compact_payload),
        "pretty_payload": bool(args.pretty_payload),
        "compress_payload": args.compress_payload,
        "payload_bytes": paths.payload.stat().st_size if paths.payload.exists() else 0,
        "html_bytes": paths.report.stat().st_size if paths.report.exists() else 0,
    }
    timing_dir = paths.root / "_internal"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / "visualization_timing.json"
    timing_path.write_text(json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return timing_summary


def _write_report_index(
    paths: _ReportPaths,
    args: argparse.Namespace,
    payload: Dict[str, Any],
    unavailable_policy: str,
) -> None:
    report_index = {
        "html": None if args.payload_only else paths.report.name,
        "payload": paths.payload.name,
        "requested_pages": payload.get("requested_pages", []),
        "rendered_pages": payload.get("pages", []),
        "omitted_modules": payload.get("omitted_modules", []),
        "unavailable_policy": unavailable_policy,
        "timing": "_internal/visualization_timing.json",
        "compact_payload": bool(args.compact_payload),
    }
    (paths.root / "report_index.json").write_text(json.dumps(report_index,
        ensure_ascii=False, indent=2), encoding=("u"
        "tf-8"))


def _generate_report(args: argparse.Namespace) -> int:
    wall_started = time.perf_counter()
    phase_seconds: Dict[str, float] = {}
    pages, detail_sections = resolve_requested_features(args.preset, args.feature)

    phase_started = time.perf_counter()
    collection = load_collection(Path(args.input))
    phase_seconds["load_collection"] = time.perf_counter() - phase_started

    unavailable_policy = args.unavailable_policy
    if unavailable_policy == "auto":
        unavailable_policy = "explain" if args.feature else "omit"
    phase_started = time.perf_counter()
    payload = build_payload(
        collection=collection,
        pages=pages,
        detail_sections=detail_sections,
        options=PayloadOptions(
            max_events=max(1, args.max_trace_events),
            max_raw_rows=max(1, args.max_raw_rows),
            unavailable_policy=unavailable_policy,
            memory_info_path=Path(args.memory_info) if args.memory_info else None,
        ),
    )
    phase_seconds["parse_and_build_payload"] = time.perf_counter() - phase_started

    paths = _prepare_output_paths(args)
    output_root, report_path, payload_path = paths
    edge_tables, source_tables = _export_debug_tables(payload, output_root, phase_seconds)

    if args.compact_payload:
        phase_started = time.perf_counter()
        compact_payload_for_delivery(payload)
        phase_seconds["compact_payload"] = time.perf_counter() - phase_started

    payload_text = _serialize_payload(payload, payload_path, args.pretty_payload, phase_seconds)
    _render_html(args, payload, report_path, payload_text, phase_seconds)
    timing_summary = _write_timing(paths, wall_started, phase_seconds, args)
    _write_report_index(paths, args, payload, unavailable_policy)
    cli_logger.info(json.dumps({
        "report": None if args.payload_only else str(report_path),
        "payload": str(payload_path),
        "report_index": str(output_root / "report_index.json"),
        "requested_pages": pages,
        "rendered_pages": payload.get("pages", []),
        "omitted_modules": payload.get("omitted_modules", []),
        "input_schema": collection.schema,
        "memory_edge_tables": edge_tables,
        "source_debug_tables": source_tables,
        "timing": timing_summary,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.list_features:
            print_feature_list()
            return 0
        if args.explain:
            print_feature_explanation(args.explain)
            return 0
        if not args.input or not args.output:
            parser.error("--input and --output are required unless --list-features or --explain is used.")
        return _generate_report(args)
    except VisualizationError as exc:
        logger.error("ERROR: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
