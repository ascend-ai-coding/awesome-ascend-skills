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

import base64
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from visualize_common import VisualizationError

PAYLOAD_TAG_PLAIN = '<script id="payload" type="application/json">'


PAYLOAD_TAG_COMPRESSED = '<script id="payload" type="application/octet-stream" data-encoding="gzip-base64">'


def render_report(
    payload: Dict[str, Any],
    template_path: Path,
    output_path: Path,
    payload_text: Optional[str] = None,
    compress_payload: bool = False,
) -> None:
    html = template_path.read_text(encoding="utf-8")
    marker = "__MSOPPROF_PAYLOAD__"
    if marker not in html:
        raise VisualizationError(f"Template missing marker: {marker}")
    if payload_text is None:
        payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compress_payload:
        # Embed the payload as base64(gzip(json)); the template inflates it
        # with DecompressionStream. Base64 needs no `</` escaping.
        encoded = base64.b64encode(gzip.compress(payload_text.encode("utf-8"))).decode("ascii")
        if PAYLOAD_TAG_PLAIN not in html:
            raise VisualizationError("Template missing payload script tag for compression swap.")
        html = html.replace(PAYLOAD_TAG_PLAIN, PAYLOAD_TAG_COMPRESSED)
        output_path.write_text(html.replace(marker, encoded), encoding="utf-8")
        return
    # Escape `</` once so an inlined `</script>` in data cannot break the
    # document (XSS guard); the JSON stays byte-identical after JS parsing.
    safe = payload_text.replace("</", "<\\/")
    output_path.write_text(html.replace(marker, safe), encoding="utf-8")


def _memory_edge_rows(block: Mapping[str, Any]) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    occurrences = block.get("occurrences") or [block]
    rows: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for occurrence in occurrences:
        topology = occurrence.get("topology") or {}
        # `edge_table` is the pre-3.4 alias of `edges`; read both.
        edges = topology.get("edge_table") or topology.get("edges") or []
        for edge in edges:
            rows.append((occurrence, edge))
    return rows


def _edge_id_of(edge: Mapping[str, Any]) -> Optional[str]:
    # New payloads omit edge_id; it is always f"{src}->{dst}".
    return edge.get("edge_id") or "{}->{}".format(
        edge.get("source") or edge.get("src"), edge.get("target") or edge.get("dst")
    )


def _edge_csv_row(block_id: str, occurrence: Mapping[str, Any], edge: Mapping[str, Any]) -> Dict[str, Any]:
    binding = (
        occurrence.get("execution_binding")
        or (occurrence.get("topology") or {}).get("execution_binding") or {}
    )
    return {
        "block_id": block_id,
        "physical_core_id": binding.get("physical_core_id"),
        "dispatch_index": binding.get("dispatch_index"),
        "dispatch_count": binding.get("dispatch_count"),
        "binding": binding.get("binding"),
        "source_ordinal": binding.get("source_ordinal"),
        "edge_id": _edge_id_of(edge),
        "memory_path": edge.get("memory_path"),
        "source": edge.get("source") or edge.get("src"),
        "target": edge.get("target") or edge.get("dst"),
        "bandwidth_gb_s": edge.get("bandwidth"),
        "requests": edge.get("request"),
        "peak_ratio": edge.get("peak_ratio"),
        "source_kind": edge.get("source_kind"),
    }


_EDGE_FIELD_ROW_KEYS = (
    "block_id", "physical_core_id", "dispatch_index", "dispatch_count", "binding",
    "source_ordinal", "source", "target", "edge_id", "memory_path",
)


def _write_memory_edge_csvs(
    debug_root: Path,
    block_id: str,
    rows: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> List[str]:
    combined = debug_root / f"block_{block_id}_edges.csv"
    with combined.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "block_id", "physical_core_id", "dispatch_index", "dispatch_count", "binding", "source_ordinal",
            "edge_id", "memory_path", "source", "target", "bandwidth_gb_s",
            "requests", "peak_ratio", "source_kind",
        ])
        writer.writeheader()
        for occurrence, edge in rows:
            writer.writerow(_edge_csv_row(block_id, occurrence, edge))
    written = [str(combined)]
    for field, suffix in (("bandwidth", "bandwidth"), ("request", "requests")):
        out = debug_root / f"block_{block_id}_{suffix}.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "block_id", "physical_core_id", "dispatch_index", "dispatch_count", "binding", "source_ordinal",
                "source", "target", "value", "edge_id", "memory_path",
            ])
            writer.writeheader()
            for occurrence, edge in rows:
                base = _edge_csv_row(block_id, occurrence, edge)
                row = {key: base.get(key) for key in _EDGE_FIELD_ROW_KEYS}
                row["value"] = edge.get(field)
                writer.writerow(row)
        written.append(str(out))
    return written


def export_memory_edge_tables(payload: Mapping[str, Any], output_root: Path) -> List[str]:
    """Export canonical directed edge tables without overwriting repeated dispatches.

    One CSV set is written per logical Block ID. When a block has multiple
    source-ordered occurrences, all dispatches are rows in the same files and
    are distinguished by binding metadata.
    """
    memory = (((payload.get("views") or {}).get("details") or {}).get("memory") or {})
    blocks = memory.get("blocks") or []
    if not blocks:
        return []
    debug_root = output_root / "_debug" / "memory_edges"
    debug_root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for block in blocks:
        block_id = str(block.get("block_id", ""))
        rows = _memory_edge_rows(block)
        if not rows:
            continue
        written.extend(_write_memory_edge_csvs(debug_root, block_id, rows))
    return written


def _export_source_files_csv(debug_root: Path, source: Mapping[str, Any]) -> str:
    files_path = debug_root / "source_files.csv"
    with files_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_id", "path", "kind", "is_user_source", "line_count",
            "mapped_line_count", "related_instruction_count", "snapshot_count", "content_hash"])
        writer.writeheader()
        for file_obj in source.get("files") or []:
            writer.writerow({key: file_obj.get(key) for key in writer.fieldnames})
    return str(files_path)


def _source_line_csv_row(file_obj: Mapping[str, Any], line: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "file_id": file_obj.get("id"), "path": file_obj.get("path"),
            "line": line.get("number"), "text": line.get("text"), "mapped": line.get("mapped"),
        "instructions_executed_all": (line.get("instructions_executed_by_core") or {}).get("ALL",
            0), "gpr_count": line.get("gpr_count"),
        "address_ranges": json.dumps(line.get("address_ranges") or [],
            ensure_ascii=False), "related_instruction_count": len(line.get(("r"
            "elated_instruction_ids")) or []),
        "stall_all_percent": line.get("stall_all_percent",
            (line.get("stall_all") or {}).get("percent", 0.0)), ("s"
            "tall_not_issue_percent"): line.get("stall_not_issue_percent",
                (line.get("stall_not_issue") or {}).get("percent", 0.0)),
        "stall_all_details": json.dumps((line.get("stall_all") or {}).get("details") or {},
            ensure_ascii=False),
        "stall_not_issue_details": json.dumps((line.get("stall_not_issue") or {}).get("details") or {},
            ensure_ascii=False),
    }


def _export_source_lines_csv(debug_root: Path, source: Mapping[str, Any]) -> str:
    lines_path = debug_root / "source_lines.csv"
    with lines_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_id", "path", "line", "text", "mapped", ("instructions_executed"
            "_all"), "gpr_count", "address_ranges", "related_instruction_count", "stall_all_percent", ("stall_not_is"
                "sue_percent"), "stall_all_details", "stall_not_issue_details"])
        writer.writeheader()
        for file_obj in source.get("files") or []:
            for line in file_obj.get("lines") or []:
                writer.writerow(_source_line_csv_row(file_obj, line))
    return str(lines_path)


def _instruction_csv_row(inst: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "instruction_id": inst.get("id"), "index": inst.get("index"), "address": inst.get("address"),
            "pipe": inst.get("pipe"), "opcode": inst.get("opcode"), ("s"
            "ource_location"): inst.get("source_location"),
        "instructions_executed_all": (inst.get("instructions_executed_by_core") or {}).get("ALL",
            0), "gpr_count": inst.get("gpr_count"),
        "gpr_status": json.dumps(inst.get("gpr_status") or [], ensure_ascii=False),
        "stall_all_percent": inst.get("stall_all_percent", (inst.get("stall_all") or {}).get("percent",
            0.0)), "stall_not_issue_percent": inst.get("stall_not_issue_percent",
                (inst.get("stall_not_issue") or {}).get("percent", 0.0)),
        "stall_all_details": json.dumps((inst.get("stall_all") or {}).get("details") or {}, ensure_ascii=False),
        "stall_not_issue_details": json.dumps((inst.get("stall_not_issue") or {}).get("details") or {},
            ensure_ascii=False),
    }


def _export_instructions_csv(debug_root: Path, source: Mapping[str, Any]) -> str:
    instructions_path = debug_root / "instructions.csv"
    with instructions_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["instruction_id", "index", "address", "pipe", "opcode", ("source_loc"
            "ation"), "instructions_executed_all", "gpr_count", "gpr_status", "stall_all_percent",
                "stall_not_issue_percent", "stall_all_details", "stall_not_issue_details"])
        writer.writeheader()
        for inst in source.get("instructions") or []:
            writer.writerow(_instruction_csv_row(inst))
    return str(instructions_path)


def _export_relations_csv(debug_root: Path, source: Mapping[str, Any]) -> str:
    relations_path = debug_root / "source_instruction_relations.csv"
    with relations_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["instruction_id", "file_id", "source_path", "line", "reasons"])
        writer.writeheader()
        for relation in source.get("relations") or []:
            writer.writerow({**relation, "reasons": ",".join(relation.get("reasons") or [])})
    return str(relations_path)


def export_source_tables(payload: Mapping[str, Any], output_root: Path) -> List[str]:
    source = ((payload.get("views") or {}).get("source") or {})
    if source.get("schema") not in {"msopprof-source-explorer/v1",
        "msopprof-source-explorer/v2"} or not source.get("available"):
        return []
    debug_root = output_root / "_debug" / "source"
    debug_root.mkdir(parents=True, exist_ok=True)
    return [
        _export_source_files_csv(debug_root, source),
        _export_source_lines_csv(debug_root, source),
        _export_instructions_csv(debug_root, source),
        _export_relations_csv(debug_root, source),
    ]


def compact_payload_for_delivery(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicated relation material that the standalone UI does not read.

    The Source page already carries both directions needed by the UI:
    `files[*].lines[*].related_instruction_ids` and
    `instructions[*].related_lines`. The top-level `relations` array is a third
    copy used only for CSV export, so it is exported first and then removed from
    the delivered JSON/HTML payload.
    """
    source = ((payload.get("views") or {}).get("source") or {})
    relations = source.get("relations") or []
    if relations:
        source["relations"] = []
        source["relation_storage"] = {
            "mode": "inline_bidirectional",
            "relation_count": len(relations),
            "full_export": "_debug/source/source_instruction_relations.csv",
        }
        contract = source.get("contract")
        if isinstance(contract, dict):
            contract["payload_compaction"] = (
                ("top-level relations omitted from report payload; equivalent bindings remain on source lines "
                    "and instructions")
            )
    payload["delivery"] = {
        "compact": True,
        "source_relations_omitted": len(relations),
    }
    return payload