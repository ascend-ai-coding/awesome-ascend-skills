# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import json
from typing import Any, Iterable, Optional


def _format_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(rows: Iterable[Iterable[Any]], headers: Iterable[Any]) -> str:
    header_cells = [_format_markdown_cell(cell) for cell in headers]
    body_rows = [[_format_markdown_cell(cell) for cell in row] for row in rows]
    column_count = max(
        [len(header_cells)] + [len(row) for row in body_rows],
        default=len(header_cells),
    )

    if len(header_cells) < column_count:
        header_cells.extend([""] * (column_count - len(header_cells)))

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    for row in body_rows:
        if len(row) < column_count:
            row = row + [""] * (column_count - len(row))
        lines.append("| " + " | ".join(row[:column_count]) + " |")
    return "\n".join(lines)


class _JsonScanState:
    def __init__(self) -> None:
        self.brace_count = 0
        self.in_string = False
        self.escape_next = False

    def consume(self, byte: bytes) -> bool:
        if self.escape_next:
            self.escape_next = False
            return False
        if self.in_string and byte == b"\\":
            self.escape_next = True
            return False
        if byte == b'"':
            self.in_string = not self.in_string
            return False
        if self.in_string:
            return False
        if byte == b"{":
            self.brace_count += 1
        elif byte == b"}":
            self.brace_count -= 1
        return self.brace_count == 0


def _marker_positions(data: bytes, marker: bytes) -> list[int]:
    positions: list[int] = []
    start = 0
    while start < len(data) - len(marker) + 1:
        found = data.find(b"Z", start)
        if found < 0:
            break
        if data[found:found + len(marker)] == marker:
            positions.append(found)
        start = found + 1
    return positions


class BinaryJsonExtractor:
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.marker = b'ZZ{'

    @staticmethod
    def _extract_json_bytes(data: bytes, start_pos: int) -> Optional[bytes]:
        """
        Extract JSON bytes starting from a position where '{' is expected.

        Args:
            data: Binary data
            start_pos: Starting position (should point to '{')

        Returns:
            JSON bytes (including the opening '{') or None if invalid
        """
        # Verify we're starting with '{'
        if start_pos >= len(data) or data[start_pos:start_pos + 1] != b'{':
            return None

        state = _JsonScanState()
        for index in range(start_pos, len(data)):
            result = state.consume(data[index:index + 1])
            if result:
                return data[start_pos:index + 1]
        return None

    def extract_json_blocks(self, filename: str) -> list[dict[str, Any]]:
        """Extract all JSON blocks found after ``b'ZZ{'`` markers."""
        json_blocks: list[dict[str, Any]] = []
        with open(filename, "rb") as file:
            data = file.read()
        for position in self._find_markers(data):
            json_bytes = self._extract_json_bytes(data, position + 2)
            if json_bytes is None:
                continue
            try:
                json_blocks.append(json.loads(json_bytes.decode(self.encoding)))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"Warning: Failed to parse JSON at position {position}: {exc}")
                try:
                    json_blocks.append(json.loads(json_bytes.decode(self.encoding, errors="replace")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        return json_blocks

    def extract_and_output(self, filename: str) -> None:
        """Extract JSON blocks and write each parsed object to a sibling file."""
        results = self.extract_with_positions(filename)
        for index, result in enumerate(results, 1):
            output_path = f"{filename[:-4]}part_{index}.json"
            with open(output_path, "w", encoding="utf-8") as output:
                output.write(json.dumps(result["json"], indent=4))

    def extract_with_positions(self, filename: str) -> list[dict[str, Any]]:
        """Extract JSON blocks together with their positions in the binary file."""
        results: list[dict[str, Any]] = []
        with open(filename, "rb") as file:
            data = file.read()
        for marker_pos in self._find_markers(data):
            json_start = marker_pos + 2
            json_bytes = self._extract_json_bytes(data, json_start)
            if json_bytes is None:
                continue
            try:
                json_obj = json.loads(json_bytes.decode(self.encoding))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            results.append(
                {
                    "json": json_obj,
                    "marker_position": marker_pos,
                    "json_start": json_start,
                    "json_end": json_start + len(json_bytes),
                    "total_size": len(json_bytes) + 2,
                    "json_bytes": json_bytes,
                }
            )
        return results

    def _find_markers(self, data: bytes) -> list[int]:
        """Find all positions where ``b'ZZ{'`` occurs in binary data."""
        return _marker_positions(data, self.marker)


class BaseInfo:
    def __init__(self, **kwargs: object):
        name = str(kwargs["name"])
        duration = float(kwargs["duration"])
        op_type = str(kwargs["op_type"])
        block_dim = int(kwargs["block_dim"])
        head_name = kwargs["head_name"]
        block_detail = kwargs["block_detail"]
        self.name = name
        self.duration = duration
        self.op_type = op_type
        self.block_dim = block_dim
        self.head_name = tuple(head_name)
        self.block_detail = tuple(block_detail)

    def print_info(self):
        res = ""
        res += f"**Name:** {self.name}\n\n"
        res += f"**Duration:** {self.duration}\n\n"
        res += f"**Op Type:** {self.op_type}\n\n"
        res += f"**Block Dim:** {self.block_dim}\n\n"
        if self.op_type == "vector":
            res += "#### Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        elif self.op_type == "mix":
            res += "#### Mix Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        elif self.op_type == "cube":
            res += "#### Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        else:
            raise AssertionError(f"op_type = {self.op_type}")

        return res


class WorkloadAnalysis:
    def __init__(self, pipe_utilization: Iterable[tuple[str]], details: dict[int, list[tuple[str]]]):
        self.pipe_utilization = tuple(pipe_utilization)
        self.details = details

    def print_info(self, *, block_id: int):
        res = ""
        res += "#### Pipe utilization (at most 20 printed)\n\n"
        headers = ["Block ID", "Name", "Utilization (%)"]
        res += _format_markdown_table(self.pipe_utilization[:20], headers) + "\n\n"
        if block_id in self.details:
            res += f"#### Details for block {block_id} (at most 20 printed)\n\n"
            headers = ["VECTOR", "Instructions", "Duration (us)", "Data volume (byte)"]
            res += _format_markdown_table(self.details[block_id][:20], headers) + "\n"
        return res


data_path_description = {
    0: "GM -> L2 Cache",
    1: "L2 Cache -> GM",
    2: "L2 Cache -> L1",
    3: "L1 -> L2 Cache",
    4: "L1 -> L0A",
    5: "L1 -> L0B",
    6: "L0A -> Cube",
    7: "L0B -> Cube",
    8: "Cube -> L0C",
    9: "L0C -> Cube",
    10: "L0C -> L2 Cache",
    11: "L0C -> L1",
    12: "L2 Cache -> UB (Vector0)",
    13: "UB -> L2 Cache (Vector0)",
    14: "UB -> Vector (Vector0)",
    15: "Vector -> UB (Vector 0)",
    16: "L2 Cache -> UB (Vector1)",
    17: "UB -> L2 Cache (Vector1)",
    18: "UB -> Vector (Vector1)",
    19: "Vector -> UB (Vector1)",
}


class CoreMemoryMap:
    def __init__(self, **kwargs: object):
        advice = kwargs["advice"]
        data_paths = kwargs["data_paths"]
        l2_cache = kwargs.get("l2_cache")
        vector = kwargs.get("vector")
        vector1 = kwargs.get("vector1")
        cube = kwargs.get("cube")
        self.advice = tuple(advice)
        self.data_paths = tuple(data_paths)
        self.l2_cache = l2_cache
        self.vector = vector
        self.vector1 = vector1
        self.cube = cube

    def print_info(self):
        res = ""
        if self.advice:
            res += "#### Advice\n\n"
            for advice in self.advice:
                res += advice + "\n\n"
        if self.l2_cache:
            res += f"**L2 Cache Hit Rate:** {float(self.l2_cache['hit_ratio']):.2f}%\n\n"
        if self.vector:
            res += f"**Vector Ratio:** {float(self.vector['ratio']):.2f}%\n\n"
        if self.vector1:
            res += f"**Vector1 Ratio:** {float(self.vector1['ratio']):.2f}%\n\n"
        if self.cube:
            res += f"**Cube Ratio:** {float(self.cube['ratio']):.2f}%\n\n"
        res += "#### Data paths\n\n"
        headers = ["Path", "Bandwidth (GB/s)", "Request"]
        res += _format_markdown_table(self.data_paths, headers) + "\n\n"
        return res


class MemoryWorkloadTable:
    def __init__(self, advice: Iterable[str], table_detail: Iterable[dict]):
        self.advice = tuple(advice)
        self.table_detail = tuple(table_detail)

    def print_info(self):
        res = ""
        if self.advice:
            res += "#### Advice\n\n"
            for advice in self.advice:
                res += advice + "\n\n"
        for table in self.table_detail:
            headers = table['header_name']
            headers[0] = table['table_name']
            data = []
            for row in table['row']:
                data.append([row['name']] + row['value'])
            res += f"#### {table['table_name']}\n\n"
            res += _format_markdown_table(data, headers) + "\n\n"
        return res


class MemoryWorkloadAnalysis:
    def __init__(self, core_memory_maps: dict[int, CoreMemoryMap],
                 workload_tables: dict[int, MemoryWorkloadTable]):
        self.core_memory_maps = core_memory_maps
        self.workload_tables = workload_tables

    def print_info(self, *, block_id: int):
        res = ""
        if block_id in self.core_memory_maps:
            res += f"### Core memory map for block {block_id}\n\n"
            res += self.core_memory_maps[block_id].print_info()
        if block_id in self.workload_tables:
            res += f"### Memory workload table for block {block_id}\n\n"
            res += self.workload_tables[block_id].print_info()
        return res


class AllInfo:
    def __init__(self, base_info: BaseInfo, workload_analysis: WorkloadAnalysis,
                 memory_workload_analysis: MemoryWorkloadAnalysis):
        self.base_info = base_info
        self.workload_analysis = workload_analysis
        self.memory_workload_analysis = memory_workload_analysis

    def print_info(self, *, block_id: int):
        res = ""
        res += "## Base Info\n\n"
        res += self.base_info.print_info() + "\n"
        res += "## Compute Workload Analysis\n\n"
        res += self.workload_analysis.print_info(block_id=block_id) + "\n"
        res += "## Memory Workload Analysis\n\n"
        res += self.memory_workload_analysis.print_info(block_id=block_id)
        return res


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _block_key(value: Any) -> str:
    return str(value)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = [_result_json_block(results, index) for index in range(5)]
    return {
        "base_info": _summarize_base_info(blocks[0]),
        "pipe_utilization": _summarize_pipe_utilization(blocks[1]),
        "instruction_wait_signals": _summarize_instruction_waits(blocks[2]),
        "memory_path_signals": _summarize_memory_paths(blocks[3]),
        "memory_load_signals": _summarize_memory_load(blocks[4]),
    }


def _result_json_block(results: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index >= len(results):
        return {}
    block = results[index].get("json", {})
    return block if isinstance(block, dict) else {}


def _summarize_base_info(block: dict[str, Any]) -> dict[str, Any]:
    detail = block.get("mix_block_detail") if block.get("op_type") == "mix" else block.get("block_detail")
    head_name, preview = _block_detail_preview(detail)
    return {
        "name": block.get("name"),
        "duration": _safe_float(block.get("duration")),
        "op_type": block.get("op_type"),
        "block_dim": int(block.get("block_dim") or 0),
        "head_name": head_name,
        "block_detail_preview": preview,
    }


def _block_detail_preview(detail: Any) -> tuple[list[str], list[list[str]]]:
    if not isinstance(detail, dict):
        return [], []
    head_name = [str(item) for item in detail.get("head_name", [])]
    preview: list[list[str]] = []
    for row in detail.get("row", [])[:20]:
        if isinstance(row, dict) and isinstance(row.get("value"), list):
            preview.append([str(item) for item in row["value"]])
    return head_name, preview


def _summarize_pipe_utilization(block: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in block.get("subblock_detail", []):
        if isinstance(item, dict):
            entries.append({
                "block_id": _block_key(item.get("block_id", "")),
                "name": str(item.get("name", "")),
                "utilization_percent": _safe_float(item.get("value")),
            })
    top_pipe = max(entries, key=_utilization_sort_key) if entries else None
    return {"entries": entries, "top_pipe": top_pipe}


def _utilization_sort_key(item: dict[str, Any]) -> float:
    value = item.get("utilization_percent")
    return value if isinstance(value, float) else float("-inf")


def _summarize_instruction_waits(block: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    vector_waits: dict[str, float] = {}
    data_sizes: dict[str, dict[str, float]] = {}
    instruction_counts: dict[str, dict[str, float]] = {}
    signals = {
        "vector_waits": vector_waits,
        "data_sizes": data_sizes,
        "instruction_counts": instruction_counts,
    }
    for item in block.get("subblock_detail", []):
        if not isinstance(item, dict):
            continue
        block_id = _block_key(item.get("block_id", ""))
        name, value = str(item.get("name", "")), _safe_float(item.get("value"))
        entries.append({"block_id": block_id, "name": name, "value": value})
        _record_instruction_signal(name, value, block_id, signals)
    return {
        "entries": entries,
        "vector_wait_by_block": vector_waits,
        "data_size_by_block": data_sizes,
        "instruction_counts_by_block": instruction_counts,
    }


def _record_instruction_signal(
    name: str,
    value: float | None,
    block_id: str,
    signals: dict[str, Any],
) -> None:
    if value is None:
        return
    if name == "Vector Wait":
        signals["vector_waits"][block_id] = value
    elif name in ("Vector Compute Data Size", "Cube Compute Data Size"):
        signals["data_sizes"].setdefault(block_id, {})[name] = value
    else:
        signals["instruction_counts"].setdefault(block_id, {})[name] = value


def _summarize_memory_paths(block: dict[str, Any]) -> dict[str, Any]:
    summary = _empty_memory_path_summary()
    all_paths: list[dict[str, Any]] = []
    for item in block.get("core_memory_map", []):
        if isinstance(item, dict):
            _record_memory_core(summary, all_paths, item)
    summary["top_bandwidth_paths"] = sorted(all_paths, key=_bandwidth_sort_key, reverse=True)[:5]
    return summary


def _empty_memory_path_summary() -> dict[str, Any]:
    return {
        "advice_by_core": {}, "l2_hit_ratio_by_core": {}, "vector_ratio_by_core": {},
        "vector1_ratio_by_core": {}, "cube_ratio_by_core": {}, "top_bandwidth_paths": [],
        "paths_by_core": {},
    }


def _record_memory_core(summary: dict[str, Any], all_paths: list[dict[str, Any]], item: dict[str, Any]) -> None:
    core_key = _block_key(item.get("core_no", ""))
    summary["advice_by_core"][core_key] = [str(entry) for entry in item.get("advice", [])]
    _record_memory_ratios(summary, core_key, item)
    core_paths = _memory_paths_for_core(core_key, item)
    summary["paths_by_core"][core_key] = core_paths
    all_paths.extend(core_paths)


def _record_memory_ratios(
    summary: dict[str, Any],
    core_key: str,
    item: dict[str, Any],
) -> None:
    l2_cache = item.get("L2cache")
    if isinstance(l2_cache, dict):
        hit_ratio = _safe_float(l2_cache.get("hit_ratio"))
        if hit_ratio is not None:
            summary["l2_hit_ratio_by_core"][core_key] = hit_ratio
    ratio_targets = (
        ("Vector", "vector_ratio_by_core"),
        ("Vector1", "vector1_ratio_by_core"),
        ("Cube", "cube_ratio_by_core"),
    )
    for source, target in ratio_targets:
        value = item.get(source)
        if isinstance(value, dict):
            ratio = _safe_float(value.get("ratio"))
            if ratio is not None:
                summary[target][core_key] = ratio


def _memory_paths_for_core(core_key: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    paths = []
    for unit_item in item.get("memory_unit", []):
        if isinstance(unit_item, dict):
            memory_path = unit_item.get("memory_path")
            paths.append({
                "core_no": core_key, "memory_path": memory_path,
                "path_name": data_path_description.get(memory_path, str(memory_path)),
                "bandwidth_gb_s": _safe_float(unit_item.get("bandwidth")),
                "request": _safe_float(unit_item.get("request")),
            })
    return paths


def _bandwidth_sort_key(item: dict[str, Any]) -> float:
    value = item.get("bandwidth_gb_s")
    return value if isinstance(value, float) else float("-inf")


def _summarize_memory_load(block: dict[str, Any]) -> dict[str, Any]:
    advice_by_block: dict[str, list[str]] = {}
    tables_by_block: dict[str, list[dict[str, Any]]] = {}
    for item in block.get("table_per_block", []):
        if isinstance(item, dict):
            block_id = _block_key(item.get("block_id", ""))
            advice_by_block[block_id] = [str(entry) for entry in item.get("advice", [])]
            tables_by_block[block_id] = _normalize_workload_tables(item.get("table_detail", []))
    return {"advice_by_block": advice_by_block, "tables_by_block": tables_by_block}


def _normalize_workload_tables(tables: Any) -> list[dict[str, Any]]:
    normalized = []
    for table in tables:
        if isinstance(table, dict):
            normalized.append({
                "table_name": str(table.get("table_name", "")),
                "header_name": [str(entry) for entry in table.get("header_name", [])],
                "row": _normalize_workload_rows(table.get("row", [])),
            })
    return normalized


def _normalize_workload_rows(rows: Any) -> list[dict[str, Any]]:
    return [
        {"name": str(row.get("name", "")), "value": [str(entry) for entry in row.get("value", [])]}
        for row in rows if isinstance(row, dict)
    ]


def summarize_file(filename: str) -> dict[str, Any]:
    extractor = BinaryJsonExtractor()
    results = extractor.extract_with_positions(filename)
    return summarize_results(results)


def get_info(results: list[dict[str, Any]]) -> AllInfo:
    base_info = _build_base_info(results[0]["json"])
    workload = _build_workload_analysis(results[1]["json"], results[2]["json"])
    memory = _build_memory_workload_analysis(results[3]["json"], results[4]["json"])
    return AllInfo(base_info, workload, memory)


def _build_base_info(result: dict[str, Any]) -> BaseInfo:
    op_type = result["op_type"]
    detail = result["mix_block_detail"] if op_type == "mix" else result["block_detail"]
    if op_type not in {"vector", "mix", "cube"}:
        raise AssertionError(f"op_type = {op_type}")
    head_name = detail["head_name"]
    block_detail = [tuple(block["value"]) for block in detail["row"]]
    return BaseInfo(
        name=result["name"], duration=result["duration"], op_type=op_type,
        block_dim=int(result["block_dim"] or 0), head_name=head_name, block_detail=block_detail,
    )


def _build_workload_analysis(
    pipe_result: dict[str, Any],
    detail_result: dict[str, Any],
) -> WorkloadAnalysis:
    pipe_utilization = [
        (item["block_id"], item["name"], item["value"])
        for item in pipe_result["subblock_detail"]
    ]
    details: dict[int, list[tuple[str, str, Any, str]]] = {}
    for item in detail_result["subblock_detail"]:
        block_id, name, value = int(item["block_id"]), item["name"], item["value"]
        details.setdefault(block_id, []).append(_workload_detail_row(name, value))
    return WorkloadAnalysis(pipe_utilization=pipe_utilization, details=details)


def _workload_detail_row(name: str, value: Any) -> tuple[str, str, Any, str]:
    if name == "Vector Wait":
        return name, "", value, ""
    if name in ("Vector Compute Data Size", "Cube Compute Data Size"):
        return name, "", "", value
    return name, value, "", ""


def _build_memory_workload_analysis(
    memory_result: dict[str, Any],
    table_result: dict[str, Any],
) -> MemoryWorkloadAnalysis:
    core_maps = _build_core_memory_maps(memory_result["core_memory_map"])
    workload_tables = {
        int(item["block_id"]): MemoryWorkloadTable(
            advice=item["advice"], table_detail=item["table_detail"]
        )
        for item in table_result["table_per_block"]
    }
    return MemoryWorkloadAnalysis(core_memory_maps=core_maps, workload_tables=workload_tables)


def _build_core_memory_maps(items: list[dict[str, Any]]) -> dict[int, CoreMemoryMap]:
    core_maps: dict[int, CoreMemoryMap] = {}
    for item in items:
        data_paths = [
            (data_path_description[path["memory_path"]], path["bandwidth"], path["request"])
            for path in item["memory_unit"]
        ]
        core_maps[int(item["core_no"])] = CoreMemoryMap(
            advice=item["advice"], data_paths=data_paths, l2_cache=item.get("L2cache"),
            vector=item.get("Vector"), vector1=item.get("Vector1"), cube=item.get("Cube"),
        )
    return core_maps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse MindStudio Profiler binary output")
    parser.add_argument("filename", help="Path to the binary profiler output file")
    parser.add_argument("--block-id", type=int, default=0, help="Block ID to analyze (default: 0)")
    args = parser.parse_args()

    extractor = BinaryJsonExtractor()
    results = extractor.extract_with_positions(args.filename)
    info = get_info(results)
    print(info.print_info(block_id=args.block_id))
