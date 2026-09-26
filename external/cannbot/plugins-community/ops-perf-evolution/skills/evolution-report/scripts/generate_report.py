#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""Evolution report generator for ops-evo and lingxi-evo pipelines.

Parses output directory files and renders a standardized HTML report.
Supports both ops-evo and lingxi-evo pipeline structures.
"""

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from string import Template
from typing import Any

# Session 统计模块（session/timing/token 统计 + 资源消耗区块），从本文件拆分
from report_session_stats import (
    _get_dir_birthtime,
    get_session_time_range,
    parse_session_stats,
    parse_round_timing,
    build_resource_stats_section,
)

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：报告输出路径等 CLI 结果走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


def parse_output_dir_name(output_dir: str) -> tuple[str, str, str]:
    """Extract op_name, pipeline_type and timestamp from output directory name.

    Returns:
        (op_name, pipeline_type, timestamp)
        pipeline_type is one of: "ops-evo", "lingxi-evo", "unknown"
    """
    dirname = os.path.basename(output_dir.rstrip("/"))
    # Match ops-evo format: {op}_ops-evo_{timestamp}
    m = re.match(r"(.+?)_ops-evo_(\d{8}_\d{6})$", dirname)
    if m:
        return m.group(1), "ops-evo", m.group(2)
    # Match lingxi-evo format: {op}_evo_{timestamp} or {op}_lingxi-evo_{timestamp}
    m = re.match(r"(.+?)_lingxi-evo_(\d{8}_\d{6})$", dirname)
    if m:
        return m.group(1), "lingxi-evo", m.group(2)
    m = re.match(r"(.+?)_evo_(\d{8}_\d{6})$", dirname)
    if m:
        return m.group(1), "lingxi-evo", m.group(2)
    # Fallback: just extract timestamp
    m2 = re.match(r"(.+?)_(\d{8}_\d{6})$", dirname)
    if m2:
        return m2.group(1), "unknown", m2.group(2)
    tz = timezone(timedelta(hours=8))
    return dirname, "unknown", datetime.now(tz).strftime("%Y%m%d_%H%M%S")


def get_model_from_config() -> str | None:
    """Read model name from Claude Code settings.json in CLAUDE_CONFIG_DIR.

    Priority:
    1. $CLAUDE_CONFIG_DIR/settings.json -> env.ANTHROPIC_MODEL
    2. ~/.claude/settings.json -> env.ANTHROPIC_MODEL
    """
    config_paths = []
    claude_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if claude_config_dir:
        config_paths.append(os.path.join(claude_config_dir, "settings.json"))
    home = os.path.expanduser("~")
    config_paths.append(os.path.join(home, ".claude", "settings.json"))

    for path in config_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            model = settings.get("env", {}).get("ANTHROPIC_MODEL")
            if model:
                return model
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return None


def load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _normalize_lingxi_format(eval_data: dict) -> dict:
    """归一化 lingxi 格式（有 comparison + evolved，无 baseline）。"""
    comp = eval_data.get("comparison", {}) or {}
    ev = eval_data.get("evolved", {}) or {}
    return {
        "op_name": eval_data.get("op_name", ""),
        "baseline": {},
        "evolved": {
            "tag": ev.get("tag", "evolved"),
            "install_path": ev.get("install_path", ""),
            "precision_passed": comp.get("precision_passed", False),
            "correctness_message": ev.get("correctness_message", ""),
            "time_us": ev.get("time_us", -1),
            "pipeline": ev.get("pipeline", {}),
            "bottleneck": ev.get("bottleneck", "unknown"),
            "cv_pct": ev.get("cv_pct", 0.0),
        },
        "comparison": {
            "compilation_success": comp.get("compilation_success", False),
            "precision_passed": comp.get("precision_passed", False),
            "speedup": comp.get("speedup", 0.0),
            "time_delta_us": comp.get("time_delta_us", 0.0),
            "cv_pct": comp.get("cv_pct", 0.0),
            "measurement_quality": comp.get("measurement_quality", "unknown"),
        },
    }


def _normalize_flat_format(eval_data: dict) -> dict:
    """归一化 flat 格式（单变体 eval 输出），包装为完整嵌套结构。"""
    time_us = eval_data.get("time_us", -1)
    precision_passed = eval_data.get("precision_passed", False)
    # If precision passed and we have a positive time, compilation implicitly succeeded
    compilation_success = eval_data.get("compilation_success")
    if compilation_success is None:
        compilation_success = precision_passed and time_us is not None and time_us > 0

    return {
        "op_name": eval_data.get("op_name", ""),
        "baseline": {},
        "evolved": {
            "tag": eval_data.get("tag", "evolved"),
            "install_path": eval_data.get("install_path", ""),
            "precision_passed": precision_passed,
            "correctness_message": eval_data.get("correctness_message", ""),
            "time_us": time_us,
            "pipeline": eval_data.get("pipeline", {}),
            "bottleneck": eval_data.get("bottleneck", "unknown"),
            "cv_pct": eval_data.get("cv_pct", 0.0),
        },
        "comparison": {
            "compilation_success": compilation_success,
            "precision_passed": precision_passed,
            "speedup": 0.0,
            "time_delta_us": 0.0,
            "cv_pct": eval_data.get("cv_pct", 0.0),
            "measurement_quality": "unknown",
        },
    }


def normalize_eval_json(eval_data: dict | None) -> dict | None:
    """Normalize evaluation_results.json to the full nested format.

    Handles two input formats:
    1. Full format (baseline eval output): {baseline: {...}, evolved: {...}, comparison: {...}}
    2. Flat format (single-variant eval output): {tag, time_us, precision_passed, ...}

    Flat format is produced when ops-partial subagents save only the evolved
    result from run_single_version(). We normalize it to full format so all
    downstream consumers work uniformly.
    """
    if not eval_data:
        return None

    # Already full format
    if "baseline" in eval_data and "evolved" in eval_data and "comparison" in eval_data:
        return eval_data

    # Lingxi format: has `comparison` and `evolved` but no `baseline`
    # （lingxi-partial 的输出结构：comparison 字段含编译/精度/加速比，
    #  evolved 字段含耗时与 pipeline）
    if "comparison" in eval_data and "evolved" in eval_data and "baseline" not in eval_data:
        return _normalize_lingxi_format(eval_data)

    # Flat format: wrap in full structure
    return _normalize_flat_format(eval_data)


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""




































































def _parse_test_cases_csv(csv_path: str) -> dict:
    """Parse the first data row of test_cases.csv into a dict of param->value."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return {}
        headers = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        return dict(zip(headers, values))
    except Exception:
        return {}


def _walk_source_files(base_dir: Path) -> list:
    """遍历目录收集符合条件的源文件相对路径。"""
    collected = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), base_dir)
            if not _should_include_modified_file(rel):
                continue
            collected.append(rel)
    return collected


def _collect_modified_files(pd: Path) -> tuple:
    """收集单个变体的修改文件列表（ops-evo: modified_files/；lingxi-evo: kernel/）。

    Returns (modified_files, modified_files_dir)。
    """
    modified_files_dir = pd / "modified_files"
    if not modified_files_dir.is_dir():
        modified_files_dir = pd / "kernel"
    modified_files = []
    if modified_files_dir.is_dir():
        modified_files = _walk_source_files(modified_files_dir)
    return modified_files, modified_files_dir


def _collect_one_variant(rd: Path, pd: Path) -> dict:
    """收集单个 round/parallel 变体的评估结果与修改文件。"""
    round_num = int(rd.name.split("_")[1])
    par_num = int(pd.name.split("_")[1])
    eval_json = load_json(str(pd / "evaluation_results.json"))
    if not eval_json:
        eval_json = load_json(str(pd / "eval.json"))
    eval_json = normalize_eval_json(eval_json)
    impl_note = load_text(str(pd / "implementation_note.txt"))
    modified_files, modified_files_dir = _collect_modified_files(pd)
    return {
        "round": round_num,
        "parallel": par_num,
        "eval": eval_json,
        "impl_note": impl_note,
        "modified_files": modified_files,
        "modified_files_dir": str(modified_files_dir),
        "path": str(pd),
    }


def collect_rounds(output_dir: str, pipeline_type: str = "ops-evo") -> list[dict]:
    """Collect all round/parallel evaluation results.

    Args:
        output_dir: Evolution output directory
        pipeline_type: "ops-evo" or "lingxi-evo", affects baseline source detection
    """
    results = []
    round_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for rd in round_dirs:
        parallel_dirs = sorted(
            [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("parallel_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        for pd in parallel_dirs:
            results.append(_collect_one_variant(rd, pd))
    return results


def enrich_rounds_from_world_model(rounds: list[dict], wm: dict | None, baseline_time: float) -> None:
    """Supplement rounds missing evaluation_results.json with world model node data.

    When eval JSON is absent but the world model has a node with a matching
    solution_ref and a positive score, synthesize eval data so the variant
    is not silently dropped.
    """
    if not wm or baseline_time <= 0:
        return
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    ref_map: dict[str, dict] = {}
    for node in nodes.values():
        ref = node.get("solution_ref")
        if ref:
            ref_map[ref] = node

    for r in rounds:
        if r.get("eval"):
            continue
        ref = f"round_{r['round']}/parallel_{r['parallel']}"
        node = ref_map.get(ref)
        if not node:
            continue
        score = node.get("score")
        if not score or score <= 0:
            continue
        # Synthesize evaluation data from world model score
        evolved_time = baseline_time / score
        r["eval"] = {
            "compilation_success": True,
            "precision_passed": True,
            "evolved": {"time_us": round(evolved_time, 2)},
            "_source": "world_model",
        }


def _eval_compile_and_precision(ev: dict) -> tuple[bool, bool]:
    """从 eval dict 提取 compilation_success / precision_passed（兼容嵌套 comparison）。"""
    compilation_ok = ev.get("compilation_success")
    precision_ok = ev.get("precision_passed")
    if compilation_ok is None and "comparison" in ev:
        compilation_ok = ev.get("comparison", {}).get("compilation_success")
        precision_ok = ev.get("comparison", {}).get("precision_passed")
    return bool(compilation_ok), bool(precision_ok)


def find_best_variant(rounds: list[dict], baseline_time: float) -> dict | None:
    """Find the variant with the highest speedup."""
    best = None
    for r in rounds:
        ev = r.get("eval")
        if not ev:
            continue
        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok, precision_ok = _eval_compile_and_precision(ev)
        if not compilation_ok or not precision_ok:
            continue
        evolved_time = ev.get("evolved", {}).get("time_us")
        if evolved_time and evolved_time > 0:
            speedup = baseline_time / evolved_time
            if best is None or speedup > best["speedup"]:
                best = {**r, "speedup": speedup, "time_us": evolved_time}
    return best


def get_node_for_variant(wm: dict, round_num: int, par_num: int) -> dict | None:
    """Find the world model node matching a round/parallel variant."""
    if not wm:
        return None
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    ref = f"round_{round_num}/parallel_{par_num}"
    for node in nodes.values():
        if node.get("solution_ref") == ref:
            return node
    return None


def _build_parent_lookup(nodes: dict) -> dict:
    """构建 child_id → parent_id 回溯表。"""
    parent_of = {}
    for nid, node in nodes.items():
        for child_id in node.get("children", []):
            parent_of[child_id] = nid
    return parent_of


def _trace_back_to_root(start_nid: str, parent_of: dict) -> list:
    """从指定节点沿 parent_of 回溯到根，返回节点 id 路径（不含 "root" 前缀）。"""
    path = [start_nid]
    current = start_nid
    while current in parent_of:
        current = parent_of.get(current)
        path.insert(0, current)
    return path


def _find_path_by_best_variant(nodes: dict, best_variant: dict,
                                parent_of: dict) -> tuple | None:
    """Strategy 1: 找匹配实际最优变体的节点并回溯路径。"""
    target_ref = f"round_{best_variant['round']}/parallel_{best_variant['parallel']}"
    target_node = None
    for nid, node in nodes.items():
        if node.get("solution_ref") == target_ref:
            target_node = nid
            break
    if not target_node:
        return None
    path = _trace_back_to_root(target_node, parent_of)
    endpoint_score = nodes.get(target_node, {}).get("score", 0)
    if endpoint_score is None:
        endpoint_score = 0.0
    return ["root"] + path, endpoint_score


def _find_path_by_highest_score(nodes: dict, parent_of: dict) -> tuple:
    """Strategy 2: 找 score 最高的节点并回溯路径。"""
    best_node_id = None
    best_node_score = -1.0
    for nid, node in nodes.items():
        score = node.get("score")
        if score is not None and score > best_node_score:
            best_node_score = score
            best_node_id = nid

    if best_node_id and best_node_score > 0:
        path = _trace_back_to_root(best_node_id, parent_of)
        return ["root"] + path, best_node_score
    return ["root"], 1.0


def find_best_path(wm: dict, best_variant: dict | None = None, baseline_time: float = 0) -> tuple[list[str], float]:
    """Trace the best path from root to the best leaf.

    Priority:
    1. Path to the node matching the actual best variant (if found in WM)
    2. Path to the highest-score node in the world model

    Returns (path, endpoint_score)
    """
    if not wm:
        return [], 0.0

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    parent_of = _build_parent_lookup(nodes)

    # Strategy 1: Try to find path to the node matching actual best variant
    if best_variant:
        result = _find_path_by_best_variant(nodes, best_variant, parent_of)
        if result is not None:
            return result

    # Strategy 2: Find the node with the highest score
    return _find_path_by_highest_score(nodes, parent_of)


def _extract_shape_groups(call_spec: dict) -> list[dict]:
    """Normalize call_spec into a list of shape groups.

    Supports two formats:
    - Legacy single-shape: top-level ``inputs`` / ``scalar_args`` / ``tensor_kwargs``.
    - Multi-shape: ``target_shapes`` / ``generalization_shapes`` arrays, each item
      carrying its own ``inputs`` / ``scalar_args`` / ``tensor_kwargs``.

    Each returned group has: ``{role, name, inputs, scalar_args, tensor_kwargs}``.
    """
    groups: list[dict] = []
    targets = call_spec.get("target_shapes") or []
    gens = call_spec.get("generalization_shapes") or []
    if targets or gens:
        for i, sh in enumerate(targets):
            groups.append({
                "role": "target",
                "name": sh.get("name") or f"T{i+1}",
                "inputs": sh.get("inputs", []),
                "scalar_args": sh.get("scalar_args", {}),
                "tensor_kwargs": sh.get("tensor_kwargs", {}),
            })
        for i, sh in enumerate(gens):
            groups.append({
                "role": "generalization",
                "name": sh.get("name") or f"G{i+1}",
                "inputs": sh.get("inputs", []),
                "scalar_args": sh.get("scalar_args", {}),
                "tensor_kwargs": sh.get("tensor_kwargs", {}),
            })
        return groups

    # Legacy single-shape format
    if call_spec.get("inputs") or call_spec.get("scalar_args") or call_spec.get("tensor_kwargs"):
        groups.append({
            "role": "single",
            "name": "",
            "inputs": call_spec.get("inputs", []),
            "scalar_args": call_spec.get("scalar_args", {}),
            "tensor_kwargs": call_spec.get("tensor_kwargs", {}),
        })
    return groups


def _render_group_rows(group: dict, with_header: bool) -> list[str]:
    """Render rows for one shape group: optional colspan=4 header + 2-col params + tensor remark."""
    rows: list[str] = []
    if with_header:
        role_tag = {"target": "Target", "generalization": "泛化", "single": ""}.get(group["role"], "")
        label_parts = [p for p in (role_tag, group["name"]) if p]
        label = " · ".join(label_parts) if label_parts else "参数"
        rows.append(
            f'<tr><td colspan="4" style="background:var(--bg-subtle);font-weight:600;">'
            f'Shape: {_escape_html(label)}</td></tr>'
        )

    structured: dict[str, str] = {}
    for inp in group.get("inputs", []) or []:
        name = inp.get("name", "")
        shape = inp.get("shape", [])
        dtype = inp.get("dtype", "unknown")
        structured[name] = f"shape={shape}, dtype={dtype}"
    for k, v in (group.get("scalar_args") or {}).items():
        structured[k] = str(v)

    tensor_kwargs_info: list[str] = []
    for k, v in (group.get("tensor_kwargs") or {}).items():
        if isinstance(v, dict):
            tensor_kwargs_info.append(
                f"{k}: shape={v.get('shape', [])}, dtype={v.get('dtype', 'unknown')}"
            )
        else:
            tensor_kwargs_info.append(f"{k}: {v}")

    items = list(structured.items())
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            rows.append(f"<tr><td>{k1}</td><td>{v1}</td><td>{k2}</td><td>{v2}</td></tr>")
        else:
            rows.append(f"<tr><td>{k1}</td><td>{v1}</td><td></td><td></td></tr>")

    if tensor_kwargs_info:
        remark = "; ".join(tensor_kwargs_info)
        rows.append(
            f'<tr><td colspan="4" style="color:var(--text-muted);font-size:0.8rem;">'
            f'<strong>辅助张量:</strong> {_escape_html(remark)}</td></tr>'
        )
    return rows


def build_test_case_rows(test_case: dict, output_dir: str = "") -> str:
    """Build HTML table rows for test case parameters.

    Multi-shape call_spec → emit one section per shape group with a header row.
    Legacy single-shape call_spec → emit a flat 2-col table (no header).
    """
    call_spec = None
    if output_dir:
        call_spec = load_json(os.path.join(output_dir, "shared", "call_spec.json"))

    groups: list[dict] = []
    if call_spec:
        groups = _extract_shape_groups(call_spec)

    # Fallback: synthesize one group from the test_case dict if call_spec missing/empty
    if not groups and test_case:
        # Treat each kv as a scalar arg so they still show up
        groups = [{
            "role": "single",
            "name": "",
            "inputs": [],
            "scalar_args": {k: str(v) for k, v in test_case.items()},
            "tensor_kwargs": {},
        }]

    if not groups:
        return '<tr><td colspan="4">参数信息不可用</td></tr>'

    multi = len(groups) > 1 or (groups and groups[0]["role"] != "single")
    all_rows: list[str] = []
    for g in groups:
        all_rows.extend(_render_group_rows(g, with_header=multi))

    if not all_rows:
        return '<tr><td colspan="4">参数信息不可用</td></tr>'
    return "\n".join(all_rows)


def build_hardware_rows(hw: dict, eval_info: dict) -> str:
    """Build HTML table rows for hardware info."""
    chip = hw.get("chip_model", "Unknown")
    cube = hw.get("core_num_cube") or hw.get("core_num", "?")
    # Fallback: for 910B series, cube and vector core counts are identical
    vector = hw.get("core_num_vector") or hw.get("vector_core_num")
    if vector is None:
        vector = cube
    rows = [
        f"<tr><td>芯片</td><td>Ascend {chip}</td></tr>",
        f"<tr><td>CubeCore / VectorCore</td><td>{cube} / {vector}</td></tr>",
    ]
    # Add a note when vector core count was inferred from cube core count
    if hw.get("core_num_vector") is None and hw.get("vector_core_num") is None:
        rows.append(
            f'<tr><td colspan="2" style="color:var(--text-muted);font-size:0.8rem;">'
            f'注：VectorCore 数量未单独提供，按 Ascend {chip} 架构推断与 CubeCore 相同（每 AI Core 含 1 Cube + 1 Vector）'
            f"</td></tr>"
        )
    if hw.get("peak_bw_gbps"):
        rows.append(f"<tr><td>峰值带宽</td><td>{hw['peak_bw_gbps']} GB/s</td></tr>")
    backend = eval_info.get("eval_backend", "forge")
    profiling_method = eval_info.get("profiling_method", "msprof")
    # Normalize: strip trailing " profiling" or "op profiling" suffixes
    for suffix in (" op profiling", " profiling"):
        if profiling_method.endswith(suffix):
            profiling_method = profiling_method[:-len(suffix)]
    rows.append(f"<tr><td>评估后端</td><td>{backend}</td></tr>")
    rows.append(f"<tr><td>性能采集</td><td>{profiling_method}</td></tr>")
    return "\n".join(rows)


def _resolve_row_node_info(node: dict | None, r: dict) -> tuple:
    """解析表格行的节点信息（node_id / strategy / desc）。

    世界模型节点缺失时标记为"分层采样"（启发式采样）。
    """
    node_id = node.get("id", "N/A") if node else "N/A"
    strategy = "+".join(node.get("strategy_combination", [])) if node else "—"
    if node and node.get("mode") == "open_exploration":
        strategy = "open"
    if not node:
        strategy = "分层采样"
    desc_raw = node.get("description", "") if node else r.get("impl_note", "")[:60]
    desc = desc_raw[:40] if len(desc_raw) > 40 else desc_raw
    return node_id, strategy, desc


@dataclass
class _RowContext:
    """表格行构建的公共上下文。"""
    vid: str
    node_id: str
    strategy: str
    desc: str


def _build_invalid_row(ctx: _RowContext, reason: str, evolved_time=None) -> str:
    """构建无效变体（编译失败 / 精度失败）的表格行。"""
    head = (f'<tr class="invalid"><td><span class="vid">{ctx.vid}</span></td>'
            f'<td><span class="tag tag-node">{ctx.node_id}</span></td>'
            f'<td><span class="sid">{ctx.strategy}</span></td>'
            f"<td>{ctx.desc}</td>")
    if evolved_time is None:
        return head + f'<td colspan="3"><span class="badge badge-skip">无效 — {reason}</span></td></tr>'
    return head + (f'<td>{evolved_time:.2f}</td><td>—</td>'
                   f'<td><span class="badge badge-fail">{reason}</span></td></tr>')


def _build_valid_row(ctx: _RowContext, evolved_time: float,
                     baseline_time: float, is_best: bool) -> str:
    """构建有效变体的表格行（最优 / 有效 / 退化徽标）。"""
    speedup = baseline_time / evolved_time if evolved_time > 0 else 0
    if is_best:
        tr_class = ' class="best"'
        badge = '<span class="badge badge-best">最优</span>'
    elif speedup >= 1.0:
        tr_class = ""
        badge = '<span class="badge badge-pass">有效</span>'
    else:
        tr_class = ""
        badge = '<span class="badge badge-fail">退化</span>'

    return (
        f"<tr{tr_class}><td><span class=\"vid\">{ctx.vid}</span></td>"
        f'<td><span class="tag tag-node">{ctx.node_id}</span></td>'
        f'<td><span class="sid">{ctx.strategy}</span></td>'
        f"<td>{ctx.desc}</td>"
        f"<td>{evolved_time:.2f}</td><td>{speedup:.3f}x</td>"
        f"<td>{badge}</td></tr>"
    )


def build_evolution_table_rows(rounds: list[dict], wm: dict, baseline_time: float, best_variant: dict | None) -> str:
    """Build HTML table rows for the full evolution trajectory."""
    rows = []
    best_round = best_variant["round"] if best_variant else -1
    best_par = best_variant["parallel"] if best_variant else -1

    for r in rounds:
        rn, pn = r["round"], r["parallel"]
        vid = f"{rn}-V{pn}"
        node = get_node_for_variant(wm, rn, pn)
        node_id, strategy, desc = _resolve_row_node_info(node, r)

        ev = r.get("eval")
        is_best = rn == best_round and pn == best_par

        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok, precision_ok = _eval_compile_and_precision(ev) if ev else (None, None)

        row_ctx = _RowContext(vid=vid, node_id=node_id, strategy=strategy, desc=desc)
        if not ev or not compilation_ok:
            reason = node["failure_reason"][:30] if node and node.get("failure_reason") else ""
            rows.append(_build_invalid_row(row_ctx, reason))
            continue

        evolved_time = ev.get("evolved", {}).get("time_us", 0)
        if evolved_time is None:
            evolved_time = 0
        if not precision_ok:
            rows.append(_build_invalid_row(row_ctx, "精度失败", evolved_time))
            continue

        rows.append(_build_valid_row(row_ctx, evolved_time, baseline_time, is_best))
    return "\n".join(rows)


def _collect_chart_data(rounds: list[dict], baseline_time: float) -> tuple:
    """收集折线图数据点（labels, data），过滤无效变体和极端离群值。"""
    labels = []
    data = []
    for r in rounds:
        ev = r.get("eval")
        if not ev:
            continue
        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok, precision_ok = _eval_compile_and_precision(ev)
        if not compilation_ok or not precision_ok:
            continue
        evolved_time = ev.get("evolved", {}).get("time_us")
        if evolved_time and evolved_time > 0:
            # Skip extreme outliers that would distort the chart
            if baseline_time > 0 and evolved_time > baseline_time * 5:
                continue
            labels.append(f"{r['round']}-V{r['parallel']}")
            data.append(round(evolved_time, 2))
    return labels, data


def build_chart_script(rounds: list[dict], baseline_time: float) -> str:
    """Build Chart.js initialization script."""
    labels, data = _collect_chart_data(rounds, baseline_time)

    if not data:
        return "// No valid data for chart"

    best_idx = data.index(min(data))
    labels_json = json.dumps(labels)
    data_json = json.dumps(data)

    return f"""
var evoCtx = document.getElementById('evolutionChart');
if (evoCtx) {{
    new Chart(evoCtx, {{
        type: 'line',
        data: {{
            labels: {labels_json},
            datasets: [{{
                label: '耗时 (us)',
                data: {data_json},
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88,166,255,0.1)',
                fill: true, tension: 0.3, pointRadius: 4,
                pointBackgroundColor: function(ctx) {{ return ctx.dataIndex === {best_idx} ? '#3fb950' : '#58a6ff'; }},
                pointBorderColor: function(ctx) {{ return ctx.dataIndex === {best_idx} ? '#3fb950' : '#58a6ff'; }},
                pointRadius: function(ctx) {{ return ctx.dataIndex === {best_idx} ? 8 : 4; }}
            }}, {{
                label: 'Baseline ({baseline_time} us)',
                data: Array({len(data)}).fill({baseline_time}),
                borderColor: 'rgba(248,81,73,0.5)',
                borderDash: [6, 4], pointRadius: 0, fill: false
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#c9d1d9' }} }},
                tooltip: {{ callbacks: {{ afterLabel: function(ctx) {{
                    if (ctx.datasetIndex === 0) return '加速比: ' + ({baseline_time} / ctx.raw).toFixed(3) + 'x';
                }} }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
                y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }},
                      title: {{ display: true, text: '耗时 (us)', color: '#8b949e' }} }}
            }}
        }}
    }});
}}"""


# Excluded directories and extensions for modified files filtering
_EXCLUDED_MOD_DIRS = {"tests", "docs", "__pycache__", ".git"}
_EXCLUDED_MOD_EXTS = {".bak", ".tmp", ".swp", ".swo", ".orig", ".rej", ".o", ".so", ".a", ".pyc", ".pyo"}


def _should_include_modified_file(rel_path: str) -> bool:
    """Filter out non-core and temporary files from modified_files listing."""
    parts = rel_path.split(os.sep)
    if any(p in _EXCLUDED_MOD_DIRS for p in parts):
        return False
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in _EXCLUDED_MOD_EXTS:
        return False
    basename = os.path.basename(rel_path)
    if basename.startswith(".") or basename.endswith("~"):
        return False
    return True


def _detect_modified_files_prefix(modified_files: list[str], op_name: str) -> str | None:
    """Detect if modified_files have a common op-name prefix directory.

    ops-partial sometimes copies files into modified_files/<op_name>/...
    instead of modified_files/... directly. This detects such prefixes.
    """
    if not modified_files:
        return None
    first_dirs = set()
    for f in modified_files:
        parts = f.split(os.sep)
        if len(parts) > 1:
            first_dirs.add(parts[0])
    if len(first_dirs) == 1:
        prefix = list(first_dirs)[0]
        op_flat = op_name.replace("_", "").lower()
        prefix_flat = prefix.replace("_", "").lower()
        if op_flat in prefix_flat or prefix_flat in op_flat:
            return prefix
    return None


def _strip_prefix(rel_path: str, prefix: str | None) -> str:
    """Strip leading directory prefix from a relative path."""
    if not prefix:
        return rel_path
    prefix_with_sep = prefix + os.sep
    if rel_path.startswith(prefix_with_sep):
        return rel_path[len(prefix_with_sep):]
    return rel_path


def _is_source_file(rel_path: str) -> bool:
    """Check if a file is a source code file that should appear in diff sections."""
    ext = os.path.splitext(rel_path)[1].lower()
    return ext in (".cpp", ".h", ".hpp", ".c", ".py", ".cc", ".cu", ".cuh")


def build_code_diff_sections(best_variant: dict, baseline_source: str | None) -> str:
    """Build HTML code diff sections for the best variant's modified files."""
    if not best_variant:
        return '<div class="card"><p style="color:var(--text-muted);">无最优变体数据</p></div>'

    sections = []
    mod_dir = best_variant.get("modified_files_dir", "")
    prefix = best_variant.get("modified_files_prefix", "")
    copy_id_counter = 0

    for rel_path in best_variant.get("modified_files", []):
        # Skip metadata files that are not actual source code
        if not _is_source_file(rel_path):
            continue
        mod_file = os.path.join(mod_dir, rel_path)
        if not os.path.isfile(mod_file):
            continue

        # Normalize path for baseline comparison (strip op-name prefix if present)
        norm_path = _strip_prefix(rel_path, prefix)

        copy_id_counter += 1
        code_id = f"diff-{copy_id_counter}"

        if not baseline_source:
            # No baseline source — cannot determine if file is new or modified, skip
            copy_id_counter -= 1
            continue

        base_file = os.path.join(baseline_source, norm_path)
        if not os.path.isfile(base_file):
            # No baseline counterpart — this is a NEW file, skip per policy
            # (report should only show diffs of existing files)
            copy_id_counter -= 1
            continue

        with open(base_file, "r", encoding="utf-8", errors="replace") as f:
            base_lines = f.readlines()
        with open(mod_file, "r", encoding="utf-8", errors="replace") as f:
            mod_lines = f.readlines()
        diff = list(difflib.unified_diff(
            base_lines, mod_lines,
            fromfile=f"a/{norm_path}", tofile=f"b/{norm_path}", lineterm=""
        ))
        if not diff:
            # Files are identical — skip entirely
            copy_id_counter -= 1
            continue
        diff_html = _format_diff_html(diff)
        sections.append(
            f'<div class="card"><h3 style="margin-top:0">{norm_path}</h3>'
            f'<div class="copy-wrapper">'
            f'<button class="copy-btn" onclick="copyCode(this,\'{code_id}\')">复制</button>'
            f'<pre><code id="{code_id}">{diff_html}</code></pre>'
            f"</div></div>"
        )

    if not sections:
        return '<div class="card"><p style="color:var(--text-muted);">无代码修改（所有文件与基线一致）</p></div>'
    return "\n".join(sections)


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_diff_html(diff_lines: list[str]) -> str:
    """Format unified diff lines as colored HTML."""
    parts = []
    for line in diff_lines:
        line_escaped = _escape_html(line.rstrip("\n"))
        if line.startswith("@@"):
            parts.append(f'<span class="diff-hunk">{line_escaped}</span>')
        elif line.startswith("+") and not line.startswith("+++"):
            parts.append(f'<span class="diff-add">{line_escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            parts.append(f'<span class="diff-del">{line_escaped}</span>')
        else:
            parts.append(line_escaped)
    return "\n".join(parts)


def _build_ref_to_time(rounds: list[dict] | None) -> dict:
    """Build a lookup from solution_ref -> evolved_time.

    Speedups shown in the tree are computed with the *unified* baseline_time
    instead of the per-variant baseline that world-model scores are based on
    (avoids small mismatches).
    """
    ref_to_time: dict[str, float] = {}
    if not rounds:
        return ref_to_time
    for r in rounds:
        ref = f"round_{r['round']}/parallel_{r['parallel']}"
        ev = r.get("eval")
        if ev:
            t = ev.get("evolved", {}).get("time_us")
            if t and t > 0:
                ref_to_time[ref] = t
    return ref_to_time


def _build_children_map(nodes: dict, root_children: list) -> dict:
    """Build unified parent→children mapping from BOTH sources:

    1. node["children"] arrays
    2. node["parent_id"] fields (reverse mapping)
    This prevents omissions when one source is incomplete.
    """
    children_map: dict[str, list[str]] = {}  # parent_id → [child_ids]
    for nid, n in nodes.items():
        if nid == "root":
            continue
        # Source 2: node's parent_id field
        pid = n.get("parent_id")
        if pid:
            children_map.setdefault(pid, [])
            if nid not in children_map[pid]:
                children_map[pid].append(nid)
    # Merge root_children into the map
    children_map.setdefault("root", [])
    for cid in root_children:
        if cid not in children_map["root"]:
            children_map["root"].append(cid)
    # Also merge each node's own children list
    for nid, n in nodes.items():
        for cid in n.get("children", []):
            children_map.setdefault(nid, [])
            if cid not in children_map[nid]:
                children_map[nid].append(cid)
    return children_map


def _bfs_one_level(current_ids: list, nodes: dict, children_map: dict,
                   visited: set) -> tuple:
    """BFS 展开一层，返回 (本层节点列表, 下层候选 id 列表)。"""
    level = []
    next_ids = []
    for nid in current_ids:
        if nid in visited:
            continue
        visited.add(nid)
        node = nodes.get(nid)
        if not node:
            continue
        level.append((nid, node))
        for child_id in children_map.get(nid, []):
            if child_id not in visited:
                next_ids.append(child_id)
    return level, next_ids


def _bfs_tree_levels(nodes: dict, children_map: dict, baseline_time: float) -> list:

    """BFS from root using unified children; orphan 节点追加为最后一层。"""
    levels = []
    # Level 0: root
    levels.append([("root", {"id": "root", "score": 1.0, "strategy_combination": [],
                              "description": f"Baseline {baseline_time:.2f}us",
                              "children": children_map.get("root", []),
                              "status": "passed", "mode": "root"})])
    current_ids = children_map.get("root", [])
    visited = {"root"}
    while current_ids:
        level, next_ids = _bfs_one_level(current_ids, nodes, children_map, visited)
        if level:
            levels.append(level)
        current_ids = next_ids

    # Append any orphaned nodes (not reachable via BFS) as an extra level.
    # These nodes exist in the tree but have broken parent references.
    orphaned = [(nid, node) for nid, node in nodes.items() if nid not in visited]
    if orphaned:
        levels.append(orphaned)
    return levels


@dataclass
class _TreeRenderContext:
    """树节点渲染的共享上下文。"""
    best_path: list[str]
    best_score: float
    baseline_time: float
    ref_to_time: dict


def _render_one_tree_node(nid: str, node: dict, ctx: _TreeRenderContext,
                          html_parts: list):
    """渲染单个树节点的 HTML。"""
    css_class = _get_node_css_class(nid, node, ctx.best_path, ctx.best_score)
    strategy = "+".join(node.get("strategy_combination", []))
    if node.get("mode") == "open_exploration":
        strategy = "open"
    desc = node.get("description", "")[:25]

    # Speedup: prefer unified-baseline eval time, else world-model score
    ref = node.get("solution_ref")
    if ref and ref in ctx.ref_to_time and ctx.baseline_time > 0:
        score = ctx.baseline_time / ctx.ref_to_time[ref]
    else:
        score = node.get("score")

    if nid == "root":
        speedup_text = f"{ctx.baseline_time:.2f}us"
        speedup_class = ""
    elif score is not None:
        speedup_text = f"{score:.3f}x"
        if score and abs(score - ctx.best_score) < 0.001:
            speedup_class = " best"
            speedup_text += " 最优"
        elif score >= 1.0:
            speedup_class = " good"
        else:
            speedup_class = " bad"
    else:
        speedup_text = "无效"
        speedup_class = " bad"

    time_text = ""
    if score and score > 0 and nid != "root":
        time_text = f"{ctx.baseline_time / score:.2f}us"

    html_parts.append(f'<div class="vtree-node {css_class}">')
    html_parts.append(f'    <div class="vnode-title">{nid}</div>')
    if strategy:
        html_parts.append(f'    <div class="vnode-strategy">{strategy}</div>')
    if desc:
        html_parts.append(f'    <div class="vnode-desc">{_escape_html(desc)}</div>')
    if time_text:
        html_parts.append(f'    <div class="vnode-perf">{time_text}</div>')
    html_parts.append(f'    <div class="vnode-speedup{speedup_class}">{speedup_text}</div>')
    html_parts.append("</div>")


def _link_orphan(level: list, pid: str, j: int, connections: list):
    """把 orphan 子节点 j 连到 level 中 id=pid 的父节点（若存在）。"""
    for i, (pnid, _) in enumerate(level):
        if pnid != pid:
            continue
        existing = next((c for c in connections if c["pi"] == i), None)
        if existing:
            existing["ci"].append(j)
        else:
            connections.append({"pi": i, "ci": [j], "bi": -1})
        break


def _connect_orphans_to_parents(level: list, next_level: list, connections: list):
    """Handle orphan nodes: connect them to their parent_id if the parent is
    in the current level (handles broken children lists)."""
    for j, (cnid, cnode) in enumerate(next_level):
        pid = cnode.get("parent_id")
        if not pid:
            continue
        # Check if this child is already connected
        already_connected = any(j in c["ci"] for c in connections)
        if already_connected:
            continue
        # Find parent in current level
        _link_orphan(level, pid, j, connections)


@dataclass
class _LevelPairContext:
    """相邻两层连接关系构建的上下文。"""
    level: list
    next_level: list
    children_map: dict
    best_path: list[str]


def _find_child_indices(children: list, ctx: _LevelPairContext) -> tuple:
    """找出 children 在 next_level 中的下标及最优路径下标。"""
    child_indices = []
    best_child_idx = -1
    for j, (cnid, _) in enumerate(ctx.next_level):
        if cnid in children:
            child_indices.append(j)
            if cnid in ctx.best_path:
                best_child_idx = j
    return child_indices, best_child_idx


def _build_level_connections(ctx: _LevelPairContext, orphaned: list, is_last: bool) -> str:

    """构建相邻两层之间的连接关系 JSON。"""
    connections = []
    for i, (nid, node) in enumerate(ctx.level):
        children = ctx.children_map.get(nid, [])
        child_indices, best_child_idx = _find_child_indices(children, ctx)
        if child_indices:
            connections.append({"pi": i, "ci": child_indices, "bi": best_child_idx})

    # Also handle orphan nodes in the last level pair
    if is_last and orphaned:
        _connect_orphans_to_parents(ctx.level, ctx.next_level, connections)

    return json.dumps(connections)


def build_decision_tree_html(wm: dict, best_path: list[str], baseline_time: float,
                               best_score: float = 0, rounds: list[dict] | None = None) -> str:
    """Build vertical decision tree HTML from world model."""
    if not wm:
        return '<p style="color:var(--text-muted);">无世界模型数据</p>'

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    root_children = wm.get("decision_tree", {}).get("root", {}).get("children", [])
    if not root_children and "root" in nodes:
        root_children = nodes["root"].get("children", [])

    best_score = best_score or wm.get("best_speedup") or wm.get("best_score", 0)

    ref_to_time = _build_ref_to_time(rounds)
    children_map = _build_children_map(nodes, root_children)
    levels = _bfs_tree_levels(nodes, children_map, baseline_time)

    render_ctx = _TreeRenderContext(
        best_path=best_path, best_score=best_score,
        baseline_time=baseline_time, ref_to_time=ref_to_time,
    )
    html_parts = []
    for lvl_idx, level in enumerate(levels):
        # Render level nodes
        html_parts.append('<div class="vtree-level">')
        for nid, node in level:
            _render_one_tree_node(nid, node, render_ctx, html_parts)
        html_parts.append("</div>")

        # Render connectors to next level using unified children_map
        if lvl_idx < len(levels) - 1:
            next_level = levels[lvl_idx + 1]
            orphaned = next_level if lvl_idx == len(levels) - 2 else []
            conn_json = _build_level_connections(
                _LevelPairContext(level, next_level, children_map, best_path),
                orphaned, is_last=(lvl_idx == len(levels) - 2),
            )
            html_parts.append(
                f'<div class="vtree-connectors" data-parent-level="{lvl_idx}" '
                f"data-connections='{conn_json}'>"
                f'<svg class="vtree-svg"></svg></div>'
            )

    return "\n".join(html_parts)


def _get_node_css_class(nid: str, node: dict, best_path: list[str], best_score: float) -> str:
    if nid == "root":
        return "node-best-path node-root"
    score = node.get("score")
    status = node.get("status", "")
    if score and abs(score - best_score) < 0.001:
        return "node-best-leaf"
    if nid in best_path:
        return "node-best-path"
    if status == "open" and node.get("failure_reason"):
        return "node-invalid"
    if score is not None and score < 1.0:
        return "node-regress"
    return "node-minor"


def _resolve_best_strategy_info(best_variant: dict, wm: dict | None) -> dict:
    """解析最优变体的描述/策略/profiling 信息（世界模型节点优先，eval 兜底）。"""
    node = get_node_for_variant(wm, best_variant["round"], best_variant["parallel"]) if wm else None
    desc = node.get("description", "") if node else best_variant.get("impl_note", "")
    strategies = node.get("strategy_combination", []) if node else []
    impl_note = best_variant.get("impl_note", "")

    # Profiling insight from node or eval
    profiling = {}
    if node and node.get("profiling_insight"):
        profiling = node["profiling_insight"]
    elif best_variant.get("eval") and best_variant["eval"].get("evolved", {}).get("pipeline"):
        profiling = best_variant["eval"]["evolved"]["pipeline"]

    # Build analysis text from available data
    analysis_parts = []
    if desc:
        analysis_parts.append(desc)
    if impl_note and impl_note != desc:
        analysis_parts.append(impl_note)

    return {
        "strategies": strategies,
        "profiling": profiling,
        "full_analysis": "\n\n".join(analysis_parts) if analysis_parts else "暂无详细分析",
    }


def _build_wm_coverage_note(wm: dict, best_variant: dict, speedup: float) -> str:
    """世界模型轮次覆盖不足时生成提示行，覆盖完整返回空串。"""
    wm_best_variant = wm.get("best_variant", "")
    wm_best_speedup = wm.get("best_speedup", 0)
    actual_round = best_variant.get("round", 0)
    actual_par = best_variant.get("parallel", 0)
    wm_rounds_covered = set()
    for n in wm.get("decision_tree", {}).get("nodes", {}).values():
        sol = n.get("solution_ref") or ""
        if sol.startswith("round_"):
            try:
                wm_rounds_covered.add(int(sol.split("/")[0].split("_")[1]))
            except ValueError:
                continue
    total_rounds = actual_round  # best_variant is from the latest round
    if total_rounds <= 0 or len(wm_rounds_covered) >= total_rounds:
        return ""
    missing = total_rounds - len(wm_rounds_covered)
    return (
        f'<li><strong>世界模型数据</strong>: '
        f'仅覆盖前 {len(wm_rounds_covered)} 轮（共 {total_rounds} 轮），'
        f'缺少后续 {missing} 轮的决策节点。'
        f'世界模型记录最优为 {wm_best_variant} ({wm_best_speedup:.3f}x)，'
        f'而实际评估最优为 round_{actual_round}/parallel_{actual_par} ({speedup:.3f}x)。</li>'
    )


def build_best_strategy_section(best_variant: dict | None, wm: dict | None, baseline_time: float = 0.0) -> str:

    """Build the best strategy analysis section with auto-filled content from world model."""
    if not best_variant:
        return ""

    info = _resolve_best_strategy_info(best_variant, wm)

    # Performance data
    best_time = best_variant.get("time_us", 0)
    speedup = baseline_time / best_time if baseline_time > 0 and best_time > 0 else 1.0

    # Strategy tags
    strategies = info["strategies"]
    strategy_tags_html = ""
    if strategies:
        strategy_tags_html = '<div style="margin-top:0.8rem;">' + "".join(
            f'<span class="tag tag-strategy">{s}</span>' for s in strategies
        ) + '</div>'

    # Profiling one-liner
    profiling = info["profiling"]
    one_liner = profiling.get("profiling_one_liner", "") if isinstance(profiling, dict) else ""
    full_analysis = info["full_analysis"]

    # World model coverage note
    wm_coverage_note = _build_wm_coverage_note(wm, best_variant, speedup) if wm else ""

    html = f'''<h2>最优策略分析</h2>
<div class="card">
<h3 style="margin-top:0">优化概述</h3>
<p style="color:var(--text-muted);font-size:0.88rem;margin-bottom:0.8rem;">{_escape_html(full_analysis[:300])}</p>

<h3>关键数据</h3>
<ul style="color:var(--text);font-size:0.88rem;line-height:1.8;">
<li><strong>Baseline</strong>: {baseline_time:.2f} us → <strong>最优</strong>: {best_time:.2f} us ({speedup:.3f}x)</li>
<li><strong>策略组合</strong>: {" + ".join(strategies) if strategies else "开放探索"}</li>
{f'<li><strong>Profiling</strong>: {_escape_html(one_liner)}</li>' if one_liner else ''}
{wm_coverage_note}
</ul>
{strategy_tags_html}
</div>'''
    return html


def _failure_reason(r: dict, ev: dict | None, compilation_ok: bool,
                    precision_ok: bool, wm: dict | None) -> str:
    """判定单个变体的失败原因。"""
    node = get_node_for_variant(wm, r["round"], r["parallel"]) if wm else None
    if node and node.get("failure_reason"):
        return node["failure_reason"]
    if not ev:
        return "evaluation_results.json 缺失"
    if not compilation_ok:
        return "编译失败"
    if not precision_ok:
        return "精度失败"
    return "未知错误"


def build_failure_analysis(rounds: list[dict], wm: dict | None) -> str:
    """Build failure analysis section if there are failed rounds."""
    failures = []
    for r in rounds:
        ev = r.get("eval")
        # Check compilation_success at top level or in comparison (same logic as table rows)
        compilation_ok, precision_ok = _eval_compile_and_precision(ev) if ev else (False, False)
        if not ev or not compilation_ok or not precision_ok:
            reason = _failure_reason(r, ev, compilation_ok, precision_ok, wm)
            failures.append((r["round"], r["parallel"], reason))

    if not failures:
        return ""

    rows = []
    for rn, pn, reason in failures:
        rows.append(f"<tr><td>{rn}-V{pn}</td><td>{_escape_html(reason)}</td></tr>")

    return (
        '<h2>失败/无效轮次分析</h2>\n<div class="card">\n'
        "<table>\n<tr><th>变体</th><th>原因</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n</div>"
    )


def _file_differs_from_baseline(base_file: str, mod_file: str) -> bool:
    """判断修改文件与基线是否存在差异（基线不存在或无差异返回 False）。"""
    if not os.path.isfile(base_file):
        return False  # New file — skip, only include modifications
    with open(base_file, "r", encoding="utf-8", errors="replace") as f:
        base_lines = f.readlines()
    with open(mod_file, "r", encoding="utf-8", errors="replace") as f:
        mod_lines = f.readlines()
    return bool(list(difflib.unified_diff(base_lines, mod_lines, lineterm="")))


def _collect_changed_files(mod_dir: str, prefix: str, baseline_source: str | None) -> list:
    """收集 modified_files 中与基线存在差异的源文件（归一化路径，去重）。"""
    changed_files: list[str] = []
    seen: set[str] = set()  # deduplication
    for root, _, files in os.walk(mod_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), mod_dir)
            if not _is_source_file(rel):
                continue
            # Normalize path for baseline comparison
            norm_rel = _strip_prefix(rel, prefix)
            # Deduplicate (same normalized path from different prefix paths)
            if norm_rel in seen:
                continue
            seen.add(norm_rel)
            if baseline_source and not _file_differs_from_baseline(
                    os.path.join(baseline_source, norm_rel),
                    os.path.join(mod_dir, rel)):
                continue
            changed_files.append(norm_rel)
    return changed_files


def build_apply_cmd(best_variant: dict | None, output_dir: str, baseline_source: str | None) -> str:

    """Build the apply command for the best variant.

    Only includes files that:
    1. Are source files (not docs, tests, configs)
    2. Actually differ from baseline (when baseline is available)
    3. Exist in baseline (new files are excluded — only modifications)

    Output is a concise shell snippet: variable definitions + cp commands only.
    No shebang, comments, or set -e — just the actionable commands.
    """
    if not best_variant:
        return "# 无最优变体"

    best_path = best_variant["path"]
    # Try modified_files first (ops-evo), then kernel (lingxi-evo)
    mod_dir = os.path.join(best_path, "modified_files")
    if not os.path.isdir(mod_dir):
        mod_dir = os.path.join(best_path, "kernel")
    if not os.path.isdir(mod_dir):
        return "# 未找到修改文件目录"

    # Determine repo path
    if baseline_source:
        repo_path = baseline_source
    else:
        repo_path = "<your_ops_repo_path>"

    prefix = best_variant.get("modified_files_prefix", "")

    # Only include source files that actually differ from baseline
    changed_files = _collect_changed_files(mod_dir, prefix, baseline_source)

    if not changed_files:
        return "# 无源代码修改（所有文件与基线一致）"

    # Use forward slash consistently for path separator in prefix
    prefix_path = prefix.replace(os.sep, "/") + "/" if prefix else ""

    # Build concise command snippet: variables + cp commands only
    lines: list[str] = [
        f'REPO_ROOT="{repo_path}"',
        f'OUTPUT_DIR="{os.path.abspath(output_dir)}"',
        f'BEST_VARIANT="{os.path.relpath(best_path, output_dir)}"',
        "",
    ]

    # Group by directory for cleaner output and merge cp commands
    dirs: dict[str, list[str]] = {}
    for rel in sorted(changed_files):
        parent = os.path.dirname(rel) or "."
        dirs.setdefault(parent, []).append(rel)

    for _, files in sorted(dirs.items()):
        for f in files:
            src = f'"$OUTPUT_DIR/$BEST_VARIANT/modified_files/{prefix_path}{f}"'
            dst = f'"$REPO_ROOT/{f}"'
            lines.append(f"cp {src} {dst}")

    return "\n".join(lines)


def _check_best_variant(best: dict | None, html: str) -> list[str]:
    """检查 2: 最优变体有效性。"""
    warnings = []
    if not best:
        return warnings
    ev = best.get("eval", {})
    compilation_ok = ev.get("compilation_success")
    if compilation_ok is None and "comparison" in ev:
        compilation_ok = ev.get("comparison", {}).get("compilation_success")
    precision_ok = ev.get("precision_passed")
    if precision_ok is None and "comparison" in ev:
        precision_ok = ev.get("comparison", {}).get("precision_passed")
    evolved_time = ev.get("evolved", {}).get("time_us")

    if not compilation_ok:
        warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} compilation_success=False，请确认")
    if not precision_ok:
        warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} precision_passed=False，请确认")
    if evolved_time is None or evolved_time <= 0:
        warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} time_us 无效 ({evolved_time})，请确认")

    # 2a. Check for redundant/unmodified files in diff section
    mod_files = best.get("modified_files", [])
    prefix = best.get("modified_files_prefix", "")
    if len(mod_files) > 20:
        warnings.append(f"代码修改部分文件数异常多 ({len(mod_files)} 个)，可能存在冗余未修改文件")
    # Check if path prefix was detected but not stripped properly
    if prefix and f'modified_files/{prefix}/' not in html:
        warnings.append(f"modified_files 路径前缀 '{prefix}' 可能未正确处理，diff 或 apply-cmd 路径可能不匹配")
    return warnings


def _check_all_failed(rounds: list[dict]) -> list[str]:
    """检查 3: 可疑的全失败模式。"""
    failed_count = 0
    for r in rounds:
        ev = r.get("eval", {})
        if ev is None:
            ev = {}
        compilation_ok = ev.get("compilation_success")
        if compilation_ok is None and "comparison" in ev:
            compilation_ok = ev.get("comparison", {}).get("compilation_success")
        if not compilation_ok:
            failed_count += 1
    if failed_count == len(rounds) and len(rounds) > 0:
        return [f"所有 {len(rounds)} 个变体均标记为编译失败，请检查 evaluation_results.json 格式或 ops-partial 子agent 输出"]
    return []


def _check_wm_consistency(wm: dict | None, rounds: list[dict], baseline_time: float) -> list[str]:
    """检查 4: 世界模型数据一致性（score 与 eval 耗时交叉验证）。"""
    warnings = []
    if not wm:
        return warnings
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    missing_nodes = []
    for r in rounds:
        ref = f"round_{r['round']}/parallel_{r['parallel']}"
        node = None
        for n in nodes.values():
            if n.get("solution_ref") == ref:
                node = n
                break
        if not node:
            missing_nodes.append(f"R{r['round']}-P{r['parallel']}")
            continue
        node_score = node.get("score")
        ev = r.get("eval", {})
        evolved_time = ev.get("evolved", {}).get("time_us")
        has_valid_score = bool(node_score) and node_score > 0
        has_valid_time = bool(evolved_time) and evolved_time > 0
        if has_valid_score and has_valid_time:
            computed_speedup = baseline_time / evolved_time
            if abs(computed_speedup - node_score) > 0.05:
                warnings.append(
                    f"R{r['round']}-P{r['parallel']} speedup 不一致: "
                    f"world_model={node_score:.3f}x, eval={computed_speedup:.3f}x"
                )
    if missing_nodes:
        warnings.append(
            f"以下变体在世界模型中无对应节点（可能 refine 被跳过）: {', '.join(missing_nodes)}"
        )
    return warnings


def _check_subtitle(html: str) -> list[str]:
    """检查 7: 副标题质量（拒绝占位符式取值）。"""
    subtitle_match = re.search(r'<p class="subtitle">(.+?)</p>', html)
    if not subtitle_match:
        return []
    subtitle_text = subtitle_match.group(1).strip()
    # Strip HTML tags for plain text check
    plain_subtitle = re.sub(r'<[^>]+>', '', subtitle_text).strip()
    bad_subtitles = {"test", "report", "title", "default", ""}
    if plain_subtitle.lower() in bad_subtitles:
        return [f"报告副标题不规范: '{plain_subtitle}'，请检查 --title 参数"]
    return []


def _check_apply_cmd(html: str, best: dict | None) -> list[str]:
    """检查 8: apply 命令冗余与路径合法性。"""
    warnings = []
    apply_match = re.search(r'id="apply-cmd">(.*?)</code>', html, re.DOTALL)
    if not apply_match:
        return warnings
    cmd_text = apply_match.group(1)
    cp_lines = [l for l in cmd_text.split('\n') if l.strip().startswith('cp ')]
    if len(cp_lines) > 20:
        warnings.append(f"应用最优变体命令数量过多 ({len(cp_lines)} 条)，可能存在冗余或路径前缀未剥离")
    # Check for potential path mismatch (op-name prefix in destination)
    if best:
        prefix = best.get("modified_files_prefix", "")
        if prefix and any(f'$REPO_ROOT/{prefix}/' in l for l in cp_lines):
            warnings.append(f"apply-cmd 目标路径包含算子名前缀 '{prefix}/'，可能与仓库实际结构不匹配")
    # Check for old-style boilerplate that should have been removed
    if '#!/bin/bash' in cmd_text or 'set -e' in cmd_text:
        warnings.append("apply-cmd 含冗余指令（shebang / set -e），应仅保留变量定义和 cp 命令")
    return warnings


def _check_round_timing(round_timing: dict | None) -> list[str]:
    """检查 9: 轮次耗时异常。"""
    warnings = []
    if not (round_timing and round_timing.get("rounds")):
        return warnings
    for rn, rd in round_timing["rounds"].items():
        duration = rd.get("duration_minutes", 0)
        if duration <= 0:
            warnings.append(f"轮次 {rn} 耗时异常 (<=0 分钟: {duration:.2f})，请检查目录时间戳数据")
        elif duration < 0.5:
            warnings.append(f"轮次 {rn} 耗时过短 ({duration:.2f} 分钟)，可能存在预创建目录时间戳干扰")
    return warnings


def _check_tree_topology(wm: dict | None, rounds: list[dict]) -> list[str]:
    """检查 11: 决策树拓扑（变体节点覆盖 + 孤立节点）。"""
    warnings = []
    if not wm:
        return warnings
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    if not nodes:
        return warnings
    # Check all variant rounds have corresponding tree nodes
    variant_refs = {f"round_{r['round']}/parallel_{r['parallel']}" for r in rounds}
    node_refs = {n.get("solution_ref") for n in nodes.values() if n.get("solution_ref")}
    missing_in_tree = variant_refs - node_refs
    if missing_in_tree:
        warnings.append(
            f"决策树缺少 {len(missing_in_tree)} 个变体节点: "
            f"{', '.join(sorted(missing_in_tree)[:5])}"
        )
    # Check for orphan nodes (no parent and not root children)
    root_children_set = set(wm.get("decision_tree", {}).get("root", {}).get("children", []))
    if "root" in nodes:
        root_children_set.update(nodes["root"].get("children", []))
    all_children = set()
    for n in nodes.values():
        all_children.update(n.get("children", []))
    all_children.update(root_children_set)
    orphans = []
    for nid in nodes:
        if nid == "root" or nid in all_children:
            continue
        if not nodes[nid].get("parent_id"):
            orphans.append(nid)
    if orphans:
        warnings.append(
            f"决策树存在 {len(orphans)} 个孤立节点（无父引用）: "
            f"{', '.join(orphans[:5])}"
        )
    return warnings


def _check_diff_new_file_leak(html: str) -> list[str]:
    """检查 12: diff 中新文件泄漏（应只展示已有文件的修改）。"""
    warnings = []
    diff_sections = re.findall(r'<pre><code class="diff">(.*?)</code></pre>', html, re.DOTALL)
    for i, diff_content in enumerate(diff_sections):
        lines = diff_content.strip().split('\n')
        add_lines = [l for l in lines if l.startswith('+') and not l.startswith('+++')]
        del_lines = [l for l in lines if l.startswith('-') and not l.startswith('---')]
        # If a diff has only additions and zero deletions, it's likely a new file leak
        if len(add_lines) > 10 and len(del_lines) == 0:
            # Check for the "new file" pattern in hunk headers
            has_dev_null = any('--- /dev/null' in l or '--- a/dev/null' in l for l in lines)
            if has_dev_null:
                warnings.append(
                    f"代码修改第 {i+1} 段 diff 为新增文件（非已有文件修改），不应出现在报告中"
                )
    return warnings


def _check_apply_dup_dest(html: str) -> list[str]:
    """检查 13: apply 命令重复目标路径。"""
    apply_match = re.search(r'id="apply-cmd">(.*?)</code>', html, re.DOTALL)
    if not apply_match:
        return []
    cmd_text = apply_match.group(1)
    cp_lines = [l for l in cmd_text.split('\n') if l.strip().startswith('cp ')]
    # Extract destination paths and check for duplicates
    destinations = []
    for cp_line in cp_lines:
        parts = cp_line.strip().split()
        if len(parts) >= 3:
            destinations.append(parts[-1])
    dup_dests = [d for d in destinations if destinations.count(d) > 1]
    if dup_dests:
        unique_dups = sorted(set(dup_dests))
        return [
            f"apply-cmd 存在 {len(unique_dups)} 个重复目标路径: "
            f"{', '.join(unique_dups[:3])}"
        ]
    return []


def _check_total_duration(round_timing: dict | None) -> list[str]:
    """检查 14: 总时长合理性（超过轮次总和 3 倍视为 idle 膨胀）。"""
    if not round_timing:
        return []
    total_dur = round_timing.get("total_evolution_minutes", 0)
    rounds_dur = sum(
        rd.get("duration_minutes", 0)
        for rd in round_timing.get("rounds", {}).values()
    )
    # Total should not exceed sum of rounds by more than 3x (indicates idle inflation)
    if rounds_dur > 0 and total_dur > rounds_dur * 3:
        return [
            f"总进化耗时 ({total_dur:.1f} 分钟) 远超各轮次总和 ({rounds_dur:.1f} 分钟)，"
            f"可能包含大量空闲等待时间"
        ]
    return []


def self_check_report(html: str, rounds: list[dict], wm: dict | None,
                      baseline_time: float, round_timing: dict | None = None) -> list[str]:
    """Run self-check on the generated report and return a list of warnings."""
    warnings = []
    best = find_best_variant(rounds, baseline_time)

    # 1. Check for placeholder residues
    if "LLM_FILL" in html:
        warnings.append("报告仍含未填充的 LLM_FILL 占位符")

    warnings.extend(_check_best_variant(best, html))
    warnings.extend(_check_all_failed(rounds))
    warnings.extend(_check_wm_consistency(wm, rounds, baseline_time))

    # 5. Check test case rows
    if '<td colspan="4">参数信息不可用</td>' in html:
        warnings.append("测试用例参数未获取到，请检查 shared/call_spec.json 是否存在")

    # 6. Check model info (accept plain model name in subtitle or explicit tag)
    _known_models = ["Kimi", "Claude", "GPT", "Gemini", "Qwen", "DeepSeek", "Llama"]
    has_model_info = any(m in html for m in _known_models) or "Model:" in html
    if not has_model_info:
        warnings.append("未包含模型信息，请检查 session JSONL 是否存在或模型字段是否可用")

    warnings.extend(_check_subtitle(html))
    warnings.extend(_check_apply_cmd(html, best))
    warnings.extend(_check_round_timing(round_timing))

    # 10. Check token table completeness (cache_creation column)
    if '词元用量统计' in html and 'Cache Creation' not in html and 'cache_creation' not in html:
        warnings.append("词元用量统计表格缺少 Cache Creation 列，统计可能不完整")

    warnings.extend(_check_tree_topology(wm, rounds))
    warnings.extend(_check_diff_new_file_leak(html))
    warnings.extend(_check_apply_dup_dest(html))
    warnings.extend(_check_total_duration(round_timing))

    return warnings


def build_strategy_legend(wm: dict | None) -> str:
    """Build strategy code legend from world model nodes."""
    if not wm:
        return ""
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    strategies = set()
    for node in nodes.values():
        for s in node.get("strategy_combination", []):
            strategies.add(s)
    if not strategies:
        return ""
    # Known strategy names
    known = {
        "P1": "双缓冲", "P2": "自适应Tiling", "P3": "多核均衡", "P4": "向量化DMA",
        "P5": "标量优化", "P7": "循环展开", "P8": "流水线优化",
        "P9": "确定性输出", "P10": "预取", "P11": "数据对齐", "P12": "融合计算",
    }
    parts = []
    for s in sorted(strategies):
        name = known.get(s, "")
        parts.append(f"{s}={name}" if name else s)
    return "策略编号: " + " ".join(parts) + " | open=开放探索模式"


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evolution optimization HTML report")
    parser.add_argument("output_dir", help="Evolution output directory path")
    parser.add_argument("--baseline-source", default=None, help="Baseline source directory for diff")
    parser.add_argument("--title", default=None, help="Custom report title")
    parser.add_argument("--pipeline", default="ops-evo",
                        help="Pipeline type (default: ops-evo)")
    parser.add_argument("--session-jsonl", default=None,
                        help="Explicit path to the session JSONL file for token/timing stats")
    return parser


def _extract_baseline_info(baseline_eval: dict | None, wm: dict | None) -> tuple:
    """从 baseline_evaluation.json / world_model 提取基线信息。

    Returns (baseline_time, test_case, hw_params, eval_info)。
    """
    baseline_time = 0.0
    test_case = {}
    hw_params = {}
    eval_info = {}

    if baseline_eval:
        baseline_time = baseline_eval.get("baseline", {}).get("time_us", 0)
        # Forge mode: baseline tested against itself stores actual time in evolved.time_us
        if baseline_time <= 0:
            baseline_time = baseline_eval.get("evolved", {}).get("time_us", 0)
        # Flat format fallback (forge raw result or standalone files)
        if baseline_time <= 0:
            baseline_time = baseline_eval.get("baseline_time_us", 0)
        test_case = baseline_eval.get("test_case", {})
        if not test_case:
            test_case = baseline_eval.get("test_config", {})
        eval_info = baseline_eval
    if wm:
        if baseline_time <= 0:
            baseline_time = wm.get("baseline", {}).get("time_us", 0)
        if baseline_time <= 0:
            baseline_time = wm.get("baseline_time_us", 0)
        hw_params = wm.get("hw_params", {})
        if not test_case:
            test_case = wm.get("test_case", {})
    return baseline_time, test_case, hw_params, eval_info


def _collect_and_enrich_rounds(output_dir: str, pipeline_type: str, wm: dict | None,
                               baseline_time: float, op_name: str) -> tuple:
    """收集轮次数据并确定最优变体/最优路径。"""
    rounds = collect_rounds(output_dir, pipeline_type)
    # Detect common prefix in modified_files for path normalization
    all_mod_files = []
    for r in rounds:
        all_mod_files.extend(r.get("modified_files", []))
    prefix = _detect_modified_files_prefix(all_mod_files, op_name)
    for r in rounds:
        r["modified_files_prefix"] = prefix
    enrich_rounds_from_world_model(rounds, wm, baseline_time)
    best_variant = find_best_variant(rounds, baseline_time) if baseline_time > 0 else None
    best_path, wm_best_score = find_best_path(wm, best_variant, baseline_time) if wm else ([], 0.0)
    return rounds, best_variant, best_path, wm_best_score


def _build_tree_summary(wm: dict | None, best_path: list, speedup: float,
                        num_rounds: int, wm_best_score: float) -> str:
    """构建决策树摘要行（含轮次覆盖警告和 wm score 差异提示）。"""
    if not wm:
        return "无世界模型数据"
    node_count = len(wm.get('decision_tree', {}).get('nodes', {}))
    # Detect world-model coverage gaps
    wm_rounds_covered = set()
    for n in wm.get("decision_tree", {}).get("nodes", {}).values():
        sol = n.get("solution_ref") or ""
        if sol.startswith("round_"):
            try:
                wm_rounds_covered.add(int(sol.split("/")[0].split("_")[1]))
            except ValueError:
                continue
    missing_rounds = num_rounds - len(wm_rounds_covered) if num_rounds > 0 else 0
    coverage_warn = ""
    if missing_rounds > 0:
        coverage_warn = (
            f' | <span style="color:var(--orange);font-weight:600;">'
            f'警告: 仅覆盖 {len(wm_rounds_covered)}/{num_rounds} 轮，缺少 {missing_rounds} 轮节点'
            f'</span>'
        )
    wm_score_note = ""
    if wm_best_score and abs(wm_best_score - speedup) > 0.01:
        wm_score_note = f" | 世界模型记录: {wm_best_score:.3f}x"
    return (
        f"共 {node_count} 个节点 | "
        f'最优路径: <span style="color:var(--green);font-weight:600;">'
        f"{' → '.join(best_path)}</span> ({speedup:.3f}x)"
        f"{wm_score_note}{coverage_warn}"
    )


def _collect_session_search_roots() -> list:
    """收集所有可能的 .claude/projects 搜索根目录（支持多用户）。"""
    search_roots = []
    home_candidates = [Path.home()]
    extra_homes_env = os.environ.get("LINGXI_EXTRA_HOME_DIRS", "")
    for extra in extra_homes_env.split(":"):
        extra = extra.strip()
        if extra:
            home_candidates.append(Path(extra))
    for home_dir in home_candidates:
        if home_dir.exists():
            projects_dir = home_dir / ".claude" / "projects"
            if projects_dir.exists():
                search_roots.append(projects_dir)
    # Also check current user's home if not already included
    current_home = Path.home()
    if not any(str(current_home) in str(r) for r in search_roots):
        projects_dir = current_home / ".claude" / "projects"
        if projects_dir.exists():
            search_roots.append(projects_dir)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for r in search_roots:
        if str(r) not in seen:
            seen.add(str(r))
            deduped.append(r)
    return deduped


def _filter_sessions_by_round1_anchor(candidate_sessions: list, output_dir: str) -> list:
    """用 round_1 的 birthtime 作为锚点过滤匹配的 session 文件。"""
    round1_dir = Path(output_dir) / "round_1"
    if not round1_dir.exists():
        return candidate_sessions
    round1_birth = _get_dir_birthtime(round1_dir)
    if not round1_birth:
        return candidate_sessions
    # Convert seconds -> milliseconds for comparison
    round1_birth_ms = round1_birth * 1000 if round1_birth < 1e12 else round1_birth
    filtered = []
    for cand in candidate_sessions:
        start_ms, end_ms = get_session_time_range(cand["session_file"])
        if start_ms is not None and end_ms is not None:
            if start_ms <= round1_birth_ms <= end_ms:
                filtered.append(cand)
    return filtered or candidate_sessions


def _find_session_files(args, output_dir: str) -> tuple:
    """定位本次 session 的目录与 JSONL 文件。

    Returns (session_dir, session_jsonl)，未找到返回 (None, None)。
    """
    # Explicit override via CLI
    if args.session_jsonl:
        jsonl_path = Path(args.session_jsonl)
        if jsonl_path.exists():
            return jsonl_path.parent, jsonl_path
        LOGGER.warning("Warning: --session-jsonl not found: %s", args.session_jsonl)

    # Search all possible .claude/projects directories (not just Path.home())
    # because the script may run as a different user than the session owner
    search_roots = _collect_session_search_roots()

    # Collect all candidate session files across ALL search roots
    candidate_sessions = []
    output_path = Path(output_dir).resolve()
    output_str = str(output_path)
    output_marker = "/output/"
    if output_marker in output_str:
        prefix_str = output_str[:output_str.index(output_marker)]
        output_as_proj = "-" + prefix_str.replace("/", "-").replace("_", "-").lstrip("-")
    else:
        output_as_proj = "-" + output_str.replace("/", "-").replace("_", "-").lstrip("-")

    for projects_dir in search_roots:
        if not projects_dir.exists():
            continue
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            if proj_dir.name != output_as_proj:
                continue
            session_files = list(proj_dir.glob("*.jsonl"))
            for sf in session_files:
                subagent_dir = sf.parent / sf.stem / "subagents"
                has_subagents = (
                    subagent_dir.exists() and len(list(subagent_dir.glob("*.jsonl"))) > 0
                )
                candidate_sessions.append({
                    "proj_dir": proj_dir,
                    "session_file": sf,
                    "has_subagents": has_subagents,
                    "mtime": sf.stat().st_mtime,
                })

    # Use round_1 birthtime as an anchor to find the matching session
    candidate_sessions = _filter_sessions_by_round1_anchor(candidate_sessions, output_dir)

    if candidate_sessions:
        best = max(candidate_sessions, key=lambda c: (c["has_subagents"], c["mtime"]))
        return best["proj_dir"], best["session_file"]
    return None, None


def _load_resource_stats(session_dir, session_jsonl, output_dir: str, pipeline_type: str) -> str:
    """解析 session 统计并构建资源消耗区块 HTML。"""
    session_stats = {}
    round_timing = {}
    if session_dir and session_jsonl:
        try:
            session_stats = parse_session_stats(str(session_dir), str(session_jsonl))
        except Exception as e:
            LOGGER.warning("Warning: Failed to parse session stats: %s", e)
            session_stats = {}

    try:
        round_timing = parse_round_timing(output_dir)
    except Exception as e:
        LOGGER.warning("Warning: Failed to parse round timing: %s", e)
        round_timing = {}

    if not (session_stats or round_timing):
        return ""
    try:
        return build_resource_stats_section(session_stats, round_timing, pipeline_type)
    except Exception as e:
        LOGGER.warning("Warning: Failed to build resource stats section: %s", e)
        return ""


def _auto_detect_baseline_source(args, output_dir: str):
    """自动检测 baseline 源目录（ops-evo: shared/original/；lingxi-evo: shared/kernel/）。"""
    if args.baseline_source:
        return args.baseline_source
    for baseline_dir in ["shared/original", "shared/kernel"]:
        auto_baseline = os.path.join(output_dir, baseline_dir)
        if os.path.isdir(auto_baseline):
            return auto_baseline
    return None


def _parse_round_timing_safe(output_dir: str) -> dict:
    try:
        return parse_round_timing(output_dir)
    except Exception as e:
        LOGGER.warning("Warning: Failed to parse round timing: %s", e)
        return {}


def _load_report_inputs(args, output_dir: str) -> tuple:
    """加载报告输入数据：baseline_eval / world_model / 输出目录名解析。

    输入缺失时返回 None（由 main 统一退出）。
    """
    op_name, pipeline_type, timestamp = parse_output_dir_name(output_dir)
    # Override with --pipeline if explicitly provided
    if args.pipeline and args.pipeline != "ops-evo":
        pipeline_type = args.pipeline
    baseline_eval = load_json(os.path.join(output_dir, "baseline_evaluation.json"))
    wm = load_json(os.path.join(output_dir, "world_model_final.json"))
    if not wm:
        wm = load_json(os.path.join(output_dir, "world_model.json"))

    if not baseline_eval and not wm:
        return None
    return op_name, pipeline_type, timestamp, baseline_eval, wm


def _build_subtitle(args, timestamp: str, chip: str, num_rounds: int,
                    num_parallels: int) -> str:
    """构建报告副标题（拒绝占位符式 --title）。"""
    default_subtitle = (f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                        f" | Ascend {chip} | {num_rounds}轮 x {num_parallels}变体")
    # Reject placeholder-like titles (e.g., "test", "report", empty)
    _bad_titles = {"test", "report", "", "title", "default"}
    return (default_subtitle if (args.title or "").strip().lower() in _bad_titles
            else (args.title or default_subtitle))


def _load_session_stats(args, output_dir: str) -> tuple:
    """解析 session 统计与轮次时间，失败降级为空 dict。"""
    session_dir, session_jsonl = _find_session_files(args, output_dir)
    session_stats = {}
    if session_dir and session_jsonl:
        try:
            session_stats = parse_session_stats(str(session_dir), str(session_jsonl))
        except Exception as e:
            LOGGER.warning("Warning: Failed to parse session stats: %s", e)
            session_stats = {}
    return session_stats, _parse_round_timing_safe(output_dir)


def _build_resource_stats_safe(session_stats: dict, round_timing: dict,
                               pipeline_type: str) -> str:
    """构建资源消耗区块，失败返回空串。"""
    if not (session_stats or round_timing):
        return ""
    try:
        return build_resource_stats_section(session_stats, round_timing, pipeline_type)
    except Exception as e:
        LOGGER.warning("Warning: Failed to build resource stats section: %s", e)
        return ""


def _build_replacements(args, ctx: dict) -> dict:
    """组装模板替换字典。"""
    return {
        "OP_NAME": ctx["op_name"],
        "SUBTITLE": ctx["subtitle"],
        "BASELINE_TIME": f"{ctx['baseline_time']:.2f}",
        "BEST_TIME": f"{ctx['best_time']:.2f}",
        "SPEEDUP": f"{ctx['speedup']:.3f}",
        "TIME_REDUCTION": f"{ctx['time_reduction']:+.1f}",
        "TEST_CASE_ROWS": build_test_case_rows(ctx["test_case"], ctx["output_dir"]),
        "HARDWARE_ROWS": build_hardware_rows(ctx["hw_params"], ctx["eval_info"]),
        "EVOLUTION_TABLE_ROWS": build_evolution_table_rows(
            ctx["rounds"], ctx["wm"], ctx["baseline_time"], ctx["best_variant"]),
        "STRATEGY_LEGEND": build_strategy_legend(ctx["wm"]),
        "BEST_STRATEGY_SECTION": build_best_strategy_section(
            ctx["best_variant"], ctx["wm"], ctx["baseline_time"]),
        "CODE_DIFF_SECTIONS": build_code_diff_sections(ctx["best_variant"], ctx["baseline_source"]),
        "TREE_SUMMARY": ctx["tree_summary"],
        "DECISION_TREE_HTML": build_decision_tree_html(
            ctx["wm"], ctx["best_path"], ctx["baseline_time"], ctx["speedup"], ctx["rounds"]),
        "FAILURE_ANALYSIS_SECTION": build_failure_analysis(ctx["rounds"], ctx["wm"]),
        "RESOURCE_STATS_SECTION": ctx["resource_stats_section"],
        "APPLY_CMD": build_apply_cmd(ctx["best_variant"], ctx["output_dir"], ctx["baseline_source"]),
        "CHART_INIT_SCRIPT": build_chart_script(ctx["rounds"], ctx["baseline_time"]),
        "MODEL_INFO": "",
    }


def _write_report(output_dir: str, op_name: str, timestamp: str, html: str) -> str:
    """写出 HTML 报告文件，返回输出路径。"""
    out_filename = f"evolution-report_{op_name}_{timestamp}.html"
    out_path = os.path.join(output_dir, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


@dataclass
class _ReportInputs:
    """报告输入数据封装（baseline_eval / world_model / 基线信息 / 测试用例）。"""
    op_name: str
    pipeline_type: str
    timestamp: str
    baseline_eval: dict | None
    wm: dict | None
    baseline_time: float
    test_case: dict
    hw_params: dict
    eval_info: dict


def _resolve_inputs_and_test_case(args, output_dir: str) -> _ReportInputs | None:
    """解析报告输入 + 基线信息 + 测试用例。输入缺失时返回 None（由 main 统一退出）。"""
    inputs = _load_report_inputs(args, output_dir)
    if inputs is None:
        return None
    op_name, pipeline_type, timestamp, baseline_eval, wm = inputs
    baseline_time, test_case, hw_params, eval_info = _extract_baseline_info(baseline_eval, wm)

    # Fallback: parse test_cases.csv for test case params
    if not test_case:
        csv_path = os.path.join(output_dir, "shared", "test_cases.csv")
        if os.path.isfile(csv_path):
            test_case = _parse_test_cases_csv(csv_path)

    return _ReportInputs(
        op_name=op_name, pipeline_type=pipeline_type, timestamp=timestamp,
        baseline_eval=baseline_eval, wm=wm, baseline_time=baseline_time,
        test_case=test_case, hw_params=hw_params, eval_info=eval_info,
    )


def _compute_metrics(rounds: list, best_variant: dict | None, baseline_time: float) -> tuple:
    """计算 best_time / speedup / time_reduction / num_rounds / num_parallels。"""
    best_time = best_variant["time_us"] if best_variant else baseline_time
    speedup = baseline_time / best_time if best_time > 0 else 1.0
    # 时间变化量：负值表示变快（耗时下降），正值表示变慢（耗时上升）
    time_reduction = (best_time / baseline_time - 1) * 100 if baseline_time > 0 else 0
    num_rounds = max((r["round"] for r in rounds), default=0) if rounds else 0
    num_parallels = max((r["parallel"] for r in rounds), default=0) + 1 if rounds else 0
    return best_time, speedup, time_reduction, num_rounds, num_parallels


def _load_template_str() -> str:
    """加载 HTML 报告模板。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "..", "templates", "evolution-report.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_main_parser().parse_args()
    output_dir = args.output_dir.rstrip("/")

    if not os.path.isdir(output_dir):
        LOGGER.error("Error: output directory not found: %s", output_dir)
        sys.exit(1)

    ri = _resolve_inputs_and_test_case(args, output_dir)
    if ri is None:
        LOGGER.error("Error: neither baseline_evaluation.json nor world_model_final.json found")
        sys.exit(2)

    rounds, best_variant, best_path, wm_best_score = _collect_and_enrich_rounds(
        output_dir, ri.pipeline_type, ri.wm, ri.baseline_time, ri.op_name)

    best_time, speedup, time_reduction, num_rounds, num_parallels = _compute_metrics(
        rounds, best_variant, ri.baseline_time)

    # Build subtitle
    subtitle = _build_subtitle(args, ri.timestamp, ri.hw_params.get("chip_model", "Unknown"),
                               num_rounds, num_parallels)

    template_str = _load_template_str()

    baseline_source = _auto_detect_baseline_source(args, output_dir)
    tree_summary = _build_tree_summary(ri.wm, best_path, speedup, num_rounds, wm_best_score)

    session_stats, round_timing = _load_session_stats(args, output_dir)
    resource_stats_section = _build_resource_stats_safe(session_stats, round_timing, ri.pipeline_type)

    # Model info: prefer config file, fallback to session JSONL
    model_name = get_model_from_config()
    if not model_name and session_stats:
        model_name = session_stats.get("model")
    # Append model info to subtitle for a cleaner header
    if model_name:
        subtitle = f"{model_name} | {subtitle}"

    ctx = {
        "output_dir": output_dir, "op_name": ri.op_name, "wm": ri.wm,
        "baseline_time": ri.baseline_time, "test_case": ri.test_case,
        "hw_params": ri.hw_params, "eval_info": ri.eval_info,
        "rounds": rounds, "best_variant": best_variant, "best_path": best_path,
        "best_time": best_time, "speedup": speedup, "time_reduction": time_reduction,
        "subtitle": subtitle, "baseline_source": baseline_source,
        "tree_summary": tree_summary, "resource_stats_section": resource_stats_section,
    }
    replacements = _build_replacements(args, ctx)

    # Render template
    tmpl = Template(template_str)
    html = tmpl.safe_substitute(replacements)

    # Self-check
    warnings = self_check_report(html, rounds, wm, baseline_time, round_timing)
    if warnings:
        LOGGER.warning("\n[报告自检警告]")
        for w in warnings:
            LOGGER.warning("  [WARN] %s", w)
    else:
        LOGGER.info("\n[报告自检通过] 无警告")

    out_path = _write_report(output_dir, ri.op_name, ri.timestamp, html)
    DATA_LOGGER.info("%s", out_path)


if __name__ == "__main__":
    main()
