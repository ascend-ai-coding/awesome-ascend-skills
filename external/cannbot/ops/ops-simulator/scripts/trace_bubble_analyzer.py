#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
trace_bubble_analyzer.py — Simulator Trace 空泡分析器

读取 msprof / cannsim 产出的 Chrome Trace Format，自动识别 pipeline 映射，
对主 pipeline（VECTOR/CUBE）的空泡进行分类归因。

用法:
    # 分析单核 trace
    python3 trace_bubble_analyzer.py path/to/trace_core0.json

    # 分析 simulator 输出目录（自动发现所有核）
    python3 trace_bubble_analyzer.py path/to/simulator_output/

    # 输出 JSON 报告
    python3 trace_bubble_analyzer.py path/to/simulator_output/ --json -o report.json

输出:
    文本模式: 逐核空泡统计 +  Top 瓶颈
    JSON 模式: 结构化数据，供 Agent 进一步分析
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

LOGGER = logging.getLogger(__name__)


# ── PID 映射 ───────────────────────────────────────────────

PIPELINE_PID_MAP = {
    10: "SCALAR",
    20: "SCALARLDST",
    30: "VECTOR",
    40: "CUBE",
    50: "MTE1",
    60: "MTE2",
    70: "MTE3",
    80: "FIXPIPE",
    90: "FLOWCTRL",
    100: "ALL",
    110: "CACHEMISS",
}

PIPELINE_NAME_MAP = {v: k for k, v in PIPELINE_PID_MAP.items()}

_CANNSIM_PID_REMAP = {
    1: 10, 2: 20, 4: 60, 5: 30, 6: 40, 7: 70, 3: 50,
    11: 30, 12: 30, 13: 30, 14: 30, 15: 90, 17: 80, 18: 30,
}

_CANNSIM_NAME_REMAP = {
    "SCALAR": 10, "SCALARLDST": 20, "VEC": 30, "VECTOR": 30,
    "CUBE": 40, "MTE1": 50, "MTE2": 60, "MTE3": 70,
    "FIXP": 80, "FIXPIPE": 80, "FLOWCTRL": 90, "FLOWCONTROL": 90,
    "RVECSU": 30, "RVECEX": 30, "RVECLD": 30, "RVECST": 30, "RVECLP": 30,
    "PUSHQ": 90,
}

PID_SCALAR = 10
PID_SCALARLDST = 20
PID_VECTOR = 30
PID_CUBE = 40
PID_MTE1 = 50
PID_MTE2 = 60
PID_MTE3 = 70
PID_FIXPIPE = 80
PID_FLOWCTRL = 90
PID_CACHEMISS = 110

DRAIN_OPS = frozenset({"VNCHWCONV", "BAR", "PipeBarrier", "PIPE_BARRIER", "VNCHWCONV_B16", "VNCHWCONV_B32"})
BARRIER_OPS = frozenset({"BAR", "PipeBarrier", "PIPE_BARRIER"})

# ── 阈值 ───────────────────────────────────────────────────

THRESHOLD_A_MAX_PS = 1.0
THRESHOLD_B_DRAIN_PS = 30.0
THRESHOLD_C_MAX_PS = 500.0

_DIR_PATTERN = re.compile(r"^core\d+\.(veccore\d+|cubecore\d+)$")


# ── 数据结构 ───────────────────────────────────────────────

@dataclass
class TraceEvent:
    name: str
    ph: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict = field(default_factory=dict)
    cname: Optional[str] = None

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur

    @property
    def pipeline_name(self) -> str:
        return PIPELINE_PID_MAP.get(self.pid, f"UNKNOWN_{self.pid}")

    def is_wait_flag(self) -> bool:
        return self.name == "WAIT_FLAG"

    def waited_pipeline(self) -> Optional[str]:
        if not self.is_wait_flag():
            return None
        detail = self.args.get("detail", "")
        m = re.search(r"PIPE:(\w+)", detail)
        return m.group(1) if m else None

    def is_drain_op(self) -> bool:
        return self.name in DRAIN_OPS


class BubbleCategory(Enum):
    NORMAL = "normal"
    STRUCTURAL = "structural"
    DATA_STALL = "data_stall"
    SCALAR_OVERHEAD = "scalar_overhead"
    RESOURCE_CONTENTION = "resource_contention"
    CROSS_CORE = "cross_core"
    CUBE_VECTOR = "cube_vector"


class BubbleSubType(Enum):
    N_ISSUE_GAP = "n_issue_gap"
    S_DRAIN = "s_drain"
    S_BARRIER = "s_barrier"
    S_COLD_START = "s_cold_start"
    S_TAIL_DRAIN = "s_tail_drain"
    S_ICACHE_MISS = "s_icache_miss"
    S_FLOWCTRL = "s_flowctrl"
    D_MTE2_WAIT = "d_mte2_wait"
    D_MTE2_IMPLICIT = "d_mte2_implicit"
    D_MTE3_WAIT = "d_mte3_wait"
    D_MTE3_IMPLICIT = "d_mte3_implicit"
    D_MTE2_UNDERSIZE = "d_mte2_undersize"
    D_MTE3_UNDERSIZE = "d_mte3_undersize"
    D_NO_OVERLAP = "d_no_overlap"
    D_PARTIAL_OVERLAP = "d_partial_overlap"
    SC_LDST_BLOCK = "sc_ldst_block"
    SC_COMPUTE_BLOCK = "sc_compute_block"
    SC_TILING_COMPLEX = "sc_tiling_complex"
    R_UB_PRESSURE = "r_ub_pressure"
    R_ICACHE_THRASH = "r_icache_thrash"
    R_BUS_CONTENTION = "r_bus_contention"
    X_TILING_IMBALANCE = "x_tiling_imbalance"
    X_TAIL_CORE = "x_tail_core"
    X_SYNC_BARRIER = "x_sync_barrier"
    CV_CUBE_WAIT = "cv_cube_wait"
    CV_VECTOR_WAIT = "cv_vector_wait"
    CV_HANDOFF = "cv_handoff"


SUBTYPE_TO_CATEGORY = {
    BubbleSubType.N_ISSUE_GAP: BubbleCategory.NORMAL,
    BubbleSubType.S_DRAIN: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_BARRIER: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_COLD_START: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_TAIL_DRAIN: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_ICACHE_MISS: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_FLOWCTRL: BubbleCategory.STRUCTURAL,
    BubbleSubType.SC_LDST_BLOCK: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.SC_COMPUTE_BLOCK: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.SC_TILING_COMPLEX: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.D_MTE2_WAIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE2_IMPLICIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_WAIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_IMPLICIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE2_UNDERSIZE: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_UNDERSIZE: BubbleCategory.DATA_STALL,
    BubbleSubType.D_NO_OVERLAP: BubbleCategory.DATA_STALL,
    BubbleSubType.D_PARTIAL_OVERLAP: BubbleCategory.DATA_STALL,
    BubbleSubType.R_UB_PRESSURE: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.R_ICACHE_THRASH: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.R_BUS_CONTENTION: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.X_TILING_IMBALANCE: BubbleCategory.CROSS_CORE,
    BubbleSubType.X_TAIL_CORE: BubbleCategory.CROSS_CORE,
    BubbleSubType.X_SYNC_BARRIER: BubbleCategory.CROSS_CORE,
    BubbleSubType.CV_CUBE_WAIT: BubbleCategory.CUBE_VECTOR,
    BubbleSubType.CV_VECTOR_WAIT: BubbleCategory.CUBE_VECTOR,
    BubbleSubType.CV_HANDOFF: BubbleCategory.CUBE_VECTOR,
}


@dataclass
class ConcurrentState:
    scalar_coverage: float = 0.0
    scalarldst_coverage: float = 0.0
    mte2_coverage: float = 0.0
    mte3_coverage: float = 0.0
    mte1_coverage: float = 0.0
    cube_coverage: float = 0.0
    fixpipe_coverage: float = 0.0
    cachemiss_active: bool = False
    flowctrl_active: bool = False

    @property
    def scalar_busy(self) -> bool:
        return self.scalar_coverage > 0.05

    @property
    def scalarldst_busy(self) -> bool:
        return self.scalarldst_coverage > 0.05

    @property
    def mte2_busy(self) -> bool:
        return self.mte2_coverage > 0.05

    @property
    def mte3_busy(self) -> bool:
        return self.mte3_coverage > 0.05

    @property
    def mte1_busy(self) -> bool:
        return self.mte1_coverage > 0.05

    @property
    def cube_busy(self) -> bool:
        return self.cube_coverage > 0.05

    @property
    def fixpipe_busy(self) -> bool:
        return self.fixpipe_coverage > 0.05

    @property
    def all_idle(self) -> bool:
        return not (self.scalar_busy or self.scalarldst_busy
                    or self.mte2_busy or self.mte3_busy
                    or self.mte1_busy or self.cube_busy
                    or self.fixpipe_busy)


@dataclass
class BubbleResult:
    sub_type: str
    category: str
    gap_ps: float
    gap_us: float
    reason: str
    optimizable: bool
    optimization_hint: Optional[str] = None
    waited_pipeline: Optional[str] = None
    concurrent_cause: Optional[str] = None
    iteration_index: Optional[int] = None


@dataclass
class CoreReport:
    core_id: str
    core_type: str
    total_duration_us: float
    main_pipeline: str
    bubble_count: int
    bubbles: list[BubbleResult] = field(default_factory=list)
    subtype_summary: dict = field(default_factory=dict)
    top_bottleneck: Optional[str] = None


@dataclass
class GapContext:
    gap_ps: float
    op_before: Optional[TraceEvent]
    op_after: Optional[TraceEvent]
    cs: ConcurrentState
    has_wait_flag: bool = False
    wait_flag_event: Optional[TraceEvent] = None
    iteration_index: Optional[int] = None
    total_iterations: Optional[int] = None


# ── Trace 解析 ─────────────────────────────────────────────

def _parse_event(raw: dict) -> Optional[TraceEvent]:
    ph = raw.get("ph", "")
    if ph != "X":
        return None
    return TraceEvent(
        name=raw.get("name", ""),
        ph=ph,
        pid=raw.get("pid", 0),
        tid=raw.get("tid", 0),
        ts=float(raw.get("ts", 0)),
        dur=float(raw.get("dur", 0)),
        args=raw.get("args", {}),
        cname=raw.get("cname"),
    )


def _build_pid_map_from_metadata(raw_events: list[dict]) -> dict[int, int]:
    """Read ph='M' metadata events to build dynamic PID -> standard PID mapping."""
    pid_map = {}
    for raw in raw_events:
        if raw.get("ph") != "M" or raw.get("name") != "process_name":
            continue
        trace_pid = raw.get("pid", -1)
        pname = raw.get("args", {}).get("name", "")
        suffix = re.sub(r'^\d+_', '', pname).upper()

        # cannsim format: names like 0_RVECEX, 1_RVECSU, FIXP, etc.
        if suffix in _CANNSIM_NAME_REMAP:
            pid_map[trace_pid] = _CANNSIM_NAME_REMAP[suffix]
        elif trace_pid in _CANNSIM_PID_REMAP:
            pid_map[trace_pid] = _CANNSIM_PID_REMAP[trace_pid]
        # msprof / standard format: names like SCALAR, VECTOR, MTE2, etc.
        elif suffix in PIPELINE_NAME_MAP:
            pid_map[trace_pid] = PIPELINE_NAME_MAP[suffix]
        # Direct numeric match to standard map (msprof already uses standard PIDs)
        elif trace_pid in PIPELINE_PID_MAP:
            pid_map[trace_pid] = trace_pid
    return pid_map


def load_core_trace(trace_path: str) -> dict[int, list[TraceEvent]]:
    with open(trace_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_events = data
    else:
        raw_events = data.get("traceEvents", [])

    pid_map = _build_pid_map_from_metadata(raw_events)

    pipelines: dict[int, list[TraceEvent]] = {}
    for raw in raw_events:
        evt = _parse_event(raw)
        if evt is None:
            continue
        if pid_map and evt.pid in pid_map:
            evt.pid = pid_map[evt.pid]
        pipelines.setdefault(evt.pid, []).append(evt)

    for pid in pipelines:
        pipelines[pid].sort(key=lambda e: e.ts)

    return pipelines


# ── 路径扫描 ───────────────────────────────────────────────

def _should_include_dir(entry: str, core_type: str) -> bool:
    if not _DIR_PATTERN.match(entry):
        return False
    if core_type == "veccore":
        return ".veccore" in entry
    if core_type == "cubecore":
        return ".cubecore" in entry
    return True


def _try_add_flat_entry(
    result: dict[str, str], entry_path: str, entry: str, core_type: str,
) -> None:
    if os.path.isdir(entry_path) and _should_include_dir(entry, core_type):
        trace_path = os.path.join(entry_path, "trace.json")
        if os.path.isfile(trace_path):
            result[entry] = trace_path
    elif os.path.isfile(entry_path):
        m = re.match(r"^trace_(core\d+)\.json$", entry)
        if m:
            cid = m.group(1)
            if core_type == "veccore" and "veccore" not in cid:
                pass
            result[cid] = entry_path


def _scan_flat_directory(simulator_dir: str, core_type: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in os.listdir(simulator_dir):
        _try_add_flat_entry(result, os.path.join(simulator_dir, entry), entry, core_type)
    return result


def _scan_deep_msprof(simulator_dir: str, core_type: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for root, dirs, _ in os.walk(simulator_dir):
        if os.path.basename(root) != "simulator":
            continue
        for d in dirs:
            if not _should_include_dir(d, core_type):
                continue
            trace_path = os.path.join(root, d, "trace.json")
            if os.path.isfile(trace_path):
                result[d] = trace_path
        break
    return result


def _scan_deep_cannsim(simulator_dir: str, core_type: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for root, _, files in os.walk(simulator_dir):
        if os.path.basename(root) != "report":
            continue
        for f in files:
            m = re.match(r"^trace_(core\d+)\.json$", f)
            if m:
                cid = m.group(1)
                result[cid] = os.path.join(root, f)
        break
    return result


def get_all_core_paths(simulator_dir: str, core_type: str = "all") -> dict[str, str]:
    if not os.path.isdir(simulator_dir):
        return {}
    result = _scan_flat_directory(simulator_dir, core_type)
    if result:
        return result
    result = _scan_deep_msprof(simulator_dir, core_type)
    if result:
        return result
    return _scan_deep_cannsim(simulator_dir, core_type)


# ── 覆盖计算 ───────────────────────────────────────────────

def compute_core_duration(pipelines: dict[int, list[TraceEvent]]) -> float:
    min_ts = float("inf")
    max_end = 0.0
    for events in pipelines.values():
        for e in events:
            if e.ts < min_ts:
                min_ts = e.ts
            end = e.ts + e.dur
            if end > max_end:
                max_end = end
    return max_end - min_ts if min_ts != float("inf") else 0.0


def compute_pipeline_coverage(
    pipelines: dict[int, list[TraceEvent]], pid: int,
    window_start: float, window_end: float,
) -> float:
    window_dur = window_end - window_start
    if window_dur <= 0:
        return 0.0
    events = pipelines.get(pid, [])
    covered = 0.0
    for e in events:
        if e.ts >= window_end:
            break
        if e.end_ts <= window_start:
            continue
        overlap_start = max(e.ts, window_start)
        overlap_end = min(e.end_ts, window_end)
        covered += overlap_end - overlap_start
    return min(covered / window_dur, 1.0)


def has_event_in_window(
    pipelines: dict[int, list[TraceEvent]],
    pid: int, window_start: float, window_end: float,
) -> bool:
    for e in pipelines.get(pid, []):
        if e.ts >= window_end:
            break
        if e.end_ts > window_start:
            return True
    return False


def find_concurrent_events(
    pipelines: dict[int, list[TraceEvent]],
    ts: float, end_ts: float, exclude_pid: Optional[int] = None,
) -> dict[int, list[TraceEvent]]:
    result: dict[int, list[TraceEvent]] = {}
    for pid, events in pipelines.items():
        if pid == exclude_pid:
            continue
        overlapping = []
        for e in events:
            if e.ts < end_ts and e.end_ts > ts:
                overlapping.append(e)
            elif e.ts >= end_ts:
                break
        if overlapping:
            result[pid] = overlapping
    return result


# ── 并发状态 ───────────────────────────────────────────────

def build_concurrent_state(
    pipelines: dict[int, list[TraceEvent]],
    window_start: float, window_end: float, core_type: str,
) -> ConcurrentState:
    state = ConcurrentState()
    state.scalar_coverage = compute_pipeline_coverage(
        pipelines, PID_SCALAR, window_start, window_end)
    state.scalarldst_coverage = compute_pipeline_coverage(
        pipelines, PID_SCALARLDST, window_start, window_end)
    state.mte2_coverage = compute_pipeline_coverage(
        pipelines, PID_MTE2, window_start, window_end)
    state.cachemiss_active = has_event_in_window(
        pipelines, PID_CACHEMISS, window_start, window_end)
    state.flowctrl_active = has_event_in_window(
        pipelines, PID_FLOWCTRL, window_start, window_end)

    if core_type == "veccore":
        state.mte3_coverage = compute_pipeline_coverage(
            pipelines, PID_MTE3, window_start, window_end)
    elif core_type == "cubecore":
        state.mte1_coverage = compute_pipeline_coverage(
            pipelines, PID_MTE1, window_start, window_end)
        state.cube_coverage = compute_pipeline_coverage(
            pipelines, PID_CUBE, window_start, window_end)
        state.fixpipe_coverage = compute_pipeline_coverage(
            pipelines, PID_FIXPIPE, window_start, window_end)

    return state


# ── 空泡分类辅助函数 ───────────────────────────────────────

def _check_threshold_a(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.gap_ps > THRESHOLD_A_MAX_PS:
        return None
    return BubbleResult("n_issue_gap", "normal", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "Normal issue gap", False)


def _check_threshold_b(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.gap_ps > THRESHOLD_B_DRAIN_PS:
        return None
    if ctx.op_before and ctx.op_before.is_drain_op():
        if ctx.op_before.name in BARRIER_OPS:
            return BubbleResult("s_barrier", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                                f"PipeBarrier sync after {ctx.op_before.name}", False)
        return BubbleResult("s_drain", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Pipeline drain after {ctx.op_before.name}", False)
    return BubbleResult("n_issue_gap", "normal", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "Small gap, non-drain", False)


def _check_threshold_c_common(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.gap_ps > THRESHOLD_C_MAX_PS:
        return None
    if ctx.cs.cachemiss_active:
        return BubbleResult("s_icache_miss", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "Instruction cache miss detected", False)
    if ctx.cs.flowctrl_active:
        return BubbleResult("s_flowctrl", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "Flow control instruction overhead", False)
    if ctx.cs.scalarldst_coverage > 0.3:
        return BubbleResult("sc_ldst_block", "scalar_overhead", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"SCALARLDST busy (coverage={ctx.cs.scalarldst_coverage:.0%})", True,
                            "Parameter prefetch: preload next tile params during compute", "SCALARLDST")
    if ctx.cs.scalar_coverage > 0.3:
        return BubbleResult("sc_compute_block", "scalar_overhead", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"SCALAR busy (coverage={ctx.cs.scalar_coverage:.0%})", True,
                            "Reduce address computation: simplify tiling or precompute offsets", "SCALAR")
    if ctx.cs.all_idle:
        return BubbleResult("s_drain", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "Extended drain / all pipelines idle", False)
    return None


def _check_threshold_c_vec(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.gap_ps > THRESHOLD_C_MAX_PS:
        return None
    if ctx.cs.mte2_coverage > 0.3:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"MTE2 busy in mid-range gap (coverage={ctx.cs.mte2_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE2 with VEC", "MTE2")
    if ctx.cs.mte3_coverage > 0.3:
        return BubbleResult("d_mte3_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"MTE3 busy in mid-range gap (coverage={ctx.cs.mte3_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE3 with VEC", "MTE3")
    return BubbleResult("s_drain", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "Mid-range gap, non-scalar cause", False)


def _check_threshold_c_cube(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.gap_ps > THRESHOLD_C_MAX_PS:
        return None
    if ctx.cs.fixpipe_coverage > 0.3:
        return BubbleResult("cv_handoff", "cube_vector", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"FIXPIPE busy (coverage={ctx.cs.fixpipe_coverage:.0%}), L0C->UB conversion",
                            False, None, "FIXPIPE")
    if ctx.cs.mte1_coverage > 0.3:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"MTE1 busy (L1->L0 load, coverage={ctx.cs.mte1_coverage:.0%})", True,
                            "Overlap MTE1 data load with CUBE compute via double buffering", "MTE1")
    if ctx.cs.mte2_coverage > 0.3:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"MTE2 busy (GM->L1 load, coverage={ctx.cs.mte2_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE2 GM->L1 with compute", "MTE2")
    return BubbleResult("s_drain", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "Mid-range gap, non-scalar cause", False)


def _check_iteration_boundary(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.iteration_index is None or ctx.total_iterations is None:
        return None
    if ctx.iteration_index == 0 and ctx.total_iterations > 1:
        return BubbleResult("s_cold_start", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "First tile pipeline fill (cold start)", False)
    if ctx.iteration_index == ctx.total_iterations - 1 and ctx.total_iterations > 1:
        return BubbleResult("s_tail_drain", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "Last tile pipeline drain (tail)", False)
    return None


def _check_wait_flag_vec(ctx: GapContext) -> Optional[BubbleResult]:
    if not ctx.has_wait_flag or not ctx.wait_flag_event:
        return None
    waited = ctx.wait_flag_event.waited_pipeline()
    if waited == "MTE2" or (waited is None and ctx.cs.mte2_busy):
        return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for MTE2 (data load stall)", True,
                            "Enable double buffering to overlap MTE2 with VEC", "MTE2")
    if waited == "MTE3" or (waited is None and ctx.cs.mte3_busy):
        return BubbleResult("d_mte3_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for MTE3 (write-back stall)", True,
                            "Enable double buffering to overlap MTE3 with VEC", "MTE3")
    if waited == "SCALAR":
        return BubbleResult("sc_compute_block", "scalar_overhead", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for SCALAR", True,
                            "Reduce scalar dependency: simplify control flow or precompute", "SCALAR")
    return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                        f"WAIT_FLAG for {waited or 'unknown'}", True)


def _check_wait_flag_cube(ctx: GapContext) -> Optional[BubbleResult]:
    if not ctx.has_wait_flag or not ctx.wait_flag_event:
        return None
    waited = ctx.wait_flag_event.waited_pipeline()
    if waited == "MTE1" or (waited is None and ctx.cs.mte1_busy):
        return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for MTE1 (L1->L0 data load stall)", True,
                            "Enable double buffering to overlap MTE1 with CUBE", "MTE1")
    if waited == "MTE2" or (waited is None and ctx.cs.mte2_busy):
        return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for MTE2 (GM->L1 data load stall)", True,
                            "Enable double buffering to overlap MTE2 with CUBE", "MTE2")
    if waited == "FIXPIPE" or (waited is None and ctx.cs.fixpipe_busy):
        return BubbleResult("cv_handoff", "cube_vector", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for FIXPIPE (L0C->UB conversion)", False,
                            None, "FIXPIPE")
    if waited == "SCALAR":
        return BubbleResult("sc_compute_block", "scalar_overhead", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "WAIT_FLAG for SCALAR", True,
                            "Reduce scalar dependency: simplify control flow or precompute", "SCALAR")
    return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                        f"WAIT_FLAG for {waited or 'unknown'}", True)


def _check_wait_flag_unavailable(ctx: GapContext) -> Optional[BubbleResult]:
    if not ctx.has_wait_flag:
        return None
    return BubbleResult("d_mte2_wait", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "WAIT_FLAG (detail unavailable)", True,
                        "Enable double buffering to overlap data transfer with compute")


def _check_implicit_vec(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.cs.mte2_coverage > 0.5:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Implicit MTE2 stall (coverage={ctx.cs.mte2_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE2 with VEC", "MTE2")
    if ctx.cs.mte3_coverage > 0.5:
        return BubbleResult("d_mte3_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Implicit MTE3 stall (coverage={ctx.cs.mte3_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE3 with VEC", "MTE3")
    return None


def _check_implicit_cube(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.cs.mte1_coverage > 0.5:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Implicit MTE1 stall (L1->L0, coverage={ctx.cs.mte1_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE1 with CUBE", "MTE1")
    if ctx.cs.mte2_coverage > 0.5:
        return BubbleResult("d_mte2_implicit", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Implicit MTE2 stall (GM->L1, coverage={ctx.cs.mte2_coverage:.0%})", True,
                            "Enable double buffering to overlap MTE2 with CUBE", "MTE2")
    if ctx.cs.fixpipe_coverage > 0.5:
        return BubbleResult("cv_handoff", "cube_vector", ctx.gap_ps, ctx.gap_ps / 1e6,
                            f"Implicit FIXPIPE stall (coverage={ctx.cs.fixpipe_coverage:.0%})", False,
                            None, "FIXPIPE")
    return None


def _check_all_idle_cachemiss(ctx: GapContext) -> Optional[BubbleResult]:
    if ctx.cs.all_idle and ctx.cs.cachemiss_active:
        return BubbleResult("s_icache_miss", "structural", ctx.gap_ps, ctx.gap_ps / 1e6,
                            "All idle + icache miss", False)
    return None


def _default_result(ctx: GapContext) -> BubbleResult:
    return BubbleResult("d_no_overlap", "data_stall", ctx.gap_ps, ctx.gap_ps / 1e6,
                        "Large gap without explicit WAIT_FLAG (implicit sync, no overlap)", True,
                        "Investigate pipeline dependency for root cause")


# ── 空泡分类主函数 ─────────────────────────────────────────

def classify_gap(ctx: GapContext, core_type: str) -> BubbleResult:
    if core_type == "cubecore":
        return _classify_gap_cube(ctx)
    return _classify_gap_vec(ctx)


def _classify_gap_vec(ctx: GapContext) -> BubbleResult:
    for checker in (
        _check_threshold_a, _check_threshold_b,
        _check_threshold_c_common, _check_threshold_c_vec,
        _check_iteration_boundary, _check_wait_flag_vec,
        _check_wait_flag_unavailable, _check_implicit_vec,
        _check_all_idle_cachemiss,
    ):
        result = checker(ctx)
        if result is not None:
            return result
    return _default_result(ctx)


def _classify_gap_cube(ctx: GapContext) -> BubbleResult:
    for checker in (
        _check_threshold_a, _check_threshold_b,
        _check_threshold_c_common, _check_threshold_c_cube,
        _check_iteration_boundary, _check_wait_flag_cube,
        _check_wait_flag_unavailable, _check_implicit_cube,
        _check_all_idle_cachemiss,
    ):
        result = checker(ctx)
        if result is not None:
            return result
    return _default_result(ctx)


# ── 分析主流程 ─────────────────────────────────────────────

def _extract_wait_flags(
    concurrent: dict[int, list[TraceEvent]],
) -> tuple[bool, Optional[TraceEvent]]:
    wait_flag_events = []
    for _, events in concurrent.items():
        for e in events:
            if e.is_wait_flag():
                wait_flag_events.append(e)
    has_wait_flag = len(wait_flag_events) > 0
    wait_flag_event = wait_flag_events[0] if wait_flag_events else None
    return has_wait_flag, wait_flag_event


def _analyze_bubbles(
    main_events: list[TraceEvent],
    pipelines: dict[int, list[TraceEvent]],
    core_type: str,
    total_iterations: int,
) -> list[BubbleResult]:
    bubbles: list[BubbleResult] = []
    main_pid = PID_CUBE if core_type == "cubecore" else PID_VECTOR
    for i in range(len(main_events) - 1):
        current = main_events[i]
        next_evt = main_events[i + 1]
        gap_ps = next_evt.ts - current.end_ts
        if gap_ps <= 0:
            continue

        window_start = current.end_ts
        window_end = next_evt.ts
        concurrent = find_concurrent_events(
            pipelines, window_start, window_end, exclude_pid=main_pid)
        has_wait_flag, wait_flag_event = _extract_wait_flags(concurrent)
        cs = build_concurrent_state(pipelines, window_start, window_end, core_type)
        iteration_index = int((i / max(1, len(main_events) - 1)) * total_iterations)

        ctx = GapContext(
            gap_ps=gap_ps,
            op_before=current,
            op_after=next_evt,
            cs=cs,
            has_wait_flag=has_wait_flag,
            wait_flag_event=wait_flag_event,
            iteration_index=iteration_index,
            total_iterations=total_iterations,
        )
        result = classify_gap(ctx, core_type)
        bubbles.append(result)
    return bubbles


def _build_core_report(
    core_id: str, core_type: str, main_pipeline_name: str,
    total_duration: float, bubbles: list[BubbleResult],
) -> CoreReport:
    subtype_counts: dict[str, int] = {}
    category_durations: dict[str, float] = {}
    for b in bubbles:
        subtype_counts[b.sub_type] = subtype_counts.get(b.sub_type, 0) + 1
        category_durations[b.category] = category_durations.get(b.category, 0) + b.gap_us

    top_bottleneck = None
    if category_durations:
        top_bottleneck = max(category_durations, key=category_durations.get)

    return CoreReport(
        core_id=core_id,
        core_type=core_type,
        total_duration_us=total_duration / 1e6,
        main_pipeline=main_pipeline_name,
        bubble_count=len(bubbles),
        bubbles=bubbles,
        subtype_summary=subtype_counts,
        top_bottleneck=top_bottleneck,
    )


def analyze_core(trace_path: str, core_id: str) -> CoreReport:
    pipelines = load_core_trace(trace_path)
    core_type = "cubecore" if ".cubecore" in core_id else "veccore"
    main_pid = PID_CUBE if core_type == "cubecore" else PID_VECTOR
    main_pipeline_name = "CUBE" if core_type == "cubecore" else "VECTOR"

    main_events = pipelines.get(main_pid, [])
    if not main_events:
        return CoreReport(core_id, core_type, compute_core_duration(pipelines) / 1e6,
                          main_pipeline_name, 0, [], {}, "no_main_pipeline_events")

    total_duration = compute_core_duration(pipelines)
    total_iterations = max(1, len(main_events) // 2)
    bubbles = _analyze_bubbles(main_events, pipelines, core_type, total_iterations)

    return _build_core_report(core_id, core_type, main_pipeline_name,
                              total_duration, bubbles)


# ── 输出 ───────────────────────────────────────────────────

def _format_single_report(report: CoreReport) -> str:
    lines = [
        f"\n{'='*60}",
        f"Core: {report.core_id} ({report.core_type})",
        f"Main Pipeline: {report.main_pipeline}",
        f"Total Duration: {report.total_duration_us:.3f} us",
        f"Bubbles Found: {report.bubble_count}",
    ]
    if report.top_bottleneck:
        lines.append(f"Top Bottleneck Category: {report.top_bottleneck}")

    if report.subtype_summary:
        lines.append("\n  Subtype Distribution:")
        for subtype, count in sorted(report.subtype_summary.items(), key=lambda x: -x[1]):
            lines.append(f"    {subtype}: {count}")

    sorted_bubbles = sorted(report.bubbles, key=lambda b: -b.gap_us)[:5]
    if sorted_bubbles:
        lines.append("\n  Top 5 Longest Bubbles:")
        for b in sorted_bubbles:
            opt_hint = f" -> {b.optimization_hint}" if b.optimization_hint else ""
            lines.append(f"    {b.sub_type}: {b.gap_us:.3f} us | {b.reason}{opt_hint}")
    return "\n".join(lines)


def print_report(report: CoreReport):
    LOGGER.info(_format_single_report(report))


def to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


# ── 主函数 ─────────────────────────────────────────────────

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulator Trace Bubble Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trace_core0.json
  %(prog)s ./simulator_output/
  %(prog)s ./simulator_output/ --json -o report.json
        """,
    )
    parser.add_argument("input", help="Trace JSON file or simulator output directory")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON format")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--core-type", "-t", choices=["all", "veccore", "cubecore"],
                        default="all", help="Filter core type")
    return parser


class TraceInputError(Exception):
    """Raised when the input path is invalid or contains no trace files."""


def _resolve_input(input_path: str, core_type: str) -> list[CoreReport]:
    if os.path.isfile(input_path):
        fname = os.path.basename(input_path)
        m = re.match(r"^trace_(core\d+)\.json$", fname)
        if m:
            core_id = m.group(1)
        else:
            core_id = os.path.basename(os.path.dirname(input_path)) or "unknown"
        return [analyze_core(input_path, core_id)]

    if os.path.isdir(input_path):
        core_paths = get_all_core_paths(input_path, core_type)
        if not core_paths:
            raise TraceInputError(f"No core traces found in {input_path}")
        return [analyze_core(trace_path, core_id)
                for core_id, trace_path in sorted(core_paths.items())]

    raise TraceInputError(f"{input_path} is not a file or directory")


def _format_reports(reports: list[CoreReport], json_mode: bool) -> str:
    if json_mode:
        return json.dumps([to_dict(r) for r in reports], indent=2, ensure_ascii=False)
    return "\n".join(_format_single_report(r) for r in reports)


def _write_output(output: str, output_path: Optional[str]) -> None:
    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        LOGGER.info("Report written to %s", output_path)
    else:
        LOGGER.info(output)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_argument_parser()
    args = parser.parse_args()

    try:
        reports = _resolve_input(args.input, args.core_type)
    except TraceInputError as exc:
        LOGGER.error("Error: %s", exc)
        sys.exit(1)

    output = _format_reports(reports, args.json)
    _write_output(output, args.output)


if __name__ == "__main__":
    main()
