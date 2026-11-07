#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Memory Snapshot 深度分析脚本
# 用于解析 PyTorch on NPU 训练产出的 memory snapshot (.pickle) 文件，提取关键内存指标。
#
# 参考: https://docs.pytorch.org/tutorials/beginner/mosaic_memory_profiling_tutorial.html
#
# 用法:
#   # 单快照分析
#   python analyze_snapshot.py <snapshot.pickle>
#
#   # 双快照对比
#   python analyze_snapshot.py <baseline.pickle> <modified.pickle>
#
#   # JSON 结构化输出
#   python analyze_snapshot.py <snapshot.pickle> --json
#
#   # 自定义 Top-N
#   python analyze_snapshot.py <snapshot.pickle> --top-n 20
#
# 输出:
#   - 设备内存信息
#   - 内存分配概览（Reserved / Allocated / Active / 碎片化率）
#   - 按语义类别分类的内存占比（激活 / 梯度 / 优化器 / 参数 / 通信 / 其他）
#   - 峰值内存分析（峰值时刻定位 + Top-N 调用栈）
#   - 时间线分析（内存增长趋势 + 阶段识别 + 泄漏检测）
#   - 碎片化深度分析（碎片分布 / 最大空闲块 / 小块占比）
#   - 双快照差异对比（可选）

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# 语义分类器
# ============================================================

# 分类优先级从高到低，命中即停
_CATEGORY_RULES = [
    # (类别名, 匹配函数名关键字列表, 匹配文件名关键字列表)
    # 优化器状态
    (
        "optimizer_state",
        {
            "names": [
                "adam",
                "adamw",
                "sgd",
                "optimizer",
                "zero_grad",
                "zeros_like",
                "step",
                "exp_avg",
                "exp_avg_sq",
                "momentum_buffer",
            ],
            "files": [
                "optim/adam",
                "optim/adamw",
                "optim/sgd",
                "optim/optimizer",
                "swap_optimizer",
            ],
        },
    ),
    # 梯度（backward pass 产生的梯度张量）
    (
        "gradient",
        {
            "names": [
                "backward",
                "autograd",
                "grad",
                "AccumulateGrad",
                "AddmmBackward",
                "MmBackward",
                "mm_mat1_backward",
                "mm_mat2_backward",
                "BmmBackward",
                "NativeLayerNormBackward",
                "SoftmaxBackward",
                "native_batch_norm_backward",
            ],
            "files": ["autograd/engine", "autograd/function"],
        },
    ),
    # 激活（前向传播保存用于反向的中间张量）
    (
        "activation",
        {
            "names": [
                "save_for_backward",
                "saved_tensors",
                "checkpoint",
                "unpack",
                "ctx",
            ],
            "files": ["torch/utils/checkpoint", "activation_checkpoint"],
        },
    ),
    # 通信缓冲（含异步通信和 DTensor 重分布）
    (
        "communication",
        {
            "names": [
                "all_reduce",
                "all_gather",
                "reduce_scatter",
                "all_to_all",
                "broadcast",
                "nccl",
                "hccl",
                "ncclx",
                "process_group",
                "_fsdp",
                "dist.",
                # 异步通信原语（泄漏高发区）
                "_c10d_functional",
                "wait_tensor",
                "async_op",
                # DTensor 重分布
                "redistribute",
                "DTensor",
                "_local_tensor",
                "placements",
                # 并行策略输出处理
                "RowwiseParallel",
                "ColwiseParallel",
                "_prepare_output_fn",
            ],
            "files": [
                "distributed/",
                "c10d/",
                "fsdp/",
                "dtensor/",
                "nccl",
                "hccl",
                "_c10d_functional",
            ],
        },
    ),
    # 参数（模型权重相关）
    (
        "parameter",
        {
            "names": [
                "weight",
                "bias",
                "embedding",
                "parameter",
                "to_empty",
                "materialize",
            ],
            "files": ["nn/modules/", "nn/parameter"],
        },
    ),
    # Loss 计算
    (
        "loss",
        {
            "names": ["loss", "cross_entropy", "nll_loss", "mse_loss", "kl_div"],
            "files": ["nn/functional", "nn/modules/loss"],
        },
    ),
    # 注意力层
    (
        "attention",
        {
            "names": [
                "attention",
                "mha",
                "mla",
                "sdpa",
                "scaled_dot_product",
                "flash_attn",
                "multi_head_attention",
            ],
            "files": [],
        },
    ),
    # MoE 层
    (
        "moe",
        {
            "names": [
                "moe",
                "expert",
                "grouped_mm",
                "topk_softmax",
                "permute",
                "unpermute",
                "router",
                "gate",
            ],
            "files": ["moe/"],
        },
    ),
    # 归一化层
    (
        "normalization",
        {
            "names": ["norm", "layer_norm", "rms_norm", "batch_norm", "group_norm"],
            "files": [],
        },
    ),
    # 线性层 / 矩阵乘
    (
        "linear",
        {
            "names": ["linear", "matmul", "mm", "addmm", "bmm", "gemm"],
            "files": [],
        },
    ),
]


def _classify_by_frames(frames: list[dict]) -> str:
    """基于调用栈帧进行语义分类。

    按优先级检查每个分类规则，在整个调用栈中搜索匹配。
    """
    if not frames:
        return "unknown"

    # 收集所有帧的名称和文件名（小写）
    all_names = []
    all_files = []
    for f in frames:
        all_names.append(f.get("name", "").lower())
        all_files.append(f.get("filename", "").lower())

    for category, rules in _CATEGORY_RULES:
        for keyword in rules.get("names", []):
            kw = keyword.lower()
            if any(kw in name for name in all_names):
                return category
        for keyword in rules.get("files", []):
            kw = keyword.lower()
            if any(kw in fn for fn in all_files):
                return category

    return "other"


def _format_callstack(frames: list[dict], max_depth: int = 5) -> list[str]:
    """格式化调用栈，返回可读字符串列表。"""
    result = []
    for f in frames[:max_depth]:
        name = f.get("name", "?")
        filename = f.get("filename", "?")
        line = f.get("line", "?")
        result.append(f"  {name} ({filename}:{line})")
    if len(frames) > max_depth:
        result.append(f"  ... ({len(frames) - max_depth} more frames)")
    return result


# ============================================================
# 类别中文名映射
# ============================================================

_CATEGORY_NAMES_CN = {
    "optimizer_state": "优化器状态",
    "gradient": "梯度",
    "activation": "激活",
    "communication": "通信缓冲",
    "parameter": "参数/权重",
    "loss": "Loss 计算",
    "attention": "注意力层",
    "moe": "MoE 专家层",
    "normalization": "归一化层",
    "linear": "线性层/矩阵乘",
    "other": "其他",
    "unknown": "未知",
}


def _cn(cat: str) -> str:
    return _CATEGORY_NAMES_CN.get(cat, cat)


def _fmt_bytes(n: int | float) -> str:
    """将字节数转为可读格式（自动选择单位）。"""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


# ============================================================
# 数据类
# ============================================================


@dataclass
class SnapshotAnalysis:
    """memory snapshot 分析结果。"""

    # 设备信息
    device_total_memory: int = 0

    # 内存概览
    total_reserved: int = 0
    total_allocated: int = 0
    total_active: int = 0
    fragmentation_rate: float = 0.0

    # 按类别分类
    category_memory: dict[str, int] = field(default_factory=dict)

    # 峰值分析
    peak_allocated: int = 0
    peak_reserved: int = 0
    peak_category_breakdown: dict[str, int] = field(default_factory=dict)
    peak_top_allocations: list[dict] = field(default_factory=list)

    # 时间线分析
    timeline_events_count: int = 0
    alloc_count: int = 0
    free_count: int = 0
    phase_summary: list[dict] = field(default_factory=list)
    potential_leak: bool = False
    leak_details: str = ""

    # 碎片化深度
    total_segments: int = 0
    total_blocks: int = 0
    free_blocks: int = 0
    free_block_sizes: list[int] = field(default_factory=list)
    largest_free_block: int = 0
    small_block_count: int = 0  # < 1MB 的空闲块
    small_block_ratio: float = 0.0

    # Top-N 大块列表
    top_allocations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为可序列化字典。"""
        return {
            "device_total_memory_bytes": self.device_total_memory,
            "device_total_memory_readable": _fmt_bytes(self.device_total_memory),
            "memory_overview": self._build_memory_overview(),
            "category_breakdown": self._build_category_breakdown(),
            "peak_analysis": self._build_peak_analysis(),
            "timeline": self._build_timeline(),
            "fragmentation_detail": self._build_fragmentation_detail(),
        }

    def _build_peak_category_breakdown(self) -> dict:
        peak_category_breakdown = {}
        for category, size in sorted(self.peak_category_breakdown.items(), key=lambda x: -x[1]):
            peak_category_breakdown[category] = {
                "bytes": size,
                "readable": _fmt_bytes(size),
            }
        return peak_category_breakdown

    def _build_category_breakdown(self) -> dict:
        category_breakdown = {}
        total_allocated = max(self.total_allocated, 1)
        for category, size in sorted(self.category_memory.items(), key=lambda x: -x[1]):
            category_breakdown[category] = {
                "bytes": size,
                "readable": _fmt_bytes(size),
                "percentage": round(size / total_allocated * 100, 1),
            }
        return category_breakdown

    def _build_memory_overview(self) -> dict:
        return {
            "reserved_bytes": self.total_reserved,
            "reserved_readable": _fmt_bytes(self.total_reserved),
            "allocated_bytes": self.total_allocated,
            "allocated_readable": _fmt_bytes(self.total_allocated),
            "active_bytes": self.total_active,
            "active_readable": _fmt_bytes(self.total_active),
            "fragmentation_rate": round(self.fragmentation_rate, 2),
        }

    def _build_peak_analysis(self) -> dict:
        return {
            "peak_allocated_bytes": self.peak_allocated,
            "peak_allocated_readable": _fmt_bytes(self.peak_allocated),
            "peak_reserved_bytes": self.peak_reserved,
            "peak_reserved_readable": _fmt_bytes(self.peak_reserved),
            "peak_category_breakdown": self._build_peak_category_breakdown(),
            "top_allocations": self.peak_top_allocations[:10],
        }

    def _build_timeline(self) -> dict:
        return {
            "total_events": self.timeline_events_count,
            "alloc_count": self.alloc_count,
            "free_count": self.free_count,
            "potential_leak": self.potential_leak,
            "leak_details": self.leak_details,
            "phases": self.phase_summary,
        }

    def _build_fragmentation_detail(self) -> dict:
        return {
            "total_segments": self.total_segments,
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "largest_free_block_bytes": self.largest_free_block,
            "largest_free_block_readable": _fmt_bytes(self.largest_free_block),
            "small_block_count_under_1mb": self.small_block_count,
            "small_block_ratio": round(self.small_block_ratio, 2),
        }


# ============================================================
# 核心分析逻辑
# ============================================================


def _extract_device_info(snapshot: dict) -> int:
    """提取设备总内存。"""
    for device_traces in snapshot.get("device_traces", []):
        for trace in device_traces:
            if "device_total_memory" in trace:
                return trace["device_total_memory"]
    return 0


def _accumulate_segment_blocks(result: SnapshotAnalysis, segments: list) -> tuple[list, list]:
    """遍历段与块，累加统计并返回 (活跃块列表, 空闲块大小列表)。"""
    all_active_blocks = []
    free_block_sizes = []

    for seg in segments:
        result.total_reserved += seg.get("total_size", 0)
        result.total_allocated += seg.get("allocated_size", 0)
        result.total_active += seg.get("active_size", 0)

        for block in seg.get("blocks", []):
            result.total_blocks += 1
            state = block.get("state", "")
            size = block.get("size", 0)

            if state == "active_allocated":
                frames = block.get("frames", [])
                category = _classify_by_frames(frames)
                result.category_memory[category] = result.category_memory.get(category, 0) + size
                all_active_blocks.append(
                    {
                        "size": size,
                        "category": category,
                        "frames": frames,
                    }
                )
            elif state in ("inactive", "active_pending_free"):
                result.free_blocks += 1
                free_block_sizes.append(size)

    return all_active_blocks, free_block_sizes


def _compute_fragmentation(result: SnapshotAnalysis, free_block_sizes: list) -> None:
    """计算碎片化率与碎片化深度指标。"""
    # 碎片化率
    if result.total_reserved > 0:
        result.fragmentation_rate = (1 - result.total_allocated / result.total_reserved) * 100

    # 碎片化深度分析
    result.free_block_sizes = sorted(free_block_sizes, reverse=True)
    result.largest_free_block = result.free_block_sizes[0] if result.free_block_sizes else 0
    small_threshold = 1 * 1024 * 1024  # 1 MB
    result.small_block_count = sum(1 for s in free_block_sizes if s < small_threshold)
    if result.free_blocks > 0:
        result.small_block_ratio = result.small_block_count / result.free_blocks * 100


def _collect_top_allocations(result: SnapshotAnalysis, all_active_blocks: list, top_n: int) -> None:
    """按大小排序选出 Top-N 大块分配并记录调用栈。"""
    all_active_blocks.sort(key=lambda b: b["size"], reverse=True)
    for i, block in enumerate(all_active_blocks[:top_n]):
        stack = _format_callstack(block["frames"], max_depth=4)
        result.top_allocations.append(
            {
                "rank": i + 1,
                "size_bytes": block["size"],
                "size_readable": _fmt_bytes(block["size"]),
                "category": block["category"],
                "category_cn": _cn(block["category"]),
                "callstack": stack,
            }
        )


def _analyze_segments(snapshot: dict, top_n: int = 10) -> SnapshotAnalysis:
    """分析内存段（segments）和块（blocks）。"""
    result = SnapshotAnalysis()
    result.device_total_memory = _extract_device_info(snapshot)

    segments = snapshot.get("segments", [])
    result.total_segments = len(segments)

    all_active_blocks, free_block_sizes = _accumulate_segment_blocks(result, segments)
    _compute_fragmentation(result, free_block_sizes)
    _collect_top_allocations(result, all_active_blocks, top_n)

    return result


def _flatten_trace_events(device_traces: list) -> list:
    """展平所有 trace 事件，仅保留 alloc/free 相关动作。"""
    events = []
    for traces in device_traces:
        for trace in traces:
            action = trace.get("action", "")
            if action in ("alloc", "free_requested", "free_completed"):
                events.append(trace)
    return events


def _simulate_allocations(events: list) -> tuple[int, int, int, dict[int, dict]]:
    """按地址追踪模拟分配过程，返回 (alloc_count, free_count, peak, active_allocs)。"""
    current_allocated = 0
    peak_allocated = 0
    alloc_count = 0
    free_count = 0

    # 记录活跃分配（addr -> {size, frames, category}）
    active_allocs: dict[int, dict] = {}

    for event in events:
        action = event.get("action", "")
        addr = event.get("addr", 0)
        size = event.get("size", 0)
        frames = event.get("frames", [])

        if action == "alloc":
            alloc_count += 1
            current_allocated += size
            category = _classify_by_frames(frames)
            active_allocs[addr] = {
                "size": size,
                "frames": frames,
                "category": category,
            }
            if current_allocated > peak_allocated:
                peak_allocated = current_allocated
        elif action in ("free_requested", "free_completed"):
            free_count += 1
            if addr in active_allocs:
                current_allocated -= active_allocs[addr]["size"]
                del active_allocs[addr]

    return alloc_count, free_count, peak_allocated, active_allocs


def _collect_peak_top_allocations(active_allocs: dict[int, dict]) -> list[dict]:
    """从活跃分配中取最大项，作为峰值时刻 Top-N 分配。"""
    active_list = sorted(active_allocs.values(), key=lambda x: x["size"], reverse=True)
    peak_top = []
    for i, alloc in enumerate(active_list[:10]):
        peak_top.append(
            {
                "rank": i + 1,
                "size_bytes": alloc["size"],
                "size_readable": _fmt_bytes(alloc["size"]),
                "category": alloc["category"],
                "category_cn": _cn(alloc["category"]),
                "callstack": _format_callstack(alloc["frames"], max_depth=4),
            }
        )
    return peak_top


def _detect_leak(active_allocs: dict[int, dict], result: SnapshotAnalysis) -> None:
    """泄漏检测：未释放分配中 '其他/未知' 类别占比过高时标记潜在泄漏。"""
    # 泄漏检测：如果最终未释放的分配数量明显大于预期
    # （正常训练结束后，应该只剩参数、优化器状态等）
    unreleased = len(active_allocs)
    if unreleased > 0:
        unreleased_total = sum(a["size"] for a in active_allocs.values())
        # 按类别统计未释放
        unreleased_cats: dict[str, int] = defaultdict(int)
        for a in active_allocs.values():
            unreleased_cats[a["category"]] += a["size"]

        # 如果 "other" 或 "unknown" 类别占未释放总量 > 30%，可能存在泄漏
        suspicious = sum(v for k, v in unreleased_cats.items() if k in ("other", "unknown"))
        if unreleased_total > 0 and suspicious / unreleased_total > 0.3:
            result.potential_leak = True
            result.leak_details = (
                f"未释放分配 {unreleased} 个，总计 {_fmt_bytes(unreleased_total)}，"
                f"其中 '其他/未知' 类别占 {suspicious / unreleased_total * 100:.1f}%，"
                f"可能存在内存泄漏"
            )


def _identify_phases(events: list) -> list[dict]:
    """简单阶段识别（基于 alloc/free 模式），返回阶段摘要列表。"""
    # 简单阶段识别（基于 alloc/free 模式）
    # 将事件序列按照 1000 个一组分段，统计每段的 alloc vs free
    chunk_size = max(1, len(events) // 10)
    phases = []
    for i in range(0, len(events), chunk_size):
        chunk_end = i + chunk_size
        chunk = events[i:chunk_end]
        a = sum(1 for e in chunk if e.get("action") == "alloc")
        f = sum(1 for e in chunk if e.get("action") in ("free_requested", "free_completed"))
        net = sum(e.get("size", 0) if e.get("action") == "alloc" else -e.get("size", 0) for e in chunk)
        if a > f * 1.5:
            phase_type = "增长 (forward/加载)"
        elif f > a * 1.5:
            phase_type = "释放 (backward/清理)"
        else:
            phase_type = "稳定"
        phases.append(
            {
                "chunk_index": len(phases),
                "events": len(chunk),
                "allocs": a,
                "frees": f,
                "net_change_readable": _fmt_bytes(abs(net)),
                "net_direction": "+" if net > 0 else ("-" if net < 0 else "0"),
                "phase_type": phase_type,
            }
        )
    return phases


def _analyze_timeline(snapshot: dict, result: SnapshotAnalysis) -> None:
    """分析内存时间线（device_traces），定位峰值和检测泄漏。"""
    device_traces = snapshot.get("device_traces", [])
    if not device_traces:
        return

    events = _flatten_trace_events(device_traces)

    result.timeline_events_count = len(events)
    if not events:
        return

    alloc_count, free_count, peak_allocated, active_allocs = _simulate_allocations(events)

    result.alloc_count = alloc_count
    result.free_count = free_count
    result.peak_allocated = peak_allocated
    result.peak_reserved = result.total_reserved

    # 峰值时刻的类别分解：因为无法精确回溯峰值时刻的所有活跃分配，
    # 使用当前 segments 的分类作为近似（snapshot 采集时刻即快照点）
    result.peak_category_breakdown = dict(result.category_memory)

    # 峰值时刻的 Top-N 分配（使用当前活跃分配的最大项）
    result.peak_top_allocations.extend(_collect_peak_top_allocations(active_allocs))

    _detect_leak(active_allocs, result)

    result.phase_summary = _identify_phases(events)


# ============================================================
# 打印输出
# ============================================================


def _print_device_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印设备内存信息一节。"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}设备内存信息")
    logger.info(f"{'=' * 70}")
    if result.device_total_memory > 0:
        logger.info(f"  设备总内存: {_fmt_bytes(result.device_total_memory)}")
    else:
        logger.info("  设备总内存: 未知（snapshot 中未记录）")


def _print_overview_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印内存分配概览一节。"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}内存分配概览")
    logger.info(f"{'=' * 70}")
    logger.info(f"  保留内存 (Reserved):   {_fmt_bytes(result.total_reserved)}")
    logger.info(f"  已分配内存 (Allocated): {_fmt_bytes(result.total_allocated)}")
    logger.info(f"  活跃内存 (Active):     {_fmt_bytes(result.total_active)}")
    logger.info(f"  碎片化率:              {result.fragmentation_rate:.1f}%")

    # 利用率
    if result.device_total_memory > 0:
        usage_pct = result.total_reserved / result.device_total_memory * 100
        logger.info(f"  设备利用率:            {usage_pct:.1f}%")


def _print_category_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印内存占用分类一节。"""
    if not result.category_memory:
        return
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}内存占用分类（按语义类别）")
    logger.info(f"{'=' * 70}")
    total = max(sum(result.category_memory.values()), 1)
    logger.info(f"  {'类别':<16} {'大小':>12} {'占比':>8}")
    logger.info(f"  {'-' * 40}")
    for cat, size in sorted(result.category_memory.items(), key=lambda x: -x[1]):
        pct = size / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {_cn(cat):<14} {_fmt_bytes(size):>12} {pct:>6.1f}%  {bar}")


def _print_peak_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印峰值内存分析一节。"""
    if result.peak_allocated <= 0:
        return
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}峰值内存分析")
    logger.info(f"{'=' * 70}")
    logger.info(f"  峰值已分配内存: {_fmt_bytes(result.peak_allocated)}")
    logger.info(f"  峰值保留内存:   {_fmt_bytes(result.peak_reserved)}")

    if result.peak_category_breakdown:
        logger.info("\n  峰值时刻类别分布:")
        total_peak = max(sum(result.peak_category_breakdown.values()), 1)
        for cat, size in sorted(
            result.peak_category_breakdown.items(),
            key=lambda x: -x[1],
        ):
            pct = size / total_peak * 100
            logger.info(f"    {_cn(cat):<14} {_fmt_bytes(size):>12} ({pct:.1f}%)")

    if result.peak_top_allocations:
        logger.info("\n  峰值时刻 Top 分配:")
        for alloc in result.peak_top_allocations[:5]:
            logger.info(f"    [{alloc['rank']}] {alloc['size_readable']} ({alloc['category_cn']})")
            for line in alloc.get("callstack", []):
                logger.info(f"      {line}")


def _print_timeline_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印时间线分析一节。"""
    if result.timeline_events_count <= 0:
        return
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}时间线分析")
    logger.info(f"{'=' * 70}")
    logger.info(f"  总事件数: {result.timeline_events_count}")
    logger.info(f"  分配操作: {result.alloc_count}")
    logger.info(f"  释放操作: {result.free_count}")

    if result.potential_leak:
        logger.info(f"\n  ⚠️  潜在内存泄漏: {result.leak_details}")

    if result.phase_summary:
        logger.info("\n  阶段识别:")
        for phase in result.phase_summary:
            logger.info(
                f"    段 {phase['chunk_index']}: "
                f"{phase['phase_type']} "
                f"(allocs={phase['allocs']}, "
                f"frees={phase['frees']}, "
                f"net={phase['net_direction']}"
                f"{phase['net_change_readable']})"
            )


def _print_fragmentation_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印碎片化深度分析一节。"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}碎片化深度分析")
    logger.info(f"{'=' * 70}")
    logger.info(f"  总段数 (Segments): {result.total_segments}")
    logger.info(f"  总块数 (Blocks):   {result.total_blocks}")
    logger.info(f"  空闲块数:          {result.free_blocks}")
    logger.info(f"  最大空闲块:        {_fmt_bytes(result.largest_free_block)}")
    logger.info(f"  小块数 (<1MB):     {result.small_block_count}")
    logger.info(f"  小块占比:          {result.small_block_ratio:.1f}%")

    frag_level = "正常"
    if result.fragmentation_rate > 30:
        frag_level = "🔴 严重"
    elif result.fragmentation_rate > 20:
        frag_level = "🟡 较高"
    elif result.fragmentation_rate > 10:
        frag_level = "🟢 轻微"
    logger.info(f"  碎片化评估:        {frag_level}")

    if result.fragmentation_rate > 20:
        logger.info("\n  碎片化建议:")
        logger.info("    - 考虑调用 torch.npu.empty_cache() 清理碎片")
        logger.info("    - 检查是否存在频繁的小块分配/释放模式")
        if result.small_block_ratio > 50:
            logger.info(f"    - 大量小空闲块 ({result.small_block_ratio:.0f}%)，")
            logger.info("      可能需要调整分配策略或增大 batch 以减少碎片")


def _print_top_allocations_section(result: SnapshotAnalysis, prefix: str) -> None:
    """打印 Top-N 大块内存分配一节。"""
    if not result.top_allocations:
        return
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}Top-{len(result.top_allocations)} 大块内存分配")
    logger.info(f"{'=' * 70}")
    for alloc in result.top_allocations:
        logger.info(f"  [{alloc['rank']}] {alloc['size_readable']} ({alloc['category_cn']})")
        for line in alloc.get("callstack", []):
            logger.info(f"    {line}")


def _print_analysis(result: SnapshotAnalysis, label: str = "") -> None:
    """以可读格式打印分析结果。"""
    prefix = f"[{label}] " if label else ""

    _print_device_section(result, prefix)
    _print_overview_section(result, prefix)
    _print_category_section(result, prefix)
    _print_peak_section(result, prefix)
    _print_timeline_section(result, prefix)
    _print_fragmentation_section(result, prefix)
    _print_top_allocations_section(result, prefix)

    # 诊断建议
    _print_diagnosis(result, prefix)


def _category_suggestion(cat: str, pct: float) -> str | None:
    """根据单个类别占比生成诊断建议，无建议时返回 None。"""
    if cat == "activation" and pct > 40:
        return (
            f"激活内存占比 {pct:.1f}% (> 40%)，建议:\n"
            f"    - 启用 activation checkpoint: "
            f'[activation_checkpoint] mode = "full" 或 "selective"\n'
            f"    - 减小 batch_size 或 seq_len"
        )
    if cat == "optimizer_state" and pct > 40:
        return (
            f"优化器状态占比 {pct:.1f}% (> 40%)，建议:\n"
            f"    - 启用 swap_optimizer: "
            f"[optimizer] swap_optimizer = true\n"
            f"    - 或使用 FSDP 分片优化器状态"
        )
    if cat == "gradient" and pct > 30:
        return (
            f"梯度内存占比 {pct:.1f}% (> 30%)，建议:\n"
            f"    - 增大 TP/PP 并行度以分散梯度\n"
            f"    - 检查是否有不必要的梯度保留"
        )
    if cat == "communication" and pct > 20:
        return f"通信缓冲占比 {pct:.1f}% (> 20%)，建议:\n    - 调整并行策略减少通信量\n    - 检查 HCCL 缓冲区配置"
    if cat in ("other", "unknown") and pct > 20:
        return (
            f"'其他/未知' 类别占比 {pct:.1f}% (> 20%)，建议:\n"
            f"    - 使用 Mosaic 工具进行深度分析:\n"
            f"      mosaic_get_memory_usage_peak --snapshot <file>\n"
            f"    - 检查是否有遗留的调试代码或不必要的张量"
        )
    return None


def _build_diagnosis_suggestions(result: SnapshotAnalysis) -> list[str]:
    """构建诊断建议列表（类别占比 + 碎片化 + 泄漏）。"""
    total = max(sum(result.category_memory.values()), 1)
    suggestions = []

    # 1. 检查各类别占比
    for cat, size in result.category_memory.items():
        pct = size / total * 100
        suggestion = _category_suggestion(cat, pct)
        if suggestion is not None:
            suggestions.append(suggestion)

    # 2. 碎片化检查
    if result.fragmentation_rate > 20:
        suggestions.append(
            f"碎片化率 {result.fragmentation_rate:.1f}% (> 20%)，建议在合适时机调用 torch.npu.empty_cache()"
        )

    # 3. 泄漏检查
    if result.potential_leak:
        suggestions.append(f"⚠️  检测到潜在内存泄漏: {result.leak_details}\n    建议对比多个 step 的 snapshot 确认")

    return suggestions


def _print_diagnosis(result: SnapshotAnalysis, prefix: str = "") -> None:
    """基于分析结果给出诊断建议。"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{prefix}诊断建议")
    logger.info(f"{'=' * 70}")

    suggestions = _build_diagnosis_suggestions(result)

    if suggestions:
        for i, s in enumerate(suggestions, 1):
            logger.info(f"  {i}. {s}")
    else:
        logger.info("  ✅ 未发现明显异常，内存分布合理。")

    logger.info("")


# ============================================================
# 双快照对比
# ============================================================


def _fmt_diff(diff: int) -> str:
    """格式化带正负号的字节差异字符串。"""
    if diff > 0:
        return f"+{_fmt_bytes(diff)}"
    if diff < 0:
        return f"-{_fmt_bytes(abs(diff))}"
    return "0"


def _print_comparison_overview(
    baseline: SnapshotAnalysis,
    modified: SnapshotAnalysis,
    label_a: str,
    label_b: str,
) -> None:
    """打印总览对比表（保留/已分配/活跃 + 碎片化）。"""
    logger.info(f"\n  {'指标':<18} {label_a:>14} {label_b:>14} {'差异':>14}")
    logger.info(f"  {'-' * 62}")

    metrics = [
        ("保留内存", baseline.total_reserved, modified.total_reserved),
        ("已分配内存", baseline.total_allocated, modified.total_allocated),
        ("活跃内存", baseline.total_active, modified.total_active),
    ]
    for name, val_a, val_b in metrics:
        diff_str = _fmt_diff(val_b - val_a)
        logger.info(f"  {name:<16} {_fmt_bytes(val_a):>14} {_fmt_bytes(val_b):>14} {diff_str:>14}")

    # 碎片化对比
    logger.info(
        f"  {'碎片化率':<16} "
        f"{baseline.fragmentation_rate:>13.1f}% "
        f"{modified.fragmentation_rate:>13.1f}% "
        f"{modified.fragmentation_rate - baseline.fragmentation_rate:>+13.1f}%"
    )


def _print_comparison_categories(
    baseline: SnapshotAnalysis,
    modified: SnapshotAnalysis,
    all_cats: list[str],
    label_a: str,
    label_b: str,
) -> None:
    """打印按类别对比表。"""
    if not all_cats:
        return
    logger.info("\n  按类别对比:")
    logger.info(f"  {'类别':<16} {label_a:>14} {label_b:>14} {'差异':>14}")
    logger.info(f"  {'-' * 62}")
    for cat in all_cats:
        val_a = baseline.category_memory.get(cat, 0)
        val_b = modified.category_memory.get(cat, 0)
        diff_str = _fmt_diff(val_b - val_a)
        logger.info(f"  {_cn(cat):<14} {_fmt_bytes(val_a):>14} {_fmt_bytes(val_b):>14} {diff_str:>14}")


def _print_comparison_insights(
    baseline: SnapshotAnalysis,
    modified: SnapshotAnalysis,
    all_cats: list[str],
    label_a: str,
    label_b: str,
) -> None:
    """打印关键洞察（总量变化 + 最大变化类别）。"""
    logger.info("\n  关键洞察:")
    total_diff = modified.total_allocated - baseline.total_allocated
    if total_diff > 0:
        logger.info(f"    ⬆ {label_b} 比 {label_a} 多使用 {_fmt_bytes(total_diff)} 内存")
    elif total_diff < 0:
        logger.info(f"    ⬇ {label_b} 比 {label_a} 节省 {_fmt_bytes(abs(total_diff))} 内存")
    else:
        logger.info("    ≈ 两者内存使用基本一致")

    # 找出变化最大的类别
    max_diff_cat = ""
    max_diff_val = 0
    for cat in all_cats:
        diff = abs(modified.category_memory.get(cat, 0) - baseline.category_memory.get(cat, 0))
        if diff > max_diff_val:
            max_diff_val = diff
            max_diff_cat = cat
    if max_diff_cat:
        val_a = baseline.category_memory.get(max_diff_cat, 0)
        val_b = modified.category_memory.get(max_diff_cat, 0)
        direction = "增加" if val_b > val_a else "减少"
        if val_a > 0:
            pct_change = abs(val_b - val_a) / val_a * 100
            logger.info(
                f"    最大变化类别: {_cn(max_diff_cat)} ({direction} {_fmt_bytes(max_diff_val)}, {pct_change:.1f}%)"
            )
        else:
            logger.info(f"    最大变化类别: {_cn(max_diff_cat)} ({direction} {_fmt_bytes(max_diff_val)})")


def _print_comparison(
    baseline: SnapshotAnalysis,
    modified: SnapshotAnalysis,
    label_a: str = "Baseline",
    label_b: str = "Modified",
) -> None:
    """对比两个快照分析结果。"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"双快照对比: {label_a} vs {label_b}")
    logger.info(f"{'=' * 70}")

    _print_comparison_overview(baseline, modified, label_a, label_b)

    all_cats = sorted(set(list(baseline.category_memory.keys()) + list(modified.category_memory.keys())))
    _print_comparison_categories(baseline, modified, all_cats, label_a, label_b)
    _print_comparison_insights(baseline, modified, all_cats, label_a, label_b)

    logger.info("")


def _comparison_to_dict(
    baseline: SnapshotAnalysis,
    modified: SnapshotAnalysis,
) -> dict:
    """双快照对比的 JSON 结构。"""
    all_cats = sorted(set(list(baseline.category_memory.keys()) + list(modified.category_memory.keys())))
    category_diffs = {}
    for cat in all_cats:
        val_a = baseline.category_memory.get(cat, 0)
        val_b = modified.category_memory.get(cat, 0)
        category_diffs[cat] = {
            "baseline_bytes": val_a,
            "modified_bytes": val_b,
            "diff_bytes": val_b - val_a,
            "diff_readable": _fmt_bytes(abs(val_b - val_a)),
        }

    return {
        "baseline": baseline.to_dict(),
        "modified": modified.to_dict(),
        "comparison": {
            "allocated_diff_bytes": (modified.total_allocated - baseline.total_allocated),
            "reserved_diff_bytes": (modified.total_reserved - baseline.total_reserved),
            "fragmentation_diff": (modified.fragmentation_rate - baseline.fragmentation_rate),
            "category_diffs": category_diffs,
        },
    }


# ============================================================
# 多步骤泄漏检测
# ============================================================


def _print_leak_growth_table(results: list[SnapshotAnalysis], labels: list[str]) -> None:
    """打印各快照已分配内存的增长趋势表。"""
    logger.info("\n  内存增长趋势:")
    logger.info(f"  {'快照':<16} {'已分配内存':>14} {'较前一步变化':>14}")
    logger.info(f"  {'-' * 48}")
    for i, (r, label) in enumerate(zip(results, labels)):
        diff_str = ""
        if i > 0:
            diff = r.total_allocated - results[i - 1].total_allocated
            diff_str = _fmt_diff(diff)
        logger.info(f"  {label:<14} {_fmt_bytes(r.total_allocated):>14} {diff_str:>14}")


def _print_leak_verdict(results: list[SnapshotAnalysis]) -> None:
    """检测总内存是否单调增长并打印结论。"""
    allocs = [r.total_allocated for r in results]
    is_monotonic = all(allocs[i] <= allocs[i + 1] for i in range(len(allocs) - 1))
    total_growth = allocs[-1] - allocs[0]

    if is_monotonic and total_growth > 0:
        logger.info(f"\n  ⚠️  内存单调增长! 总增长 {_fmt_bytes(total_growth)}")
        logger.info("  高度怀疑存在内存泄漏。")
    elif total_growth > 0:
        logger.info(f"\n  📈 内存总体增长 {_fmt_bytes(total_growth)}（非单调）")
        logger.info("  可能存在泄漏，也可能是 warmup 阶段的正常增长。")
    else:
        logger.info("\n  ✅ 内存未持续增长，未检测到明显泄漏。")


def _print_leak_category_trends(results: list[SnapshotAnalysis]) -> None:
    """按类别分析并打印增长趋势及泄漏嫌疑。"""
    all_cats = set()
    for r in results:
        all_cats.update(r.category_memory.keys())
    all_cats = sorted(all_cats)

    if not all_cats:
        return

    logger.info("\n  各类别增长趋势（首→末）:")
    logger.info(f"  {'类别':<16} {'首快照':>12} {'末快照':>12} {'变化':>12} {'趋势':>8}")
    logger.info(f"  {'-' * 64}")

    leaking_cats = []
    for cat in all_cats:
        values = [r.category_memory.get(cat, 0) for r in results]
        first, last = values[0], values[-1]
        diff = last - first
        mono = all(values[i] <= values[i + 1] for i in range(len(values) - 1))

        if diff > 0 and mono:
            trend = "⚠️ 泄漏"
            leaking_cats.append((cat, diff))
        elif diff > 0:
            trend = "📈 增长"
        elif diff < 0:
            trend = "📉 下降"
        else:
            trend = "— 稳定"

        diff_str = f"+{_fmt_bytes(diff)}" if diff > 0 else (f"-{_fmt_bytes(abs(diff))}" if diff < 0 else "0")
        logger.info(f"  {_cn(cat):<14} {_fmt_bytes(first):>12} {_fmt_bytes(last):>12} {diff_str:>12} {trend:>8}")

    if leaking_cats:
        logger.info("\n  🔍 泄漏嫌疑类别:")
        for cat, diff in sorted(leaking_cats, key=lambda x: -x[1]):
            logger.info(f"    - {_cn(cat)}: 累计增长 {_fmt_bytes(diff)}")
        logger.info("\n  建议:")
        logger.info("    1. 检查上述类别相关代码中是否有异步操作未正确 wait/同步")
        logger.info("    2. 检查 activation checkpoint + TP/通信的交互是否导致 tensor 泄漏")
        logger.info("    3. 检查是否有 tensor 被意外持有引用（如调试代码、全局变量）")
        logger.info("    4. 参考 SKILL.md Step 3.6 进行代码审查")


def _analyze_leak_trend(results: list[SnapshotAnalysis], labels: list[str]) -> None:
    """分析多个快照的内存增长趋势，检测泄漏。"""
    logger.info(f"\n{'=' * 70}")
    logger.info("多步骤内存泄漏检测")
    logger.info(f"{'=' * 70}")
    logger.info(f"  快照数量: {len(results)}")
    logger.info(f"  快照标签: {', '.join(labels)}")

    _print_leak_growth_table(results, labels)
    _print_leak_verdict(results)
    _print_leak_category_trends(results)

    logger.info("")


def _leak_trend_to_dict(results: list[SnapshotAnalysis], labels: list[str]) -> dict:
    """多步骤泄漏检测的 JSON 结构。"""
    allocs = [r.total_allocated for r in results]
    is_monotonic = all(allocs[i] <= allocs[i + 1] for i in range(len(allocs) - 1))

    all_cats = set()
    for r in results:
        all_cats.update(r.category_memory.keys())

    category_trends = {}
    for cat in sorted(all_cats):
        values = [r.category_memory.get(cat, 0) for r in results]
        mono = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        category_trends[cat] = {
            "values_bytes": values,
            "first_bytes": values[0],
            "last_bytes": values[-1],
            "diff_bytes": values[-1] - values[0],
            "monotonic_growth": mono and values[-1] > values[0],
        }

    return {
        "snapshots": [{"label": label, "analysis": r.to_dict()} for r, label in zip(results, labels)],
        "leak_detection": {
            "total_allocated_trend": allocs,
            "total_growth_bytes": allocs[-1] - allocs[0],
            "is_monotonic_growth": is_monotonic and allocs[-1] > allocs[0],
            "category_trends": category_trends,
            "leaking_categories": [cat for cat, info in category_trends.items() if info["monotonic_growth"]],
        },
    }


# ============================================================
# 主入口
# ============================================================


def analyze_snapshot(snapshot_path: str, top_n: int = 10) -> SnapshotAnalysis:
    """加载并分析一个 memory snapshot 文件。

    注意：基于 pickle 反序列化加载，请仅分析可信来源（自己训练产出）的 snapshot 文件，
    不要加载来历不明的 .pickle，以免触发反序列化代码执行风险。
    """
    try:
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise RuntimeError(
            f"无法读取或解析 snapshot 文件 {snapshot_path}: {exc}"
        ) from exc

    result = _analyze_segments(snapshot, top_n=top_n)
    _analyze_timeline(snapshot, result)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory Snapshot 深度分析工具",
        epilog=(
            "示例:\n"
            "  %(prog)s snapshot.pickle\n"
            "  %(prog)s snapshot.pickle --json\n"
            "  %(prog)s baseline.pickle modified.pickle\n"
            "  %(prog)s step5.pickle step10.pickle step20.pickle  # 泄漏检测\n"
            "  %(prog)s snapshot.pickle --top-n 20\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="snapshot .pickle 文件路径（1个=单独分析，2个=对比分析，3个+=泄漏检测）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结构化分析结果",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N 大块内存分配数量（默认: 10）",
    )
    parser.add_argument(
        "--label-a",
        default="Baseline",
        help="对比模式下第一个快照的标签（默认: Baseline）",
    )
    parser.add_argument(
        "--label-b",
        default="Modified",
        help="对比模式下第二个快照的标签（默认: Modified）",
    )
    return parser


def _dispatch_single(args) -> None:
    """单快照分析。"""
    result = analyze_snapshot(args.snapshots[0], top_n=args.top_n)
    if args.json:
        logger.info(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_analysis(result)


def _dispatch_comparison(args) -> None:
    """双快照对比。"""
    result_a = analyze_snapshot(args.snapshots[0], top_n=args.top_n)
    result_b = analyze_snapshot(args.snapshots[1], top_n=args.top_n)
    if args.json:
        output = _comparison_to_dict(result_a, result_b)
        logger.info(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        _print_analysis(result_a, label=args.label_a)
        _print_analysis(result_b, label=args.label_b)
        _print_comparison(
            result_a,
            result_b,
            label_a=args.label_a,
            label_b=args.label_b,
        )


def _dispatch_leak_trend(args) -> None:
    """3+ 快照：多步骤泄漏检测。"""
    results = []
    labels = []
    for path in args.snapshots:
        results.append(analyze_snapshot(path, top_n=args.top_n))
        labels.append(os.path.splitext(os.path.basename(path))[0])
    if args.json:
        output = _leak_trend_to_dict(results, labels)
        logger.info(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # 打印首尾快照的详细分析
        _print_analysis(results[0], label=labels[0])
        _print_analysis(results[-1], label=labels[-1])
        # 打印泄漏趋势
        _analyze_leak_trend(results, labels)


def _dispatch(args) -> None:
    """根据快照数量分派到对应分析流程。"""
    if len(args.snapshots) == 1:
        _dispatch_single(args)
    elif len(args.snapshots) == 2:
        _dispatch_comparison(args)
    else:
        _dispatch_leak_trend(args)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        _dispatch(args)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
