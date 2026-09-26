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
"""
profiling_evidence.py — 空泡诊断→世界模型策略映射

将 profiling 分析结果 (T1/T2/T3) 转换为世界模型可用的策略建议，
用于 lingxi-evo Refine 步骤中注入子节点的 strategy_combination。

═══════════════════════════════════════════════════════════════════
v3.2 三阶段诊断管线（Phase B 引入）
═══════════════════════════════════════════════════════════════════

旧 API（保留向后兼容）：
    extract_profiling_evidence(eval_results) -> {bottleneck_type, suggested_strategies, ...}

新 API（v3.2，推荐使用）：
    Stage 1: extract_facts(eval_results) -> {dominant_pipe, mte2_ratio, bw_utilization, ...}
             纯事实抽取，无策略推荐，无瓶颈结论
    Stage 2: refine 阶段 LLM 在 prompt 里读 facts + cannbot quickref，
             输出 {diagnosis_text, bottleneck_labels[], confidence}
    Stage 3: match_strategies_by_labels(labels) -> {candidate_source_keys, candidate_ids, ...}
             按 labels 反查 INDEX.json triggers，纯脚本无 LLM

设计文档：docs/design/knowledge-strategy-architecture-v3.2.md §3.4
═══════════════════════════════════════════════════════════════════

用法 (旧)：
    from profiling_evidence import extract_profiling_evidence
    evidence = extract_profiling_evidence(evaluation_results)
    # → {"bottleneck_type": "mte2_stall", "suggested_strategies": ["P1", "P10"], ...}

用法 (新 v3.2)：
    from profiling_evidence import extract_facts, match_strategies_by_labels
    facts = extract_facts(evaluation_results)
    # ↑ 注入 refine prompt 让 LLM 给 labels
    candidates = match_strategies_by_labels(["fake_mte2_bound", "undersize_transfer"])
    # → {"candidate_source_keys": [...], "candidate_ids": [...], ...}
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger(__name__)


# 瓶颈类型 → 推荐/反推荐策略映射
BOTTLENECK_STRATEGY_MAP = {
    "mte2_stall": {
        "primary": ["P1", "P10"],
        "secondary": ["P2", "P25", "P66", "P7", "P19", "P18", "P56", "P59"],
        "anti": ["P3"],
        "description": "MTE2 数据加载等待是主瓶颈，应通过双缓冲重叠搬运与计算；辅以 DataCopyPad 对齐 (P25) 和 512B 对齐 (P66) 减少带宽折损",
    },
    "mte3_stall": {
        "primary": ["P1", "P10"],
        "secondary": ["P8", "P19", "P40", "P56", "P59"],
        "anti": [],
        "description": "MTE3 写回等待是主瓶颈，双缓冲可重叠写回与下一轮计算",
    },
    "tiling_imbalance": {
        "primary": ["P4", "P2"],
        "secondary": ["P11", "P47", "P51", "P58", "P72", "P73", "P75"],
        "anti": ["P3"],
        "description": "多核负载不均衡，应优化分块策略使各核工作量均匀",
    },
    "scalar_loading": {
        "primary": ["P5", "P67"],
        "secondary": ["P2", "P10", "P54", "P84", "P97", "P99", "P100"],
        "anti": [],
        "description": "标量参数加载阻塞 VEC，应预取参数或简化地址计算；次级含零拷贝(P97)、广播批处理(P99)、GatherMask替换(P100)",
    },
    "scalar_compute": {
        "primary": ["P5", "P67", "P94", "P95"],
        "secondary": ["P10", "P84", "P96", "P97", "P98", "P99", "P100"],
        "anti": [],
        "description": "标量计算阻塞 VEC：地址计算(P5/P67)或算术循环(P94 ReduceSum/P95 AddsMuls)；"
                       "次级含硬件排序(P96)、零拷贝(P97)、混合调度(P98)、广播批处理(P99)、GatherMask替换(P100)",
    },
    "compute_bound": {
        "primary": ["D1", "P46"],
        "secondary": ["A1", "P47", "P68", "P69", "P79", "P84", "P94", "P95", "P98", "P99"],
        "anti": [],
        "description": "计算密集型瓶颈，应考虑算法优化或混合精度；次级含标量归约向量化(P94)、逐元素向量化(P95)、混合调度(P98)、广播批处理(P99)",
    },
    "near_optimal": {
        "primary": [],
        "secondary": [],
        "anti": [],
        "description": "内核已接近最优，无明显可优化空泡",
    },
    # ── 新增子类型 ──
    "no_overlap": {
        "primary": ["P1"],
        "secondary": ["P10", "P19", "P63"],
        "anti": [],
        "description": "CopyIn/Compute/CopyOut 无时间重叠，需启用双缓冲",
    },
    "partial_overlap": {
        "primary": ["P1", "P2"],
        "secondary": ["P8", "P18", "P26", "P53", "P83"],
        "anti": [],
        "description": "双缓冲已启用但重叠不充分，需调整 tile 大小或 UB 分区",
    },
    "undersize_transfer": {
        "primary": ["P2", "P10"],
        "secondary": ["P7", "P25", "P45", "P66"],
        "anti": [],
        "description": "DMA 搬运粒度过小，增大 tile 或使用向量化搬运、合并搬运",
    },
    "icache_miss": {
        "primary": ["P8", "P54"],
        "secondary": [],
        "anti": [],
        "description": "指令缓存未命中，内核代码体积过大，应简化或拆分（P8）、合并小 kernel 减少 launch 开销（P54）",
    },
    "bus_contention": {
        "primary": ["P8", "P65"],
        "secondary": ["P1", "P28", "P64"],
        "anti": [],
        "description": "MTE2/MTE3 总线竞争，应错开搬运时序或使用 HardEvent 同步",
    },
    "l2_cache_thrash": {
        "primary": ["P52", "P74"],
        "secondary": ["P61", "P78"],
        "anti": [],
        "description": "L2 cache 频繁失效，多核竞争导致缓存抖动，应禁用或引导 L2 缓存",
    },
    "ub_memory_pressure": {
        "primary": ["P8", "P85"],
        "secondary": ["P81", "P88"],
        "anti": ["P20"],
        "description": "UB 内存压力过大，应优化 buffer 分区复用或减少同时驻留的 tensor 数量",
    },
}


def classify_bottleneck(profiling_analysis: dict) -> str:
    """
    从 profiling_analysis 字段判定主瓶颈类型。

    Args:
        profiling_analysis: evaluation_results.json 中的 profiling_analysis 字段

    Returns:
        瓶颈类型字符串 (BOTTLENECK_STRATEGY_MAP 的 key)
    """
    if not profiling_analysis:
        return "near_optimal"

    imbalance_ratio = profiling_analysis.get("imbalance_ratio", 1.0)
    d_class_pct = profiling_analysis.get("d_class_pct", 0)
    c_class_pct = profiling_analysis.get("c_class_pct", 0)
    primary_bottleneck = profiling_analysis.get("primary_bottleneck")

    # 优先级: tiling_imbalance > D类 > C类 > near_optimal
    if imbalance_ratio > 1.3:
        return "tiling_imbalance"

    if d_class_pct > 30:
        # 细化 D 类子类型 — 使用 T7/T8/T9 数据
        overlap_status = profiling_analysis.get("overlap_status")
        if overlap_status == "no_overlap":
            return "no_overlap"
        if overlap_status == "partial_overlap":
            return "partial_overlap"

        # DMA 效率
        dma_eff = profiling_analysis.get("dma_efficiency", {})
        mte2_short = dma_eff.get("mte2_short_pct", 0)
        if mte2_short > 30:
            return "undersize_transfer"

        # 周期性 icache miss
        pattern_type = profiling_analysis.get("pattern_type")
        dominant_subtype = profiling_analysis.get("dominant_subtype", "")
        if pattern_type == "periodic" and "icache" in str(dominant_subtype):
            return "icache_miss"

        if primary_bottleneck == "MTE2":
            return "mte2_stall"
        elif primary_bottleneck == "MTE3":
            return "mte3_stall"
        else:
            return "mte2_stall"  # 默认归为 MTE2

    if c_class_pct > 20:
        c_cause = profiling_analysis.get("c_class_primary_cause")
        if c_cause == "SCALARLDST":
            return "scalar_loading"
        elif c_cause == "SCALAR":
            return "scalar_compute"
        return "scalar_loading"  # 默认

    # 高计算占比 → compute_bound
    compute_pct = profiling_analysis.get("pure_compute_pct", 0)
    if compute_pct > 70:
        return "compute_bound"

    return "near_optimal"


def extract_profiling_evidence(evaluation_results: dict) -> Optional[dict]:
    """
    从 evaluation_results.json 提取 profiling 证据，生成世界模型可用的结构。

    Args:
        evaluation_results: 完整的 evaluation_results.json 内容

    Returns:
        profiling_evidence dict (可直接写入世界模型节点)，
        若无 profiling 数据则返回 None
    """
    profiling = evaluation_results.get("profiling_analysis")
    if not profiling:
        return None

    bottleneck_type = classify_bottleneck(profiling)
    strategy_info = BOTTLENECK_STRATEGY_MAP.get(bottleneck_type, {})

    suggested = strategy_info.get("primary", []) + strategy_info.get("secondary", [])
    anti = strategy_info.get("anti", [])

    return {
        "bottleneck_type": bottleneck_type,
        "d_class_pct": profiling.get("d_class_pct", 0),
        "c_class_pct": profiling.get("c_class_pct", 0),
        "imbalance_ratio": profiling.get("imbalance_ratio", 1.0),
        "primary_bottleneck": profiling.get("primary_bottleneck"),
        "suggested_strategies": suggested,
        "anti_strategies": anti,
        "description": strategy_info.get("description", ""),
        "top_recommendation": profiling.get("top_recommendation", ""),
        # ── 新增字段 ──
        "pattern_type": profiling.get("pattern_type"),
        "overlap_status": profiling.get("overlap_status"),
        "dominant_subtype": profiling.get("dominant_subtype"),
        "dma_efficiency": profiling.get("dma_efficiency"),
    }


def synthesize_analysis_from_pipeline(
    pipeline: Optional[dict],
    bottleneck_hint: Optional[str] = None,
) -> Optional[dict]:
    """Coarse-grained pipeline → profiling_analysis mapping (CSV-level fallback).

    evaluate_ops_direct.py only writes pipeline ratios (aiv_mte2_ratio,
    aiv_vec_ratio, aiv_scalar_ratio, aiv_mte3_ratio). This helper promotes
    those ratios to the subset of profiling_analysis fields that
    classify_bottleneck() relies on, so extract_profiling_evidence() can be
    used from the main refine loop without requiring deep-profiling.

    Fields that truly need instruction-level trace (pattern_type, overlap_status,
    dominant_subtype, dma_efficiency) are left None — deep-profiling still owns
    those.
    """
    if not pipeline:
        return None

    def _pct(key: str) -> float:
        v = pipeline.get(key)
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        # Normalize ratios (0..1) to percentages (0..100)
        if 0.0 <= f <= 1.0:
            f *= 100.0
        return f

    mte2 = _pct("aiv_mte2_ratio")
    mte3 = _pct("aiv_mte3_ratio")
    vec = _pct("aiv_vec_ratio")
    scalar = _pct("aiv_scalar_ratio")
    # icache miss rate is usually stored as a ratio (0..1); _pct already
    # normalizes it to percentage.
    icache_miss_pct = _pct("aiv_icache_miss_rate")

    # D-class: non-VEC ratio (MTE2 + MTE3 dominate)
    d_class_pct = max(mte2, mte3)
    if mte2 >= mte3:
        primary_bn = "MTE2"
    else:
        primary_bn = "MTE3"

    # C-class proxy: scalar ratio above ~20% is a scalar pressure signal.
    c_class_pct = scalar if scalar > 15 else 0.0
    c_cause = "SCALARLDST" if scalar > 15 else None

    # CSV-level icache miss signal. classify_bottleneck() gates icache_miss on
    # (d_class_pct > 30) AND pattern_type=="periodic" AND "icache" in
    # dominant_subtype. Synthesize the latter two from the CSV column so the
    # main refine loop can route to icache_miss without waiting for deep
    # profiling. Only flag when miss rate is meaningfully high (>10%).
    if icache_miss_pct > 10.0:
        pattern_type = "periodic"
        dominant_subtype = "icache"
        # Ensure d_class gate passes so classify_bottleneck picks icache_miss
        # over compute_bound fallback.
        if d_class_pct <= 30:
            d_class_pct = 31.0
    else:
        pattern_type = None
        dominant_subtype = None

    return {
        "imbalance_ratio": 1.0,   # not observable from CSV
        "d_class_pct": d_class_pct,
        "c_class_pct": c_class_pct,
        "primary_bottleneck": primary_bn if d_class_pct > 30 else None,
        "pure_compute_pct": vec,
        "c_class_primary_cause": c_cause,
        "overlap_status": None,
        "dominant_subtype": dominant_subtype,
        "pattern_type": pattern_type,
        "dma_efficiency": {},
        "icache_miss_pct": icache_miss_pct,
        "_synthesis_source": "pipeline_csv",
        "_bottleneck_hint": bottleneck_hint,
    }


@dataclass
class MergeConfig:
    """merge_strategies_with_evidence 的窗口配置。"""
    max_total: int = 5
    new_slots: int = 3
    offset: int = 0


def merge_strategies_with_evidence(
    base_strategies: list[str],
    evidence: Optional[dict],
    ancestry_failed: Optional[Iterable[str]] = None,
    parent_hint: Optional[dict] = None,
    config: MergeConfig = None,
) -> list[str]:
    """
    派生子节点的 strategy_combination 计算（v3.3 滑动窗口 + LLM-hint 引导）。

    设计要点：
    - 父血缘前 (max_total - new_slots) 个作为"地基"传承（跳过第 offset 位，让兄弟差异化）
    - 给本轮 profiling 新建议留 new_slots 个"滚动槽"
    - LLM 父节点 diagnosis.next_round_hint.prefer 抢占新槽位最高优先级
    - 自动应用 anti 集合：profiling anti_strategies ∪ 父链 failed 节点策略 ∪ hint.avoid

    Args:
        base_strategies: 父节点 strategy_combination
        evidence: profiling_evidence dict (有 suggested_strategies / anti_strategies)
        ancestry_failed: 父链上 status=failed 节点的 strategy_combination 集合
        parent_hint: 父节点 diagnosis.next_round_hint dict (prefer / avoid / rationale)
        config: 窗口配置（max_total: cap 上限；new_slots: 新建议滚动槽位数；
                offset: 兄弟差异化偏移，跳过父血缘第 offset 位 + 切 suggested[offset:]）

    Returns:
        策略 list，长度 ≤ max_total
    """
    if evidence is None:
        evidence = {}
    if parent_hint is None:
        parent_hint = {}
    if config is None:
        config = MergeConfig()
    max_total, new_slots, offset = config.max_total, config.new_slots, config.offset

    # 合并 anti 集合：profiling 当下负面 + 父链历史失败 + LLM hint 主动避开
    anti: set[str] = set(evidence.get("anti_strategies", []) or [])
    if ancestry_failed:
        anti |= set(ancestry_failed)
    anti |= set(parent_hint.get("avoid", []) or [])

    # 父血缘部分：跳过第 offset 位 + anti 过滤；取前 (max_total - new_slots)
    inherited = [
        s for i, s in enumerate(base_strategies)
        if i != offset and s not in anti
    ]
    keep_from_parent = max(0, max_total - new_slots)
    result = inherited[:keep_from_parent]

    # 新槽位：LLM hint.prefer 抢占最高优先级
    for s in parent_hint.get("prefer", []) or []:
        if len(result) >= max_total:
            break
        if s not in result and s not in anti:
            result.append(s)

    # 兜底：profiling.suggested 按 offset 切，填到 cap
    for s in (evidence.get("suggested_strategies", []) or [])[offset:]:
        if len(result) >= max_total:
            break
        if s not in result and s not in anti:
            result.append(s)

    return result


# ═══════════════════════════════════════════════════════════════════
# v3.2 新 API: 三阶段诊断管线
# ═══════════════════════════════════════════════════════════════════

# 标准化 bottleneck_labels 词表（18 个 = 14 主标签 + 4 Stage 2 精细化标签）
# 与 plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/profiling_reference/INDEX.md 对齐
KNOWN_BOTTLENECK_LABELS = frozenset({
    # 主 14 个，与 cards/SCHEMA.md 词表一致
    "mte2_stall", "mte3_stall", "tiling_imbalance",
    "scalar_loading", "scalar_compute", "compute_bound",
    "near_optimal", "no_overlap", "partial_overlap",
    "undersize_transfer", "icache_miss", "bus_contention",
    "l2_cache_thrash", "ub_memory_pressure",
    # 增量 4 个，Stage 2 LLM 精细化标签（真假 bound + bank conflict + db_not_effective）
    "fake_mte2_bound", "fake_compute_bound",
    "bank_conflict", "db_not_effective",
})


def _normalize_ratio(value, default=None):
    """把可能是 0~1 或 0~100 的数值统一归一到 0~1。"""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v / 100.0 if v > 1.0 else v


def _select_fact_source(evaluation_results: dict):
    """选择 facts 数据来源：优先 deep profiling，回退 pipeline CSV。

    返回 (source, data)；无 profiling 数据返回 (None, None)。
    """
    profiling = evaluation_results.get("profiling_analysis") if evaluation_results else None
    pipeline = evaluation_results.get("pipeline") if evaluation_results else None

    if not profiling and not pipeline:
        return None, None
    if profiling:
        return "deep_profiling", profiling
    # 用原始 pipeline 字段，避免 synthesize 丢失 mte3/mte2 等原始 ratio
    return "pipeline_csv", dict(pipeline) if pipeline else {}


def extract_facts(evaluation_results: dict) -> Optional[dict]:
    """Stage 1: 纯事实抽取（无策略推荐，无瓶颈结论）。

    Stage 1 的职责是把 evaluation_results.json 里散落的 profiling 数据
    标准化为一个 dict，**只描述事实**，不做"是什么瓶颈"的判定——
    那是 Stage 2 LLM 的工作。

    Args:
        evaluation_results: 完整的 evaluation_results.json 内容
            支持两种来源：deep_profiling (profiling_analysis 字段)
                       或 CSV-level pipeline (pipeline 字段)

    Returns:
        facts dict，含字段：
            _source           : "deep_profiling" 或 "pipeline_csv"
            dominant_pipe     : 流水占比最高的 ("mte2"/"vec"/...) 或 None
            mte2_ratio        : MTE2 占比 (0~1)
            mte3_ratio        : MTE3 占比
            vec_ratio         : VEC 占比
            scalar_ratio      : Scalar 占比
            bw_utilization    : 带宽利用率 (用于真假 MTE2 bound 区分)
            bank_cflt_ratio   : Bank conflict 占比
            l2_hit_rate       : L2 cache 命中率
            icache_miss_rate  : icache miss 率
            imbalance_ratio   : 各核耗时不均衡比 (max/min)
            pattern_type      : deep profiling 才有
            overlap_status    : 双缓冲重叠状态
            dma_efficiency    : DMA 效率细节

        若无任何 profiling 数据返回 None。
    """
    source, data = _select_fact_source(evaluation_results)
    if source is None:
        return None

    # 流水占比（统一归一到 0~1）
    mte2_ratio = _normalize_ratio(data.get("aiv_mte2_ratio"))
    mte3_ratio = _normalize_ratio(data.get("aiv_mte3_ratio"))
    vec_ratio = _normalize_ratio(data.get("aiv_vec_ratio"))
    scalar_ratio = _normalize_ratio(data.get("aiv_scalar_ratio"))

    # 兜底：如果 ratio 字段缺失，用 d/c_class_pct 反推
    if mte2_ratio is None:
        mte2_ratio = _normalize_ratio(data.get("d_class_pct"))
    if vec_ratio is None:
        vec_ratio = _normalize_ratio(data.get("pure_compute_pct"))
    if scalar_ratio is None:
        scalar_ratio = _normalize_ratio(data.get("c_class_pct"))

    # 确定 dominant_pipe
    pipes = {
        "mte2": mte2_ratio or 0,
        "mte3": mte3_ratio or 0,
        "vec": vec_ratio or 0,
        "scalar": scalar_ratio or 0,
    }
    dominant = max(pipes, key=pipes.get) if any(v for v in pipes.values()) else None

    # 不均衡
    imbalance = data.get("imbalance_ratio", 1.0)

    return {
        "_source": source,
        "dominant_pipe": dominant,
        "mte2_ratio": mte2_ratio,
        "mte3_ratio": mte3_ratio,
        "vec_ratio": vec_ratio,
        "scalar_ratio": scalar_ratio,
        "bw_utilization": _normalize_ratio(data.get("bw_utilization") or data.get("bw_usage_rate")),
        "bank_cflt_ratio": _normalize_ratio(
            data.get("bank_cflt_ratio") or data.get("aiv_vec_total_cflt_ratio")
        ),
        "l2_hit_rate": _normalize_ratio(data.get("l2_hit_rate")),
        "icache_miss_rate": _normalize_ratio(
            data.get("icache_miss_pct")
            or data.get("aiv_icache_miss_rate")
            or data.get("icache_miss_rate")
        ),
        "imbalance_ratio": imbalance,
        "imbalance_pct": max(0.0, imbalance - 1.0) if imbalance else 0.0,
        "pattern_type": data.get("pattern_type"),
        "overlap_status": data.get("overlap_status"),
        "dominant_subtype": data.get("dominant_subtype"),
        "dma_efficiency": data.get("dma_efficiency", {}),
    }


# INDEX.json 加载（带缓存）
_INDEX_CACHE: Optional[dict] = None
_INDEX_PATH_CACHE: Optional[Path] = None


def _find_index_path() -> Optional[Path]:
    """搜寻 evolution-strategies 的 INDEX.json 路径。

    优先用相对 CWD 路径，回退到相对本脚本路径。
    """
    cwd_candidate = Path("plugins-community/ops-perf-evolution/skills/evolution-strategies/references/INDEX.json")
    if cwd_candidate.exists():
        return cwd_candidate

    here = Path(__file__).resolve().parent
    for up in [here, here.parent, here.parent.parent, here.parent.parent.parent, here.parent.parent.parent.parent]:
        candidate = up / "plugins-community/ops-perf-evolution/skills/evolution-strategies/references/INDEX.json"
        if candidate.exists():
            return candidate
    return None


def _load_index(force_reload: bool = False) -> Optional[dict]:
    global _INDEX_CACHE, _INDEX_PATH_CACHE
    if not force_reload and _INDEX_CACHE is not None:
        return _INDEX_CACHE

    path = _find_index_path()
    if not path:
        return None
    _INDEX_PATH_CACHE = path
    _INDEX_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _INDEX_CACHE


def match_strategies_by_labels(
    labels: list[str],
    *,
    include_unknown: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """Stage 3: 按 bottleneck_labels 反查 INDEX.json triggers。

    输入由 Stage 2 LLM 给出的 bottleneck_labels，返回适用的 source_keys 列表。
    纯脚本无 LLM。

    Args:
        labels: bottleneck_labels 列表（应为 KNOWN_BOTTLENECK_LABELS 子集）
        include_unknown: 是否在 unknown_labels 字段返回非词表标签
        limit: candidate_source_keys 返回数量上限（默认全返回）

    Returns:
        dict {
            "candidate_source_keys": [str],   # 按命中 label 数降序
            "candidate_ids":         [str],   # 同上的 strategy ID
            "by_label":              {label: [ids]},
            "unknown_labels":        [str],   # 词表外的 labels
        }
    """
    idx = _load_index()
    if not idx:
        return {
            "candidate_source_keys": [],
            "candidate_ids": [],
            "by_label": {},
            "unknown_labels": list(labels),
            "_error": "INDEX.json not found (searched cwd and script-relative paths)",
        }

    unknown = [l for l in labels if l not in KNOWN_BOTTLENECK_LABELS]
    valid_labels = [l for l in labels if l in KNOWN_BOTTLENECK_LABELS]

    by_label: dict[str, list[str]] = {l: [] for l in valid_labels}
    hit_count: dict[str, int] = {}

    for card in idx.get("cards", []):
        card_labels = set(card.get("triggers", {}).get("bottleneck_labels", []))
        sid = card["id"]
        for l in valid_labels:
            if l in card_labels:
                by_label[l].append(sid)
                hit_count[sid] = hit_count.get(sid, 0) + 1

    # 按命中 label 数降序，相同命中数按 ID 升序
    sorted_ids = sorted(hit_count.keys(), key=lambda x: (-hit_count[x], x))
    if limit is not None:
        sorted_ids = sorted_ids[:limit]

    id_to_card = {c["id"]: c for c in idx.get("cards", [])}
    source_keys = [id_to_card[sid]["source_key"] for sid in sorted_ids]

    result = {
        "candidate_source_keys": source_keys,
        "candidate_ids": sorted_ids,
        "by_label": by_label,
    }
    if include_unknown:
        result["unknown_labels"] = unknown
    elif unknown:
        result["unknown_labels"] = unknown
    return result


def validate_labels(labels: list[str]) -> dict:
    """校验 bottleneck_labels 列表是否合法（用于 wm_ops.refine 校验 LLM 输出）。

    Returns:
        {"valid": bool, "unknown": [labels not in vocab], "vocabulary_size": 18}
    """
    unknown = [l for l in labels if l not in KNOWN_BOTTLENECK_LABELS]
    return {
        "valid": len(unknown) == 0,
        "unknown": unknown,
        "vocabulary_size": len(KNOWN_BOTTLENECK_LABELS),
    }


# ═══════════════════════════════════════════════════════════════════
# v3.2 Stage 3 收口: expected_gain 计算
# ═══════════════════════════════════════════════════════════════════

# bottleneck label → 该 label 在 facts dict 中的占比来源 key 列表
# - 值是 facts 字段名 list，compute_expected_gain 按 LABEL_AGGREGATION 聚合
# - 值为 None 表示无直接 facts 字段，用 LABEL_FALLBACK_RATIO 保守默认值
# - 与 KNOWN_BOTTLENECK_LABELS 一一对应（18 项全覆盖）
LABEL_TO_FACTS_RATIO_KEYS: dict[str, Optional[list[str]]] = {
    "mte2_stall": ["mte2_ratio"],
    "fake_mte2_bound": ["mte2_ratio"],
    "mte3_stall": ["mte3_ratio"],
    "undersize_transfer": ["mte2_ratio"],            # 短搬运 → MTE2 总占比代理
    "scalar_loading": ["scalar_ratio"],
    "scalar_compute": ["scalar_ratio"],
    "icache_miss": ["icache_miss_rate"],
    "tiling_imbalance": ["imbalance_pct"],
    "no_overlap": ["mte2_ratio", "mte3_ratio"],
    "partial_overlap": ["mte2_ratio", "mte3_ratio"],
    "db_not_effective": ["mte2_ratio", "mte3_ratio"],
    "bus_contention": ["mte2_ratio", "mte3_ratio"],  # 总线竞争 → 用 sum 聚合
    "bank_conflict": ["bank_cflt_ratio"],
    "compute_bound": ["vec_ratio"],
    "fake_compute_bound": ["vec_ratio"],
    "l2_cache_thrash": None,
    "ub_memory_pressure": None,
    "near_optimal": None,
}

# 无 facts 字段时的保守默认占比（避免有效策略被零收益完全抑制）
LABEL_FALLBACK_RATIO: dict[str, float] = {
    "l2_cache_thrash": 0.10,
    "ub_memory_pressure": 0.05,
    "near_optimal": 0.0,
}

# label 内部多字段的聚合方式，默认 max；bus_contention 是 sum
# 理由：bus_contention 物理上是两条流水**同时**繁忙才触发，sum 才反映真实占比
LABEL_AGGREGATION: dict[str, str] = {
    "bus_contention": "sum",
}


def compute_expected_gain(
    strategy_combination: list[str],
    parent_facts: Optional[dict],
    parent_labels: Optional[list[str]],
    *,
    addressed_cap: float = 0.8,
    expected_gain_cap: float = 4.0,
) -> float:
    """估算策略组合若完全消除父节点诊断标签覆盖的瓶颈，理论可获得的 delta speedup 上限。

    Amdahl 风格保守估计：
      1. 加载 INDEX.json，对 strategy_combination 中每个 sid 取 triggers.bottleneck_labels
      2. addressable_labels = ⋃(strategy 触发 labels) ∩ parent_labels
         —— 父诊断认为的瓶颈中本策略组合能解决的子集
      3. addressed_ratio = ⋃ 各 label 映射到 facts 的占比，**facts 字段层面去重**
         —— mte2_stall 和 no_overlap 都映射到 mte2_ratio，物理上是同一段时间，不累加
      4. expected_gain = 1/(1-addressed) - 1，cap 在 expected_gain_cap

    Early return 0.0 的情况：
      - strategy_combination/parent_facts/parent_labels 任一缺失
      - INDEX.json 加载失败
      - addressable_labels 为空（策略不针对父诊断的任一瓶颈）

    Returns: float in [0, expected_gain_cap]，0 表示无前瞻收益（utility 退化为旧公式）
    """
    if not strategy_combination or not parent_facts or not parent_labels:
        return 0.0

    idx = _load_index()
    if not idx:
        return 0.0

    id_to_card = {c["id"]: c for c in idx.get("cards", [])}
    parent_label_set = set(parent_labels)

    # 收集策略组合覆盖到的、且在父诊断 labels 内的标签
    addressable: set[str] = set()
    for sid in strategy_combination:
        card = id_to_card.get(sid)
        if not card:
            continue
        triggers = set(card.get("triggers", {}).get("bottleneck_labels", []))
        addressable |= (triggers & parent_label_set)

    if not addressable:
        return 0.0

    # 对每个 addressable label 算其贡献占比，**分摊到具体的 facts 字段**
    # 关键去重原则：facts_used 以 facts 字段名为 key，多 label 命中同字段时取 max
    # —— 物理意义：一段时间不能被两个 label 都"声称解决"
    facts_used: dict[str, float] = {}

    for label in addressable:
        keys = LABEL_TO_FACTS_RATIO_KEYS.get(label)
        if keys is None:
            # fallback 类 label 没有真 facts 字段，单独用 label 名做桶
            fallback = LABEL_FALLBACK_RATIO.get(label, 0.0)
            if fallback > 0:
                facts_used[label] = max(facts_used.get(label, 0.0), fallback)
            continue

        # 收集该 label 在 facts 中能取到的值
        key_values: dict[str, float] = {}
        for k in keys:
            v = parent_facts.get(k)
            if v is None:
                continue
            try:
                key_values[k] = float(v)
            except (TypeError, ValueError):
                continue
        if not key_values:
            continue

        agg = LABEL_AGGREGATION.get(label, "max")
        if agg == "sum":
            # sum 类（bus_contention）：每个字段都计入自己的值
            # 物理意义：两条流水同时繁忙时，两者都贡献到总占比
            for k, v in key_values.items():
                facts_used[k] = max(facts_used.get(k, 0.0), v)
        else:
            # max 类：只把"赢家字段"记入，避免跨 label 重复计入
            # 物理意义：no_overlap 实际取自 mte2_ratio 时，不应同时占用 mte3_ratio
            winner_k = max(key_values, key=key_values.get)
            winner_v = key_values[winner_k]
            facts_used[winner_k] = max(facts_used.get(winner_k, 0.0), winner_v)

    addressed_ratio = min(sum(facts_used.values()), addressed_cap)
    if addressed_ratio <= 0.0:
        return 0.0

    gain = 1.0 / (1.0 - addressed_ratio) - 1.0
    return min(gain, expected_gain_cap)


if __name__ == "__main__":
    # smoke test
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # 模拟 evaluation_results 输入
        mock = {
            "pipeline": {
                "aiv_mte2_ratio": 0.85,
                "aiv_vec_ratio": 0.40,
                "aiv_scalar_ratio": 0.05,
                "aiv_mte3_ratio": 0.10,
                "aiv_icache_miss_rate": 0.03,
            }
        }
        LOGGER.info("=== extract_facts ===")
        facts = extract_facts(mock)
        LOGGER.info(json.dumps(facts, indent=2))
        LOGGER.info("")
        LOGGER.info("=== match_strategies_by_labels(['mte2_stall', 'no_overlap']) ===")
        candidates = match_strategies_by_labels(["mte2_stall", "no_overlap"], limit=5)
        LOGGER.info(json.dumps(candidates, indent=2))
        LOGGER.info("")
        LOGGER.info("=== validate_labels(['mte2_stall', 'bogus_label']) ===")
        LOGGER.info(json.dumps(validate_labels(["mte2_stall", "bogus_label"]), indent=2))
        LOGGER.info("")
        LOGGER.info("=== compute_expected_gain demos ===")
        demo_facts = {"mte2_ratio": 0.6, "mte3_ratio": 0.1, "vec_ratio": 0.25, "scalar_ratio": 0.05}
        for name, sids, labels, facts in [
            ("mte2_stall + P1", ["P1"], ["mte2_stall"], demo_facts),
            ("no_overlap + P1 (facts dedup)", ["P1"], ["mte2_stall", "no_overlap"], demo_facts),
            ("unrelated A1", ["A1"], ["mte2_stall"], demo_facts),
            ("open_exploration (empty strat)", [], ["mte2_stall"], demo_facts),
            ("no diagnosis", ["P1"], [], demo_facts),
            ("bus_contention sum (P28)", ["P28"], ["bus_contention"],
             {"mte2_ratio": 0.3, "mte3_ratio": 0.3}),
            ("addressed_cap clamp", ["P1"], ["mte2_stall"],
             {"mte2_ratio": 0.95}),
        ]:
            g = compute_expected_gain(sids, facts, labels)
            LOGGER.info("  %-40s → expected_gain = %.3f", name, g)
