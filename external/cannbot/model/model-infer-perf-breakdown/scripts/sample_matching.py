# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Sample-driven layer detection core.

无 IO 的纯函数 + dataclass。调用方（detect_structure.py）负责 load
raw_ops.json / structure_spec.json，把数据传进来；本模块只算。

主入口：
    run_sample_mode(ops, structure_spec, samples, strict_composition=False)
        -> SampleDraft（包含 components、unmatched_regions、warnings、
                       validation、samples_used）

子函数（按调用顺序）：
    extract_fingerprint(ops, lo, hi, k=3) -> Fingerprint
    build_prefix_counts(ops) -> 用于 O(distinct) Jaccard
    scan_seeds(prefix, fingerprint, direction) -> [(direction, pos)]
    match_instances(ops, prefix, fingerprint, seeds, expected_len)
                                              -> [InstanceCandidate]
    endpoint_adjust(ops, prefix, fingerprint, candidate) -> InstanceCandidate
    nms_instances(candidates, overlap=0.30) -> [Instance]
    arbitrate_overlaps(instances_by_component) -> [Instance]
    classify_unmatched(instances, total_ops, repeat_threshold=2)
                                              -> [UnmatchedRegion]
    assemble_layers(instances, structure_spec) -> (layers, warnings)
    build_validation(structure_spec, instances) -> dict

阈值常量集中在文件顶部，便于回归时调。
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

K = 3                          # head / tail 指纹窗口大小
SEED_THRESHOLD = 0.8           # head/tail Jaccard 种子阈值
ACCEPT_THRESHOLD = 0.85        # body_ms Jaccard 验收阈值
MICROADJUST_LOWER = 0.70       # body_ms 落在 [0.70, 0.85) 触发端点微调
SHAPE_CONFIDENCE_THRESHOLD = 0.75    # body_shape_ms < 0.75 → low_shape_confidence
LENGTH_BAND = (0.85, 1.15)     # tail / head 窗口落在 [L*0.85, L*1.15]
ENDPOINT_RADIUS = 3            # 端点微调 ±K
NMS_OVERLAP = 0.30             # 区间重叠 > 30% 丢弃后者
ARBITRATE_SCORE_TOLERANCE = 0.05      # 两个 component 差 < 0.05 进入 shape 仲裁
ARBITRATE_SHAPE_TOLERANCE = 0.02      # shape 也差 < 0.02 → ambiguous_match
SUSPECTED_UNDECLARED_MIN_REGIONS = 2  # 末尾 ≥ N 个相似 region 报警
SUSPECTED_UNDECLARED_SIMILARITY = 0.80  # 区间长度互相 ≥ 80% → 视为重复 region
LARGE_UNMATCHED_REGION_OPS = 30       # 单段 inter_layer_region > N ops → 升级为 suspected
# aux 时间脱节体检：形态无关，只度量"并入辅流是否撑大 component 包络"。
AUX_DISPLACEMENT_RATIO_LIMIT = 1.0    # 单 occurrence 膨胀比 (total_span-primary_span)/primary_span 超此 → 显著
AUX_DISPLACEMENT_FRACTION = 0.5       # 一条 aux 流多数(>此比例)occurrence 脱节 → 判流级 displaced
# 判据全用相对/结构量（inflation 比、流级多数），不设绝对时间阈值——避免量级过拟合。

AUXILIARY_NAMES = {
    "Cast", "Reshape", "Transpose", "Contiguous",
    "DequantSwiglu", "DequantBmm", "Slice", "Concat",
}


@dataclass(frozen=True)
class Fingerprint:
    """单个 sample 的指纹"""
    component: str
    op_range: tuple[int, int]
    length: int
    head_ms: dict[tuple[str, str], int]   # Counter on (normalized_name, canon_shape)
    tail_ms: dict[tuple[str, str], int]
    body_ms: dict[str, int]               # Counter on normalized_name
    body_shape_ms: dict[tuple[str, str], int]


@dataclass
class InstanceCandidate:
    component: str
    start: int
    end: int
    score_body: float
    score_body_shape: float
    score_head: float
    score_tail: float
    seed_direction: str
    endpoint_adjusted: bool


@dataclass
class Instance:
    """通过 NMS 仲裁后的最终 instance"""
    component: str
    start: int
    end: int
    layer_idx: int = -1
    phase: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    seed_direction: str = "head"
    endpoint_adjusted: bool = False


@dataclass
class UnmatchedRegion:
    start: int
    end: int
    # classification ∈ pre_arch / post_arch / inter_layer_region /
    # suspected_undeclared_component / intra_layer_gap / intra_layer_outlier
    classification: str


@dataclass
class DraftWarning:
    code: str
    message: str
    extra: dict = field(default_factory=dict)


@dataclass
class SampleDraft:
    samples_used: list[dict]
    components: list[Instance]
    unmatched_regions: list[UnmatchedRegion]
    warnings: list[DraftWarning]
    validation: dict


@dataclass
class StreamSegment:
    stream_id: str
    role: str
    op_indices: list[int]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamComponent:
    component_id: str
    type: str
    phase: str
    layer_idx: int
    occurrence_idx: int
    primary_stream_id: str
    op_indices: list[int]
    stream_segments: list[StreamSegment]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamSampleDraft:
    structure_spec: dict
    samples_used: list[dict]
    components: list[StreamComponent]
    op_to_component: dict[str, str]
    unmatched_op_indices: list[int]
    unmatched_stream_segments: list[dict]
    warnings: list[DraftWarning]
    validation: dict
    displaced_streams: list = field(default_factory=list)   # [(component_type, stream_id)]


# 其余函数将在 Task 5+ 添加
def canon_shape(op: dict) -> str:
    """规范化 input_shapes + output_shapes 为单字符串。

    跨 run 稳定指纹的关键：去空格、统一定界符、合并 in/out。
    """
    ins = (op.get("input_shapes") or "").replace(" ", "").rstrip(";")
    outs = (op.get("output_shapes") or "").replace(" ", "").rstrip(";")
    return f"{ins}|{outs}"


def _name_of(op: dict) -> str:
    return op.get("normalized_name") or op.get("name", "")


def _op_index(op: dict, fallback: int) -> int:
    idx = op.get("index", fallback)
    return fallback if idx is None else int(idx)


def _stream_id_of(op: dict) -> str:
    sid = op.get("stream_id")
    return "unknown" if sid in (None, "") else str(sid)


def _start_of(op: dict, fallback: int) -> float:
    st = op.get("start_time_us")
    if st is None:
        return float(fallback)
    try:
        return float(st)
    except (TypeError, ValueError):
        return float(fallback)


def build_stream_index(ops: list[dict]) -> tuple[dict[str, list[int]], dict[int, dict]]:
    """Build stream-local op order.

    Returns:
      streams: stream_id -> global op indices sorted by (start_time_us, index)
      op_to_stream_pos: op index -> {stream_id, pos}
    """
    streams: dict[str, list[int]] = {}
    for pos, op in enumerate(ops):
        idx = _op_index(op, pos)
        streams.setdefault(_stream_id_of(op), []).append(idx)

    pos_by_idx = {_op_index(op, pos): pos for pos, op in enumerate(ops)}
    for _sid, indices in streams.items():
        indices.sort(key=lambda i: (_start_of(ops[pos_by_idx[i]], pos_by_idx[i]), i))

    op_to_stream_pos: dict[int, dict] = {}
    for sid, indices in streams.items():
        for p, idx in enumerate(indices):
            op_to_stream_pos[idx] = {"stream_id": sid, "pos": p}
    return streams, op_to_stream_pos


def _ops_by_index(ops: list[dict]) -> dict[int, dict]:
    return {_op_index(op, pos): op for pos, op in enumerate(ops)}


def _counts_for_indices(ops_by_idx: dict[int, dict], indices: list[int]) -> Counter:
    return Counter(_name_of(ops_by_idx[i]) for i in indices)


def _shape_counts_for_indices(ops_by_idx: dict[int, dict], indices: list[int]) -> Counter:
    return Counter((_name_of(ops_by_idx[i]), canon_shape(ops_by_idx[i])) for i in indices)


def _sequence_score(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if len(a) != len(b) or not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _stream_window_scores(ops_by_idx: dict[int, dict],
                          window: list[int],
                          sample_names: list[str],
                          sample_counts: Counter,
                          sample_shape_counts: Counter) -> dict[str, float]:
    names = [_name_of(ops_by_idx[i]) for i in window]
    return {
        "stream_sequence": _sequence_score(names, sample_names),
        "stream_body": multiset_jaccard(
            _counts_for_indices(ops_by_idx, window), sample_counts),
        "stream_shape": multiset_jaccard(
            _shape_counts_for_indices(ops_by_idx, window), sample_shape_counts),
    }


def _scan_stream_segment(ops_by_idx: dict[int, dict],
                         stream_indices: list[int],
                         sample_indices: list[int]) -> list[StreamSegment]:
    if not sample_indices or len(sample_indices) > len(stream_indices):
        return []
    sample_names = [_name_of(ops_by_idx[i]) for i in sample_indices]
    sample_counts = _counts_for_indices(ops_by_idx, sample_indices)
    sample_shape_counts = _shape_counts_for_indices(ops_by_idx, sample_indices)
    sample_len = len(sample_indices)
    out: list[StreamSegment] = []
    for start in range(0, len(stream_indices) - sample_len + 1):
        window = stream_indices[start:start + sample_len]
        scores = _stream_window_scores(
            ops_by_idx, window, sample_names, sample_counts, sample_shape_counts)
        if (scores["stream_sequence"] >= 0.75
                and scores["stream_body"] >= 0.85
                and scores["stream_shape"] >= SHAPE_CONFIDENCE_THRESHOLD):
            out.append(StreamSegment(
                stream_id=_stream_id_of(ops_by_idx[window[0]]),
                role="",
                op_indices=list(window),
                scores=scores,
            ))
    return out


def _low_shape_stream_matches(ops_by_idx: dict[int, dict],
                              stream_indices: list[int],
                              sample_indices: list[int]) -> list[dict]:
    if not sample_indices or len(sample_indices) > len(stream_indices):
        return []
    sample_names = [_name_of(ops_by_idx[i]) for i in sample_indices]
    sample_counts = _counts_for_indices(ops_by_idx, sample_indices)
    sample_shape_counts = _shape_counts_for_indices(ops_by_idx, sample_indices)
    sample_len = len(sample_indices)
    out = []
    for start in range(0, len(stream_indices) - sample_len + 1):
        window = stream_indices[start:start + sample_len]
        scores = _stream_window_scores(
            ops_by_idx, window, sample_names, sample_counts, sample_shape_counts)
        if (scores["stream_sequence"] >= 0.75
                and scores["stream_body"] >= 0.85
                and scores["stream_shape"] < SHAPE_CONFIDENCE_THRESHOLD):
            out.append({"op_indices": list(window), "scores": scores})
    return out


def _segment_time_bounds(seg: StreamSegment,
                         ops_by_idx: dict[int, dict]) -> tuple[float, float]:
    starts = []
    ends = []
    for idx in seg.op_indices:
        op = ops_by_idx[idx]
        st = _start_of(op, idx)
        dur = float(op.get("duration_us") or 0.0)
        starts.append(st)
        ends.append(st + dur)
    if not starts:
        return 0.0, 0.0
    return min(starts), max(ends)


def _segment_pair_metrics(primary_seg: StreamSegment,
                          aux_seg: StreamSegment,
                          ops_by_idx: dict[int, dict]) -> dict[str, float]:
    p_start, p_end = _segment_time_bounds(primary_seg, ops_by_idx)
    a_start, a_end = _segment_time_bounds(aux_seg, ops_by_idx)
    overlap = max(0.0, min(p_end, a_end) - max(p_start, a_start))
    if overlap > 0:
        gap = 0.0
    else:
        gap = min(abs(a_start - p_end), abs(p_start - a_end))
    center_delta = abs(((p_start + p_end) / 2.0) - ((a_start + a_end) / 2.0))
    return {
        "pair_overlap_us": overlap,
        "pair_gap_us": gap,
        "pair_center_delta_us": center_delta,
    }


def _match_auxiliary_segments(primary_segs: list[StreamSegment],
                              aux_cands: list[StreamSegment],
                              ops_by_idx: dict[int, dict]
                              ) -> tuple[dict[int, tuple[int, StreamSegment, dict]],
                                         dict[int, dict]]:
    """全局最优一一匹配 primary occurrence ↔ aux segment。

    对每条 aux 流，在 primary occurrence 与该流的 aux segment 之间求一组一一匹配，使匹配数
    最多、总时间距离最小。当辅流步距小于主流、逐层领先时，最近的 aux 会先被前段 occurrence
    占走、后段只能配到远期 segment——按全局最优分配可避免这种错配，逐层单调时自然退化为序号
    对齐（occ k ↔ aux 第 k 段）。

    算法：代价 = 两段中心时间距离 pair_center_delta_us（一维点匹配）。一维点匹配的最优解
    非交叉（保序）只在两序列各自按"中心时间"排序时成立；滑窗产出的 segment 按 *起点* 升序、
    duration 非单调时起点序 ≠ 中心序，会破坏该前提。因此入口先按中心时间稳定排序 primary/aux，
    再跑非交叉 DP，回溯后映射回原始 occurrence/cand 位置。目标按 (-matched_count, total_cost)
    字典序最小化：先最大化匹配数，平局再比总代价。

    cost 用 center_delta：等长窗口下它单调代表 overlap，且辅流通常短于 primary（重叠的 aux
    中心必落在 primary 跨度内 → center_delta 小），故并发优先自然成立；仅当 aux 段反常地长于
    primary 时，一个中心更近的脱节段可能赢过重叠段（aux 不长于 primary 时不出现）。

    返回:
      assignment: {occ_idx -> (cand_pos, seg, pair_metrics)}
      ambiguous:  {occ_idx -> ambiguity_dict}（全局最优非唯一时，该 occ 的配对可替换）
    """
    n, m = len(primary_segs), len(aux_cands)
    if n == 0 or m == 0:
        return {}, {}

    def _center(seg):
        a, b = _segment_time_bounds(seg, ops_by_idx)
        return (a + b) / 2.0

    # 按中心时间稳定排序，使一维点距离严格满足 Monge（非交叉 DP 才是全局最优）。
    p_order = sorted(range(n), key=lambda i: (_center(primary_segs[i]), i))
    a_order = sorted(range(m), key=lambda j: (_center(aux_cands[j]), j))
    primary_sorted = [primary_segs[i] for i in p_order]
    aux_sorted = [aux_cands[j] for j in a_order]
    metrics = [[_segment_pair_metrics(primary_sorted[i], aux_sorted[j], ops_by_idx)
                for j in range(m)] for i in range(n)]
    cost = [[metrics[i][j]["pair_center_delta_us"] for j in range(m)] for i in range(n)]

    def _rank(state):
        return (-state[0], state[1])

    def _best_cell(dp, i, j, blocked_col):
        """单格 DP 最优决策；扁平化以控制嵌套深度。"""
        best, ch = dp[i - 1][j], "skip_occ"
        if _rank(dp[i][j - 1]) < _rank(best):
            best, ch = dp[i][j - 1], "skip_seg"
        if j - 1 == blocked_col:
            return best, ch
        cand = (dp[i - 1][j - 1][0] + 1,
                dp[i - 1][j - 1][1] + cost[i - 1][j - 1])
        if _rank(cand) < _rank(best):
            best, ch = cand, "match"
        return best, ch

    def _solve(blocked_col):
        """非交叉 DP 最优匹配；blocked_col(已排序列索引) 被禁用，用于唯一性检验。"""
        dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
        choice = [[""] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j], choice[i][j] = _best_cell(dp, i, j, blocked_col)
        pairs = {}
        i, j = n, m
        while i > 0 and j > 0:
            if choice[i][j] == "match":
                pairs[i - 1] = j - 1
                i -= 1
                j -= 1
            elif choice[i][j] == "skip_occ":
                i -= 1
            else:
                j -= 1
        return dp[n][m], pairs

    opt, pairs = _solve(blocked_col=-1)

    # 映射回原始下标
    assignment: dict[int, tuple[int, StreamSegment, dict]] = {}
    for si, sj in pairs.items():
        occ, pos = p_order[si], a_order[sj]
        assignment[occ] = (pos, aux_cands[pos], metrics[si][sj])

    # ambiguity: 禁用某 occ 选中的列后重跑，最优 (count,cost) 不变 → 该配对全局可替换。
    ambiguous: dict[int, dict] = {}
    for si, sj in pairs.items():
        opt2, _ = _solve(blocked_col=sj)
        if opt2[0] == opt[0] and abs(opt2[1] - opt[1]) < 1e-9:
            occ, pos = p_order[si], a_order[sj]
            alt_sj = next((k for k in sorted(range(m), key=lambda k: (cost[si][k], k))
                           if k != sj), None)
            ambiguous[occ] = {
                "selected_op_indices": assignment[occ][1].op_indices,
                "alternative_op_indices": (
                    aux_cands[a_order[alt_sj]].op_indices if alt_sj is not None else []),
                "selected_pair_metrics": assignment[occ][2],
            }
    return assignment, ambiguous


def _normalize_stream_ack(sample_ack: dict) -> dict:
    if sample_ack.get("schema_version") == "stream_sample_ack.v1":
        return sample_ack
    raise ValueError(
        "stream sample mode requires sample_ack.json schema_version=stream_sample_ack.v1")


def extract_fingerprint(ops: list[dict], component: str,
                        lo: int, hi: int, k: int = K) -> Fingerprint:
    """从 [lo, hi] 闭区间提取 4 个指纹（head_ms / tail_ms / body_ms / body_shape_ms）。"""
    if lo < 0 or hi >= len(ops) or lo > hi:
        raise ValueError(f"bad sample range [{lo}, {hi}] for ops of len {len(ops)}")
    if hi - lo + 1 < 2 * k:
        raise ValueError(f"sample length {hi - lo + 1} < 2K ({2 * k})")

    window = ops[lo:hi + 1]
    head_ms = Counter((_name_of(o), canon_shape(o)) for o in window[:k])
    tail_ms = Counter((_name_of(o), canon_shape(o)) for o in window[-k:])
    body_ms = Counter(_name_of(o) for o in window)
    body_shape_ms = Counter((_name_of(o), canon_shape(o)) for o in window)

    return Fingerprint(
        component=component,
        op_range=(lo, hi),
        length=hi - lo + 1,
        head_ms=dict(head_ms),
        tail_ms=dict(tail_ms),
        body_ms=dict(body_ms),
        body_shape_ms=dict(body_shape_ms),
    )


def build_prefix_counts(ops: list[dict]) -> dict[str, list[int]]:
    """每个 normalized_name 一个 prefix-sum 数组。

    prefix[name][i] = count of `name` in ops[0:i]，i ∈ [0, N]。
    单次 multiset Jaccard 取 [s, e+1] - [s] 即得窗口内 count。
    """
    names = set(_name_of(o) for o in ops)
    out: dict[str, list[int]] = {n: [0] * (len(ops) + 1) for n in names}
    for i, op in enumerate(ops):
        name = _name_of(op)
        for n in names:
            out[n][i + 1] = out[n][i] + (1 if n == name else 0)
    return out


def window_name_counts(prefix: dict[str, list[int]],
                       s: int, e: int) -> dict[str, int]:
    """O(distinct_names) 取 [s, e] 闭区间内的 name multiset。"""
    return {
        n: prefix[n][e + 1] - prefix[n][s]
        for n in prefix
        if prefix[n][e + 1] - prefix[n][s] > 0
    }


def multiset_jaccard(a, b) -> float:
    """两个 dict-like multiset 的 Jaccard = sum(min) / sum(max)。

    支持 a / b 是 dict 或 Counter。空多集对空多集 → 1.0；
    一方非空一方空 → 0.0。
    """
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


def window_shape_counts(ops: list[dict], s: int, e: int) -> dict[tuple[str, str], int]:
    """[s, e] 闭区间内 (name, canon_shape) 多集——给 head_ms/tail_ms/body_shape_ms 用。"""
    c: Counter = Counter()
    for i in range(s, e + 1):
        c[(_name_of(ops[i]), canon_shape(ops[i]))] += 1
    return dict(c)


def scan_head_seeds(ops: list[dict], fp: Fingerprint, k: int = K) -> list[int]:
    """主方向：扫所有起点 s 使 window[s:s+k] 的 (name, shape) 多集与 head_ms Jaccard ≥ SEED_THRESHOLD。

    返回 s 列表（升序），每个 s 是候选 instance 起点。
    """
    n = len(ops)
    seeds = []
    for s in range(n - k + 1):
        wc = window_shape_counts(ops, s, s + k - 1)
        if multiset_jaccard(wc, fp.head_ms) >= SEED_THRESHOLD:
            seeds.append(s)
    return seeds


def scan_tail_seeds(ops: list[dict], fp: Fingerprint, k: int = K) -> list[int]:
    """兜底方向：扫所有终点 e 使 window[e-k+1:e+1] 与 tail_ms Jaccard ≥ SEED_THRESHOLD。

    返回 e 列表（升序）。只在 head seeds 全空时调用。
    """
    n = len(ops)
    seeds = []
    for e in range(k - 1, n):
        wc = window_shape_counts(ops, e - k + 1, e)
        if multiset_jaccard(wc, fp.tail_ms) >= SEED_THRESHOLD:
            seeds.append(e)
    return seeds


def _body_jaccard(prefix, fp: Fingerprint, s: int, e: int) -> float:
    return multiset_jaccard(window_name_counts(prefix, s, e), fp.body_ms)


def _body_shape_jaccard(ops, fp: Fingerprint, s: int, e: int) -> float:
    return multiset_jaccard(window_shape_counts(ops, s, e), fp.body_shape_ms)


def scan_body_seeds(ops, prefix, fp: Fingerprint, step: int = 1,
                    threshold: float = MICROADJUST_LOWER) -> list[int]:
    """body_ms Jaccard 滑窗：固定窗长 L = fp.length，按 step 扫所有起点。

    body 多集是 instance validity 的真信号（与 head/tail boundary 是否被
    fused kernel 吞掉无关）。head/tail seed 卡掉的 instance（典型 case：
    pre-norm 架构入口 RmsNorm 被前一层 InplaceAddRmsNorm 吸收）由此路径
    找回，再由 endpoint_adjust 精化 (s, e)。

    threshold 取 MICROADJUST_LOWER 与 head-seed 路径一致：seed 宽 →
    endpoint_adjust 收紧 → 验收 ≥ ACCEPT_THRESHOLD。
    """
    n = len(ops)
    window_len = fp.length
    if window_len > n:
        return []
    return [
        s for s in range(0, n - window_len + 1, step)
        if _body_jaccard(prefix, fp, s, s + window_len - 1) >= threshold
    ]


def _best_tail_in_band(ops, fp: Fingerprint, start: int, n: int,
                      k: int = K) -> Optional[tuple[int, float]]:
    """在 [start + L*0.85 - k + 1, start + L*1.15 - k + 1) 内找 tail Jaccard 最高的 e。"""
    lo_end = start + int(fp.length * LENGTH_BAND[0]) - 1
    hi_end = min(n - 1, start + int(fp.length * LENGTH_BAND[1]) - 1)
    best: Optional[tuple[int, float]] = None
    for e in range(max(lo_end, start + k - 1), hi_end + 1):
        wc = window_shape_counts(ops, e - k + 1, e)
        j = multiset_jaccard(wc, fp.tail_ms)
        if j >= SEED_THRESHOLD and (best is None or j > best[1]):
            best = (e, j)
    return best


def _best_head_in_band(ops, fp: Fingerprint, end: int,
                      k: int = K) -> Optional[tuple[int, float]]:
    """tail-seeded 兜底：给定 e，在 [e - L*1.15 + 1, e - L*0.85 + 1] 内找 head 最佳起点。"""
    lo_start = max(0, end - int(fp.length * LENGTH_BAND[1]) + 1)
    hi_start = max(0, end - int(fp.length * LENGTH_BAND[0]) + 1)
    best: Optional[tuple[int, float]] = None
    for s in range(lo_start, hi_start + 1):
        if s + k - 1 > end:
            continue
        wc = window_shape_counts(ops, s, s + k - 1)
        j = multiset_jaccard(wc, fp.head_ms)
        if j >= SEED_THRESHOLD and (best is None or j > best[1]):
            best = (s, j)
    return best


def endpoint_adjust(ops, prefix, fp: Fingerprint, s: int, e: int) -> tuple[int, int, float]:
    """在 (s, e) 周围 ±ENDPOINT_RADIUS 内枚举 (2*r+1)² 组合，返回 body_ms 最高的 (s, e, score)。

    用于 body_ms ∈ [MICROADJUST_LOWER, ACCEPT_THRESHOLD) 时的救援。
    """
    n = len(ops)
    radius = ENDPOINT_RADIUS
    best = (s, e, _body_jaccard(prefix, fp, s, e))
    for ds in range(-radius, radius + 1):
        for de in range(-radius, radius + 1):
            ns, ne = s + ds, e + de
            if ns < 0 or ne >= n or ns >= ne:
                continue
            score = _body_jaccard(prefix, fp, ns, ne)
            if score > best[2]:
                best = (ns, ne, score)
    return best


@dataclass
class SeedWindow:
    """一个待验收的 seed 窗口：起止行 + head/tail Jaccard + 种子方向。"""
    start: int
    end: int
    head_j: float
    tail_j: float
    direction: str


def _accept_seeded_candidate(ops, prefix, fp: Fingerprint,
                             seed: SeedWindow) -> Optional[InstanceCandidate]:
    """head/tail seed 共用的验收逻辑：body_j 不足时端点微调，仍不足则丢弃。

    返回验收通过的 InstanceCandidate，否则 None。
    """
    s, e = seed.start, seed.end
    head_j, tail_j = seed.head_j, seed.tail_j
    body_j = _body_jaccard(prefix, fp, s, e)
    adjusted = False
    if body_j < ACCEPT_THRESHOLD:
        if body_j < MICROADJUST_LOWER:
            return None
        ns, ne, body_j = endpoint_adjust(ops, prefix, fp, s, e)
        if body_j < ACCEPT_THRESHOLD:
            return None
        s, e = ns, ne
        adjusted = True
        head_j = multiset_jaccard(window_shape_counts(ops, s, min(s + K - 1, e)), fp.head_ms)
        tail_j = multiset_jaccard(window_shape_counts(ops, max(e - K + 1, s), e), fp.tail_ms)
    body_shape_j = _body_shape_jaccard(ops, fp, s, e)
    return InstanceCandidate(
        component=fp.component, start=s, end=e,
        score_body=body_j, score_body_shape=body_shape_j,
        score_head=head_j, score_tail=tail_j,
        seed_direction=seed.direction, endpoint_adjusted=adjusted,
    )


def _head_seed_candidates(ops, prefix, fp: Fingerprint, n: int) -> list[InstanceCandidate]:
    accepted: list[InstanceCandidate] = []
    for s in scan_head_seeds(ops, fp, k=K):
        head_j = multiset_jaccard(window_shape_counts(ops, s, s + K - 1), fp.head_ms)
        tail = _best_tail_in_band(ops, fp, s, n, k=K)
        if tail is None:
            continue
        e, tail_j = tail
        cand = _accept_seeded_candidate(
            ops, prefix, fp, SeedWindow(s, e, head_j, tail_j, "head"))
        if cand is not None:
            accepted.append(cand)
    return accepted


def _body_seed_candidates(ops, prefix, fp: Fingerprint, n: int) -> list[InstanceCandidate]:
    accepted: list[InstanceCandidate] = []
    for s in scan_body_seeds(ops, prefix, fp, step=1):
        e = s + fp.length - 1
        if e >= n:
            continue
        ns, ne, body_j = endpoint_adjust(ops, prefix, fp, s, e)
        if body_j < ACCEPT_THRESHOLD:
            continue
        head_j = multiset_jaccard(window_shape_counts(ops, ns, min(ns + K - 1, ne)), fp.head_ms)
        tail_j = multiset_jaccard(window_shape_counts(ops, max(ne - K + 1, ns), ne), fp.tail_ms)
        body_shape_j = _body_shape_jaccard(ops, fp, ns, ne)
        accepted.append(InstanceCandidate(
            component=fp.component, start=ns, end=ne,
            score_body=body_j, score_body_shape=body_shape_j,
            score_head=head_j, score_tail=tail_j,
            seed_direction="body", endpoint_adjusted=(ns != s or ne != e),
        ))
    return accepted


def _tail_seed_candidates(ops, prefix, fp: Fingerprint) -> list[InstanceCandidate]:
    accepted: list[InstanceCandidate] = []
    for e in scan_tail_seeds(ops, fp, k=K):
        tail_j = multiset_jaccard(window_shape_counts(ops, e - K + 1, e), fp.tail_ms)
        head = _best_head_in_band(ops, fp, e, k=K)
        if head is None:
            continue
        s, head_j = head
        cand = _accept_seeded_candidate(
            ops, prefix, fp, SeedWindow(s, e, head_j, tail_j, "tail"))
        if cand is not None:
            accepted.append(cand)
    return accepted


def match_component(ops, prefix, fp: Fingerprint) -> list[InstanceCandidate]:
    """主入口：对一个 component 的指纹扫全文，输出候选 instance 列表（已 NMS）。

    body_ms Jaccard 是 instance validity 的真信号——multiset 与 boundary
    kernel 是否被融合无关。head/tail 都只是加速 anchor，可能被 fused boundary
    kernel（如入口 RmsNorm 被前一层 residual-add 融合吃掉）卡掉。

    路径（执行顺序，并行附加候选，最后 NMS 去重）：
      1. head-seed（快，head 完整时给出精确起点 + length-band tail；
         与 body-seed 重叠时由 NMS 按 body_j 合并，相同分时插入序在前的胜出）
      2. body-seed 滑窗（O(N) step=1）：扫所有 s，body_j ≥ MICROADJUST_LOWER
         即 seed，endpoint_adjust 精化 (s, e)，验收 body_j ≥ ACCEPT_THRESHOLD。
         覆盖 head 被融合吞掉的 instance。
      3. tail-seed（兜底；仅 head 和 body 都空时尝试）
    """
    n = len(ops)
    accepted = _head_seed_candidates(ops, prefix, fp, n)
    accepted.extend(_body_seed_candidates(ops, prefix, fp, n))

    if accepted:
        return nms_candidates(accepted)

    return _tail_seed_candidates(ops, prefix, fp)


def _overlap_ratio(a: InstanceCandidate, b: InstanceCandidate) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if lo > hi:
        return 0.0
    inter = hi - lo + 1
    shorter = min(a.end - a.start + 1, b.end - b.start + 1)
    return inter / shorter


def nms_candidates(cands: list[InstanceCandidate],
                   overlap: float = NMS_OVERLAP) -> list[InstanceCandidate]:
    """按 score_body 降序贪心，与已选区间重叠 > overlap 的丢弃。"""
    ordered = sorted(cands, key=lambda c: -c.score_body)
    kept: list[InstanceCandidate] = []
    for c in ordered:
        if any(_overlap_ratio(c, k) > overlap for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda c: c.start)


def arbitrate_overlaps(all_cands_by_component: dict[str, list[InstanceCandidate]]
                       ) -> tuple[list[Instance], list[DraftWarning]]:
    """跨 component 仲裁：

    1. 把所有候选混进同一池，按 score_body 降序贪心
    2. 与已选区间重叠 > NMS_OVERLAP 时：
       - 如果是同 component → 直接丢弃后者（NMS 已做完，理论上不会进来）
       - 如果是异 component → 比 score_body：差 < 0.05 比 score_body_shape；
         差 < 0.02 → ambiguous_match 警告 + 都保留？不，选高的，
         同时记 warning 让 AI 提示用户重挑 sample。
    """
    pool = []
    for cands in all_cands_by_component.values():
        pool.extend(cands)
    pool.sort(key=lambda c: -c.score_body)

    kept: list[InstanceCandidate] = []
    warnings: list[DraftWarning] = []
    for c in pool:
        conflict = next((k for k in kept if _overlap_ratio(c, k) > NMS_OVERLAP), None)
        if conflict is None:
            kept.append(c)
            continue
        if conflict.component == c.component:
            continue
        # 异 component 冲突：比较 body / body_shape
        d_body = abs(conflict.score_body - c.score_body)
        if d_body < ARBITRATE_SCORE_TOLERANCE:
            d_shape = abs(conflict.score_body_shape - c.score_body_shape)
            if d_shape < ARBITRATE_SHAPE_TOLERANCE:
                warnings.append(DraftWarning(
                    code="ambiguous_match",
                    message=(f"component {conflict.component} vs {c.component} "
                             f"at rows [{c.start},{c.end}] body 差 {d_body:.3f} "
                             f"shape 差 {d_shape:.3f}"),
                    extra={"component_a": conflict.component,
                           "component_b": c.component,
                           "row_range": [c.start, c.end]},
                ))
        # 选 score 更高的；conflict 已经在 kept 里且分数 ≥ c（因为降序），保留 conflict

    instances = [
        Instance(component=c.component, start=c.start, end=c.end,
                 scores={"body_ms": c.score_body, "body_shape_ms": c.score_body_shape,
                         "head_ms": c.score_head, "tail_ms": c.score_tail},
                 seed_direction=c.seed_direction,
                 endpoint_adjusted=c.endpoint_adjusted)
        for c in sorted(kept, key=lambda x: x.start)
    ]
    return instances, warnings


def _are_lengths_similar(lens: list[int]) -> bool:
    if len(lens) < SUSPECTED_UNDECLARED_MIN_REGIONS:
        return False
    mn, mx = min(lens), max(lens)
    return mn / mx >= SUSPECTED_UNDECLARED_SIMILARITY if mx else False


def _global_gap_regions(sorted_inst: list[Instance], n: int) -> list[UnmatchedRegion]:
    """构造 pre_arch / inter_layer_region / post_arch 全局空隙。"""
    regions: list[UnmatchedRegion] = []
    if sorted_inst[0].start > 0:
        regions.append(UnmatchedRegion(0, sorted_inst[0].start - 1, "pre_arch"))
    for a, b in zip(sorted_inst, sorted_inst[1:]):
        if a.end + 1 <= b.start - 1:
            regions.append(UnmatchedRegion(a.end + 1, b.start - 1, "inter_layer_region"))
    if sorted_inst[-1].end < n - 1:
        regions.append(UnmatchedRegion(sorted_inst[-1].end + 1, n - 1, "post_arch"))
    return regions


def _collect_tail_regions(regions: list[UnmatchedRegion]) -> list[UnmatchedRegion]:
    """从尾部回收连续的 post_arch / inter_layer_region（最多 5 段）。"""
    tail_regions: list[UnmatchedRegion] = []
    for r in reversed(regions):
        if r.classification not in ("post_arch", "inter_layer_region"):
            break
        tail_regions.append(r)
        if len(tail_regions) > 4:
            break
    tail_regions.reverse()
    return tail_regions


def _flag_tail_repeats(regions: list[UnmatchedRegion]) -> Optional[DraftWarning]:
    """末尾连续相似长度的重复 region → suspected_undeclared_component。"""
    tail_regions = _collect_tail_regions(regions)
    lens = [r.end - r.start + 1 for r in tail_regions]
    if not _are_lengths_similar(lens):
        return None
    for r in tail_regions:
        r.classification = "suspected_undeclared_component"
    return DraftWarning(
        code="suspected_undeclared_component",
        message=(f"末尾发现 {len(tail_regions)} 个长度 ~{lens[0]} 的重复区间未匹配，"
                 f"可能漏声明 MTP / lm_head 等 component"),
        extra={"row_ranges": [[r.start, r.end] for r in tail_regions]},
    )


def _large_inter_layer_regions(regions: list[UnmatchedRegion]) -> list[UnmatchedRegion]:
    large_regions: list[UnmatchedRegion] = []
    for r in regions:
        if r.classification != "inter_layer_region":
            continue
        if (r.end - r.start + 1) > LARGE_UNMATCHED_REGION_OPS:
            large_regions.append(r)
    return large_regions


def _flag_large_regions(regions: list[UnmatchedRegion]) -> Optional[DraftWarning]:
    """单段 inter_layer_region > 阈值 → 升级 suspected_undeclared_component。"""
    large_regions = _large_inter_layer_regions(regions)
    if not large_regions:
        return None
    for r in large_regions:
        r.classification = "suspected_undeclared_component"
    return DraftWarning(
        code="large_unmatched_region",
        message=(f"{len(large_regions)} 段 inter_layer_region > "
                 f"{LARGE_UNMATCHED_REGION_OPS} ops 被升级为 "
                 f"suspected_undeclared_component；可能漏声明 component "
                 f"(sampler / lm_head / draft head 等)，回 Phase 0a/0b 补"),
        extra={
            "threshold": LARGE_UNMATCHED_REGION_OPS,
            "regions": [
                {"op_range": [r.start, r.end], "length": r.end - r.start + 1}
                for r in large_regions
            ],
        },
    )


def classify_unmatched(instances: list[Instance], n: int
                       ) -> tuple[list[UnmatchedRegion], list[DraftWarning]]:
    """把 instance 之间的空隙分类为 pre_arch / post_arch / inter_layer_region.

    intra_layer_gap / intra_layer_outlier 在 assemble_layers 阶段才能算（需要先拼
    layer），这里只处理 component 间的全局空隙。
    """
    if not instances:
        return [], []
    sorted_inst = sorted(instances, key=lambda i: i.start)
    regions = _global_gap_regions(sorted_inst, n)
    warnings: list[DraftWarning] = []

    # 末尾"重复 region"检测必须先于 LARGE_UNMATCHED_REGION_OPS：尾部重复模式
    # 比"单段巨大"更有信息量。
    tail_warning = _flag_tail_repeats(regions)
    if tail_warning is not None:
        warnings.append(tail_warning)

    large_warning = _flag_large_regions(regions)
    if large_warning is not None:
        warnings.append(large_warning)

    return regions, warnings


def _pick_composition_by_lookahead(
    candidates: list[dict], by_start: list[Instance],
    consumed: list[bool], inst_idx: int, n: int,
) -> tuple[dict, bool]:
    """多 composition 共享 first.component 时，扫接下来 K 条未消费 instance
    的 type 多集，挑能被该 multiset 最大覆盖的 composition。

    返回 (chosen, tie)：tie=True 表示最高分仍有多个 composition 并列。
    K 取当前 candidate 的 components 数；若 candidate 间长度不同，分别用各自长度
    取 lookahead 切片。
    """
    if len(candidates) == 1:
        return candidates[0], False
    scores = []
    for comp in candidates:
        need = list(comp["components"])
        pending_types: list[str] = []
        for j in range(inst_idx, n):
            if consumed[j]:
                continue
            pending_types.append(by_start[j].component)
            if len(pending_types) >= len(need):
                break
        remaining = list(pending_types)
        s = 0
        for t in need:
            if t in remaining:
                remaining.remove(t)
                s += 1
        scores.append(s)
    top = max(scores)
    winners = [c for c, s in zip(candidates, scores) if s == top]
    return winners[0], len(winners) > 1


@dataclass
class _AssemblyState:
    """assemble_layers 的可变游标：跨 phase / layer / type 共享。"""
    by_start: list[Instance]
    consumed: list[bool]
    n: int
    inst_idx: int = 0
    next_layer_id: int = 0


def _consume_one_of_type(state: _AssemblyState, ctype: str, phase_name: str) -> bool:
    """从 inst_idx 起顺序消费一个 component==ctype 的未消费 instance；成功返回 True。"""
    for j in range(state.inst_idx, state.n):
        if state.consumed[j] or state.by_start[j].component != ctype:
            continue
        state.by_start[j].layer_idx = state.next_layer_id
        state.by_start[j].phase = phase_name
        state.consumed[j] = True
        if j == state.inst_idx:
            while state.inst_idx < state.n and state.consumed[state.inst_idx]:
                state.inst_idx += 1
        return True
    return False


def _select_composition(state: _AssemblyState, declared: list[dict],
                        phase_name: str) -> tuple[Optional[dict], list[DraftWarning]]:
    """为当前 layer slot 选 composition；返回 (comp, warnings)，comp 为 None 表示跳过。"""
    warnings: list[DraftWarning] = []
    first = state.by_start[state.inst_idx]
    candidates = [c for c in declared if first.component in c["components"]]
    if not candidates:
        warnings.append(DraftWarning(
            code="composition_mismatch",
            message=(f"phase {phase_name} layer {state.next_layer_id}: "
                     f"first instance type={first.component} "
                     f"不属于任何声明 composition"),
            extra={"phase": phase_name,
                   "layer_idx": state.next_layer_id,
                   "unexpected_type": first.component},
        ))
        state.inst_idx += 1
        return None, warnings
    # 多 composition 共享 first.component（如 main phase 同时声明 [mla,dense] 和
    # [mla,moe]）时，靠 lookahead 看接下来 K 条未消费 instance 的 type 多集，选
    # 与之最匹配的那个 composition。
    comp, tie = _pick_composition_by_lookahead(
        candidates, state.by_start, state.consumed, state.inst_idx, state.n)
    if tie:
        warnings.append(DraftWarning(
            code="ambiguous_composition",
            message=(f"phase {phase_name} layer {state.next_layer_id}: "
                     f"type={first.component} 同时属于多个声明 composition，"
                     f"lookahead 仍无法区分"),
            extra={"phase": phase_name,
                   "layer_idx": state.next_layer_id,
                   "type": first.component},
        ))
    return comp, warnings


def _assemble_one_layer(state: _AssemblyState, declared: list[dict],
                        phase_name: str) -> tuple[bool, list[DraftWarning]]:
    """拼一个 layer slot；返回 (emitted, warnings)。emitted=False 表示该 slot 被跳过。"""
    comp, warnings = _select_composition(state, declared, phase_name)
    if comp is None:
        return False, warnings
    for t in list(comp["components"]):
        if _consume_one_of_type(state, t, phase_name):
            continue
        warnings.append(DraftWarning(
            code="composition_mismatch",
            message=(f"phase {phase_name} layer {state.next_layer_id}: "
                     f"composition {comp['components']} 缺 type={t}"),
            extra={"phase": phase_name,
                   "layer_idx": state.next_layer_id,
                   "missing_type": t},
        ))
    state.next_layer_id += 1
    return True, warnings


def _assemble_phase(state: _AssemblyState, phase: dict) -> list[DraftWarning]:
    """拼一个 phase 内声明的所有 layer。"""
    warnings: list[DraftWarning] = []
    declared = phase["layer_compositions"]
    layers_expected = phase["layers"]
    layers_emitted = 0
    for _ in range(layers_expected):
        if state.inst_idx >= state.n:
            warnings.append(DraftWarning(
                code="layer_count_mismatch",
                message=(f"phase {phase['name']} 期望 {layers_expected} 层，"
                         f"实测仅拼出 {layers_emitted}"),
                extra={"phase": phase["name"],
                       "declared": layers_expected,
                       "detected": layers_emitted},
            ))
            break
        emitted, layer_warnings = _assemble_one_layer(state, declared, phase["name"])
        warnings.extend(layer_warnings)
        if emitted:
            layers_emitted += 1
    return warnings


def assemble_layers(instances: list[Instance], structure_spec: dict
                    ) -> tuple[list[Instance], list[DraftWarning]]:
    """row-order composition 推断.

    遍历每个 phase 的声明 layer 数 N：
      - pending = phase row 范围内未消费 instance（按 row 升序）
      - 对每个 layer slot：看 pending[0].component，找匹配的 declared composition
        （component ∈ comp.components 且 comp 与已选 composition 之间可区分）
      - 选中 composition → 从 pending 顺序消费一个该 composition 包含的每个 type
      - 给消费的 instance 填 layer_idx / phase

    返回填好 layer_idx / phase 的 instances + warnings。
    """
    if not structure_spec.get("phases"):
        return instances, []
    sorted_inst = sorted(instances, key=lambda i: i.start)
    state = _AssemblyState(
        by_start=sorted_inst,
        consumed=[False] * len(sorted_inst),
        n=len(sorted_inst),
    )
    warnings: list[DraftWarning] = []
    for phase in structure_spec["phases"]:
        warnings.extend(_assemble_phase(state, phase))

    return [i for i in state.by_start if i.layer_idx >= 0], warnings


def build_validation(structure_spec: dict, instances: list[Instance]
                     ) -> tuple[dict, list[DraftWarning]]:
    """component_count_mismatch 校验.

    某 type 实测 instance 数 ≠ Σ phase.layers × count(type ∈ composition)。
    """
    warnings: list[DraftWarning] = []
    expected_per_type: Counter = Counter()
    for phase in structure_spec.get("phases", []):
        for comp in phase["layer_compositions"]:
            for t in comp["components"]:
                expected_per_type[t] += phase["layers"] // len(phase["layer_compositions"])
                # 上面是粗估；精确期望需要知道每种 composition 在 phase 内的比例，
                # 但 row-order 推断本身就决定了比例，所以这里只能给一个上界
    detected_per_type = Counter(i.component for i in instances)

    declared_layers = sum(p["layers"] for p in structure_spec.get("phases", []))
    detected_layers = len({(i.phase, i.layer_idx) for i in instances})

    per_phase_match = {}
    for phase in structure_spec.get("phases", []):
        decl = phase["layers"]
        det = len({i.layer_idx for i in instances if i.phase == phase["name"]})
        per_phase_match[phase["name"]] = [decl, det]
        if det != decl:
            warnings.append(DraftWarning(
                code="layer_count_mismatch",
                message=f"phase {phase['name']}: 声明 {decl} 层，实测 {det} 层",
                extra={"phase": phase["name"], "declared": decl, "detected": det},
            ))

    expected_components = set(structure_spec.get("expected_components", []))
    detected_components = set(detected_per_type)
    missing = expected_components - detected_components
    for t in missing:
        warnings.append(DraftWarning(
            code="component_count_mismatch",
            message=f"component {t}: 期望 ≥ 1，实测 0",
            extra={"component": t, "detected": 0},
        ))

    validation = {
        "declared_layer_count": declared_layers,
        "detected_layer_count": detected_layers,
        "per_phase_layer_match": per_phase_match,
        "missing_samples": [],
        "ambiguous_matches": [],
    }
    return validation, warnings


def _low_shape_confidence_warnings(instances: list[Instance]) -> list[DraftWarning]:
    out = []
    for i in instances:
        bs = i.scores.get("body_shape_ms")
        if bs is not None and bs < SHAPE_CONFIDENCE_THRESHOLD:
            out.append(DraftWarning(
                code="low_shape_confidence",
                message=(f"layer {i.layer_idx} type={i.component}: "
                         f"body_shape_ms={bs:.2f} (<{SHAPE_CONFIDENCE_THRESHOLD})"),
                extra={"layer_idx": i.layer_idx, "component": i.component,
                       "body_shape_ms": bs},
            ))
    return out


def run_sample_mode(ops: list[dict], structure_spec: dict,
                    samples: dict[str, tuple[int, int]],
                    strict_composition: bool = False) -> SampleDraft:
    """主入口：从 ops + spec + samples 跑全流程.

    Args:
        ops: raw_ops.json 的 operators 列表
        structure_spec: 0a 解析结果 dict
        samples: {component_name: (lo, hi)}（闭区间 row 号）
        strict_composition: 若 True，遇到 composition_mismatch warning 抛 RuntimeError

    Returns:
        SampleDraft（结构见 dataclass）
    """
    expected = list(structure_spec.get("expected_components", []))
    missing = [c for c in expected if c not in samples]
    if missing:
        raise ValueError(
            f"missing samples for components: {missing}; "
            f"expected_components={expected}"
        )

    prefix = build_prefix_counts(ops)
    fingerprints: dict[str, Fingerprint] = {
        comp: extract_fingerprint(ops, comp, lo, hi, k=K)
        for comp, (lo, hi) in samples.items()
    }

    candidates_by_comp: dict[str, list[InstanceCandidate]] = {}
    for comp, fp in fingerprints.items():
        cands = match_component(ops, prefix, fp)
        cands = nms_candidates(cands)
        candidates_by_comp[comp] = cands

    instances, arb_warnings = arbitrate_overlaps(candidates_by_comp)

    layers_filled, asm_warnings = assemble_layers(instances, structure_spec)

    unmatched, unmatched_warnings = classify_unmatched(layers_filled, n=len(ops))

    validation, val_warnings = build_validation(structure_spec, layers_filled)
    shape_warnings = _low_shape_confidence_warnings(layers_filled)

    all_warnings = (arb_warnings + asm_warnings + unmatched_warnings
                    + val_warnings + shape_warnings)

    if strict_composition and any(w.code == "composition_mismatch" for w in all_warnings):
        mismatches = [w.message for w in all_warnings if w.code == "composition_mismatch"]
        for message in mismatches:
            logger.error("strict-composition: %s", message)
        raise RuntimeError(
            "strict-composition failed: "
            + "; ".join(mismatches)
        )

    samples_used = [
        {
            "component": comp,
            "op_range": list(samples[comp]),
            "length": fingerprints[comp].length,
            "head_size": K,
            "tail_size": K,
        }
        for comp in samples
    ]

    return SampleDraft(
        samples_used=samples_used,
        components=layers_filled,
        unmatched_regions=unmatched,
        warnings=all_warnings,
        validation=validation,
    )


def _component_expected_samples(structure_spec: dict, sample_ack: dict) -> list[str]:
    expected = list(structure_spec.get("expected_components", []))
    if not expected:
        expected = sorted(sample_ack.get("components", {}).keys())
    return expected


def _stream_samples_for_component(sample_ack: dict, component: str) -> tuple[str, list[dict]]:
    entry = (sample_ack.get("components") or {}).get(component)
    if not entry:
        raise ValueError(f"missing stream sample for component {component}")
    stream_samples = entry.get("stream_samples") or []
    if not stream_samples:
        raise ValueError(f"component {component} has no stream_samples")
    primary_entries = [s for s in stream_samples if s.get("role") == "primary"]
    if len(primary_entries) != 1:
        raise ValueError(
            f"component {component} must have exactly one primary stream sample, "
            f"got {len(primary_entries)}"
        )
    primary_stream_id = str(entry.get("primary_stream_id") or primary_entries[0]["stream_id"])
    return primary_stream_id, stream_samples


def _endpoint_mismatch_warnings(ops_by_idx: dict[int, dict], component: str,
                                sid: str, sample: dict,
                                indices: list[int]) -> list[DraftWarning]:
    """校验 sample 声明的 head_op / tail_op 与实际首尾 op 是否一致。"""
    if not indices:
        return []
    warnings: list[DraftWarning] = []
    actual_head = _name_of(ops_by_idx[indices[0]]) if indices[0] in ops_by_idx else None
    actual_tail = _name_of(ops_by_idx[indices[-1]]) if indices[-1] in ops_by_idx else None
    checks = (
        ("head_op", sample.get("head_op"), actual_head, indices[0]),
        ("tail_op", sample.get("tail_op"), actual_tail, indices[-1]),
    )
    for field_name, expected, actual, op_index in checks:
        if expected and actual and str(expected) != str(actual):
            warnings.append(DraftWarning(
                code="sample_ack_mismatch",
                message=(f"component {component}: stream {sid} {field_name} "
                         f"{expected!r} != actual {actual!r}"),
                extra={"component": component, "stream_id": sid,
                       "field": field_name, "expected": expected,
                       "actual": actual, "op_index": op_index},
            ))
    return warnings


def _low_shape_warnings(ops_by_idx: dict[int, dict], component: str, sid: str,
                        indices: list[int], streams: dict[str, list[int]]
                        ) -> list[DraftWarning]:
    """无 shape-confident match 时，回报最多 3 条 name/body-only 候选。"""
    warnings: list[DraftWarning] = []
    low_shape = _low_shape_stream_matches(ops_by_idx, streams.get(sid, []), indices)
    for match in low_shape[:3]:
        warnings.append(DraftWarning(
            code="stream_shape_mismatch",
            message=(f"component {component}: stream {sid} has name/body "
                     f"match but shape score "
                     f"{match['scores']['stream_shape']:.2f} below "
                     f"{SHAPE_CONFIDENCE_THRESHOLD:.2f}"),
            extra={"component": component, "stream_id": sid,
                   "op_indices": match["op_indices"],
                   "scores": match["scores"]},
        ))
    return warnings


def _scan_component_streams(ops_by_idx: dict[int, dict],
                            streams: dict[str, list[int]],
                            component: str,
                            primary_stream_id: str,
                            stream_samples: list[dict]
                            ) -> tuple[dict[str, list[StreamSegment]],
                                       dict[str, str], dict, list[DraftWarning]]:
    """扫每条声明 stream 的 segment，返回 (stream_cands, role_by_stream, echo, warnings)。"""
    warnings: list[DraftWarning] = []
    stream_cands: dict[str, list[StreamSegment]] = {}
    role_by_stream: dict[str, str] = {}
    sample_echo = {
        "component": component,
        "primary_stream_id": primary_stream_id,
        "stream_samples": [],
    }
    for sample in stream_samples:
        sid = str(sample["stream_id"])
        role = sample.get("role") or "unknown"
        role_by_stream[sid] = role
        indices = [int(i) for i in sample.get("op_indices") or []]
        sample_echo["stream_samples"].append({
            "stream_id": sid,
            "role": role,
            "op_indices": indices,
        })
        warnings.extend(_endpoint_mismatch_warnings(
            ops_by_idx, component, sid, sample, indices))
        if role == "unknown":
            continue
        cands = _scan_stream_segment(ops_by_idx, streams.get(sid, []), indices)
        if not cands:
            warnings.extend(_low_shape_warnings(
                ops_by_idx, component, sid, indices, streams))
        for c in cands:
            c.role = role
        stream_cands[sid] = cands
    return stream_cands, role_by_stream, sample_echo, warnings


def _auxiliary_sids(stream_cands: dict[str, list[StreamSegment]],
                    role_by_stream: dict[str, str],
                    primary_stream_id: str) -> list[str]:
    out: list[str] = []
    for sid in stream_cands:
        if sid == primary_stream_id:
            continue
        if role_by_stream.get(sid, "auxiliary") == "unknown":
            continue
        out.append(sid)
    return out


def _match_aux_streams(primary: list[StreamSegment],
                       stream_cands: dict[str, list[StreamSegment]],
                       aux_sids: list[str],
                       ops_by_idx: dict[int, dict]
                       ) -> tuple[dict[str, dict], dict[str, dict]]:
    """每条 aux 流先做一次全局最优匹配（occurrence ↔ segment）。"""
    aux_match: dict[str, dict[int, tuple[int, StreamSegment, dict]]] = {}
    aux_ambig: dict[str, dict[int, dict]] = {}
    for sid in aux_sids:
        assignment, ambiguous = _match_auxiliary_segments(
            primary, stream_cands[sid], ops_by_idx)
        aux_match[sid] = assignment
        aux_ambig[sid] = ambiguous
    return aux_match, aux_ambig


def _auxiliary_coverage(segments: list[StreamSegment],
                        role_by_stream: dict[str, str],
                        primary_stream_id: str) -> float:
    if not any(role == "auxiliary" for role in role_by_stream.values()):
        return 1.0
    declared_aux = sum(1 for sid, role in role_by_stream.items()
                       if sid != primary_stream_id and role == "auxiliary")
    return (len(segments) - 1) / max(1, declared_aux)


@dataclass
class _OccurrenceContext:
    """一个 component 的 aux-matching 上下文：跨该 component 所有 occurrence 共享。"""
    component: str
    primary_stream_id: str
    aux_sids: list[str]
    aux_match: dict[str, dict]
    aux_ambig: dict[str, dict]
    role_by_stream: dict[str, str]


def _build_occurrence_candidate(ctx: _OccurrenceContext, occ: int,
                                primary_seg: StreamSegment
                                ) -> tuple[dict, list[DraftWarning]]:
    """组装单个 occurrence 的 candidate dict + warnings。"""
    warnings: list[DraftWarning] = []
    segments = [primary_seg]
    for sid in ctx.aux_sids:
        entry = ctx.aux_match.get(sid, {}).get(occ)
        if entry is None:
            warnings.append(DraftWarning(
                code="auxiliary_stream_missing",
                message=(f"component {ctx.component} occurrence {occ}: "
                         f"auxiliary stream {sid} has no matching segment"),
                extra={"component": ctx.component, "occurrence_idx": occ,
                       "stream_id": sid},
            ))
            continue
        _pos, seg, metrics = entry
        seg.scores.update(metrics)
        segments.append(seg)
        if occ in ctx.aux_ambig.get(sid, {}):
            warnings.append(DraftWarning(
                code="auxiliary_stream_ambiguous",
                message=(f"component {ctx.component} occurrence {occ}: "
                         f"auxiliary stream {sid} has multiple time-near matches"),
                extra={"component": ctx.component, "occurrence_idx": occ,
                       "stream_id": sid, **ctx.aux_ambig[sid][occ]},
            ))
    op_indices = sorted({idx for seg in segments for idx in seg.op_indices})
    scores = {
        "primary_stream_sequence": primary_seg.scores.get("stream_sequence", 0.0),
        "stream_body": min((s.scores.get("stream_body", 0.0) for s in segments), default=0.0),
        "stream_shape": min((s.scores.get("stream_shape", 0.0) for s in segments), default=0.0),
        "auxiliary_stream_coverage": _auxiliary_coverage(
            segments, ctx.role_by_stream, ctx.primary_stream_id),
    }
    candidate = {
        "type": ctx.component,
        "occurrence_idx": occ,
        "primary_stream_id": ctx.primary_stream_id,
        "op_indices": op_indices,
        "stream_segments": segments,
        "scores": scores,
    }
    return candidate, warnings


def _component_candidates_for_stream_ack(
    ops_by_idx: dict[int, dict],
    streams: dict[str, list[int]],
    component: str,
    primary_stream_id: str,
    stream_samples: list[dict],
) -> tuple[list[dict], list[DraftWarning], dict]:
    stream_cands, role_by_stream, sample_echo, warnings = _scan_component_streams(
        ops_by_idx, streams, component, primary_stream_id, stream_samples)

    primary = stream_cands.get(primary_stream_id) or []
    if not primary:
        warnings.append(DraftWarning(
            code="primary_stream_missing",
            message=f"component {component}: primary stream {primary_stream_id} matched no segment",
            extra={"component": component, "stream_id": primary_stream_id},
        ))
        return [], warnings, sample_echo

    aux_sids = _auxiliary_sids(stream_cands, role_by_stream, primary_stream_id)
    aux_match, aux_ambig = _match_aux_streams(primary, stream_cands, aux_sids, ops_by_idx)

    ctx = _OccurrenceContext(
        component=component,
        primary_stream_id=primary_stream_id,
        aux_sids=aux_sids,
        aux_match=aux_match,
        aux_ambig=aux_ambig,
        role_by_stream=role_by_stream,
    )
    candidates = []
    for occ, primary_seg in enumerate(primary):
        candidate, occ_warnings = _build_occurrence_candidate(ctx, occ, primary_seg)
        warnings.extend(occ_warnings)
        candidates.append(candidate)
    return candidates, warnings, sample_echo


def _arbitrate_stream_candidates(candidates: list[dict]) -> tuple[list[dict], list[DraftWarning]]:
    kept: list[dict] = []
    occupied: dict[int, dict] = {}
    warnings: list[DraftWarning] = []
    ordered = sorted(
        candidates,
        key=lambda c: (
            -c["scores"].get("primary_stream_sequence", 0.0),
            -c["scores"].get("stream_body", 0.0),
            c["type"],
            c["occurrence_idx"],
        ),
    )
    for cand in ordered:
        conflicts = [idx for idx in cand["op_indices"] if idx in occupied]
        if not conflicts:
            kept.append(cand)
            for idx in cand["op_indices"]:
                occupied[idx] = cand
            continue
        warnings.append(DraftWarning(
            code="op_membership_conflict",
            message=(f"component {cand['type']} occurrence {cand['occurrence_idx']} "
                     f"shares {len(conflicts)} ops with another candidate"),
            extra={"component": cand["type"],
                   "occurrence_idx": cand["occurrence_idx"],
                   "op_indices": conflicts[:20]},
        ))
    return sorted(kept, key=lambda c: (c["type"], c["occurrence_idx"])), warnings


def _composition_local_indices(comp: dict, phase_name: str
                               ) -> tuple[list[int], Optional[DraftWarning]]:
    """从 composition 解析它覆盖的 local layer 索引；缺声明时返回 warning。"""
    if "layer_range" in comp:
        lo, hi = comp["layer_range"]
        return list(range(int(lo), int(hi) + 1)), None
    if "layer_indices" in comp:
        return [int(i) for i in comp["layer_indices"]], None
    warning = DraftWarning(
        code="composition_schedule_missing",
        message=(f"phase {phase_name}: layer_compositions item "
                 f"{comp.get('components')} has no layer_range/layer_indices"),
        extra={"phase": phase_name, "components": comp.get("components", [])},
    )
    return [], warning


def _assign_layer_composition(per_layer: dict[int, list[str]], comp: dict,
                              local_idx: int, layers: int, phase_name: str
                              ) -> Optional[DraftWarning]:
    """把一个 composition 绑到 local_idx；越界/重复时返回 warning（重复仍绑定）。"""
    if local_idx < 0 or local_idx >= layers:
        return DraftWarning(
            code="composition_mismatch",
            message=f"phase {phase_name}: layer index {local_idx} out of range",
            extra={"phase": phase_name, "layer_idx": local_idx},
        )
    warning = None
    if local_idx in per_layer:
        warning = DraftWarning(
            code="composition_mismatch",
            message=f"phase {phase_name}: layer {local_idx} matches multiple compositions",
            extra={"phase": phase_name, "layer_idx": local_idx},
        )
    per_layer[local_idx] = list(comp.get("components", []))
    return warning


def _phase_per_layer(phase: dict, phase_name: str, layers: int
                     ) -> tuple[dict[int, list[str]], list[DraftWarning]]:
    """把一个 phase 的 layer_compositions 展开为 {local_idx -> components}。"""
    warnings: list[DraftWarning] = []
    per_layer: dict[int, list[str]] = {}
    for comp in phase.get("layer_compositions", []):
        local_indices, missing = _composition_local_indices(comp, phase_name)
        if missing is not None:
            warnings.append(missing)
            continue
        for local_idx in local_indices:
            warning = _assign_layer_composition(
                per_layer, comp, local_idx, layers, phase_name)
            if warning is not None:
                warnings.append(warning)
    return per_layer, warnings


def _expand_schedule(structure_spec: dict) -> tuple[list[dict], list[DraftWarning]]:
    warnings: list[DraftWarning] = []
    schedule: list[dict] = []
    global_base = 0
    for phase in structure_spec.get("phases", []):
        phase_name = phase.get("name", "")
        layers = int(phase.get("layers", 0))
        per_layer, phase_warnings = _phase_per_layer(phase, phase_name, layers)
        warnings.extend(phase_warnings)
        for local_idx in range(layers):
            comps = per_layer.get(local_idx)
            if comps is None:
                warnings.append(DraftWarning(
                    code="composition_schedule_missing",
                    message=f"phase {phase_name}: layer {local_idx} has no composition schedule",
                    extra={"phase": phase_name, "layer_idx": local_idx},
                ))
                continue
            schedule.append({
                "phase": phase_name,
                "layer_idx": global_base + local_idx,
                "local_layer_idx": local_idx,
                "components": comps,
            })
        global_base += layers
    return schedule, warnings


def _assemble_stream_components(candidates: list[dict], structure_spec: dict
                                ) -> tuple[list[StreamComponent], list[DraftWarning]]:
    schedule, warnings = _expand_schedule(structure_spec)
    if any(w.code == "composition_schedule_missing" for w in warnings):
        return [], warnings

    by_type: dict[str, list[dict]] = {}
    for cand in candidates:
        by_type.setdefault(cand["type"], []).append(cand)
    for cands in by_type.values():
        cands.sort(key=lambda c: c["occurrence_idx"])

    cursors: Counter = Counter()
    components: list[StreamComponent] = []
    for layer in schedule:
        for ctype in layer["components"]:
            idx = cursors[ctype]
            available = by_type.get(ctype, [])
            if idx >= len(available):
                warnings.append(DraftWarning(
                    code="composition_mismatch",
                    message=(f"phase {layer['phase']} layer {layer['layer_idx']}: "
                             f"missing component {ctype} occurrence {idx}"),
                    extra={"phase": layer["phase"], "layer_idx": layer["layer_idx"],
                           "missing_type": ctype, "occurrence_idx": idx},
                ))
                continue
            cand = available[idx]
            cursors[ctype] += 1
            component_id = (
                f"{layer['phase']}.L{int(layer['layer_idx']):03d}."
                f"{ctype}.{cand['occurrence_idx']}"
            )
            components.append(StreamComponent(
                component_id=component_id,
                type=ctype,
                phase=layer["phase"],
                layer_idx=layer["layer_idx"],
                occurrence_idx=cand["occurrence_idx"],
                primary_stream_id=cand["primary_stream_id"],
                op_indices=list(cand["op_indices"]),
                stream_segments=list(cand["stream_segments"]),
                scores=dict(cand["scores"]),
            ))

    for ctype, available in by_type.items():
        if cursors[ctype] < len(available):
            warnings.append(DraftWarning(
                code="composition_mismatch",
                message=(f"component {ctype}: {len(available) - cursors[ctype]} "
                         f"matched occurrences were not consumed by schedule"),
                extra={"component": ctype, "unused_count": len(available) - cursors[ctype]},
            ))
    return components, warnings


def _unmatched_stream_segments(unmatched: list[int],
                               op_to_stream_pos: dict[int, dict]) -> list[dict]:
    by_stream: dict[str, list[int]] = {}
    for idx in unmatched:
        info = op_to_stream_pos.get(idx, {"stream_id": "unknown", "pos": idx})
        by_stream.setdefault(str(info["stream_id"]), []).append(idx)
    out = []
    for sid, indices in sorted(by_stream.items()):
        indices.sort(key=lambda i: op_to_stream_pos.get(i, {}).get("pos", i))
        current = []
        last_pos = None
        for idx in indices:
            pos = op_to_stream_pos.get(idx, {}).get("pos", idx)
            if current and last_pos is not None and pos != last_pos + 1:
                out.append({"stream_id": sid, "op_indices": current})
                current = []
            current.append(idx)
            last_pos = pos
        if current:
            out.append({"stream_id": sid, "op_indices": current})
    return out


def _is_displaced_record(rec: dict) -> bool:
    """单 occurrence 是否时间脱节：与主流不重叠且膨胀比超限。"""
    return rec["overlap"] <= 0.0 and rec["inflation"] > AUX_DISPLACEMENT_RATIO_LIMIT


def detect_displaced_aux_streams(candidates: list[dict],
                                 ops_by_idx: dict[int, dict]
                                 ) -> tuple[list[DraftWarning], set[tuple[str, str]]]:
    """并入膨胀体检——形态无关，只度量"撑大包络"这一危害本身。

    对每个 component 实例量"并入某条 aux 流让 component 时间包络相对主流自身膨胀多少"：
        primary_span = max_end(primary) − min_start(primary)
        total_span   = max_end(primary∪aux) − min_start(primary∪aux)
        inflation    = (total_span − primary_span) / primary_span
    按 (component_type, aux_stream) 聚合：若多数 occurrence 满足 overlap==0 且
    inflation > AUX_DISPLACEMENT_RATIO_LIMIT，则该流与主流时间脱节，并入会撑大 bubble。

    不识别任何具体流形态，只看包络是否被撑大；并发辅流（overlap>0）天然不进判定。
    只发 hard warning（detect_structure 会 block），不改 membership —— displaced 流的 op
    默认仍保留在 component（单列 displaced_op_indices 供 render 从 cluster/bubble/TOTAL 剔除
    并标注）；用户仅在确认该流不属于该 component 时才于 Phase 0b 改判为 unmatched。

    口径/边界：overlap 与 total 用 segment 时间包络（min_start/max_end），非 op 级 union；
    对内部时间稀疏的 aux seg 是近似，per-layer 紧凑 aux 下足够。退化主流（p_len<=0 或
    primary/aux 段为空）信息不足，跳过。判据全是相对/结构量（inflation 比 + 流级多数），
    无绝对时间阈值——避免量级过拟合（prefill/小模型/不同芯片量级各异）；偶发单点膨胀由
    流级多数（FRACTION）过滤，不靠绝对地板。
    """
    by_comp_aux: dict[tuple[str, str], list[dict]] = {}
    for cand in candidates:
        segs = cand.get("stream_segments") or []
        primary = next((s for s in segs if s.role == "primary"), None)
        if primary is None or not primary.op_indices:
            continue
        p_start, p_end = _segment_time_bounds(primary, ops_by_idx)
        p_len = p_end - p_start
        if p_len <= 0:        # 主流跨度为 0：膨胀比无意义，不纳入判定
            continue
        for s in segs:
            if s.role == "primary" or not s.op_indices:
                continue
            a_start, a_end = _segment_time_bounds(s, ops_by_idx)
            overlap = max(0.0, min(p_end, a_end) - max(p_start, a_start))
            displacement = (max(p_end, a_end) - min(p_start, a_start)) - p_len
            by_comp_aux.setdefault((cand["type"], s.stream_id), []).append({
                "overlap": overlap,
                "inflation": displacement / p_len,
                "occ": cand.get("occurrence_idx"),
                "op_indices": list(s.op_indices),
            })
    out: list[DraftWarning] = []
    displaced_set: set[tuple[str, str]] = set()
    for (ctype, sid), recs in sorted(by_comp_aux.items()):
        displaced = [r for r in recs if _is_displaced_record(r)]
        if recs and len(displaced) / len(recs) > AUX_DISPLACEMENT_FRACTION:
            displaced_set.add((ctype, sid))
            ratios = sorted(r["inflation"] for r in displaced)
            median_ratio = ratios[len(ratios) // 2]
            out.append(DraftWarning(
                code="auxiliary_stream_temporally_displaced",
                message=(
                    f"component {ctype}: auxiliary stream {sid} 与主流时间脱节"
                    f"（{len(displaced)}/{len(recs)} occurrence overlap=0 且膨胀比 > "
                    f"{AUX_DISPLACEMENT_RATIO_LIMIT}）。op 仍保留在本 component，render 会"
                    f"自动从 cluster/bubble/TOTAL 剔除并标注；仅当确认该流不属于本 component "
                    f"时才在 Phase 0b 改判为 unmatched。"),
                extra={
                    "component": ctype,
                    "stream_id": sid,
                    "displaced_occurrence_count": len(displaced),
                    "total_occurrence_count": len(recs),
                    "displaced_median_inflation_ratio": round(median_ratio, 2),
                    "example_op_indices": displaced[0]["op_indices"][:20],
                },
            ))
    return out, displaced_set


def _collect_stream_candidates(ack: dict, expected: list[str],
                               ops_by_idx: dict[int, dict],
                               streams: dict[str, list[int]]
                               ) -> tuple[list[dict], list[DraftWarning], list[dict]]:
    """遍历每个 expected component，收集 candidates / warnings / sample echo。"""
    all_candidates: list[dict] = []
    all_warnings: list[DraftWarning] = []
    samples_used: list[dict] = []
    for component in expected:
        primary, stream_samples = _stream_samples_for_component(ack, component)
        cands, warnings, sample_echo = _component_candidates_for_stream_ack(
            ops_by_idx, streams, component, primary, stream_samples)
        all_candidates.extend(cands)
        all_warnings.extend(warnings)
        samples_used.append(sample_echo)
    return all_candidates, all_warnings, samples_used


def _build_op_to_component(components: list[StreamComponent]) -> dict[str, str]:
    op_to_component: dict[str, str] = {}
    for comp in components:
        for idx in comp.op_indices:
            op_to_component[str(idx)] = comp.component_id
    return op_to_component


def _unmatched_op_indices(ops: list[dict], op_to_component: dict[str, str]) -> list[int]:
    all_indices = sorted(_op_index(op, pos) for pos, op in enumerate(ops))
    return [idx for idx in all_indices if str(idx) not in op_to_component]


def _build_stream_validation(structure_spec: dict,
                             components: list[StreamComponent],
                             all_warnings: list[DraftWarning]) -> dict:
    declared = sum(int(p.get("layers", 0)) for p in structure_spec.get("phases", []))
    detected = len({(c.phase, c.layer_idx) for c in components})
    per_phase = {}
    for phase in structure_spec.get("phases", []):
        name = phase.get("name", "")
        per_phase[name] = [
            int(phase.get("layers", 0)),
            len({c.layer_idx for c in components if c.phase == name}),
        ]
    return {
        "declared_layer_count": declared,
        "detected_layer_count": detected,
        "per_phase_layer_match": per_phase,
        "missing_samples": [],
        "ambiguous_matches": [
            w.extra for w in all_warnings if w.code == "op_membership_conflict"
        ],
    }


def run_stream_sample_mode(ops: list[dict], structure_spec: dict,
                           sample_ack: dict
                           ) -> StreamSampleDraft:
    """Stream-aware Step 2 implementation.

    Consumes stream_sample_ack.v1. Row/time ranges must be converted during
    Phase 0b before this function runs; they are not accepted as structure
    facts.
    """
    ack = _normalize_stream_ack(sample_ack)
    streams, op_to_stream_pos = build_stream_index(ops)
    ops_by_idx = _ops_by_index(ops)
    expected = _component_expected_samples(structure_spec, ack)

    all_candidates, all_warnings, samples_used = _collect_stream_candidates(
        ack, expected, ops_by_idx, streams)

    kept_candidates, arb_warnings = _arbitrate_stream_candidates(all_candidates)
    all_warnings.extend(arb_warnings)
    disp_warnings, displaced_streams = detect_displaced_aux_streams(kept_candidates, ops_by_idx)
    all_warnings.extend(disp_warnings)
    components, asm_warnings = _assemble_stream_components(kept_candidates, structure_spec)
    all_warnings.extend(asm_warnings)

    op_to_component = _build_op_to_component(components)
    unmatched = _unmatched_op_indices(ops, op_to_component)
    unmatched_segments = _unmatched_stream_segments(unmatched, op_to_stream_pos)
    validation = _build_stream_validation(structure_spec, components, all_warnings)

    return StreamSampleDraft(
        structure_spec=structure_spec,
        samples_used=samples_used,
        components=components,
        op_to_component=op_to_component,
        unmatched_op_indices=unmatched,
        unmatched_stream_segments=unmatched_segments,
        warnings=all_warnings,
        validation=validation,
        displaced_streams=sorted(displaced_streams),
    )


def _displaced_op_indices(component: StreamComponent,
                          displaced: set[tuple[str, str]]) -> list[int]:
    """收集该 component 中属于 displaced (type, stream) 的 op 下标（升序去重）。"""
    disp_ops: set[int] = set()
    for s in component.stream_segments:
        if (component.type, s.stream_id) in displaced:
            disp_ops.update(s.op_indices)
    return sorted(disp_ops)


def _stream_component_to_dict(c: StreamComponent,
                              displaced: set[tuple[str, str]]) -> dict:
    d = {
        "component_id": c.component_id,
        "type": c.type,
        "phase": c.phase,
        "layer_idx": c.layer_idx,
        "occurrence_idx": c.occurrence_idx,
        "primary_stream_id": c.primary_stream_id,
        "op_indices": c.op_indices,
        "stream_segments": [
            {"stream_id": s.stream_id, "role": s.role, "op_indices": s.op_indices}
            for s in c.stream_segments
        ],
        "scores": c.scores,
    }
    # 时间脱节辅流的 op 仍留在 op_indices 全集（matched 不丢、partition 不破），
    # 但单列出来供 render 从 TOTAL/bubble 排除并标注。
    disp_ops = _displaced_op_indices(c, displaced)
    if disp_ops:
        d["displaced_op_indices"] = disp_ops
    return d


def stream_draft_to_dict(draft: StreamSampleDraft,
                         structure_spec: Optional[dict] = None) -> dict:
    displaced = {(t, s) for t, s in draft.displaced_streams}
    return {
        "mode": "stream_sample_driven",
        "schema_version": "structure_draft.stream.v1",
        "structure_spec": structure_spec if structure_spec is not None else draft.structure_spec,
        "samples_used": draft.samples_used,
        "components": [_stream_component_to_dict(c, displaced) for c in draft.components],
        "op_to_component": draft.op_to_component,
        "unmatched_op_indices": draft.unmatched_op_indices,
        "unmatched_stream_segments": draft.unmatched_stream_segments,
        "warnings": [
            {"code": w.code, "message": w.message, **w.extra}
            for w in draft.warnings
        ],
        "validation": draft.validation,
    }
