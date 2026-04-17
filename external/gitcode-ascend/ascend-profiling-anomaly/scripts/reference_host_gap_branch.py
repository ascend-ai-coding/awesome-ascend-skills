from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict, Any

import math
import pandas as pd


@dataclass
class Interval:
    start_us: float
    end_us: float

    @property
    def dur_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    items = sorted((i for i in intervals if i.end_us > i.start_us), key=lambda x: (x.start_us, x.end_us))
    if not items:
        return []
    merged: List[Interval] = [Interval(items[0].start_us, items[0].end_us)]
    for cur in items[1:]:
        last = merged[-1]
        if cur.start_us <= last.end_us:
            last.end_us = max(last.end_us, cur.end_us)
        else:
            merged.append(Interval(cur.start_us, cur.end_us))
    return merged


def interval_union_us(intervals: Sequence[Interval]) -> float:
    return sum(i.dur_us for i in merge_intervals(intervals))


def interval_intersection_us(a: Interval, b: Interval) -> float:
    left = max(a.start_us, b.start_us)
    right = min(a.end_us, b.end_us)
    return max(0.0, right - left)


def union_overlap_ratio(target: Interval, others: Sequence[Interval]) -> Optional[float]:
    if target.dur_us <= 0:
        return None
    clipped: List[Interval] = []
    for x in others:
        s = max(target.start_us, x.start_us)
        e = min(target.end_us, x.end_us)
        if e > s:
            clipped.append(Interval(s, e))
    if not clipped:
        return 0.0
    return interval_union_us(clipped) / target.dur_us


def build_device_intervals(device_df: pd.DataFrame, step_start_us: float, step_end_us: float) -> List[Interval]:
    sub = device_df[(device_df['start_us'] < step_end_us) & ((device_df['start_us'] + device_df['duration_us']) > step_start_us)]
    out: List[Interval] = []
    for row in sub.itertuples(index=False):
        s = max(step_start_us, float(row.start_us))
        e = min(step_end_us, float(row.start_us) + float(row.duration_us))
        if e > s:
            out.append(Interval(s, e))
    return out


def compute_step_bubble_metrics(step_start_us: float, step_end_us: float, device_intervals: Sequence[Interval]) -> Dict[str, Any]:
    service_us = max(0.0, step_end_us - step_start_us)
    merged = merge_intervals(device_intervals)
    if not merged:
        return {
            'service_ms': service_us / 1000.0,
            'device_busy_union_ms': 0.0,
            'underfeed_ms': service_us / 1000.0,
            'underfeed_ratio': 1.0 if service_us > 0 else 0.0,
            'prelaunch_gap_ms': service_us / 1000.0,
            'tail_gap_ms': 0.0,
            'internal_bubble_total_ms': 0.0,
            'largest_internal_bubble_ms': 0.0,
            'bubble_count': 0,
            'bubble_intervals': [],
        }

    busy_union_us = sum(seg.dur_us for seg in merged)
    prelaunch_us = max(0.0, merged[0].start_us - step_start_us)
    tail_us = max(0.0, step_end_us - merged[-1].end_us)

    bubbles: List[Interval] = []
    for left, right in zip(merged[:-1], merged[1:]):
        if right.start_us > left.end_us:
            bubbles.append(Interval(left.end_us, right.start_us))

    bubble_total_us = sum(b.dur_us for b in bubbles)
    largest_bubble_us = max((b.dur_us for b in bubbles), default=0.0)
    underfeed_us = max(0.0, service_us - busy_union_us)
    underfeed_ratio = underfeed_us / service_us if service_us > 0 else 0.0

    return {
        'service_ms': service_us / 1000.0,
        'device_busy_union_ms': busy_union_us / 1000.0,
        'underfeed_ms': underfeed_us / 1000.0,
        'underfeed_ratio': underfeed_ratio,
        'prelaunch_gap_ms': prelaunch_us / 1000.0,
        'tail_gap_ms': tail_us / 1000.0,
        'internal_bubble_total_ms': bubble_total_us / 1000.0,
        'largest_internal_bubble_ms': largest_bubble_us / 1000.0,
        'bubble_count': len(bubbles),
        'bubble_intervals': bubbles,
    }


def score_wait_anchor(duration_us: float, wait_us: float, total_cost_rank: int) -> Dict[str, Any]:
    total = max(0.0, duration_us + wait_us)
    wait_ratio = (wait_us / total) if total > 0 else 0.0
    is_false_hotspot_risk = bool(wait_ratio > 0.95 and duration_us < 10.0 and total_cost_rank <= 10)
    return {
        'wait_ratio': wait_ratio,
        'is_false_hotspot_risk': is_false_hotspot_risk,
    }


def soft_attribution_for_bubble(
    bubble: Interval,
    host_intervals: Sequence[Interval],
    sync_intervals: Sequence[Interval],
    comm_intervals: Sequence[Interval],
    host_parallelism_hint: Optional[float] = None,
) -> Dict[str, Any]:
    host_cov = union_overlap_ratio(bubble, host_intervals)
    sync_cov = union_overlap_ratio(bubble, sync_intervals)
    comm_cov = union_overlap_ratio(bubble, comm_intervals)
    labels: List[str] = []

    if (sync_cov or 0.0) >= 0.2:
        labels.append('possible_sync_or_h2d')
    if (comm_cov or 0.0) >= 0.2:
        labels.append('possible_comm_wait')
    if (host_cov or 0.0) < 0.05:
        labels.append('possible_untraced_host_blocking')
    if not labels and (host_cov or 0.0) >= 0.1:
        labels.append('possible_host_launch_lag')
    if not labels and (host_parallelism_hint is not None) and host_parallelism_hint < 1.2:
        labels.append('possible_python_serialization_or_lock')
    if not labels:
        labels.append('insufficient_evidence')

    return {
        'host_visible_coverage_ratio': host_cov,
        'sync_marker_overlap_ratio': sync_cov,
        'comm_marker_overlap_ratio': comm_cov,
        'soft_root_cause_labels': labels,
    }


def classify_hidden_issue(metrics: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    service_ms = float(metrics['service_ms'])
    if metrics['underfeed_ratio'] >= 0.30:
        tags.append('DEVICE_IDLE_GAP_HEAVY')
    if metrics['prelaunch_gap_ms'] >= max(1.0, 0.10 * service_ms):
        tags.append('PRELAUNCH_GAP_HEAVY')
    if metrics['tail_gap_ms'] >= max(1.0, 0.10 * service_ms):
        tags.append('TAIL_GAP_HEAVY')
    if metrics['largest_internal_bubble_ms'] >= max(1.0, 0.10 * service_ms):
        tags.append('INTERNAL_BUBBLE_HEAVY')
    return tags


def aggregate_group_metrics(step_rows: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [
        'service_ms', 'device_busy_union_ms', 'underfeed_ratio', 'prelaunch_gap_ms',
        'tail_gap_ms', 'internal_bubble_total_ms', 'largest_internal_bubble_ms'
    ]
    out: Dict[str, Any] = {'group_size': int(len(step_rows))}
    for c in numeric_cols:
        out[f'{c}_avg'] = float(step_rows[c].mean()) if len(step_rows) else 0.0
        out[f'{c}_median'] = float(step_rows[c].median()) if len(step_rows) else 0.0
    out['largest_internal_bubble_ms_p95'] = float(step_rows['largest_internal_bubble_ms'].quantile(0.95)) if len(step_rows) else 0.0

    recurrence = 0.0
    if len(step_rows):
        recurrence = float((step_rows['bubble_count'] > 0).mean())
    out['recurring_bubble_pattern'] = recurrence >= 0.60

    dominant = {
        'prelaunch': out['prelaunch_gap_ms_avg'],
        'tail': out['tail_gap_ms_avg'],
        'internal_bubble': out['internal_bubble_total_ms_avg'],
    }
    best_name, best_val = max(dominant.items(), key=lambda x: x[1])
    out['dominant_idle_pattern'] = best_name if best_val > 0 else 'none'
    return out
