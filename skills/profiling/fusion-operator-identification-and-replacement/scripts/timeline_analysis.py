#!/usr/bin/env python3
"""Timeline analysis for kernel_details.csv: per-stream busy/gaps, cross-stream
overlap, adjacent-op pair sequences, and (optional) step-count inference from a
per-step signature op.

Usage:
    python timeline_analysis.py <kernel_details.csv> [--step-op TYPE] [--step-gap-ms N]

Key outputs (the evidence a fusion-analysis report needs):
  - busy ratio over the capture window (sum of durations / window) -> dispatch-bound?
  - per-stream gap distribution -> launch overhead evidence
  - adjacent pairs on each stream -> unfused sequence candidates (with counts)
  - step count via a per-step signature op (e.g. the LM-head matmul)

Handles trailing tabs and epoch-microsecond Start Time.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def clean(s):
    return s.strip().strip('\t').strip() if isinstance(s, str) else s


def merged_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, d in iv:
        e = s + d
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kernel_details')
    ap.add_argument('--step-op', default=None,
                    help='op Type that occurs once per step (e.g. LM-head matmul type)')
    ap.add_argument('--step-gap-ms', type=float, default=500.0,
                    help='gap (ms) above which two --step-op calls are separate steps')
    args = ap.parse_args()

    rows = []
    with open(args.kernel_details, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                rows.append((clean(row['Stream ID']), float(clean(row['Start Time(us)'])),
                             float(clean(row['Duration(us)'])), clean(row['Type'])))
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r[1])

    t0 = rows[0][1] if rows else 0.0
    t1 = max(s + d for _, s, d, _ in rows) if rows else 0.0
    busy = sum(d for _, _, d, _ in rows)
    print(f'window_us={t1 - t0:.1f} sum_duration_us={busy:.1f} busy_ratio={busy / (t1 - t0) * 100:.2f}%')

    by_stream = defaultdict(list)
    for sid, s, d, t in rows:
        by_stream[sid].append((s, d, t))

    result = {'window_us': round(t1 - t0, 1), 'sum_duration_us': round(busy, 1),
              'busy_ratio_pct': round(busy / (t1 - t0) * 100, 2), 'streams': {}}

    for sid, lst in sorted(by_stream.items(), key=lambda kv: -len(kv[1])):
        lst.sort()
        gaps = [lst[i + 1][0] - (lst[i][0] + lst[i][1]) for i in range(len(lst) - 1)]
        gaps = [g for g in gaps if g > 0]
        sd = {'n_kernels': len(lst), 'sum_duration_us': round(sum(d for _, d, _ in lst), 1)}
        if gaps:
            sd.update({'gap_count': len(gaps), 'gap_sum_us': round(sum(gaps), 1),
                       'gap_median_us': round(statistics.median(gaps), 1),
                       'gap_mean_us': round(statistics.mean(gaps), 1),
                       'gap_max_us': round(max(gaps), 1)})
        result['streams'][sid] = sd
        print(f"stream {sid}: {sd}")

        # adjacent pairs within 200us on this stream
        pairs = defaultdict(int)
        for i in range(len(lst) - 1):
            if lst[i + 1][0] - lst[i][0] < 200:
                pairs[(lst[i][2], lst[i + 1][2])] += 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:20]
        result['streams'][sid]['top_adjacent_pairs'] = [f'{a} -> {b} ({c}x)' for (a, b), c in top]
        for (a, b), c in top[:10]:
            print(f'    pair: {a} -> {b}  x{c}')

    # cross-stream overlap
    sids = list(result['streams'])
    if len(sids) >= 2:
        m = [merged_intervals([(s, d) for s, d, _ in by_stream[sid]]) for sid in sids[:2]]
        i = j = 0
        ov = 0.0
        while i < len(m[0]) and j < len(m[1]):
            s = max(m[0][i][0], m[1][j][0]); e = min(m[0][i][1], m[1][j][1])
            if e > s:
                ov += e - s
            if m[0][i][1] < m[1][j][1]:
                i += 1
            else:
                j += 1
        result['cross_stream_overlap_us'] = round(ov, 1)
        print(f"cross-stream overlap ({sids[0]} vs {sids[1]}): {ov:.1f}us")

    # step inference
    if args.step_op:
        times = sorted(s for sid, s, d, t in rows if t == args.step_op)
        if times:
            steps = [[times[0]]]
            for t in times[1:]:
                if t - steps[-1][-1] > args.step_gap_ms * 1000:
                    steps.append([t])
                else:
                    steps[-1].append(t)
            per_step = len(rows) / len(steps)
            per_step_busy = busy / len(steps)
            result['steps'] = {'signature_op': args.step_op, 'step_count': len(steps),
                               'signature_calls': len(times),
                               'kernels_per_step': round(per_step, 1),
                               'busy_us_per_step': round(per_step_busy, 1)}
            print(f"steps via {args.step_op}: count={len(steps)}, kernels/step={per_step:.0f}, "
                  f"busy/step={per_step_busy:.1f}us")

    out = Path('timeline_analysis.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'JSON written to {out}')


if __name__ == '__main__':
    main()
