#!/usr/bin/env python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Split kernel_details.csv + trace_view.json per component / region.

Driven by structure_draft.json (stream_sample_driven mode). Partitions the op
membership into a complete set of folders — every op appears in exactly one
folder, no duplicates, no gaps:

  - one folder per component instance, named L<layer:03d>_<phase>_<type>
  - one folder per unmatched region between components, named after the
    surrounding components: gap_L<idx>_<phase>_<type>_to_L<idx>_<phase>_<type>,
    gap_before_L<...> / gap_after_L<...>, pre, post, suspected_...

Each folder contains:

  kernels.csv      rows of the source CSV restricted to this folder's ops
                   (header preserved + leading `org_index` column)
  trace_view.json  trace evidence for this folder. When --trace is supplied it
                   is a time-envelope context window around the folder's ops,
                   so parallel ops from other folders may be visible. Without
                   --trace it is synthesized exactly from this folder's
                   op_indices (one 'X' event per kernel with ts/dur/name/stream).

Top-level `manifest.json` maps folder → {kind, op_range, op_indices?,
ops_count, files} for quick navigation. `op_range` is only the display
envelope; `op_indices` is present when the target membership is non-contiguous.
`files.trace_scope` distinguishes exact synthesized traces from source-trace
time-envelope context. No per-folder summary.json is written.

The script asserts coverage = exactly [0, N-1] with no overlap. If the
draft's components + unmatched_op_indices don't partition the op space, the
run aborts (indicates a draft / raw_ops mismatch).
"""

import argparse
import csv
import dataclasses
import json
import logging
import os
import re
import sys
import typing

logger = logging.getLogger(__name__)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_op_index(ops):
    return [
        {
            "org_index": op["org_index"],
            "start_time_us": op["start_time_us"],
            "end_time_us": op["start_time_us"] + op.get("duration_us", 0.0),
            "normalized_name": op.get("normalized_name", op.get("name", "")),
            "accelerator_core": op.get("accelerator_core", ""),
        }
        for op in ops
    ]


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(s):
    s = _SLUG_RE.sub("_", str(s).strip())
    return s.strip("_") or "unnamed"


def component_folder_name(c):
    return f"L{int(c['layer_idx']):03d}_{slugify(c['phase'])}_{slugify(c['type'])}"


def time_range(ops_idx, op_indices):
    sub = [ops_idx[i] for i in op_indices]
    if not sub:
        return None, None
    return (min(o["start_time_us"] for o in sub),
            max(o["end_time_us"] for o in sub))


def _op_indices_from_component(c):
    return sorted(int(i) for i in c.get("op_indices") or [])


def validate_draft_schema(draft):
    if (draft.get("mode") != "stream_sample_driven"
            or draft.get("schema_version") != "structure_draft.stream.v1"):
        raise ValueError(
            f"draft mode={draft.get('mode')!r}, "
            f"schema_version={draft.get('schema_version')!r} not supported. "
            f"Only stream_sample_driven / structure_draft.stream.v1 drafts are accepted."
        )
    for i, component in enumerate(draft.get("components", [])):
        if "op_indices" not in component:
            raise ValueError(f"draft component[{i}] missing op_indices")


def _contiguous_runs(indices):
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append(list(range(start, prev + 1)))
        start = prev = idx
    runs.append(list(range(start, prev + 1)))
    return runs


class Entry(typing.NamedTuple):
    """A position-sorted component/region entry over the op space."""
    min_idx: int
    max_idx: int
    kind: str
    payload: dict
    op_indices: list


def _build_entries(draft):
    """Build a unified, position-sorted list of component/region entries."""
    components = sorted(
        draft.get("components", []),
        key=lambda c: min(_op_indices_from_component(c) or [10 ** 18]),
    )
    entries = []
    for c in components:
        indices = _op_indices_from_component(c)
        if indices:
            entries.append(Entry(min(indices), max(indices), "component", c, indices))
    for run_indices in _contiguous_runs(draft.get("unmatched_op_indices") or []):
        indices = sorted(int(i) for i in run_indices)
        if indices:
            region = {
                "op_indices": run_indices,
                "op_range": [run_indices[0], run_indices[-1]],
                "classification": "unmatched",
            }
            entries.append(
                Entry(min(indices), max(indices), "unmatched", region, indices))
    entries.sort(key=lambda e: e.min_idx)
    return entries


def _assert_membership(entry, owner, n_ops):
    """Record op ownership, raising on out-of-range or overlapping ops."""
    kind = entry.kind
    for idx in entry.op_indices:
        if idx < 0 or idx >= n_ops:
            raise ValueError(
                f"draft references invalid op {idx} in {kind} "
                f"[{entry.min_idx}, {entry.max_idx}]")
        if idx in owner:
            raise ValueError(
                f"draft overlap: op {idx} appears in both {owner[idx]} and {kind}."
            )
        owner[idx] = kind


def _neighbor_component(entries, start, step):
    """Return the nearest component payload scanning entries from start by step."""
    j = start
    while 0 <= j < len(entries):
        if entries[j].kind == "component":
            return entries[j].payload
        j += step
    return None


def _component_label(c):
    return (f"L{int(c['layer_idx']):03d}_"
            f"{slugify(c['phase'])}_{slugify(c['type'])}")


def _region_folder_name(entries, i, kind, lo, hi):
    """Derive a layer/component-aware folder name for an inter-layer region."""
    prev_comp = _neighbor_component(entries, i - 1, -1)
    next_comp = _neighbor_component(entries, i + 1, 1)
    prefix = "suspected" if kind == "suspected_undeclared_component" else "gap"
    if prev_comp is not None and next_comp is not None:
        return f"{prefix}_{_component_label(prev_comp)}_to_{_component_label(next_comp)}"
    if prev_comp is not None:
        return f"{prefix}_after_{_component_label(prev_comp)}"
    if next_comp is not None:
        return f"{prefix}_before_{_component_label(next_comp)}"
    return f"{prefix}_{lo}_{hi}"


def _folder_name_for_entry(entries, i):
    """Compute the (pre-disambiguation) folder name for a single entry."""
    entry = entries[i]
    kind, lo, hi = entry.kind, entry.min_idx, entry.max_idx
    if kind == "component":
        return component_folder_name(entry.payload)
    if kind == "pre_arch":
        return "pre"
    if kind == "post_arch":
        return "post"
    if kind in ("inter_layer_region", "suspected_undeclared_component"):
        return _region_folder_name(entries, i, kind, lo, hi)
    return f"{slugify(kind)}_{lo}_{hi}"


def _assert_full_coverage(owner, n_ops):
    """Raise if any op in [0, n_ops-1] is unclaimed by a target."""
    missing = [i for i in range(n_ops) if i not in owner]
    if not missing:
        return
    first = missing[0]
    last = first
    for idx in missing[1:]:
        if idx == last + 1:
            last = idx
        else:
            break
    raise ValueError(
        f"draft coverage gap: ops [{first}, {last}] unclaimed, "
        f"but raw_ops has {n_ops} ops. Re-run Step 2 against the current "
        f"raw_ops, then retry split."
    )


def _disambiguate_folder_names(targets):
    """Append op ranges / dup suffixes so every folder name is unique."""
    name_counts = {}
    for folder, _, _, _ in targets:
        name_counts[folder] = name_counts.get(folder, 0) + 1
    final = []
    seen = {}
    for folder, kind, op_indices, payload in targets:
        lo, hi = min(op_indices), max(op_indices)
        if name_counts[folder] > 1:
            folder = f"{folder}_{lo}_{hi}"
        if folder in seen:
            folder = f"{folder}_dup{seen[folder]}"
        seen[folder] = seen.get(folder, 0) + 1
        final.append((folder, kind, op_indices, payload))
    return final


def enumerate_targets(draft, n_ops):
    """Yield (folder_name, kind, op_indices, payload) covering [0, n_ops-1].

    Order is by op position. Asserts no overlap and full coverage.
    """
    entries = _build_entries(draft)

    owner = {}
    targets = []
    for i, entry in enumerate(entries):
        _assert_membership(entry, owner, n_ops)
        folder = _folder_name_for_entry(entries, i)
        targets.append((folder, entry.kind, entry.op_indices, entry.payload))

    _assert_full_coverage(owner, n_ops)
    return _disambiguate_folder_names(targets)


AI_CPU_CORES = {"AI_CPU", "AICPU"}


def _is_ai_cpu_row(row, accel_idx):
    return (accel_idx is not None and accel_idx < len(row)
            and row[accel_idx].strip().upper() in AI_CPU_CORES)


def slice_csv(csv_path, org_to_step_index, out_path):
    """Slice CSV rows whose 0-based row index is in `org_to_step_index`.

    Prepends an `org_index` column. AI_CPU rows are dropped to stay aligned
    with raw_ops.json semantics (analyze_kernels filters AI_CPU by default).
    If the CSV has no 'Accelerator Core' column the filter is skipped.
    """
    with open(csv_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header is None:
            return 0
        try:
            accel_idx = header.index("Accelerator Core")
        except ValueError:
            accel_idx = None
        writer.writerow(["org_index", *header])
        written = 0
        for i, row in enumerate(reader):
            if i not in org_to_step_index:
                continue
            if _is_ai_cpu_row(row, accel_idx):
                continue
            writer.writerow([i, *row])
            written += 1
        return written


def synthesize_trace_events(ops_idx_full, op_indices, ops_data):
    events = []
    operators = ops_data["operators"]
    for pos in op_indices:
        if pos >= len(operators):
            break
        op = operators[pos]
        tid = op.get("stream_id", "0")
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            tid_int = 0
        events.append({
            "ph": "X",
            "name": op.get("normalized_name") or op.get("name") or op.get("original_name", "op"),
            "pid": 0,
            "tid": tid_int,
            "ts": op["start_time_us"],
            "dur": op.get("duration_us", 0.0),
            "cat": op.get("task_type", "kernel"),
            "args": {
                "op_index": op.get("index"),
                "org_index": op.get("org_index"),
                "stream_id": op.get("stream_id"),
            },
        })
    return events


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def slice_trace_events(events, t_lo, t_hi):
    """Keep metadata + events strictly contained in [t_lo, t_hi].

    Strict containment (not interval overlap) avoids pulling in neighbor
    blocks' kernels that touch this slice's window at the boundary. Source
    profiles encode `ts` as a string; normalize to float so downstream
    viewers parse it correctly.
    """
    if t_lo is None or t_hi is None:
        return []
    kept = []
    for ev in events:
        ph = ev.get("ph")
        if ph in ("M", "I", "i"):
            ts_f = _to_float(ev.get("ts"))
            if ts_f is not None:
                ev["ts"] = ts_f
            kept.append(ev)
            continue
        ts_f = _to_float(ev.get("ts"))
        if ts_f is None:
            continue
        dur_f = _to_float(ev.get("dur")) or 0.0
        end_f = ts_f + dur_f
        if ts_f >= t_lo and end_f <= t_hi:
            ev["ts"] = ts_f
            if "dur" in ev:
                ev["dur"] = dur_f
            kept.append(ev)
    return kept


def load_trace(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, "list", None
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"], "dict", data
    raise ValueError(f"Unrecognized trace_view format in {path}")


def write_trace_fragment(events, kind, container, out_path):
    if kind == "list":
        out = events
    else:
        out = {k: v for k, v in container.items()}
        out["traceEvents"] = events
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--draft", required=True,
                   help="structure_draft.json from Step 2 sample mode")
    p.add_argument("-r", "--raw_ops", required=True,
                   help="raw_ops.json from Step 1")
    p.add_argument("-f", "--csv", default=None,
                   help="source kernel_details.csv (omit to skip CSV slicing)")
    p.add_argument("-t", "--trace", default=None,
                   help="source trace_view.json. If omitted, each folder's "
                        "trace_view.json is synthesized from raw_ops.")
    p.add_argument("-o", "--output", required=True,
                   help="output directory (will be created)")
    p.add_argument("--dry-run", action="store_true",
                   help="list targets without writing csv/trace files")
    return p.parse_args(argv)


def _check_draft_op_bounds(draft, n_ops):
    """Raise if the draft references an op beyond the available op count."""
    max_end = max(
        (max(_op_indices_from_component(c) or [-1])
         for c in draft.get("components", [])),
        default=-1,
    )
    if max_end >= n_ops:
        raise ValueError(
            f"draft references op {max_end} but raw_ops only has {n_ops} ops. "
            f"raw_ops.json was likely regenerated without re-running "
            f"detect_structure. Re-run Step 2 against current raw_ops."
        )


def _build_manifest_source(args):
    return {
        "draft": os.path.abspath(args.draft),
        "raw_ops": os.path.abspath(args.raw_ops),
        "csv": os.path.abspath(args.csv) if args.csv else None,
        "trace": os.path.abspath(args.trace) if args.trace else None,
    }


@dataclasses.dataclass
class SliceContext:
    """Per-run context shared by every target's csv/trace write."""
    args: argparse.Namespace
    ops_idx: list
    ops_data: dict
    trace_ctx: tuple


def _write_csv_slice(ctx, folder_path, op_indices, files_written):
    args = ctx.args
    if not (args.csv and not args.dry_run):
        return
    csv_out = os.path.join(folder_path, "kernels.csv")
    org_to_step = {ctx.ops_idx[i]["org_index"]: i for i in op_indices}
    rows = slice_csv(args.csv, org_to_step, csv_out)
    files_written["csv"] = "kernels.csv"
    files_written["csv_scope"] = "op_membership_exact"
    files_written["csv_rows"] = rows


def _write_trace_slice(ctx, folder_path, op_indices, files_written):
    args = ctx.args
    if args.dry_run:
        return
    trace_out = os.path.join(folder_path, "trace_view.json")
    if args.trace:
        trace_events, trace_kind, trace_container = ctx.trace_ctx
        t_lo, t_hi = time_range(ctx.ops_idx, op_indices)
        sliced = slice_trace_events(trace_events, t_lo, t_hi)
        write_trace_fragment(sliced, trace_kind, trace_container, trace_out)
        files_written["trace"] = "trace_view.json"
        files_written["trace_source"] = "sliced"
        files_written["trace_scope"] = "time_envelope_context"
        files_written["trace_time_range_us"] = [t_lo, t_hi]
        files_written["trace_events"] = len(sliced)
        return
    synth = synthesize_trace_events(ctx.ops_idx, op_indices, ctx.ops_data)
    with open(trace_out, "w") as f:
        json.dump(synth, f, ensure_ascii=False)
    files_written["trace"] = "trace_view.json"
    files_written["trace_source"] = "synthesized"
    files_written["trace_scope"] = "op_membership_exact"
    files_written["trace_events"] = len(synth)


def _manifest_entry(folder, kind, op_indices, files_written):
    start_pos, end_pos = min(op_indices), max(op_indices)
    contiguous = len(op_indices) == end_pos - start_pos + 1
    return {
        "folder": folder,
        "kind": kind,
        "op_range": [start_pos, end_pos],
        "op_indices": None if contiguous else op_indices,
        "ops_count": len(op_indices),
        "files": files_written,
    }


def _process_targets(ctx, targets):
    manifest_targets = []
    for folder, kind, op_indices, _payload in targets:
        folder_path = os.path.join(ctx.args.output, folder)
        os.makedirs(folder_path, exist_ok=True)
        files_written = {}
        _write_csv_slice(ctx, folder_path, op_indices, files_written)
        _write_trace_slice(ctx, folder_path, op_indices, files_written)
        manifest_targets.append(
            _manifest_entry(folder, kind, op_indices, files_written))
    return manifest_targets


def _log_summary(args, targets, manifest, n_ops, manifest_path):
    total_covered = sum(t["ops_count"] for t in manifest["targets"])
    logger.info("split_artifacts → %s", args.output)
    logger.info("  targets: %d  ·  ops covered: %d/%d",
                len(targets), total_covered, n_ops)
    logger.info("  manifest: %s", manifest_path)
    for t in manifest["targets"][:8]:
        op_lo, op_hi = t["op_range"]
        noncontig = " noncontig" if t.get("op_indices") else ""
        extra = ""
        if "csv_rows" in t["files"]:
            extra += f"  csv={t['files']['csv_rows']}"
        if "trace_events" in t["files"]:
            extra += f"  trace={t['files']['trace_events']}"
        logger.info("  - %-48s %-22s ops %s-%s%s%s",
                    t["folder"], t["kind"], op_lo, op_hi, noncontig, extra)
    if len(targets) > 8:
        logger.info("  ... +%d more", len(targets) - 8)


def run(args):
    draft = load_json(args.draft)
    validate_draft_schema(draft)
    ops_data = load_json(args.raw_ops)
    ops_idx = build_op_index(ops_data["operators"])
    n_ops = len(ops_idx)

    _check_draft_op_bounds(draft, n_ops)
    targets = enumerate_targets(draft, n_ops)

    os.makedirs(args.output, exist_ok=True)

    trace_ctx = (None, None, None)
    if args.trace and not args.dry_run:
        trace_ctx = load_trace(args.trace)

    ctx = SliceContext(args=args, ops_idx=ops_idx, ops_data=ops_data,
                       trace_ctx=trace_ctx)
    manifest = {
        "source": _build_manifest_source(args),
        "n_ops": n_ops,
        "targets": _process_targets(ctx, targets),
    }

    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log_summary(args, targets, manifest, n_ops, manifest_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        run(args)
    except Exception as e:  # 顶层入口兜底，转非零退出码
        logger.error("split_artifacts failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
