#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer Norm 性能对比：工程 torch.ops.npu.layer_norm（ascend_kernel） vs
同设备上 torch.nn.functional.layer_norm（原生 dispatch）。

采集必须使用 torch_npu.profiler；warmup/active 固定为 5/5（技能 ascendc-operator-performance-eval）。
仅输出 Markdown 报告，不写 *_profiler_results.json。
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import torch
import torch_npu

from layer_norm_profiler_common import (
    PROFILER_SCHEDULE_ACTIVE,
    PROFILER_SCHEDULE_WARMUP,
    default_case_file,
    default_trace_root,
    load_cases,
    render_layer_norm_comparison_report_md,
    run_layer_norm_profiler_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer Norm：工程 layer_norm vs F.layer_norm（torch_npu.profiler，warmup/active=5）"
    )
    parser.add_argument("--case-file", default=None, help="用例 JSONL（仅 .jsonl）")
    parser.add_argument(
        "--trace-root",
        default=None,
        help="tensorboard_trace_handler 根目录（默认: 本 test/profiler_trace）",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--divisor-mode",
        choices=("active_steps", "active_only"),
        default="active_steps",
        help="sum(Total Time) 的除数：active_steps=active×repeat；active_only=仅 active",
    )
    parser.add_argument("--only-case", type=int, default=None)
    parser.add_argument("--report-md", default=None)
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU 不可用", file=sys.stderr)
        sys.exit(2)

    case_file = args.case_file or default_case_file()
    if not os.path.isfile(case_file):
        print(f"ERROR: case file not found: {case_file}", file=sys.stderr)
        sys.exit(2)

    all_cases = load_cases(case_file)
    if not all_cases:
        print("ERROR: no cases loaded", file=sys.stderr)
        sys.exit(2)

    if args.only_case is not None:
        if args.only_case < 0 or args.only_case >= len(all_cases):
            print("ERROR: --only-case out of range", file=sys.stderr)
            sys.exit(2)
        idxs = [args.only_case]
    else:
        idxs = None

    trace_root = args.trace_root or default_trace_root()
    device = torch.device(args.device)
    div = (
        PROFILER_SCHEDULE_ACTIVE * args.repeat
        if args.divisor_mode == "active_steps"
        else PROFILER_SCHEDULE_ACTIVE
    )
    print(f"[INFO] cases={len(idxs or all_cases)} case_file={case_file}")
    print(
        f"[INFO] schedule wait={args.wait} warmup={PROFILER_SCHEDULE_WARMUP} "
        f"active={PROFILER_SCHEDULE_ACTIVE} repeat={args.repeat} divisor_mode={args.divisor_mode} -> /{div}"
    )

    results = run_layer_norm_profiler_cases(
        all_cases=all_cases,
        case_indices=idxs,
        device=device,
        trace_root=trace_root,
        trace_subdir="layer_norm_torch_npu",
        modes=["custom", "native"],
        load_ascend_kernel=True,
        wait=args.wait,
        warmup=PROFILER_SCHEDULE_WARMUP,
        active=PROFILER_SCHEDULE_ACTIVE,
        repeat=args.repeat,
        divisor_mode=args.divisor_mode,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    md_path = args.report_md or os.path.join(here, "layer_norm_torch_npu_profiler_report.md")
    base_out = os.path.join(trace_root, "layer_norm_torch_npu")
    generated_at = datetime.datetime.now().isoformat(timespec="seconds")

    md_body = render_layer_norm_comparison_report_md(
        title="Layer Norm torch_npu.profiler 性能对比",
        generated_at=generated_at,
        case_file=os.path.abspath(case_file),
        trace_root=os.path.abspath(base_out),
        div=float(div),
        divisor_mode=args.divisor_mode,
        results=results,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_body)

    print(f"[INFO] wrote {md_path}")


if __name__ == "__main__":
    main()
