#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import collect  # type: ignore
import visualize  # type: ignore

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)

COLLECTION_ONLY_FEATURES = {"kernel-scale"}

VISUAL_FEATURE_MAP = {
    "memory-detail": "memory",
    "timeline-detail": "timeline-detail",
}


def normalize_feature(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _base_collect_args(args: argparse.Namespace) -> List[str]:
    return [
        "--operator-path", args.operator_path,
        "--output", args.output,
        "--preset", args.preset,
        "--launch-count", str(args.launch_count),
        "--timeout", str(args.timeout),
        "--preflight-mode", args.preflight_mode,
        "--preflight-timeout", str(args.preflight_timeout),
        "--preflight-cache-seconds", str(getattr(args, "preflight_cache_seconds", 300)),
        "--heartbeat-seconds", str(args.heartbeat_seconds),
        "--simt", args.simt,
        "--source-stall", args.source_stall,
        "--env-profile-mode", args.env_profile_mode,
        "--env-source-timeout", str(args.env_source_timeout),
    ]


def _append_app_collect_args(out: List[str], args: argparse.Namespace) -> None:
    if args.app:
        out += ["--app", args.app]
    if args.app_cwd:
        out += ["--app-cwd", args.app_cwd]
    if args.memory_info:
        out += ["--memory-info", args.memory_info]
    if args.kernel_name:
        out += ["--kernel-name", args.kernel_name]


def _append_profiler_collect_args(out: List[str], args: argparse.Namespace) -> None:
    if args.launch_skip_before_match is not None:
        out += ["--launch-skip-before-match", str(args.launch_skip_before_match)]
    if args.replay_mode:
        out += ["--replay-mode", args.replay_mode]
    if args.warm_up is not None:
        out += ["--warm-up", str(args.warm_up)]
    if args.kill:
        out += ["--kill", args.kill]
    if args.mstx:
        out += ["--mstx", args.mstx]
    if args.mstx_include:
        out += ["--mstx-include", args.mstx_include]
    if args.dump:
        out += ["--dump", args.dump]
    if args.core_id:
        out += ["--core-id", args.core_id]


def _append_tool_collect_args(out: List[str], args: argparse.Namespace) -> None:
    if args.op_type:
        out += ["--op-type", args.op_type]
    if args.msprof:
        out += ["--msprof", args.msprof]
    if args.env_profile:
        out += ["--env-profile", args.env_profile]
    if args.instr_timeline_pipe:
        out += ["--instr-timeline-pipe", args.instr_timeline_pipe]
    if args.debug_rebuild_command:
        out += ["--debug-rebuild-command", args.debug_rebuild_command]


def _append_repeatable_collect_args(out: List[str], args: argparse.Namespace) -> None:
    for value in args.env_source:
        out += ["--env-source", value]
    for value in args.env_var:
        out += ["--env-var", value]
    out += ["--debug-rebuild-timeout", str(args.debug_rebuild_timeout)]
    out += ["--kernel-scale", args.kernel_scale]
    for value in args.build_config:
        out += ["--build-config", value]
    for value in args.validation_note:
        out += ["--validation-note", value]
    for value in args.app_arg:
        out += ["--app-arg", value]
    for value in args.block_timeout:
        out += ["--block-timeout", value]
    for feature in args.feature:
        out += ["--feature", normalize_feature(feature)]


def _append_flag_collect_args(out: List[str], args: argparse.Namespace) -> None:
    if args.clean:
        out.append("--clean")
    if args.strict:
        out.append("--strict")
    if args.dry_run:
        out.append("--dry-run")
    if not args.reuse_existing:
        out.append("--no-reuse-existing")
    if getattr(args, "independent_default", False):
        out.append("--independent-default")
    if not args.circuit_breaker:
        out.append("--no-circuit-breaker")
    if not args.adaptive_timeout:
        out.append("--no-adaptive-timeout")


def build_collect_args(args: argparse.Namespace) -> List[str]:
    out = _base_collect_args(args)
    _append_app_collect_args(out, args)
    _append_profiler_collect_args(out, args)
    _append_tool_collect_args(out, args)
    _append_repeatable_collect_args(out, args)
    _append_flag_collect_args(out, args)
    return out


def build_visualize_args(args: argparse.Namespace) -> List[str]:
    preset = "complete" if args.preset == "deep" else args.preset
    out = [
        "--input", args.output,
        "--output", args.output,
        "--preset", preset,
        "--max-trace-events", str(args.max_trace_events),
        "--max-raw-rows", str(args.max_raw_rows),
        "--report-name", args.report_name,
        "--unavailable-policy", args.unavailable_policy,
    ]
    seen = set()
    if args.memory_info:
        out += ["--memory-info", args.memory_info]
    if not args.compact_payload:
        out.append("--no-compact-payload")
    if args.pretty_payload:
        out.append("--pretty-payload")
    for raw in args.feature:
        normalized = normalize_feature(raw)
        if normalized in COLLECTION_ONLY_FEATURES:
            continue
        feature = VISUAL_FEATURE_MAP.get(normalized, normalized)
        if feature not in seen:
            seen.add(feature)
            out += ["--feature", feature]
    return out


def write_pipeline_timing(output: str, wall_started: float, phase_seconds: dict, summary: dict) -> dict:
    timing = {
        "schema": "msopprof-pipeline-timing/v1",
        "wall_seconds": round(time.perf_counter() - wall_started, 6),
        "phases": [
            {"phase": name, "elapsed_seconds": round(value, 6)}
            for name, value in phase_seconds.items()
        ],
        "scope_note": ("Measures run_pipeline.py execution only; agent reading and manual preparation are "
            "outside this timer."),
    }
    timing_dir = Path(output).resolve() / "_internal"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / "pipeline_timing.json"
    tmp = timing_path.with_suffix(timing_path.suffix + ".tmp")
    tmp.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(timing_path)
    summary["timing"] = timing
    return timing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-step msOpProf collection + universal visualization pipeline."
    )
    _add_collection_arguments(parser)
    _add_visualize_arguments(parser)
    return parser


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--operator-path", help="Operator sample/project root.")
    parser.add_argument("--output", required=True, help="Unified collection and report output root.")
    parser.add_argument("--mode", choices=["full", "collect", "visualize"], default="full")
    parser.add_argument("--app")
    parser.add_argument("--app-cwd", help="Application working directory; defaults to --operator-path.")
    parser.add_argument("--kernel-name")
    parser.add_argument("--op-type")
    parser.add_argument("--simt", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--source-stall", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--kernel-scale", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--app-arg", action="append", default=[])
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--env-profile", help=("Reusable CANN/msOpProf environment profile; relative paths "
        "resolve from --operator-path."))
    parser.add_argument("--env-profile-mode", choices=["auto", "readonly", "refresh", "off"], default="auto")
    parser.add_argument("--env-source", action="append", default=[], help=("Repeatable shell source/init "
        "command used only when creating or refreshing the environment profile."))
    parser.add_argument("--env-var", action="append", default=[], metavar="KEY=VALUE", help=("Repeatable "
        "non-secret environment value saved in the profile."))
    parser.add_argument("--env-source-timeout", type=int, default=120)
    parser.add_argument("--preset", choices=["fast", "core", "complete", "deep"], default="complete")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--launch-skip-before-match", type=int)
    parser.add_argument("--replay-mode", choices=["kernel", "application", "range"])
    parser.add_argument("--warm-up", type=int)
    parser.add_argument("--kill", choices=["on", "off"])
    parser.add_argument("--mstx", choices=["on", "off"])
    parser.add_argument("--mstx-include")
    parser.add_argument("--dump", choices=["on", "off"])
    parser.add_argument("--core-id")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--preflight-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--preflight-timeout", type=int, default=60)
    parser.add_argument("--preflight-cache-seconds", type=int, default=300)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--circuit-breaker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-timeout", action="append", default=[], metavar="BLOCK=SECONDS")
    parser.add_argument("--instr-timeline-pipe")
    parser.add_argument("--debug-rebuild-command")
    parser.add_argument("--debug-rebuild-timeout", type=int, default=1800)
    parser.add_argument("--build-config", action="append", default=[])
    parser.add_argument("--validation-note", action="append", default=[])


def _add_visualize_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-trace-events", type=int, default=10000)
    parser.add_argument("--max-raw-rows", type=int, default=5000)
    parser.add_argument("--report-name", default="report.html", help="Output HTML filename inside --output.")
    parser.add_argument("--memory-info", help="Explicit memory_info.json for targeted On-Chip Memory visualization.")
    parser.add_argument("--unavailable-policy", choices=["auto", "omit", "explain"], default="auto")
    parser.add_argument("--compact-payload", action=argparse.BooleanOptionalAction, default=True, help=("Compact "
        "duplicated Source relation data in report artifacts."))
    parser.add_argument("--pretty-payload", action="store_true", help=("Pretty-print report_payload.json "
        "instead of compact JSON."))
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--independent-default", action="store_true", help=("Force an independent Default "
        "replay instead of reusing Roofline."))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def emit_summary(output: str, wall_started: float, phase_seconds: dict, summary: dict, rc: int) -> int:
    write_pipeline_timing(output, wall_started, phase_seconds, summary)
    cli_logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return rc


def run_collect_mode(args: argparse.Namespace, wall_started: float,
    phase_seconds: dict, summary: dict) -> Optional[int]:
    phase_started = time.perf_counter()
    rc = collect.main(build_collect_args(args))
    phase_seconds["collection"] = time.perf_counter() - phase_started
    summary["collection"] = {"return_code": rc, "elapsed_seconds": round(phase_seconds["collection"], 6)}
    if rc != 0:
        return emit_summary(args.output, wall_started, phase_seconds, summary, rc)
    if args.dry_run or args.mode == "collect":
        return emit_summary(args.output, wall_started, phase_seconds, summary, 0)
    return None


def run_visualize_mode(args: argparse.Namespace, wall_started: float, phase_seconds: dict, summary: dict) -> int:
    normalized_features = [normalize_feature(x) for x in args.feature]
    if args.mode == "full" and normalized_features and all(x in COLLECTION_ONLY_FEATURES for x in normalized_features):
        summary["visualization"] = {"return_code": 0, "skipped": True, "reason": ("Requested features are "
            "collection-only.")}
        return emit_summary(args.output, wall_started, phase_seconds, summary, 0)
    phase_started = time.perf_counter()
    rc = visualize.main(build_visualize_args(args))
    phase_seconds["visualization"] = time.perf_counter() - phase_started
    summary["visualization"] = {
        "return_code": rc,
        "report": str(Path(args.output).resolve() / Path(args.report_name).name),
        "elapsed_seconds": round(phase_seconds["visualization"], 6),
    }
    return emit_summary(args.output, wall_started, phase_seconds, summary, rc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode in {"full", "collect"} and not args.operator_path:
        parser.error("--operator-path is required for collection.")

    wall_started = time.perf_counter()
    phase_seconds = {}
    summary = {"mode": args.mode, "collection": None, "visualization": None}

    if args.mode in {"full", "collect"}:
        rc = run_collect_mode(args, wall_started, phase_seconds, summary)
        if rc is not None:
            return rc

    if args.mode in {"full", "visualize"}:
        return run_visualize_mode(args, wall_started, phase_seconds, summary)

    return emit_summary(args.output, wall_started, phase_seconds, summary, 0)


if __name__ == "__main__":
    raise SystemExit(main())
