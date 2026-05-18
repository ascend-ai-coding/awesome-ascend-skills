#!/usr/bin/env python3
"""auto_optimize.py — Auto-optimization driver for vllm bench serve.

Finds the maximum concurrency or request rate that satisfies SLO constraints
using exponential probe + binary search.

Usage:
    python3 auto_optimize.py \
        --base-url http://ip:port --model /path/to/weights --served-model-name MODEL_NAME \
        --backend openai-chat \
        --dataset-name random --random-input-len 1024 --random-output-len 128 \
        --slo "p99_ttft:500" --slo "mean_tpot:50" --slo "success_rate:95" \
        --search-mode A \
        --coarse-multiplier 3 --fine-multiplier 6 --validation-multiplier 10 \
        --result-dir ./bench_results/optimize/opt_20260314

SLO key format:
    Latency: {mean|median|p50|p90|p95|p99}_{ttft|tpot|itl|e2el}:VALUE_MS  (check <=)
    success_rate:PERCENT   (check >=, 0-100)
    goodput_ratio:RATIO    (check >=, 0.0-1.0, requires --goodput-config)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow importing common.py from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BACKEND_ENDPOINT_MAP


def run_benchmark(base_args: str, search_param: str, search_value: int | float,
                  num_prompts: int, result_dir: str, result_filename: str,
                  fixed_param: str | None = None, fixed_value: int | float | None = None) -> dict:
    """Execute a single benchmark run and return parsed results."""
    cmd_parts = [
        "vllm bench serve",
        base_args,
        f"--{search_param} {search_value}",
        f"--num-prompts {num_prompts}",
        '--percentile-metrics "ttft,tpot,itl,e2el"',
        '--metric-percentiles "50,90,95,99"',
        "--save-result --save-detailed",
        f"--result-dir {result_dir}",
        f"--result-filename {result_filename}",
    ]

    if fixed_param and fixed_value is not None:
        cmd_parts.insert(3, f"--{fixed_param} {fixed_value}")

    cmd = " ".join(cmd_parts)
    print(f"\n  Running: {search_param}={search_value}, num_prompts={num_prompts}")
    print(f"  Command: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  ERROR: Exit code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return {"error": f"exit_code_{result.returncode}", "success": False}
    except subprocess.TimeoutExpired:
        print("  ERROR: Benchmark timed out (600s)")
        return {"error": "timeout", "success": False}

    # Parse result JSON
    result_path = os.path.join(result_dir, result_filename)
    if not os.path.exists(result_path):
        print(f"  WARNING: Result file not found: {result_path}")
        return {"error": "no_result_file", "success": False}

    with open(result_path) as f:
        data = json.load(f)

    return data


def parse_slo_specs(slo_list: list[str]) -> list[dict]:
    """Parse --slo KEY:VALUE pairs into structured SLO targets.

    Supported keys:
      Latency:  {mean|median|p50|p90|p95|p99}_{ttft|tpot|itl|e2el}  (ms, check <=)
      success_rate          (%, check >=)
      goodput_ratio         (0.0-1.0, check >=)

    Returns list of dicts: {key, value, json_key, direction}
    """
    import re

    LATENCY_METRICS = {"ttft", "tpot", "itl", "e2el"}
    AGGREGATIONS = {"mean", "median"}
    targets = []

    for spec in slo_list:
        if ":" not in spec:
            raise ValueError(f"Invalid SLO format '{spec}', expected 'KEY:VALUE'")
        key, val_str = spec.split(":", 1)
        value = float(val_str)

        if key == "success_rate":
            targets.append({
                "key": key, "value": value, "json_key": "_special_success_rate",
                "direction": ">=", "unit": "%",
            })
        elif key == "goodput_ratio":
            targets.append({
                "key": key, "value": value, "json_key": "_special_goodput_ratio",
                "direction": ">=", "unit": "",
            })
        else:
            # Try {agg}_{metric} pattern: mean_ttft, median_tpot, p99_e2el, etc.
            m = re.match(r"^(mean|median|p\d+)_(ttft|tpot|itl|e2el)$", key)
            if not m:
                raise ValueError(
                    f"Unknown SLO key '{key}'. Expected: "
                    f"{{mean|median|p50|p90|p95|p99}}_{{ttft|tpot|itl|e2el}}, "
                    f"success_rate, or goodput_ratio"
                )
            agg, metric = m.group(1), m.group(2)
            if agg in AGGREGATIONS:
                json_key = f"{agg}_{metric}_ms"
            else:
                # p99 → p99_ttft_ms (percentile key in result JSON)
                json_key = f"{agg}_{metric}_ms"
            targets.append({
                "key": key, "value": value, "json_key": json_key,
                "direction": "<=", "unit": "ms",
            })

    return targets


def check_slo(data: dict, slo_targets: list[dict]) -> tuple[bool, list[str]]:
    """Check if benchmark results meet all SLO targets.

    slo_targets: list from parse_slo_specs()
    """
    if "error" in data:
        return False, [f"Benchmark failed: {data['error']}"]

    violations = []

    for slo in slo_targets:
        key = slo["key"]
        target = slo["value"]
        direction = slo["direction"]

        if slo["json_key"] == "_special_success_rate":
            completed = data.get("completed", 0)
            failed = data.get("failed", 0)
            total = completed + failed
            actual = (completed / total * 100) if total > 0 else 0
            if actual < target:
                violations.append(f"success_rate: {actual:.1f}% < {target}%")

        elif slo["json_key"] == "_special_goodput_ratio":
            goodput = data.get("request_goodput")
            throughput = data.get("request_throughput")
            if goodput is not None and throughput and throughput > 0:
                actual = goodput / throughput
                if actual < target:
                    violations.append(
                        f"goodput_ratio: {actual:.3f} < {target} "
                        f"(goodput={goodput:.2f}, throughput={throughput:.2f})"
                    )
            else:
                violations.append(
                    "goodput_ratio: cannot compute — request_goodput not in results "
                    "(did you pass --goodput-config?)"
                )

        else:
            # Latency metric: look up json_key directly in result data
            json_key = slo["json_key"]
            actual = data.get(json_key)
            if actual is None:
                # For percentile keys like p99_ttft_ms, also try the percentiles list
                actual = _get_metric_from_percentiles(data, json_key)
            if actual is not None:
                if direction == "<=" and actual > target:
                    violations.append(f"{key}: {actual:.1f}ms > {target}ms")
                elif direction == ">=" and actual < target:
                    violations.append(f"{key}: {actual:.1f}ms < {target}ms")
            # If actual is None, metric not available — skip silently
            # (may happen for itl/tpot with non-streaming backends)

    return len(violations) == 0, violations


def _get_metric_from_percentiles(data: dict, json_key: str) -> float | None:
    """Fallback: extract metric from percentiles list if not a direct key.

    E.g., json_key='p99_ttft_ms' → look in data['percentiles_ttft_ms'] for p=99.
    """
    import re
    m = re.match(r"^p(\d+)_(ttft|tpot|itl|e2el)_ms$", json_key)
    if not m:
        return None
    percentile = float(m.group(1))
    metric = m.group(2)
    plist = data.get(f"percentiles_{metric}_ms")
    if plist and isinstance(plist, list):
        for p, v in plist:
            if abs(float(p) - percentile) < 0.1:
                return float(v)
    return None


def build_base_args(args, slo_targets: list[dict]) -> str:
    """Build the common benchmark arguments string."""
    parts = [
        f"--base-url {args.base_url}",
        f"--model {args.model}",
    ]
    if args.served_model_name:
        parts.append(f"--served-model-name {args.served_model_name}")
    endpoint = BACKEND_ENDPOINT_MAP.get(args.backend, "/v1/completions")
    parts.extend([
        f"--backend {args.backend}",
        f"--endpoint {endpoint}",
        f"--dataset-name {args.dataset_name}",
    ])
    # Inject --goodput flags for per-request SLO when goodput_ratio is used
    # vllm bench serve uses: --goodput "ttft:500" "tpot:50" (nargs="+", space-separated)
    if args.goodput_config:
        goodput_args = " ".join(f'"{gc}"' for gc in args.goodput_config)
        parts.append(f"--goodput {goodput_args}")
    # Use dataset-specific parameter names
    input_len = args.random_input_len or args.input_len
    output_len = args.random_output_len or args.output_len
    ds = args.dataset_name

    if ds in ("random", "random-mm", "random-rerank"):
        if input_len:
            parts.append(f"--random-input-len {input_len}")
        if output_len:
            parts.append(f"--random-output-len {output_len}")
    elif ds == "sonnet":
        if input_len:
            parts.append(f"--sonnet-input-len {input_len}")
        if output_len:
            parts.append(f"--sonnet-output-len {output_len}")
    elif ds == "sharegpt":
        if output_len:
            parts.append(f"--sharegpt-output-len {output_len}")
    elif ds in ("custom", "custom_mm"):
        if output_len:
            parts.append(f"--custom-output-len {output_len}")
    elif ds == "hf":
        if output_len:
            parts.append(f"--hf-output-len {output_len}")
    else:
        if input_len:
            parts.append(f"--input-len {input_len}")
        if output_len:
            parts.append(f"--output-len {output_len}")
    if args.dataset_path:
        parts.append(f"--dataset-path {args.dataset_path}")
    if args.random_range_ratio is not None:
        parts.append(f"--random-range-ratio {args.random_range_ratio}")
    if args.random_prefix_len:
        parts.append(f"--random-prefix-len {args.random_prefix_len}")
    if args.num_warmups:
        parts.append(f"--num-warmups {args.num_warmups}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Auto-optimize vllm bench serve")

    # Service params
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Model weight path (for tokenizer)")
    parser.add_argument("--served-model-name", default=None, help="API model name (if different from --model)")
    parser.add_argument("--backend", default="openai-chat")

    # Dataset params
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=None, help="General input length (alias)")
    parser.add_argument("--output-len", type=int, default=None, help="General output length (alias)")
    parser.add_argument("--random-input-len", type=int, default=None, help="Random dataset input length")
    parser.add_argument("--random-output-len", type=int, default=None, help="Random dataset output length")
    parser.add_argument("--random-range-ratio", type=float, default=None)
    parser.add_argument("--random-prefix-len", type=int, default=None)
    parser.add_argument("--num-warmups", type=int, default=5)

    # SLO targets — generic format: --slo "KEY:VALUE" (repeatable)
    # Keys: {mean|median|p50|p90|p95|p99}_{ttft|tpot|itl|e2el}, success_rate, goodput_ratio
    parser.add_argument("--slo", action="append", default=None, metavar="KEY:VALUE",
                        help='SLO constraint, e.g. "p99_ttft:500", "mean_tpot:50", '
                             '"success_rate:95", "goodput_ratio:0.9". Repeatable.')
    # Per-request goodput config — passed to vllm bench serve as --goodput "KEY:VALUE"
    # Required when using goodput_ratio SLO
    parser.add_argument("--goodput-config", action="append", default=None, metavar="METRIC:MS",
                        help='Per-request SLO for goodput, e.g. "ttft:500" "tpot:50". '
                             'Passed to vllm bench serve --goodput. Required for goodput_ratio SLO.')

    # Search configuration
    parser.add_argument("--search-mode", default="A", choices=["A", "B", "C", "D", "E"],
                        help="A=concurrency only, B=rate only, C=fixed concurrency search rate, "
                             "D=fixed rate search concurrency, E=joint")
    parser.add_argument("--fixed-concurrency", type=int, default=None, help="For mode C")
    parser.add_argument("--fixed-rate", type=float, default=None, help="For mode D")
    parser.add_argument("--coarse-multiplier", type=float, default=3.0)
    parser.add_argument("--fine-multiplier", type=float, default=6.0)
    parser.add_argument("--validation-multiplier", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--convergence-threshold", type=float, default=0.05)

    # Output
    parser.add_argument("--result-dir", default=None)

    args = parser.parse_args()

    # Build SLO targets
    if not args.slo:
        print("ERROR: At least one --slo required, e.g. --slo 'p99_ttft:500' --slo 'success_rate:95'")
        sys.exit(1)

    slo_targets = parse_slo_specs(args.slo)

    # Validate: goodput_ratio requires --goodput-config
    has_goodput_slo = any(s["key"] == "goodput_ratio" for s in slo_targets)
    if has_goodput_slo and not args.goodput_config:
        print("ERROR: --slo 'goodput_ratio:...' requires --goodput-config to define per-request SLOs")
        print("  Example: --goodput-config 'ttft:500' --goodput-config 'tpot:50'")
        sys.exit(1)

    # Determine search dimension
    mode = args.search_mode
    if mode == "A":
        search_param = "max-concurrency"
        fixed_param, fixed_value = None, None
    elif mode == "B":
        search_param = "request-rate"
        fixed_param, fixed_value = None, None
    elif mode == "C":
        search_param = "request-rate"
        fixed_param = "max-concurrency"
        fixed_value = args.fixed_concurrency
        if fixed_value is None:
            print("ERROR: Mode C requires --fixed-concurrency")
            sys.exit(1)
    elif mode == "D":
        search_param = "max-concurrency"
        fixed_param = "request-rate"
        fixed_value = args.fixed_rate
        if fixed_value is None:
            print("ERROR: Mode D requires --fixed-rate")
            sys.exit(1)
    elif mode == "E":
        # Two-stage: first A, then C at optimal
        search_param = "max-concurrency"
        fixed_param, fixed_value = None, None

    # Result directory
    if not args.result_dir:
        args.result_dir = f"./bench_results/optimize/opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.result_dir, exist_ok=True)

    base_args = build_base_args(args, slo_targets)
    exploration_history = []

    slo_display = [f"{s['key']}{s['direction']}{s['value']}{s['unit']}" for s in slo_targets]
    print("=" * 60)
    print("Auto-Optimization")
    print(f"  Mode: {mode}")
    print(f"  Search: {search_param}")
    print(f"  SLO targets: {', '.join(slo_display)}")
    print(f"  Multipliers: coarse={args.coarse_multiplier}, fine={args.fine_multiplier}, "
          f"validation={args.validation_multiplier}")
    print(f"  Result dir: {args.result_dir}")
    print("=" * 60)

    # === Phase 1: Warmup ===
    print("\n--- Phase 1: Warmup ---")
    warmup_data = run_benchmark(
        base_args, search_param, 1, 50, args.result_dir, "warmup.json",
        fixed_param, fixed_value,
    )
    slo_pass, violations = check_slo(warmup_data, slo_targets)
    exploration_history.append({
        "phase": "warmup", "value": 1, "num_prompts": 50,
        "slo_pass": slo_pass, "violations": violations, "file": "warmup.json",
    })

    if not slo_pass:
        print(f"\n  ABORT: SLO violated at minimum load!")
        for v in violations:
            print(f"    - {v}")
        _save_report(args.result_dir, None, slo_targets, exploration_history,
                     "Service cannot meet SLO constraints at minimum load.")
        sys.exit(1)

    # === Phase 2: Exponential Probe ===
    print("\n--- Phase 2: Exponential Probe ---")
    value = 1
    last_good = 1
    upper_bound = None

    for i in range(args.max_iterations):
        num_prompts = max(int(value * args.coarse_multiplier), 50)
        filename = f"probe_{i+1:03d}_{search_param[0]}{value}.json"

        data = run_benchmark(
            base_args, search_param, value, num_prompts, args.result_dir, filename,
            fixed_param, fixed_value,
        )
        slo_pass, violations = check_slo(data, slo_targets)
        exploration_history.append({
            "phase": "probe", "value": value, "num_prompts": num_prompts,
            "slo_pass": slo_pass, "violations": violations, "file": filename,
        })

        if slo_pass:
            print(f"  {search_param}={value}: PASS")
            last_good = value
            value = value * 2
        else:
            print(f"  {search_param}={value}: FAIL")
            for v in violations:
                print(f"    - {v}")
            upper_bound = value
            break

    if upper_bound is None:
        print(f"\n  Service not saturated at {search_param}={value//2}")
        _save_report(args.result_dir, last_good, slo_targets, exploration_history,
                     f"Service not saturated. Best tested: {search_param}={last_good}")
        sys.exit(0)

    lower_bound = last_good
    print(f"\n  Bounds: [{lower_bound}, {upper_bound}]")

    # === Phase 3: Binary Search ===
    print("\n--- Phase 3: Binary Search ---")
    is_integer_search = search_param == "max-concurrency"
    for i in range(8):
        mid = (lower_bound + upper_bound) / 2
        if is_integer_search:
            mid = int(mid)
            if mid == lower_bound:
                mid = lower_bound + 1
        else:
            mid = round(mid, 1)
            if mid <= lower_bound:
                mid = round(lower_bound + 0.1, 1)

        num_prompts = max(int(mid * args.fine_multiplier), 100)
        filename = f"search_{i+1:03d}_{search_param[0]}{mid}.json"

        data = run_benchmark(
            base_args, search_param, mid, num_prompts, args.result_dir, filename,
            fixed_param, fixed_value,
        )
        slo_pass, violations = check_slo(data, slo_targets)
        exploration_history.append({
            "phase": "search", "value": mid, "num_prompts": num_prompts,
            "slo_pass": slo_pass, "violations": violations, "file": filename,
        })

        if slo_pass:
            print(f"  {search_param}={mid}: PASS")
            lower_bound = mid
        else:
            print(f"  {search_param}={mid}: FAIL")
            upper_bound = mid

        # Check convergence
        if lower_bound > 0 and (upper_bound - lower_bound) / lower_bound < args.convergence_threshold:
            print(f"  Converged: [{lower_bound}, {upper_bound}]")
            break

    optimal = lower_bound
    print(f"\n  Optimal candidate: {search_param}={optimal}")

    # === Phase 4: Validation ===
    print("\n--- Phase 4: Validation ---")
    for retry in range(3):
        num_prompts = max(int(optimal * args.validation_multiplier), 200)
        filename = f"validation{'_retry' + str(retry) if retry > 0 else ''}.json"

        data = run_benchmark(
            base_args, search_param, optimal, num_prompts, args.result_dir, filename,
            fixed_param, fixed_value,
        )
        slo_pass, violations = check_slo(data, slo_targets)
        exploration_history.append({
            "phase": "validation", "value": optimal, "num_prompts": num_prompts,
            "slo_pass": slo_pass, "violations": violations, "file": filename,
        })

        if slo_pass:
            print(f"  Validation PASSED at {search_param}={optimal}")
            _save_report(args.result_dir, optimal, slo_targets, exploration_history,
                         f"Optimal {search_param}={optimal}. Validated with {num_prompts} prompts.",
                         data)
            print(f"\n{'='*60}")
            print(f"RESULT: Optimal {search_param} = {optimal}")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print(f"  Validation FAILED at {search_param}={optimal}")
            optimal = int(optimal * 0.9)
            if optimal < 1:
                optimal = 1

    print(f"\n  WARNING: Could not validate. Best effort: {search_param}={optimal}")
    _save_report(args.result_dir, optimal, slo_targets, exploration_history,
                 f"Could not fully validate. Best effort: {search_param}={optimal}")


def _save_report(result_dir: str, optimal_value: int | None, slo_targets: list[dict],
                 history: list, recommendation: str, metrics_data: dict | None = None):
    """Save optimization report JSON."""
    report = {
        "optimal_value": optimal_value,
        "slo_targets": [{"key": s["key"], "direction": s["direction"], "value": s["value"]}
                        for s in slo_targets],
        "recommendation": recommendation,
        "exploration_history": history,
        "timestamp": datetime.now().isoformat(),
    }
    if metrics_data:
        report["metrics_at_optimal"] = {
            "request_throughput": metrics_data.get("request_throughput"),
            "request_goodput": metrics_data.get("request_goodput"),
            "output_throughput": metrics_data.get("output_throughput"),
            "completed": metrics_data.get("completed"),
            "failed": metrics_data.get("failed"),
        }

    report_path = os.path.join(result_dir, "optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
