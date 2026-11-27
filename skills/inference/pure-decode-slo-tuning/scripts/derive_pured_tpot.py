#!/usr/bin/env python3
"""Derive pure-D TPOT from real acceptance rates and fake-KV ITL results."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", required=True, help="real Prefill+Decode bench JSON")
    parser.add_argument("--acceptance-expected-requests", type=int)
    parser.add_argument("--spec-tokens", type=int, help="number of speculative positions; default: all measured")
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--min-success-rate", type=float, required=True)
    parser.add_argument("fake_results", nargs="+", help="fake-KV bench JSON files")
    args = parser.parse_args()

    if not 0 < args.min_success_rate <= 1:
        parser.error("--min-success-rate must be in (0, 1]")
    if args.slo_ms <= 0:
        parser.error("--slo-ms must be positive")

    measured = read_json(args.acceptance)
    rates = measured.get("spec_decode_per_position_acceptance_rates")
    if not isinstance(rates, list) or not rates:
        parser.error("acceptance JSON has no position-wise rates")
    if any(not isinstance(rate, (int, float)) or not 0 <= rate <= 1 for rate in rates):
        parser.error("acceptance rates must be numeric values in [0, 1]")

    positions = len(rates) if args.spec_tokens is None else args.spec_tokens
    if not 1 <= positions <= len(rates):
        parser.error("--spec-tokens exceeds measured positions")
    real_completed = int(measured.get("completed", 0))
    real_failed = int(measured.get("failed", 0))
    if real_completed <= 0 or real_failed:
        parser.error("acceptance run has no successful requests or contains failures")
    if args.acceptance_expected_requests is not None:
        if real_completed != args.acceptance_expected_requests or real_failed:
            parser.error("acceptance run did not complete the requested sample successfully")

    al = 1 + sum(rates[:positions])
    rows = []
    for filename in args.fake_results:
        result = read_json(filename)
        completed = int(result.get("completed", 0))
        failed = int(result.get("failed", 0))
        total = completed + failed
        if total <= 0:
            parser.error(f"{filename}: no completed or failed requests")
        percentiles = {}
        for percentile in (50, 90, 99):
            key = f"p{percentile}_itl_ms"
            if key not in result:
                parser.error(f"{filename}: missing {key}")
            percentiles[f"tpot_p{percentile}_ms"] = result[key] / al
        success_rate = completed / total
        rows.append(
            {
                "file": filename,
                "completed": completed,
                "failed": failed,
                "success_rate": success_rate,
                "request_qps": result.get("request_throughput"),
                "output_tokens_per_s": result.get("output_throughput"),
                "ttft_p50_ms": result.get("p50_ttft_ms"),
                "itl_p50_ms": result["p50_itl_ms"],
                **percentiles,
                "bench_native_tpot_p50_ms": result.get("p50_tpot_ms"),
                "slo_pass": success_rate >= args.min_success_rate
                and percentiles["tpot_p50_ms"] < args.slo_ms,
            }
        )

    print(
        json.dumps(
            {
                "acceptance_file": args.acceptance,
                "acceptance_completed": real_completed,
                "acceptance_failed": real_failed,
                "position_rates": rates[:positions],
                "spec_tokens": positions,
                "al": al,
                "slo_ms": args.slo_ms,
                "min_success_rate": args.min_success_rate,
                "runs": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
