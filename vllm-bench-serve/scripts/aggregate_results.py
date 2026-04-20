#!/usr/bin/env python3
"""aggregate_results.py — Aggregate vllm bench serve result JSONs into a comparison table.

Usage:
    python3 aggregate_results.py --result-dir ./bench_results/batch/batch_20260314 --format markdown

Supported formats: markdown (default), json, csv
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_result_file(filepath: str) -> dict | None:
    """Parse a vllm bench serve result JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {"file": filepath, "error": str(e)}

    # Handle both flat format and nested format
    # vllm bench serve saves results with various keys depending on version
    result = {
        "file": os.path.basename(filepath),
        "model": data.get("model", "unknown"),
        "backend": data.get("backend", "unknown"),
        "dataset": data.get("dataset_name", "unknown"),
    }

    # Load parameters
    result["num_prompts"] = data.get("num_prompts", data.get("total_requests", "?"))
    result["request_rate"] = data.get("request_rate", "?")
    result["max_concurrency"] = data.get("max_concurrency", "?")
    result["input_len"] = data.get("random_input_len", data.get("input_len", "?"))
    result["output_len"] = data.get("random_output_len", data.get("output_len", "?"))

    # Core metrics
    result["completed"] = data.get("completed", 0)
    result["failed"] = data.get("failed", 0)
    total = result["completed"] + result["failed"]
    result["success_rate"] = (
        f"{result['completed'] / total * 100:.1f}%" if total > 0 else "N/A"
    )

    result["req_throughput"] = _fmt(data.get("request_throughput"), 2)
    result["output_tok_s"] = _fmt(data.get("output_throughput"), 1)
    result["total_tok_s"] = _fmt(data.get("total_token_throughput"), 1)

    # Latency metrics (generation tasks)
    result["ttft_mean"] = _fmt(data.get("mean_ttft_ms"), 2)
    result["ttft_p99"] = _extract_percentile(data, "ttft", 99)
    result["tpot_mean"] = _fmt(data.get("mean_tpot_ms"), 2)
    result["tpot_p99"] = _extract_percentile(data, "tpot", 99)
    result["e2e_mean"] = _fmt(data.get("mean_e2el_ms"), 2)
    result["e2e_p99"] = _extract_percentile(data, "e2el", 99)

    return result


def _fmt(value, decimals=2) -> str:
    """Format a numeric value or return '—'."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _extract_percentile(data: dict, metric: str, percentile: int) -> str:
    """Extract a specific percentile from result data."""
    # Try percentiles list format: [(p, value), ...]
    key = f"percentiles_{metric}_ms"
    plist = data.get(key)
    if plist and isinstance(plist, list):
        for p, v in plist:
            if abs(float(p) - percentile) < 0.1:
                return _fmt(v, 2)

    # Try direct key format
    direct_key = f"{metric}_p{percentile}"
    if direct_key in data:
        return _fmt(data[direct_key], 2)

    return "—"


def format_markdown(results: list[dict]) -> str:
    """Format results as a Markdown table."""
    if not results:
        return "No results found."

    headers = [
        "File", "Concurrency", "Rate", "Prompts",
        "Req/s", "Out tok/s", "TTFT mean", "TTFT P99",
        "TPOT mean", "TPOT P99", "E2E P99", "Success%",
    ]

    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["file"], f"ERROR: {r['error']}"] + ["—"] * (len(headers) - 2))
            continue
        rows.append([
            r["file"],
            str(r["max_concurrency"]),
            str(r["request_rate"]),
            str(r["num_prompts"]),
            r["req_throughput"],
            r["output_tok_s"],
            r["ttft_mean"],
            r["ttft_p99"],
            r["tpot_mean"],
            r["tpot_p99"],
            r["e2e_p99"],
            r["success_rate"],
        ])

    # Calculate column widths
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]

    # Build table
    lines = []
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(headers))
        ) + " |")

    return "\n".join(lines)


def format_csv(results: list[dict]) -> str:
    """Format results as CSV."""
    headers = [
        "file", "model", "backend", "dataset", "max_concurrency", "request_rate",
        "num_prompts", "req_throughput", "output_tok_s", "total_tok_s",
        "ttft_mean", "ttft_p99", "tpot_mean", "tpot_p99", "e2e_mean", "e2e_p99",
        "completed", "failed", "success_rate",
    ]
    lines = [",".join(headers)]
    for r in results:
        if "error" in r:
            lines.append(f"{r['file']},ERROR,{r['error']}" + ",," * (len(headers) - 3))
            continue
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate vllm bench serve results")
    parser.add_argument("--result-dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json", "csv"])
    parser.add_argument("--sort-by", default="req_throughput",
                        help="Sort by this field (default: req_throughput)")
    parser.add_argument("--baseline", default=None,
                        help="Path to a baseline result JSON for regression detection")

    args = parser.parse_args()

    # Find all JSON result files
    pattern = os.path.join(args.result_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSON files found in {args.result_dir}")
        sys.exit(1)

    # Parse all files
    results = []
    for f in files:
        # Skip summary/report files
        basename = os.path.basename(f)
        if basename in ("batch_summary.json", "optimization_report.json"):
            continue
        parsed = parse_result_file(f)
        if parsed:
            results.append(parsed)

    if not results:
        print("No valid result files found.")
        sys.exit(1)

    # Sort by throughput (descending), handling non-numeric values
    def sort_key(r):
        try:
            return float(r.get(args.sort_by, 0))
        except (TypeError, ValueError):
            return 0

    results.sort(key=sort_key, reverse=True)

    # Output
    if args.format == "markdown":
        print(format_markdown(results))
    elif args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "csv":
        print(format_csv(results))

    # Regression detection
    if args.baseline and args.format == "markdown":
        baseline = parse_result_file(args.baseline)
        if baseline and "error" not in baseline:
            print("\n### Regression Check (vs baseline)")
            print(f"Baseline: `{args.baseline}`\n")
            _compare_metrics = [
                ("req_throughput", "Req/s", True),     # higher is better
                ("output_tok_s", "Out tok/s", True),   # higher is better
                ("ttft_p99", "TTFT P99", False),       # lower is better
                ("tpot_p99", "TPOT P99", False),       # lower is better
                ("e2e_p99", "E2E P99", False),         # lower is better
            ]
            for r in [x for x in results if "error" not in x]:
                deltas = []
                for key, label, higher_is_better in _compare_metrics:
                    try:
                        cur = float(r.get(key, 0))
                        base = float(baseline.get(key, 0))
                        if base > 0:
                            pct = (cur - base) / base * 100
                            marker = ""
                            if higher_is_better and pct < -5:
                                marker = " REGRESSION"
                            elif not higher_is_better and pct > 5:
                                marker = " REGRESSION"
                            deltas.append(f"  {label}: {pct:+.1f}%{marker}")
                    except (TypeError, ValueError):
                        pass
                if deltas:
                    print(f"**{r['file']}**")
                    for d in deltas:
                        print(d)
                    print()

    # Summary stats
    if args.format == "markdown":
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            print(f"\n**Summary:** {len(valid_results)} results aggregated, "
                  f"sorted by {args.sort_by} (descending)")


if __name__ == "__main__":
    main()
