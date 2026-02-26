#!/usr/bin/env python3
"""HCCL Test Result Parser - 解析 HCCL Test 输出并生成汇总表格

Usage:
  ./parse-hccl-result.py output.log              Parse from file
  cat output.log | ./parse-hccl-result.py        Parse from stdin
  ./parse-hccl-result.py output.log -f markdown  Output as Markdown table
"""

import re
import sys
import argparse
from typing import List, Dict


def parse_hccl_output(content: str) -> List[Dict]:
    pattern = re.compile(r"^\s*(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\w+)", re.MULTILINE)
    results = []

    for match in pattern.finditer(content):
        results.append(
            {
                "data_size": int(match.group(1)),
                "avg_time": float(match.group(2)),
                "alg_bandwidth": float(match.group(3)),
                "check_result": match.group(4),
            }
        )

    return results


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def print_markdown_table(results: List[Dict]) -> None:
    if not results:
        print("No results found.")
        return

    print("| Data Size | Avg Time (us) | Bandwidth (GB/s) | Check Result |")
    print("|-----------|---------------|------------------|--------------|")

    total_bandwidth = 0
    for r in results:
        print(
            f"| {format_size(r['data_size']):>9} | {r['avg_time']:>13.2f} | {r['alg_bandwidth']:>16.4f} | {r['check_result']:>12} |"
        )
        total_bandwidth += r["alg_bandwidth"]

    avg_bandwidth = total_bandwidth / len(results)
    print()
    print(f"**Total Tests:** {len(results)}")
    print(f"**Average Bandwidth:** {avg_bandwidth:.4f} GB/s")
    print(f"**Max Bandwidth:** {max(r['alg_bandwidth'] for r in results):.4f} GB/s")
    print(f"**Min Bandwidth:** {min(r['alg_bandwidth'] for r in results):.4f} GB/s")


def print_summary_table(results: List[Dict]) -> None:
    if not results:
        print("No results found.")
        return

    print("-" * 70)
    print(f"{'Data Size':>12} {'Time (us)':>12} {'BW (GB/s)':>12} {'Check':>10}")
    print("-" * 70)

    total_bandwidth = 0
    for r in results:
        print(
            f"{format_size(r['data_size']):>12} {r['avg_time']:>12.2f} {r['alg_bandwidth']:>12.4f} {r['check_result']:>10}"
        )
        total_bandwidth += r["alg_bandwidth"]

    print("-" * 70)
    avg_bandwidth = total_bandwidth / len(results)
    max_bw = max(r["alg_bandwidth"] for r in results)
    min_bw = min(r["alg_bandwidth"] for r in results)
    print(
        f"Total: {len(results)} tests | Avg BW: {avg_bandwidth:.4f} GB/s | Max: {max_bw:.4f} GB/s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse HCCL Test output and generate summary table"
    )
    parser.add_argument(
        "file", nargs="?", help="HCCL test output file (default: stdin)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--sort",
        "-s",
        choices=["size", "bandwidth", "time"],
        default="size",
        help="Sort by column (default: size)",
    )

    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        content = sys.stdin.read()

    results = parse_hccl_output(content)

    if not results:
        print("No HCCL test results found in input.", file=sys.stderr)
        sys.exit(1)

    sort_key = "data_size" if args.sort == "size" else args.sort
    reverse = args.sort == "bandwidth"
    results.sort(key=lambda x: x[sort_key], reverse=reverse)

    if args.format == "markdown":
        print_markdown_table(results)
    else:
        print_summary_table(results)


if __name__ == "__main__":
    main()
