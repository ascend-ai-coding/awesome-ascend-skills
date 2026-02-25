#!/usr/bin/env python3
"""
AISBench Result Parser
Parse and summarize evaluation results from AISBench output directories.

Usage:
    python parse_results.py <output_dir>
    python parse_results.py <output_dir> --format json
    python parse_results.py <output_dir> --format csv --output results.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def find_summary_files(output_dir: str) -> list:
    """Find all summary files in output directory."""
    summary_files = []
    output_path = Path(output_dir)

    for f in output_path.rglob("summary_*.csv"):
        summary_files.append(f)

    return sorted(summary_files)


def parse_summary_csv(csv_path: Path) -> list:
    """Parse summary CSV file."""
    results = []

    with open(csv_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return results

    # Parse header
    header = lines[0].strip().split(",")

    # Parse data rows
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.strip().split(",")
        result = dict(zip(header, values))
        results.append(result)

    return results


def parse_results_json(json_path: Path) -> dict:
    """Parse results JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def find_result_jsons(output_dir: str) -> list:
    """Find all result JSON files."""
    result_files = []
    output_path = Path(output_dir)

    for f in output_path.rglob("results/**/*.json"):
        result_files.append(f)

    return sorted(result_files)


def format_results_table(results: list) -> str:
    """Format results as ASCII table."""
    if not results:
        return "No results found."

    # Get all columns
    columns = list(results[0].keys())

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for r in results:
        for col in columns:
            widths[col] = max(widths[col], len(str(r.get(col, ""))))

    # Build table
    lines = []

    # Header
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    lines.append(header)
    lines.append(separator)

    # Data rows
    for r in results:
        row = " | ".join(str(r.get(col, "")).ljust(widths[col]) for col in columns)
        lines.append(row)

    return "\n".join(lines)


def format_results_json(results: list) -> str:
    """Format results as JSON."""
    return json.dumps(results, indent=2, ensure_ascii=False)


def format_results_csv(results: list) -> str:
    """Format results as CSV."""
    if not results:
        return ""

    columns = list(results[0].keys())
    lines = [",".join(columns)]

    for r in results:
        row = ",".join(str(r.get(col, "")) for col in columns)
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse AISBench evaluation results")
    parser.add_argument("output_dir", help="AISBench output directory")
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Find and parse summary files
    summary_files = find_summary_files(output_dir)

    all_results = []
    for sf in summary_files:
        results = parse_summary_csv(sf)
        all_results.extend(results)

    if not all_results:
        # Try to find individual result JSONs
        json_files = find_result_jsons(output_dir)
        for jf in json_files:
            try:
                data = parse_results_json(jf)
                if isinstance(data, list):
                    all_results.extend(data)
                elif isinstance(data, dict):
                    all_results.append(data)
            except json.JSONDecodeError:
                continue

    # Format output
    if args.format == "table":
        output = format_results_table(all_results)
    elif args.format == "json":
        output = format_results_json(all_results)
    elif args.format == "csv":
        output = format_results_csv(all_results)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
