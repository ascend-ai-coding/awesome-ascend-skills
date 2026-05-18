#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from typing import Any


ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"traceback",
        r"runtimeerror",
        r"assertionerror",
        r"valueerror",
        r"error",
        r"exception",
        r"hccl",
        r"npu",
        r"cuda",
        r"timeout",
        r"killed",
        r"exit code",
    ]
]


def read_tail(path: pathlib.Path, max_lines: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def extract_error_block(lines: list[str], context_before: int, context_after: int) -> dict[str, Any]:
    last_match = None
    for idx, line in enumerate(lines):
        if any(pattern.search(line) for pattern in ERROR_PATTERNS):
            last_match = idx

    if last_match is None:
        excerpt = lines[-min(len(lines), context_before + context_after + 40):]
        return {
            "matched": False,
            "anchor_line": None,
            "excerpt": excerpt,
            "summary": "No explicit error marker found in the inspected tail.",
        }

    start = max(0, last_match - context_before)
    end = min(len(lines), last_match + context_after + 1)
    excerpt = lines[start:end]
    summary = lines[last_match].strip()
    return {
        "matched": True,
        "anchor_line": last_match + 1,
        "excerpt": excerpt,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the most relevant failure block from a training log tail.")
    parser.add_argument("path", help="Path to the training log.")
    parser.add_argument("--tail-lines", type=int, default=300, help="Number of tail lines to inspect.")
    parser.add_argument("--context-before", type=int, default=40, help="Lines to keep before the last match.")
    parser.add_argument("--context-after", type=int, default=80, help="Lines to keep after the last match.")
    args = parser.parse_args()

    path = pathlib.Path(args.path)
    lines = read_tail(path, args.tail_lines)
    result = extract_error_block(lines, args.context_before, args.context_after)
    result["path"] = str(path)
    result["tail_lines"] = args.tail_lines
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
