#!/usr/bin/env python3
"""Verify training script has required PrecisionDebugger profiler flags."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED = [
    (r"global_profiler\.tool\s*=\s*precision_debugger|tool:\s*precision_debugger", "global_profiler.tool=precision_debugger"),
    (r"global_profiler\.steps\s*=|steps:\s*\[", "global_profiler.steps"),
    (r"precision_debugger\.config_path|config_path:", "precision_debugger.config_path"),
    (r"profiler\.enable\s*=\s*True|enable:\s*True", "role profiler.enable=True"),
]


def check(text: str) -> tuple[bool, list[str]]:
    missing = []
    for pattern, label in REQUIRED:
        if not re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            missing.append(label)
    if re.search(r'"dump_path"\s*:', text):
        missing.append("config.json should NOT set dump_path (verl controls output path)")
    return len(missing) == 0, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    args = parser.parse_args()

    if not args.script.is_file():
        print(f"FAIL: missing script {args.script}")
        return 1

    ok, missing = check(args.script.read_text(encoding="utf-8", errors="replace"))
    if ok:
        print(f"PASS: {args.script}")
        return 0

    print(f"FAIL: {args.script}")
    for m in missing:
        print(f"  - missing or invalid: {m}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
