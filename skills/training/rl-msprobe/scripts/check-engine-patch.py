#!/usr/bin/env python3
"""Verify msprobe dump hooks exist in verl engine transformer_impl.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_MARKERS = [
    "_ensure_debugger",
    "_log_update_actor_step",
    "DUMP_ON",
    "update_actor_log.jsonl",
    "PrecisionDebugger",
]

BACKEND_FILES = {
    "fsdp": "verl/workers/engine/fsdp/transformer_impl.py",
    "megatron": "verl/workers/engine/megatron/transformer_impl.py",
}


def check_file(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [f"missing file: {path}"]

    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [m for m in REQUIRED_MARKERS if m not in text]
    if missing:
        return False, [f"missing markers: {', '.join(missing)}"]

    # backend-specific hook points
    if "fsdp" in str(path):
        if "forward_backward_batch" not in text or "_debugger.start" not in text:
            return False, ["FSDP: expected dump hooks in forward_backward_batch"]
    if "megatron" in str(path):
        if "_finish_dump_step" not in text and "_dump_should_record" not in text:
            return False, [
                "Megatron: expected _finish_dump_step or _dump_should_record hooks"
            ]

    return True, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True)
    parser.add_argument("--backend", choices=["fsdp", "megatron"], required=True)
    args = parser.parse_args()

    rel = BACKEND_FILES[args.backend]
    target = Path(args.verl_root) / rel
    ok, errors = check_file(target)

    if ok:
        print(f"PASS: {target}")
        return 0

    print(f"FAIL: {target}")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
