#!/usr/bin/env python3
"""Verify inference dump flags in a verl training script."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def check_vllm(text: str) -> tuple[bool, list[str]]:
    missing = []
    if not re.search(r"enforce_eager\s*=\s*True|enforce_eager:\s*True", text, re.I):
        missing.append("actor_rollout_ref.rollout.enforce_eager=True (required for dump)")
    if not re.search(r"dump_config_path", text):
        missing.append("engine_kwargs.vllm.additional_config.dump_config_path")
    return len(missing) == 0, missing


def check_sglang(text: str) -> tuple[bool, list[str]]:
    missing = []
    if not re.search(r"enforce_eager\s*=\s*True|enforce_eager:\s*True", text, re.I):
        missing.append("actor_rollout_ref.rollout.enforce_eager=True (maps to disable_cuda_graph)")
    has_native = bool(re.search(r"msprobe_dump_config|msprobe-dump-config", text, re.I))
    # Old path: user may have patched sglang ModelRunner — cannot auto-detect in script
    if not has_native:
        missing.append(
            "engine_kwargs.sglang.msprobe_dump_config (SGLang>=0.5.11) "
            "OR confirm SGLang ModelRunner patch (SGLang<0.5.11)"
        )
    return len(missing) == 0, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    args = parser.parse_args()

    if not args.script.is_file():
        print(f"FAIL: missing {args.script}")
        return 1

    text = args.script.read_text(encoding="utf-8", errors="replace")
    ok, missing = check_vllm(text) if args.backend == "vllm" else check_sglang(text)

    if ok:
        print(f"PASS: {args.script} ({args.backend})")
        return 0

    print(f"FAIL: {args.script} ({args.backend})")
    for m in missing:
        print(f"  - {m}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
