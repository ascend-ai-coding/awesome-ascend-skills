#!/usr/bin/env python3
"""
Resolve the alignment status between a MindSpeed branch and Megatron branches.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


KNOWN_MAPPINGS = {
    "master": "core_v0.12.1",
    "2.3.0_core_r0.12.1": "core_v0.12.1",
    "dev": "dev",
    "core_r0.15.3": "core_v0.15.3",
}


def resolve_alignment(
    mindspeed_branch: str,
    megatron_target_branch: str | None,
    megatron_base_branch: str | None,
) -> dict[str, Any]:
    known_base = KNOWN_MAPPINGS.get(mindspeed_branch)
    resolved_base = megatron_base_branch or known_base
    target = megatron_target_branch

    if not resolved_base:
        return {
            "mode": "unresolved",
            "confidence": 0.0,
            "mindspeed_branch": mindspeed_branch,
            "megatron_base_branch": None,
            "megatron_target_branch": target,
            "reason": "No explicit or known Megatron base branch mapping for this MindSpeed branch.",
            "known_mapping": False,
        }

    if target is None:
        return {
            "mode": "strict",
            "confidence": 0.9 if known_base else 0.75,
            "mindspeed_branch": mindspeed_branch,
            "megatron_base_branch": resolved_base,
            "megatron_target_branch": resolved_base,
            "reason": "Using the resolved Megatron base branch as both baseline and target.",
            "known_mapping": bool(known_base),
        }

    if target == "main":
        return {
            "mode": "exploration",
            "confidence": 0.6,
            "mindspeed_branch": mindspeed_branch,
            "megatron_base_branch": resolved_base,
            "megatron_target_branch": target,
            "reason": "Megatron main has no confirmed strict MindSpeed mapping and should be treated as exploratory.",
            "known_mapping": bool(known_base),
        }

    if target == resolved_base:
        return {
            "mode": "strict",
            "confidence": 0.95 if known_base else 0.8,
            "mindspeed_branch": mindspeed_branch,
            "megatron_base_branch": resolved_base,
            "megatron_target_branch": target,
            "reason": "MindSpeed branch is aligned directly with the specified Megatron branch.",
            "known_mapping": bool(known_base),
        }

    return {
        "mode": "strict",
        "confidence": 0.85 if known_base else 0.7,
        "mindspeed_branch": mindspeed_branch,
        "megatron_base_branch": resolved_base,
        "megatron_target_branch": target,
        "reason": "Using a confirmed or explicit baseline branch and a different target branch for migration planning.",
        "known_mapping": bool(known_base),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindspeed-branch", required=True)
    parser.add_argument("--megatron-target-branch")
    parser.add_argument("--megatron-base-branch")
    args = parser.parse_args()

    print(
        json.dumps(
            resolve_alignment(
                mindspeed_branch=args.mindspeed_branch,
                megatron_target_branch=args.megatron_target_branch,
                megatron_base_branch=args.megatron_base_branch,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
