#!/usr/bin/env python3
"""
Normalize a Megatron tracking request into a simple change-set JSON document.

This script does not call remote APIs. It exists to keep request parsing and
artifact shape deterministic inside the skill.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def normalize_request(args: argparse.Namespace) -> dict[str, Any]:
    selector = {
        "pr": args.pr,
        "commit": args.commit,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "since": args.since,
        "until": args.until,
    }
    return {
        "repo": "NVIDIA/Megatron-LM",
        "branch": args.branch,
        "source_type": args.source_type,
        "selector": selector,
        "resolved": {
            "commits": [],
            "head_sha": args.head_sha,
            "base_sha": args.base_sha,
        },
        "analysis_mode": args.analysis_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument(
        "--source-type",
        required=True,
        choices=["pr", "commit", "range", "time_window", "scheduled"],
    )
    parser.add_argument("--analysis-mode", default="summary")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--commit")
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--since")
    parser.add_argument("--until")
    args = parser.parse_args()

    print(json.dumps(normalize_request(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
