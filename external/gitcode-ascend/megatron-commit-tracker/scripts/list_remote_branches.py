#!/usr/bin/env python3
"""
List remote Git branches for an official upstream repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="https://github.com/NVIDIA/Megatron-LM.git")
    parser.add_argument("--pattern", action="append")
    args = parser.parse_args()

    cmd = ["git", "ls-remote", "--heads", args.repo_url]
    if args.pattern:
        cmd.extend(args.pattern)
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    branches = []
    for line in result.stdout.splitlines():
        sha, ref = line.split("\t", 1)
        branches.append({"sha": sha, "ref": ref, "branch": ref.removeprefix("refs/heads/")})
    print(json.dumps({"repo_url": args.repo_url, "branches": branches}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
