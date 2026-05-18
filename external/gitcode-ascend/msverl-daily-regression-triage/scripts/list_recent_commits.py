#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
from typing import Any


def run(cmd: list[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def resolve_branch(repo_path: pathlib.Path, branch: str) -> str:
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=str(repo_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if probe.returncode == 0:
        return branch
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path).strip()


def list_recent_commits(repo_path: pathlib.Path, branch: str, since: str, until: str, limit: int) -> list[dict[str, Any]]:
    branch = resolve_branch(repo_path, branch)
    pretty = "%H%x1f%ad%x1f%s%x1f%an"
    log_cmd = [
        "git",
        "log",
        branch,
        f"--since={since}",
        f"--until={until}",
        f"--max-count={limit}",
        "--date=iso-strict",
        f"--pretty=format:{pretty}",
        "--name-only",
    ]
    output = run(log_cmd, repo_path)
    commits: list[dict[str, Any]] = []
    current = None

    for raw_line in output.splitlines():
        if "\x1f" in raw_line:
            if current:
                current["files"] = sorted(set(current["files"]))
                commits.append(current)
            sha, authored_at, subject, author = raw_line.split("\x1f", 3)
            current = {
                "sha": sha,
                "authored_at": authored_at,
                "subject": subject,
                "author": author,
                "files": [],
            }
            continue
        if current and raw_line.strip():
            current["files"].append(raw_line.strip())

    if current:
        current["files"] = sorted(set(current["files"]))
        commits.append(current)
    return commits


def main() -> None:
    parser = argparse.ArgumentParser(description="List recent commits with touched files for a repo and time window.")
    parser.add_argument("repo_path", help="Local git repository path.")
    parser.add_argument("--branch", required=True, help="Branch name to inspect.")
    parser.add_argument("--since", required=True, help="Start of time window, for example 2026-03-29 00:00:00 +0800.")
    parser.add_argument("--until", required=True, help="End of time window.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum commit count to inspect.")
    args = parser.parse_args()

    result = {
        "repo_path": str(pathlib.Path(args.repo_path)),
        "branch": args.branch,
        "since": args.since,
        "until": args.until,
        "commits": list_recent_commits(pathlib.Path(args.repo_path), args.branch, args.since, args.until, args.limit),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
