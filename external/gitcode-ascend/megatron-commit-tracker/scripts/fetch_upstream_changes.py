#!/usr/bin/env python3
"""
Fetch a branch-aware Megatron change-set from the official upstream repository.

This script keeps upstream collection deterministic and reusable for:
- PR tracking
- single-commit lookup
- commit ranges
- branch time windows
- scheduled incremental tracking
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/NVIDIA/Megatron-LM.git"
DEFAULT_CACHE_DIR = Path.home() / ".codex" / "skill-cache" / "megatron-lm.git"


def run_git(repo_dir: Path, *args: str) -> str:
    cmd = ["git", "--git-dir", str(repo_dir), *args]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def ensure_cache_repo(repo_dir: Path, repo_url: str) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        try:
            subprocess.run(["git", "init", "--bare", str(repo_dir)], check=True, text=True)
        except subprocess.CalledProcessError:
            if not repo_dir.exists():
                raise
        try:
            run_git(repo_dir, "remote", "add", "origin", repo_url)
        except subprocess.CalledProcessError:
            pass

    try:
        current = run_git(repo_dir, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        current = ""
    if current != repo_url:
        if current:
            run_git(repo_dir, "remote", "set-url", "origin", repo_url)
        else:
            run_git(repo_dir, "remote", "add", "origin", repo_url)


def fetch_branch(repo_dir: Path, branch: str) -> str:
    refspec = f"+refs/heads/{branch}:refs/remotes/origin/{branch}"
    subprocess.run(
        ["git", "--git-dir", str(repo_dir), "fetch", "--prune", "origin", refspec],
        check=True,
        text=True,
        capture_output=True,
    )
    return f"refs/remotes/origin/{branch}"


def fetch_pr(repo_dir: Path, pr_number: int) -> str:
    local_ref = f"refs/pull/{pr_number}/head"
    refspec = f"+refs/pull/{pr_number}/head:{local_ref}"
    subprocess.run(
        ["git", "--git-dir", str(repo_dir), "fetch", "--prune", "origin", refspec],
        check=True,
        text=True,
        capture_output=True,
    )
    return local_ref


def commit_exists(repo_dir: Path, rev: str) -> bool:
    result = subprocess.run(
        ["git", "--git-dir", str(repo_dir), "rev-parse", "--verify", "--quiet", rev],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def get_head_sha(repo_dir: Path, ref: str) -> str:
    return run_git(repo_dir, "rev-parse", ref)


def list_commits(repo_dir: Path, rev_args: list[str], limit: int) -> list[str]:
    args = ["rev-list", "--reverse", f"--max-count={limit}", *rev_args]
    output = run_git(repo_dir, *args)
    return [line for line in output.splitlines() if line]


def commit_metadata(repo_dir: Path, sha: str) -> dict[str, Any]:
    fmt = "%H%x1f%an%x1f%ae%x1f%aI%x1f%s"
    raw = run_git(repo_dir, "show", "-s", f"--format={fmt}", sha)
    commit_sha, author, email, authored_date, title = raw.split("\x1f", 4)
    files = run_git(repo_dir, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    touched_files = [line for line in files.splitlines() if line]
    return {
        "sha": commit_sha,
        "author": author,
        "author_email": email,
        "authored_date": authored_date,
        "title": title,
        "touched_files_count": len(touched_files),
        "touched_files": touched_files,
    }


def normalize_selector(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pr": args.pr,
        "commit": args.commit,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "since": args.since,
        "until": args.until,
    }


def build_change_set(args: argparse.Namespace, repo_dir: Path) -> dict[str, Any]:
    branch_ref = fetch_branch(repo_dir, args.branch)
    resolved_base_sha = args.base_sha
    resolved_head_sha = args.head_sha
    commits: list[str] = []

    if args.source_type == "pr":
        if args.pr is None:
            raise ValueError("--pr is required for source type 'pr'")
        pr_ref = fetch_pr(repo_dir, args.pr)
        resolved_head_sha = get_head_sha(repo_dir, pr_ref)
        commits = list_commits(repo_dir, [pr_ref], args.limit)
    elif args.source_type == "commit":
        if not args.commit:
            raise ValueError("--commit is required for source type 'commit'")
        if not commit_exists(repo_dir, args.commit):
            subprocess.run(
                ["git", "--git-dir", str(repo_dir), "fetch", "origin", args.commit],
                check=True,
                text=True,
                capture_output=True,
            )
        resolved_head_sha = get_head_sha(repo_dir, args.commit)
        commits = [resolved_head_sha]
    elif args.source_type == "range":
        if not args.base_sha or not args.head_sha:
            raise ValueError("--base-sha and --head-sha are required for source type 'range'")
        resolved_base_sha = get_head_sha(repo_dir, args.base_sha)
        resolved_head_sha = get_head_sha(repo_dir, args.head_sha)
        commits = list_commits(repo_dir, [f"{resolved_base_sha}..{resolved_head_sha}"], args.limit)
    elif args.source_type in {"time_window", "scheduled"}:
        rev_args = []
        if args.since:
            rev_args.append(f"--since={args.since}")
        if args.until:
            rev_args.append(f"--until={args.until}")
        rev_args.append(branch_ref)
        commits = list_commits(repo_dir, rev_args, args.limit)
        if commits:
            resolved_base_sha = commits[0]
            resolved_head_sha = commits[-1]
        else:
            resolved_head_sha = get_head_sha(repo_dir, branch_ref)
    else:
        raise ValueError(f"Unsupported source type: {args.source_type}")

    commit_objects = [commit_metadata(repo_dir, sha) for sha in commits]
    return {
        "repo": "NVIDIA/Megatron-LM",
        "repo_url": args.repo_url,
        "branch": args.branch,
        "source_type": args.source_type,
        "selector": normalize_selector(args),
        "resolved": {
            "commits": commits,
            "head_sha": resolved_head_sha,
            "base_sha": resolved_base_sha,
        },
        "analysis_mode": args.analysis_mode,
        "commit_table": commit_objects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--branch", required=True)
    parser.add_argument(
        "--source-type",
        required=True,
        choices=["pr", "commit", "range", "time_window", "scheduled"],
    )
    parser.add_argument("--analysis-mode", default="summary")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--commit")
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--since")
    parser.add_argument("--until")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(os.path.expanduser(args.cache_dir))
    ensure_cache_repo(repo_dir, args.repo_url)
    change_set = build_change_set(args, repo_dir)
    print(json.dumps(change_set, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
