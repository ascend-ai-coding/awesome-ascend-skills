#!/usr/bin/env python3
"""
Search the official MindSpeed repository on a specific branch for likely
adaptation points.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://gitcode.com/Ascend/MindSpeed.git"
DEFAULT_CACHE_ROOT = Path.home() / ".codex" / "skill-cache" / "mindspeed"


def sanitize_branch(branch: str) -> str:
    return branch.replace("/", "__")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def ensure_checkout(repo_url: str, branch: str, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_root / sanitize_branch(branch)
    if not repo_dir.exists():
        run(["git", "clone", "--branch", branch, "--single-branch", repo_url, str(repo_dir)])
        return repo_dir

    run(["git", "fetch", "origin", branch], cwd=repo_dir)
    run(["git", "checkout", branch], cwd=repo_dir)
    run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir)
    return repo_dir


def search_with_rg(repo_dir: Path, terms: list[str], limit: int) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {}
    for term in terms:
        completed = subprocess.run(
            [
                "rg",
                "--line-number",
                "--no-heading",
                "--smart-case",
                "--max-count",
                str(limit),
                term,
                ".",
            ],
            cwd=repo_dir,
            text=True,
            capture_output=True,
        )
        if completed.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                completed.returncode, completed.args, completed.stdout, completed.stderr
            )
        rows: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            path, line_no, text = line.split(":", 2)
            rows.append({"path": path, "line": int(line_no), "text": text})
            if len(rows) >= limit:
                break
        matches[term] = rows
    return matches


def search_without_rg(repo_dir: Path, terms: list[str], limit: int) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {}
    for term in terms:
        rows: list[dict[str, Any]] = []
        lowered = term.lower()
        for path in repo_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                content = path.read_text(errors="ignore").splitlines()
            except Exception:
                continue
            for idx, line in enumerate(content, start=1):
                if lowered in line.lower():
                    rows.append(
                        {
                            "path": str(path.relative_to(repo_dir)),
                            "line": idx,
                            "text": line,
                        }
                    )
                    if len(rows) >= limit:
                        break
            if len(rows) >= limit:
                break
        matches[term] = rows
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--term", action="append", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    repo_dir = ensure_checkout(
        repo_url=args.repo_url,
        branch=args.branch,
        cache_root=Path(os.path.expanduser(args.cache_root)),
    )
    if shutil.which("rg"):
        matches = search_with_rg(repo_dir, args.term, args.limit)
    else:
        matches = search_without_rg(repo_dir, args.term, args.limit)

    print(
        json.dumps(
            {
                "repo": "Ascend/MindSpeed",
                "repo_url": args.repo_url,
                "branch": args.branch,
                "checkout": str(repo_dir),
                "matches": matches,
            },
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
