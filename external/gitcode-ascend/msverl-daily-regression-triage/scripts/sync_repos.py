#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
from typing import Any


REPOS = {
    "verl": {
        "url": "https://github.com/verl-project/verl.git",
        "branch": "main",
    },
    "mindspeed": {
        "url": "https://gitcode.com/Ascend/MindSpeed.git",
        "branch": "master",
    },
}


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def sync_repo(cache_root: pathlib.Path, name: str, config: dict[str, str]) -> dict[str, Any]:
    repo_dir = cache_root / name
    if not repo_dir.exists():
        run(["git", "clone", "--branch", config["branch"], "--single-branch", config["url"], str(repo_dir)])
        action = "cloned"
    else:
        run(["git", "fetch", "origin", config["branch"]], cwd=repo_dir)
        run(["git", "checkout", config["branch"]], cwd=repo_dir)
        run(["git", "reset", "--hard", f"origin/{config['branch']}"], cwd=repo_dir)
        action = "updated"

    head = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    return {
        "name": name,
        "path": str(repo_dir),
        "branch": config["branch"],
        "url": config["url"],
        "action": action,
        "head": head,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or refresh local repo caches for msverl triage.")
    parser.add_argument(
        "--cache-root",
        default="/tmp/msverl-skill-cache",
        help="Directory used for local repo caches.",
    )
    args = parser.parse_args()

    cache_root = pathlib.Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    results = [sync_repo(cache_root, name, config) for name, config in REPOS.items()]
    print(json.dumps({"cache_root": str(cache_root), "repos": results}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
