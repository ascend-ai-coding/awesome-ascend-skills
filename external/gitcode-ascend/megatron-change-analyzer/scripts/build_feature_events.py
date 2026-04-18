#!/usr/bin/env python3
"""
Build implementation-oriented Megatron feature events from a normalized change-set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


NOISE_TITLE_PATTERNS = (
    r"^chore",
    r"copy-pr-bot",
    r"goldenvalues?",
    r"golden values?",
    r"\bbump\b",
    r"skip ci",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def is_noise_commit(title: str, files: list[str]) -> bool:
    lowered = title.lower()
    if any(re.search(pattern, lowered) for pattern in NOISE_TITLE_PATTERNS):
        return True
    if files and all(path in {"uv.lock", "pyproject.toml", "megatron/core/package_info.py"} or path.startswith(".github/") for path in files):
        return True
    return False


def infer_areas(files: list[str]) -> list[str]:
    areas: set[str] = set()
    for path in files:
        if "dist_checkpointing" in path or "checkpoint" in path:
            areas.add("checkpointing")
        if "training/" in path or path.startswith("megatron/training"):
            areas.add("training")
        if "config/" in path or "arguments.py" in path or "training_config.py" in path:
            areas.add("config")
        if "distributed" in path or "parallel" in path:
            areas.add("distributed")
        if "optimizer" in path:
            areas.add("optimizer")
        if path.startswith("tests/"):
            areas.add("tests")
    return sorted(areas) or ["misc"]


def clean_title(title: str) -> str:
    title = re.sub(r"\s*\(#\d+\)$", "", title).strip()
    title = re.sub(r"^[A-Z][a-z]+:\s*", "", title)
    return title


def build_units(files: list[str], title: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    config_files = [path for path in files if "arguments.py" in path or "training_config.py" in path or "/config/" in path]
    if config_files:
        units.append(
            {
                "name": "Expose or extend config surface",
                "kind": "config_surface",
                "upstream_files": config_files,
                "summary": f"Expose the feature via configuration, CLI, or training config for '{clean_title(title)}'.",
            }
        )

    runtime_files = [
        path
        for path in files
        if any(token in path for token in ["global_vars.py", "training.py", "async_utils.py", "serialization.py", "fully_parallel.py", "torch.py"])
    ]
    if runtime_files:
        units.append(
            {
                "name": "Port runtime behavior",
                "kind": "runtime_logic",
                "upstream_files": runtime_files,
                "summary": f"Recreate the runtime logic added or changed by '{clean_title(title)}'.",
            }
        )

    wrapper_files = [path for path in files if any(token in path for token in ["checkpointing.py", "serialization.py", "async_utils.py", "strategies/"])]
    if wrapper_files:
        units.append(
            {
                "name": "Update wrappers or adapter flow",
                "kind": "wrapper_logic",
                "upstream_files": wrapper_files,
                "summary": "Carry upstream wrapper, adapter, or orchestration changes into the local integration layer.",
            }
        )

    test_files = [path for path in files if path.startswith("tests/")]
    if test_files:
        units.append(
            {
                "name": "Refresh tests and examples",
                "kind": "tests_and_examples",
                "upstream_files": test_files,
                "summary": "Mirror the most relevant test or example updates so the downstream feature is easier to validate.",
            }
        )

    if not units:
        units.append(
            {
                "name": "Port primary implementation path",
                "kind": "runtime_logic",
                "upstream_files": files,
                "summary": f"Port the main logic implied by '{clean_title(title)}'.",
            }
        )

    return units


def migration_relevance(units: list[dict[str, Any]], files: list[str]) -> str:
    kinds = {unit["kind"] for unit in units}
    if {"runtime_logic", "wrapper_logic", "config_surface"} & kinds:
        return "high"
    if any("tests/" not in path for path in files):
        return "medium"
    return "low"


def breaking_risk(files: list[str]) -> str:
    if any("config" in path or "arguments.py" in path for path in files):
        return "medium"
    if any("checkpoint" in path or "distributed" in path for path in files):
        return "medium"
    return "low"


def build_events(change_set: dict[str, Any]) -> dict[str, Any]:
    commit_table = change_set.get("commit_table", [])
    events: list[dict[str, Any]] = []
    for commit in commit_table:
        title = str(commit.get("title", "")).strip()
        files = [str(path) for path in commit.get("touched_files", [])]
        if is_noise_commit(title, files):
            continue

        units = build_units(files, title)
        areas = infer_areas(files)
        notes = f"Feature event derived from commit '{clean_title(title)}' with emphasis on implementation-bearing files."
        events.append(
            {
                "title": clean_title(title),
                "kind": "new_feature" if "fix" not in title.lower() else "feature_fix",
                "commits": [commit["sha"]],
                "primary_commit": commit["sha"],
                "areas": areas,
                "breaking_risk": breaking_risk(files),
                "migration_relevance": migration_relevance(units, files),
                "notes": notes,
                "evidence": files[:8],
                "upstream_changed_files": files,
                "implementation_units": units,
                "porting_notes": [
                    "Use the upstream commit as the implementation reference before drafting MindSpeed changes.",
                    "Do not mark the feature as fully migrated unless each major implementation unit has a local target.",
                ],
            }
        )

    return {
        "repo": change_set.get("repo", "NVIDIA/Megatron-LM"),
        "branch": change_set.get("branch", "unknown"),
        "source_type": change_set.get("source_type", "unknown"),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--change-set", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    result = build_events(load_json(Path(args.change_set)))
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
