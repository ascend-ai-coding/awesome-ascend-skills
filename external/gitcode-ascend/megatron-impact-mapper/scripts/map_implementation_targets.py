#!/usr/bin/env python3
"""
Map Megatron implementation units onto likely MindSpeed implementation targets.
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

HINTS_BY_KIND = {
    "config_surface": [
        "mindspeed/arguments.py",
        "tests_extend/system_tests/yaml_args_example/example.yaml",
    ],
    "runtime_logic": [
        "mindspeed/training.py",
        "mindspeed/core/training.py",
    ],
    "wrapper_logic": [
        "mindspeed/checkpointing.py",
        "mindspeed/core/dist_checkpointing/checkpoint_adaptor.py",
        "mindspeed/features_manager/ckpt_acceleration/ckpt_acceleration.py",
        "mindspeed/core/megatron_basic/megatron_basic.py",
    ],
    "tests_and_examples": [
        "tests_extend/system_tests/yaml_args_example/example.yaml",
        "tests_extend/unit_tests",
        "tests_extend/system_tests",
    ],
}


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def search_with_rg(repo_dir: Path, terms: list[str], limit: int) -> list[str]:
    results: list[str] = []
    for term in terms:
        completed = subprocess.run(
            ["rg", "--files-with-matches", "--smart-case", term, "."],
            cwd=repo_dir,
            text=True,
            capture_output=True,
        )
        if completed.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                completed.returncode, completed.args, completed.stdout, completed.stderr
            )
        for path in completed.stdout.splitlines():
            path = normalize_repo_path(path)
            if not is_local_impl_path(path):
                continue
            if path not in results:
                results.append(path)
            if len(results) >= limit:
                return results
    return results


def search_without_rg(repo_dir: Path, terms: list[str], limit: int) -> list[str]:
    results: list[str] = []
    lowered_terms = [term.lower() for term in terms if term]
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_dir))
        rel = normalize_repo_path(rel)
        if not is_local_impl_path(rel):
            continue
        if any(term in rel.lower() for term in lowered_terms):
            if rel not in results:
                results.append(rel)
            if len(results) >= limit:
                return results
    return results


def repo_search(repo_dir: Path, terms: list[str], limit: int) -> list[str]:
    if shutil.which("rg"):
        return search_with_rg(repo_dir, terms, limit)
    return search_without_rg(repo_dir, terms, limit)


def normalize_repo_path(path: str) -> str:
    while path.startswith("./"):
        path = path[2:]
    return path


def is_local_impl_path(path: str) -> bool:
    path = normalize_repo_path(path)
    return path.startswith("mindspeed/") or path.startswith("tests_extend/")


def path_exists(repo_dir: Path, rel_path: str) -> bool:
    return (repo_dir / normalize_repo_path(rel_path)).exists()


def title_terms(title: str) -> list[str]:
    tokens = []
    for raw in title.replace("/", " ").replace("-", " ").split():
        clean = raw.strip().lower()
        if len(clean) >= 4:
            tokens.append(clean)
    return tokens[:6]


def derive_terms(event: dict[str, Any], unit: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for path in unit.get("upstream_files", []):
        base = Path(path).name
        stem = Path(base).stem
        if stem and stem not in terms:
            terms.append(stem)
        for token in stem.replace("_", " ").split():
            if len(token) >= 4 and token not in terms:
                terms.append(token)

    for token in title_terms(event.get("title", "")):
        if token not in terms:
            terms.append(token)

    if unit.get("kind") == "config_surface":
        terms.extend(["argument", "config", "yaml"])
    elif unit.get("kind") == "runtime_logic":
        terms.extend(["training", "signal", "shutdown"])
    elif unit.get("kind") == "wrapper_logic":
        terms.extend(["checkpoint", "dist_checkpointing", "wrapper", "preload", "async"])
    elif unit.get("kind") == "tests_and_examples":
        terms.extend(["tests_extend", "system_tests", "unit_tests"])

    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def build_targets(repo_dir: Path, event: dict[str, Any], units: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for unit in units:
        hints = [path for path in HINTS_BY_KIND.get(unit.get("kind", ""), []) if path_exists(repo_dir, path)]
        matches = repo_search(repo_dir, derive_terms(event, unit), limit)
        candidates: list[str] = []
        for path in hints + matches:
            if path not in candidates:
                candidates.append(path)
        targets.append(
            {
                "name": unit.get("name", "Untitled implementation target"),
                "source_unit": unit.get("name", "unknown"),
                "candidate_paths": candidates[:limit],
                "required_change": unit.get("summary", "Port the upstream implementation unit locally."),
                "confidence": 0.85 if hints else (0.7 if candidates else 0.45),
            }
        )
    return targets


def item_status(event: dict[str, Any], targets: list[dict[str, Any]]) -> str:
    if not targets:
        return "report_only_candidate"
    avg = sum(float(target.get("confidence", 0)) for target in targets) / len(targets)
    if avg >= 0.8 and event.get("migration_relevance") == "high":
        return "high_priority_candidate"
    if avg >= 0.65:
        return "partial_coverage_candidate"
    return "watchlist_candidate"


def build_impact_report(events_payload: dict[str, Any], repo_dir: Path, mindspeed_branch: str, megatron_base_branch: str, megatron_target_branch: str, mode: str, limit: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for event in events_payload.get("events", []):
        if event.get("migration_relevance") == "low":
            continue
        units = [unit for unit in event.get("implementation_units", []) if isinstance(unit, dict)]
        targets = build_targets(repo_dir, event, units, limit)
        candidate_paths: list[str] = []
        for target in targets:
            for path in target.get("candidate_paths", []):
                if path not in candidate_paths:
                    candidate_paths.append(path)
        items.append(
            {
                "event_title": event.get("title", "Untitled event"),
                "status": item_status(event, targets),
                "confidence": round(sum(float(target.get("confidence", 0)) for target in targets) / max(len(targets), 1), 2),
                "reason": event.get("notes", "No rationale provided."),
                "primary_commit": event.get("primary_commit"),
                "commits": event.get("commits", []),
                "upstream_changed_files": event.get("upstream_changed_files", []),
                "candidate_paths": candidate_paths,
                "implementation_units": units,
                "implementation_targets": targets,
                "covered_scope": [
                    f"已为上游实现单元 `{target.get('source_unit', 'unknown')}` 识别本地实现目标"
                    for target in targets
                    if target.get("candidate_paths")
                ],
                "omitted_scope": [
                    f"上游实现单元 `{unit.get('name', 'unknown')}` 还需要进一步细化本地改动方案"
                    for unit in units
                    if not any(target.get("source_unit") == unit.get("name") and target.get("candidate_paths") for target in targets)
                ],
                "manual_followups": [
                    "逐个对照上游 commit diff，把每个 implementation unit 展开成多文件本地补丁。",
                    "不要只修改首个 candidate path；优先补 config、runtime、wrapper、tests 四类实现单元。",
                ],
                "upstream_evidence": event.get("evidence", []),
            }
        )

    return {
        "mindspeed_branch": mindspeed_branch,
        "megatron_base_branch": megatron_base_branch,
        "megatron_target_branch": megatron_target_branch,
        "mode": mode,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--mindspeed-branch", required=True)
    parser.add_argument("--megatron-base-branch", required=True)
    parser.add_argument("--megatron-target-branch", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()

    repo_dir = ensure_checkout(
        repo_url=args.repo_url,
        branch=args.mindspeed_branch,
        cache_root=Path(os.path.expanduser(args.cache_root)),
    )
    report = build_impact_report(
        events_payload=load_json(Path(args.events)),
        repo_dir=repo_dir,
        mindspeed_branch=args.mindspeed_branch,
        megatron_base_branch=args.megatron_base_branch,
        megatron_target_branch=args.megatron_target_branch,
        mode=args.mode,
        limit=args.limit,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
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
