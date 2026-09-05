#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Canonical target-repository paths for the Issue Handler runtime.

All paths are relative to the target repository root.  Compatibility helpers
only fall back to the former repository-root inputs when the caller is using a
canonical default and the new input does not exist.  They never move, rewrite,
or delete a user's existing file.
"""

from __future__ import annotations

from pathlib import Path

RUNTIME_ROOT = Path(".cannbot") / "gitcode-issue-handler"
CONFIG_DIR = RUNTIME_ROOT / "config"
DATA_DIR = RUNTIME_ROOT / "data"
CACHE_DIR = RUNTIME_ROOT / "cache"
REPORTS_DIR = RUNTIME_ROOT / "reports"
LOGS_DIR = RUNTIME_ROOT / "logs"
REPRO_DIR = RUNTIME_ROOT / "repro"
IMAGES_DIR = RUNTIME_ROOT / "images"
TMP_DIR = RUNTIME_ROOT / "tmp"
WORKTREES_DIR = RUNTIME_ROOT / "worktrees"

CLASSIFY_CONFIG = CONFIG_DIR / "classify_config.yaml"
OPERATOR_OWNERS_CONFIG = CONFIG_DIR / "operator_owners.yaml"
ISSUES_DATA = DATA_DIR / "issues.json"
GROUPS_DATA = DATA_DIR / "groups.json"
LAST_CHECK_STATE = DATA_DIR / "last_check.json"
FOLLOWUP_WATCH_STATE = DATA_DIR / "followup-watch.json"
CLASSIFICATION_REPORT = REPORTS_DIR / "classification.txt"
FETCH_CACHE = CACHE_DIR / "issues"
GITCODE_RATE_LIMIT = CACHE_DIR / "gitcode-rate-limit"
KNOWLEDGE_CACHE = CACHE_DIR / "knowledge-comments"
KNOWLEDGE_CORPUS = DATA_DIR / "issue-history.json"
KNOWLEDGE_STATE = DATA_DIR / "issue-knowledge-state.json"
KNOWLEDGE_LOCK = CACHE_DIR / "issue-knowledge-refresh.lock"
KNOWLEDGE_REPORT = REPORTS_DIR / "knowledge-corpus.md"
LATEST_REPORT = REPORTS_DIR / "latest.md"

LEGACY_CLASSIFY_CONFIG = Path("classify_config.yaml")
LEGACY_OPERATOR_OWNERS_CONFIG = Path("operator_owners.yaml")
LEGACY_LAST_CHECK_STATE = Path("issue_analysis_data/last_check.json")
LEGACY_CLASSIFICATION_REPORT = Path("issue_analysis_data/classify_report.txt")
LEGACY_FETCH_CACHE = Path("issue_analysis_data/cache")


def path_text(path: Path) -> str:
    """Return a stable POSIX spelling for CLI defaults and documentation."""
    return path.as_posix()


def rate_limit_path(cache_dir: str | Path | None = None) -> Path:
    """Place shared transport state beside a configured handler cache."""
    if not cache_dir:
        return GITCODE_RATE_LIMIT
    return Path(cache_dir).parent / GITCODE_RATE_LIMIT.name


def compatible_read_path(
    requested: str | Path,
    *,
    canonical: Path,
    legacy: Path,
) -> Path:
    """Resolve a canonical default to a legacy input as a read-only fallback.

    Explicit non-default paths never fall back.  A canonical file always wins
    when both files exist, so an old root-level file cannot shadow new config.
    """
    requested_path = Path(requested)
    if (
        requested_path == canonical
        and not canonical.exists()
        and legacy.exists()
    ):
        return legacy
    return requested_path


def migrate_legacy_runtime_defaults(config: dict, source: Path) -> dict:
    """Redirect known former defaults without editing their source config.

    This also covers users who copied an old root config into the canonical
    config directory.  Only exact former defaults in those two conventional
    locations change; an explicitly selected custom config remains fully
    authoritative.
    """
    if source not in {LEGACY_CLASSIFY_CONFIG, CLASSIFY_CONFIG}:
        return config
    replacements = {
        "last_check_file": (LEGACY_LAST_CHECK_STATE, LAST_CHECK_STATE),
        "report_file": (LEGACY_CLASSIFICATION_REPORT, CLASSIFICATION_REPORT),
        "cache_dir": (LEGACY_FETCH_CACHE, FETCH_CACHE),
    }
    for key, (legacy, canonical) in replacements.items():
        if config.get(key) == path_text(legacy):
            config[key] = path_text(canonical)
    return config
