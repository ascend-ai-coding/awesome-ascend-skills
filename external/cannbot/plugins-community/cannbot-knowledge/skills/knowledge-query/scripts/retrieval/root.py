# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Knowledge-base root resolution for knowledge-query.

Resolution is intentionally deterministic:
1. explicit --knowledge-root / --knowledge-roots
2. environment variables
3. the persisted cannbot env file
4. bounded structural discovery
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shlex
from typing import Iterable

from retrieval.errors import CliError

MIN_KNOWLEDGE_SCORE = 6
# Discovery source for cannbot installer dependency checkouts. Kept as one constant so
# root discovery (_is_discoverable) and the readiness gate (is_cannbot_dependency_root)
# share a single trust rule instead of two drifting heuristics.
DEPENDENCY_DISCOVERY_SOURCE = "discover:cannbot-dependency"
# Root selected in this process: {"path": str, "source": str}. Recorded when the
# resolution is built, and consulted by the readiness gate.
_SELECTED: dict[str, str | None] = {"path": None, "source": None}
ROOT_ENV_NAMES = (
    "CANNBOT_KNOWLEDGE_ROOT",
    "OKF_KNOWLEDGE_ROOT",
    "KNOWLEDGE_ROOT",
)
ROOTS_ENV_NAMES = (
    "CANNBOT_KNOWLEDGE_ROOTS",
    "OKF_KNOWLEDGE_ROOTS",
    "KNOWLEDGE_ROOTS",
)
SKIP_DISCOVERY_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "lost+found",
    "node_modules",
    "third_party",
}

CONTENT_MARKERS = ("reference/", "ops/", "runbooks/")
SPEC_MARKERS = ("SPEC-Retrieve.md", "SPEC-Graph.md", "SPEC-frontmatter.md")


@dataclass(frozen=True)
class Candidate:
    path: str
    source: str
    score: int
    markers: tuple[str, ...]


def default_env_file() -> Path:
    override = os.environ.get("CANNBOT_KNOWLEDGE_ENV")
    if override:
        return Path(override).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return config_home / "cannbot" / "knowledge.env"


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def score_knowledge_root(path: str | os.PathLike[str]) -> tuple[int, tuple[str, ...]]:
    root = Path(path).expanduser()
    if not _safe_is_dir(root):
        return 0, ()

    checks = (
        ("search/okf.index.json", 9, _safe_is_file(root / "search" / "okf.index.json")),
        ("search/", 1, _safe_is_dir(root / "search")),
        ("reference/", 5, _safe_is_dir(root / "reference")),
        ("ops/", 4, _safe_is_dir(root / "ops")),
        ("runbooks/", 4, _safe_is_dir(root / "runbooks")),
        ("SPEC-Retrieve.md", 4, _safe_is_file(root / "SPEC-Retrieve.md")),
        ("SPEC-Graph.md", 2, _safe_is_file(root / "SPEC-Graph.md")),
        ("SPEC-frontmatter.md", 2, _safe_is_file(root / "SPEC-frontmatter.md")),
        ("README.md", 1, _safe_is_file(root / "README.md")),
        ("index.md", 2, _safe_is_file(root / "index.md")),
    )
    score = 0
    markers: list[str] = []
    for marker, points, ok in checks:
        if ok:
            score += points
            markers.append(marker)
    return score, tuple(markers)


def has_strong_knowledge_markers(markers: Iterable[str]) -> bool:
    marker_set = set(markers)
    if "search/okf.index.json" in marker_set:
        return True
    content_count = sum(1 for marker in CONTENT_MARKERS if marker in marker_set)
    has_spec = any(marker in marker_set for marker in SPEC_MARKERS)
    has_index = "index.md" in marker_set
    return (has_spec and content_count >= 1) or (has_index and content_count >= 2)


def _is_discoverable(candidate: Candidate) -> bool:
    """Whether a structurally discovered candidate may be auto-selected.

    cannbot plugin dependency checkouts (<project>/.cannbot/dependencies/<plugin>/
    cannbot-knowledge/knowledge) are installed by the cannbot installer and are trusted
    with just an ops/ content marker, even before the retrieval index is built.
    """
    if is_probable_knowledge_root(candidate.path):
        return True
    return candidate.source == DEPENDENCY_DISCOVERY_SOURCE and "ops/" in candidate.markers


def is_probable_knowledge_root(path: str | os.PathLike[str]) -> bool:
    score, markers = score_knowledge_root(path)
    return score >= MIN_KNOWLEDGE_SCORE and has_strong_knowledge_markers(markers)


def _real_path(path: str | os.PathLike[str]) -> str | None:
    """Return the real path used for identity comparisons, or None when unusable."""
    try:
        return os.path.realpath(os.path.abspath(str(path)))
    except OSError:
        return None


def selected_dependency_root() -> str | None:
    """The root chosen in this process by the cannbot dependency probe, if any.

    Root discovery resolves every candidate path (``str(path.resolve())``), so a
    checkout reached through a symlink -- e.g. the knowledge repo cached once under
    ``~/.cache/cannbot/cannbot-knowledge`` and linked into several projects -- loses the
    ``dependencies`` / ``cannbot-knowledge`` path segments.  The discovery *source*
    survives that resolution, so it is the fact the trust decision must key on.
    """
    if _SELECTED.get("source") != DEPENDENCY_DISCOVERY_SOURCE:
        return None
    path = _SELECTED.get("path")
    return str(path) if path else None


def _is_selected_dependency_root(path: str | os.PathLike[str]) -> bool:
    selected = selected_dependency_root()
    if selected is None:
        return False
    return _real_path(selected) == _real_path(path)


def _is_probed_dependency_root(path: str | os.PathLike[str]) -> bool:
    """Re-run the discovery probe and compare by real path.

    Same fact source as ``discover_candidates``, and symlink-safe: the probe yields the
    in-project paths while this compares real paths, so a linked checkout still matches.
    """
    target = _real_path(path)
    if target is None:
        return False
    try:
        probed = _cannbot_dependency_paths(Path.cwd().resolve())
    except OSError:
        return False
    return any(_real_path(candidate) == target for candidate in probed)


def is_cannbot_dependency_root(path: str | os.PathLike[str]) -> bool:
    """Whether `path` is a cannbot-installer dependency checkout of the knowledge repo.

    The installer clones cannbot-knowledge into
    ``<project>/.cannbot/dependencies/<plugin>/cannbot-knowledge`` and the queryable
    root is its ``knowledge/`` subdirectory.  Such a checkout is installed by the
    cannbot installer, so it is trusted with just an ``ops/`` content marker even
    before the retrieval index is built -- the same trust rule ``_is_discoverable``
    applies when auto-selecting a root.

    Trust is keyed on the discovery fact, not on the path literal, because discovery
    resolves candidates and a symlinked checkout no longer contains those segments.
    """
    expanded = Path(path).expanduser()
    if not _safe_is_dir(expanded / "ops"):
        return False
    if _is_selected_dependency_root(expanded) or _is_probed_dependency_root(expanded):
        return True
    # Fallback for an explicitly configured root (CANNBOT_KNOWLEDGE_ROOT /
    # --knowledge-root) whose layout did not go through the probe.  Deliberately uses
    # the given path instead of Path.resolve() so the segments are still visible.
    # The `.cannbot` segment is required: without it, any directory named
    # ``.../dependencies/<x>/cannbot-knowledge/knowledge`` would be trusted and skip the
    # MIN_KNOWLEDGE_SCORE gate, which is wider than what the installer produces.
    literal = Path(os.path.abspath(str(expanded))).parts
    return (
        expanded.name == "knowledge"
        and ".cannbot" in literal
        and "dependencies" in literal
        and "cannbot-knowledge" in literal
    )


def is_trusted_knowledge_root(path: str | os.PathLike[str]) -> bool:
    """Whether `path` may be auto-built without requiring a pre-built index.

    Covers first-class knowledge repositories (strong content markers) and
    cannbot installer dependency checkouts (trusted by install provenance).
    """
    return is_probable_knowledge_root(path) or is_cannbot_dependency_root(path)


def consume_knowledge_root_args(argv: list[str]) -> tuple[list[str], list[str]]:
    roots: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--knowledge-root", "--knowledge-roots"):
            if i + 1 >= len(argv):
                raise CliError(f"{arg} requires a path")
            roots.extend(split_roots(argv[i + 1]))
            i += 2
            continue
        if arg.startswith("--knowledge-root="):
            roots.extend(split_roots(arg.split("=", 1)[1]))
            i += 1
            continue
        if arg.startswith("--knowledge-roots="):
            roots.extend(split_roots(arg.split("=", 1)[1]))
            i += 1
            continue
        rest.append(arg)
        i += 1
    return roots, rest


def split_roots(value: str | None) -> list[str]:
    if not value:
        return []
    roots: list[str] = []
    for chunk in value.split(os.pathsep):
        for part in chunk.split(","):
            part = part.strip()
            if part:
                roots.append(part)
    return roots


def _normalize_path(raw: str, base: Path | None = None) -> str:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = base / path
    return str(path.resolve())


def _candidate(raw: str, source: str, base: Path | None = None) -> Candidate:
    path = _normalize_path(raw, base)
    score, markers = score_knowledge_root(path)
    return Candidate(path=path, source=source, score=score, markers=markers)


def _env_file_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            values[key] = value
    return values


def _roots_from_env_map(values: dict[str, str]) -> list[str]:
    roots: list[str] = []
    for name in ROOT_ENV_NAMES:
        roots.extend(split_roots(values.get(name)))
    for name in ROOTS_ENV_NAMES:
        roots.extend(split_roots(values.get(name)))
    return roots


def _dedupe(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for cand in candidates:
        key = (cand.path, cand.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _choose_configured(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    # 显式配置（--knowledge-root / CANNBOT_KNOWLEDGE_ROOT）也要接受 cannbot 安装器的
    # 依赖检出布局，否则用户按文档指向已安装的知识库副本时会因“没有索引”被判不合法。
    valid = [c for c in candidates if is_trusted_knowledge_root(c.path)]
    return valid[0] if valid else None


def _walk_dirs(base: Path, max_depth: int = 2, max_dirs: int = 1500) -> Iterable[Path]:
    stack: list[tuple[Path, int]] = [(base, 0)]
    yielded = 0
    while stack and yielded < max_dirs:
        cur, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            if child.name in SKIP_DISCOVERY_DIRS or not _safe_is_dir(child):
                continue
            yielded += 1
            yield child
            stack.append((child, depth + 1))
            if yielded >= max_dirs:
                break


def _cannbot_dependency_paths(start_path: Path) -> list[Path]:
    """Probe cannbot plugin dependency checkouts installed by the cannbot installer.

    Plugins declaring cannbot-knowledge in plugin-install.json get a shallow clone at
    <project>/.cannbot/dependencies/<plugin>/cannbot-knowledge; the knowledge root is its
    knowledge/ subdirectory. Probe ancestors of cwd so sessions started anywhere inside
    the project still find it.
    """
    found: list[Path] = []
    for base in [start_path, *list(start_path.parents)[:6]]:
        dep_root = base / ".cannbot" / "dependencies"
        if not _safe_is_dir(dep_root):
            continue
        try:
            plugins = sorted(dep_root.iterdir())
        except OSError:
            continue
        for plugin_dir in plugins:
            candidate = plugin_dir / "cannbot-knowledge" / "knowledge"
            if _safe_is_dir(candidate / "ops"):
                found.append(candidate)
    return found


def discover_candidates(start: str | os.PathLike[str] | None = None) -> list[Candidate]:
    start_path = Path(start or os.getcwd()).resolve()
    cands: list[Candidate] = []
    seen_paths: set[str] = set()

    ancestor_paths = [start_path, *list(start_path.parents)[:6]]
    child_paths = list(_walk_dirs(start_path))
    for source, paths in (("discover:ancestor", ancestor_paths), ("discover:child", child_paths),
                          ("discover:cannbot-dependency", _cannbot_dependency_paths(start_path))):
        for path in paths:
            path_str = str(path.resolve())
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            score, markers = score_knowledge_root(path_str)
            if score <= 0:
                continue
            cands.append(Candidate(path=path_str, source=source, score=score, markers=markers))
    return sorted(
        cands,
        key=lambda c: (
            0 if _is_discoverable(c) else 1,
            0 if c.source == "discover:ancestor" else 1,
            -c.score,
            c.path,
        ),
    )


def _configured_tiers(cli_roots, env_file):
    tiers = []
    if cli_roots:
        tiers.append(("cli", cli_roots, Path.cwd()))
    for name in ROOT_ENV_NAMES:
        roots = split_roots(os.environ.get(name))
        if roots:
            tiers.append((f"env:{name}", roots, Path.cwd()))
    for name in ROOTS_ENV_NAMES:
        roots = split_roots(os.environ.get(name))
        if roots:
            tiers.append((f"env:{name}", roots, Path.cwd()))
    env_file_roots = _roots_from_env_map(_env_file_values(env_file))
    if env_file_roots:
        tiers.append((f"env-file:{env_file}", env_file_roots, env_file.parent))
    return tiers


def _resolution(selected, candidates, env_file, configured, error=None):
    if isinstance(selected, Candidate):
        _SELECTED["path"] = selected.path
        _SELECTED["source"] = selected.source
    result = {
        "selected": selected,
        "candidates": _dedupe(candidates),
        "env_file": str(env_file),
        "configured": configured,
    }
    if error is not None:
        result["error"] = error
    return result


def resolve_knowledge_roots(
    cli_roots: list[str] | None = None,
    start: str | os.PathLike[str] | None = None,
    include_discovery: bool = True,
) -> dict[str, object]:
    all_candidates: list[Candidate] = []
    env_file = default_env_file()
    for source, roots, base in _configured_tiers(cli_roots, env_file):
        tier_candidates = [_candidate(root, source, base) for root in roots]
        all_candidates.extend(tier_candidates)
        selected = _choose_configured(tier_candidates)
        if selected is not None:
            return _resolution(selected, all_candidates, env_file, True)

    if all_candidates:
        error = "configured knowledge base root candidates do not look like cannbot knowledge-base roots"
        return _resolution(None, all_candidates, env_file, True, error)

    if include_discovery:
        discovered = discover_candidates(start)
        all_candidates.extend(discovered)
        valid = [c for c in discovered if _is_discoverable(c)]
        selected = valid[0] if valid else None
        return _resolution(selected, all_candidates, env_file, False)

    return _resolution(None, all_candidates, env_file, False)


def configure_environment_from_argv(argv: list[str]) -> tuple[list[str], dict[str, object]]:
    cli_roots, rest = consume_knowledge_root_args(argv[1:])
    resolution = resolve_knowledge_roots(cli_roots=cli_roots)
    selected = resolution.get("selected")
    if isinstance(selected, Candidate):
        os.environ["CANNBOT_KNOWLEDGE_ROOT"] = selected.path
    elif resolution.get("configured"):
        candidates = resolution.get("candidates") or []
        first = candidates[0] if candidates and isinstance(candidates[0], Candidate) else None
        if first is not None:
            os.environ["CANNBOT_KNOWLEDGE_ROOT"] = first.path
        if resolution.get("error"):
            os.environ["CANNBOT_KNOWLEDGE_RESOLUTION_ERROR"] = str(resolution["error"])
    return [argv[0], *rest], resolution


def resolution_to_dict(resolution: dict[str, object]) -> dict[str, object]:
    selected = resolution.get("selected")
    candidates = resolution.get("candidates") or []
    return {
        "selected": asdict(selected) if isinstance(selected, Candidate) else None,
        "env_file": resolution.get("env_file"),
        "configured": resolution.get("configured", False),
        "error": resolution.get("error"),
        "candidates": [asdict(c) for c in candidates if isinstance(c, Candidate)],
    }
