#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Plan conflict-free Issue groups and manage their temporary Git worktrees.

The manifest is the ownership boundary for cleanup. A worktree can only be
removed when it was recorded by this script, is in a terminal lifecycle state,
and has a clean Git status. Local branches are intentionally retained.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, NamedTuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from cli_output import write_stderr, write_stdout  # noqa: E402

MANIFEST_VERSION = 1
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TERMINAL_CLEANUP_STATES = {"published", "no_changes", "cancelled_clean"}
LIFECYCLE_STATES = TERMINAL_CLEANUP_STATES | {"active", "blocked", "cleaned"}
GIT_EXECUTABLE = shutil.which("git")


class WorktreeError(RuntimeError):
    """A safe worktree operation could not be completed."""


def _run(
    *args: str,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise WorktreeError(f"{' '.join(args)}: {detail}")
    return result


def _git(*args: str, cwd: Path | None = None, check: bool = True):
    """Run the resolved Git executable instead of relying on PATH at launch."""
    if GIT_EXECUTABLE is None:
        raise WorktreeError("git executable was not found")
    return _run(GIT_EXECUTABLE, *args, cwd=cwd, check=check)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise WorktreeError(f"{label} must match {SAFE_ID_RE.pattern}: {value!r}")
    return value


def _resolve_repo(path: str) -> Path:
    root = Path(path).resolve()
    top = _git("-C", str(root), "rev-parse", "--show-toplevel").stdout.strip()
    resolved_top = Path(top).resolve()
    if resolved_top != root:
        raise WorktreeError(
            f"repo root must be the Git top-level directory: {resolved_top}"
        )
    return root


def _git_common_dir(repo_root: Path) -> Path:
    raw = _git("-C", str(repo_root), "rev-parse", "--git-common-dir").stdout.strip()
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate_worktree_root(worktree_root: Path, repo_root: Path) -> None:
    if worktree_root == Path("/"):
        raise WorktreeError("worktree root must not be the filesystem root")
    if (
        worktree_root == repo_root
        or worktree_root in repo_root.parents
        or repo_root in worktree_root.parents
    ):
        raise WorktreeError(
            "worktree root must be outside and not an ancestor of the target repository"
        )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _manifest_lock(manifest_path: Path) -> Iterator[None]:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorktreeError(f"manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorktreeError(f"manifest is invalid JSON: {path}: {exc}") from exc
    if payload.get("version") != MANIFEST_VERSION or not isinstance(
        payload.get("groups"), dict
    ):
        raise WorktreeError(f"unsupported or invalid manifest: {path}")
    return payload


def _normalize_path(raw: str) -> str:
    value = raw.strip().replace("\\", "/")
    if not value or value.startswith("/"):
        raise WorktreeError(
            f"planned path must be a non-empty repository-relative path: {raw!r}"
        )
    parts = [part for part in value.split("/") if part not in ("", ".")]
    if not parts or ".." in parts or parts[0] == ".git":
        raise WorktreeError(f"unsafe planned path: {raw!r}")
    return "/".join(parts)


def _path_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _group_conflicts(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    left_paths = left["planned_paths"]
    right_paths = right["planned_paths"]
    if not left_paths or not right_paths:
        reasons.append("unknown_path_scope")
    else:
        for left_path in left_paths:
            overlaps = [
                right_path
                for right_path in right_paths
                if _path_overlap(left_path, right_path)
            ]
            reasons.extend(
                f"path:{left_path}<->{right_path}" for right_path in overlaps
            )
    shared_resources = sorted(
        set(left["exclusive_resources"]) & set(right["exclusive_resources"])
    )
    reasons.extend(f"resource:{resource}" for resource in shared_resources)
    return sorted(set(reasons))


def _read_groups(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorktreeError(f"cannot read groups JSON {path}: {exc}") from exc
    raw_groups = payload.get("groups") if isinstance(payload, dict) else payload
    if not isinstance(raw_groups, list):
        raise WorktreeError(
            "groups JSON must be an array or an object containing a groups array"
        )

    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise WorktreeError("every group must be an object")
        group_id = _safe_id(str(raw_group.get("group_id", "")), "group_id")
        if group_id in seen:
            raise WorktreeError(f"duplicate group_id: {group_id}")
        seen.add(group_id)
        raw_paths = (
            raw_group.get("planned_paths", raw_group.get("change_locations", [])) or []
        )
        if not isinstance(raw_paths, list):
            raise WorktreeError(f"planned_paths must be an array for {group_id}")
        resources = raw_group.get("exclusive_resources", []) or []
        if not isinstance(resources, list):
            raise WorktreeError(f"exclusive_resources must be an array for {group_id}")
        groups.append(
            {
                "group_id": group_id,
                "planned_paths": sorted(
                    {_normalize_path(str(item)) for item in raw_paths}
                ),
                "exclusive_resources": sorted(
                    {str(item).strip() for item in resources if str(item).strip()}
                ),
            }
        )
    return groups


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    groups = _read_groups(Path(args.groups_json))
    conflicts: list[dict[str, Any]] = []
    conflict_pairs: set[frozenset[str]] = set()
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            reasons = _group_conflicts(left, right)
            if reasons:
                conflict_pairs.add(frozenset((left["group_id"], right["group_id"])))
                conflicts.append(
                    {
                        "groups": [left["group_id"], right["group_id"]],
                        "reasons": reasons,
                    }
                )

    waves: list[list[str]] = []
    for group in groups:
        group_id = group["group_id"]
        for wave in waves:
            if all(
                frozenset((group_id, member)) not in conflict_pairs for member in wave
            ):
                wave.append(group_id)
                break
        else:
            waves.append([group_id])

    assignments = {
        group_id: wave_index
        for wave_index, wave in enumerate(waves, start=1)
        for group_id in wave
    }
    return {
        "groups": groups,
        "conflicts": conflicts,
        "waves": [
            {"wave": index, "groups": wave, "parallel": len(wave) > 1}
            for index, wave in enumerate(waves, start=1)
        ],
        "assignments": assignments,
    }


def _new_manifest(repo_root: Path, worktree_root: Path, run_id: str) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "run_id": run_id,
        "repo_root": str(repo_root),
        "git_common_dir": str(_git_common_dir(repo_root)),
        "worktree_root": str(worktree_root),
        "created_at": _now(),
        "groups": {},
    }


def _validate_manifest_repository(manifest: dict[str, Any], repo_root: Path) -> None:
    if Path(manifest["repo_root"]).resolve() != repo_root:
        raise WorktreeError("manifest belongs to a different repository root")
    if Path(manifest["git_common_dir"]).resolve() != _git_common_dir(repo_root):
        raise WorktreeError("manifest belongs to a different Git common directory")
    _validate_worktree_root(Path(manifest["worktree_root"]).resolve(), repo_root)


class CreateContext(NamedTuple):
    args: argparse.Namespace
    repo_root: Path
    run_id: str
    group_id: str
    manifest_path: Path
    worktree_root: Path
    target: Path


def _prepare_create(args: argparse.Namespace) -> CreateContext:
    repo_root = _resolve_repo(args.repo_root)
    run_id = _safe_id(args.run_id, "run_id")
    group_id = _safe_id(args.group_id, "group_id")
    manifest_path = Path(args.manifest).resolve()
    worktree_root = Path(args.worktree_root).resolve()
    _validate_worktree_root(worktree_root, repo_root)
    target = (worktree_root / run_id / group_id).resolve()
    if worktree_root not in target.parents:
        raise WorktreeError(
            "resolved worktree path escaped the configured worktree root"
        )

    _git("check-ref-format", "--branch", args.branch)
    if run_id not in args.branch:
        raise WorktreeError(
            "branch must contain the run_id to remain unique across runs"
        )
    _git(
        "-C",
        str(repo_root),
        "rev-parse",
        "--verify",
        f"{args.base_ref}^{{commit}}",
    )
    return CreateContext(
        args,
        repo_root,
        run_id,
        group_id,
        manifest_path,
        worktree_root,
        target,
    )


def _load_create_manifest(context: CreateContext) -> dict[str, Any]:
    manifest = (
        _load_manifest(context.manifest_path)
        if context.manifest_path.exists()
        else _new_manifest(context.repo_root, context.worktree_root, context.run_id)
    )
    _validate_manifest_repository(manifest, context.repo_root)
    manifest_root = Path(manifest["worktree_root"]).resolve()
    if manifest["run_id"] != context.run_id or manifest_root != context.worktree_root:
        raise WorktreeError(
            "run_id or worktree_root does not match the existing manifest"
        )
    if context.group_id in manifest["groups"]:
        raise WorktreeError(f"group already exists in manifest: {context.group_id}")
    if context.target.exists() and not context.target.is_dir():
        raise WorktreeError(
            f"worktree target exists and is not a directory: {context.target}"
        )
    if context.target.exists() and any(context.target.iterdir()):
        raise WorktreeError(
            f"worktree target already exists and is not empty: {context.target}"
        )
    return manifest


def _add_managed_worktree(
    context: CreateContext, manifest: dict[str, Any]
) -> dict[str, Any]:
    args = context.args
    context.target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(
            "-C",
            str(context.repo_root),
            "worktree",
            "add",
            "-b",
            args.branch,
            str(context.target),
            args.base_ref,
        )
        head = _git("-C", str(context.target), "rev-parse", "HEAD").stdout.strip()
        group = {
            "group_id": context.group_id,
            "branch": args.branch,
            "base_ref": args.base_ref,
            "base_commit": head,
            "worktree_path": str(context.target),
            "wave": args.wave,
            "planned_paths": sorted(
                {_normalize_path(item) for item in args.planned_path}
            ),
            "exclusive_resources": sorted(set(args.exclusive_resource)),
            "lifecycle_status": "active",
            "created_at": _now(),
        }
        manifest["groups"][context.group_id] = group
        _atomic_write(context.manifest_path, manifest)
        return group
    except (OSError, TypeError, ValueError, WorktreeError, subprocess.SubprocessError):
        if context.target.exists():
            _git(
                "-C",
                str(context.repo_root),
                "worktree",
                "remove",
                str(context.target),
                check=False,
            )
        raise


def command_create(args: argparse.Namespace) -> dict[str, Any]:
    context = _prepare_create(args)

    with _manifest_lock(context.manifest_path):
        manifest = _load_create_manifest(context)
        return _add_managed_worktree(context, manifest)


def command_mark(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    if args.status not in LIFECYCLE_STATES - {"cleaned"}:
        raise WorktreeError(f"unsupported lifecycle status: {args.status}")
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        try:
            group = manifest["groups"][args.group_id]
        except KeyError as exc:
            raise WorktreeError(
                f"group is not managed by this manifest: {args.group_id}"
            ) from exc
        if group["lifecycle_status"] == "cleaned":
            raise WorktreeError(f"group is already cleaned: {args.group_id}")
        group["lifecycle_status"] = args.status
        group["updated_at"] = _now()
        if args.commit_sha:
            group["commit_sha"] = args.commit_sha
        if args.published_ref:
            group["published_ref"] = args.published_ref
        if args.pr_url:
            group["pr_url"] = args.pr_url
        if args.reason:
            group["reason"] = args.reason
        _atomic_write(manifest_path, manifest)
        return group


def _registered_worktrees(repo_root: Path) -> set[Path]:
    result = _git("-C", str(repo_root), "worktree", "list", "--porcelain")
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in result.stdout.splitlines()
        if line.startswith("worktree ")
    }


def command_cleanup(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    cleaned: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        repo_root = _resolve_repo(manifest["repo_root"])
        _validate_manifest_repository(manifest, repo_root)
        worktree_root = Path(manifest["worktree_root"]).resolve()
        requested = args.group_id or list(manifest["groups"])
        registered = _registered_worktrees(repo_root)

        for group_id in requested:
            group = manifest["groups"].get(group_id)
            if group is None:
                skipped.append({"group_id": group_id, "reason": "not_managed"})
                continue
            status = group.get("lifecycle_status")
            if status == "cleaned":
                skipped.append({"group_id": group_id, "reason": "already_cleaned"})
                continue
            if status not in TERMINAL_CLEANUP_STATES:
                skipped.append(
                    {"group_id": group_id, "reason": f"non_terminal:{status}"}
                )
                continue

            path = Path(group["worktree_path"]).resolve()
            expected_path = (worktree_root / manifest["run_id"] / group_id).resolve()
            if path != expected_path or worktree_root not in path.parents:
                raise WorktreeError(f"refusing unexpected managed path: {path}")
            if path not in registered:
                skipped.append({"group_id": group_id, "reason": "not_registered"})
                continue
            if not path.is_dir():
                skipped.append({"group_id": group_id, "reason": "missing_path"})
                continue
            status_output = _git("-C", str(path), "status", "--porcelain").stdout
            if status_output.strip():
                skipped.append({"group_id": group_id, "reason": "dirty_worktree"})
                continue

            _git("-C", str(repo_root), "worktree", "remove", str(path))
            group["lifecycle_status"] = "cleaned"
            group["cleaned_at"] = _now()
            cleaned.append({"group_id": group_id, "worktree_path": str(path)})

        run_root = (worktree_root / manifest["run_id"]).resolve()
        if run_root.parent == worktree_root:
            with contextlib.suppress(OSError):
                run_root.rmdir()
        _atomic_write(manifest_path, manifest)
    return {"cleaned": cleaned, "skipped": skipped, "manifest": str(manifest_path)}


def command_inspect(args: argparse.Namespace) -> dict[str, Any]:
    return _load_manifest(Path(args.manifest).resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="build conflict-free execution waves")
    plan.add_argument("--groups-json", required=True)
    plan.set_defaults(handler=command_plan)

    create = subparsers.add_parser(
        "create", help="create and register one managed worktree"
    )
    create.add_argument("--repo-root", required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--worktree-root", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--group-id", required=True)
    create.add_argument("--branch", required=True)
    create.add_argument("--base-ref", required=True)
    create.add_argument("--wave", type=int, required=True)
    create.add_argument("--planned-path", action="append", default=[])
    create.add_argument("--exclusive-resource", action="append", default=[])
    create.set_defaults(handler=command_create)

    mark = subparsers.add_parser("mark", help="record a group's lifecycle state")
    mark.add_argument("--manifest", required=True)
    mark.add_argument("--group-id", required=True)
    mark.add_argument("--status", required=True)
    mark.add_argument("--commit-sha")
    mark.add_argument("--published-ref")
    mark.add_argument("--pr-url")
    mark.add_argument("--reason")
    mark.set_defaults(handler=command_mark)

    cleanup = subparsers.add_parser(
        "cleanup", help="remove clean terminal managed worktrees"
    )
    cleanup.add_argument("--manifest", required=True)
    cleanup.add_argument("--group-id", action="append")
    cleanup.set_defaults(handler=command_cleanup)

    inspect = subparsers.add_parser("inspect", help="print a worktree manifest")
    inspect.add_argument("--manifest", required=True)
    inspect.set_defaults(handler=command_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except WorktreeError as exc:
        write_stderr(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
        )
        return 2
    write_stdout(
        json.dumps({"status": "ok", "result": result}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
