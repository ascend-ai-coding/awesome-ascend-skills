#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""User-visible report rounds, independent of internal scan identities."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import re
import time
import uuid

from fetch_cache import _atomic_write_json
from pipeline_evidence import content_hash
from protocol_output import write_json

ZONE = timezone(timedelta(hours=8))
RUN_NAME = re.compile(r"\d{8}-\d{6}-P0800(?:-\d{2,})?")


def create_run(reports, repository, mode, now=None, iid=None):
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    started = datetime.fromtimestamp(time.time() if now is None else now, ZONE)
    stem = started.strftime("%Y%m%d-%H%M%S-P0800")
    suffix = 1
    while True:
        name = stem if suffix == 1 else f"{stem}-{suffix:02d}"
        directory = reports / name
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1
    value = {
        "version": 1,
        "run_id": name,
        "mode": mode,
        "repository": repository,
        "started_at": started.isoformat(),
        "timezone": "Asia/Shanghai",
        "internal_run_id": uuid.uuid4().hex,
        "scan_ids": [],
    }
    if iid is not None:
        value["issue_iid"] = str(iid)
    save_run(reports, value)
    return value


def run_path(reports, value):
    name = value.get("run_id", "")
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        raise ValueError("invalid_report_run_id")
    path = Path(reports) / name
    if path.is_symlink():
        raise ValueError("invalid_report_run_directory")
    return path


def save_run(reports, value):
    directory = run_path(reports, value)
    # Publication bookkeeping belongs to queue state, not the user metadata.
    metadata = {key: item for key, item in value.items() if key != "material_versions"}
    _atomic_write_json(directory / "run.json", metadata)


def ensure_run(state, store, now, new=False):
    value = state.get("report_run")
    if new or not value:
        value = create_run(store.root / "reports", state["repository"], "batch", now)
        state["report_run"] = value
        state.pop("delivery_report", None)
    if value.get("repository") != state["repository"]:
        raise ValueError("report_run_repository_mismatch")
    run_path(store.root / "reports", value)
    scan = state.get("scan", {}).get("id")
    if not new and scan and scan not in value["scan_ids"]:
        value["scan_ids"].append(scan)
    save_run(store.root / "reports", value)
    return Path("reports") / value["run_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start"])
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--mode", choices=["single"], default="single")
    parser.add_argument("--iid", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    from handler_config import load_handler_config

    root = Path(args.repository_root).resolve()
    runtime = root / ".cannbot/gitcode-issue-handler"
    cfg = load_handler_config(Path(args.config) if args.config else runtime / "config/classify_config.yaml")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", cfg.get("repo", "")) or not args.iid.isdigit() or int(args.iid) <= 0:
        raise ValueError("repository_and_positive_issue_id_required")
    value = create_run(runtime / "reports", cfg["repo"], args.mode, iid=args.iid)
    directory = run_path(runtime / "reports", value)
    state_file = directory / "_internal/run_state.json"
    _atomic_write_json(
        state_file,
        {
            "run": {
                **value,
                "repository_root": str(root),
                "report_directory": str(directory),
                "overall_status": "running",
                "authorization_mode": "interactive",
            },
            "issues": [],
            "groups": [],
            "external_operations": [],
        },
    )
    write_json(
        {
            "run_id": value["run_id"],
            "report_directory": str(directory),
            "run_file": str(directory / "run.json"),
            "state_file": str(state_file),
        },
        ensure_ascii=False,
    )
    return 0


def publish_materials(state, store, now, report):
    """Publish current Markdown copies; keep source artifacts immutable."""
    base = ensure_run(state, store, now)
    versions = state["report_run"].setdefault("material_versions", {})
    eligible = {str(issue["iid"]): issue for issue in report["issues"]}
    for iid in set(versions) | set(eligible):
        _publish_issue_materials(store, base, iid, versions, eligible)
    return report


def _publish_issue_materials(store, base, iid, versions, eligible):
    previous = versions.get(iid, {})
    sources = eligible.get(iid, {}).get("response_artifacts", {})
    contents = {
        name: Path(path).read_text(encoding="utf-8")
        for name, path in sources.items()
        if name in {"analysis.md", "reply.md", "assign.md"}
    }
    signature = content_hash(contents) if contents else None
    directory = base / "issues" / f"issue-{iid}"
    version = _updated_material_version(store, directory, previous, contents, signature)
    if version is not None:
        versions[iid] = version
    if iid not in eligible:
        return
    # Report links only to this round's visible Markdown and actual review file.
    files = dict(versions.get(iid, {}).get("files", {}))
    for name, path in sources.items():
        if name not in contents:
            files[name] = store.text_artifact(
                base / "_internal/issues" / f"issue-{iid}" / name, Path(path).read_text(encoding="utf-8")
            )
    eligible[iid]["response_artifacts"] = files


def _updated_material_version(store, directory, previous, contents, signature):
    if previous.get("signature") == signature and all((store.root / directory / name).is_file() for name in contents):
        return None
    revision = previous.get("revision", 0) + 1
    for name in previous.get("files", {}):
        current = store.root / directory / name
        if current.is_file():
            store.text_artifact(
                directory / "history" / f"revision-{revision - 1}" / name, current.read_text(encoding="utf-8")
            )
            current.unlink()
    files = {name: store.text_artifact(directory / name, text) for name, text in contents.items()}
    return {"signature": signature, "revision": revision, "files": files}


if __name__ == "__main__":
    raise SystemExit(main())
