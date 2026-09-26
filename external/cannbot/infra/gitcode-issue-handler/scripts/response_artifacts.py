# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persist reviewable, per-Issue response drafts before any external writes."""
from __future__ import annotations

import hashlib
from pathlib import Path
from report_runs import RUN_NAME

from fetch_cache import _atomic_write_json, _read_json
from gitcode_client import parse_issue_url


def save_response_artifacts(result_file, issue_url, analysis, *, reply=None, owner=None):
    """Share analysis across comment and assignment without losing either draft."""
    _, _, number = parse_issue_url(issue_url)
    parent = Path(result_file).parent
    directory = parent if parent.name == f"issue-{number}" else parent / "issues" / f"issue-{number}"
    run_directory = next((p for p in (parent, *parent.parents)
                          if RUN_NAME.fullmatch(p.name) and (p / 'run.json').is_file()), None)
    if run_directory:
        directory = run_directory / 'issues' / f'issue-{number}'
        manifest_path = run_directory / '_internal/issues' / f'issue-{number}' / 'response-artifacts.json'
    else:
        manifest_path = directory / "response-artifacts.json"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_path) or {"issue_url": issue_url, "analysis": {}, "files": {}}
    if manifest.get("issue_url") != issue_url:
        raise ValueError("response artifact directory belongs to another Issue")
    analysis_path = directory / "analysis.md"
    if analysis_path.exists():
        existing = analysis_path.read_text(encoding="utf-8")
        previous_digest = manifest["files"].get("analysis.md", {}).get("digest")
        if hashlib.sha256(existing.encode()).hexdigest() != previous_digest:
            manifest["analysis"]["review"] = existing.strip()
    kind = "assignment" if owner else "reply"
    manifest["analysis"][kind] = str(analysis).strip()
    drafts = {"analysis.md": f"# Issue #{number}\n\n{issue_url}\n\n" + "\n\n".join(
        part for part in manifest["analysis"].values() if part
    ) + "\n"}
    if reply is not None:
        drafts["reply.md"] = reply
    if owner:
        drafts["assign.md"] = f"/assign @{owner}\n"
    if run_directory and any((directory / name).is_file() and
                             (directory / name).read_text(encoding='utf-8') != body
                             for name, body in drafts.items()):
        history = directory / 'history'
        revision = 1
        while (history / f'revision-{revision}').exists():
            revision += 1
        archive = history / f'revision-{revision}'
        archive.mkdir(parents=True)
        for existing in directory.glob('*.md'):
            (archive / existing.name).write_text(existing.read_text(encoding='utf-8'), encoding='utf-8')
    for name, body in drafts.items():
        path = directory / name
        path.write_text(body, encoding="utf-8")
        manifest["files"][name] = {"path": str(path), "digest": hashlib.sha256(body.encode()).hexdigest()}
    _atomic_write_json(str(manifest_path), manifest)
    return manifest["files"]
