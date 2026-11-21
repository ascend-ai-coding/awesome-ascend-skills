# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Task inventory preserves declarations, missing identities and bounded delivery."""

import hashlib
import json
import subprocess
import sys

import pytest

from _checks import host_io, task_context
from _checks.paths import SKILL_ROOT


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("raise RuntimeError('must not import')\n")
    manifest = tmp_path / "task.json"
    data = {
        "schema": "pypto-task-context/1",
        "artifacts": [
            {
                "path": source.name,
                "role": "source",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        "inline_evidence": [
            {"id": "cost", "kind": "synthetic_fixture", "value": {"elapsed_us": 7}}
        ],
        "unavailable": ["No native measurement was supplied."],
    }
    manifest.write_text(json.dumps(data))
    return manifest, source, data, host_io.policy(workspace=tmp_path)


def test_only_declared_bytes_are_checked_without_import_or_directory_discovery(
    task, monkeypatch
):
    manifest, source, _, config = task
    monkeypatch.setattr(
        type(source), "rglob", lambda *args: pytest.fail("must not enumerate")
    )
    result = task_context.report(manifest, config=config)
    assert result["items"][0]["status"] == "verified_bytes"
    assert result["inline_evidence"][0]["storage"] == "inline"
    assert result["inline_evidence"][0]["authenticated"] is False
    assert set(result["validation"].values()) == {"not_established"}
    assert result["coverage"]["files_read"] == 2


@pytest.mark.parametrize(
    "action, expected", [("change", "changed"), ("delete", "missing")]
)
def test_stale_or_missing_artifact_is_unverified(task, action, expected):
    manifest, source, _, config = task
    source.write_text("different\n") if action == "change" else source.unlink()
    result = task_context.report(manifest, config=config)
    assert result["status"] == "unverified" and not result["coverage"]["complete"]
    assert result["items"][0]["status"] == expected


def test_task_reads_do_not_exceed_reported_scan_budget(task, monkeypatch):
    manifest, _, _, config = task
    original = host_io._bytes
    observed = []

    def record(path, limit):
        payload = original(path, limit)
        observed.append((path, len(payload)))
        return payload

    monkeypatch.setattr(host_io, "_bytes", record)
    result = task_context.report(manifest, config=config)
    assert sum(size for _, size in observed) == result["coverage"]["scan_bytes"]
    assert len(observed) == result["coverage"]["files_read"]


def test_task_file_budget_leaves_declared_artifact_unvisited(task):
    manifest, _, _, config = task
    config["limits"]["max_files"] = 1
    result = task_context.report(manifest, config=config)
    assert result["status"] == "partial" and result["reason"] == "file_scan_budget"
    assert (
        result["coverage"]["files_read"] == 1
        and result["remaining_artifact_index"] == 0
    )
    assert result["items"] == []


def test_manifest_mutation_during_artifact_read_is_unverified(task, monkeypatch):
    manifest, source, data, config = task
    original = host_io._bytes

    def mutate(path, limit):
        payload = original(path, limit)
        if path == source:
            data["unavailable"] = ["changed evidence declaration"]
            manifest.write_text(json.dumps(data))
        return payload

    monkeypatch.setattr(host_io, "_bytes", mutate)
    with pytest.raises(ValueError, match="manifest changed"):
        task_context.report(manifest, config=config)


@pytest.mark.parametrize("mutation", ["role", "kind", "sha", "duplicate", "outside"])
def test_invalid_inventory_does_not_become_success(task, mutation):
    manifest, _, data, config = task
    if mutation == "role":
        data["artifacts"][0]["role"] = "execute"
    if mutation == "kind":
        data["inline_evidence"][0]["kind"] = "authenticated_native"
    if mutation == "sha":
        data["artifacts"][0]["sha256"] = "unknown"
    if mutation == "duplicate":
        data["artifacts"] *= 2
    if mutation == "outside":
        config = {**config, "roots": (manifest.parent,)}
        data["artifacts"][0]["path"] = "../outside.py"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        task_context.report(manifest, config=config)


def test_cli_inventory_and_small_error_reply_are_measured(task):
    manifest, _, data, _ = task
    command = [
        sys.executable,
        str(SKILL_ROOT / "scripts/check.py"),
        "query",
        "--task",
        str(manifest),
    ]
    result = subprocess.run(
        command, cwd=manifest.parent, capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert len(result.stdout.encode()) == report["budget"]["result_bytes"]
    policy = manifest.parent / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(manifest.parent)],
                "limits": {"max_output_bytes": 2048},
            }
        )
    )
    data["inline_evidence"][0]["value"] = "long" * 2000
    manifest.write_text(json.dumps(data))
    result = subprocess.run(
        command + ["--policy", str(policy)],
        cwd=manifest.parent,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 2 and json.loads(result.stdout)["status"] == "unverified"
    )
    assert len(result.stdout.encode()) <= 2048


@pytest.mark.parametrize(
    "content", ["Find a kernel for this task.\n", "{}", '"plain question"']
)
def test_prose_and_non_inventory_json_explain_the_knowledge_entry(task, content):
    manifest, _, _, _ = task
    manifest.write_text(content)
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "query",
            "--task",
            str(manifest),
        ],
        cwd=manifest.parent,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 2 and report["status"] == "unverified"
    assert (
        "pypto-task-context/1 JSON inventory" in report["error"]
        and "query TEXT" in report["error"]
    )
    assert report["items"] == [] and not report["coverage"]["complete"]


def test_batch_ranges_share_budget_and_preserve_order(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("a\nb\nc\n")
    second.write_text("一\n二\n三\n")
    config = host_io.policy(workspace=tmp_path)
    result = host_io.read_requests(
        [
            {"path": str(first), "start_line": 2, "max_lines": 1},
            {"path": str(second), "start_line": 3, "max_lines": 1},
        ],
        config=config,
    )
    assert [item["text"] for item in result["items"]] == ["b\n", "三\n"]
    assert [item["request_index"] for item in result["items"]] == [0, 1]
    assert result["status"] == "complete"
    assert (
        len((host_io.encode(result) + "\n").encode())
        == result["budget"]["result_bytes"]
    )


def test_batch_partial_page_identifies_current_request_and_unvisited_tail(tmp_path):
    source = tmp_path / "long.txt"
    source.write_text("长" * 5000)
    config = host_io.policy(workspace=tmp_path)
    config["limits"]["max_output_bytes"] = 4096
    result = host_io.read_requests(
        [{"path": str(source)}, {"path": str(source), "max_lines": 1}], config=config
    )
    assert result["status"] == "partial" and result["next"]["request_index"] == 0
    assert result["remaining_request_index"] == 1
    assert result["budget"]["result_bytes"] <= 4096


@pytest.mark.parametrize("batch", [False, True])
def test_explicit_reads_stop_at_the_shared_file_budget(tmp_path, batch, monkeypatch):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("first\n")
    second.write_text("second\n")
    config = host_io.policy(workspace=tmp_path)
    config["limits"]["max_files"] = 1
    original = host_io._bytes
    observed = []

    def record(path, limit):
        observed.append(path)
        return original(path, limit)

    monkeypatch.setattr(host_io, "_bytes", record)
    result = (
        host_io.read_requests(
            [{"path": str(first)}, {"path": str(second)}], config=config
        )
        if batch
        else host_io.read([first, second], config=config)
    )
    assert result["status"] == "partial" and result["reason"] == "file_scan_budget"
    assert result["coverage"]["files_read"] == 1 and observed == [first]
    assert result["remaining_request_index" if batch else "remaining_path_index"] == 1
