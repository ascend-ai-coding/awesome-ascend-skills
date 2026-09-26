# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Report-round boundaries and layouts; all fixtures are local/offline."""

from datetime import datetime
from pathlib import Path
import copy
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_runs as rr
import issue_pipeline as pipeline
from pipeline_store import PipelineStore, new_state
from response_artifacts import save_response_artifacts

NOW = datetime.fromisoformat("2026-09-23T09:16:34+08:00").timestamp()


def test_fixed_timezone_collision_and_stored_internal_ids(tmp_path):
    first = rr.create_run(tmp_path, "team/repo", "single", NOW, "297")
    second = rr.create_run(tmp_path, "team/repo", "batch", NOW)
    assert first["run_id"] == "20260923-091634-P0800"
    assert second["run_id"] == "20260923-091634-P0800-02"
    assert first["internal_run_id"] != second["internal_run_id"]
    assert json.loads((tmp_path / first["run_id"] / "run.json").read_text())["issue_iid"] == "297"


def test_refresh_keeps_round_new_request_creates_round_and_preserves_queue(tmp_path):
    state = new_state("team/repo")
    state["tasks"]["fixture"] = {"status": "waiting"}
    store = PipelineStore(tmp_path, "team/repo")
    first = rr.ensure_run(state, store, NOW)
    state["scan"]["id"] = "scan-one"
    rr.ensure_run(state, store, NOW + 10)
    state["scan"]["id"] = "scan-two"
    assert rr.ensure_run(state, store, NOW + 20) == first
    metadata = json.loads((store.root / first / "run.json").read_text())
    assert metadata["scan_ids"] == ["scan-one", "scan-two"]
    second = rr.ensure_run(state, store, NOW + 30, new=True)
    assert first != second
    assert state["tasks"]["fixture"]["status"] == "waiting"
    assert json.loads((store.root / first / "run.json").read_text()) == metadata


def test_published_drafts_are_local_to_round_and_stale_drafts_go_to_history(tmp_path):
    store = PipelineStore(tmp_path / ".cannbot/gitcode-issue-handler", "team/repo")
    state = new_state("team/repo")
    legacy = tmp_path / "legacy-hash"
    legacy.mkdir()
    draft = legacy / "reply.md"
    review = legacy / "response-review.json"
    draft.write_text("first draft")
    review.write_text("{}")
    report = {
        "issues": [{"iid": "297", "response_artifacts": {"reply.md": str(draft), "response-review.json": str(review)}}]
    }
    view = rr.publish_materials(state, store, NOW, copy.deepcopy(report))
    current = Path(view["issues"][0]["response_artifacts"]["reply.md"])
    assert current == store.root / "reports/20260923-091634-P0800/issues/issue-297/reply.md"
    assert "/_internal/issues/" in view["issues"][0]["response_artifacts"]["response-review.json"]
    rr.publish_materials(state, store, NOW + 1, copy.deepcopy(report))
    assert not (current.parent / "history").exists()
    draft.write_text("second draft")
    rr.publish_materials(state, store, NOW + 2, copy.deepcopy(report))
    assert current.read_text() == "second draft"
    assert (current.parent / "history/revision-1/reply.md").read_text() == "first draft"
    rr.publish_materials(state, store, NOW + 3, {"issues": []})
    assert not current.exists()
    assert (current.parent / "history/revision-2/reply.md").read_text() == "second draft"
    assert draft.read_text() == "second draft"
    rr.ensure_run(state, store, NOW + 10, new=True)
    view = rr.publish_materials(state, store, NOW + 10, copy.deepcopy(report))
    assert Path(view["issues"][0]["response_artifacts"]["reply.md"]).read_text() == "second draft"
    assert "20260923-091644-P0800" in view["issues"][0]["response_artifacts"]["reply.md"]


def test_single_executor_writes_user_markdown_and_internal_manifest(tmp_path):
    value = rr.create_run(tmp_path, "team/repo", "single", NOW, "297")
    root = tmp_path / value["run_id"]
    result = root / "_internal/issues/issue-297/comment.json"
    files = save_response_artifacts(result, "https://gitcode.com/team/repo/issues/297", "Analysis", reply="Draft")
    assert Path(files["reply.md"]["path"]) == root / "issues/issue-297/reply.md"
    assert (root / "_internal/issues/issue-297/response-artifacts.json").is_file()
    assert not (root / "issues/issue-297/response-artifacts.json").exists()
    save_response_artifacts(result, "https://gitcode.com/team/repo/issues/297", "Analysis", reply="Revised")
    assert (root / "issues/issue-297/history/revision-1/reply.md").read_text() == "Draft"


def test_offline_new_run_and_refresh_keep_distinct_scan_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / pipeline.RUNTIME / "config/classify_config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("repo: team/repo\n")
    snapshot = tmp_path / "fetch.json"
    snapshot.write_text(
        json.dumps(
            {
                "filters": {"repository": "team/repo", "mode": "batch", "since": None, "follow_up": {"complete": True}},
                "primary_scan": {"complete": True},
                "issues": [],
            }
        )
    )
    args = ["--repository-root", str(tmp_path), "--input", str(snapshot), "--offline"]
    first = pipeline.run(pipeline.parser().parse_args(["resume", "--new-run", *args]))
    metadata = json.loads(Path(first["run_file"]).read_text())
    second = pipeline.run(pipeline.parser().parse_args(["resume", "--refresh", *args]))
    assert second["report_directory"] == first["report_directory"]
    newer = json.loads(Path(second["run_file"]).read_text())
    assert len(newer["scan_ids"]) == 2 and newer["scan_ids"][0] == metadata["scan_ids"][0]
    third = pipeline.run(pipeline.parser().parse_args(["resume", "--new-run", *args]))
    assert third["report_directory"] != first["report_directory"]
    assert len(json.loads(Path(third["run_file"]).read_text())["scan_ids"]) == 1
    assert Path(first["status_file"]).is_file()
    assert "/_internal/" in first["status_file"]
    assert not (tmp_path / pipeline.RUNTIME / "reports/pipeline").exists()


def test_single_start_does_not_create_or_scan_batch_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / pipeline.RUNTIME / "config/classify_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("repo: team/repo\n")
    rr.main(["start", "--repository-root", str(tmp_path), "--mode", "single", "--iid", "297"])
    result = json.loads(capsys.readouterr().out)
    state = json.loads(Path(result["state_file"]).read_text())
    assert state["run"]["mode"] == "single" and state["run"]["issue_iid"] == "297"
    assert "/_internal/" in result["state_file"]
    assert not (tmp_path / pipeline.RUNTIME / "data/pipeline-state.json").exists()
