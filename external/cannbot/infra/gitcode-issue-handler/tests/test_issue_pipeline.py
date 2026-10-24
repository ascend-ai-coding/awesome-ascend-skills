# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Offline workflow replay: no GitCode requests or publication."""

import copy
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import issue_pipeline as p
from pipeline_store import PipelineStore, new_state
from scope_cache import make_entry, restore
from responsibility import policy_digest


@pytest.fixture
def cfg():
    return {
        "repo": "cann/samples",
        "responsibility": {"handle": ["具体问题"], "list-only": ["规划"], "ignore": ["用户指定"]},
        "ignored_issue_ids": [],
        "cache_dir": "/nonexistent-pipeline-fixture-cache",
    }


def issue(iid="1", **extra):
    return {
        "iid": iid,
        "number": iid,
        "state": "open",
        "title": "构建报错",
        "description": "cmake/shmem.cmake 使用错误编译器",
        "author": "reporter",
        "updated_at": "2026-09-21T01:00:00+00:00",
        "comments_count": 0,
        "created_at": "2020-01-01T00:00:00+00:00",
        **extra,
    }


def snapshot(issues, *, complete=True, full=True, cursor="2026-09-22T00:00:00+00:00"):
    return {
        "filters": {
            "mode": "batch",
            "repository": "cann/samples",
            "since": None,
            "follow_up": {
                "complete": complete,
                "cursor_proposed": cursor if complete else None,
                "updated_scan": {"closed": []},
                "watchlist_refresh": {"closed": []},
            },
        },
        "primary_scan": {"complete": complete, "skipped": not full},
        "issues": issues,
    }


NOW = 1790035200.0


def init(cfg, items=None):
    state = new_state(cfg["repo"])
    p.ingest(state, snapshot(items if items is not None else [issue()]), cfg, NOW)
    p.synchronize(state, cfg, NOW)
    return state


def review(cfg, level="handle"):
    return {
        "level": level,
        "summary": "实际文件属于本仓具体问题",
        "evidence": ["已核查固定提交中的构建入口"],
        "policy_digest": policy_digest(cfg["responsibility"]),
    }


def accept_scope(state, cfg, root):
    task = p.queue.claim(state, "worker", NOW + 1)
    p.queue.submit(
        state,
        task["task_id"],
        task["attempt_id"],
        {"decision": "handle", "summary": "本仓构建缺陷", "evidence": ["cmake/shmem.cmake 的配置命令"]},
        NOW + 2,
    )
    p.accept_result(state, task["task_id"], cfg, root, NOW + 3)
    p.synchronize(state, cfg, NOW + 3)
    return task


def test_first_run_includes_old_open_issue(cfg):
    state = init(cfg)
    assert state["issues"]["1"]["phase"] == "scope"
    assert not state["needs_full_scan"]
    assert len(state["tasks"]) == 1


@pytest.mark.parametrize("invalid", [[], {"decision": "handle", "evidence": ["file"]}])
def test_rejected_result_can_be_corrected_with_new_attempt(cfg, tmp_path, invalid):
    state = init(cfg)
    task = p.queue.claim(state, "worker", NOW + 1)
    p.queue.submit(state, task["task_id"], task["attempt_id"], invalid, NOW + 2)
    with pytest.raises(ValueError, match="invalid_result_object|missing_summary"):
        p.accept_result(state, task["task_id"], cfg, tmp_path, NOW + 3)
    p.queue.requeue(state, task["task_id"], "审核修正", NOW + 4)
    renewed = p.queue.claim(state, "worker", NOW + 5)
    assert renewed["attempt_id"] != task["attempt_id"]
    p.queue.submit(
        state,
        renewed["task_id"],
        renewed["attempt_id"],
        {"decision": "handle", "summary": "构建错误", "evidence": ["固定源码"]},
        NOW + 6,
    )
    p.accept_result(state, renewed["task_id"], cfg, tmp_path, NOW + 7)
    assert state["issues"]["1"]["phase"] == "classify"


def test_waiting_task_summary_exposes_reason_after_resume(cfg, tmp_path):
    state = init(cfg)
    task = p.queue.claim(state, "worker", NOW + 1)
    p.queue.submit(
        state,
        task["task_id"],
        task["attempt_id"],
        {"decision": "needs_evidence", "summary": "缺少源码版本", "evidence": ["正文无版本"]},
        NOW + 2,
    )
    p.accept_result(state, task["task_id"], cfg, tmp_path, NOW + 3)
    p.synchronize(state, cfg, NOW + 4)
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        summary = p.output_status(state, store, NOW + 5)
    assert summary["tasks"]["waiting"][0]["reason"] == "缺少源码版本"
    assert summary["next_action"] == "resume_waiting_work"


def test_partial_scan_cannot_advance_cursor_or_complete(cfg, tmp_path):
    state = init(cfg)
    original = state["scan"]["cursor"]
    p.ingest(state, snapshot([issue("2")], complete=False), cfg, NOW + 10)
    assert state["scan"]["cursor"] == original
    assert set(state["issues"]) == {"1", "2"}
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        assert p.output_status(state, store, NOW + 11)["next_action"] != "complete"


def test_pending_issue_survives_incremental_scan_without_it(cfg):
    state = init(cfg)
    p.ingest(state, snapshot([], full=False), cfg, NOW + 10)
    assert state["issues"]["1"]["phase"] == "refresh"
    assert "1" in state["issues"]


def test_missing_from_full_scan_requires_get_not_inferred_closed(cfg):
    state = init(cfg)
    p.ingest(state, snapshot([]), cfg, NOW + 10)
    assert state["issues"]["1"]["issue"]["state"] == "open"
    assert state["issues"]["1"]["phase"] == "refresh"


def test_closed_issue_supersedes_lease_and_reopen_restores_task(cfg):
    state = init(cfg)
    task = p.queue.claim(state, "worker", NOW)
    p.ingest(state, snapshot([issue(state="closed")]), cfg, NOW + 10)
    p.synchronize(state, cfg, NOW + 10)
    assert state["tasks"][task["task_id"]]["status"] == "superseded"
    p.ingest(state, snapshot([issue()]), cfg, NOW + 20)
    p.synchronize(state, cfg, NOW + 20)
    assert state["issues"]["1"]["phase"] == "scope"
    with pytest.raises(ValueError):
        p.queue.submit(state, task["task_id"], task["attempt_id"], {}, NOW + 21)


def test_scope_reuse_after_new_session_but_activity_requires_delta(cfg, tmp_path):
    state = init(cfg)
    accept_scope(state, cfg, tmp_path)
    state = json.loads(json.dumps(state))
    p.ingest(state, snapshot([issue()]), cfg, NOW + 10)
    p.synchronize(state, cfg, NOW + 10)
    assert state["issues"]["1"]["scope_cache_reason"] == "reused"
    assert state["issues"]["1"]["phase"] == "classify"
    p.ingest(state, snapshot([issue(comments_count=1)]), cfg, NOW + 20)
    p.synchronize(state, cfg, NOW + 20)
    assert state["issues"]["1"]["scope_cache_reason"] == "activity_changed"
    assert state["issues"]["1"]["phase"] == "scope"


def test_exact_ignore_invalidates_only_matching_task_and_can_be_removed(cfg, tmp_path):
    state = init(cfg, [issue("1"), issue("2")])
    old = {t["iid"]: t["task_id"] for t in state["tasks"].values()}
    cfg["ignored_issue_ids"] = [1]
    p.synchronize(state, cfg, NOW + 1)
    assert state["issues"]["1"]["phase"] == "ignored"
    assert state["tasks"][old["1"]]["status"] == "superseded"
    assert state["tasks"][old["2"]]["status"] == "ready"
    cfg["ignored_issue_ids"] = []
    p.synchronize(state, cfg, NOW + 2)
    assert state["issues"]["1"]["phase"] == "scope"


def test_late_result_after_policy_change_cannot_be_accepted(cfg):
    state = init(cfg)
    task = p.queue.claim(state, "worker", NOW)
    cfg["responsibility"]["handle"] = ["仅文档"]
    p.synchronize(state, cfg, NOW + 1)
    with pytest.raises(ValueError, match="stale_input"):
        p.queue.submit(state, task["task_id"], task["attempt_id"], {}, NOW + 2)


def test_delta_task_contains_previous_evidence(cfg, tmp_path):
    state = init(cfg)
    accept_scope(state, cfg, tmp_path)
    p.ingest(state, snapshot([issue(updated_at="2026-09-22T00:00:00+00:00")]), cfg, NOW + 10)
    p.synchronize(state, cfg, NOW + 10)
    task = p.queue.claim(state, "next", NOW + 11)
    assert task["payload"]["previous_review"]["review"]["level"] == "handle"
    assert task["payload"]["reason"] == "activity_changed"


def test_fixed_and_moving_source_reuse(cfg):
    fixed = make_entry(issue(), cfg["responsibility"], review(cfg))
    assert restore(fixed, issue(), cfg["responsibility"])[1] == "reused"
    moving = make_entry(issue(), cfg["responsibility"], review(cfg), source_mode="moving")
    assert restore(moving, issue(), cfg["responsibility"])[0] is None


def test_one_ready_result_advances_without_other_workers(cfg, tmp_path):
    state = init(cfg, [issue("1"), issue("2"), issue("3")])
    tasks = [p.queue.claim(state, f"w{i}", NOW) for i in range(3)]
    first = tasks[0]
    p.queue.submit(
        state,
        first["task_id"],
        first["attempt_id"],
        {"decision": "handle", "summary": "已核实", "evidence": ["代码入口"]},
        NOW + 1,
    )
    p.accept_result(state, first["task_id"], cfg, tmp_path, NOW + 2)
    p.synchronize(state, cfg, NOW + 2)
    assert state["issues"][first["iid"]]["phase"] == "classify"
    assert sum(t["status"] == "running" for t in state["tasks"].values()) == 2


def test_needs_evidence_stays_waiting_after_status_and_resume(cfg, tmp_path):
    state = init(cfg)
    task = p.queue.claim(state, "w", NOW)
    p.queue.submit(
        state,
        task["task_id"],
        task["attempt_id"],
        {"decision": "needs_evidence", "summary": "需提供问题版本", "evidence": ["正文未给版本"]},
        NOW + 1,
    )
    p.accept_result(state, task["task_id"], cfg, tmp_path, NOW + 2)
    p.synchronize(state, cfg, NOW + 3)
    assert p.queue.claim(state, "other", NOW + 100) is None
    assert state["tasks"][task["task_id"]]["status"] == "waiting"


def test_no_attention_is_not_resolution_and_pr_refreshes(cfg, tmp_path):
    state = init(cfg)
    accept_scope(state, cfg, tmp_path)
    entry = state["issues"]["1"]
    p.apply_classification(
        state,
        {
            "issues": [
                {
                    "number": "1",
                    "bucket": "no_attention",
                    "category": "our_team_done_with_pr",
                    "responsibility": "handle",
                    "linked_prs": [{"pr_number": 2}],
                }
            ]
        },
        [entry],
        "classification.json",
        NOW + 5,
    )
    assert entry["phase"] == "observe"
    assert "resolution_status" not in entry
    p.ingest(state, snapshot([], full=False), cfg, NOW + 10)
    assert entry["phase"] == "refresh"


def test_list_only_observation_reconciles_without_response_task(cfg):
    state = init(cfg)
    entry = state["issues"]["1"]
    p.apply_classification(
        state,
        {"observations": [{"number": "1", "bucket": "no_attention", "responsibility": "list-only"}]},
        [entry],
        "result",
        NOW,
    )
    assert entry["phase"] == "observe"
    assert not any(t["stage"] == "response" for t in state["tasks"].values())


def test_missing_classification_is_not_completion(cfg):
    state = init(cfg)
    entry = state["issues"]["1"]
    entry["phase"] = "classify"
    p.apply_classification(state, {"issues": []}, [entry], "result", NOW)
    assert entry["phase"] == "classify"
    assert entry["classification_error"] == "missing_classification"


def test_legacy_import_does_not_trust_old_completion(cfg, tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    old_issue = issue(responsibility_review=review(cfg))
    (folder / "old-reviewed.json").write_text(
        json.dumps({"filters": {"repository": cfg["repo"]}, "issues": [old_issue]})
    )
    state = new_state(cfg["repo"])
    p.migrate(state, tmp_path, cfg)
    assert state["issues"]["1"]["scope"]["review"]["level"] == "handle"
    assert state["issues"]["1"]["phase"] == "refresh"
    assert state["needs_full_scan"]


def test_atomic_store_restores_corruption_and_requires_audit(cfg, tmp_path):
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        state = init(cfg)
        store.save(state)
        store.save(state)
        store.path.write_text("{broken")
        restored = store.load()
        assert "1" in restored["issues"]
        assert restored["needs_full_scan"]
        assert restored["recovery"]["requires_operation_audit"]
        assert Path(restored["recovery"]["preserved"]).read_text() == "{broken"


def test_atomic_write_failure_keeps_previous_generation(cfg, tmp_path, monkeypatch):
    import pipeline_store

    with PipelineStore(tmp_path, cfg["repo"]) as store:
        store.save(init(cfg))
        before = store.path.read_bytes()
        monkeypatch.setattr(
            pipeline_store, "_atomic_write_json", lambda *_: (_ for _ in ()).throw(OSError("disk_full"))
        )
        with pytest.raises(OSError):
            store.save(init(cfg, [issue("2")]))
        assert store.path.read_bytes() == before


def test_repository_lock_prevents_concurrent_owners(cfg, tmp_path):
    with PipelineStore(tmp_path, cfg["repo"]):
        with pytest.raises(ValueError, match="pipeline_busy"):
            with PipelineStore(tmp_path, cfg["repo"]):
                pass


def test_repository_mismatch_does_not_overwrite_state(cfg, tmp_path):
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        store.save(init(cfg))
    with PipelineStore(tmp_path, "other/repo") as store:
        with pytest.raises(RuntimeError, match="repository_mismatch"):
            store.load()


def setup_cli(tmp_path, cfg):
    import yaml

    root = tmp_path / "repo"
    path = root / p.RUNTIME / "config/classify_config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(cfg))
    captured = tmp_path / "snapshot.json"
    captured.write_text(json.dumps(snapshot([issue()])))
    return root, captured


def test_cli_new_session_resume_and_unknown_write_requires_verification(cfg, tmp_path, monkeypatch):
    root, captured = setup_cli(tmp_path, cfg)
    monkeypatch.setattr(p.time, "time", lambda: NOW + 20)

    def call(command, *extra):
        return p.run(p.parser().parse_args([command, "--repository-root", str(root), "--offline", *extra]))

    result = call("resume", "--input", str(captured))
    assert result["next_action"] == "claim_task"
    task = call("claim")["task"]
    restarted = call("status")
    assert restarted["tasks"]["counts"]["running"] == 1
    call("record-operation", "--operation-id", "comment-1", "--iid", "1")
    assert call("resume")["next_action"] == "verify_operations"
    assert call("status")["tasks"]["running"][0]["attempt_id"] == task["attempt_id"]


def test_acceptance_checkpoint_survives_classifier_failure(cfg, tmp_path, monkeypatch):
    root, captured = setup_cli(tmp_path, cfg)
    monkeypatch.setattr(p.time, "time", lambda: NOW + 20)

    def args(command, *extra, offline=True):
        return p.parser().parse_args(
            [command, "--repository-root", str(root), *(["--offline"] if offline else []), *extra]
        )

    p.run(args("resume", "--input", str(captured)))
    task = p.run(args("claim"))["task"]
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"decision": "handle", "summary": "已核查", "evidence": ["构建入口"]}))
    p.run(args("submit", "--task", task["task_id"], "--attempt", task["attempt_id"], "--result-file", str(result_path)))
    monkeypatch.setenv("GITCODE_TOKEN", "fixture-token")
    monkeypatch.setattr(p, "classify_ready", lambda *_: (_ for _ in ()).throw(ValueError("classifier_failed")))
    with pytest.raises(ValueError, match="classifier_failed"):
        p.run(args("accept", "--task", task["task_id"], offline=False))
    with PipelineStore(root / p.RUNTIME, cfg["repo"]) as store:
        state = store.load()
        assert state["tasks"][task["task_id"]]["status"] == "accepted"
        assert state["issues"]["1"]["scope"]["review"]["level"] == "handle"


def test_migrated_report_without_core_state_stays_refresh(cfg):
    state = new_state(cfg["repo"])
    state["issues"]["1"] = {"issue": {"iid": "1", "title": "历史未完成任务"}, "phase": "refresh"}
    p.synchronize(state, cfg, NOW)
    assert state["issues"]["1"]["phase"] == "refresh"


def test_moving_source_creates_new_task_on_next_scan(cfg, tmp_path):
    state = init(cfg)
    first = accept_scope(state, cfg, tmp_path)
    entry = state["issues"]["1"]
    entry["scope"]["source_mode"] = "moving"
    p.ingest(state, snapshot([issue()]), cfg, NOW + 10)
    p.synchronize(state, cfg, NOW + 10)
    task = p.queue.claim(state, "new", NOW + 11)
    assert task is not None
    assert task["task_id"] != first["task_id"]
    assert task["payload"]["reason"] == "moving_source_requires_review"


def test_old_response_cannot_skip_classification_on_new_scan(cfg, tmp_path):
    state = init(cfg)
    accept_scope(state, cfg, tmp_path)
    entry = state["issues"]["1"]
    result = {
        "issues": [
            {"number": "1", "bucket": "need_attention", "category": "needs_first_look", "responsibility": "handle"}
        ]
    }
    p.apply_classification(state, result, [entry], "classification.json", NOW + 5)
    task = p.queue.claim(state, "draft", NOW + 6)
    p.queue.submit(
        state,
        task["task_id"],
        task["attempt_id"],
        {"decision": "prepared", "summary": "分析完成", "evidence": ["代码"], "artifacts": ["reply.md"]},
        NOW + 7,
    )
    p.ingest(state, snapshot([issue()]), cfg, NOW + 10)
    p.synchronize(state, cfg, NOW + 10)
    with pytest.raises(ValueError, match="stale_classification"):
        p.accept_result(state, task["task_id"], cfg, tmp_path, NOW + 11)
    assert entry["phase"] == "classify"


@pytest.mark.parametrize("damage", ["generation", "migration_done", "issue_entry", "task_entry"])
def test_valid_json_but_damaged_schema_recovers(cfg, tmp_path, damage):
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        state = init(cfg)
        store.save(state)
        store.save(state)
        damaged = copy.deepcopy(state)
        if damage == "issue_entry":
            damaged["issues"]["1"] = None
        elif damage == "task_entry":
            damaged["tasks"][next(iter(damaged["tasks"]))] = None
        else:
            del damaged[damage]
        store.path.write_text(json.dumps(damaged))
        recovered = store.load()
        assert recovered["recovery"]["requires_operation_audit"]
        assert recovered["issues"]["1"] is not None


def test_full_scan_may_take_longer_than_one_minute(cfg, tmp_path, monkeypatch):
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        state = new_state(cfg["repo"])
        later_cursor = "2026-09-22T00:03:00+00:00"
        monkeypatch.setattr(p, "read_command", lambda *_: (snapshot([issue()], cursor=later_cursor), 180))
        monkeypatch.setattr(p.time, "time", lambda: NOW + 180)
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW)
        assert state["scan"]["cursor"] == later_cursor


def test_repeat_resume_incomplete_scan_does_not_drop_old_queue(cfg, tmp_path):
    with PipelineStore(tmp_path, cfg["repo"]) as store:
        state = init(cfg)
        store.save(state)
        # Simulate restart after durable ingest but before scope review.
        restarted = store.load()
        p.ingest(restarted, snapshot([issue("2")], complete=False), cfg, NOW + 10)
        store.save(restarted)
        again = store.load()
        assert set(again["issues"]) == {"1", "2"}
        assert not again["scan"]["complete"]


def test_natural_policy_change_invalidates_scope(cfg, tmp_path):
    state = init(cfg)
    accept_scope(state, cfg, tmp_path)
    cfg["responsibility"]["handle"] = ["仅处理A5"]
    p.synchronize(state, cfg, NOW + 10)
    assert state["issues"]["1"]["scope_cache_reason"] == "policy_changed"
    assert state["issues"]["1"]["phase"] == "scope"


def test_invalid_scope_result_can_be_corrected_in_same_attempt(cfg, tmp_path, monkeypatch, capsys):
    root, captured = setup_cli(tmp_path, cfg)
    monkeypatch.setattr(p.time, "time", lambda: NOW + 20)

    def args(command, *extra):
        return [command, "--repository-root", str(root), "--offline", *extra]

    p.run(p.parser().parse_args(args("resume", "--input", str(captured))))
    bundle = p.run(p.parser().parse_args(args("claim")))["task"]
    assert p.main(args("submit", "--claim-file", bundle["claim_file"])) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["next_action"] == "correct_result_file"
    with PipelineStore(root / p.RUNTIME, cfg["repo"]) as store:
        task = store.load()["tasks"][bundle["task_id"]]
    assert task["status"] == "running"
    assert task["attempt_id"] == bundle["attempt_id"]
    Path(bundle["result_file"]).write_text(
        json.dumps(
            {"decision": "handle", "summary": "构建问题属于范围", "evidence": ["问题正文"], "source_mode": "fixed"}
        )
    )
    p.run(p.parser().parse_args(args("submit", "--claim-file", bundle["claim_file"])))
    p.run(p.parser().parse_args(args("accept", "--claim-file", bundle["claim_file"])))
    with PipelineStore(root / p.RUNTIME, cfg["repo"]) as store:
        assert store.load()["tasks"][bundle["task_id"]]["status"] == "accepted"


def test_scan_checkpoint_is_session_scoped_and_new_session_refreshes(cfg, tmp_path, monkeypatch):
    state = init(cfg, items=[])
    state.update(needs_full_scan=False)
    state["scan"].update(complete=True, completed_at=NOW, full_completed_at=NOW, session_id="session-a")
    monkeypatch.setenv("CODEX_THREAD_ID", "session-a")
    monkeypatch.setattr(p.time, "time", lambda: NOW + 600)
    calls = []

    def fetch(*args):
        calls.append(args)
        return snapshot([]), 0.1

    monkeypatch.setattr(p, "read_command", fetch)
    with PipelineStore(tmp_path / p.RUNTIME, cfg["repo"]) as store:
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 600)
        assert calls == []  # Finishing a long pass must not restart its scope work.
        monkeypatch.setenv("CODEX_THREAD_ID", "session-b")
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 601)
        assert len(calls) == 1 and state["scan"]["session_id"] == "session-b"
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 602, force=True)
        assert len(calls) == 2
        state["scan"]["complete"] = False
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 603)
        assert len(calls) == 3


def test_host_without_session_id_keeps_time_based_refresh(cfg, tmp_path, monkeypatch):
    state = init(cfg, items=[])
    state.update(needs_full_scan=False)
    state["scan"].update(complete=True, completed_at=NOW, full_completed_at=NOW)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CODEX_SESSION_ID", raising=False)
    monkeypatch.setattr(p.time, "time", lambda: NOW + 301)
    calls = []

    def fetch(*args):
        calls.append(args)
        return snapshot([]), 0.1

    monkeypatch.setattr(p, "read_command", fetch)
    with PipelineStore(tmp_path / p.RUNTIME, cfg["repo"]) as store:
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 10)
        assert not calls
        p.refresh_remote(state, cfg, tmp_path / "config", tmp_path, store, NOW + 301)
        assert len(calls) == 1


def test_protocol_pages_preserve_all_waiting_items_without_mutating_full_state():
    original = {
        "next_action": "resume_waiting_work",
        "tasks": {"counts": {"waiting": 23}},
        "waiting_issues": [{"iid": str(i), "artifacts": ["large/" * 100]} for i in range(23)],
        "status_file": "/local/status.json",
    }
    before = copy.deepcopy(original)
    seen = []
    offset = 0
    while True:
        args = p.parser().parse_args(["status", "--section", "waiting_issues", "--offset", str(offset)])
        page = p.protocol_view(original, args)
        seen.extend(item["iid"] for item in page["items"])
        offset = page["next_offset"]
        if offset is None:
            break
    assert seen == [str(i) for i in range(23)]
    assert original == before
    args = p.parser().parse_args(["status", "--iid", "21"])
    assert p.protocol_view(original, args)["waiting_issues"] == [{"iid": "21"}]
    args = p.parser().parse_args(["status", "--details"])
    assert p.protocol_view(original, args)["waiting_issues"] == before["waiting_issues"]


def test_accept_receipt_does_not_grow_with_waiting_artifacts():
    def receipt(count):
        full = {
            "next_action": "resume_waiting_work",
            "counts": {"waiting": count},
            "tasks": {"counts": {"waiting": count}, "waiting": [{"iid": str(i)} for i in range(count)]},
            "waiting_issues": [{"iid": str(i), "artifacts": ["x" * 5000]} for i in range(count)],
            "task": {"iid": "1", "status": "accepted"},
            "status_file": "/status.json",
        }
        return json.dumps(p.protocol_view(full, p.parser().parse_args(["accept"])))

    assert len(receipt(1000)) < 2000
    assert len(receipt(1000)) - len(receipt(1)) < 20


def test_required_references_follow_stage_and_native_assignment_contract():
    scope = p.task_references({"stage": "scope"})
    assert [Path(v).name for v in scope] == ["responsibility-scope.md"]
    classification = {"auto_action": {"type": "assign_candidate", "response_requirement": "exempt_self_authored_pr"}}
    task = {"stage": "response", "payload": {"classification": classification}}
    assert [Path(v).name for v in p.task_references(task)] == ["automation.md"]
    classification["auto_action"]["response_requirement"] = "required"
    assert [Path(v).name for v in p.task_references(task)] == ["response-writing.md", "automation.md"]
    for path in scope + p.task_references(task):
        assert Path(path).is_absolute() and Path(path).is_file()
        assert Path(path).parent.parent == SCRIPTS.parent.resolve()


def test_nested_inspection_lists_can_be_paged_independently():
    records = [
        {"path": f"file{i}.md", "excerpt": "x" * 5000, "links": [{"target": str(j)} for j in range(17)]}
        for i in range(8)
    ]
    original = {"records": records, "reported_documents": [], "evidence_file": "/facts.json"}
    view = p.protocol_view(original, p.parser().parse_args(["inspect"]))
    assert len(view["records"]) == 5
    assert len(view["records"][0]["excerpt"]) == 2400
    assert view["records"][0]["excerpt_truncated"]
    args = p.parser().parse_args(["inspect", "--section", "records.file7.md.links", "--offset", "5"])
    assert [v["target"] for v in p.protocol_view(original, args)["items"]] == list(map(str, range(5, 10)))
    assert len(original["records"][0]["excerpt"]) == 5000


@pytest.mark.parametrize(
    "arguments",
    [
        ["--limit", "0"],
        ["--limit", "101"],
        ["--offset", "-1"],
        ["--lines", "0:5"],
        ["--lines", "5:4"],
        ["--lines", "1:161"],
    ],
)
def test_invalid_output_arguments_fail_before_runtime(arguments):
    with pytest.raises(BaseException) as caught:
        p.parser().parse_args(["inspect", *arguments])
    assert caught.value.code == 2


def test_details_respects_explicit_section_and_issue_filters():
    full = {
        "tasks": {"waiting": [{"iid": "1"}, {"iid": "2"}]},
        "waiting_issues": [{"iid": "1"}, {"iid": "2"}],
        "issue_detail": {
            "phase": "waiting",
            "issue": {"description": "x" * 40000},
            "response_file_map": {"reply": "/reply.md"},
        },
        "state_file": "/state.json",
    }
    selected = p.protocol_view(full, p.parser().parse_args(["status", "--iid", "2", "--details"]))
    assert selected["waiting_issues"] == [{"iid": "2"}]
    assert selected["tasks"]["waiting"] == [{"iid": "2"}]
    section = p.protocol_view(full, p.parser().parse_args(["status", "--section", "waiting_issues", "--details"]))
    assert section["items"] == full["waiting_issues"]
    assert "issue_detail" not in section
    compact = p.protocol_view(full, p.parser().parse_args(["status", "--iid", "2"]))
    assert compact["issue_detail"]["response_file_map"]["reply"] == "/reply.md"
    assert len(json.dumps(compact)) < 2000


def test_prepared_inventory_reports_only_existing_files(cfg, tmp_path):
    state = init(cfg)
    reply = tmp_path / "reply.md"
    reply.write_text("Draft")
    state["issues"]["1"].update(
        phase="waiting",
        prepared_response={"summary": "Ready"},
        response_file_map={"reply.md": str(reply), "assign.md": str(tmp_path / "absent.md")},
    )
    with PipelineStore(tmp_path / p.RUNTIME, cfg["repo"]) as store:
        summary = p.output_status(state, store, NOW)
    assert summary["prepared_items"] == [{"iid": "1", "files": ["reply.md"]}]
    assert p.protocol_view(summary, p.parser().parse_args(["status"]))["prepared_items"] == summary["prepared_items"]
    assert "prepared_items" not in p.protocol_view(summary, p.parser().parse_args(["accept"]))
