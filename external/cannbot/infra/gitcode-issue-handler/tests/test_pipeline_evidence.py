# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Facts and quality gates, including the live-evaluation false-dead-link regression."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline_evidence as pe
import pipeline_delivery as pd
import issue_pipeline as p
from pipeline_store import new_state, PipelineStore


@pytest.fixture
def facts(monkeypatch, tmp_path):
    paths = ["story/recipes/examples/quant/README.md", "story/docs/量化介绍.md", "kernel.h"]
    contents = {paths[0]: "[量化](../../../docs/量化介绍.md)\n[坏链](missing.md)\n", paths[2]: "void copy() {}"}

    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return b"a" * 40
        if args[0] == "ls-tree":
            return ("\0".join(paths) + "\0").encode()
        if args[0] == "log":
            return ("b" * 40 + "\tAlice\ta@example.test\tImplement copy\n").encode()
        if args[0] == "show":
            return contents.get(args[1].split(":", 1)[1]).encode()
        raise AssertionError(args)

    monkeypatch.setattr(pe, "git", fake_git)
    return pe.inspect_sources(tmp_path, "team/repo", [paths[0], paths[2]])


def test_relative_links_resolve_from_readme_not_repo_root(facts):
    links = facts["records"][0]["links"]
    assert links[0]["resolved"] == "story/docs/量化介绍.md"
    assert links[0]["status"] == "tracked_target_exists"
    assert links[1]["status"] == "target_missing_at_revision"
    assert facts["records"][0]["revision"] == "a" * 40
    assert facts["records"][0]["history"][0]["author_name"] == "Alice"
    assert "login" not in facts["records"][0]["history"][0]


def make_result(tmp_path, facts):
    task = {"task_id": "1/response/test", "attempt_id": "attempt1", "payload": {"issue": {"title": "README 死链"}}}
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts))
    state = {
        "source_evidence": {
            "facts.json": {
                "task_id": task["task_id"],
                "attempt_id": task["attempt_id"],
                "digest": pe.content_hash(facts),
            }
        }
    }
    result = {
        "inspection_files": ["facts.json"],
        "analysis": "分析文件和版本",
        "reply": "回复具体核查结果",
        "next_action": "等待提交版本",
        "source_verdict": "needs_version",
        "owner_review": {"status": "not_needed", "reason": "当前版本无法复现，先核对提交", "candidates": []},
        "response_review": {"checked_facts": ["目标文档存在"], "remaining_unknowns": ["报告提交"]},
    }
    return state, task, result


def test_unsubstantiated_dead_link_claim_is_rejected(tmp_path, facts):
    facts["records"][0]["links"] = facts["records"][0]["links"][:1]
    state, task, result = make_result(tmp_path, facts)
    result["source_verdict"] = "confirmed_at_revision"
    with pytest.raises(pd.MaterialError, match="response_materials_incomplete") as exc:
        pd.validate_response(state, task, result, tmp_path)
    assert any("事实冲突" in e for e in exc.value.errors)
    result["source_verdict"] = "not_reproduced_at_revision"
    assert pd.validate_response(state, task, result, tmp_path)


def test_missing_and_tampered_evidence_cannot_be_accepted(tmp_path, facts):
    state, task, result = make_result(tmp_path, facts)
    (tmp_path / "facts.json").write_text("{}")
    with pytest.raises(pd.MaterialError) as exc:
        pd.validate_response(state, task, result, tmp_path)
    assert any("修改" in e for e in exc.value.errors)
    result["inspection_files"] = []
    with pytest.raises(pd.MaterialError):
        pd.validate_response(state, task, result, tmp_path)


def test_owner_login_cannot_be_guessed_from_git_email(tmp_path, facts):
    state, task, result = make_result(tmp_path, facts)
    result["owner_review"] = {
        "status": "candidates",
        "operators": ["copy"],
        "reason": "实现贡献",
        "candidates": [{"login": "Alice"}],
    }
    with pytest.raises(pd.MaterialError) as exc:
        pd.validate_response(state, task, result, tmp_path)
    assert any("提交元数据" in e for e in exc.value.errors)


def test_directory_listing_cannot_replace_source_inspection(tmp_path, facts):
    facts["records"] = [{"path": "story", "kind": "directory", "tracked": True, "history": []}]
    state, task, result = make_result(tmp_path, facts)
    with pytest.raises(pd.MaterialError) as exc:
        pd.validate_response(state, task, result, tmp_path)
    assert any("至少检查一个" in e for e in exc.value.errors)


def test_claim_file_preserves_attempt_identity_without_manual_copy(tmp_path):
    state = new_state("team/repo")
    state["issues"]["1"] = {"issue": {"iid": "1"}, "phase": "scope"}
    p.queue.ensure_task(state, "1", "scope", "fingerprint", {"issue": {"iid": "1"}}, 1)
    task = p.queue.claim(state, "worker", 2)
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        bundle = p.claim_bundle(task, state, tmp_path, store)
    args = p.parser().parse_args(["submit", "--claim-file", bundle["claim_file"]])
    p.open_claim(args, state, tmp_path)
    assert args.task == task["task_id"] and args.attempt == task["attempt_id"]
    assert args.result_file == bundle["result_file"]
    p.queue.requeue(state, task["task_id"], "retry", 3)
    p.queue.claim(state, "worker", 4)
    with pytest.raises(ValueError, match="stale_claim_file"):
        p.open_claim(args, state, tmp_path)


def test_explicit_rejection_is_known_failure_not_unknown_write(tmp_path):
    state = new_state("team/repo")
    path = tmp_path / "reports/old/run_state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run": {"repository": "team/repo"},
                "external_operations": [
                    {
                        "operation_id": "old-push",
                        "kind": "branch_push",
                        "status": "failed",
                        "target": "origin/fix/old",
                        "execution_evidence": {"remote_http_result": "rejected"},
                    },
                    {
                        "operation_id": "uncertain",
                        "kind": "branch_push",
                        "status": "failed",
                        "target": "origin/other",
                        "execution_evidence": {"reason": "timeout"},
                    },
                ],
            }
        )
    )
    p.migrate(state, tmp_path, {"responsibility": {}})
    rejected = state["operations"]["legacy:old:old-push"]
    assert rejected["status"] == "verified" and rejected["outcome"] == "confirmed_rejected"
    assert rejected["target"] == "origin/fix/old"
    assert state["operations"]["legacy:old:uncertain"]["status"] == "needs_review"


def _response_case_state(cfg, now, response_requirement):
    from scope_cache import make_entry, task_digest
    from responsibility import policy_digest

    state = new_state("team/repo")
    state.update(migration_done=True, needs_full_scan=False)
    state["scan"] = {"id": "testscan", "complete": True, "attempted_at": now, "completed_at": now}
    issue = {
        "iid": "1",
        "number": "1",
        "title": "README 死链",
        "state": "open",
        "author": "reporter",
        "description": "README target not found",
        "url": "https://gitcode.com/team/repo/issues/1",
    }
    review = {
        "level": "handle",
        "summary": "具体文档问题",
        "evidence": ["报告正文"],
        "policy_digest": policy_digest(cfg["responsibility"]),
    }
    entry = {
        "issue": issue,
        "phase": "classify",
        "input_digest": task_digest(issue, cfg["responsibility"], set()),
        "scope": make_entry(issue, cfg["responsibility"], review),
    }
    state["issues"]["1"] = entry
    item = {"number": "1", "bucket": "need_attention", "category": "first_response", "responsibility": "handle"}
    if response_requirement:
        item["auto_action"] = {
            "type": "assign_candidate",
            "candidate": "alice",
            "candidates": ["alice"],
            "response_requirement": response_requirement,
        }
    only_assignment = pd.assignment_only(item)
    return state, entry, item, only_assignment


def _prepare_response_case(tmp_path, facts, monkeypatch, response_requirement):
    from handler_config import load_handler_config

    config_path = tmp_path / p.RUNTIME / "config/classify_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("repo: team/repo\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_handler_config(config_path)
    now = p.time.time()
    state, entry, item, only_assignment = _response_case_state(cfg, now, response_requirement)
    p.apply_classification(state, {"issues": [item]}, [entry], "classification.json", now)
    task = p.queue.claim(state, "worker", now)
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        bundle = p.claim_bundle(task, state, tmp_path, store)
        relative = str(Path(bundle["result_file"]).parent.relative_to(tmp_path) / "facts.json")
        (tmp_path / relative).write_text(json.dumps(facts))
        state["source_evidence"] = {
            relative: {"task_id": task["task_id"], "attempt_id": task["attempt_id"], "digest": pe.content_hash(facts)}
        }
        _, _, result = make_result(tmp_path, facts)
        result.update(
            decision="prepared", summary="核对版本", evidence=["Git tracked snapshot"], inspection_files=[relative]
        )
        # not_needed may carry descriptive metadata; it is not a candidate list.
        result["owner_review"]["operators"] = [{"name": "existing owner context"}]
        if response_requirement:
            result["assignment"] = {"login": "alice", "reason": "平台原生关联 PR 作者"}
        if only_assignment:
            result.update(reply="", inspection_files=[])
        Path(bundle["result_file"]).write_text(json.dumps(result))
        store.save(state)
    args = ["--repository-root", str(tmp_path), "--claim-file", bundle["claim_file"], "--offline"]
    return {
        "args": args,
        "cfg": cfg,
        "now": now,
        "item": item,
        "result": result,
        "only_assignment": only_assignment,
        "task": task,
    }


def _assert_response_revision(tmp_path, accepted, case):
    item, cfg, now = case["item"], case["cfg"], case["now"]
    result, args, task = case["result"], case["args"], case["task"]
    only_assignment = case["only_assignment"]
    # A new scan cannot present a previous draft before classification refresh.
    refreshed = copy.deepcopy(accepted)
    refreshed["scan"]["id"] = "nextscan"
    refreshed["issues"]["1"].pop("classification_scan_id")
    assert pd.report_state(refreshed, tmp_path, now)["issues"] == []
    p.apply_classification(refreshed, {"issues": [item]}, [refreshed["issues"]["1"]], "next.json", now)
    assert len(pd.report_state(refreshed, tmp_path, now)["issues"]) == 1
    # PR-only changes and changed issue contents both archive obsolete drafts.
    for reason in ("pr", "body"):
        changed = copy.deepcopy(accepted)
        changed_entry = changed["issues"]["1"]
        if reason == "pr":
            updated = dict(item, linked_prs=[{"number": 9, "state": "open"}])
            p.apply_classification(changed, {"issues": [updated]}, [changed_entry], "new.json", now)
        else:
            changed_entry["issue"]["description"] += " new evidence"
            p.synchronize(changed, cfg, now)
        assert "prepared_response" not in changed_entry
        assert changed_entry["response_history"][-1]["result"]["summary"] == result["summary"]
        assert not pd.report_state(changed, tmp_path, now)["issues"]
        assert "delivery_report" not in changed
        with pytest.raises(ValueError, match="stale_claim_file"):
            p.open_claim(p.parser().parse_args(["check", *args]), changed, tmp_path)
    original_reply = Path(accepted["issues"]["1"]["response_file_map"]["assign.md" if only_assignment else "reply.md"])
    before = original_reply.read_text()
    p.run(
        p.parser().parse_args(
            ["revise", "--iid", "1", "--reason", "事实需要修正", "--repository-root", str(tmp_path), "--offline"]
        )
    )
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        revised = store.load()
    assert revised["issues"]["1"]["response_revision"] == 1
    assert "prepared_response" not in revised["issues"]["1"]
    assert any(t["status"] == "ready" for t in revised["tasks"].values())
    assert original_reply.read_text() == before
    assert revised["tasks"][task["task_id"]]["status"] == "superseded"


@pytest.mark.parametrize("response_requirement", [None, "exempt_self_authored_pr", "satisfied", "required"])
def test_response_accept_generates_strict_report_and_revision_preserves_original(
    tmp_path, facts, monkeypatch, response_requirement
):
    case = _prepare_response_case(tmp_path, facts, monkeypatch, response_requirement)
    args, cfg, now, item = (case[key] for key in ("args", "cfg", "now", "item"))
    result, only_assignment, task = (case[key] for key in ("result", "only_assignment", "task"))
    p.run(p.parser().parse_args(["submit", *args]))
    output = p.run(p.parser().parse_args(["accept", *args]))
    report = Path(output["delivery_report"])
    assert report.is_file()
    report_json = json.loads((report.parent / "_internal/run_state.json").read_text())
    assert report_json["run"]["report_generated"] is True
    assert report_json["issues"][0]["response_status"] == (
        "exempt_self_authored_pr" if response_requirement == "exempt_self_authored_pr" else "prepared"
    )
    assert bool(report_json["issues"][0]["related_code"]) is (not only_assignment)
    if only_assignment:
        assert "reply.md" not in report_json["issues"][0]["response_artifacts"]
    if response_requirement:
        assignment_file = Path(report_json["issues"][0]["response_artifacts"]["assign.md"])
        assert assignment_file.read_text() == "/assign @alice\n"
    assert "owner_candidate_analysis" not in report_json["issues"][0]
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        accepted = store.load()
    case = dict(item=item, cfg=cfg, now=now, result=result, args=args, task=task, only_assignment=only_assignment)
    _assert_response_revision(tmp_path, accepted, case)


def test_reported_document_lookup_covers_whole_tree(monkeypatch, tmp_path):
    files = ["Features/memory/指南.md", "Performance/example/README.md"]

    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return b"a" * 40
        if args[0] == "ls-tree":
            return ("\0".join(files) + "\0").encode()
        if args[0] == "show":
            return b"# Sample"
        if args[0] == "log":
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(pe, "git", fake_git)
    data = pe.inspect_sources(
        tmp_path,
        "team/repo",
        ["Performance/example/README.md"],
        reported_text="`.../memory/指南.md` 和 `missing.md` 不存在",
    )
    found = {r["name"]: r for r in data["reported_documents"]}
    assert found["指南.md"]["matches"] == ["Features/memory/指南.md"]
    assert found["missing.md"]["match_count"] == 0


def test_assignment_preparation_uses_verified_classifier_candidate(tmp_path):
    classification = {
        "auto_action": {
            "type": "assign_candidate",
            "candidate": "author",
            "candidates": ["author", "other"],
            "response_requirement": "exempt_self_authored_pr",
        }
    }
    result = pd.response_template(classification)
    result.update(analysis="作者已有有效 PR，仅补分配", next_action="等待审核分配")
    result["assignment"]["reason"] = "自提 PR 作者优先"
    result["response_review"]["checked_facts"] = ["有效关联 PR 作者是 author"]
    task = {"payload": {"classification": classification}}
    assert pd.validate_response({}, task, result, tmp_path) == []
    result["assignment"]["login"] = "other"
    with pytest.raises(pd.MaterialError):
        pd.validate_response({}, task, result, tmp_path)
    result["assignment"]["login"] = "author"
    result["reply"] = "多余的首响"
    with pytest.raises(pd.MaterialError):
        pd.validate_response({}, task, result, tmp_path)


def test_malformed_review_fields_fail_before_report(tmp_path, facts):
    state, task, result = make_result(tmp_path, facts)
    result["owner_review"].update(status="unresolved", operators=[{"name": "kernel"}])
    with pytest.raises(pd.MaterialError):
        pd.validate_response(state, task, result, tmp_path)
    result["owner_review"].update(status="not_needed")
    result["source_verdict"] = []
    with pytest.raises(pd.MaterialError):
        pd.validate_response(state, task, result, tmp_path)
    result["source_verdict"] = "needs_version"
    result["response_review"]["checked_facts"] = {"unsupported": "shape"}
    with pytest.raises(pd.MaterialError):
        pd.validate_response(state, task, result, tmp_path)


@pytest.mark.parametrize(
    "owner",
    [
        {"status": "candidates", "operators": ["kernel"], "reason": "证据", "candidates": [{"login": ["bad"]}]},
        {"status": "unresolved", "operators": ["kernel"], "reason": "缺身份", "candidates": [{"operators": [{}]}]},
    ],
)
def test_invalid_candidate_shapes_return_material_errors(tmp_path, facts, owner):
    state, task, result = make_result(tmp_path, facts)
    result["owner_review"] = owner
    with pytest.raises(pd.MaterialError):
        pd.validate_response(state, task, result, tmp_path)


def test_inspect_explicit_line_range_preserves_fixed_revision_and_full_links(monkeypatch, tmp_path):
    content = "\n".join(f"line {i}" for i in range(1, 301)) + "\n[late](late.md)"

    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return b"a" * 40
        if args[0] == "ls-tree":
            return b"file.md\0late.md\0"
        if args[0] == "show":
            return content.encode()
        if args[0] == "log":
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(pe, "git", fake_git)
    facts = pe.inspect_sources(tmp_path, "team/repo", ["file.md"], lines=(161, 180))
    record = facts["records"][0]
    assert record["excerpt"].startswith("161: line 161")
    assert record["excerpt"].endswith("180: line 180")
    assert record["excerpt_range"] == {"start": 161, "end": 180}
    assert record["excerpt_truncated"]
    assert record["links"][0]["status"] == "tracked_target_exists"
    assert record["revision"] == "a" * 40
    for paths, lines in [
        (["file.md"], (999, 1000)),
        (["file.md", "late.md"], (1, 2)),
        (["absent"], (1, 2)),
        (["file.md"], (1, 161)),
    ]:
        with pytest.raises(ValueError):
            pe.inspect_sources(tmp_path, "team/repo", paths, lines=lines)


def test_existing_evidence_paging_does_not_collect_or_rewrite_state(tmp_path, facts, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / p.RUNTIME / "config/classify_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("repo: team/repo\n")
    monkeypatch.setattr(p, "load_handler_config", lambda path: {"repo": "team/repo"})
    state = new_state("team/repo")
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts))
    state["source_evidence"] = {"facts.json": {"digest": pe.content_hash(facts)}}
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        store.save(state)
        before = store.path.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("existing evidence must not collect, synchronize or migrate")

    monkeypatch.setattr(p, "inspect_sources", unexpected)
    monkeypatch.setattr(p, "synchronize", unexpected)
    monkeypatch.setattr(p, "migrate", unexpected)
    args = p.parser().parse_args(
        [
            "inspect",
            "--repository-root",
            str(tmp_path),
            "--evidence-file",
            str(path),
            "--section",
            "records",
            "--offset",
            "1",
            "--offline",
        ]
    )
    result = p.run(args)
    assert p.protocol_view(result, args)["items"] == [facts["records"][1]]
    assert "detail_command" in result
    assert (tmp_path / p.RUNTIME / "data/pipeline-state.json").read_bytes() == before
    path.write_text("{}")
    with pytest.raises(ValueError, match="modified_evidence_file"):
        p.run(args)


def test_material_links_deduplicate_same_revision_but_keep_distinct_versions(tmp_path):
    record = {"path": "kernel.h", "url": "https://example.test/a/kernel.h", "kind": "file", "revision": "a" * 40}
    second = {**record, "revision": "b" * 40, "url": "https://example.test/b/kernel.h"}
    result = pd.response_template()
    result["analysis"] = "Static source review"
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        files, related = pd.materialize(result, [record, record, second], store, Path("reports/test"))
    assert related == [record, second]
    assert Path(files["analysis.md"]).read_text().count("https://example.test/a/kernel.h") == 1


def test_response_claim_discloses_validator_enums_and_next_command_preserves_context(tmp_path):
    state = new_state("team/repo")
    p.queue.ensure_task(state, "1", "response", "digest", {"issue": {"iid": "1"}, "classification": {}}, 1)
    task = p.queue.claim(state, "worker", 2)
    with PipelineStore(tmp_path / p.RUNTIME, "team/repo") as store:
        bundle = p.claim_bundle(task, state, tmp_path, store)
    contract = json.loads(Path(bundle["task_file"]).read_text())["allowed_values"]
    assert contract["source_verdict"] == list(pd.SOURCE_VERDICTS)
    assert contract["owner_review.status"] == list(pd.OWNER_STATUSES)
    args = p.parser().parse_args(
        ["check", "--claim-file", bundle["claim_file"], "--offline", "--config", str(tmp_path / "custom.yaml")]
    )
    command = p.command_argv("submit", tmp_path, args)
    parsed = p.parser().parse_args(command[2:])
    assert parsed.command == "submit" and parsed.offline
    assert parsed.claim_file == bundle["claim_file"]
    assert parsed.config == str(tmp_path / "custom.yaml")


def test_next_submit_preserves_checked_result_override(tmp_path):
    result = tmp_path / "corrected.json"
    args = p.parser().parse_args(["check", "--claim-file", str(tmp_path / "claim.json"), "--result-file", str(result)])
    followup = p.parser().parse_args(p.command_argv("submit", tmp_path, args)[2:])
    assert followup.result_file == str(result)
