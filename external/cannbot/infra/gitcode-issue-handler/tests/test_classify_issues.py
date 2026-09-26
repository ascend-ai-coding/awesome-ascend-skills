# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import io
import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "classify_issues.py"
SPEC = importlib.util.spec_from_file_location("classify_issues", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)


def test_classifier_cache_resolves_shared_rate_limit_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "issues"

    assert CLASSIFIER.rate_limit_path(cache_dir) == cache_dir.parent / "gitcode-rate-limit"


def classification_options(issue_pr_map=None, **overrides):
    values = {
        "issue_pr_map": issue_pr_map or {},
        "post_fn": lambda *_: True,
        "dry_run": True,
        **overrides,
    }
    return CLASSIFIER.ClassificationOptions(**values)


def linkage_options(**overrides):
    values = {
        "api_base": "https://api.example.test",
        "repo": "cann/ops-math",
        "token": "token",
        **overrides,
    }
    return CLASSIFIER.LinkageOptions(**values)


def classify(issue_data: dict) -> dict:
    return CLASSIFIER.classify_one(issue_data, classification_options())


def comment(
    author="maintainer",
    body="已受理，正在核对",
    *,
    comment_id=1,
    created_at="2026-08-14T02:11:02Z",
    raw_user=False,
) -> dict:
    author_field = {"user": {"login": author}} if raw_user else {"author": author}
    return {
        "id": comment_id,
        **author_field,
        "body": body,
        "created_at": created_at,
    }


def issue(**overrides) -> dict:
    values = {
        "number": 1,
        "state": "open",
        "author": "reporter",
        "assignee": None,
        "comments": [],
    }
    values.update(overrides)
    return values


def pull_request(number=4487, **overrides) -> dict:
    values = {
        "number": number,
        "state": "open",
        "body": "",
        "title": f"PR {number}",
        "head": {"ref": f"feature-{number}"},
        "user": {"login": "developer"},
    }
    values.update(overrides)
    return values


def assert_classification(issue_data, bucket, category, *, options=None):
    result = CLASSIFIER.classify_one(
        issue_data,
        options or classification_options(),
    )
    assert (result["bucket"], result["category"]) == (bucket, category)
    return result


def followup_comments() -> list[dict]:
    return [
        comment(
            body="请补充完整日志。",
            comment_id=10,
            created_at="2026-08-07T08:00:00Z",
        ),
        comment(
            "reporter",
            "日志已经补充，请继续看下。",
            comment_id=11,
            created_at="2026-08-13T08:00:00Z",
        ),
    ]


def followup_issue(**overrides) -> dict:
    values = issue(
        number=2535,
        assignee="maintainer",
        state="open",
        issue_state="挂起",
        comments=followup_comments(),
        fetch_sources=["updated"],
    )
    values.update(overrides)
    return values


def assignee_watch(**overrides) -> dict:
    values = {
        "conversation_state": "awaiting_assignee",
        "assignee": "operator-owner",
        "last_maintainer_comment_id": 184760205,
        "last_maintainer_comment_at": "2026-08-14T02:11:02Z",
    }
    values.update(overrides)
    return values


def assignee_conversation_issue(*extra_comments: dict, **overrides) -> dict:
    values = issue(
        number=2617,
        assignee="operator-owner",
        state="open",
        issue_state="挂起",
        comments=[
            comment(
                "triager",
                "已联系算子责任人，请稍等",
                comment_id=184760205,
            ),
            *extra_comments,
        ],
    )
    values.update(overrides)
    return values


CANONICAL_RUNTIME_PATHS = {
    "last_check_file": ".cannbot/gitcode-issue-handler/data/last_check.json",
    "report_file": ".cannbot/gitcode-issue-handler/reports/classification.txt",
    "cache_dir": ".cannbot/gitcode-issue-handler/cache/issues",
}
LEGACY_RUNTIME_PATHS = {
    "last_check_file": "issue_analysis_data/last_check.json",
    "report_file": "issue_analysis_data/classify_report.txt",
    "cache_dir": "issue_analysis_data/cache",
}


class TestRuntimePaths:
    @staticmethod
    def _assert_paths(cfg: dict, expected: dict) -> None:
        assert {key: cfg[key] for key in expected} == expected

    @staticmethod
    def _write(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @classmethod
    def _load_unchanged(
        cls,
        path: Path,
        values: dict,
        *,
        requested: str | None = None,
        **load_options,
    ) -> dict:
        source = cls._write(
            path,
            "".join(f"{key}: {value}\n" for key, value in values.items()),
        )
        before = source.read_text(encoding="utf-8")
        cfg = CLASSIFIER.load_config(requested, **load_options)
        assert source.read_text(encoding="utf-8") == before
        return cfg

    @pytest.fixture(autouse=True)
    def _run_from_tmp_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

    def test_canonical_config_wins_when_legacy_also_exists(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "classify_config.yaml", "repo: legacy/repo\n")
        self._write(
            tmp_path / CLASSIFIER.DEFAULT_CONFIG_FILE,
            "repo: canonical/repo\n",
        )

        cfg = CLASSIFIER.load_config(CLASSIFIER.DEFAULT_CONFIG_FILE)

        assert cfg["repo"] == "canonical/repo"
        self._assert_paths(cfg, CANONICAL_RUNTIME_PATHS)

    def test_legacy_config_is_read_but_former_defaults_write_to_new_tree(
        self, tmp_path: Path
    ) -> None:
        cfg = self._load_unchanged(
            tmp_path / "classify_config.yaml",
            {"repo": "legacy/repo", **LEGACY_RUNTIME_PATHS},
        )

        assert cfg["repo"] == "legacy/repo"
        self._assert_paths(cfg, CANONICAL_RUNTIME_PATHS)

    def test_custom_legacy_output_paths_remain_authoritative(
        self, tmp_path: Path
    ) -> None:
        cfg = self._load_unchanged(
            tmp_path / "classify_config.yaml",
            {"repo": "legacy/repo", "report_file": "custom/report.txt"},
        )

        assert cfg["report_file"] == "custom/report.txt"

    def test_explicit_config_never_falls_back_to_legacy(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "classify_config.yaml", "repo: legacy/repo\n")
        custom = tmp_path / "custom/selected.yaml"
        cfg = self._load_unchanged(
            tmp_path / "custom/selected.yaml",
            {
                "repo": "explicit/repo",
                "report_file": LEGACY_RUNTIME_PATHS["report_file"],
            },
            requested=str(custom.relative_to(tmp_path)),
        )

        assert cfg["repo"] == "explicit/repo"
        assert cfg["report_file"] == LEGACY_RUNTIME_PATHS["report_file"]

    def test_explicit_canonical_path_does_not_enable_legacy_fallback(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path / "classify_config.yaml",
            "repo: legacy/repo\nreport_file: custom/legacy.txt\n",
        )

        with pytest.raises(ValueError, match="config file does not exist"):
            CLASSIFIER.load_config(
                CLASSIFIER.DEFAULT_CONFIG_FILE,
                repo_override="cli/repo",
                allow_legacy=False,
            )


    def test_explicit_canonical_config_does_not_redirect_legacy_values(
        self, tmp_path: Path
    ) -> None:
        cfg = self._load_unchanged(
            tmp_path / CLASSIFIER.DEFAULT_CONFIG_FILE,
            {"repo": "explicit/repo", **LEGACY_RUNTIME_PATHS},
            requested=CLASSIFIER.DEFAULT_CONFIG_FILE,
            allow_legacy=False,
        )

        self._assert_paths(cfg, LEGACY_RUNTIME_PATHS)


class TestEffectiveComments:
    @pytest.mark.parametrize(
        "issue_data,bucket,category",
        [
            (
                issue(
                    number=2552,
                    author="LS_462ss",
                    comments=[comment("LS_462ss", "补充源码基线和适配范围")],
                ),
                "need_attention",
                "needs_first_look",
            ),
            (
                issue(comments=[comment()]),
                "no_attention",
                "replied_no_owner",
            ),
            (
                issue(comments=[comment("unknown", "some text")]),
                "need_attention",
                "needs_first_look",
            ),
            (
                issue(
                    comments=[
                        comment("reporter", "补充一些背景"),
                        comment(body="/assign @owner"),
                    ]
                ),
                "need_attention",
                "needs_only_assign_cmd",
            ),
            (
                issue(
                    author="Reporter",
                    assignee="owner",
                    comments=[comment("reporter", "补充复现信息", raw_user=True)],
                ),
                "need_attention",
                "our_team_needs_work",
            ),
        ],
        ids=["author", "maintainer", "unknown", "assign-only", "raw-api"],
    )
    def test_comment_evidence(self, issue_data, bucket, category) -> None:
        assert_classification(issue_data, bucket, category)


class TestResponseScope:
    @staticmethod
    @pytest.mark.parametrize("body,bucket", [
        ("当前接口传入空张量后报错，能否先确认如何绕过？", "need_attention"),
        ("请问接口是否支持空张量？", "need_attention"),
        ("计划优化接口；当前接口仍然报错，请帮忙确认原因。", "need_attention"),
        ("- [ ] 计划修复空张量报错。\n- [ ] 确认新增接口的支持范围。", "no_attention"),
        ("计划确认是否支持空张量，以及如何改善错误提示。", "no_attention"),
    ])
    def test_planning_body_distinguishes_requests_from_planned_work(body, bucket):
        source = issue(title="Roadmap", description=f"## 总体方向\n{body}\n## 详细计划\n待办事项。")
        assert classify(source)["bucket"] == bucket

    @staticmethod
    @pytest.mark.parametrize("scan_complete", [False, True])
    def test_pure_planning_is_exempt_despite_comment_updates_or_scan_failure(scan_complete):
        source = followup_issue(title="Roadmap", description="## 详细计划\n计划扩展算子覆盖。")
        assert_classification(source, "no_attention", "planning_record", options=classification_options(
            comment_scan_complete=scan_complete, association_scan_complete=False,
        ))

    @staticmethod
    def test_pure_planning_does_not_require_responsibility_investigation():
        source = issue(title="Roadmap", description="## 详细计划\n计划扩展算子覆盖。")
        result = CLASSIFIER.classify_with_responsibility(source, classification_options(), {})
        assert (result["bucket"], result["category"]) == ("no_attention", "planning_record")
        assert result["responsibility"] == "pending"

    @staticmethod
    def test_planning_record_is_not_a_feedback_response() -> None:
        source = issue(
            title="Development Roadmap (2026 Q3)",
            description="# 2026 Q3 Roadmap\n## 总体方向\n算子适配、接口与工具。\n## 详细计划\n待办事项。",
            assignee="reporter", comments=[comment(body="/assign @reporter")],
        )
        assert_classification(source, "no_attention", "planning_record")

    @staticmethod
    @pytest.mark.parametrize("type_fields", [
        {"title": "[Requirement|需求建议]: [Roadmap] 使用 tensorapi 优化 cube 算子"},
        {"title": "增强日志接口", "labels": [{"name": "需求建议"}]},
        {"title": "增强日志接口", "issue_type": {"name": "Requirement"}},
        {"title": "增强日志接口", "description": "### 需求描述\n增加日志字段。\n### 验收标准\n示例验证通过。"},
    ])
    @pytest.mark.parametrize("assignee,bucket,category", [
        ("reporter", "no_attention", "self_assigned"),
        ("someone-else", "need_attention", "our_team_needs_work"),
    ])
    def test_requirement_uses_account_identity_not_type_exemption(
        type_fields, assignee, bucket, category,
    ) -> None:
        source = issue(assignee=assignee, description="增强接口能力并完成验证。", **{
            key: value for key, value in type_fields.items() if key != "description"
        })
        source.update(type_fields)
        assert_classification(source, bucket, category)

    @staticmethod
    @pytest.mark.parametrize("fields", [
        {"title": "[Bug-Report|缺陷反馈]: Hist V2 确定性路径性能不达标，请优化"},
        {"title": "Roadmap parser crashes on empty input"},
        {"title": "[Requirement|需求建议]: 修复算子崩溃"},
        {"title": "Development Roadmap (2026 Q3)", "labels": ["bug"]},
        {"title": "[Requirement|需求建议]: 优化接口", "description": "### 问题描述\n输入 shape 后抛出异常。\n### 需求描述\n修复。\n### 验收标准\n通过。"},
        {"title": "普通反馈：接口性能需要优化"},
    ])
    def test_other_assignee_and_roadmap_mentions_do_not_exempt_defects(fields) -> None:
        source = issue(assignee="other-owner", description="## 总体方向\n待办事项。")
        source.update(fields)
        assert_classification(source, "need_attention", "our_team_needs_work")

    @staticmethod
    def test_unassigned_requirement_still_needs_response_and_assignment() -> None:
        assert_classification(
            issue(title="[Requirement|需求建议]: 扩展 genop", description="增加生成模板。"),
            "need_attention", "needs_first_look",
        )

    @staticmethod
    def test_requirement_does_not_hide_new_reporter_question() -> None:
        source = followup_issue(
            title="[Requirement|需求建议]: 增强日志", description="增加日志字段。",
        )
        assert_classification(source, "need_attention", "reporter_followup")

    @staticmethod
    def test_requirement_does_not_hide_new_assignee_reply() -> None:
        source = assignee_conversation_issue(
            comment("operator-owner", "已实现接口，请确认使用方式。", comment_id=184760300,
                    created_at="2026-08-14T04:00:00Z"),
            title="[Requirement|需求建议]: 增强日志", description="增加日志字段。",
        )
        assert_classification(source, "need_attention", "assignee_followup")


class TestFollowupConversation:
    @staticmethod
    def test_reporter_reply_after_maintainer_is_actionable() -> None:
        issue_data = followup_issue()

        result = classify(issue_data)
        conversation = CLASSIFIER.analyze_conversation(issue_data)

        assert result["bucket"] == "need_attention"
        assert result["category"] == "reporter_followup"
        assert conversation["activate_required"] is True
        assert conversation["pending_since"] == "2026-08-13T08:00:00Z"

    @staticmethod
    def test_closed_terminal_issue_requires_reopen_and_activation() -> None:
        issue_data = followup_issue(state="closed", issue_state="已完成")

        result = classify(issue_data)
        conversation = CLASSIFIER.analyze_conversation(issue_data)

        assert result["category"] == "reopened_followup"
        assert conversation["reopen_required"] is True
        assert conversation["activate_required"] is True

    @staticmethod
    def test_watched_issue_without_reporter_reply_stays_non_actionable() -> None:
        issue_data = followup_issue(
            number=42,
            comments=followup_comments()[:1],
            followup_watch={
                "last_maintainer_comment_id": 10,
                "last_maintainer_comment_at": "2026-08-07T08:00:00Z",
            },
        )

        result = classify(issue_data)
        conversation = CLASSIFIER.analyze_conversation(issue_data)

        assert result["bucket"] == "no_attention"
        assert result["category"] == "awaiting_reporter"
        assert conversation["waiting_status_reconcile_required"] is False

    @staticmethod
    def test_reporter_watch_repairs_a_drifted_custom_status() -> None:
        issue_data = followup_issue(
            number=42,
            issue_state="进行中",
            comments=followup_comments()[:1],
            followup_watch={
                "conversation_state": "awaiting_reporter",
                "last_maintainer_comment_id": 10,
                "last_maintainer_comment_at": "2026-08-07T08:00:00Z",
            },
        )

        assert_classification(issue_data, "need_attention", "awaiting_reporter_setup")

    @staticmethod
    @pytest.mark.parametrize("custom_state", ["挂起", "进行中"])
    def test_answered_reporter_watch_does_not_restore_old_wait(custom_state):
        source = followup_issue(issue_state=custom_state, followup_watch={
            "conversation_state": "awaiting_reporter",
            "last_maintainer_comment_id": 10,
            "last_maintainer_comment_at": "2026-08-07T08:00:00Z",
        })
        source["comments"].append(comment(
            body="日志已确认，原因是输入格式不符，请按示例调整。", comment_id=12,
            created_at="2026-08-14T08:00:00Z",
        ))
        assert_classification(source, "no_attention", "our_team_replied")
        assert source["followup_watch"]["last_maintainer_comment_id"] == 10

    @staticmethod
    def test_historical_handoff_does_not_require_watch_or_status_backfill() -> None:
        issue_data = assignee_conversation_issue(
            issue_state="进行中",
            fetch_sources=["updated"],
        )

        result = classify(issue_data)
        conversation = CLASSIFIER.analyze_conversation(issue_data)

        assert result["bucket"] == "no_attention"
        assert result["category"] == "awaiting_assignee"
        assert conversation["state"] == "awaiting_assignee"
        assert conversation["waiting_on"] == "assignee"
        assert conversation["waiting_watch_required"] is False
        assert conversation["waiting_status_reconcile_required"] is False

    @staticmethod
    @pytest.mark.parametrize("with_watch", [False, True])
    @pytest.mark.parametrize(
        ("reporter_reply_at", "bucket", "category"),
        [
            (None, "no_attention", "awaiting_assignee"),
            ("2026-08-14T03:00:00Z", "no_attention", "awaiting_assignee"),
            ("2026-08-14T06:00:00Z", "need_attention", "reporter_followup"),
        ],
    )
    def test_historical_handoff_uses_latest_reply(reporter_reply_at, bucket, category, with_watch) -> None:
        source = assignee_conversation_issue(
            comment("operator-owner", "我来处理修复。", comment_id=184760300,
                    created_at="2026-08-14T04:00:00Z"),
            comment("triager", "已收到，请继续按确认的方案处理。", comment_id=184760400,
                    created_at="2026-08-14T05:00:00Z"),
            issue_state="进行中",
        )
        if with_watch:
            source["followup_watch"] = assignee_watch()
        if reporter_reply_at:
            source["comments"].append(comment(
                "reporter", "补充一下复现日志。", comment_id=184760250,
                created_at=reporter_reply_at,
            ))

        assert_classification(source, bucket, category)
        if bucket == "need_attention":
            conversation = CLASSIFIER.analyze_conversation(source)
            assert conversation["pending_since"] == reporter_reply_at

    @staticmethod
    def test_assignee_watch_stays_suspended_until_assignee_replies() -> None:
        issue_data = assignee_conversation_issue(
            followup_watch=assignee_watch(waiting_since="2026-08-14T02:11:02Z")
        )

        assert_classification(issue_data, "no_attention", "awaiting_assignee")

    @staticmethod
    def test_reporter_reply_interrupts_an_assignee_watch() -> None:
        issue_data = assignee_conversation_issue(
            comment(
                "reporter",
                "我这里再补充一个现象。",
                comment_id=184760250,
                created_at="2026-08-14T03:00:00Z",
            ),
            followup_watch=assignee_watch(),
        )

        assert_classification(issue_data, "need_attention", "reporter_followup")

    @staticmethod
    def test_assignee_reply_reactivates_suspended_issue() -> None:
        issue_data = assignee_conversation_issue(
            comment(
                "operator-owner",
                "已确认原因，我来补充处理结论。",
                comment_id=184760300,
                created_at="2026-08-14T04:00:00Z",
            ),
            followup_watch=assignee_watch(),
        )

        result = classify(issue_data)
        conversation = CLASSIFIER.analyze_conversation(issue_data)

        assert result["bucket"] == "need_attention"
        assert result["category"] == "assignee_followup"
        assert conversation["activate_required"] is True

    @staticmethod
    def test_historical_assignee_reply_is_not_hidden_without_a_watch() -> None:
        issue_data = assignee_conversation_issue(
            comment(
                "operator-owner",
                "我已经给出处理建议。",
                comment_id=184760300,
                created_at="2026-08-14T04:00:00Z",
            ),
            fetch_sources=["updated"],
        )

        assert_classification(issue_data, "need_attention", "assignee_followup")

    @staticmethod
    def test_ordinary_acknowledgement_does_not_imply_assignee_wait() -> None:
        issue_data = issue(
            number=2618,
            assignee="maintainer",
            state="open",
            issue_state="进行中",
            comments=[comment(body="已受理，正在排查问题。")],
        )

        assert classify(issue_data)["category"] == "our_team_replied"

    @staticmethod
    def test_followup_overrides_active_pr_and_self_assignment() -> None:
        active_pr = {"2535": [{"pr_state": "open", "pr_merged": False}]}

        assert_classification(
            followup_issue(assignee="reporter"),
            "need_attention",
            "reporter_followup",
            options=classification_options(issue_pr_map=active_pr),
        )

    @staticmethod
    def test_comment_timestamps_control_turn_order() -> None:
        issue_data = followup_issue(
            assignee=None,
            comments=list(reversed(followup_comments())),
        )

        assert classify(issue_data)["category"] == "reporter_followup"

    @staticmethod
    def test_followup_is_sorted_before_ordinary_first_look() -> None:
        items = [
            {
                "category": "needs_first_look",
                "first_response_sla": "breached",
                "created_at": "2026-08-01T00:00:00Z",
            },
            {
                "category": "reporter_followup",
                "followup_sla": "pending",
                "created_at": "2026-08-13T00:00:00Z",
            },
        ]

        items.sort(key=CLASSIFIER.attention_sort_key)

        assert items[0]["category"] == "reporter_followup"

    @pytest.mark.parametrize(
        "body",
        [
            "已经联系相关责任人，请稍等。",
            "已经转交给算子负责人处理。",
            "Forwarded to the operator owner for analysis.",
        ],
    )
    def test_explicit_assignee_wait_phrases_are_recognized(self, body: str) -> None:
        issue_data = issue(
            number=2619,
            assignee="operator-owner",
            state="open",
            issue_state="进行中",
            comments=[comment("triager", body)],
        )

        assert classify(issue_data)["category"] == "awaiting_assignee"


class TestCommentFetchPolicy:
    @staticmethod
    def test_comment_fetch_failure_blocks_external_action() -> None:
        result = CLASSIFIER.classify_one(
            issue(number=4),
            classification_options(
                dry_run=False,
                comment_scan_complete=False,
            ),
        )

        assert result["bucket"] == "need_attention"
        assert result["category"] == "comment_scan_incomplete"
        assert result["auto_action"] is None

    @pytest.mark.parametrize(
        "issue_data,pr_map,expected",
        [
            (
                issue(author="developer", assignee="Developer"),
                {},
                (True, "classification_required"),
            ),
            (
                issue(number=2),
                {"2": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                (True, "first_response_required"),
            ),
            (
                issue(number=2, author="developer"),
                {"2": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                (True, "first_response_required"),
            ),
            (
                issue(number=3, assignee="owner"),
                {},
                (True, "classification_required"),
            ),
            (
                issue(number=2, assignee="owner", fetch_sources=["updated"]),
                {"2": [{"pr_state": "open", "pr_merged": False}]},
                (True, "followup_detection_required"),
            ),
        ],
        ids=["self-assigned", "active-pr", "self-authored-pr", "assigned", "followup"],
    )
    def test_fetch_policy(self, issue_data, pr_map, expected) -> None:
        assert CLASSIFIER.should_fetch_comments(issue_data, pr_map) == expected


class TestTimeScope:
    @pytest.mark.parametrize(
        "raw,ignore_last_check,expected",
        [
            (
                {"filters": {"since": "2026-08-04"}, "issues": []},
                False,
                ("2026-08-04T00:00:00+08:00", False, "fetch_since"),
            ),
            ({"issues": []}, True, (None, False, "full_input")),
            (
                {"filters": {"mode": "single"}, "issues": []},
                False,
                (None, False, "single_issue"),
            ),
        ],
        ids=["fetch-since", "complete-input", "single-issue"],
    )
    def test_scope(self, raw, ignore_last_check, expected) -> None:
        assert CLASSIFIER.resolve_time_scope(
            raw,
            ignore_last_check,
            "unused.json",
            "cann/ops-math",
        ) == expected


class TestProcessingMode:
    @staticmethod
    def test_explicit_single_issue_cannot_be_silently_skipped() -> None:
        routed = CLASSIFIER.apply_processing_mode(
            {
                "bucket": "no_attention",
                "category": "our_team_replied",
                "reason": "already replied",
                "auto_action": None,
            },
            single_mode=True,
        )

        assert routed["bucket"] == "need_attention"
        assert routed["classification_bucket"] == "no_attention"
        assert routed["must_handle"] is True
        assert routed["single_issue_override"] is True

    @staticmethod
    def test_batch_mode_preserves_classifier_bucket() -> None:
        routed = CLASSIFIER.apply_processing_mode(
            {
                "bucket": "no_attention",
                "category": "self_assigned",
                "reason": "self assigned",
                "auto_action": None,
            },
            single_mode=False,
        )

        assert routed["bucket"] == "no_attention"
        assert routed["must_handle"] is False
        assert routed["single_issue_override"] is False


class TestAuthorizationMode:
    @staticmethod
    def test_default_mode_is_interactive() -> None:
        assert CLASSIFIER.parse_args([]).authorization_mode == "interactive"

    @staticmethod
    def test_approved_batch_is_rejected_for_single_issue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="valid only for batch"):
            TestAuthorizationMode._runtime(
                tmp_path,
                monkeypatch,
                input_mode="single",
                authorization_mode="approved_batch",
            )

    @staticmethod
    def _runtime(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        input_mode: str = "batch",
        authorization_mode: str = "interactive",
        no_auto_assign: bool = False,
        token_available: bool = True,
    ):
        if token_available:
            monkeypatch.setenv("GITCODE_TOKEN", "test-token")
        else:
            monkeypatch.delenv("GITCODE_TOKEN", raising=False)
        input_path = tmp_path / "issues.json"
        input_path.write_text(
            json.dumps(
                {
                    "filters": {
                        "mode": input_mode,
                        "repository": "cann/ops-math",
                    },
                    "issues": [],
                }
            ),
            encoding="utf-8",
        )
        config_path = tmp_path / "classify.yaml"
        config_path.write_text("repo: cann/ops-math\n", encoding="utf-8")
        argv = [
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--authorization-mode",
            authorization_mode,
        ]
        if no_auto_assign:
            argv.append("--no-auto-assign")
        return CLASSIFIER.load_runtime(CLASSIFIER.parse_args(argv))

    @pytest.mark.parametrize(
        "runtime_kwargs,expected_mode,expected_dry_run",
        [
            ({}, "interactive", True),
            ({"authorization_mode": "approved_batch"}, "approved_batch", True),
            (
                {"authorization_mode": "approved_batch", "no_auto_assign": True},
                "approved_batch",
                True,
            ),
            (
                {"authorization_mode": "approved_batch", "token_available": False},
                "approved_batch",
                True,
            ),
        ],
        ids=["interactive", "approved", "no-auto-assign", "missing-token"],
    )
    def test_runtime_policy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        runtime_kwargs,
        expected_mode,
        expected_dry_run,
    ) -> None:
        runtime = self._runtime(tmp_path, monkeypatch, **runtime_kwargs)

        assert runtime.args.authorization_mode == expected_mode
        assert runtime.dry_run is expected_dry_run

    def test_classifier_assignment_callback_is_read_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._runtime(
            tmp_path,
            monkeypatch,
            authorization_mode="approved_batch",
        )
        assert runtime.post_fn(7, "/assign @expected-owner", "expected-owner") is False


class TestPullRequestAssociation:
    @staticmethod
    def test_linked_pr_without_substantive_response_requires_first_response() -> None:
        result = assert_classification(
            issue(number=2591, author="reporter"),
            "need_attention",
            "needs_first_response_with_pr",
            options=classification_options(
                {"2591": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                dry_run=False,
            ),
        )

        assert result["auto_action"]["response_requirement"] == "required"
        assert result["auto_action"]["candidate"] == "developer"

    @staticmethod
    def test_same_assignee_exempts_first_response_even_with_other_author_pr() -> None:
        assert_classification(
            issue(number=2590, author="reporter", assignee="reporter"),
            "no_attention",
            "self_assigned",
            options=classification_options(
                {"2590": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]}
            ),
        )

    @staticmethod
    def test_assignee_matching_reporter_without_pr_is_self_assigned() -> None:
        assert_classification(
            issue(number=2589, author="reporter", assignee="reporter"),
            "no_attention",
            "self_assigned",
        )

    @staticmethod
    @pytest.mark.parametrize("assignee,pr_author,bucket,category,assignment", [
        (None, None, "need_attention", "needs_first_look", None),
        ("owner", None, "need_attention", "our_team_needs_work", None),
        ("reporter", None, "no_attention", "self_assigned", None),
        (None, "developer", "need_attention", "needs_first_response_with_pr", "developer"),
        ("owner", "developer", "need_attention", "needs_first_response_with_pr", None),
        ("reporter", "developer", "no_attention", "self_assigned", None),
        (None, "reporter", "need_attention", "needs_pr_owner_handoff", "reporter"),
        ("owner", "reporter", "no_attention", "self_assigned", None),
        ("reporter", "reporter", "no_attention", "self_assigned", None),
    ])
    def test_first_response_decision_branches_are_read_only(
        assignee, pr_author, bucket, category, assignment,
    ) -> None:
        posted = []
        linked = {"1": [{"pr_state": "open", "pr_author": pr_author}]} if pr_author else {}
        result = assert_classification(
            issue(assignee=assignee), bucket, category,
            options=classification_options(
                linked, dry_run=False, post_fn=lambda *args: posted.append(args),
            ),
        )
        action = result["auto_action"]
        assert (action["candidate"] if action else None) == assignment
        assert posted == []
        if action:
            assert action["read_only"] is True

    @staticmethod
    @pytest.mark.parametrize("author,assignee", [
        ("Reporter", "reporter"), (" reporter ", "REPORTER"),
    ])
    def test_no_pr_self_assignment_compares_normalized_nonempty_accounts(author, assignee) -> None:
        assert_classification(
            issue(author=author, assignee=assignee), "no_attention", "self_assigned",
        )

    @staticmethod
    @pytest.mark.parametrize("author,assignee", [(None, None), ("", ""), (" ", " ")])
    def test_empty_accounts_do_not_create_self_authorship(author, assignee) -> None:
        result = classify(issue(author=author, assignee=assignee))
        assert result["bucket"] == "need_attention"
        assert result["category"] != "self_assigned"

    @staticmethod
    def test_no_pr_self_assignment_does_not_hide_reporter_followup() -> None:
        assert_classification(
            followup_issue(assignee="reporter"), "need_attention", "reporter_followup",
        )

    @staticmethod
    @pytest.mark.parametrize("linked", [
        [], [{"pr_state": "open"}], [{"pr_state": "unknown"}],
        [{"pr_state": "open", "pr_author": "someone-else"}],
    ])
    @pytest.mark.parametrize("scan_complete", [False, True])
    def test_self_assignment_needs_no_pr_author_or_scan_to_exempt_first_response(
        linked, scan_complete,
    ) -> None:
        assert_classification(
            issue(author="Reporter", assignee=" reporter "), "no_attention", "self_assigned",
            options=classification_options({"1": linked}, association_scan_complete=scan_complete),
        )

    @staticmethod
    def test_self_assignment_still_requires_complete_comment_evidence() -> None:
        assert_classification(
            issue(assignee="reporter"), "need_attention", "comment_scan_incomplete",
            options=classification_options(association_scan_complete=False, comment_scan_complete=False),
        )

    @staticmethod
    def test_self_assignment_reporter_followup_precedes_incomplete_pr_scan() -> None:
        assert_classification(
            followup_issue(assignee="reporter"), "need_attention", "reporter_followup",
            options=classification_options(association_scan_complete=False),
        )

    @staticmethod
    def test_self_assignment_keeps_new_reply_from_watched_assignee_actionable() -> None:
        source = assignee_conversation_issue(
            comment("operator-owner", "已确认处理方案，请确认。", comment_id=184760300,
                    created_at="2026-08-14T04:00:00Z"),
            assignee="reporter", followup_watch=assignee_watch(),
        )
        assert_classification(
            source, "need_attention", "assignee_followup",
            options=classification_options(association_scan_complete=False),
        )

    @staticmethod
    @pytest.mark.parametrize("author,assignee", [(None, None), ("", ""), (" ", " ")])
    def test_empty_identity_cannot_bypass_incomplete_pr_scan(author, assignee) -> None:
        assert_classification(
            issue(author=author, assignee=assignee), "need_attention", "association_scan_incomplete",
            options=classification_options(association_scan_complete=False),
        )

    @staticmethod
    def test_substantive_response_emits_read_only_pr_owner_strategy() -> None:
        posted = []
        result = assert_classification(
            issue(number=2588, author="reporter", comments=[comment()]),
            "need_attention",
            "needs_pr_owner_handoff",
            options=classification_options(
                {"2588": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                post_fn=lambda *args: posted.append(args) or True,
                dry_run=False,
                automation_policy={"auto_response": True, "auto_assign": False},
            ),
        )

        assert result["auto_action"]["strategy"] == "linked_pr_author_during_response"
        assert result["auto_action"]["candidate"] == "developer"
        assert result["auto_action"]["read_only"] is True
        assert posted == []

    @staticmethod
    def test_multiple_pr_authors_request_coverage_selection_during_response() -> None:
        posted = []
        result = assert_classification(
            issue(number=2586, author="reporter", comments=[comment()]),
            "need_attention",
            "needs_pr_owner_handoff",
            options=classification_options(
                {"2586": [
                    {"pr_state": "open", "pr_merged": False, "pr_author": "z-owner"},
                    {"pr_state": "open", "pr_merged": False, "pr_author": "a-owner"},
                ]},
                post_fn=lambda *args: posted.append(args) or True,
                dry_run=False,
                automation_policy={"auto_response": True, "auto_assign": True},
            ),
        )
        assert result["auto_action"]["selection"] == "issue_coverage"
        assert result["auto_action"]["candidates"] == ["a-owner", "z-owner"]
        assert posted == []

    @staticmethod
    def test_auto_assign_flag_does_not_enable_classifier_side_effect() -> None:
        posted = []
        result = assert_classification(
            issue(number=2585, author="reporter", comments=[comment()]),
            "need_attention",
            "needs_pr_owner_handoff",
            options=classification_options(
                {"2585": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                post_fn=lambda *args: posted.append(args) or True,
                dry_run=False,
                automation_policy={"auto_response": False, "auto_assign": False},
            ),
        )
        assert result["auto_action"]["strategy"] == "linked_pr_author_during_response"
        assert posted == []

    @staticmethod
    def test_incomplete_comment_scan_blocks_even_with_linked_pr() -> None:
        result = assert_classification(
            issue(number=2587, author="reporter"),
            "need_attention",
            "comment_scan_incomplete",
            options=classification_options(
                {"2587": [{"pr_state": "open", "pr_merged": False, "pr_author": "developer"}]},
                comment_scan_complete=False,
                dry_run=False,
            ),
        )

        assert result["auto_action"] is None

    @staticmethod
    def test_issue_author_with_own_pr_needs_assignment_without_first_response() -> None:
        issue_data = issue(number=2592, author="developer")
        issue_pr_map = {
            "2592": [
                {"pr_state": "open", "pr_merged": False, "pr_author": "developer"}
            ]
        }
        posted = []

        result = assert_classification(
            issue_data,
            "need_attention",
            "needs_pr_owner_handoff",
            options=classification_options(
                issue_pr_map,
                post_fn=lambda *args: posted.append(args) or True,
                dry_run=False,
            ),
        )

        assert result["auto_action"]["response_requirement"] == "exempt_self_authored_pr"
        assert result["auto_action"]["candidate"] == "developer"
        assert posted == []

    @staticmethod
    def test_failed_pr_scan_defers_external_action() -> None:
        result = assert_classification(
            issue(number=6),
            "need_attention",
            "association_scan_incomplete",
            options=classification_options(
                dry_run=False,
                association_scan_complete=False,
            ),
        )

        assert result["auto_action"] is None

    @staticmethod
    def test_text_link_is_mapped_without_linkage_api_call() -> None:
        prs = [
            pull_request(
                body="https://gitcode.com/cann/ops-math/issues/2561",
                title="stride=0 precision fix",
                user={"login": "hw-zhangpanpan"},
            )
        ]

        with patch.object(
            CLASSIFIER,
            "fetch_pr_linked_issues",
        ) as fetch_linkage:
            issue_map, diagnostics = CLASSIFIER.build_issue_pr_map(
                prs,
                linkage_options(target_issue_numbers=[2561]),
            )

        assert issue_map["2561"][0]["pr_number"] == 4487
        assert diagnostics["api_calls"] == 0
        fetch_linkage.assert_not_called()

    @staticmethod
    def test_linkage_api_calls_are_bounded() -> None:
        prs = [
            pull_request(number, head={"ref": f"fix-issue-2552-{number}"})
            for number in range(20)
        ]

        with patch.object(
            CLASSIFIER,
            "fetch_pr_linked_issues",
            return_value=[],
        ) as fetch_linkage:
            _, diagnostics = CLASSIFIER.build_issue_pr_map(
                prs,
                linkage_options(
                    target_issue_numbers=[2552], api_budget=3, cache_dir=None
                ),
            )

        assert fetch_linkage.call_count == 3
        assert diagnostics["budget_exhausted"] is True
        assert diagnostics["complete"] is False
        assert diagnostics["incomplete_issue_numbers"] == ["2552"]

    @staticmethod
    def test_unrelated_prs_do_not_use_native_linkage_api() -> None:
        prs = [
            pull_request(number, title=f"Unrelated PR {number}")
            for number in range(20)
        ]

        with patch.object(CLASSIFIER, "fetch_pr_linked_issues") as fetch_linkage:
            _, diagnostics = CLASSIFIER.build_issue_pr_map(
                prs,
                linkage_options(target_issue_numbers=[2552], cache_dir=None),
            )

        fetch_linkage.assert_not_called()
        assert diagnostics["candidates"] == 0
        assert diagnostics["complete"] is True

    @staticmethod
    def test_native_linkage_result_is_reused_from_cache(tmp_path: Path) -> None:
        pr = pull_request(
            updated_at="2026-08-11T08:00:00+08:00",
            title="Fix stride precision",
            head={"ref": "fix-issue-2552", "sha": "abc"},
        )

        with patch.object(
            CLASSIFIER,
            "fetch_pr_linked_issues",
            return_value=[{"number": 2552}],
        ):
            CLASSIFIER.build_issue_pr_map(
                [pr],
                linkage_options(target_issue_numbers=[2552], cache_dir=tmp_path),
            )

        with patch.object(
            CLASSIFIER,
            "fetch_pr_linked_issues",
        ) as fetch_linkage:
            issue_map, diagnostics = CLASSIFIER.build_issue_pr_map(
                [pr],
                linkage_options(target_issue_numbers=[2552], cache_dir=tmp_path),
            )

        fetch_linkage.assert_not_called()
        assert diagnostics["cache_hits"] == 1
        assert issue_map["2552"][0]["pr_number"] == 4487

    @staticmethod
    def test_closed_unmerged_pr_is_not_completion_evidence() -> None:
        issue_data = issue(number=4, assignee="owner")
        issue_pr_map = {
            "4": [
                {"pr_state": "closed", "pr_merged": False, "pr_author": "owner"}
            ]
        }

        assert_classification(
            issue_data,
            "need_attention",
            "our_team_needs_work",
            options=classification_options(issue_pr_map),
        )


class TestInputFailure:
    @staticmethod
    def test_empty_stdin_has_actionable_error() -> None:
        args = types.SimpleNamespace(input=None)

        with patch.object(CLASSIFIER.sys, "stdin", io.StringIO("")):
            with pytest.raises(ValueError) as raised:
                CLASSIFIER.read_input(args)

        assert "upstream fetch command may have failed" in str(raised.value)


class TestReport:
    @staticmethod
    def test_report_only_lists_need_attention() -> None:
        report = CLASSIFIER.format_report(
            need_attention=[
                {
                    "number": 101,
                    "title": "需要处理",
                    "assignee": None,
                    "comments_count": 0,
                    "reason": "缺少负责人",
                    "url": "https://example.test/issues/101",
                }
            ],
            no_attention=[
                {
                    "number": 102,
                    "title": "自提 Issue",
                    "reason": "自提 issue，已在自行处理",
                    "url": "https://example.test/issues/102",
                }
            ],
            since_iso="2026-08-04T00:00:00+08:00",
            all_clear=False,
        )

        assert "#101 需要处理" in report
        assert "#102" not in report
        assert "自提 Issue" not in report
        assert "不需要关注" not in report


@pytest.mark.parametrize("level", ["list-only", "ignore", "pending"])
@pytest.mark.parametrize("single", [False, True])
def test_responsibility_gate_prevents_assignment_even_when_authorized(level, single):
    policy = {"handle": ["公共问题"], "list-only": ["其他芯片"], "ignore": ["不处理"]}
    data = issue(comments=[comment(body="/assign @developer")])
    if level != "pending":
        data["responsibility_review"] = {
            "level": level, "summary": "核查结论", "evidence": ["具体版本 kernel 路径"],
            "policy_digest": CLASSIFIER.policy_digest(policy),
        }
    with patch.object(CLASSIFIER, "classify_one") as classify_mock:
        result = CLASSIFIER.classify_with_responsibility(
            data, classification_options(dry_run=False), policy, single
        )
    if level == "list-only":
        classify_mock.assert_called_once()
        assert result["bucket"] == "no_attention"
        assert not result["must_handle"]
        assert not result["single_issue_override"]
    else:
        classify_mock.assert_not_called()
    assert result["responsibility"] == level
    assert not result.get("auto_action")


def test_handle_preserves_existing_first_response_classification():
    policy = {"handle": ["公共问题"], "list-only": [], "ignore": []}
    data = issue(responsibility_review={
        "level": "handle", "summary": "公共 API 问题", "evidence": ["接口校验源码"],
        "policy_digest": CLASSIFIER.policy_digest(policy),
    })
    result = CLASSIFIER.classify_with_responsibility(data, classification_options(), policy)
    assert result["responsibility"] == "handle"
    assert result["category"] == classify(data)["category"]


@pytest.mark.parametrize("mutation", ["config", "evidence", "summary", "level"])
def test_stale_or_incomplete_review_requires_investigation(mutation):
    policy = {"handle": ["公共问题"], "list-only": [], "ignore": []}
    review = {"level": "handle", "summary": "公共问题", "evidence": ["路径证据"],
              "policy_digest": CLASSIFIER.policy_digest(policy)}
    if mutation == "config":
        policy["handle"] = ["950 算子"]
    else:
        review[mutation] = [] if mutation == "evidence" else ""
    assert CLASSIFIER.review_responsibility(issue(responsibility_review=review), policy)["level"] == "pending"


@pytest.mark.parametrize("fields", [{"merged": True}, {"merged_at": "2026-09-15T00:00:00Z"}, {"state": "merged"}])
def test_gitcode_pr_merge_variants_are_recognized(fields):
    invert_pr_refs = getattr(CLASSIFIER, "_invert_pr_refs")
    result = invert_pr_refs([pull_request(**fields)], {4487: {"1"}})
    assert result["1"][0]["pr_merged"] is True


def test_recent_pr_fetch_uses_configured_token():
    options = CLASSIFIER.PRFetchOptions(
        api_base="https://api.example.test", repo="cann/math", token="test-token", since_iso=None
    )
    with patch.object(CLASSIFIER, "make_session"), patch.object(CLASSIFIER, "api_get", return_value=[]) as get:
        prs, diagnostics = CLASSIFIER.fetch_recent_prs(options)
    assert prs == []
    assert diagnostics["complete"]
    assert get.call_args.args[2] == "test-token"


def test_cli_lists_only_reviewed_items_that_still_need_attention(tmp_path):
    import yaml
    policy = {"handle": ["公共问题"], "list-only": ["其他芯片"], "ignore": ["不处理"]}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "repo": "test/repo", "responsibility": policy,
        "report_file": str(tmp_path / "classification.txt"),
        "cache_dir": str(tmp_path / "cache"),
        "last_check_file": str(tmp_path / "last_check.json"),
    }), encoding="utf-8")
    data = []
    for number, level in enumerate(("list-only", "list-only", "ignore", "pending"), 1):
        item = issue(number=number, title=f"Issue {number}", url=f"https://gitcode.com/test/repo/issues/{number}")
        if number == 2:
            item.update(assignee="owner", comments=[comment()])
        if level != "pending":
            item["responsibility_review"] = {
                "level": level, "summary": f"核查结果 {number}", "evidence": ["源码证据"],
                "policy_digest": CLASSIFIER.policy_digest(policy),
            }
        data.append(item)
    source = tmp_path / "issues.json"
    source.write_text(json.dumps({"issues": data}), encoding="utf-8")
    with patch.object(CLASSIFIER, "_write_stdout") as output, patch.object(
        CLASSIFIER, "fetch_recent_prs", return_value=([], {"complete": True})
    ) as fetch, patch("issue_pr_evidence.api_get", return_value=[]):
        status = CLASSIFIER.main(["--config", str(config), "--input", str(source), "--ignore-last-check"])
    assert status == 0
    fetch.assert_called_once()
    result = json.loads(output.call_args.args[0])
    assert result["by_responsibility"] == {"handle": 0, "list-only": 2, "ignore": 1, "pending": 1}
    assert [entry["number"] for entry in result["listed_issues"]] == [1]
    assert [entry["number"] for entry in result["issues"]] == [4]
    assert result["ignored_count"] == 1
    assert "核查结果 1" in (tmp_path / "classification.txt").read_text()
    assert "核查结果 2" not in (tmp_path / "classification.txt").read_text()
    assert "核查结果 3" not in (tmp_path / "classification.txt").read_text()


def test_classifier_conflicting_input_stops_before_evidence(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("repo: cann/math\n")
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"filters": {"repository": "user/math"}, "issues": []}))
    with patch.object(CLASSIFIER, "_write_stdout") as output, patch.object(CLASSIFIER, "_collect_evidence") as evidence:
        code = CLASSIFIER.main(["--config", str(config), "--input", str(source)])
    assert code == 2
    assert json.loads(output.call_args.args[0])["status"] == "needs_selection"
    evidence.assert_not_called()
    assert config.read_text() == "repo: cann/math\n"
