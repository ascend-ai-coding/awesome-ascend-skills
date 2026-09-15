# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import re
import subprocess
import sys
from pathlib import Path

HANDLER_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = HANDLER_ROOT.parent / "gitcode-toolkit"
REPO_ROOT = HANDLER_ROOT.parents[1]


def _read_interaction_documents() -> dict[str, str]:
    handler_references = HANDLER_ROOT / "references"
    paths = {
        "skill": HANDLER_ROOT / "SKILL.md",
        "setup": handler_references / "runtime-setup.md",
        "capability": handler_references / "runtime-capability-checks.md",
        "state": handler_references / "runtime-state.md",
        "policy": handler_references / "policy-error-handling.md",
        "reporting": handler_references / "delivery-reporting.md",
        "execution": handler_references / "delivery-confirmation.md",
        "intake": handler_references / "issue-intake.md",
        "comment_workflow": handler_references / "issue-comment-workflow.md",
        "authorization": handler_references / "authorization-contract.md",
        "evals": HANDLER_ROOT / "evals" / "evals.json",
    }
    return {name: path.read_text(encoding="utf-8") for name, path in paths.items()}


def _assert_handler_interaction_contract(documents: dict[str, str]) -> None:
    skill = documents["skill"]
    policy = documents["policy"]
    assert "步骤 -1：区分 policy_query 与真实执行" in skill
    assert "`policy_query`" in skill
    assert "不运行任何环境检查" in documents["setup"]
    assert "runtime-capability-checks.md" in skill
    assert 'ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh' in documents["setup"]
    assert 'GITCODE_TOOLKIT_ROOT/scripts/preflight.sh' not in documents["setup"]
    assert "--checks api" in documents["setup"]
    assert "--checks git" in documents["setup"]
    assert "--checks tmp" in documents["setup"]
    assert "--checks author" in documents["setup"]
    assert "缺失算子责任人请求最多 1 次" in policy
    assert "配置责任人并指派（推荐）" in policy
    assert "不能静默回退到 Agent 正常处理" in policy
    assert "统一执行预览前" in policy
    assert "这次方案选择不构成执行批准" in policy
    assert "need_attention / operator_routing_required" in documents["intake"]
    assert "authorization_mode: interactive | approved_batch" in documents["state"]
    assert "`single` 和 `batch` 都从 `interactive` 开始" in documents["execution"]
    assert "本检查点不授权实际 push" in documents["execution"]
    assert "`single` 和 `batch` 共用一个分析后统一执行检查点" in policy
    assert "初始“处理/自动处理/auto apply”请求不是批准" in policy
    assert "Issue 处理请求已授权常规交付写操作" not in documents["evals"]
    assert "选择 `direct` 时只授权当前 Issue" in policy
    assert "generate_summary_report.py" in documents["reporting"]
    assert "--strict" in documents["reporting"]
    assert "历史报告不可覆盖其他 `run_id`" in documents["reporting"]
    assert "`issues` 只保存和展示本轮**实际处理**的 Issue" in documents["reporting"]
    assert "最终 diff 和验证结果" in documents["execution"]
    assert "确认前禁止" in documents["execution"]

    authorization = documents["authorization"]
    comment_workflow = documents["comment_workflow"]
    assert "| `interactive` |" in authorization
    assert "| `approved_batch` |" in authorization
    assert "精确 Issue 清单" in authorization
    assert "`auto-close-stale` 是独立维护路径" in authorization
    assert "算子转交" in comment_workflow
    assert "等待责任人的 Issue 不参与" in comment_workflow
    assert "Handler 限流与日志" in comment_workflow


def test_issue_workflow_documents_keep_business_contracts_in_handler():
    documents = _read_interaction_documents()
    _assert_handler_interaction_contract(documents)


def test_toolkit_references_do_not_depend_on_handler_business_contracts():
    toolkit_references = TOOLKIT_ROOT / "references"
    generic_document_parts = []
    for name in (
        "authorization-contract.md",
        "issue-comment-workflow.md",
        "gitcode-api.md",
    ):
        generic_document_parts.append(
            (toolkit_references / name).read_text(encoding="utf-8")
        )
    generic_documents = "\n".join(generic_document_parts)

    forbidden_business_fragments = (
        "approved_batch",
        "issue_iids",
        "followup_state.py",
        ".cannbot/gitcode-issue-handler",
        "watchlist",
        "算子责任人",
        "auto-close-stale",
    )
    assert not any(
        fragment in generic_documents for fragment in forbidden_business_fragments
    )


def test_environment_checks_are_deferred_until_the_protected_operation():
    documents = _read_interaction_documents()
    combined = "\n".join(
        documents[name]
        for name in ("skill", "setup", "capability", "state", "policy", "execution")
    )

    assert "policy_query" in documents["skill"]
    assert "capability_checks:" in documents["state"]
    assert "preflight_completed" not in documents["state"]
    assert "credential_ready" not in documents["state"]
    assert "步骤 -1：一次性启动预检" not in combined
    assert "git author 必须在步骤 -1" not in combined
    assert "--checks api" in documents["setup"]
    assert "--checks author" in documents["setup"]
    assert "ISSUE_HANDLER_TMP_DIR" in documents["capability"]
    assert ".cannbot/gitcode-issue-handler/tmp" in documents["capability"]


def test_missing_token_creates_one_resumable_wait_and_stops_the_turn():
    documents = _read_interaction_documents()
    skill = documents["skill"]
    capability = documents["capability"]
    state = documents["state"]

    assert "只汇总询问一次并停止本轮" in skill
    for forbidden_progress in ("API 探测", "真实 Issue 拉取", "测试框架读取"):
        assert forbidden_progress in skill
    assert "overall_status: running | waiting_for_input" in state
    assert "api: not_started | ready | waiting_for_input" in state
    assert "input_id: gitcode_token" in state
    assert "request_count: 1" in state
    assert "resume_from:" in state
    assert "立即停止本轮" in capability
    assert "不得重复询问" in capability
    assert "测试框架读取" in capability
    assert "从保存的 `resume_from` 继续" in capability


def test_unified_execution_gate_precedes_every_external_or_publish_write():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    execution = (HANDLER_ROOT / "references" / "delivery-confirmation.md").read_text(
        encoding="utf-8"
    )
    delivery = (HANDLER_ROOT / "references" / "delivery-publish.md").read_text(
        encoding="utf-8"
    )
    diagnosis = (HANDLER_ROOT / "references" / "issue-routing.md").read_text(
        encoding="utf-8"
    )
    intake = (HANDLER_ROOT / "references" / "issue-intake.md").read_text(
        encoding="utf-8"
    )
    policy = (HANDLER_ROOT / "references" / "policy-error-handling.md").read_text(
        encoding="utf-8"
    )

    assert "初始“处理/自动处理/auto apply”请求不是批准" in policy
    assert "任何 GitCode 写入或发布动作前" in execution
    assert "POST/PUT/PATCH/DELETE Issue" in execution
    assert "暂存、commit、push、创建 PR 或触发 CI" in execution
    assert "Issue 评论" in execution
    assert "指派" in execution
    assert "实际 changed files/diff 摘要" in execution
    assert "commit message" in execution
    assert "功能分支 push" in execution
    assert "标题和完整正文" in execution
    assert "首次 CI" in execution
    assert (
        "execution_confirmation_status: not_required | pending | approved | rejected | invalidated"
        in execution
    )
    assert "POST 后 GET" in execution
    assert "不在这里再次询问" in delivery
    assert "完整正文加入统一执行预览，禁止发送" in diagnosis
    assert "本次 intake 固定传 `interactive`" in intake
    assert "禁止在分类阶段传" in intake
    assert "plan_confirmation_status" not in skill
    assert "delivery_confirmation_status" not in skill


def test_direct_push_remains_outside_the_unified_confirmation():
    execution = (HANDLER_ROOT / "references" / "delivery-confirmation.md").read_text(
        encoding="utf-8"
    )
    delivery = (HANDLER_ROOT / "references" / "delivery-publish.md").read_text(
        encoding="utf-8"
    )

    assert "本次不授权；commit 形成后凭 SHA 独立确认" in execution
    assert "该强确认不能被 `approved_batch` 或统一执行" in execution
    assert "仍按步骤 8B 独立确认" in delivery


def test_auto_close_apply_requires_its_own_exact_preview_confirmation():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    maintenance = (
        HANDLER_ROOT / "references" / "maintenance-stale-close.md"
    ).read_text(encoding="utf-8")

    assert "默认 dry-run" in skill
    assert "完整固定评论和关闭操作" in maintenance
    assert "初始“自动\n处理 / auto apply”请求不是这次批准" in maintenance
    assert "单独记录的部署授权" in maintenance


def test_followup_state_is_fetched_routed_and_authorized_end_to_end():
    intake = (HANDLER_ROOT / "references" / "issue-intake.md").read_text(
        encoding="utf-8"
    )
    followup = (HANDLER_ROOT / "references" / "issue-followup.md").read_text(
        encoding="utf-8"
    )
    execution = (HANDLER_ROOT / "references" / "delivery-confirmation.md").read_text(
        encoding="utf-8"
    )
    comment_workflow = (
        HANDLER_ROOT / "references" / "issue-comment-workflow.md"
    ).read_text(encoding="utf-8")
    api = (TOOLKIT_ROOT / "references" / "gitcode-api.md").read_text(encoding="utf-8")

    assert "自定义状态迁移到`挂起`" in (
        HANDLER_ROOT / "references" / "policy-error-handling.md"
    ).read_text(encoding="utf-8")
    assert "updated_at` 增量" in intake
    assert "watchlist 不受核心 open/closed" in intake
    assert "`reporter_followup`" in intake
    assert "`awaiting_assignee_setup`" in intake
    assert "`assignee_followup`" in intake
    assert "失败时停止，不改状态、不写 watch" in followup
    assert "--waiting-on assignee" in followup
    assert "等待责任人的 Issue 不参与" in comment_workflow
    assert "普通受理、进展" in comment_workflow
    assert "issue_state_change" in execution
    assert "状态 ID 会随组配置变化" in api
    assert "禁止硬编码" in api
    assert (HANDLER_ROOT / "scripts" / "followup_state.py").is_file()


def test_skill_documents_use_repository_installation_contract():
    setup = (HANDLER_ROOT / "references" / "runtime-setup.md").read_text(
        encoding="utf-8"
    )

    assert "Marketplace" in setup
    assert "install-helper" in setup
    assert "npx skills" in setup
    assert "requirements.txt" in setup
    assert "安装到同一 `skills/` 根目录" in setup
    assert "不要另建 toolkit 副本" in setup
    assert "单 Issue 不需要" in setup
    assert 'ISSUE_HANDLER_RUNTIME_ROOT=".cannbot/gitcode-issue-handler"' in setup
    assert "{config,data,reports,logs,cache,images,repro,tmp,worktrees}" in setup
    assert "git rev-parse --git-path info/exclude" in setup
    assert "grep -Fqx '/.cannbot/gitcode-issue-handler/'" in setup
    assert "printf '/.cannbot/gitcode-issue-handler/\\n'" in setup
    assert "始终优先读取" in setup
    assert "不会自动移动或删除" in setup
    assert "不创建或覆盖工作树里的\n`.gitignore`" in setup


def test_skill_entrypoint_stays_concise_and_routes_detailed_contracts():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert len(skill.splitlines()) <= 200
    assert "## Reference 读取路由" in skill
    assert "## 授权模型与卡点" not in skill
    assert "## 运行状态" not in skill
    assert "authorization_mode: interactive | approved_batch" not in skill
    assert "# 运行时：授权与状态契约" in (
        HANDLER_ROOT / "references" / "runtime-state.md"
    ).read_text(encoding="utf-8")


def test_reference_layout_stays_flat_and_grouped_by_concern():
    references = HANDLER_ROOT / "references"
    expected = {
            "runtime-setup.md",
            "runtime-state.md",
            "runtime-capability-checks.md",
            "runtime-knowledge.md",
        "issue-intake.md",
        "issue-routing.md",
        "issue-followup.md",
        "issue-comment-workflow.md",
        "authorization-contract.md",
        "code-worktree.md",
        "code-root-cause.md",
        "code-git-history.md",
        "code-validation.md",
        "delivery-confirmation.md",
        "delivery-publish.md",
        "delivery-reporting.md",
        "policy-error-handling.md",
        "maintenance-stale-close.md",
    }

    top_level = {path.name for path in references.glob("*.md")}
    all_references = {path.relative_to(references) for path in references.rglob("*.md")}
    assert top_level == expected
    assert all(path.parent == Path(".") for path in all_references)


def test_repository_exposes_only_the_handler_public_name():
    marketplace = (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(
        encoding="utf-8"
    )
    registry = (
        REPO_ROOT
        / "plugins-community"
        / "install-helper"
        / "src"
        / "core"
        / "skill-registry.ts"
    ).read_text(encoding="utf-8")
    public_name = "gitcode-issue-handler"

    assert marketplace.count(f'"./{public_name}"') == 1
    assert registry.count(f'id: "{public_name}"') == 1
    assert (HANDLER_ROOT / "scripts" / "knowledge_query.py").is_file()


def test_handler_and_toolkit_have_no_retired_branding():
    retired_fragments = (
        "-".join(("issue", "fix", "helper")),
        "_".join(("issue", "fix", "helper")),
        "_".join(("ISSUE", "FIX")),
        "-".join(("issue", "knowledge", "query")),
        "_".join(("issue", "knowledge", "query")),
    )
    for root in (HANDLER_ROOT, TOOLKIT_ROOT):
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            assert not any(fragment in text for fragment in retired_fragments), path


def test_gitcode_issue_workflow_uses_pr_terminology():
    documentation_paths = (
        REPO_ROOT / "docs" / "feature-list.md",
        REPO_ROOT / "docs" / "skills-usage.md",
        REPO_ROOT / "tests" / "system" / "docs" / "ST_DESIGN_AND_DEVELOPMENT_GUIDE.md",
    )
    paths = list(documentation_paths)
    for root in (HANDLER_ROOT, TOOLKIT_ROOT):
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )

    legacy_term_pattern = re.compile(r"\b" + "m" + r"rs?\b", re.IGNORECASE)
    forbidden_fragments = (
        "_".join(("m" + "r", "url")),
        "_".join(("m" + "r", "create")),
        "--" + "-".join(("m" + "r", "url")),
    )
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert legacy_term_pattern.search(text) is None, path
        assert not any(fragment in text for fragment in forbidden_fragments), path


def test_renamed_knowledge_query_verifies_bundled_knowledge():
    script = HANDLER_ROOT / "scripts" / "knowledge_query.py"
    result = subprocess.run(
        [sys.executable, str(script), "verify"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["findings"] == []


def test_runtime_knowledge_refresh_is_wired_before_first_query():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    setup = (HANDLER_ROOT / "references" / "runtime-setup.md").read_text(
        encoding="utf-8"
    )
    diagnosis = (HANDLER_ROOT / "references" / "issue-routing.md").read_text(
        encoding="utf-8"
    )
    lifecycle = (HANDLER_ROOT / "references" / "runtime-knowledge.md").read_text(
        encoding="utf-8"
    )

    assert "步骤 0a：刷新运行时历史证据" in skill
    assert "runtime-knowledge.md" in setup
    assert "refresh_issue_knowledge.py" in lifecycle
    assert "knowledge_refresh_status" in lifecycle
    assert "未记录时返回" in diagnosis
    assert '--repository-root "$ISSUE_HANDLER_REPOSITORY_ROOT"' in diagnosis
    assert "首次全量，日常增量，周期校准" in lifecycle
    assert "provisional/low" in lifecycle
    assert "stale_fallback" in lifecycle


def test_runtime_defaults_are_confined_to_the_canonical_tree():
    scripts = HANDLER_ROOT / "scripts"
    runtime_paths = (scripts / "runtime_paths.py").read_text(encoding="utf-8")
    for path in scripts.glob("*.py"):
        if path.name == "runtime_paths.py":
            continue
        assert "issue_analysis_data" not in path.read_text(encoding="utf-8"), path

    assert 'RUNTIME_ROOT = Path(".cannbot") / "gitcode-issue-handler"' in runtime_paths
    assert "LEGACY_CLASSIFY_CONFIG" in runtime_paths
    assert "LEGACY_OPERATOR_OWNERS_CONFIG" in runtime_paths

    setup = (HANDLER_ROOT / "references" / "runtime-setup.md").read_text(
        encoding="utf-8"
    )
    assert 'ISSUE_HANDLER_RUNTIME_ROOT=".cannbot/gitcode-issue-handler"' in setup
    assert "└── tmp/" in setup
    assert "issue_analysis_data/tmp" not in setup

    help_result = subprocess.run(
        [
            sys.executable,
            str(scripts / "build_issue_knowledge.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    compact_help = "".join(help_result.stdout.split())
    assert ".cannbot/gitcode-issue-handler/data/issue-history.json" in compact_help
    assert ".cannbot/gitcode-issue-handler/reports/knowledge-corpus.md" in compact_help
