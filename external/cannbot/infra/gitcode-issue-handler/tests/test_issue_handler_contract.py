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

import yaml

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
    documents = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    # Validate the full contract, including its conditionally loaded schema.
    documents["state"] += (handler_references / "runtime-state-schema.md").read_text(encoding="utf-8")
    return documents


def _assert_handler_interaction_contract(documents: dict[str, str]) -> None:
    skill = documents["skill"]
    policy = documents["policy"]
    assert "只询问规则（policy_query）" in skill
    assert "policy_query" in skill
    assert "不检查 Token/Git/CANN" in skill
    assert "runtime-capability-checks.md" in skill
    assert 'ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh' in documents["capability"]
    assert 'GITCODE_TOOLKIT_ROOT/scripts/preflight.sh' not in documents["setup"]
    assert "--checks api" in documents["capability"]
    assert "--checks git" in documents["capability"]
    assert "--checks tmp" in documents["capability"]
    assert "--checks author" in documents["capability"]
    assert "缺失算子责任人请求最多 1 次" in policy
    assert "候选表随 summary" in policy
    assert "不能静默回退到 Agent 正常处理" in policy
    assert "统一执行预览前" in policy
    assert "候选调查不延迟回复，本身不授权 assign 或 direct" in policy
    assert "need_attention / operator_routing_required" in documents["intake"]
    assert "authorization_mode: interactive | approved_batch" in documents["state"]
    assert "`single`、`batch` 均从 `interactive` 开始" in documents["execution"]
    assert "但不授权实际 push" in documents["execution"]
    assert "`single` 和 `batch` 共用回复与交付两种检查点" in policy
    assert "初始“处理/自动处理/auto apply”请求不是代码交付批准" in policy
    assert "Issue 处理请求已授权常规交付写操作" not in documents["evals"]
    assert "确认具体账号或当前 Issue 的 `direct`" in policy
    assert "generate_summary_report.py" in documents["reporting"]
    assert "--strict" in documents["reporting"]
    assert "不可覆盖其他 run" in documents["reporting"]
    assert "`issues` 只保存本轮实际处理" in documents["reporting"]
    assert "changed files/diff" in documents["execution"]
    assert "禁止未经对应授权的 Issue" in documents["execution"]

    authorization = documents["authorization"]
    comment_workflow = documents["comment_workflow"]
    assert "| `interactive` |" in authorization
    assert "| `approved_batch` |" in authorization
    assert "精确 Issue 清单" in authorization
    assert "`auto-close-stale` 独立于 `approved_batch`" in authorization
    assert "算子转交" in comment_workflow
    assert "等待期间不静默关闭" in comment_workflow
    assert "默认 45 次/60 秒" in comment_workflow


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
    assert "--checks api" in documents["capability"]
    assert "--checks author" in documents["capability"]
    assert "ISSUE_HANDLER_TMP_DIR" in documents["capability"]
    assert ".cannbot/gitcode-issue-handler/tmp" in documents["capability"]


def test_missing_token_creates_one_resumable_wait_and_stops_the_turn():
    documents = _read_interaction_documents()
    skill = documents["skill"]
    capability = documents["capability"]
    state = documents["state"]

    assert "只汇总询问一次并停止本轮" in capability
    for forbidden_progress in ("API 探测", "真实 Issue 拉取", "测试框架读取"):
        assert forbidden_progress in capability
    assert "overall_status: running | waiting_for_input" in state
    assert "api: not_started | ready | waiting_for_input" in state
    assert "input_id: gitcode_token" in state
    assert "request_count: 1" in state
    assert "`resume_from`" in state
    assert "暂停整轮" in capability
    assert "保存唯一输入" in capability
    assert "测试读取" in capability
    assert "恢复时只重跑失败的 `api` 组" in capability


def test_response_and_delivery_gates_cover_external_or_publish_writes():
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

    assert "初始“处理/自动处理/auto apply”请求不是代码交付批准" in policy
    assert "任何 GitCode 写入、暂存、提交、推送、PR 或 CI 前" in execution
    assert "Issue/评论/指派/标签/状态 POST/PUT/PATCH/DELETE" in execution
    assert "暂存、commit、push、PR 和 CI" in execution
    assert "评论" in execution
    assert "指派" in execution
    assert "changed files/diff" in execution
    assert "message" in execution
    assert "目标功能分支" in execution
    assert "标题、完整正文" in execution
    assert "首次 CI" in execution
    assert (
        "execution_confirmation_status: not_required | pending | approved | rejected | invalidated"
        in execution
    )
    assert "POST/GET" in execution
    assert "按授权契约校验" in delivery
    assert "operator-handoff.md" in diagnosis
    assert "完整正文加入回复执行预览，禁止发送未经授权的正文" in (HANDLER_ROOT / "references/operator-handoff.md").read_text(encoding="utf-8")
    assert "本次 intake 固定传 `interactive`" in intake
    assert "分类器不执行外部写入" in intake
    assert "plan_confirmation_status" not in skill
    assert "delivery_confirmation_status" not in skill


def test_direct_push_remains_outside_the_unified_confirmation():
    execution = (HANDLER_ROOT / "references" / "delivery-confirmation.md").read_text(
        encoding="utf-8"
    )
    delivery = (HANDLER_ROOT / "references" / "delivery-publish.md").read_text(
        encoding="utf-8"
    )

    assert "commit 后需独立确认" in execution
    assert "该确认不能被统一批准吞并" in execution
    assert "commit 后独立展示并确认" in delivery


def test_auto_close_apply_requires_its_own_exact_preview_confirmation():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    maintenance = (
        HANDLER_ROOT / "references" / "maintenance-stale-close.md"
    ).read_text(encoding="utf-8")

    assert "默认 dry-run" in skill
    assert "完整固定评论和关闭操作" in maintenance
    assert "初始“自动处理 / auto apply”请求不是这次批准" in maintenance
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

    assert "挂起回查" in (
        HANDLER_ROOT / "references" / "policy-error-handling.md"
    ).read_text(encoding="utf-8")
    assert "updated_at` 增量" in intake
    assert "显式 single、增量更新和 watchlist 均不能绕过此过滤" in intake
    assert "`reporter_followup`" in intake
    assert "classification-categories.md" in intake
    assert "`awaiting_assignee_setup`" in (
        HANDLER_ROOT / "references/classification-categories.md"
    ).read_text(encoding="utf-8")
    assert "`assignee_followup`" in intake
    assert "失败时停止，不改状态、不写 watch" in followup
    assert "--waiting-on assignee" in followup
    assert "等待期间不静默关闭" in comment_workflow
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
    assert "优先读取" in setup
    assert "不自动移动或删除" in setup
    assert "不改 `.gitignore`" in setup


def test_skill_entrypoint_stays_concise_and_routes_detailed_contracts():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert len(skill.splitlines()) <= 200
    assert "## 按需读取" in skill
    assert "## 授权模型与卡点" not in skill
    assert "## 运行状态" not in skill
    assert "authorization_mode: interactive | approved_batch" not in skill
    assert "# 运行时：最小授权与状态契约" in (
        HANDLER_ROOT / "references" / "runtime-state.md"
    ).read_text(encoding="utf-8")


def test_reference_layout_stays_flat_and_grouped_by_concern():
    references = HANDLER_ROOT / "references"
    expected = {
        "pipeline.md",
        "batch-analysis.md",
        "automation.md",
        "runtime-setup.md",
        "configuration-setup.md",
        "runtime-state.md",
        "runtime-state-schema.md",
        "classification-categories.md",
        "operator-handoff.md",
        "response-quality-rubric.md",
        "runtime-capability-checks.md",
        "runtime-knowledge.md",
        "knowledge-maintenance.md",
        "responsibility-scope.md",
        "response-writing.md",
        "issue-intake.md",
        "issue-routing.md",
        "operator-owner-candidates.md",
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


def test_runtime_knowledge_query_is_not_blocked_by_refresh():
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

    assert "复用" in lifecycle and "快照" in lifecycle and "受审" in lifecycle
    assert "knowledge-maintenance.md" in skill
    maintenance = (HANDLER_ROOT / "references/knowledge-maintenance.md").read_text(encoding="utf-8")
    assert "knowledge-maintenance.md" in lifecycle
    assert "runtime-knowledge.md" in setup
    assert "refresh_issue_knowledge.py" in lifecycle
    assert "knowledge_refresh_status" in maintenance
    assert "不要求本轮已执行 refresh" in diagnosis
    assert '--repository-root "$ISSUE_HANDLER_REPOSITORY_ROOT"' in diagnosis
    assert "快照缺失、过期或无命中均不阻塞首响" in lifecycle
    assert "provisional/low" in lifecycle
    assert "stale_fallback" in maintenance


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
    assert "repro,tmp,worktrees" in setup
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


def test_conditional_references_remain_reachable_and_local_links_resolve():
    documents = [HANDLER_ROOT / "SKILL.md", *sorted((HANDLER_ROOT / "references").glob("*.md"))]
    targets = set()
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            assert resolved.exists(), (path, target)
            targets.add(resolved)
    names = (
        "runtime-state-schema.md",
        "classification-categories.md",
        "operator-handoff.md",
        "response-quality-rubric.md",
    )
    for name in names:
        assert (HANDLER_ROOT / "references" / name).resolve() in targets


def test_scope_and_candidate_reviews_require_matching_evidence():
    intake = (HANDLER_ROOT / "references/responsibility-scope.md").read_text(encoding="utf-8")
    owners = (HANDLER_ROOT / "references/operator-owner-candidates.md").read_text(encoding="utf-8")
    assert "问题版本 + 完整故障文件路径 + 实际入口/构建选择" in intake
    assert "不能用当前 master 替代问题版本" in intake
    assert "本算子核心行为补丁" in owners
    assert "该补丁作者到 login 的映射" in owners
    assert "README、UT 或局部防御性修补" in owners
    assert "不能进入候选表" in owners


def test_reply_quality_rubric_is_conditional_and_keeps_all_score_bands():
    workflow = (HANDLER_ROOT / "references/response-writing.md").read_text(encoding="utf-8")
    rubric = (HANDLER_ROOT / "references/response-quality-rubric.md").read_text(encoding="utf-8")
    assert "正式评分或上述检查发现质量不足时" in workflow
    assert "| 分数 |" not in workflow
    for score in (100, 80, 60, 40, 20, 0):
        assert f"| {score} |" in rubric
    for category in ("漏洞", "缺陷", "需求", "文档", "咨询", "任务", "审查", "其他"):
        assert category in rubric
    assert "github.com" not in rubric
    assert "https://gitcode.com/xujiachen8/cannbot-skills/issues/1" in rubric


def test_reply_workflow_avoids_unsupported_commitments():
    workflow = (HANDLER_ROOT / "references/response-writing.md").read_text(encoding="utf-8")
    rubric = (HANDLER_ROOT / "references/response-quality-rubric.md").read_text(encoding="utf-8")
    assert "不轻易承诺未来动作或结果" in workflow
    assert "我们将修复/补充/完成/推进/上线/在某版本支持" in workflow
    assert "一个可行的解决" in workflow
    assert "持续跟踪" in workflow
    assert "承诺检查" in workflow
    assert "先区分错误与工程选择" in workflow
    assert "不一边倒赞同" in workflow
    assert "实现与维护成本" in workflow
    assert "兼容性/迁移风险" in workflow
    assert "我们将继续评估该方案及其影响" in workflow
    assert "不为追求“明确下一步”承诺" in rubric
    assert "可行/潜在方案" in rubric
    assert "不等于赞同提出者方案或承诺采纳" in rubric


def test_auto_response_can_link_reviewed_covering_pr_before_temporary_assignment():
    skill = (HANDLER_ROOT / "SKILL.md").read_text(encoding="utf-8")
    automation = (HANDLER_ROOT / "references/automation.md").read_text(encoding="utf-8")
    workflow = (HANDLER_ROOT / "references/issue-comment-workflow.md").read_text(encoding="utf-8")

    assert "associate_issue_pr.py" in automation
    assert "coverage_verified: true" in automation
    assert "additional_risk_reviewed: true" in automation
    assert "unresolved_risks" in automation
    assert "PR→Issue" in automation and "Issue→PR" in automation
    assert "duplicate_cross_reference" in automation
    assert "一个 PR 只能关联一个 Issue" in automation
    assert "PR 关联必须先于" in workflow
    response = (HANDLER_ROOT / "references/response-writing.md").read_text(encoding="utf-8")
    assert "自动关联" in response and "临时指派 PR 作者" in response


def test_assignment_commands_use_state_not_exact_comment_body():
    handoff = (HANDLER_ROOT / "references/operator-handoff.md").read_text(encoding="utf-8")
    generic = (TOOLKIT_ROOT / "references/issue-comment-workflow.md").read_text(encoding="utf-8")
    api = (TOOLKIT_ROOT / "references/gitcode-api.md").read_text(encoding="utf-8")
    assert "平台可能将 mention 改写为 Markdown" in handoff
    assert "成功依据是 assignee login" in handoff
    assert "仅授权发送评论时不能扩大为原生指派" in handoff
    assert 'PATCH `{"assignee":"<owner>"}`' in handoff
    assert "不能用完整正文相等判定成功" in generic
    assert '单数字段 `{"assignee":"<login>"}`' in api


def test_runtime_schema_preserves_established_fields_enums_and_defaults():
    # Frozen public state contract; do not derive expectations from the edited docs.
    contract = json.loads((Path(__file__).parent / "fixtures/runtime_state_contract.json").read_text(encoding="utf-8"))
    text = (HANDLER_ROOT / "references/runtime-state-schema.md").read_text(encoding="utf-8")
    schema = {}
    for block in re.findall(r"```yaml\n(.*?)```", text, re.S):
        schema.update(yaml.safe_load(block))

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
            return result
        if isinstance(value, list) and value:
            return flatten(value[0], prefix + "[]")
        return {prefix: value}

    authorization_text = (HANDLER_ROOT / "references/authorization-contract.md").read_text(encoding="utf-8")
    authorization = yaml.safe_load(re.search(r"```yaml\n(.*?)```", authorization_text, re.S).group(1))
    actual = {**flatten(schema), **flatten(authorization)}
    expected_fields = {**contract["fields"], **contract["authorization_fields"]}
    for field, expected in expected_fields.items():
        assert field in actual, f"Missing established state field: {field}"
        if isinstance(expected, str) and " | " in expected:
            assert set(expected.split(" | ")) <= set(actual.get(field, "").split(" | ")), field
        elif expected is not None:
            assert actual.get(field) == expected, field
