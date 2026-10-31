#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate response evidence and project pipeline state into the existing report."""

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from pipeline_evidence import content_hash


class MaterialError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("response_materials_incomplete")


def assignment_action(classification):
    action = classification.get("auto_action") or {}
    return action if action.get("type") == "assign_candidate" else {}


def assignment_only(classification):
    return assignment_action(classification).get("response_requirement") in {"exempt_self_authored_pr", "satisfied"}


SOURCE_VERDICTS = ("confirmed_at_revision", "not_reproduced_at_revision", "needs_version", "partial", "not_link_issue")
OWNER_STATUSES = ("candidates", "unresolved", "not_needed")


def response_enums():
    return {"source_verdict": list(SOURCE_VERDICTS), "owner_review.status": list(OWNER_STATUSES)}


def response_template(classification=None):
    classification = classification or {}
    result = {
        "decision": "prepared",
        "summary": "",
        "evidence": [],
        "inspection_files": [],
        "analysis": "",
        "reply": "",
        "source_verdict": "needs_version",
        "owner_review": {"status": "unresolved", "operators": [], "reason": "", "candidates": []},
        "response_review": {"checked_facts": [], "remaining_unknowns": []},
        "next_action": "",
    }
    action = assignment_action(classification)
    if action:
        result["assignment"] = {"login": action.get("candidate") or "", "reason": ""}
    if assignment_only(classification):
        result["source_verdict"] = "not_link_issue"
        result["owner_review"].update(status="not_needed", reason="按已有关联 PR 准备责任人分配")
    return result


def response_instruction(classification):
    if assignment_only(classification):
        return "按分类中的关联 PR 候选与证据填写分配理由及分析；脚本生成 assign.md 和报告。"
    return "inspect 关键文件后，依据固定版本证据填写 result_file；需要候选责任人时加 --identities。脚本生成材料与报告。"


def string_list(value, nonempty=False):
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(x, str) and x.strip() for x in value)
    )


def _valid_assignment(assignment, allowed):
    if not isinstance(assignment, dict):
        return False
    login = assignment.get("login")
    reason = assignment.get("reason")
    return (
        isinstance(login, str)
        and login in allowed
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", login))
        and isinstance(reason, str)
        and bool(reason.strip())
    )


def _valid_owner(owner, assignment_exempt=False):
    if not isinstance(owner, dict):
        return False
    if assignment_exempt:
        return (
            owner.get("status") == "not_needed"
            and isinstance(owner.get("reason"), str)
            and bool(owner["reason"].strip())
        )
    return (
        isinstance(owner.get("status"), str)
        and owner.get("status") in OWNER_STATUSES
        and isinstance(owner.get("reason"), str)
        and bool(owner["reason"].strip())
    )


def _read_inspection(state, task, root, name, errors):
    if not isinstance(name, str):
        errors.append("inspection_files: 文件路径必须是字符串")
        return None
    receipt = state.get("source_evidence", {}).get(name)
    if not receipt or receipt.get("task_id") != task["task_id"] or receipt.get("attempt_id") != task["attempt_id"]:
        errors.append("inspection_files: 证据未由当前任务 attempt 的 inspect 生成")
        return None
    try:
        value = json.loads((Path(root) / name).read_text())
        if content_hash(value) != receipt["digest"] or value.get("repository") != state.get(
            "repository", value.get("repository")
        ):
            raise ValueError()
        return value
    except (OSError, ValueError):
        errors.append("inspection_files: 证据文件不可读或被修改，请重新 inspect")
        return None


def _inspection_records(state, task, result, root, errors):
    inspections = []
    files = result.get("inspection_files")
    if not isinstance(files, list) or not files:
        errors.append(
            "inspection_files: 先运行 inspect --claim-file CLAIM --path 仓库相对路径，引用返回的 evidence_file"
        )
    else:
        for name in files:
            value = _read_inspection(state, task, root, name, errors)
            if value is not None:
                inspections.append(value)
    records = [r for e in inspections for r in e["records"]]
    if records and not any(r["tracked"] for r in records):
        errors.append("inspection_files: 仅检查不存在的路径不够；请检查实际报告入口、父目录或保留 needs_evidence")
    if records and not any(r["kind"] == "file" for r in records):
        errors.append("inspection_files: 目录列表不能替代关键源码/README；至少检查一个相关文件")
    return records


def _check_source_verdict(task, result, records, errors):
    issue = task["payload"]["issue"]
    link_issue = bool(re.search(r"死链|链接|404|broken.?link|missing.*\.md", issue.get("title", ""), re.I))
    verdict = result.get("source_verdict")
    if not isinstance(verdict, str) or verdict not in SOURCE_VERDICTS:
        errors.append(
            "source_verdict: 选择 confirmed_at_revision/not_reproduced_at_revision/needs_version/partial/not_link_issue"
        )
    links = [link for r in records for link in r.get("links", [])]
    if link_issue:
        if not any(r.get("path", "").lower().endswith(".md") and r["kind"] == "file" for r in records):
            errors.append("链接问题必须 inspect 报告引用链接的 Markdown 文件")
        if verdict == "not_link_issue":
            errors.append("链接问题不能填写 not_link_issue；按 inspect 的逐链接事实填写")
        if verdict == "confirmed_at_revision" and not any(l["status"] == "target_missing_at_revision" for l in links):
            errors.append("事实冲突：inspect 未发现缺失目标，不能确认当前版本存在死链；核对报告版本")


def _check_candidate(candidate, known, errors):
    if (
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("login"), str)
        or candidate["login"] not in known
    ):
        errors.append("候选 login 须由 inspect --identities 的提交元数据验证；否则保留 unresolved")
    elif not string_list(candidate.get("operators"), True):
        errors.append("候选 operators 必须是非空算子名称字符串列表")
    elif any(
        not isinstance(candidate.get(k), str) or not candidate[k].strip() for k in ("contribution", "limitation")
    ) or any(not string_list(candidate.get(k), True) for k in ("evidence", "identity_evidence")):
        errors.append("候选 contribution/limitation 使用文字，evidence/identity_evidence 使用非空字符串列表")


def _check_owner_review(result, records, errors):
    owner = result.get("owner_review")
    if not _valid_owner(owner):
        errors.append("owner_review: 填写 candidates/unresolved/not_needed 及理由；未验证账号不能猜测")
        return
    if owner["status"] != "not_needed" and not string_list(owner.get("operators", [])):
        errors.append("owner_review.operators: 使用算子名称字符串列表")
    history = [h for r in records for h in r.get("history", [])]
    known = {h.get("login") for h in history if h.get("login")}
    candidates = owner.get("candidates", [])
    if not isinstance(candidates, list):
        errors.append("owner_review.candidates 必须是列表")
    elif owner["status"] == "candidates":
        if not candidates or not owner.get("operators"):
            errors.append("候选分析需要 operators 和至少一名候选")
        for candidate in candidates:
            _check_candidate(candidate, known, errors)
    elif owner["status"] == "unresolved":
        if candidates:
            errors.append("owner_review: 有可靠候选时使用 candidates，否则候选列表保持空白")
        if not history:
            errors.append("未确认 owner 时先 inspect 相关实现以取得贡献历史，记录具体身份或归属缺口")


def _check_response_fields(task, result, errors):
    classification = task["payload"].get("classification", {})
    assignment_exempt = assignment_only(classification)
    for field in ("analysis", "next_action") if assignment_exempt else ("analysis", "reply", "next_action"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            errors.append(f"{field}: 填写完整的分析、可发布草稿及下一步；无条件生成草稿时使用 needs_evidence")
    action = assignment_action(classification)
    assignment = result.get("assignment")
    if action:
        allowed = {action["candidate"]} if action.get("candidate") else set(action.get("candidates", []))
        if not _valid_assignment(assignment, allowed):
            errors.append("assignment: 选择分类中已核实的关联 PR 作者并填写分配理由；证据不足使用 needs_evidence")
    elif assignment:
        errors.append("assignment: 当前分类没有待分配动作")
    return assignment_exempt


def _check_assignment_exempt(result, errors):
    if result.get("reply"):
        errors.append("reply: 本项仅需分配，保持空白")
    owner = result.get("owner_review")
    if not _valid_owner(owner, assignment_exempt=True):
        errors.append("owner_review: 已有关联 PR 候选，填写 not_needed 及依据")
    review = result.get("response_review")
    if (
        not isinstance(review, dict)
        or not string_list(review.get("checked_facts"), True)
        or not string_list(review.get("remaining_unknowns"))
    ):
        errors.append("response_review: 核对关联 PR 作者和分配依据，记录剩余疑点")


def validate_response(state, task, result, root):
    if not isinstance(result, dict):
        raise MaterialError(["result_file: 必须是 JSON 对象"])
    errors = []
    assignment_exempt = _check_response_fields(task, result, errors)
    if assignment_exempt:
        _check_assignment_exempt(result, errors)
        if errors:
            raise MaterialError(errors)
        return []
    records = _inspection_records(state, task, result, root, errors)
    _check_source_verdict(task, result, records, errors)
    _check_owner_review(result, records, errors)
    review = result.get("response_review")
    if (
        not isinstance(review, dict)
        or not string_list(review.get("checked_facts"), True)
        or not string_list(review.get("remaining_unknowns"))
    ):
        errors.append(
            "response_review: 列出逐条核对的 checked_facts 和 remaining_unknowns；不得把 Issue 原文当作已验证根因"
        )
    if errors:
        raise MaterialError(errors)
    return records


def materialize(result, records, store, relative):
    related = [
        {k: r[k] for k in ("path", "url", "kind", "revision")} for r in records if r["kind"] in {"file", "directory"}
    ]
    related = list({(r["path"], r["revision"], r["kind"], r["url"]): r for r in related}.values())
    links = "\n".join(f"- [{r['path']}]({r['url']})（{r['revision'][:12]}）" for r in related)
    analysis = result["analysis"].rstrip() + "\n"
    if links:
        analysis += "\n## 已核查代码与文档\n\n" + links + "\n"
    files = {"analysis.md": store.text_artifact(relative / "analysis.md", analysis)}
    if result.get("reply"):
        files["reply.md"] = store.text_artifact(relative / "reply.md", result["reply"].rstrip() + "\n")
    if result.get("assignment"):
        files["assign.md"] = store.text_artifact(
            relative / "assign.md", "/assign @" + result["assignment"]["login"] + "\n"
        )
    review = {
        "status": "prepared_for_review",
        "source_verdict": result.get("source_verdict"),
        "owner_review": result["owner_review"],
        "response_review": result["response_review"],
        "inspection_files": result.get("inspection_files", []),
        "body_digest": content_hash(result.get("reply", "")),
        "assignment": result.get("assignment"),
    }
    files["response-review.json"] = store.artifact(relative / "response-review.json", review)
    store.artifact(
        relative / "response-artifacts.json", {"files": files, "status": "pending_approval", "external_write": False}
    )
    return files, related


def _iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _prepared_issue(entry, classification, result, now):
    owner = result["owner_review"]
    candidates = owner.get("candidates", []) if owner["status"] != "not_needed" else []
    operators = owner.get("operators", []) if owner["status"] != "not_needed" else []
    covered = {op for c in candidates for op in c.get("operators", [])}
    issue = {
        **entry["issue"],
        "handled_in_run": True,
        "bucket": "need_attention",
        "category": classification.get("category", "awaiting_maintainer"),
        "handling_status": "waiting_for_approval",
        "resolution_status": "unresolved",
        "result_summary": result["summary"],
        "next_action": result["next_action"],
        "responsibility": "handle",
        "response_status": "prepared",
        "response_artifacts": entry["response_file_map"],
        "related_code": entry["related_code"],
        "response_review": result["response_review"],
        "problem_summary": result["summary"],
        "owner_candidate_analysis": {
            "status": owner["status"],
            "summary": owner["reason"],
            "operators": operators,
            "candidates": candidates,
            "uncovered_operators": sorted(set(operators) - covered),
        },
        "process_log": [
            {
                "time": _iso(now),
                "stage": stage,
                "action": action,
                "result": result["summary"],
                "evidence": result["evidence"],
            }
            for stage, action in [("triage", "脚本分类与责任范围审核"), ("diagnose", "准备证据、分析与待审核回复")]
        ],
    }
    if assignment_only(classification):
        issue["related_code_note"] = "仅准备已有关联 PR 的责任人分配，无新增源码结论。"
        if assignment_action(classification).get("response_requirement") == "exempt_self_authored_pr":
            issue["response_status"] = "exempt_self_authored_pr"
    if result.get("assignment"):
        issue["assignment_plan"] = result["assignment"]
    if owner["status"] == "not_needed":
        issue.pop("owner_candidate_analysis", None)
    return issue


def report_state(state, root, now):
    issues, listed = [], []
    for iid, entry in state["issues"].items():
        classification = entry.get("classification", {})
        if classification.get("responsibility") == "list-only" and classification.get("bucket") == "need_attention":
            listed.append(
                {
                    "iid": iid,
                    "url": entry["issue"]["url"],
                    "responsibility": "list-only",
                    "responsibility_summary": classification.get("responsibility_summary", "仅列举"),
                }
            )
        result = entry.get("prepared_response")
        current_scan = entry.get("classification_scan_id") == state["scan"].get("id")
        actionable = (
            classification.get("bucket") == "need_attention" and classification.get("responsibility") == "handle"
        )
        prepared = result and entry["phase"] == "waiting" and current_scan and actionable
        if not prepared:
            continue
        issues.append(_prepared_issue(entry, classification, result, now))
    scan_id = state["scan"].get("id", "startup")
    return {
        "run": {
            "run_id": state.get("report_run", {}).get("run_id", f"pipeline-{scan_id}"),
            "mode": "batch",
            "repository": state["repository"],
            "repository_root": str(root),
            "started_at": state.get("report_run", {}).get("started_at", _iso(state["scan"].get("attempted_at", now))),
            "completed_at": _iso(now),
            "overall_status": "waiting_for_input" if issues else "no_issues",
            "authorization_mode": "interactive",
            "response_confirmation_status": "pending" if issues else "not_required",
            "issues_scanned_total": len(state["issues"]),
            "issues_total": len(issues),
            "issues_listed_total": len(listed),
        },
        "issues": issues,
        "listed_issues": listed,
        "groups": [],
        "external_operations": [],
    }
