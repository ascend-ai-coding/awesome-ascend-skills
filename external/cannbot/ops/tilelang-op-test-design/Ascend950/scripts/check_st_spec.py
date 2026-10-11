#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""校验当前仓库 ST 的契约与用例清单。

该检查器验证可审查的证据，并识别常见的不实结论。它不会执行 pytest，
也不能证明 reference 或算子实现正确。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_KINDS = {
    "user_requirement",
    "interface_doc",
    "design_doc",
    "public_docstring",
    "api_validation",
    "mathematical_definition",
    "reference",
    "cuda_implementation",
    "ascend_implementation",
    "existing_test",
    "implementation",
}
PRIMARY_ORACLES = {
    "pytorch_reference",
    "cpu_reference",
    "exact_expected",
    "mathematical",
    "metamorphic",
    "exception_contract",
}
SUPPORTING_ORACLES = {"cuda_differential", "legacy_differential"}
CASE_KINDS = {"normal", "boundary", "negative", "gradient", "stateful"}
ASSERTIONS = {
    "shape",
    "dtype",
    "device",
    "values",
    "gradients",
    "aux_outputs",
    "state_changed",
    "state_unchanged",
    "protected_storage",
    "exception_type",
    "exception_message",
}
LIFECYCLE = {"designed": 0, "implemented": 1, "collected": 2, "executed": 3}
RESULTS = {"not_run", "passed", "failed", "skipped", "xfailed", "error"}
COVERAGE_DIMENSIONS = (
    "functionality",
    "precision",
    "boundary",
    "gradient",
    "state_mutation",
    "invalid_rejection",
    "layout_interface",
    "backend_branch",
    "randomness",
    "execution_evidence",
)
COVERAGE_STATUSES = {"covered", "partial", "missing", "unknown", "not_applicable"}
SEMANTIC_SOURCE_KINDS = {
    "user_requirement",
    "interface_doc",
    "design_doc",
    "public_docstring",
    "mathematical_definition",
    "reference",
}


class Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.case_errors: dict[str, int] = {}

    def add(self, severity: str, code: str, message: str, context: str = "") -> None:
        record = {"code": code, "message": message}
        if context:
            record["context"] = context
        if severity == "error":
            self.errors.append(record)
            if context.startswith("case:"):
                case_id = context.split(":", 1)[1]
                self.case_errors[case_id] = self.case_errors.get(case_id, 0) + 1
        else:
            self.warnings.append(record)

    def error(self, code: str, message: str, context: str = "") -> None:
        self.add("error", code, message, context)

    def warn(self, code: str, message: str, context: str = "") -> None:
        self.add("warning", code, message, context)


def require_nonempty_string(
    value: Any, field: str, findings: Findings, context: str = ""
) -> bool:
    if isinstance(value, str) and value.strip():
        return True
    findings.error("invalid_field", f"{field} 必须是非空字符串", context)
    return False


def unique_records(
    records: Any, field: str, findings: Findings, context_prefix: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        findings.error("invalid_field", f"{field} 必须是列表")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        context = f"{context_prefix}:{index}"
        if not isinstance(record, dict):
            findings.error("invalid_record", f"{field}[{index}] 必须是对象", context)
            continue
        record_id = record.get("id")
        if not require_nonempty_string(
            record_id, f"{field}[{index}].id", findings, context
        ):
            continue
        if record_id in result:
            findings.error(
                "duplicate_id", f"{field} 中存在重复 ID {record_id!r}", context
            )
            continue
        result[record_id] = record
    return result


def validate_sources(
    data: dict[str, Any], repo_root: Path, findings: Findings
) -> dict[str, dict[str, Any]]:
    sources = unique_records(
        data.get("contract_sources"), "contract_sources", findings, "source"
    )
    for source_id, source in sources.items():
        context = f"source:{source_id}"
        kind = source.get("kind")
        if kind not in SOURCE_KINDS:
            findings.error("invalid_source_kind", f"不支持的来源种类 {kind!r}", context)
        require_nonempty_string(source.get("claim"), "claim", findings, context)
        path_value = source.get("path")
        if path_value is None:
            if kind != "user_requirement":
                findings.warn("source_without_path", "本地来源没有提供路径", context)
        elif not isinstance(path_value, str) or not path_value.strip():
            findings.error("invalid_source_path", "path 必须是非空字符串", context)
        elif not (repo_root / path_value).exists():
            findings.error(
                "missing_source_path", f"来源路径不存在：{path_value}", context
            )
    return sources


def validate_requirements(
    data: dict[str, Any], sources: dict[str, dict[str, Any]], findings: Findings
) -> dict[str, dict[str, Any]]:
    requirements = unique_records(
        data.get("requirements"), "requirements", findings, "requirement"
    )
    for requirement_id, requirement in requirements.items():
        context = f"requirement:{requirement_id}"
        require_nonempty_string(
            requirement.get("statement"), "statement", findings, context
        )
        applicability = requirement.get("applicability")
        if applicability not in {"confirmed", "assumed", "unknown", "not_applicable"}:
            findings.error(
                "invalid_applicability",
                f"不支持的 applicability 值 {applicability!r}",
                context,
            )
        source_ids = requirement.get("source_ids")
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) for item in source_ids
        ):
            findings.error("invalid_source_ids", "source_ids 必须是字符串列表", context)
            continue
        missing = [source_id for source_id in source_ids if source_id not in sources]
        if missing:
            findings.error("unknown_source", f"未知的来源 ID：{missing}", context)
        if applicability == "confirmed" and not source_ids:
            findings.error(
                "confirmed_without_source",
                "状态为 confirmed 的需求必须引用至少一个来源",
                context,
            )
        source_kinds = {sources[item]["kind"] for item in source_ids if item in sources}
        if (
            applicability == "confirmed"
            and source_kinds
            and not (source_kinds & SEMANTIC_SOURCE_KINDS)
        ):
            findings.warn(
                "implementation_only_contract",
                "状态为 confirmed 的需求只依赖实现或当前测试证据，应将其视为暂定契约",
                context,
            )
    return requirements


def validate_oracles(
    case: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    findings: Findings,
    context: str,
) -> set[str]:
    oracles = case.get("oracles")
    if not isinstance(oracles, list) or not oracles:
        findings.error("missing_oracle", "用例必须声明至少一个判定基准", context)
        return set()
    kinds: set[str] = set()
    for index, oracle in enumerate(oracles):
        if not isinstance(oracle, dict):
            findings.error("invalid_oracle", f"oracles[{index}] 必须是对象", context)
            continue
        kind = oracle.get("kind")
        if kind not in PRIMARY_ORACLES | SUPPORTING_ORACLES:
            findings.error(
                "invalid_oracle_kind", f"不支持的判定基准种类 {kind!r}", context
            )
        else:
            kinds.add(kind)
        source_id = oracle.get("source_id")
        if not isinstance(source_id, str) or source_id not in sources:
            findings.error(
                "unknown_oracle_source",
                f"判定基准引用了未知来源 {source_id!r}",
                context,
            )
    if kinds and not (kinds & PRIMARY_ORACLES):
        findings.error(
            "differential_only_oracle",
            "CUDA 或旧实现的差分证据不能作为唯一的主要判定基准",
            context,
        )
    return kinds


def validate_assertions(
    case: dict[str, Any], findings: Findings, context: str
) -> set[str]:
    assertions = case.get("assertions")
    if (
        not isinstance(assertions, list)
        or not assertions
        or not all(isinstance(item, str) for item in assertions)
    ):
        findings.error("invalid_assertions", "assertions 必须是非空字符串列表", context)
        return set()
    names = set(assertions)
    unknown = sorted(names - ASSERTIONS)
    if unknown:
        findings.error("unknown_assertion", f"未知的断言名称：{unknown}", context)
    kind = case.get("kind")
    if kind in {"normal", "boundary"} and not ({"values", "state_changed"} & names):
        findings.error(
            "shape_only_correctness",
            f"{kind} 用例必须检查数值或状态变化，不能只检查元数据",
            context,
        )
    if kind == "gradient" and "gradients" not in names:
        findings.error("missing_gradient_assertion", "梯度用例必须检查梯度", context)
    if kind == "stateful" and "state_changed" not in names:
        findings.error(
            "missing_state_assertion", "有状态用例必须检查预期的状态变化", context
        )
    if kind == "negative" and "exception_type" not in names:
        findings.error(
            "weak_negative_assertion", "负向用例必须检查明确的异常类型", context
        )
    return names


def validate_path_evidence(
    case: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    findings: Findings,
    context: str,
) -> None:
    evidence = case.get("path_evidence")
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        findings.error("invalid_path_evidence", "path_evidence 必须是对象", context)
        return
    source_id = evidence.get("source_id")
    if not isinstance(source_id, str) or source_id not in sources:
        findings.error(
            "unknown_path_source",
            f"path_evidence 引用了未知来源 {source_id!r}",
            context,
        )
    if evidence.get("kind") != "tail":
        return
    logical_size = evidence.get("logical_size")
    block_size = evidence.get("block_size")
    if not isinstance(logical_size, int) or logical_size < 0:
        findings.error("invalid_tail_size", "尾块 logical_size 必须是非负整数", context)
    if not isinstance(block_size, int) or block_size <= 0:
        findings.error("invalid_block_size", "尾块 block_size 必须是正整数", context)
    if (
        isinstance(logical_size, int)
        and isinstance(block_size, int)
        and block_size > 0
        and logical_size % block_size == 0
    ):
        findings.error(
            "false_tail_claim",
            f"{logical_size} 可以被块大小 {block_size} 整除，并不构成尾块",
            context,
        )
    require_nonempty_string(
        evidence.get("axis"), "path_evidence.axis", findings, context
    )


def validate_lifecycle(case: dict[str, Any], findings: Findings, context: str) -> None:
    status = case.get("status")
    result = case.get("result")
    if status not in LIFECYCLE:
        findings.error("invalid_status", f"不支持的生命周期状态 {status!r}", context)
        return
    if result not in RESULTS:
        findings.error("invalid_result", f"不支持的运行结果 {result!r}", context)
        return
    if LIFECYCLE[status] >= LIFECYCLE["collected"] and not isinstance(
        case.get("test_nodeid"), str
    ):
        findings.error(
            "missing_nodeid",
            "状态为 collected 或 executed 的用例必须包含 test_nodeid",
            context,
        )
    if status == "executed" and result == "not_run":
        findings.error(
            "executed_without_result", "状态为 executed 的用例必须包含执行结果", context
        )
    if status != "executed" and result != "not_run":
        findings.error(
            "result_without_execution",
            "只有状态为 executed 的用例才能报告通过、失败、跳过或错误",
            context,
        )


def validate_cases(
    data: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    requirements: dict[str, dict[str, Any]],
    findings: Findings,
) -> list[dict[str, Any]]:
    cases_by_id = unique_records(data.get("cases"), "cases", findings, "case-index")
    cases = list(cases_by_id.values())
    for case_id, case in cases_by_id.items():
        context = f"case:{case_id}"
        kind = case.get("kind")
        if kind not in CASE_KINDS:
            findings.error("invalid_case_kind", f"不支持的用例种类 {kind!r}", context)
        level = case.get("level")
        if level not in {0, 1, 2, None}:
            findings.error("invalid_level", "level 必须为 0、1、2 或 null", context)
        requirement_ids = case.get("requirement_ids")
        if not isinstance(requirement_ids, list) or not requirement_ids:
            findings.error(
                "missing_requirements", "用例必须映射到至少一个需求", context
            )
            requirement_ids = []
        unknown_requirements = [
            item for item in requirement_ids if item not in requirements
        ]
        if unknown_requirements:
            findings.error(
                "unknown_requirement", f"未知的需求 ID：{unknown_requirements}", context
            )
        for requirement_id in requirement_ids:
            requirement = requirements.get(requirement_id)
            if requirement and requirement.get("applicability") == "unknown":
                findings.error(
                    "unknown_contract",
                    f"用例断言了契约尚未知的需求 {requirement_id}",
                    context,
                )
            if requirement and requirement.get("applicability") == "not_applicable":
                findings.error(
                    "not_applicable_requirement",
                    f"用例断言了不适用的需求 {requirement_id}",
                    context,
                )
        oracle_kinds = validate_oracles(case, sources, findings, context)
        validate_assertions(case, findings, context)
        if kind == "negative" and "exception_contract" not in oracle_kinds:
            findings.error(
                "missing_exception_contract",
                "负向用例必须使用 exception_contract 判定基准",
                context,
            )
        validate_path_evidence(case, sources, findings, context)
        validate_lifecycle(case, findings, context)
    return cases


def validate_coverage(
    data: dict[str, Any],
    cases: list[dict[str, Any]],
    findings: Findings,
    require_complete: bool,
) -> dict[str, dict[str, Any]]:
    coverage = data.get("coverage")
    if coverage is None:
        if require_complete:
            findings.error("missing_coverage", "完整交付必须包含十个维度的覆盖评估")
        return {}
    if not isinstance(coverage, dict):
        findings.error("invalid_coverage", "coverage 必须是对象")
        return {}

    unknown_dimensions = sorted(set(coverage) - set(COVERAGE_DIMENSIONS))
    if unknown_dimensions:
        findings.error(
            "unknown_coverage_dimension", f"未知的覆盖维度：{unknown_dimensions}"
        )

    cases_by_id = {str(case.get("id")): case for case in cases}
    validated: dict[str, dict[str, Any]] = {}
    for dimension in COVERAGE_DIMENSIONS:
        context = f"coverage:{dimension}"
        entry = coverage.get(dimension)
        if entry is None:
            if require_complete:
                findings.error(
                    "missing_coverage_dimension", f"缺少覆盖维度 {dimension!r}", context
                )
            else:
                findings.warn(
                    "missing_coverage_dimension", f"缺少覆盖维度 {dimension!r}", context
                )
            continue
        if not isinstance(entry, dict):
            findings.error("invalid_coverage_entry", "覆盖项必须是对象", context)
            continue
        validated[dimension] = entry
        status = entry.get("status")
        if status not in COVERAGE_STATUSES:
            findings.error(
                "invalid_coverage_status", f"不支持的覆盖状态 {status!r}", context
            )
        case_ids = entry.get("case_ids")
        if not isinstance(case_ids, list) or not all(
            isinstance(item, str) for item in case_ids
        ):
            findings.error(
                "invalid_coverage_cases", "case_ids 必须是字符串列表", context
            )
            case_ids = []
        unknown_cases = [case_id for case_id in case_ids if case_id not in cases_by_id]
        if unknown_cases:
            findings.error(
                "unknown_coverage_case", f"未知的用例 ID：{unknown_cases}", context
            )
        if status == "covered" and not case_ids:
            findings.error(
                "covered_without_case",
                "状态为 covered 的维度必须引用至少一个用例",
                context,
            )
        if status in {"missing", "unknown", "not_applicable"}:
            require_nonempty_string(
                entry.get("rationale"), "rationale", findings, context
            )
        if status == "not_applicable" and case_ids:
            findings.error(
                "not_applicable_with_cases",
                "状态为 not_applicable 的维度不能引用用例",
                context,
            )
        untrusted_cases = [
            case_id for case_id in case_ids if findings.case_errors.get(case_id, 0)
        ]
        if untrusted_cases:
            findings.error(
                "coverage_uses_untrusted_case",
                f"覆盖项引用了无效用例：{untrusted_cases}",
                context,
            )
        if dimension == "execution_evidence" and status == "covered":
            not_passed = [
                case_id
                for case_id in case_ids
                if case_id in cases_by_id
                and (
                    cases_by_id[case_id].get("status") != "executed"
                    or cases_by_id[case_id].get("result") != "passed"
                )
            ]
            if not_passed:
                findings.error(
                    "execution_coverage_not_passed",
                    f"执行证据引用了未达到 executed/passed 状态的用例：{not_passed}",
                    context,
                )
        if require_complete and status not in {"covered", "not_applicable"}:
            findings.error(
                "incomplete_coverage",
                f"交付覆盖状态必须为 covered 或 not_applicable，实际为 {status!r}",
                context,
            )
    return validated


def validate(
    data: Any, repo_root: Path, require_complete_coverage: bool = False
) -> dict[str, Any]:
    findings = Findings()
    if not isinstance(data, dict):
        findings.error("invalid_document", "JSON 顶层值必须是对象")
        data = {}
    if data.get("schema_version") != 1:
        findings.error("schema_version", "schema_version 必须为 1")
    require_nonempty_string(data.get("operator"), "operator", findings)
    sources = validate_sources(data, repo_root, findings)
    requirements = validate_requirements(data, sources, findings)
    cases = validate_cases(data, sources, requirements, findings)
    coverage = validate_coverage(data, cases, findings, require_complete_coverage)
    credible_cases = [
        case.get("id")
        for case in cases
        if findings.case_errors.get(str(case.get("id")), 0) == 0
    ]
    return {
        "operator": data.get("operator"),
        "valid": not findings.errors,
        "summary": {
            "sources": len(sources),
            "requirements": len(requirements),
            "cases": len(cases),
            "credible_cases": len(credible_cases),
            "coverage_dimensions": len(coverage),
            "errors": len(findings.errors),
            "warnings": len(findings.warnings),
        },
        "credible_case_ids": credible_cases,
        "errors": findings.errors,
        "warnings": findings.warnings,
        "limitations": [
            "清单校验通过不能证明 pytest 已被收集或执行、目标路径确实可达，也不能证明算子正确。",
            "仍需通过代码审查确认 reference 的独立性和语义准确性。",
        ],
    }


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    verdict = "PASS" if report["valid"] else "FAIL"
    lines = [
        f"{verdict}: {report.get('operator') or '<未知算子>'}",
        (
            f"来源={summary['sources']} 需求={summary['requirements']} 用例={summary['cases']} "
            f"可信用例={summary['credible_cases']} 覆盖维度={summary['coverage_dimensions']} "
            f"错误={summary['errors']} 警告={summary['warnings']}"
        ),
    ]
    severity_labels = {"errors": "错误", "warnings": "警告"}
    for severity in ("errors", "warnings"):
        for finding in report[severity]:
            context = f" [{finding['context']}]" if finding.get("context") else ""
            lines.append(
                f"{severity_labels[severity]} {finding['code']}{context}: {finding['message']}"
            )
    lines.append("检查器通过只表示清单有效，不能证明 pytest 已执行或算子正确。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict-warnings", action="store_true")
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="要求十个维度全部为 covered，或明确标记为 not_applicable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = json.loads(args.spec.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: 无法加载 {args.spec}：{exc}")
        return 2
    report = validate(data, args.repo_root.resolve(), args.require_complete_coverage)
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text(report), end="")
    if not report["valid"] or (args.strict_warnings and report["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
