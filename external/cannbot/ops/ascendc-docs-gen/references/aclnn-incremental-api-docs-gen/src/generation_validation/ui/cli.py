# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""CLI UI for CodeDocAutoEditor - Rich terminal interface."""

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import questionary
from rich.box import SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.llm import LLMClient
from src.logger import info as log_info
from src.logger import setup_logger
from src.logger import warning as log_warning

from .. import config
from ..core import ValidationResult
from ..dashboard import (
    dashboard_done,
    dashboard_error,
    dashboard_log,
    dashboard_progress,
    dashboard_result,
    dashboard_step,
    dashboard_warn,
    get_current_doc_part,
    get_dashboard,
)
from ..generator.get_example import (
    get_all_example_codes,
    get_example_code,
    get_example_codes_by_describe,
    get_example_content_with_source,
    get_example_doc_part,
)
from ..pipeline import GenerationStage, StyleStage, ValidationStage
from ..pipeline.deterministic_checks import (
    build_sig_corpus,
    collect_issues,
    resolve_orig_doc_full,
)
from ..pipeline.helpers import get_output_dir, setup_pipeline
from ..validator import validate_example

console = Console()


@dataclass
class EvidenceFile:
    path: Path
    relative_path: str
    doc_part: str
    basename: str


@dataclass
class LogCapture:
    lines: list = field(default_factory=list)

    def append(self, msg: str):
        self.lines.append(msg)


@dataclass
class PipelineResult:
    parsed: dict = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)
    style_summary: dict = field(default_factory=dict)
    style_extraction: dict = field(default_factory=dict)
    generated: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    history: list = field(default_factory=list)
    runtime_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    evidence_conflict: bool = False
    no_change_needed: bool = False
    conflict_details: Optional[str] = None
    max_retries_hit: bool = False
    attempts_used: int = 1


def detect_doc_part(filename: str) -> Optional[str]:
    match = re.search(config.DOC_PART_PATTERN, filename)
    return match.group(1) if match else None


def scan_evidence_files(evidence_dir: Path) -> list[EvidenceFile]:
    files = []
    if not evidence_dir.exists():
        return files

    for f in evidence_dir.rglob("*.json"):
        if f.is_file():
            rel_path = str(f.relative_to(evidence_dir))
            doc_part = detect_doc_part(f.name)
            if doc_part:
                files.append(
                    EvidenceFile(
                        path=f,
                        relative_path=rel_path,
                        doc_part=doc_part,
                        basename=f.stem,
                    )
                )
    # 给证据文件列表中增加 example 的 dummy 选项添加到列表中
    new_evidence = EvidenceFile(
        path=evidence_dir / "example.json",
        relative_path="example.json",
        doc_part="example",
        basename="example",
    )
    files.append(new_evidence)
    return sorted(files, key=lambda x: (x.doc_part, x.relative_path))


def group_files_by_doc_part(files: list[EvidenceFile]) -> dict[str, list[EvidenceFile]]:
    grouped = {}
    for f in files:
        if f.doc_part not in grouped:
            grouped[f.doc_part] = []
        grouped[f.doc_part].append(f)
    return grouped


console_style = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),
        ("pointer", "fg:green bold"),
        ("highlighted", "fg:green bold"),
        ("selected", "fg:green bold"),
    ]
)


def select_evidence_file(files: list[EvidenceFile]) -> EvidenceFile:
    grouped = group_files_by_doc_part(files)

    choices = []
    for doc_part in config.DOC_PARTS:
        if doc_part in grouped:
            for f in grouped[doc_part]:
                choices.append(f)

    return questionary.select(
        "Select an evidence file:",
        choices=[questionary.Choice(f.relative_path, f) for f in choices],
        style=console_style,
    ).ask()


def confirm_doc_part(detected: str) -> str:
    console.print(f"Auto-detected doc_part: [bold cyan]{detected}[/bold cyan]")
    selected = questionary.select(
        "Would you like to keep it or change?",
        choices=["Keep detected", "Change..."],
        style=console_style,
    ).ask()

    if selected == "Keep detected":
        return detected
    return questionary.select(
        "Select doc_part:",
        choices=[questionary.Choice(dp, dp) for dp in config.DOC_PARTS],
        style=console_style,
    ).ask()


def select_mode() -> str:
    return questionary.select(
        "Select mode:",
        choices=[
            questionary.Choice("Generate & Validate (full)", "full"),
            questionary.Choice("Validate Only", "validate_only"),
        ],
        style=console_style,
    ).ask()


def resolve_evidence_file(evidence_dir: Path, doc_part: str) -> Optional[EvidenceFile]:
    """根据 doc_part 从 evidence_dir 中解析对应的证据文件。doc_part 为 example 时返回 None。"""
    if doc_part == "example":
        return None
    files = scan_evidence_files(evidence_dir)
    for f in files:
        if f.doc_part == doc_part:
            return f
    return None


def select_doc_part(available_parts: list[str]) -> Optional[str]:
    """从 available_parts 列表中选择 doc_part。列表只有一个则 auto-select。"""
    if len(available_parts) == 1:
        return available_parts[0]
    return questionary.select(
        "Select doc_part to process:",
        choices=available_parts,
        style=console_style,
    ).ask()


def display_header():
    header = Text()
    logo = r"""
     _    ____ _     _   _ _   _
    / \  / ___| |   | \ | | \ | |
   / _ \| |   | |   |  \| |  \| |
  / ___ \ |___| |___| |\  | |\  |
 /_/   \_\____|_____|_| \_|_| \_|
                                 """
    header.append(
        f"{logo}\n"
        + "A CLI Tool for Incremental Generation and Validation of CANN Operator Documentation\n"
        + "CANN 算子文档增量生成与一致性校验命令行工具",
        style="bold cyan",
    )
    header.append("\n")
    header.append("─" * 85, style="dim")
    header.append("\n")
    header.append("Currently supported operators:  ACLNN", style="italic dim")
    console.print(Panel(header, border_style="cyan", expand=False))


def _is_validation_passed(v) -> bool:
    if isinstance(v, dict):
        consistency_passed = v.get("consistency_passed", False)
        metric_passed = v.get("metric_passed")
    else:
        consistency_passed = getattr(v, "consistency_passed", False)
        metric_passed = getattr(v, "metric_passed", None)

    if not consistency_passed:
        return False

    return metric_passed is None or metric_passed == "N/A" or metric_passed == "PASS"


def run_full_pipeline(
    evidence_file: EvidenceFile,
    doc_part: str,
    log_capture: LogCapture,
    style_dir: Optional[Path] = None,
    change_reason: Optional[dict] = None,
    **kwargs,
) -> PipelineResult:
    result = PipelineResult()
    start = time.time()
    modified_files = kwargs.get("modified_files", [])

    try:
        llm = LLMClient()

        if doc_part == "example":
            dashboard_step("验证文档模块")
            modified_op_dir = kwargs.get("modified_op_dir", "")
            operator_name = kwargs.get("operator_name", "")
            modified_example = get_all_example_codes(modified_op_dir, operator_name)

            all_success = True
            lines = [f"\n### {describe}\n\n```cpp\n{code}\n```" for describe, code in modified_example.items()]
            generated_contents = (config.EXAMPLE_PREFIX + "\n" + "\n".join(lines)) if lines else ""
            result.generated = [
                {
                    "sub_key": "",
                    "generated_text": generated_contents,
                    "modification_summary": {},
                }
            ]

            if not modified_example:
                all_success = False
                result.success = False
                result.error = "See the detailed error information above."

            validation_result = ValidationResult(
                consistency_passed=all_success,
                accuracy_rate=1.0 if all_success else 0.0,
                metric_passed="PASS" if all_success else "FAIL",
                edit_distance_ratio=1.0 if all_success else 0.0,
                avg_confidence=1.0 if all_success else 0.0,
                issues=[],
                suggestion="Generated for example.",
                passed=True,
            )
            result.validation = [validation_result]
            dashboard_progress(1, 1)
            dashboard_done()
        else:
            if get_dashboard():
                dashboard_step("加载代码证据")
                dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 1:[/bold] 加载代码证据")
            else:
                console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 1:[/bold] 加载代码证据")
            evidence_stage, raw_input, parsed, ranked, _, _ = setup_pipeline(
                str(evidence_file.path), llm, deduplicate=False
            )
            if get_dashboard():
                dashboard_log(f"  [green]✓[/green] 加载 {parsed.total_evidences} 个证据")
                targets_summary = ", ".join(f"{k}({len(v)})" for k, v in ranked.evidences_by_target.items())
                dashboard_log(f"  [dim]  targets: {targets_summary}[/dim]")
            else:
                console.print(f"  [green]✓[/green] 加载 {parsed.total_evidences} 个证据")
                console.print(
                    f"  [dim]  目标: {', '.join(f'{k}({len(v)})' for k, v in ranked.evidences_by_target.items())}[/dim]"
                )

            if get_dashboard():
                dashboard_step("编排代码证据")
                dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 2:[/bold] 编排代码证据")
            else:
                console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 2:[/bold] 编排代码证据")
            get_output_dir()

            extracted, source_conflicts = evidence_stage.extract_evidence(
                ranked=ranked,
                doc_part=doc_part,
                doc_original=parsed.doc_original_parsed,
                progressive=True,
                evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
                modified_files=modified_files,
            )
            if get_dashboard():
                dashboard_log(f"  [green]✓[/green] Extracted fields: {', '.join(extracted.keys())}")
            else:
                console.print(f"  [green]✓[/green] 提取字段: {', '.join(extracted.keys())}")

            definite_conflicts = [c for c in source_conflicts if c.get("definite", False)]
            uncertain_conflicts = [c for c in source_conflicts if not c.get("definite", False)]
            if uncertain_conflicts:
                uncertain_msg = "; ".join(f"{c.get('fact', '?')}: {c.get('values', [])}" for c in uncertain_conflicts)
                log_info(f"[{config.format_doc_part(doc_part)}] 忽略不能完全确定的证据差异（不告警）: {uncertain_msg}")

            if definite_conflicts:
                conflict_details = "; ".join(f"{c.get('fact', '?')}: {c.get('values', [])}" for c in definite_conflicts)
                result.evidence_conflict = True
                result.conflict_details = conflict_details
                conflict_msg = f"提取阶段检测到明确的证据冲突: {conflict_details}"
                if get_dashboard():
                    dashboard_log(f"  [red]⚠[/red] {conflict_msg}")
                    db = get_dashboard()
                    dp = get_current_doc_part()
                    if db and dp:
                        db.update(
                            dp,
                            status="✗",
                            result="FAIL",
                            progress_current=1,
                            progress_total=1,
                        )
                else:
                    console.print(f"  [red]⚠[/red] {conflict_msg}")
                log_info(f"[{config.format_doc_part(doc_part)}] {conflict_msg}")
                result.generated = []
                result.validation = []
                result.runtime_seconds = time.time() - start
                return result

            if get_dashboard():
                dashboard_step("加载领域信息")
                dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 3:[/bold] 加载领域信息")
            else:
                console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 3:[/bold] 加载领域信息")
            style_stage = StyleStage(llm, style_dir)
            style_summary = style_stage.get_style_summary()
            style_extraction = style_stage.get_style_extraction()
            if get_dashboard():
                dashboard_log(f"  [green]✓[/green] Core topic: {style_summary.core_topic}")
            else:
                console.print(f"  [green]✓[/green] 核心主题: {style_summary.core_topic}")

            if get_dashboard():
                dashboard_step("生成文档")
                dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 4:[/bold] 生成文档模块")
            else:
                console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 4:[/bold] 生成文档模块")
            validation_results_list = []
            final_results = []
            history: list[dict] = []

            if doc_part == "support":
                support_matrix = extracted.get("support_matrix", {})
                md_content = support_matrix.get("md_content")
                if get_dashboard():
                    dashboard_log("  [green]✓[/green] Support matrix generated")
                    dashboard_step("验证文档模块")
                    dashboard_log(
                        f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 5:[/bold] 验证文档模块"
                    )
                else:
                    console.print("  [green]✓[/green] 支持矩阵已生成")
                    console.print(
                        f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 5:[/bold] 验证文档模块"
                    )

                validation_stage = ValidationStage(llm)
                parsed_dedup = evidence_stage.parser.parse_raw_input(raw_input, deduplicate=True)
                doc_original_for_validation = parsed.doc_original_md[0] if parsed.doc_original_md else ""
                validation_result = validation_stage.validate(
                    doc_part=doc_part,
                    doc_original=doc_original_for_validation,
                    evidences=parsed_dedup.evidences,
                    doc_generated=md_content,
                    evidences_extracted=extracted,
                    check_metric=True,
                )
                ValidationStage.print_result(validation_result)
                result_dict = ValidationStage.format_result(validation_result)
                result_dict["passed"] = _is_validation_passed(result_dict)
                validation_results_list.append(result_dict)

                status_icon = "[green]✓ PASS[/green]" if result_dict.get("passed") else "[red]✗ FAIL[/red]"
                if get_dashboard():
                    dashboard_log(f"  {status_icon} support_matrix (acc={result_dict.get('accuracy_rate', 0):.2f})")
                else:
                    console.print(f"  {status_icon} support_matrix (acc={result_dict.get('accuracy_rate', 0):.2f})")
                final_results.append(
                    {
                        "sub_key": "support_matrix",
                        "generated_text": md_content,
                        "modification_summary": {},
                    }
                )
            else:
                generation_stage = GenerationStage(llm, style_stage.template_loader)
                validation_stage = ValidationStage(llm)

                previous_issues = []
                accumulated_issues = []
                previous_generated_doc: dict[str, str] = {}
                all_passed = False
                passed_indices = []

                for attempt in range(config.MAX_REGENERATION_RETRIES):
                    if attempt > 0:
                        msg = f"Regeneration attempt {attempt + 1}/{config.MAX_REGENERATION_RETRIES}"
                        if get_dashboard():
                            dashboard_log(f"  [yellow]→[/yellow] {msg}")
                        else:
                            console.print(f"\n  [yellow]→[/yellow] {msg}")
                        log_info(f"[{config.format_doc_part(doc_part)}] {msg}")
                        if passed_indices:
                            skip_msg = f"Skipping {len(passed_indices)} already-passed section(s)"
                            if get_dashboard():
                                dashboard_log(f"  [dim]  {skip_msg}[/dim]")
                            else:
                                console.print(f"  [dim]  {skip_msg}[/dim]")
                            log_info(f"[{config.format_doc_part(doc_part)}] {skip_msg}")

                    if get_dashboard():
                        dashboard_step("生成文档")
                    generated_results = generation_stage.generate(
                        original_content={
                            "parsed": parsed.doc_original_parsed,
                            "md": parsed.doc_original_md,
                        },
                        evidences=extracted,
                        style_summary=style_summary,
                        style_extraction=style_extraction,
                        doc_part=doc_part,
                        previous_issues=previous_issues,
                        generated_doc=previous_generated_doc,
                        passed_indices=passed_indices,
                        change_reason=change_reason,
                    )
                    if get_dashboard():
                        dashboard_log(f"  [dim]  Generated {len(generated_results)} sections[/dim]")
                    else:
                        console.print(f"  [dim]  生成 {len(generated_results)} 个模块[/dim]")
                        display_generation_results(generated_results)

                    all_no_change = True
                    any_conflict = False
                    conflict_reasons = []
                    for res in generated_results:
                        analysis = res.modification_summary.get("analysis") or {}
                        if analysis.get("need_change", True):
                            all_no_change = False
                        if analysis.get("evidence_conflict", False):
                            any_conflict = True
                            conflict_reasons.append(analysis.get("reason", ""))

                    if all_no_change and not any_conflict and generated_results:
                        no_change_reason = (generated_results[0].modification_summary.get("analysis") or {}).get(
                            "reason", ""
                        )
                        result.no_change_needed = True
                        result.conflict_details = no_change_reason
                        result.attempts_used = attempt + 1
                        final_results.clear()
                        for i, res in enumerate(generated_results):
                            sub_key = res.modification_summary.get("sub_key", f"section_{i + 1}")
                            final_results.append(
                                {
                                    "sub_key": sub_key,
                                    "generated_text": res.generated_text,
                                    "modification_summary": res.modification_summary,
                                }
                            )
                        no_change_msg = f"LLM 判断无需修改: {no_change_reason}"
                        if get_dashboard():
                            dashboard_log(f"  [yellow]→[/yellow] {no_change_msg}")
                        else:
                            console.print(f"  [yellow]→[/yellow] {no_change_msg}")
                        log_info(f"[{config.format_doc_part(doc_part)}] {no_change_msg}")
                        all_passed = True
                        break

                    if any_conflict:
                        result.evidence_conflict = True
                        result.conflict_details = "; ".join(conflict_reasons)
                        conflict_msg = f"Writer 检测到证据冲突: {result.conflict_details}"
                        if get_dashboard():
                            dashboard_log(f"  [yellow]⚠[/yellow] {conflict_msg}")
                        else:
                            console.print(f"  [yellow]⚠[/yellow] {conflict_msg}")
                        log_info(f"[{config.format_doc_part(doc_part)}] {conflict_msg}")

                    if attempt == 0:
                        if get_dashboard():
                            dashboard_step("验证文档模块")
                            dashboard_log(
                                f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 5:[/bold] 验证文档模块"
                            )
                        else:
                            console.rule("[bold cyan]Code - Doc Validating[/bold cyan]")
                            console.print(
                                f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]步骤 5:[/bold] 验证文档模块"
                            )

                    validation_results_list.clear()
                    final_results.clear()
                    all_passed = True
                    current_issues = []
                    current_generated_doc: dict[str, str] = {}
                    current_passed_indices = []

                    parsed_dedup = evidence_stage.parser.parse_raw_input(raw_input, deduplicate=True)
                    evidences_for_validation = parsed_dedup.evidences
                    _orig = parsed.total_evidences
                    _dedup = _orig - parsed_dedup.total_evidences
                    if _dedup > 0:
                        if get_dashboard():
                            dedup_total = parsed_dedup.total_evidences
                            dashboard_log(
                                f"  [dim]Dedup evidences for validation: {_orig} -> {dedup_total} "
                                f"(removed {_dedup} redundant)[/dim]"
                            )
                        else:
                            console.print(
                                f"  [dim]验证前去重证据: {_orig} -> {parsed_dedup.total_evidences} (移除 {_dedup} 个重复)[/dim]"
                            )

                    total_sections = len(generated_results)
                    has_unresolvable = False
                    # 签名语料：本 doc_part 全部原章节 + 证据链原文 + 编排后证据（供参数放行判断；
                    # changed_only 证据被过滤掉 aclnn 签名时由辅助函数回补兄弟全量证据）
                    sig_corpus = build_sig_corpus(parsed.doc_original_md, raw_input, extracted, evidence_file.path)
                    # 原文档全文（跨 doc_part），供约束节参数越界检查取到其他小节的参数名；
                    # anchor 的 doc_path 可能是目录（须拼 doc_name），为相对目录时经证据路径回溯
                    orig_doc_full = resolve_orig_doc_full(
                        raw_input.doc.doc_path,
                        raw_input.doc.doc_name,
                        evidence_file.path,
                    )
                    for i, res in enumerate(generated_results):
                        validation_result = validation_stage.validate(
                            doc_part=doc_part,
                            doc_original=parsed.doc_original_md[i] if i < len(parsed.doc_original_md) else "",
                            evidences=evidences_for_validation,
                            doc_generated=res.generated_text,
                            evidences_extracted=extracted,
                            check_metric=True,
                            idx=i + 1,
                        )
                        ValidationStage.print_result(validation_result)
                        result_dict = ValidationStage.format_result(validation_result)

                        # 确定性巡检：链接改写 / 参数表越界 → 计入 issues 触发重试再生成
                        orig_section = parsed.doc_original_md[i] if i < len(parsed.doc_original_md) else ""
                        new_issues = collect_issues(
                            orig_section,
                            res.generated_text,
                            doc_part,
                            sig_corpus,
                            orig_doc_full,
                        )
                        if new_issues:
                            result_dict["consistency_passed"] = False
                            result_dict["issues"] = result_dict.get("issues", []) + new_issues
                            for issue in new_issues:
                                log_info(f"[{config.format_doc_part(doc_part)}] [{issue['id']}] {issue['reason']}")

                        passed = _is_validation_passed(result_dict)
                        result_dict["passed"] = passed

                        status_icon = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
                        sub_key = res.modification_summary.get("sub_key", f"section_{i + 1}")
                        if get_dashboard():
                            dashboard_progress(i + 1, total_sections)
                            dashboard_log(
                                f"  {status_icon} {sub_key} (acc={result_dict.get('accuracy_rate', 0):.2f}, "
                                f"metric={result_dict.get('metric_passed', 'N/A')})"
                            )
                        else:
                            console.print(
                                f"  {status_icon} {sub_key} (acc={result_dict.get('accuracy_rate', 0):.2f}, "
                                f"metric={result_dict.get('metric_passed', 'N/A')})"
                            )

                        if not passed:
                            all_passed = False
                            issues = result_dict.get("issues", [])
                            if issues:
                                for issue in issues:
                                    issue_id = issue.get("id", "?")
                                    reason = issue.get("reason", "")
                                    issue["sub_key"] = sub_key
                                    if get_dashboard():
                                        dashboard_log(f"    [yellow]→[/yellow] [{issue_id}] {reason}")
                                    else:
                                        console.print(f"    [yellow]→[/yellow] [{issue_id}] {reason}")
                            current_issues.extend(issues)
                            current_generated_doc[sub_key] = res.generated_text
                            # metric（编辑距离启发式）FAIL 不再视为不可调和：否则一处 metric
                            # 失败会跳过整节重试，其余可修复的 consistency 问题永远得不到再生成
                            if any(issue.get("unresolvable", False) for issue in issues):
                                has_unresolvable = True
                                unresolvable_issues = [iss for iss in issues if iss.get("unresolvable", False)]
                                result.evidence_conflict = True
                                result.conflict_details = "; ".join(
                                    iss.get("reason", "") for iss in unresolvable_issues
                                )
                                unresolvable_msg = f"检测到不可调和的证据冲突: {result.conflict_details}"
                                if get_dashboard():
                                    dashboard_log(f"  [red]⚠[/red] {unresolvable_msg}")
                                else:
                                    console.print(f"  [red]⚠[/red] {unresolvable_msg}")
                                log_info(f"[{config.format_doc_part(doc_part)}] {unresolvable_msg}")
                        else:
                            current_passed_indices.append(i)

                        validation_results_list.append(result_dict)
                        final_results.append(
                            {
                                "sub_key": res.modification_summary.get("sub_key"),
                                "generated_text": res.generated_text,
                                "modification_summary": res.modification_summary,
                            }
                        )

                    # 记录本次 attempt 的完整快照（生成内容 + 注入的 issues + 验证结果）
                    history.append(
                        {
                            "attempt": attempt + 1,
                            "injected_issues": [issue.copy() for issue in previous_issues],
                            "generated_sections": [
                                {
                                    "sub_key": item.get("sub_key"),
                                    "generated_text": item.get("generated_text"),
                                    "modification_summary": item.get("modification_summary"),
                                }
                                for item in final_results
                            ],
                            "validation_results": [
                                {k: v for k, v in vdict.items()} for vdict in validation_results_list
                            ],
                        }
                    )

                    if has_unresolvable:
                        result.attempts_used = attempt + 1
                        break
                    elif all_passed:
                        result.attempts_used = attempt + 1
                        break
                    elif attempt < config.MAX_REGENERATION_RETRIES - 1:
                        for issue in current_issues:
                            tagged = issue.copy()
                            tagged["from_attempt"] = attempt + 1
                            accumulated_issues.append(tagged)
                        previous_issues = accumulated_issues
                        previous_generated_doc = current_generated_doc
                        passed_indices = current_passed_indices
                        if get_dashboard():
                            dashboard_log(
                                f"  [yellow]→[/yellow] Collecting {len(current_issues)} issues "
                                f"({len(accumulated_issues)} total accumulated) for regeneration..."
                            )
                        else:
                            console.print(
                                f"  [yellow]→[/yellow] 收集 {len(current_issues)} 个问题 "
                                f"(累计 {len(accumulated_issues)} 个) 用于重新生成..."
                            )
                    else:
                        result.max_retries_hit = True
                        result.attempts_used = attempt + 1
                        max_msg = (
                            f"已达到最大重新生成次数 ({config.MAX_REGENERATION_RETRIES})。" f"使用最后的结果继续。"
                        )
                        if get_dashboard():
                            dashboard_log(f"  [yellow]{max_msg}[/yellow]")
                        else:
                            console.print(f"  [yellow]{max_msg}[/yellow]")
                        log_warning(f"[{config.format_doc_part(doc_part)}] {max_msg}")

            result.parsed = {
                "doc_part": parsed.doc_part,
                "total_evidences": parsed.total_evidences,
                "evidences_by_target": ranked.evidences_by_target,
            }
            result.extracted = extracted
            result.style_summary = {
                "core_topic": style_summary.core_topic,
                "key_points": style_summary.key_points,
                "context_dependencies": style_summary.context_dependencies,
            }
            result.style_extraction = {
                "terminology": style_extraction.terminology,
                "structure": style_extraction.structure,
                "format": style_extraction.format,
            }
            result.generated = final_results
            result.validation = validation_results_list
            result.history = history

    except Exception as e:
        result.success = False
        result.error = str(e)
        if get_dashboard():
            dashboard_error(f"Error: {e}")
        else:
            console.print(f"\n[red]✗ 错误:[/red] {e}")

    if not result.success:
        dashboard_result(None)
    elif result.evidence_conflict:
        db = get_dashboard()
        dp = get_current_doc_part()
        if db and dp:
            db.update(dp, status="✗", result="FAIL")
    elif result.no_change_needed:
        db = get_dashboard()
        dp = get_current_doc_part()
        if db and dp:
            db.update(dp, status="⚠", result="SKIP")
    else:
        all_passed = all(_is_validation_passed(v) for v in result.validation)
        dashboard_result(all_passed)
        dashboard_done()
    result.runtime_seconds = time.time() - start
    return result


def run_validate_only(evidence_file: EvidenceFile, doc_part: str, log_capture: LogCapture, **kwargs) -> PipelineResult:
    result = PipelineResult()
    start = time.time()

    try:
        if doc_part == "example":
            original_op_dir = kwargs.get("original_op_dir", "")
            doc_json_dir = kwargs.get("doc_json_dir", "")

            doc_example = get_example_doc_part(doc_json_dir)
            describe_list = list(doc_example.keys()) if doc_example else []
            code_example = get_example_codes_by_describe(original_op_dir, describe_list)

            all_passed = True
            per_example_results = []
            issue_id = 0
            total_examples = len(doc_example)
            completed = 0

            for describe, doc_code in doc_example.items():
                disk_code = code_example.get(describe, "")
                old_single = {"": doc_code} if doc_code else {}
                new_single = {"": disk_code} if disk_code else {}
                val_result = validate_example(old_single, new_single)

                if val_result.get("results", "FAIL") != "PASS":
                    hunks = val_result.get("diff_hunks") or [
                        {
                            "location": "doc_json vs source_file",
                            "reason": "example code mismatch",
                        }
                    ]
                    for hunk in hunks:
                        issue_id += 1
                        per_example_results.append(
                            {
                                "id": issue_id,
                                "location": hunk["location"],
                                "reason": hunk["reason"],
                                "describe": describe,
                            }
                        )
                    all_passed = False
                    dashboard_log(f"  [red]✗ FAIL[/red] 描述: [cyan]{describe}[/cyan]")
                else:
                    dashboard_log(f"  [green]✓ PASS[/green] 描述: [cyan]{describe}[/cyan]")

                completed += 1
                dashboard_progress(completed, total_examples)

            validation_result = ValidationResult(
                consistency_passed=all_passed,
                accuracy_rate=1.0 if all_passed else 0.0,
                metric_passed="PASS" if all_passed else "FAIL",
                edit_distance_ratio=1.0 if all_passed else 0.0,
                avg_confidence=1.0 if all_passed else 0.0,
                issues=per_example_results,
                suggestion="",
                passed=all_passed,
            )
            result.validation = [validation_result]

        else:
            llm = LLMClient()

            dashboard_step("Load Evidence")
            dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 1:[/bold] 加载代码证据")
            evidence_stage, raw_input, parsed, ranked, _, _ = setup_pipeline(
                str(evidence_file.path), llm, deduplicate=True
            )
            _orig = len(raw_input.anchor_tuple)
            _dedup = _orig - parsed.total_evidences
            _dedup_info = (
                f" [dim](去重: {_orig} -> {parsed.total_evidences}, 移除 {_dedup} 个重复)[/dim]" if _dedup > 0 else ""
            )
            dashboard_log(f"  [green]✓[/green] 加载 {parsed.total_evidences} 个证据{_dedup_info}")

            dashboard_step("验证文档模块")
            dashboard_log(f"[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]步骤 2:[/bold] 验证文档模块")
            validation_stage = ValidationStage(llm)
            validation_results_list = []

            extracted = None
            if doc_part == "support":
                extracted, _ = evidence_stage.extract_evidence(
                    ranked=ranked,
                    doc_part=doc_part,
                    doc_original=parsed.doc_original_md[0] if parsed.doc_original_md else "",
                    progressive=True,
                    evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
                )
                indices = [None]
            elif doc_part == "sub_function_hash":
                indices = validation_stage.get_sub_function_indices(parsed)
            else:
                indices = [None]

            for idx in indices:
                if isinstance(idx, int):
                    doc_original = parsed.doc_original_md[idx - 1] if idx - 1 < len(parsed.doc_original_md) else ""
                else:
                    doc_original = parsed.doc_original_md[0] if parsed.doc_original_md else ""
                validation_result = validation_stage.validate(
                    doc_part=doc_part,
                    doc_original=doc_original,
                    evidences=parsed.evidences,
                    evidences_extracted=extracted,
                    check_metric=False,
                    idx=idx,
                )
                result_dict = ValidationStage.format_result(validation_result)
                ValidationStage.print_result(validation_result)
                validation_result.passed = _is_validation_passed(result_dict)
                validation_results_list.append(validation_result)

            result.parsed = {
                "doc_part": parsed.doc_part,
                "total_evidences": parsed.total_evidences,
            }
            result.validation = validation_results_list

    except Exception as e:
        result.success = False
        result.error = str(e)
        dashboard_error(f"Error: {e}")

    if not result.success:
        dashboard_result(None)
    else:
        all_passed = all(_is_validation_passed(v) for v in result.validation)
        dashboard_result(all_passed)
    result.runtime_seconds = time.time() - start
    return result


def display_generation_results(results: list, mode: str = "full"):
    if get_dashboard():
        return
    if not results:
        console.print("  [dim]未生成任何模块[/dim]")
        return

    table = Table(show_header=False, border_style="cyan", box=SIMPLE)
    table.add_column(style="cyan")
    table.add_column(highlight=False, overflow="fold")

    for i, item in enumerate(results):
        sub_key = ""
        text = ""
        if hasattr(item, "modification_summary"):
            sub_key = item.modification_summary.get("sub_key", f"section_{i + 1}")
            text = item.generated_text
        elif isinstance(item, dict):
            sub_key = item.get("sub_key", f"section_{i + 1}")
            text = item.get("generated_text", "")
        else:
            sub_key = f"section_{i + 1}"
            text = str(item)

        label = f"[bold]{sub_key}[/bold]" if sub_key else f"[bold]section_{i + 1}[/bold]"
        content_panel = Panel(text, border_style="cyan", expand=False)
        table.add_row(label, content_panel)

    console.print(Panel(table, title="已生成模块", border_style="cyan", expand=False))


def display_results(result: PipelineResult, mode: str):
    console.print()

    if result.evidence_conflict:
        status_text = f"[bold red]⚠ Evidence conflict: {result.conflict_details or 'see logs'}[/bold red]"
        border_style = "red"
        title = "Results"
    elif result.no_change_needed:
        status_text = f"[bold yellow]⚠ No change needed: {result.conflict_details or 'see logs'}[/bold yellow]"
        border_style = "yellow"
        title = "Results"
    else:
        has_failures = any(not _is_validation_passed(v) for v in result.validation)
        if has_failures:
            failed_icon = "[red]✗ Validations failed[/red]"
            status_text = failed_icon
            border_style = "red"
            title = "Results"
        else:
            status_text = "[bold green]✓ All validations passed[/bold green]"
            border_style = "cyan"
            title = "Results"

    runtime_text = f"{result.runtime_seconds:.2f}s"

    if mode == "validate_only":
        summary_table = Table(show_header=False, border_style=border_style, box=SIMPLE)
        summary_table.add_column(style="cyan")
        summary_table.add_column(highlight=False, overflow="fold")

        if result.validation:
            v_status = (
                "[green]PASS[/green]" if all(_is_validation_passed(v) for v in result.validation) else "[red]FAIL[/red]"
            )
        else:
            v_status = "[red]FAIL[/red]"
        summary_table.add_row(Text("Status", style="bold"), v_status)
        summary_table.add_row(Text("Runtime", style="bold"), runtime_text)
        summary_table.add_row(Text("Attempts", style="bold"), str(result.attempts_used))
        if result.max_retries_hit:
            summary_table.add_row(Text("Max retries", style="bold"), "[bold yellow]HIT[/bold yellow]")

        for vi, v in enumerate(result.validation):
            vdict = v if isinstance(v, dict) else v.model_dump()

            if len(result.validation) > 1:
                v_passed = _is_validation_passed(v)
                v_icon = "[green]PASS[/green]" if v_passed else "[red]FAIL[/red]"
                summary_table.add_row(Text(""), "")
                summary_table.add_row(Text(f"── Section {vi + 1} ──", style="bold"), v_icon)

            summary_table.add_row(
                Text("consistency_passed:", style="bold cyan"),
                str(vdict.get("consistency_passed")),
            )
            summary_table.add_row(
                Text("accuracy_rate:", style="bold cyan"),
                str(vdict.get("accuracy_rate")),
            )
            summary_table.add_row(
                Text("metric_passed:", style="bold cyan"),
                str(vdict.get("metric_passed")),
            )
            summary_table.add_row(
                Text("edit_distance_ratio:", style="bold cyan"),
                str(vdict.get("edit_distance_ratio")),
            )
            summary_table.add_row(
                Text("avg_confidence:", style="bold cyan"),
                str(vdict.get("avg_confidence")),
            )
            summary_table.add_row(Text("passed:", style="bold cyan"), str(vdict.get("passed", "?")))

            issues = vdict.get("issues", [])
            if issues and isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        issue_lines = []
                        issue_lines.append(f"  [yellow]id:[/yellow] {issue.get('id', '?')}")
                        issue_lines.append(f"  [yellow]location:[/yellow] {issue.get('location', '?')}")
                        issue_lines.append(f"  [yellow]reason:[/yellow] {issue.get('reason', '')}")
                        suggestion = issue.get("suggestion", "")
                        if suggestion and str(suggestion) != "None":
                            issue_lines.append(f"  [yellow]suggestion:[/yellow] {suggestion}")
                        evidence = issue.get("evidence", "")
                        if evidence and str(evidence) != "None":
                            issue_lines.append(f"  [yellow]evidence:[/yellow] {evidence}")
                        summary_table.add_row(Text("", style="dim"), "")
                        summary_table.add_row(Text("Issue", style="bold red"), "\n".join(issue_lines))
            else:
                suggestion = vdict.get("suggestion", "")
                if suggestion and str(suggestion) != "None" and str(suggestion):
                    summary_table.add_row(Text("Suggestion", style="bold cyan"), str(suggestion))

        console.print(Panel(summary_table, title=title, border_style=border_style, expand=False))
        return

    else:
        if result.generated:
            content_parts = []
            for i, item in enumerate(result.generated):
                sub_key = item.get("sub_key", "")
                text = item.get("generated_text", "")
                if sub_key:
                    content_parts.append(f"\n### {sub_key}\n\n{text}")
                else:
                    content_parts.append(text)

            if content_parts:
                content_text = "\n\n".join(content_parts).lstrip("\n")
                outer_table = Table(show_header=False, border_style=border_style, box=SIMPLE)
                outer_table.add_column(style="cyan")
                outer_table.add_column(overflow="fold")
                outer_table.add_row("[bold]Status[/bold]", status_text)
                outer_table.add_row("[bold]Runtime[/bold]", runtime_text)
                outer_table.add_row("[bold]Attempts[/bold]", str(result.attempts_used))
                if result.max_retries_hit:
                    outer_table.add_row("[bold]Max retries[/bold]", "[bold yellow]HIT[/bold yellow]")
                outer_table.add_row()
                outer_table.add_row("[bold cyan]Generated Content[/bold cyan]", "")
                content_panel = Panel(content_text, border_style=border_style, expand=False)
                outer_table.add_row("", content_panel)
                console.print(
                    Panel(
                        outer_table,
                        title=title,
                        border_style=border_style,
                        expand=False,
                    )
                )
            else:
                table = Table(
                    show_header=True,
                    header_style="bold cyan",
                    border_style=border_style,
                )
                table.add_column("Status", style="bold")
                table.add_column("Runtime", justify="right")
                table.add_column("Attempts", justify="right")
                table.add_row(status_text, runtime_text, str(result.attempts_used))
                console.print(Panel(table, title=title, border_style=border_style, expand=False))
        elif result.validation:
            # No generated content, show validation details
            table = Table(show_header=True, header_style="bold cyan", border_style=border_style)
            table.add_column("Describe", style="cyan")
            table.add_column("Doc Length", justify="right")
            table.add_column("Disk Length", justify="right")
            table.add_column("Result", style="bold")
            for vi, v in enumerate(result.validation):
                vdict = v if isinstance(v, dict) else v.model_dump()

                if len(result.validation) > 1:
                    v_passed = _is_validation_passed(v)
                    v_icon = "[green]PASS[/green]" if v_passed else "[red]FAIL[/red]"
                    table.add_row(Text(""), "")
                    table.add_row(Text(f"── Section {vi + 1} ──", style="bold"), v_icon)

                table.add_row(
                    Text("consistency_passed:", style="bold cyan"),
                    str(vdict.get("consistency_passed")),
                )

            console.print(Panel(table, title=title, border_style=border_style, expand=False))
        else:
            table = Table(show_header=True, header_style="bold cyan", border_style=border_style)
            table.add_column("Status", style="bold")
            table.add_column("Runtime", justify="right")
            table.add_column("Attempts", justify="right")
            table.add_row(status_text, runtime_text, str(result.attempts_used))
            console.print(Panel(table, title=title, border_style=border_style, expand=False))


def run_cli():
    screen_console = Console()
    if sys.stdout.isatty():
        screen_console.clear()
    try:
        _run_cli_impl(screen_console)
    finally:
        if sys.stdout.isatty():
            screen_console.clear()


def _run_cli_impl(console: Console):
    setup_logger(console=False)

    parser = argparse.ArgumentParser(description="Doc IncrGen&Val - CLI")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to evidence file (relative to data/evidences/)",
    )
    parser.add_argument("--doc-part", "-d", type=str, choices=config.DOC_PARTS, help="Document part")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["full", "validate_only"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available evidence files and exit",
    )
    parser.add_argument(
        "--ops-dir",
        "-o",
        type=str,
        default="",
        help="Operator source directory path (required for example doc_part)",
    )
    args = parser.parse_args()

    files = scan_evidence_files(config.EVIDENCE_EXAMPLE_DIR)

    if not files:
        console.print("[red]✗ 在 data/evidences/ 中未找到证据文件[/red]")
        sys.exit(1)

    if args.list:
        console.print(Panel("可用证据文件", border_style="cyan"))
        grouped = group_files_by_doc_part(files)
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Doc Part", style="cyan")
        table.add_column("File", style="white")
        for doc_part in config.DOC_PARTS:
            if doc_part in grouped:
                for f in grouped[doc_part]:
                    table.add_row(doc_part, f.relative_path)
        console.print(table)
        sys.exit(0)

    if sys.stdout.isatty():
        console.clear()
    display_header()
    console.print("[dim]Scanning evidence files...[/dim]")
    console.print(f"[green]✓[/green] 找到 [bold]{len(files)}[/bold] 个证据文件")
    console.print()

    selected_file = select_evidence_file(files)
    console.print(f"  [dim]已选择:[/dim] {selected_file.relative_path}")

    doc_part = confirm_doc_part(selected_file.doc_part)
    console.print(f"  [dim]Doc part:[/dim] [cyan]{doc_part}[/cyan]")

    console.print()
    mode = select_mode()
    console.print(f"  [dim]模式:[/dim] [yellow]{mode}[/yellow]")

    if not selected_file or not doc_part or not mode:
        return

    console.print()

    log_capture = LogCapture()

    ops_dir = args.ops_dir
    doc_json_dir = args.doc_json_dir if hasattr(args, "doc_json_dir") else ""
    if doc_part == "example":
        if not ops_dir:
            ops_dir = questionary.text(
                "Enter operator source directory (path containing operator/examples/*.cpp):",
                style=console_style,
                default="./data/outputs/snapshot/operator_storage/addcmul/base",
            ).ask()
        if not ops_dir:
            console.print("[red]✗ 调用示例 doc_part 需要算子目录[/red]")
            return
        ops_dir = ops_dir.strip()
        if mode == "validate_only":
            if not doc_json_dir:
                doc_json_dir = questionary.text(
                    "Enter doc_json directory (path containing the JSON file with example content):",
                    style=console_style,
                    default="./data/outputs/snapshot/addcmul/base/doc_json",
                ).ask()
            if doc_json_dir:
                doc_json_dir = doc_json_dir.strip()

    kwargs = {}
    if doc_part == "example":
        kwargs["modified_op_dir"] = ops_dir
        kwargs["original_op_dir"] = ops_dir
        kwargs["doc_json_dir"] = doc_json_dir

    if mode == "full":
        console.rule("[bold cyan]Doc Generating[/bold cyan]")
        result = run_full_pipeline(selected_file, doc_part, log_capture, **kwargs)
    else:
        console.rule("[bold cyan]Code - Doc Validating[/bold cyan]")
        result = run_validate_only(selected_file, doc_part, log_capture, **kwargs)

    display_results(result, mode)

    console.print()
    if questionary.confirm("Run again?", default=False).ask():
        _run_cli_impl(console)


if __name__ == "__main__":
    # TEST CMD: python -m src.generation_validation.ui.cli
    run_cli()
