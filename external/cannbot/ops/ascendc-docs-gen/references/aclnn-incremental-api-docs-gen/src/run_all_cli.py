# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Run All CLI - Entry point orchestrating generation and validation workflows.

Usage:
  Interactive:  python src/run_all_cli.py
  Non-interactive:
    python src/run_all_cli.py --original_op_repo <repo> --original_op_path <path> \
        --modified_op_repo <repo> --modified_op_path <path> \
        [--md-stem <stem> [<stem> ...]] [--validate-only] [--skip-fix]
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import questionary
from rich.console import Console
from rich.text import Text

from src.change_detection.doc_part_change_detection.docpart_file_extractor import (
    DocpartFileExtractor,
)
from src.generation_validation.config import (
    ENABLE_EVIDENCE_FILE_FILTER,
    ENABLE_PRE_VALIDATION,
    GEN_VAL_MAX_WORKERS,
    POST_ANALYZE_NO_CHANGE_THRESHOLD,
    format_doc_part,
    format_doc_parts,
)
from src.generation_validation.dashboard import (
    ParallelDashboard,
    set_dashboard_context,
)
from src.generation_validation.ui.cli import (
    LogCapture,
    PipelineResult,
    display_header,
)
from src.generation_validation.ui.cli import display_results as _display_results
from src.generation_validation.ui.cli import (
    resolve_evidence_file,
    run_full_pipeline,
    run_validate_only,
    select_doc_part,
    select_mode,
)
from src.logger import (
    debug,
    error,
    highlight,
    info,
    redirect_stdout_to_log,
    section,
    setup_logger,
    subsection,
    warning,
)
from src.settings import apply_settings, load_settings
from src.snapshot_construction.anchor_construction.common.doc_selector import (
    select_doc_file,
)

CANCEL_SENTINEL = "CANCEL_SENTINEL"


@dataclass
class PipelineContext:
    """Holds all paths, configuration, and mutable state for a single pipeline run."""

    workspace_dir: Path
    output_data_path: Path
    base_output_data_path: Path = field(default_factory=lambda: Path("."))
    original_op_repo: Optional[Path] = None
    original_op_dir: Optional[Path] = None
    modified_op_repo: Optional[Path] = None
    modified_op_dir: Optional[Path] = None
    modified_op_dir_md_stem: Optional[Path] = None
    selected_op: str = ""
    md_stems: list[str] = field(default_factory=list)
    validate_only: bool = False
    skip_fix: bool = False
    interactive: bool = False
    dashboard: bool = False

    selected_md_stem: str = ""
    version_dir: Path = field(default_factory=lambda: Path("."))
    snapshot_op_dir: Path = field(default_factory=lambda: Path("."))
    evidence_dir: Path = field(default_factory=lambda: Path("."))
    summary_dir: Path = field(default_factory=lambda: Path("."))
    selected_file: Optional[Path] = None
    doc_content: str = ""
    summary_file_dir: Path = field(default_factory=lambda: Path("."))
    re_detection_output_path: Path = field(default_factory=lambda: Path("."))

    doc_parts: list[str] = field(default_factory=list)
    modified_files: list = field(default_factory=list)
    evaluation_reasons: dict[str, dict] = field(default_factory=dict)
    repo_other_files: list = field(default_factory=list)
    backfill_map: dict[str, str] = field(default_factory=dict)
    passed_map: dict[str, bool] = field(default_factory=dict)
    detailed_results: dict[str, PipelineResult] = field(default_factory=dict)
    failed_doc_parts: list[str] = field(default_factory=list)
    pre_val_passed_map: dict[str, bool] = field(default_factory=dict)
    pre_val_detailed_results: dict[str, PipelineResult] = field(default_factory=dict)
    pre_check_inconsistent_list: list[str] = field(default_factory=list)
    blocked_parts: list[str] = field(default_factory=list)

    elapsed: dict[str, float] = field(default_factory=dict)
    all_elapsed: list[dict] = field(default_factory=list)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    project_root = Path(__file__).resolve().parent.parent
    resolved = project_root / path
    if resolved.exists():
        return resolved.resolve()
    return path.resolve()


def _log_and_print(console: Console, msg: str, style: str = "", level: str = "info"):
    """Write a message to both the logger and the rich console."""
    {
        "info": info,
        "warn": lambda m: info(f"[WARN] {m}"),
        "error": error,
        "debug": debug,
        "highlight": highlight,
    }[
        level
    ](msg)
    console.print(Text(msg, style=style) if style else msg)


def _run_step(
    console: Console,
    name: str,
    fn: Callable[[], Any],
    *,
    ctx: PipelineContext,
    key: str = None,
) -> Any:
    """Execute a pipeline step with unified timing, logging, and error handling."""
    subsection(name)
    start = time.perf_counter()
    info(f"{name} started")
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        ctx.elapsed[key or name] = elapsed
        highlight(f"{name} completed ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start
        ctx.elapsed[key or name] = elapsed
        error(f"{name} failed ({elapsed:.2f}s): {e}")
        console.print(Text(f"{name} 失败: {e}", style="bold red"))
        raise


def _interactive_select_doc_file(console: Console, doc_files: list[Path]) -> Optional[Path]:
    categories: dict[str, list[Path]] = {
        "default": [],
        "inplace": [],
        "compound": [],
        "other": [],
    }
    for f in doc_files:
        name = f.name
        if "&" in name:
            categories["compound"].append(f)
        elif "Inplace" in name:
            categories["inplace"].append(f)
        elif name.startswith("aclnn"):
            categories["default"].append(f)
        else:
            categories["other"].append(f)

    default_files = categories["default"]
    inplace_files = categories["inplace"]
    compound_files = categories["compound"]
    other_files = categories["other"]

    if len(default_files) == 1 and not any(v for k, v in categories.items() if k != "default"):
        console.print(f"    [dim]自动选择 (默认):[/dim] [cyan]{default_files[0].name}[/cyan]")
        return default_files[0]

    if not default_files:
        console.print("    [yellow]未找到默认格式文档。[/yellow]")
        return None

    has_unsupported = inplace_files or compound_files or other_files
    if has_unsupported:
        console.print(f"    [dim]找到 {len(doc_files)} 个 .md 文件:[/dim]")
        console.print(f"        [dim]默认:[/dim]       {len(default_files)} 个文件")
        if inplace_files:
            console.print(f"        [yellow]内联:[/yellow]     {len(inplace_files)} 个文件")
        if compound_files:
            console.print(f"        [yellow]复合:[/yellow]    {len(compound_files)} 个文件")
        if other_files:
            console.print(f"        [dim]其他:[/dim]        {len(other_files)} 个文件")
        console.print("    [dim]目前不支持处理内联/复合/其他格式。[/dim]")
        console.print()

    if len(default_files) == 1:
        selected = questionary.confirm(f"Process {default_files[0].name}?", default=True).ask()
        if selected:
            return default_files[0]

    choices = [questionary.Choice([("bold green", "(*) "), ("", f.name)], value=f) for f in default_files]
    console.print()
    choices.append(questionary.Choice([("bold red", "Cancel")], value=CANCEL_SENTINEL))

    try:
        selected = questionary.select("Select a document to process (Enter to confirm):", choices=choices).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if selected in (CANCEL_SENTINEL, None):
        return None
    return selected


def _navigate_to_folder(
    console: Console,
    workspace_dir: Path,
    prompt: str,
    default_path: str,
    title: str = "Navigate to the target operator folder:",
) -> Optional[tuple[Path, Path]]:
    start_input = questionary.text(prompt, default=default_path).ask()
    if not start_input:
        return None

    current_dir = Path(start_input.rstrip("/").rstrip("\\"))
    if not current_dir.is_absolute():
        current_dir = (workspace_dir / current_dir).resolve()
    if not current_dir.is_dir():
        console.print(f"[red]目录未找到:[/red] [cyan]{current_dir}[/cyan]")
        return None

    repo_root = current_dir

    confirm = "__CONFIRM__"
    go_back = "__GO_BACK__"
    cancel = "__CANCEL__"

    while True:
        console.print()
        console.print(f"  [dim]当前:[/dim] [cyan]{current_dir}[/cyan]")

        subfolders = sorted(d.name for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

        choices = []
        choices.append(
            questionary.Choice(
                [("bold green", "[Confirm] "), ("", "Current directory is the target")],
                value=confirm,
            )
        )
        if current_dir != current_dir.parent:
            choices.append(
                questionary.Choice(
                    [("bold yellow", "[Back] "), ("", "Go to parent directory")],
                    value=go_back,
                )
            )
        choices.append(questionary.Choice([("bold red", "[Cancel]")], value=cancel))
        for name in subfolders:
            choices.append(questionary.Choice(name, value=name))

        try:
            selected = questionary.select(title, choices=choices).ask()
        except (KeyboardInterrupt, EOFError):
            return None

        if selected is None or selected == cancel:
            return None

        if selected == confirm:
            result = current_dir.resolve()
            console.print(f"  [dim]已选择:[/dim] [cyan]{result.name}[/cyan]")
            console.print(f"  [dim]路径:[/dim] {result}")
            console.print()
            return repo_root.resolve(), result

        if selected == go_back:
            current_dir = current_dir.parent
            continue

        current_dir = (current_dir / selected).resolve()


def _interactive_snapshot(console: Console, ctx: PipelineContext) -> tuple[Optional[Path], Optional[Path], str]:
    result = _navigate_to_folder(
        console,
        ctx.workspace_dir,
        prompt="Enter operator repository path:",
        default_path="./data/test/ops-math-master",
        title="Navigate to the target operator folder:",
    )
    if result is None:
        return None, None, ""

    repo_root, op_dir = result
    selected_op = op_dir.name
    op_path = str(op_dir.relative_to(repo_root))

    if questionary.confirm("Run snapshot?", default=True).ask():
        try:
            from src.main_snapshot import main as snapshot_main

            ctx.output_data_path.mkdir(parents=True, exist_ok=True)
            with redirect_stdout_to_log():
                snapshot_main(op_path, str(ctx.output_data_path), str(repo_root))
            console.print()
            console.print(f"[bold green]OK[/bold green] '{selected_op}' 快照构建完成")
        except Exception as e:
            err_msg = str(e).strip()
            msg = (
                f"Snapshot failed for '{selected_op}': {err_msg}" if err_msg else f"Snapshot failed for '{selected_op}'"
            )
            console.print(Text(msg, style="bold red"))
            raise

    return repo_root, op_dir, selected_op


def _select_md_stems(console: Console, ctx: PipelineContext, available_md_stems: list[str]) -> list[str]:
    if ctx.interactive:
        if len(available_md_stems) == 1:
            md_stem = available_md_stems[0]
            console.print(f"    [dim]自动选择 md_stem (仅一个):[/dim] [cyan]{md_stem}[/cyan]")
            return [md_stem]

        console.print(f"    [dim]找到 {len(available_md_stems)} 个 md_stem:[/dim]")
        for stem in available_md_stems:
            console.print(f"        [cyan]{stem}[/cyan]")
        console.print()

        choices = [questionary.Choice(stem, checked=True) for stem in available_md_stems]
        choices.append(questionary.Choice([("bold red", "Cancel")], value=CANCEL_SENTINEL))

        try:
            selected = questionary.checkbox(
                "Select md_stem(s) to process (Space to toggle, Enter to confirm):",
                choices=choices,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return []
        if selected is None or CANCEL_SENTINEL in selected:
            return []
        return selected

    if not ctx.md_stems:
        console.print(f"  [dim]md_stems:[/dim] [cyan]all ({len(available_md_stems)})[/cyan]")
        return available_md_stems

    valid = [s for s in ctx.md_stems if s in available_md_stems]
    invalid = [s for s in ctx.md_stems if s not in available_md_stems]
    if invalid:
        console.print(f"  [yellow]md_stems 未找到:[/yellow] {', '.join(invalid)}")
        info(f"md_stems not found in {available_md_stems}: {invalid}")
    if not valid:
        console.print("  [yellow]无有效 md_stems, 回退到全部[/yellow]")
        info(f"No valid md_stems, fallback to all: {available_md_stems}")
        return available_md_stems

    console.print("  [dim]md_stems:[/dim] " + ", ".join(f"[cyan]{s}[/cyan]" for s in valid))
    return valid


def _step_snapshot(console: Console, ctx: PipelineContext) -> bool:
    """Step 1: Build operator code snapshot. Returns False to abort."""
    console.rule("[bold magenta]快照构建[/bold magenta]")

    if ctx.interactive:
        ctx.original_op_repo, ctx.original_op_dir, ctx.selected_op = _interactive_snapshot(console, ctx)
        if ctx.original_op_dir is None:
            return False
    else:
        ctx.selected_op = ctx.original_op_dir.name
        if not ctx.original_op_dir.is_dir():
            _log_and_print(console, f"Directory not found: {ctx.original_op_dir}", style="bold red")
            return False
        info(f"Operator: {ctx.selected_op}")
        console.print(f"  [dim]算子:[/dim] [cyan]{ctx.selected_op}[/cyan]")
        console.print(f"  [dim]算子路径:[/dim] {ctx.original_op_dir}")
        console.print()

        def _run():
            from src.main_snapshot import main as snapshot_main

            op_path = str(ctx.original_op_dir.relative_to(ctx.original_op_repo))
            with redirect_stdout_to_log():
                snapshot_main(op_path, str(ctx.output_data_path), str(ctx.original_op_repo))

        _run_step(console, "Snapshot", _run, ctx=ctx, key="snapshot")

    md_stems_dir = ctx.output_data_path / ctx.selected_op

    if not md_stems_dir.exists():
        _log_and_print(console, "No operator snapshot data available.", style="yellow")
        return False

    return True


def _step_config(console: Console, ctx: PipelineContext):
    """Step: Select document and derive paths for current stem."""
    subsection("配置：文档与路径")

    md_stems_dir = ctx.base_output_data_path / ctx.selected_op
    ctx.version_dir = md_stems_dir / ctx.selected_md_stem
    ctx.snapshot_op_dir = ctx.original_op_dir
    ctx.evidence_dir = ctx.version_dir / "anchor"
    source_docs = ctx.original_op_dir / "docs"
    ctx.summary_dir = ctx.version_dir / "style"
    ctx.re_detection_output_path = ctx.version_dir / "re_detection"

    ctx.selected_file = None
    ctx.doc_content = ""
    if source_docs.exists():
        doc_files = sorted(source_docs.glob("*.md"))
        if doc_files:
            if ctx.interactive:
                ctx.selected_file = _interactive_select_doc_file(console, doc_files)
            else:
                ctx.selected_file = select_doc_file(doc_files, selected_md_stem=ctx.selected_md_stem)
                if ctx.selected_file:
                    console.print(f"    [dim]自动选择:[/dim] [cyan]{ctx.selected_file.name}[/cyan]")
            if ctx.selected_file is None:
                console.print("[yellow]未选择文档。流程结束。[/yellow]")
                return
            ctx.doc_content = Path(ctx.selected_file).read_text(encoding="utf-8")
        else:
            console.print("[yellow]在 docs/ 中未找到 .md 文件[/yellow]")
    else:
        console.print(f"[yellow]文档目录未找到:[/yellow] {source_docs}")

    info(f"Selected file: {ctx.selected_file.name if ctx.selected_file else None}")
    info(f"Validate only: {ctx.validate_only}")
    console.print()


def _step_summary(console: Console, ctx: PipelineContext):
    """Step 2: Generate original doc summary."""
    console.rule("\n[bold yellow]生成原始文档摘要[/bold yellow]")

    if ctx.selected_file is None:
        return

    filename = f"{ctx.selected_md_stem}-summary.json"
    ctx.summary_file_dir = ctx.summary_dir / filename

    def _run():
        from src.generation_validation.style.summarizer import gen_doc_summarize

        info(f"Generating summary for {ctx.selected_file.name}...")
        console.print(f"[dim]正在为 {ctx.selected_file.name} 生成摘要...[/dim]")
        gen_doc_summarize(ctx.doc_content, str(ctx.summary_file_dir), force=False)

    _run_step(console, "Summary generation", _run, ctx=ctx, key="summary")
    _log_and_print(console, f"{ctx.selected_file.name} -> {filename}", style="green")


def _step_change_detection(console: Console, ctx: PipelineContext):
    """Step 3: Detect code changes between original and modified operator."""
    console.rule("[bold yellow]变更检测[/bold yellow]")

    ctx.modified_op_dir_md_stem = ctx.modified_op_dir

    if not (ctx.evidence_dir.is_dir() and ctx.modified_op_dir):
        return

    info(f"Evidence dir: {ctx.evidence_dir}")
    info(f"Original op dir: {ctx.original_op_dir}")
    info(f"Modified op dir: {ctx.modified_op_dir}")
    console.print(f"  [dim]证据目录:[/dim]    {ctx.evidence_dir}")
    console.print(f"  [dim]原始算子目录:[/dim] {ctx.original_op_dir}")
    console.print(f"  [dim]修改后算子目录:[/dim]  {ctx.modified_op_dir}")

    run_cd = True
    if ctx.interactive:
        console.print()
        run_cd = questionary.confirm("Run change detection?", default=True).ask()
    if not run_cd:
        return

    def _run():
        from src.change_detection.doc_part_change_detection.change_evaluator import (
            run_evaluation,
        )

        console.print()
        doc_parts, all_reasons, modified_files, repo_other_changed_files = run_evaluation(
            evidence_dir=str(ctx.evidence_dir),
            before_dir=str(ctx.original_op_dir),
            after_dir=str(ctx.modified_op_dir_md_stem),
            doc_content=ctx.doc_content,
            fine_grained=True,
            console=console,
            repo_before_dir=str(ctx.original_op_repo) if ctx.original_op_repo else None,
            repo_after_dir=str(ctx.modified_op_repo) if ctx.modified_op_repo else None,
        )
        ctx.evaluation_reasons = all_reasons
        if all_reasons:
            reasons_path = ctx.version_dir / "results" / "evaluation_reasons.json"
            reasons_path.parent.mkdir(parents=True, exist_ok=True)
            reasons_path.write_text(json.dumps(all_reasons, ensure_ascii=False, indent=2), encoding="utf-8")
            info(f"Evaluation reasons written to: {reasons_path}")
        return doc_parts, modified_files, repo_other_changed_files

    ctx.doc_parts, ctx.modified_files, ctx.repo_other_files = _run_step(
        console, "变更检测", _run, ctx=ctx, key="change_detection"
    )

    if ctx.pre_check_inconsistent_list:
        merged = list(dict.fromkeys(ctx.doc_parts + ctx.pre_check_inconsistent_list))
        info(
            f"Union doc_parts with pre_check_inconsistent_list: {ctx.doc_parts} "
            f"+ {ctx.pre_check_inconsistent_list} -> {merged}"
        )
        console.print(f"  [dim]+ 预验证不一致:[/dim] {format_doc_parts(ctx.pre_check_inconsistent_list)}")
        console.print(f"  [bold green]最终处理:[/bold green] {format_doc_parts(merged)}")
        ctx.doc_parts = merged

    highlight(f"{len(ctx.doc_parts)} subparts to process")
    if ctx.doc_parts:
        info(f"Subparts to be processed: {ctx.doc_parts}")


def _step_redetect(console: Console, ctx: PipelineContext):
    """Step 4: Re-detect evidence for changed document parts."""
    console.rule("[bold yellow]证据重检测[/bold yellow]")

    if not ctx.doc_parts:
        info("No part of the document needs to be changed, skipping evidence re-detection")
        console.print("[dim]文档无需变更![/dim]")
        return

    run_re = True
    if ctx.interactive:
        run_re = questionary.confirm("开始证据重检测？", default=True).ask()
    else:
        console.print(f"  [dim]Todo list:[/dim] {format_doc_parts(ctx.doc_parts)}")
    if not run_re:
        return

    redirection_doc_parts = [p for p in ctx.doc_parts if p != "example"]
    info(f"Re-detection output path: {ctx.re_detection_output_path}")
    info(f"Redirection doc_parts: {redirection_doc_parts}")

    def _run():
        from src.change_detection.merge_evidence import merge_redirect_to_anchor
        from src.change_detection.run_change_detection_cli import change_deteciton_cli

        with redirect_stdout_to_log():
            change_deteciton_cli(
                str(ctx.original_op_dir),
                str(ctx.modified_op_dir),
                str(ctx.output_data_path),
                ctx.selected_md_stem,
                str(ctx.re_detection_output_path),
                redirection_doc_parts,
                ctx.selected_op,
                ctx.modified_files,
                add_file_list=ctx.repo_other_files,
                workspace_dir=str(ctx.modified_op_repo) if ctx.modified_op_repo else None,
                changed_operator_path_split=str(ctx.modified_op_dir_md_stem) if ctx.modified_op_dir_md_stem else None,
                verbose=False,
            )
        console.print("  > Redirecting and merging evidence")
        merge_redirect_to_anchor(ctx.re_detection_output_path, redirection_doc_parts, verbose=False)

    _run_step(console, "证据重检测", _run, ctx=ctx, key="redetect")


def _anchor_matches_changed_files(anchor: dict, changed_basenames: set[str]) -> bool:
    for ev in anchor.get("evidence", []):
        ev_info = ev.get("evidence_information", {})
        file_path = ev_info.get("file_path", "")
        if file_path and Path(file_path).name in changed_basenames:
            return True
    anchor_info = anchor.get("anchor_information", {})
    level1 = anchor_info.get("level1", {})
    file_path = level1.get("file_path", "")
    if file_path and Path(file_path).name in changed_basenames:
        return True
    return False


def _step_pre_validate(console: Console, ctx: PipelineContext) -> bool:
    """Pre-validation step for full mode: validate existing doc before generation.

    Reuses _step_gen_val under validate_only mode. Returns True if all doc_parts passed.
    """
    console.rule("[bold yellow]预验证[/bold yellow]")
    info("Running pre-validation before full pipeline")

    extractor = DocpartFileExtractor(ctx.evidence_dir)
    mappings = extractor.extract_fine_grained_mappings()
    ctx.doc_parts = [subpart for subpart, files in mappings.items() if files]
    examples_dir = ctx.original_op_dir / "examples"
    if examples_dir.exists() and "example" not in ctx.doc_parts:
        ctx.doc_parts.append("example")

    if not ctx.doc_parts:
        info("No doc_parts found for pre-validation, skipping")
        console.print("[dim]未找到用于预验证的 doc_parts, 跳过。[/dim]")
        return True

    info(f"Pre-validation doc_parts: {ctx.doc_parts}")
    console.print(f"[dim]Pre-validation doc_parts ({len(ctx.doc_parts)}):[/dim] {format_doc_parts(ctx.doc_parts)}")

    original_interactive = ctx.interactive
    ctx.interactive = False
    pre_val_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    _step_gen_val(console, ctx, validate_only=True, dashboard=pre_val_dashboard)
    ctx.interactive = original_interactive

    ctx.pre_val_passed_map = dict(ctx.passed_map)
    ctx.pre_val_detailed_results = dict(ctx.detailed_results)
    ctx.passed_map.clear()
    ctx.detailed_results.clear()

    pre_val_results_path = ctx.version_dir / "results" / "pre_val_detailed_results.json"
    try:
        pre_val_results_path.parent.mkdir(parents=True, exist_ok=True)
        pre_val_results_path.write_text(
            json.dumps(
                {k: _serialize_result(v) for k, v in ctx.pre_val_detailed_results.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log_and_print(console, f"预验证结果已写入: {pre_val_results_path}", style="dim")
    except Exception as e:
        error(f"Failed to write pre-validation results: {e}")
        console.print(f"[yellow]写入预验证结果失败: {e}[/yellow]")

    info(f"pre_val_passed_map: {ctx.pre_val_passed_map}")

    if not ctx.dashboard:
        lines = "\n".join(
            f"  [dim]{format_doc_part(k)}[/dim]: {'[green]True[/green]' if v else '[red]False[/red]'}"
            for k, v in ctx.pre_val_passed_map.items()
        )
        console.print(f"\n预验证结果:\n{lines}")

        all_passed = all(ctx.pre_val_passed_map.values())
        ctx.pre_check_inconsistent_list = [k for k, v in ctx.pre_val_passed_map.items() if not v]
        info(f"pre_check_inconsistent_list: {ctx.pre_check_inconsistent_list}")

        if all_passed:
            info("Pre-validation all passed!")
            console.print("[green]预验证全部通过![/green]")
        else:
            info("Pre-validation has failures!")
            console.print("\n[yellow]预验证存在失败项![/yellow]")
            console.print(f"\n[dim]不一致的 doc_parts:[/dim] {format_doc_parts(ctx.pre_check_inconsistent_list)}\n")
    else:
        all_passed = all(ctx.pre_val_passed_map.values())
        ctx.pre_check_inconsistent_list = [k for k, v in ctx.pre_val_passed_map.items() if not v]
        info(f"pre_check_inconsistent_list: {ctx.pre_check_inconsistent_list}")

        if not all_passed:
            console.print("\n[yellow]预验证存在失败项![/yellow]")
            console.print(f"\n[dim]不一致的 doc_parts:[/dim] {format_doc_parts(ctx.pre_check_inconsistent_list)}\n")

    return all_passed


def _step_filter_evidence(console: Console, ctx: PipelineContext):
    """Filter evidence for passed doc_parts to only include changed files."""
    console.rule("[bold yellow]证据过滤[/bold yellow]")

    # Shared dependencies can change even when no operator-local files change.
    changed_basenames = {Path(f).name for f in [*ctx.modified_files, *ctx.repo_other_files]}
    info(f"Changed file basenames: {changed_basenames}")
    console.print(f"[dim]变更文件名:[/dim] {changed_basenames}")

    evidences_dir = ctx.re_detection_output_path / "evidences"
    filtered_dir = ctx.re_detection_output_path / "evidences_changed_only"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    skipped_parts = []

    for doc_part in list(ctx.doc_parts):
        if doc_part == "example":
            continue
        should_filter = ENABLE_EVIDENCE_FILE_FILTER or ctx.pre_val_passed_map.get(doc_part, False)
        if not should_filter:
            info(f"{doc_part}: evidence file filter disabled and pre_val not passed, keep full evidence")
            console.print(f"[dim]{format_doc_part(doc_part)}: 证据文件过滤已关闭且预验证未通过, 保留完整证据[/dim]")
            continue

        evidence_file = resolve_evidence_file(evidences_dir, doc_part)
        if evidence_file is None:
            continue

        data = json.loads(evidence_file.path.read_text(encoding="utf-8"))
        anchor_tuples = data.get("anchor_tuple", [])

        filtered = [a for a in anchor_tuples if _anchor_matches_changed_files(a, changed_basenames)]

        if not filtered:
            warning(f"{doc_part}: no changed-file evidence found, skipping generation")
            console.print(f"[yellow]{format_doc_part(doc_part)}: 无变更文件相关证据, 跳过。[/yellow]")
            skipped_parts.append(doc_part)
            continue

        data["anchor_tuple"] = filtered
        out_path = filtered_dir / f"{doc_part}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        info(f"{doc_part}: filtered {len(anchor_tuples)} -> {len(filtered)} anchors")
        console.print(
            f"[dim]{format_doc_part(doc_part)}: 过滤 {len(anchor_tuples)} -> {len(filtered)} 个 anchors[/dim]"
        )

    for dp in skipped_parts:
        ctx.doc_parts.remove(dp)


def _step_gen_val(
    console: Console,
    ctx: PipelineContext,
    validate_only: bool = False,
    op_dir_override: Optional[Path] = None,
    evidence_dir_override: Optional[Path] = None,
    dashboard: Optional[ParallelDashboard] = None,
):
    """Generate and validate document parts.

    Args:
        validate_only: If True, only validate without generating.
        op_dir_override: Override the operator directory for generation (used by fix step).
        evidence_dir_override: Override the evidence directory (used by fix step in
            validate-only mode, where re_detection/evidences is never generated).
    """

    evidence_info: dict[str, str] = {}
    label = "一致性验证" if validate_only else "生成并验证"

    def _process_one(doc_part: str):
        if evidence_dir_override is not None:
            evidence_dir_path = evidence_dir_override
        elif not validate_only:
            changed_only_dir = ctx.re_detection_output_path / "evidences_changed_only"
            if (changed_only_dir / f"{doc_part}.json").exists():
                evidence_dir_path = changed_only_dir
            else:
                evidence_dir_path = ctx.re_detection_output_path / "evidences"
        else:
            evidence_dir_path = ctx.evidence_dir
        evidence_file = resolve_evidence_file(evidence_dir_path, doc_part)

        if evidence_file is None and doc_part != "example":
            reason = f"No evidence file found for doc_part '{doc_part}' in {evidence_dir_path}"
            info(reason)
            # 记为失败而非静默跳过: 进入 passed_map/detailed_results 视野,
            # fix 步骤可重试, 导出报告如实呈现, 不再"无痕缺失"
            ctx.passed_map[doc_part] = False
            ctx.detailed_results[doc_part] = PipelineResult(success=False, error=reason)
            return

        evidence_path_str = evidence_file.relative_path if evidence_file else "[## 调用示例]"
        evidence_info[doc_part] = evidence_path_str
        info(f"Processing doc_part: {doc_part}, evidence: {evidence_path_str}")

        log_capture = LogCapture()

        if validate_only:
            info(f"Code - Doc Validating ({doc_part})")
            result = run_validate_only(
                evidence_file,
                doc_part,
                log_capture,
                original_op_dir=str(ctx.original_op_dir),
                doc_json_dir=ctx.version_dir.parent / "doc_json" / f"{ctx.selected_md_stem}.json",
            )
        else:  # Post-validation fix
            op_dir = op_dir_override or ctx.modified_op_dir
            info(f"Doc Generating ({doc_part})")
            change_reason = ctx.evaluation_reasons.get(doc_part)
            result = run_full_pipeline(
                evidence_file,
                doc_part,
                log_capture,
                style_dir=ctx.summary_file_dir,
                modified_op_dir=str(op_dir),
                operator_name=ctx.selected_op,
                modified_files=list(dict.fromkeys([*ctx.modified_files, *ctx.repo_other_files])),
                change_reason=change_reason,
            )
            ctx.backfill_map[doc_part] = "\n\n".join(item["generated_text"] for item in result.generated)

        if not result.success:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} passed: False (error: {result.error})")
        elif result.evidence_conflict:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} passed: False (evidence conflict)")
        elif result.no_change_needed:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} passed: False (no change needed)")
        else:
            # result.validation 的元素是 dict：getattr(dict, "passed", True) 恒为 True，
            # 会导致 FAIL 也被记为通过；按 dict/对象两种形态取真实 passed 值
            ctx.passed_map[doc_part] = all(
                (item.get("passed") if isinstance(item, dict) else getattr(item, "passed", True))
                for item in result.validation
            )
            info(f"doc_part {doc_part} passed: {ctx.passed_map[doc_part]}")
        old = ctx.detailed_results.get(doc_part)
        if old:
            result.attempts_used += old.attempts_used
        ctx.detailed_results[doc_part] = result

    def _print_one(doc_part: str):
        if doc_part not in ctx.detailed_results:
            return
        console.rule(f"[bold cyan]{label} ({doc_part})[/bold cyan]")
        console.print(f"  [dim]文件:[/dim]       {evidence_info.get(doc_part, '?')}")
        console.print()
        _display_results(ctx.detailed_results[doc_part], "validate_only" if validate_only else "full")

    remaining = list(ctx.doc_parts)

    if ctx.interactive:
        while remaining:
            console.print()
            remaining_str = ", ".join(f"[cyan]{format_doc_part(p)}[/cyan]" for p in remaining)
            info(f"Remaining doc_parts ({len(remaining)}): {remaining}")
            console.print(f"[dim]剩余 doc_parts ({len(remaining)}):[/dim] {remaining_str}")

            doc_part = select_doc_part(remaining)
            if not doc_part:
                error("No doc_part selected")
                return

            _process_one(doc_part)
            _print_one(doc_part)

            if questionary.confirm("Finish gen&val?", default=False).ask():
                info("User requested to finish gen&val early")
                break
            remaining.remove(doc_part)
    else:
        info(f"Processing {len(remaining)} doc_parts in parallel (max_workers={GEN_VAL_MAX_WORKERS})")

        failures: list[str] = []

        def _record_future_failure(dp: str, e: Exception):
            """future 异常不再静默吞掉: 记录失败 + 聚合, 非交互模式最终抛出。"""
            error(f"doc_part {dp} failed: {e}")
            ctx.passed_map[dp] = False
            ctx.detailed_results[dp] = PipelineResult(success=False, error=f"doc_part 处理异常: {e}")
            failures.append(f"{format_doc_part(dp)}: {e}")

        if dashboard:
            console.rule("[bold yellow]生成并验证[/bold yellow]")

            def _run_with_context(dp):
                tokens = set_dashboard_context(dashboard, dp)
                try:
                    _process_one(dp)
                finally:
                    for t in reversed(tokens):
                        t.var.reset(t)

            with dashboard.live_context():
                with ThreadPoolExecutor(max_workers=GEN_VAL_MAX_WORKERS) as executor:
                    futures = {executor.submit(_run_with_context, dp): dp for dp in remaining}
                    for future in as_completed(futures):
                        dp = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            _record_future_failure(dp, e)
                            dashboard.update(dp, status="✗", result="ERROR")
                            dashboard.append_log(dp, f"[red]Error: {e}[/red]")
        else:
            console.print(f"[dim]并行处理 {len(remaining)} 个 doc_parts (max_workers={GEN_VAL_MAX_WORKERS})[/dim]")
            with ThreadPoolExecutor(max_workers=GEN_VAL_MAX_WORKERS) as executor:
                futures = {executor.submit(_process_one, dp): dp for dp in remaining}
                for future in as_completed(futures):
                    dp = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        _record_future_failure(dp, e)
                        console.print(
                            Text(
                                f"doc_part {format_doc_part(dp)} 失败: {e}",
                                style="bold red",
                            )
                        )

            for dp in remaining:
                _print_one(dp)

        if failures and not ctx.interactive:
            # 非交互模式: 基础设施级异常聚合抛出 → failed_stems 机制 → exit 1,
            # 避免主进程 exit 0 被批量执行器 (run_pr_batch.py/CI) 误计为 SUCCESS
            raise RuntimeError(f"{len(failures)} 个 doc_part 处理异常: " + "; ".join(failures))


def _step_fix_val(console: Console, ctx: PipelineContext):
    """Fix failed doc_parts by re-generating with full pipeline."""
    if ctx.skip_fix:
        info("Skipping fix step (--skip-fix)")
        console.print("[dim]跳过修复步骤 (--skip-fix)[/dim]")
        return

    failed_doc_parts = [dp for dp, passed in ctx.passed_map.items() if not passed]
    if not failed_doc_parts:
        info("All doc_parts passed validation, no fix needed")
        console.print("[dim]所有 doc_parts 通过验证, 无需修复。[/dim]")
        return

    console.rule("[bold red]验证后修复[/bold red]")
    info(f"Failed doc_parts: {failed_doc_parts}")
    console.print(f"[dim]需修复的失败 doc_parts:[/dim] {format_doc_parts(failed_doc_parts)}")

    ctx.failed_doc_parts = failed_doc_parts
    ctx.doc_parts = failed_doc_parts

    t = time.perf_counter()
    _step_summary(console, ctx)
    ctx.elapsed["fix_summary"] = time.perf_counter() - t

    t = time.perf_counter()
    fix_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    # validate-only 模式从不执行变更检测, re_detection/evidences 不存在;
    # fix 重新生成必须回退到 snapshot anchor 证据目录, 否则全部 doc_part 静默跳过
    _step_gen_val(
        console,
        ctx,
        op_dir_override=ctx.original_op_dir,
        evidence_dir_override=ctx.evidence_dir if ctx.validate_only else None,
        dashboard=fix_dashboard,
    )
    ctx.elapsed["fix_gen_val"] = time.perf_counter() - t


def _analyze_gen_val_result(doc_part: str, result: PipelineResult, ctx: PipelineContext) -> Optional[dict]:
    def _vget(v, key, default=None):
        if isinstance(v, dict):
            return v.get(key, default)
        return getattr(v, key, default)

    if result.no_change_needed:
        return {
            "success": False,
            "reason": result.conflict_details or "LLM 判断证据不要求修改，无实际变更。",
        }
    if result.evidence_conflict:
        return {
            "success": False,
            "reason": f"证据间存在不可调和的冲突: {result.conflict_details or '详见验证日志'}",
        }

    validations = result.validation
    if validations and all(
        (_vget(v, "edit_distance_ratio") or 1.0) < POST_ANALYZE_NO_CHANGE_THRESHOLD for v in validations
    ):
        return {
            "success": False,
            "reason": "生成未产生实际变更，证据可能未要求修改文档内容。",
        }

    if result.max_retries_hit:
        failed = [v for v in validations if not _vget(v, "passed", True)]
        if failed:
            return {
                "success": False,
                "reason": f"验证反复失败，达到最大重试次数后仍有 {len(failed)} 个模块未通过。",
            }

    return {"success": True, "reason": "生成验证正常通过"}


def _step_post_analyze(console: Console, ctx: PipelineContext):
    """Post-generation analysis: detect no-change/conflict cases, update evaluation_reasons.json."""
    console.rule("[bold yellow]生成验证后分析[/bold yellow]")

    reasons_path = ctx.version_dir / "results" / "evaluation_reasons.json"
    if not reasons_path.exists():
        info("No evaluation_reasons.json found, skipping post-analyze")
        console.print("[dim]未找到 evaluation_reasons.json，跳过后置分析[/dim]")
        return

    all_reasons = json.loads(reasons_path.read_text(encoding="utf-8"))
    blocked_parts = []

    for doc_part in list(ctx.passed_map.keys()):
        if doc_part not in all_reasons:
            continue
        entry = all_reasons[doc_part]
        if not entry.get("keep", False):
            continue

        result = ctx.detailed_results.get(doc_part)
        if not result or not result.success:
            continue

        analysis = _analyze_gen_val_result(doc_part, result, ctx)
        if analysis:
            entry["gen_val_result"] = analysis
            if not analysis["success"]:
                blocked_parts.append(doc_part)
                ctx.passed_map[doc_part] = False

    reasons_path.write_text(json.dumps(all_reasons, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"Updated evaluation_reasons.json with gen_val_result for {len(all_reasons)} doc_parts")
    console.print(f"[dim]已更新 evaluation_reasons.json（{len(all_reasons)} 个 doc_parts）[/dim]")

    for dp in blocked_parts:
        ctx.backfill_map.pop(dp, None)
        reason = all_reasons[dp]["gen_val_result"]["reason"]
        console.print(f"[yellow]{format_doc_part(dp)}: 阻止回填 — {reason}[/yellow]\n")
        info(f"Blocked backfill for {dp}: {reason}")

    ctx.blocked_parts = blocked_parts


def _step_fill_back(console: Console, ctx: PipelineContext):
    """Step 6: Fill generated content back to markdown.

    基底优先 head (modified repo) 同名 md —— 保留 PR 中人工修改但未被生成
    重写覆盖的章节；head 缺失该文件时回退 original (merge_base) 的 selected_file。
    """
    subsection("6. 回填到 Markdown 文档")

    if not (ctx.selected_file and ctx.backfill_map):
        info("No backfill needed (no selected_file or empty backfill_map)")
        return

    base_file = ctx.selected_file
    head_md = (ctx.modified_op_dir / "docs" / ctx.selected_file.name) if ctx.modified_op_dir else None
    if head_md is not None and head_md.is_file():
        base_file = head_md
        info(f"Backfill base (head): {base_file}")
        console.print(f"  [dim]回填基底(head):[/dim] [cyan]{base_file}[/cyan]")
    else:
        info(f"Head md not found ({head_md}), fallback to original base: {ctx.selected_file}")
        console.print(f"  [dim]回填基底(original回退):[/dim] [cyan]{ctx.selected_file}[/cyan]")

    results_dir = ctx.version_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(results_dir / f"[new]{ctx.selected_md_stem}_{timestamp}.md")
    info(f"Filling back to: {output_path}")

    def _run():
        from src.md_fill.main_fill import update_md_with_dict

        update_md_with_dict(ctx.backfill_map, str(base_file), output_path)

    _run_step(console, "Markdown backfill", _run, ctx=ctx, key="backfill")
    console.print(f"[dim]新 markdown 文件已生成至: {output_path}[/dim]")


def _step_export_results(console: Console, ctx: PipelineContext):
    """Step 7: Export detailed results to JSON."""
    detailed_results_path = ctx.version_dir / "results" / "detailed_results.json"
    try:
        detailed_results_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_results_path.write_text(
            json.dumps(
                {k: _serialize_result(v) for k, v in ctx.detailed_results.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log_and_print(
            console,
            f"Detailed results written to: {detailed_results_path}",
            style="dim",
        )
    except Exception as e:
        error(f"Failed to write detailed results: {e}")
        console.print(f"[yellow]写入详细结果失败: {e}[/yellow]")


def _serialize_result(r: PipelineResult) -> dict:
    d = asdict(r)
    d["validation"] = [v.model_dump() if hasattr(v, "model_dump") else v for v in r.validation]
    return d


def _run_pipeline_for_stem(console: Console, ctx: PipelineContext, stem: str):
    """Run pipeline steps 2-10 for a single md_stem."""
    ctx.selected_md_stem = stem
    section(f"Processing: {stem}")

    # Step 2: 设置当前 stem 的路径和文档
    _step_config(console, ctx)
    if not ctx.selected_file:
        return

    # Step 3: 预校验 — 用原始证据验证现有文档与代码一致性，产出 pre_val_passed_map
    # validate-only 模式下预校验是主体流程, 无条件执行;
    # ENABLE_PRE_VALIDATION 只控制完整模式的可选预校验
    t = time.perf_counter()
    if ctx.validate_only or ENABLE_PRE_VALIDATION:
        _step_pre_validate(console, ctx)
    else:
        console.rule("[bold yellow]预验证（已关闭）[/bold yellow]")
        console.print("[dim]完整模式下 ENABLE_PRE_VALIDATION=False, 跳过预验证[/dim]")
    ctx.elapsed["pre_validate"] = time.perf_counter() - t

    # 单校验模式分支
    if ctx.validate_only:
        # --validate-only: 预校验 → 修复+回填(除非 --skip-fix) → 导出 → 结束
        ctx.passed_map = dict(ctx.pre_val_passed_map)
        console.print(ctx.passed_map)
        failed_parts = ", ".join(k for k, v in ctx.passed_map.items() if not v)
        if ctx.interactive and failed_parts:
            run_fix = questionary.confirm(
                f"Pre-validation failed for: {failed_parts}. Run fix?",
                default=True,
            ).ask()
            if not run_fix:
                ctx.skip_fix = True
        _step_fix_val(console, ctx)
        _step_fill_back(console, ctx)
        _step_export_results(console, ctx)
        _print_stats(ctx, console)
        section("Pipeline Complete")
        console.rule("[bold cyan]仅验证流程结束。[/bold cyan]")
        return

    # Step 4: 变更检测 — 对比 before/after 代码，识别受影响的 doc_parts 和变更文件列表
    _step_change_detection(console, ctx)

    # 取并集：预校验失败 ∪ 代码变更影响的 doc_parts 为空
    if not ctx.doc_parts:
        info("No doc_parts need processing (pre-validation all passed and no code changes)")
        console.print("[green]无需处理的 doc_parts (预验证全部通过且无代码变更)[/green]")
        _step_export_results(console, ctx)
        _print_stats(ctx, console)
        section("Pipeline Complete")
        console.rule("[bold cyan]流水线结束（无需变更）。[/bold cyan]")
        return

    # Step 5: 生成文档风格摘要（summary），供后续生成阶段使用
    _step_summary(console, ctx)

    # Step 6: 证据重检 — 对变更的 doc_parts 执行多级 anchor redirect，合并新旧证据
    _step_redetect(console, ctx)

    # Step 7: 证据过滤 — 对 pre_val pass 的 doc_part 只保留变更文件相关证据，fail 的保留全量
    t = time.perf_counter()
    _step_filter_evidence(console, ctx)
    ctx.elapsed["filter_evidence"] = time.perf_counter() - t

    # Step 8: 生成与验证 — 根据 ctx.doc_parts 调用 LLM 生成文档并验证一致性
    t = time.perf_counter()
    gen_val_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    _step_gen_val(console, ctx, dashboard=gen_val_dashboard)
    ctx.elapsed["gen_val"] = time.perf_counter() - t

    # Step 8.5: 生成验证后分析 — 检测无变更/证据冲突，更新 evaluation_reasons.json
    t = time.perf_counter()
    _step_post_analyze(console, ctx)
    ctx.elapsed["post_analyze"] = time.perf_counter() - t

    # Step 9: 回填 — 将生成的文档内容写回原始 markdown 文件
    _step_fill_back(console, ctx)

    # Step 10: 导出结果 — 将各 doc_part 的生成/验证详情输出为 JSON
    _step_export_results(console, ctx)

    _print_stats(ctx, console)


def _run_pipeline(
    console: Console,
    interactive: bool = False,
    original_op_repo: Optional[Path] = None,
    original_op_dir: Optional[Path] = None,
    modified_op_repo: Optional[Path] = None,
    modified_op_dir: Optional[Path] = None,
    validate_only: bool = False,
    skip_fix: bool = False,
    md_stems: Optional[list[str]] = None,
    workspace_dir: Optional[Path] = None,
    output_data_path: Optional[Path] = None,
    dashboard: bool = False,
):
    setup_logger(level=logging.INFO, console=False)
    section("Pipeline Start")
    mode_label = "interactive" if interactive else "non-interactive"
    info(f"Mode: {mode_label}, validate_only={validate_only}, skip_fix={skip_fix}")
    if sys.stdout.isatty():
        console.clear()
    display_header()

    ws = (workspace_dir or Path(__file__).resolve().parent.parent).resolve()
    output_data = (output_data_path or (ws / "data" / "outputs" / "snapshot")).resolve()
    output_data.mkdir(parents=True, exist_ok=True)

    subsection("0. 工作区")
    info(f"Workspace: {ws}")
    info(f"Data path: {output_data}")
    console.print(f"  [dim]工作目录:[/dim]  {ws}")
    console.print(f"  [dim]数据路径:[/dim]  {output_data}")

    ctx = PipelineContext(
        workspace_dir=ws,
        output_data_path=output_data,
        base_output_data_path=output_data,
        original_op_repo=original_op_repo,
        original_op_dir=original_op_dir,
        modified_op_repo=modified_op_repo,
        modified_op_dir=modified_op_dir,
        md_stems=md_stems or [],
        validate_only=validate_only,
        skip_fix=skip_fix,
        interactive=interactive,
        dashboard=dashboard,
    )

    start_total = time.perf_counter()
    error_msg: Optional[str] = None

    try:
        # Step 0: LLM 连通性测试 — 失败视为流水线错误 (非零退出, timing.json 记录 error)
        info("测试 LLM 模型连通性...")
        from src.llm.client import LLMClient

        try:
            client = LLMClient()
            console.print(f"  [dim]{client.model} 模型连通性测试...[/dim]")
            client.test_connection()
            info(f"LLM 连通性测试通过 [OK] model={client.model}")
            console.print("  [dim][LLM OK][/dim]")
        except Exception as e:
            error(f"LLM 连通性测试失败，流水线终止: {e}")
            console.print(f"  [red][LLM ERROR] LLM 模型不可用，全流程终止: {e}[/red]")
            raise RuntimeError(f"LLM 连通性测试失败: {e}") from e

        # Step 1: 快照原始算子代码和证据到 version_dir，用于后续对比基准
        if not _step_snapshot(console, ctx):
            raise RuntimeError(f"Snapshot 阶段失败 (operator: {ctx.selected_op or original_op_dir})")
        ctx.elapsed["snapshot"] = time.perf_counter() - start_total
        info(f"Snapshot elapsed: {ctx.elapsed['snapshot']:.2f}s")

        # 交互模式：提前询问 mode 和 modified_op_repo（只问一次）
        if ctx.interactive:
            console.print()
            mode = select_mode()
            if not mode:
                error("No mode selected")
                return
            ctx.validate_only = mode == "validate_only"

            if not ctx.validate_only:
                console.print("现在，您可以编辑算子代码了。")
                console.print()
                if not questionary.confirm("FINISH?", default=True).ask():
                    return
                console.print()
                modified_result = _navigate_to_folder(
                    console,
                    ctx.workspace_dir,
                    prompt="Enter modified operator repository path:",
                    default_path="./data/test/ops-math-master_modified",
                    title="Navigate to the modified operator folder:",
                )
                if modified_result is None:
                    return
                ctx.modified_op_repo, ctx.modified_op_dir = modified_result

        # 选择要处理的 md_stems
        md_stems_dir = output_data / ctx.selected_op
        available_md_stems = sorted(d.name for d in md_stems_dir.iterdir() if d.is_dir() and (d / "anchor").exists())
        info(f"Available md_stems: {available_md_stems}")

        stems_to_process = _select_md_stems(console, ctx, available_md_stems)
        if not stems_to_process:
            if interactive:
                # 交互模式: 用户主动取消, 正常退出
                error("No md_stems selected")
                return
            raise RuntimeError(f"无可处理的 md_stems (available: {available_md_stems})")

        info(f"Stems to process: {stems_to_process}")
        console.print("  [dim]待处理的 stems:[/dim] " + ", ".join(f"[cyan]{s}[/cyan]" for s in stems_to_process))
        console.print()

        # 循环处理每个 stem
        failed_stems: list[str] = []
        for i, stem in enumerate(stems_to_process):
            console.rule(f"[bold magenta]处理中：{stem} ({i + 1}/{len(stems_to_process)})[/bold magenta]")
            _reset_per_stem_state(ctx)
            ctx.md_stems = stems_to_process
            try:
                _run_pipeline_for_stem(console, ctx, stem)
            except Exception as e:
                error(f"Pipeline failed for stem '{stem}': {e}")
                console.print(Text(f"stem '{stem}' 流水线失败: {e}", style="bold red"))
                failed_stems.append(stem)
                continue

        section("Pipeline Complete")
        console.rule("[bold cyan]所有 stems 处理完成[/bold cyan]")
        if failed_stems:
            msg = f"以下 stems 流水线失败: {', '.join(failed_stems)}"
            error(msg)
            console.print(Text(msg, style="bold red"))
            if not interactive:
                # 非交互模式: 任一 stem 失败 → 非零退出, 避免批量/CI 误判为成功
                raise RuntimeError(msg)
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        _export_timing(ctx, output_data, time.perf_counter() - start_total, error_msg)


def _print_stats(ctx: PipelineContext, console: Console):
    """Print pipeline statistics."""
    subsection("统计")
    for name, secs in ctx.elapsed.items():
        info(f"Elapsed time for {name}: {secs:.2f}s")
    info(f"backfill_map: {ctx.backfill_map}")
    info(f"passed_map: {ctx.passed_map}")
    console.rule("[bold cyan]统计[/bold cyan]")
    for name, secs in ctx.elapsed.items():
        console.print(f"[dim]{name}:\t{secs:.2f}s[/dim]")
    console.print(
        f"[dim]backfill_map[/dim] {{{', '.join(f'{k!r}: {v[:50]!r}...' for k, v in ctx.backfill_map.items())}}}"
    )
    console.print(f"[dim]passed_map[/dim] {ctx.passed_map}")


def _export_timing(
    ctx: PipelineContext,
    output_data: Path,
    total_elapsed: float,
    error_msg: Optional[str] = None,
):
    """Export pipeline timing data to timing.json for external consumption."""
    # Final commit if elapsed has uncommitted data (last stem not committed by _reset_per_stem_state)
    if ctx.elapsed:
        ctx.all_elapsed.append(
            {
                "stem": ctx.selected_md_stem,
                "steps": dict(ctx.elapsed),
            }
        )
        ctx.elapsed.clear()
    timing = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "total_elapsed": total_elapsed,
        "error": error_msg,
        "stems": ctx.all_elapsed,
    }
    timing_path = output_data / "timing.json"
    try:
        timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
        info(f"Timing exported to: {timing_path}")
    except Exception as e:
        error(f"Failed to export timing: {e}")


def _reset_per_stem_state(ctx: PipelineContext):
    """Reset per-stem mutable state in PipelineContext for a new iteration."""
    ctx.selected_file = None
    ctx.doc_content = ""
    ctx.doc_parts.clear()
    ctx.modified_files.clear()
    ctx.repo_other_files.clear()
    ctx.backfill_map.clear()
    ctx.passed_map.clear()
    ctx.detailed_results.clear()
    ctx.failed_doc_parts.clear()
    ctx.pre_val_passed_map.clear()
    ctx.pre_val_detailed_results.clear()
    ctx.pre_check_inconsistent_list.clear()
    if ctx.elapsed:
        ctx.all_elapsed.append(
            {
                "stem": ctx.selected_md_stem,
                "steps": dict(ctx.elapsed),
            }
        )
        ctx.elapsed.clear()


# 非 TTY 条件必填 flag：4 路径 flag + --output_data_path（Task 13 扩展；TTY 模式走 TUI/默认推导，见下方调用点）
_REQUIRED_PATH_FLAGS = (
    "original_op_repo",
    "original_op_path",
    "modified_op_repo",
    "modified_op_path",
)


def _check_headless_required(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    required_attrs: tuple[str, ...],
) -> list[str]:
    """非 TTY 缺参 → parser.error（exit 2，stderr 含"非交互"与 flag 名）；TTY → 返回缺失列表由调用方走 TUI。"""
    missing = [attr for attr in required_attrs if not getattr(args, attr, None)]
    if missing and not sys.stdin.isatty():
        flags = "/".join(f"--{attr}" for attr in missing)
        parser.error(f"非交互模式缺少必需参数 {flags}（或请提供全部参数以进入非交互模式）")
    return missing


def run_all(
    original_op_repo: str,
    original_op_path: str,
    modified_op_repo: str,
    modified_op_path: str,
    validate_only: bool = False,
    skip_fix: bool = False,
    md_stems: Optional[list[str]] = None,
    workspace_dir: Optional[str] = None,
    output_data_path: Optional[str] = None,
    dashboard: bool = False,
):
    console = Console()
    if sys.stdout.isatty():
        console.clear()
    ws = Path(workspace_dir).resolve() if workspace_dir else Path(__file__).resolve().parent.parent
    out_path = Path(output_data_path).resolve() if output_data_path else None
    og_repo = _resolve_path(original_op_repo)
    original_op_dir = (og_repo / original_op_path).resolve()
    mod_repo = _resolve_path(modified_op_repo)
    modified_op_dir = (mod_repo / modified_op_path).resolve()
    try:
        _run_pipeline(
            console,
            original_op_repo=og_repo,
            original_op_dir=original_op_dir,
            modified_op_repo=mod_repo,
            modified_op_dir=modified_op_dir,
            validate_only=validate_only,
            skip_fix=skip_fix,
            md_stems=md_stems,
            workspace_dir=ws,
            output_data_path=out_path,
            interactive=False,
            dashboard=dashboard,
        )
    finally:
        if sys.stdout.isatty():
            console.clear()


def run_cli():
    console = Console()
    if sys.stdout.isatty():
        console.clear()
    try:
        _run_pipeline(console, interactive=True)
    finally:
        if sys.stdout.isatty():
            console.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANN doc generation pipeline")
    parser.add_argument(
        "--original_op_repo",
        help="Original operator repository root path (e.g. ./data/test/ops-math-master)",
    )
    parser.add_argument(
        "--original_op_path",
        help="Relative operator path within the repo (e.g. math/add)",
    )
    parser.add_argument("--modified_op_repo", help="Modified operator repository root path")
    parser.add_argument(
        "--modified_op_path",
        help="Relative modified operator path within the repo (e.g. math/add_modified)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        dest="validate_only",
        help="Only run pre-validation, skip change detection and generation",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        dest="skip_fix",
        help="Skip the fix step for failed validations (only affects --validate-only)",
    )
    parser.add_argument(
        "--md-stem",
        nargs="+",
        default=[],
        dest="md_stems",
        help="Markdown stem(s) to process (e.g. aclnnAbs aclnnAdd). Auto-selected if omitted.",
    )
    parser.add_argument(
        "--workspace_dir",
        default=None,
        help="Workspace root directory (default: project root)",
    )
    parser.add_argument(
        "--output_data_path",
        default=None,
        help="Output data directory (default: <workspace>/data/outputs/snapshot)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Enable live dashboard for parallel doc_part processing",
    )
    args = parser.parse_args()

    missing_path_flags = _check_headless_required(parser, args, _REQUIRED_PATH_FLAGS)
    # --output_data_path 仅在 4 路径 flag 齐全且非 TTY 时必填（TTY 默认推导 <workspace>/data/outputs/snapshot 保留）；
    # 该检查必须先于 apply_settings——缺参报错不依赖 env 状态
    if not missing_path_flags:
        _check_headless_required(parser, args, ("output_data_path",))

    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    try:
        if not missing_path_flags:
            run_all(
                original_op_repo=args.original_op_repo,
                original_op_path=args.original_op_path,
                modified_op_repo=args.modified_op_repo,
                modified_op_path=args.modified_op_path,
                validate_only=args.validate_only,
                skip_fix=args.skip_fix,
                md_stems=args.md_stems,
                workspace_dir=args.workspace_dir,
                output_data_path=args.output_data_path,
                dashboard=args.dashboard,
            )
        else:
            run_cli()
    except SystemExit:
        raise
    except Exception as e:
        # 流水线失败 → 非零退出码, 供 run_pr_batch.py / CI 感知 (详情见 timing.json 的 error 字段)
        print(f"\nPipeline FAILED: {e}", file=sys.stderr)
        sys.exit(1)
