# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Literal, cast

from baseline.check import (
    baseline_gate_issues,
    load_baseline_state,
)
from round.contract import ROUND_STATE_REQUIRED_FIELDS
from round.kernel_continuity import analyze_triton_kernel_continuity
from round.local_optimum import collect_local_optimum_warnings
from shared.json_io import load_json_object, optional_str
from shared.models import OptimizeCheckResult, RoundArtifactsInspection, RoundState
from shared.paths import baseline_dir, declared_state_file, existing_file, missing_issue
from shared.results import append_pass_issues_to_summary, build_check_result
from shared.round_naming import expected_round_operator_name, expected_round_perf_name, resolve_workspace_operator_file


_OPTIMIZE_DELETE_PT_FILES_ENV = "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES"
OptimizePtCleanupMode = Literal["never", "round", "run-test"]
_PT_CLEANUP_MODES = frozenset({"never", "round", "run-test"})
_LEGACY_ROUND_CLEANUP_VALUES = frozenset({"1", "true", "yes", "on"})
_LEGACY_NEVER_CLEANUP_VALUES = frozenset({"0", "false", "no", "off"})
_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "perf-analysis.md",
    "round-state.json",
}


def ordinary_optimize_pt_cleanup_mode() -> OptimizePtCleanupMode:
    raw_value = os.environ.get(_OPTIMIZE_DELETE_PT_FILES_ENV)
    if raw_value is None:
        return "round"
    value = raw_value.strip().lower()
    if value in _PT_CLEANUP_MODES:
        return cast(OptimizePtCleanupMode, value)
    if value in _LEGACY_ROUND_CLEANUP_VALUES:
        return "round"
    if value in _LEGACY_NEVER_CLEANUP_VALUES:
        return "never"
    return "round"


def ordinary_optimize_pt_cleanup_enabled() -> bool:
    return ordinary_optimize_pt_cleanup_mode() == "round"


def is_ordinary_pt_result_file(path: Path) -> bool:
    name_lower = path.name.lower()
    return name_lower == "test_result.pt" or name_lower.endswith("_result.pt")


def cleanup_pt_file(pt_file: Path) -> str | None:
    if not pt_file.is_file() or not is_ordinary_pt_result_file(pt_file):
        return None
    try:
        pt_file.unlink()
        return pt_file.name
    except OSError:
        return None


def cleanup_dir_pt_files(directory: Path) -> list[str]:
    cleaned: list[str] = []
    try:
        candidates = sorted(directory.iterdir())
    except OSError:
        return cleaned
    for pt_file in candidates:
        cleaned_name = cleanup_pt_file(pt_file)
        if cleaned_name is not None:
            cleaned.append(cleaned_name)
    return cleaned


def cleanup_dir_prof_artifacts(directory: Path) -> list[str]:
    cleaned: list[str] = []
    try:
        candidates = sorted(directory.iterdir())
    except OSError:
        return cleaned
    for artifact in candidates:
        if not artifact.name.startswith("PROF_"):
            continue
        try:
            if artifact.is_dir() and not artifact.is_symlink():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
            cleaned.append(artifact.name)
        except OSError:
            pass
    return cleaned


def load_round_state(round_dir: Path) -> RoundState:
    round_state_path = round_dir / "round-state.json"
    data = load_json_object(round_state_path, display_name="round-state.json")
    missing_fields = [
        field_name for field_name in ROUND_STATE_REQUIRED_FIELDS if field_name not in data
    ]
    if missing_fields:
        raise ValueError("missing required round-state fields: " + ", ".join(missing_fields))

    evidence_sources_value = data["evidence_sources"]
    if not isinstance(evidence_sources_value, list):
        raise ValueError("round-state evidence_sources must be a list of strings")
    evidence_sources_raw = cast(list[Any], evidence_sources_value)
    evidence_sources: list[str] = []
    for item in evidence_sources_raw:
        if not isinstance(item, str):
            raise ValueError("round-state evidence_sources must be a list of strings")
        evidence_sources.append(item)

    return RoundState(
        round_name=str(data["round"]),
        parent_round=str(data["parent_round"]),
        hypothesis=str(data["hypothesis"]),
        evidence_sources=tuple(evidence_sources),
        correctness_status=str(data["correctness_status"]),
        benchmark_status=str(data["benchmark_status"]),
        perf_artifact=str(data["perf_artifact"]),
        comparison_target=str(data["comparison_target"]),
        effective_metric_source=str(data["effective_metric_source"]),
        summary_path=str(data["summary_path"]),
        opt_note_updated=bool(data["opt_note_updated"]),
        analysis_skipped_reason=optional_str(data.get("analysis_skipped_reason")),
        profile_dir=optional_str(data.get("profile_dir")),
        ir_dir=optional_str(data.get("ir_dir")),
        perf_analysis_path=optional_str(data.get("perf_analysis_path")),
    )


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    workspace = round_dir.parent
    attempts_path = existing_file(round_dir / "attempts.md")
    round_state_path = existing_file(round_dir / "round-state.json")
    state = _load_optional_round_state(round_dir, round_state_path)
    declared_paths = _declared_round_paths(round_dir, workspace, state)
    summary_path, perf_path = _resolve_round_artifact_paths(round_dir, declared_paths)
    operator_path = resolve_round_operator_file(round_dir)
    scan = _RoundArtifactScan(
        workspace=workspace,
        attempts_path=attempts_path,
        round_state_path=round_state_path,
        state=state,
        declared=declared_paths,
        summary_path=summary_path,
        perf_path=perf_path,
        operator_path=operator_path,
    )
    issues = _round_artifact_issues(scan)
    return RoundArtifactsInspection(
        round_dir=round_dir,
        operator_path=operator_path,
        attempts_path=attempts_path,
        summary_path=summary_path,
        perf_path=perf_path,
        perf_analysis_path=declared_paths.perf_analysis_path,
        round_state_path=round_state_path,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class _DeclaredRoundPaths:
    summary_name: str | None
    perf_name: str | None
    analysis_name: str | None
    summary_path: Path | None
    perf_path: Path | None
    perf_analysis_path: Path | None


@dataclass(frozen=True)
class _RoundArtifactScan:
    workspace: Path
    attempts_path: Path | None
    round_state_path: Path | None
    state: RoundState | None
    declared: _DeclaredRoundPaths
    summary_path: Path | None
    perf_path: Path | None
    operator_path: Path | None


def _load_optional_round_state(round_dir: Path, path: Path | None) -> RoundState | None:
    if path is None:
        return None
    try:
        return load_round_state(round_dir)
    except ValueError:
        return None


def _declared_round_paths(
    round_dir: Path,
    workspace: Path,
    state: RoundState | None,
) -> _DeclaredRoundPaths:
    summary_name = state.summary_path if state is not None else None
    perf_name = state.perf_artifact if state is not None else None
    analysis_name = state.perf_analysis_path if state is not None else None
    return _DeclaredRoundPaths(
        summary_name=summary_name,
        perf_name=perf_name,
        analysis_name=analysis_name,
        summary_path=declared_state_file(round_dir, workspace, summary_name) if state else None,
        perf_path=declared_state_file(round_dir, workspace, perf_name) if state else None,
        perf_analysis_path=(
            declared_state_file(round_dir, workspace, analysis_name) if state else None
        ),
    )


def _resolve_round_artifact_paths(
    round_dir: Path,
    declared_paths: _DeclaredRoundPaths,
) -> tuple[Path | None, Path | None]:
    summary_path = declared_paths.summary_path or existing_file(round_dir / "summary.md")
    perf_path = declared_paths.perf_path or resolve_round_perf_file(round_dir)
    return summary_path, perf_path


def _round_artifact_issues(scan: _RoundArtifactScan) -> list[str]:
    operator_name, perf_name = _expected_round_artifact_names(scan.workspace)
    issues = _required_round_artifact_issues(scan, operator_name, perf_name)
    if scan.state is not None:
        _append_declared_path_issues(
            issues,
            scan.declared,
            scan.summary_path,
            scan.perf_path,
            perf_name,
        )
    return issues


def _required_round_artifact_issues(
    scan: _RoundArtifactScan,
    operator_name: str,
    perf_name: str,
) -> list[str]:
    issues: list[str] = []
    if scan.attempts_path is None:
        issues.append("missing attempts.md")
    if scan.summary_path is None:
        issues.append(missing_issue(scan.declared.summary_name, default_path="summary.md"))
    if scan.round_state_path is None:
        issues.append("missing round-state.json")
    if scan.perf_path is None:
        issues.append(missing_issue(scan.declared.perf_name, default_path=perf_name))
    if scan.operator_path is None:
        issues.append(f"missing {operator_name}")
    return issues


def _append_declared_path_issues(
    issues: list[str],
    declared: _DeclaredRoundPaths,
    summary_path: Path | None,
    perf_path: Path | None,
    expected_perf_name: str,
) -> None:
    summary_name_mismatch = (
        declared.summary_name is not None
        and summary_path is not None
        and Path(declared.summary_name).name != summary_path.name
    )
    if summary_name_mismatch:
        issues.append("summary_path must be summary.md")
    perf_name_mismatch = (
        declared.perf_name is not None
        and perf_path is not None
        and Path(declared.perf_name).name != perf_path.name
    )
    if perf_name_mismatch:
        issues.append(f"perf_artifact must be {expected_perf_name}")
    if declared.analysis_name is not None and declared.perf_analysis_path is None:
        issues.append(missing_issue(declared.analysis_name, default_path="perf-analysis.md"))


def is_completed_round_directory(round_dir: Path) -> bool:
    if not round_dir.is_dir():
        return False
    name = round_dir.name
    if not name.startswith("opt-round-"):
        return False
    suffix = name[len("opt-round-"):]
    if not suffix.isdigit():
        return False

    inspection, round_state, _state_error = _inspect_round_minimum_artifact_package(round_dir)
    if inspection.issues or round_state is None:
        return False

    return (
        round_state.correctness_status == "passed"
        and round_state.benchmark_status == "passed"
    )


def iter_completed_round_directories(workspace: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(workspace.glob("opt-round-*"))
        if is_completed_round_directory(path)
    )


def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
    inspection, round_state, state_error = _inspect_round_minimum_artifact_package(round_dir)
    preflight = _round_preflight_failure(inspection, round_state, state_error)
    if preflight is not None:
        return preflight
    if round_state is None:
        raise ValueError("round-state validation completed without a round state")
    semantic_issues, baseline_perf_path = _round_semantic_issues(round_dir, round_state)
    if semantic_issues:
        return build_check_result(kind="round", status="fail", issues=tuple(semantic_issues))
    continuity_failure = _round_continuity_failure(round_dir, inspection)
    if continuity_failure is not None:
        return continuity_failure
    result = _build_round_pass_result(
        round_dir,
        round_state,
        baseline_perf_path,
        optimize_target,
    )
    return _add_next_round_guidance(result, current_round, final_round)


def _round_preflight_failure(
    inspection: RoundArtifactsInspection,
    round_state: RoundState | None,
    state_error: str | None,
) -> OptimizeCheckResult | None:
    if inspection.issues:
        return build_check_result(kind="round", status="fail", issues=inspection.issues)
    if state_error is not None:
        return build_check_result(kind="round", status="fail", issues=(state_error,))
    if round_state is None:
        raise ValueError("round_state is None")
    status_issue = _round_status_issue(round_state)
    if status_issue is not None:
        return build_check_result(kind="round", status="fail", issues=(status_issue,))
    baseline_issues = baseline_gate_issues(inspection.round_dir.parent)
    if baseline_issues:
        return build_check_result(kind="round", status="fail", issues=baseline_issues)
    return None


def _round_status_issue(round_state: RoundState) -> str | None:
    if round_state.correctness_status != "passed":
        return f"correctness_status={round_state.correctness_status}"
    if round_state.benchmark_status != "passed":
        return f"benchmark_status={round_state.benchmark_status}"
    return None


def _round_semantic_issues(
    round_dir: Path,
    round_state: RoundState,
) -> tuple[list[str], Path | None]:
    issues, baseline_perf_path = _comparison_target_issues(round_dir, round_state)
    if round_state.effective_metric_source not in {"kernel", "total-op", "mixed"}:
        issues.append(f"effective_metric_source={round_state.effective_metric_source}")
    if not round_state.evidence_sources:
        issues.append("missing supporting evidence sources")
    return issues, baseline_perf_path


def _comparison_target_issues(
    round_dir: Path,
    round_state: RoundState,
) -> tuple[list[str], Path | None]:
    issues: list[str] = []
    baseline_perf_path: Path | None = None
    try:
        baseline = load_baseline_state(round_dir.parent)
        baseline_perf_path = declared_state_file(
            baseline_dir(round_dir.parent), round_dir.parent, baseline.perf_artifact
        )
        issue = _comparison_target_issue(
            round_dir,
            round_state,
            baseline.perf_artifact,
            baseline_perf_path,
        )
        if issue is not None:
            issues.append(issue)
    except ValueError:
        issues.append("cannot validate comparison_target: baseline state is invalid")
    return issues, baseline_perf_path


def _comparison_target_issue(
    round_dir: Path,
    round_state: RoundState,
    baseline_perf_artifact: str,
    baseline_perf_path: Path | None,
) -> str | None:
    comparison_target = round_state.comparison_target
    comparison_target_path = declared_state_file(
        round_dir,
        round_dir.parent,
        comparison_target,
    )
    expected_target = (
        os.path.relpath(baseline_perf_path.resolve(), round_dir)
        if baseline_perf_path is not None
        else None
    )
    if comparison_target_path is None:
        return missing_issue(comparison_target, default_path=expected_target or comparison_target)
    if baseline_perf_path is not None and comparison_target_path.resolve() != baseline_perf_path.resolve():
        return (
            f"comparison_target={comparison_target} "
            f"(expected {expected_target or baseline_perf_artifact})"
        )
    return None


def _round_continuity_failure(
    round_dir: Path,
    inspection: RoundArtifactsInspection,
) -> OptimizeCheckResult | None:
    operator_path = inspection.operator_path
    if operator_path is None:
        operator_name, _perf_name = _expected_round_artifact_names(round_dir.parent)
        return build_check_result(kind="round", status="fail", issues=(f"missing {operator_name}",))
    continuity = analyze_triton_kernel_continuity(operator_path)
    if not continuity.ok:
        issue = continuity.reason or "round operator failed Triton continuity check"
        return build_check_result(kind="round", status="fail", issues=(issue,))
    return None


def _build_round_pass_result(
    round_dir: Path,
    round_state: RoundState,
    baseline_perf_path: Path | None,
    optimize_target: Literal["kernel", "operator"] | None,
) -> OptimizeCheckResult:
    if ordinary_optimize_pt_cleanup_enabled():
        cleanup_dir_pt_files(round_dir)
    cleanup_dir_prof_artifacts(round_dir)
    issues = _round_runtime_warnings(round_state, optimize_target)
    if baseline_perf_path is not None:
        issues.extend(collect_local_optimum_warnings(round_dir, baseline_perf_path=baseline_perf_path))
    return build_check_result(kind="round", status="pass", issues=tuple(issues))


def _round_runtime_warnings(
    round_state: RoundState,
    optimize_target: Literal["kernel", "operator"] | None,
) -> list[str]:
    if optimize_target != "kernel" or round_state.effective_metric_source not in {"total-op", "mixed"}:
        return []
    return [
        "kernel optimize target fell back to "
        f"effective_metric_source={round_state.effective_metric_source}; "
        "the round may still participate in best-round selection, but review the comparison basis."
    ]


def _add_next_round_guidance(
    result: OptimizeCheckResult,
    current_round: int | None,
    final_round: int | None,
) -> OptimizeCheckResult:
    if current_round is None or final_round is None:
        return result
    if current_round >= final_round:
        summary = "round check passed. This round satisfied the current worker batch target."
        return _round_result_with_guidance(result, summary, None)
    next_round_name = f"opt-round-{current_round + 1}"
    summary = (
        f"round check passed. Round {current_round}/{final_round} in the current worker batch is complete. "
        f"Next round: {next_round_name}. Use the staged `triton-npu-optimize-state` skill's "
        f"`start-round` subcommand to open {next_round_name} before beginning the next round."
    )
    return _round_result_with_guidance(result, summary, next_round_name)


def _round_result_with_guidance(
    result: OptimizeCheckResult,
    summary: str,
    next_option: str | None,
) -> OptimizeCheckResult:
    return build_check_result(
        kind="round",
        status="pass",
        issues=result.issues,
        summary=append_pass_issues_to_summary(summary, result.issues),
        next_option=next_option,
    )


def resolve_round_perf_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        perf_name = expected_round_perf_name(workspace)
    except ValueError:
        perf_name = None
    if perf_name is not None:
        perf_path = round_dir / perf_name
        if perf_path.is_file():
            return perf_path

    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt

    perf_files = sorted(path for path in round_dir.glob("*_perf.txt") if path.is_file())
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def resolve_round_operator_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        operator_name = expected_round_operator_name(workspace)
    except ValueError:
        operator_name = None
    if operator_name is not None:
        operator_path = round_dir / operator_name
        if operator_path.is_file():
            return operator_path

    try:
        legacy_operator_name = resolve_workspace_operator_file(workspace).name
    except ValueError:
        legacy_operator_name = None
    if legacy_operator_name is not None:
        legacy_operator_path = round_dir / legacy_operator_name
        if legacy_operator_path.is_file():
            return legacy_operator_path

    candidates = []
    for path in sorted(round_dir.iterdir()):
        if (
            path.is_file()
            and path.name not in _ROUND_METADATA_FILENAMES
            and not path.name.endswith("_perf.txt")
        ):
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        preferred_python = [path for path in candidates if path.suffix == ".py"]
        if len(preferred_python) == 1:
            return preferred_python[0]
        return candidates[0]
    return None


def _inspect_round_minimum_artifact_package(
    round_dir: Path,
) -> tuple[RoundArtifactsInspection, RoundState | None, str | None]:
    artifact_inspection = inspect_round_artifacts(round_dir)
    if artifact_inspection.issues:
        return artifact_inspection, None, None
    try:
        return artifact_inspection, load_round_state(round_dir), None
    except ValueError as exc:
        return artifact_inspection, None, str(exc)


def _expected_round_artifact_names(workspace: Path) -> tuple[str, str]:
    try:
        return expected_round_operator_name(workspace), expected_round_perf_name(workspace)
    except ValueError:
        return "opt_<operator>.py", "opt_<operator>_perf.txt"
