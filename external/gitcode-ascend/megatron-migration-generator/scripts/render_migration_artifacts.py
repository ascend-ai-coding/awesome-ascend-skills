#!/usr/bin/env python3
"""
Render first-pass migration artifacts from an impact report JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def format_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value or "unknown")


def fenced_block(language: str, body: str) -> list[str]:
    clean = body.rstrip("\n")
    return [f"```{language}", clean, "```", ""]


def render_path_table(paths: list[str]) -> list[str]:
    lines = ["| Candidate Path |", "|---|"]
    if not paths:
        lines.append("| `<none>` |")
    else:
        for path in paths:
            lines.append(f"| `{path}` |")
    return lines


def render_path_table_zh(paths: list[str]) -> list[str]:
    lines = ["| 候选路径 |", "|---|"]
    if not paths:
        lines.append("| `<无>` |")
    else:
        for path in paths:
            lines.append(f"| `{path}` |")
    return lines


def slugify(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s/]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "item"


def slugify_filename(text: str, suffix: str = ".patch") -> str:
    base = slugify(text)
    return f"{base}{suffix}"


def ensure_patch_header(patch_text: str, item: dict[str, Any]) -> str:
    patch = patch_text.rstrip("\n")
    if patch.startswith("diff --git "):
        return patch + "\n"
    if patch.startswith("--- a/"):
        first_line = patch.splitlines()[0]
        path = first_line.removeprefix("--- a/").strip()
        return f"diff --git a/{path} b/{path}\n{patch}\n"

    candidate_paths = item.get("candidate_paths", [])
    target_path = candidate_paths[0] if candidate_paths else "UNKNOWN_PATH"
    target_path = target_path.lstrip("./")
    header = [
        f"diff --git a/{target_path} b/{target_path}",
        f"--- a/{target_path}",
        f"+++ b/{target_path}",
    ]
    if patch.startswith("@@"):
        header.append(patch)
    else:
        header.append("@@")
        header.append(patch)
    return "\n".join(header).rstrip("\n") + "\n"


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def extract_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git a/") and " b/" in line:
            path = line.split(" b/", 1)[0].removeprefix("diff --git a/").strip()
            paths.append(path)
        elif line.startswith("--- a/"):
            path = line.removeprefix("--- a/").strip()
            if path and path not in paths:
                paths.append(path)
    return paths


def build_patch_series(item: dict[str, Any]) -> list[dict[str, str]]:
    series = item.get("patch_series")
    if isinstance(series, list) and series:
        entries: list[dict[str, str]] = []
        for idx, part in enumerate(series, start=1):
            if not isinstance(part, dict):
                continue
            raw_patch = part.get("patch") or part.get("patch_snippet") or part.get("content")
            if not raw_patch:
                continue
            name = str(part.get("name") or f"part-{idx}")
            entries.append(
                {
                    "name": name,
                    "purpose": str(part.get("purpose") or "").strip(),
                    "patch": ensure_patch_header(str(raw_patch), item),
                    "filename": f"{idx:04d}-{slugify_filename(name)}",
                }
            )
        if entries:
            return entries

    raw_patch = item.get("full_patch") or item.get("patch_snippet") or item.get("code_snippet")
    if not raw_patch:
        return []
    return [
        {
            "name": "full-migration-draft",
            "purpose": "Single-file migration draft generated from the available impact data.",
            "patch": ensure_patch_header(str(raw_patch), item),
            "filename": "0001-full-migration-draft.patch",
        }
    ]


def package_level(item: dict[str, Any], patch_series: list[dict[str, str]]) -> str:
    if item.get("full_patch") or (item.get("patch_series") and len(patch_series) > 1):
        return "full_reference_package"
    if patch_series:
        return "single_patch_reference"
    return "report_only"


def render_bullet_section(title: str, values: list[str]) -> list[str]:
    lines = [title, ""]
    if values:
        for value in values:
            lines.append(f"- {value}")
    else:
        lines.append("- `<无>`")
    lines.append("")
    return lines


def render_bullet_section_en(title: str, values: list[str]) -> list[str]:
    lines = [title, ""]
    if values:
        for value in values:
            lines.append(f"- {value}")
    else:
        lines.append("- `<none>`")
    lines.append("")
    return lines


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n")


def render_implementation_targets_zh(targets: list[dict[str, Any]]) -> list[str]:
    lines = ["## 本地实现目标", ""]
    if not targets:
        lines.append("- `<无>`")
        lines.append("")
        return lines
    for idx, target in enumerate(targets, start=1):
        lines.append(f"### {idx}. {target.get('name', '未命名实现目标')}")
        lines.append("")
        lines.append(f"- 来源实现单元：{target.get('source_unit', '未提供')}")
        lines.append(f"- 需要改动：{target.get('required_change', '未提供')}")
        lines.append(f"- 置信度：`{format_confidence(target.get('confidence'))}`")
        target_paths = listify(target.get('candidate_paths'))
        if target_paths:
            lines.append("- 候选路径：")
            for path in target_paths:
                lines.append(f"  - `{path}`")
        lines.append("")
    return lines


def render_upstream_units_zh(units: list[dict[str, Any]]) -> list[str]:
    lines = ["## 上游实现单元", ""]
    if not units:
        lines.append("- `<无>`")
        lines.append("")
        return lines
    for idx, unit in enumerate(units, start=1):
        lines.append(f"### {idx}. {unit.get('name', '未命名单元')}")
        lines.append("")
        lines.append(f"- 类型：`{unit.get('kind', 'unknown')}`")
        summary = unit.get("summary")
        if summary:
            lines.append(f"- 说明：{summary}")
        upstream_files = listify(unit.get("upstream_files"))
        if upstream_files:
            lines.append("- 上游文件：")
            for path in upstream_files:
                lines.append(f"  - `{path}`")
        lines.append("")
    return lines


def render_implementation_targets_en(targets: list[dict[str, Any]]) -> list[str]:
    lines = ["## Local Implementation Targets", ""]
    if not targets:
        lines.append("- `<none>`")
        lines.append("")
        return lines
    for idx, target in enumerate(targets, start=1):
        lines.append(f"### {idx}. {target.get('name', 'Untitled target')}")
        lines.append("")
        lines.append(f"- Source unit: {target.get('source_unit', 'unknown')}")
        lines.append(f"- Required change: {target.get('required_change', 'Not provided')}")
        lines.append(f"- Confidence: `{format_confidence(target.get('confidence'))}`")
        target_paths = listify(target.get('candidate_paths'))
        if target_paths:
            lines.append("- Candidate paths:")
            for path in target_paths:
                lines.append(f"  - `{path}`")
        lines.append("")
    return lines


def render_upstream_units_en(units: list[dict[str, Any]]) -> list[str]:
    lines = ["## Upstream Implementation Units", ""]
    if not units:
        lines.append("- `<none>`")
        lines.append("")
        return lines
    for idx, unit in enumerate(units, start=1):
        lines.append(f"### {idx}. {unit.get('name', 'Untitled unit')}")
        lines.append("")
        lines.append(f"- Kind: `{unit.get('kind', 'unknown')}`")
        summary = unit.get("summary")
        if summary:
            lines.append(f"- Summary: {summary}")
        upstream_files = listify(unit.get("upstream_files"))
        if upstream_files:
            lines.append("- Upstream files:")
            for path in upstream_files:
                lines.append(f"  - `{path}`")
        lines.append("")
    return lines


def write_item_artifacts(outdir: Path, data: dict[str, Any], locale: str) -> list[dict[str, str]]:
    items_dir = outdir / "features"
    items_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for idx, item in enumerate(data.get("items", []), start=1):
        folder_name = f"{idx:02d}-{slugify(item.get('event_title', f'item-{idx}'))}"
        item_dir = items_dir / folder_name
        item_dir.mkdir(parents=True, exist_ok=True)
        patches_dir = item_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        patch_series = build_patch_series(item)
        main_patch_path = item_dir / "full.patch"
        if patch_series:
            main_patch_path.write_text("".join(entry["patch"] for entry in patch_series))
        else:
            main_patch_path.write_text("")

        for entry in patch_series:
            (patches_dir / entry["filename"]).write_text(entry["patch"])

        candidate_path = item_dir / "candidate.patch"
        if patch_series:
            candidate_path.write_text(patch_series[0]["patch"])
        else:
            candidate_path.write_text("")

        notes_path = item_dir / "README.md"
        package_manifest_path = item_dir / "package_manifest.json"
        checklist_path = item_dir / "checklist.md"
        upstream_reference_path = item_dir / "upstream_reference.json"

        covered_scope = listify(item.get("covered_scope"))
        omitted_scope = listify(item.get("omitted_scope"))
        manual_followups = listify(item.get("manual_followups"))
        upstream_changed_files = listify(item.get("upstream_changed_files"))
        local_changed_files = listify(item.get("local_changed_files")) or listify(item.get("candidate_paths"))
        implementation_notes = listify(item.get("implementation_notes"))
        validation_checks = listify(item.get("validation_checks"))
        package_notes = listify(item.get("package_notes"))
        implementation_targets = list_of_dicts(item.get("implementation_targets"))
        upstream_units = list_of_dicts(item.get("implementation_units"))
        upstream_commits = listify(item.get("commits"))
        primary_commit = str(item.get("primary_commit") or "").strip()
        if primary_commit and primary_commit not in upstream_commits:
            upstream_commits = [primary_commit] + upstream_commits
        package_mode = package_level(item, patch_series)
        touched_paths: list[str] = []
        for entry in patch_series:
            touched_paths.extend(extract_patch_paths(entry["patch"]))
        touched_paths = list(dict.fromkeys(touched_paths))

        if not covered_scope and touched_paths:
            covered_scope = [f"当前 patch 已直接覆盖 `{path}`" for path in touched_paths]

        if not omitted_scope:
            for path in item.get("candidate_paths", []):
                if path not in touched_paths:
                    omitted_scope.append(f"候选路径 `{path}` 尚未被当前 patch 直接覆盖")

        if not manual_followups and omitted_scope:
            manual_followups = [
                "结合上游完整 diff，继续补齐 README 中标记为“尚未覆盖”的本地适配文件。",
                "补充参数透传、配置暴露、测试用例和调用链兼容改动，再决定是否进入正式提交。",
            ]

        if locale == "zh_CN":
            note_lines = [
                f"# {item.get('event_title', '未命名事项')}",
                "",
                f"- 状态：`{item.get('status', 'unknown')}`",
                f"- 置信度：`{format_confidence(item.get('confidence'))}`",
                f"- 包类型：`{package_mode}`",
                f"- 建议改动：{item.get('proposed_edit', '未提供说明。')}",
                "",
                "## 迁移包内容",
                "",
                f"- 主补丁：`{main_patch_path.name}`",
                f"- 兼容补丁入口：`{candidate_path.name}`",
                f"- 拆分补丁目录：`{patches_dir.name}/`",
                f"- 清单文件：`{package_manifest_path.name}`",
                f"- 上游参考：`{upstream_reference_path.name}`",
                f"- 检查单：`{checklist_path.name}`",
                "",
                "## 上游参考提交",
                "",
            ]
            if upstream_commits:
                for commit in upstream_commits:
                    note_lines.append(f"- `{commit}`")
            else:
                note_lines.append("- `<无>`")
            note_lines.extend(
                [
                    "",
                "## 候选路径",
                "",
                ]
            )
            for path in item.get("candidate_paths", []):
                note_lines.append(f"- `{path}`")
            note_lines.append("")
            note_lines.extend(render_bullet_section("## 上游改动文件", upstream_changed_files))
            note_lines.extend(render_upstream_units_zh(upstream_units))
            note_lines.extend(render_implementation_targets_zh(implementation_targets))
            note_lines.extend(render_bullet_section("## 上游覆盖范围", covered_scope))
            note_lines.extend(render_bullet_section("## 暂未覆盖的内容", omitted_scope))
            note_lines.extend(render_bullet_section("## 人工继续开发建议", manual_followups))
            note_lines.extend(render_bullet_section("## 实现说明", implementation_notes))
            note_lines.extend(render_bullet_section("## 静态检查建议", validation_checks))
            note_lines.extend(render_bullet_section("## 包说明", package_notes))
            note_lines.extend(
                [
                    "## 使用方式",
                    "",
                    f"```bash\ngit apply {main_patch_path.name}\n```",
                    "",
                    "也可以按顺序应用拆分补丁：",
                    "",
                    "```bash",
                    f"git apply {patches_dir.name}/*.patch",
                    "```",
                    "",
                    "## 说明",
                    "",
                    "- 该迁移包以“完整迁移参考”为目标，便于开发人员快速理解改动范围。",
                    "- 如果 package 仍不完整，README 中的“暂未覆盖的内容”和“人工继续开发建议”会明确标出。",
                ]
            )
        else:
            note_lines = [
                f"# {item.get('event_title', 'Untitled event')}",
                "",
                f"- Status: `{item.get('status', 'unknown')}`",
                f"- Confidence: `{format_confidence(item.get('confidence'))}`",
                f"- Package type: `{package_mode}`",
                f"- Intended change: {item.get('proposed_edit', 'No description provided.')}",
                "",
                "## Package Contents",
                "",
                f"- Main patch: `{main_patch_path.name}`",
                f"- Compatibility entry patch: `{candidate_path.name}`",
                f"- Patch series directory: `{patches_dir.name}/`",
                f"- Manifest: `{package_manifest_path.name}`",
                f"- Upstream reference: `{upstream_reference_path.name}`",
                f"- Checklist: `{checklist_path.name}`",
                "",
                "## Upstream Reference Commits",
                "",
            ]
            if upstream_commits:
                for commit in upstream_commits:
                    note_lines.append(f"- `{commit}`")
            else:
                note_lines.append("- `<none>`")
            note_lines.extend(
                [
                    "",
                "## Candidate Paths",
                "",
                ]
            )
            for path in item.get("candidate_paths", []):
                note_lines.append(f"- `{path}`")
            note_lines.append("")
            note_lines.extend(render_bullet_section_en("## Upstream Changed Files", upstream_changed_files))
            note_lines.extend(render_upstream_units_en(upstream_units))
            note_lines.extend(render_implementation_targets_en(implementation_targets))
            note_lines.extend(render_bullet_section_en("## Covered Upstream Scope", covered_scope))
            note_lines.extend(render_bullet_section_en("## Omitted Scope", omitted_scope))
            note_lines.extend(render_bullet_section_en("## Manual Follow-ups", manual_followups))
            note_lines.extend(render_bullet_section_en("## Implementation Notes", implementation_notes))
            note_lines.extend(render_bullet_section_en("## Suggested Static Checks", validation_checks))
            note_lines.extend(render_bullet_section_en("## Package Notes", package_notes))
            note_lines.extend(
                [
                    "## Usage",
                    "",
                    f"```bash\ngit apply {main_patch_path.name}\n```",
                    "",
                    "Or apply the split patch series in order:",
                    "",
                    "```bash",
                    f"git apply {patches_dir.name}/*.patch",
                    "```",
                    "",
                    "## Notes",
                    "",
                    "- This migration package is intended as a fuller engineering reference.",
                    "- Any remaining gaps should be called out in the omitted scope and follow-up sections above.",
                ]
            )
        write_text(notes_path, note_lines)

        checklist_lines = []
        if locale == "zh_CN":
            checklist_lines.extend(["# 检查单", ""])
            checklist_lines.extend(render_bullet_section("## 本地候选改动文件", local_changed_files))
            checklist_lines.extend(render_bullet_section("## 上游涉及文件", upstream_changed_files))
            checklist_lines.extend(render_bullet_section("## 静态检查建议", validation_checks))
            checklist_lines.extend(render_bullet_section("## 人工继续开发建议", manual_followups))
        else:
            checklist_lines.extend(["# Checklist", ""])
            checklist_lines.extend(render_bullet_section_en("## Candidate Local Files", local_changed_files))
            checklist_lines.extend(render_bullet_section_en("## Upstream Files", upstream_changed_files))
            checklist_lines.extend(render_bullet_section_en("## Suggested Static Checks", validation_checks))
            checklist_lines.extend(render_bullet_section_en("## Manual Follow-ups", manual_followups))
        write_text(checklist_path, checklist_lines)

        upstream_reference = {
            "event_title": item.get("event_title", f"item-{idx}"),
            "primary_commit": primary_commit or None,
            "commits": upstream_commits,
            "upstream_changed_files": upstream_changed_files,
            "implementation_units": upstream_units,
            "upstream_evidence": listify(item.get("upstream_evidence")),
        }
        upstream_reference_path.write_text(json.dumps(upstream_reference, indent=2, ensure_ascii=False))

        package_manifest = {
            "event_title": item.get("event_title", f"item-{idx}"),
            "status": item.get("status", "unknown"),
            "confidence": item.get("confidence", "unknown"),
            "package_type": package_mode,
            "main_patch": str(main_patch_path),
            "candidate_patch": str(candidate_path),
            "patch_series_dir": str(patches_dir),
            "patch_series_files": [str(patches_dir / entry["filename"]) for entry in patch_series],
            "upstream_reference_file": str(upstream_reference_path),
            "primary_commit": primary_commit or None,
            "commits": upstream_commits,
            "candidate_paths": item.get("candidate_paths", []),
            "local_changed_files": local_changed_files,
            "upstream_changed_files": upstream_changed_files,
            "implementation_targets": implementation_targets,
            "implementation_units": upstream_units,
            "covered_scope": covered_scope,
            "omitted_scope": omitted_scope,
            "manual_followups": manual_followups,
        }
        package_manifest_path.write_text(json.dumps(package_manifest, indent=2, ensure_ascii=False))

        manifest.append(
            {
                "event_title": item.get("event_title", f"item-{idx}"),
                "folder": str(item_dir),
                "patch_file": str(candidate_path),
                "full_patch_file": str(main_patch_path),
                "patch_series_dir": str(patches_dir),
                "readme_file": str(notes_path),
                "checklist_file": str(checklist_path),
                "package_manifest_file": str(package_manifest_path),
                "upstream_reference_file": str(upstream_reference_path),
                "package_type": package_mode,
            }
        )

    return manifest


def render_code_section(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    patch_snippet = item.get("patch_snippet") or item.get("code_snippet")
    implementation_notes = item.get("implementation_notes")
    validation_checks = item.get("validation_checks", [])

    if patch_snippet:
        lines.append("#### Suggested Code")
        lines.append("")
        language = item.get("snippet_language", "diff")
        lines.extend(fenced_block(language, patch_snippet))

    if implementation_notes:
        lines.append("#### Implementation Notes")
        lines.append("")
        if isinstance(implementation_notes, list):
            for note in implementation_notes:
                lines.append(f"- {note}")
        else:
            lines.append(f"- {implementation_notes}")
        lines.append("")

    if validation_checks:
        lines.append("#### Suggested Static Checks")
        lines.append("")
        for check in validation_checks:
            lines.append(f"- {check}")
        lines.append("")

    return lines


def build_integrated_report(data: dict[str, Any], report_title: str, item_manifest: list[dict[str, str]] | None = None) -> str:
    lines: list[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("> This report is a first-pass migration package generated from an impact report.")
    lines.append("> It is designed for review and refinement, not as a claim of runtime validation.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This document consolidates branch alignment, upstream change impact, suggested migration work, and candidate code snippets into one reviewable markdown report."
    )
    lines.append("")
    lines.append("## Migration Context")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| MindSpeed branch | `{data.get('mindspeed_branch', 'unknown')}` |")
    lines.append(f"| Megatron base branch | `{data.get('megatron_base_branch', 'unknown')}` |")
    lines.append(f"| Megatron target branch | `{data.get('megatron_target_branch', 'unknown')}` |")
    lines.append(f"| Alignment mode | `{data.get('mode', 'unknown')}` |")
    lines.append("")

    items = data.get("items", [])
    lines.append("## Key Findings")
    lines.append("")
    lines.append("| Event | Status | Confidence |")
    lines.append("|---|---|---|")
    if not items:
        lines.append("| No impact items provided | `n/a` | `n/a` |")
    else:
        for item in items:
            lines.append(
                f"| {item.get('event_title', 'Untitled event')} | `{item.get('status', 'unknown')}` | `{format_confidence(item.get('confidence'))}` |"
            )
    lines.append("")

    lines.append("## Detailed Migration Analysis")
    lines.append("")
    if not items:
        lines.append("No impact items were provided.")
        lines.append("")
    else:
        for idx, item in enumerate(items, start=1):
            lines.append(f"### {idx}. {item.get('event_title', 'Untitled event')}")
            lines.append("")
            lines.append("#### Assessment")
            lines.append("")
            lines.append(f"- Status: `{item.get('status', 'unknown')}`")
            lines.append(f"- Confidence: `{format_confidence(item.get('confidence'))}`")
            lines.append(f"- Why it matters: {item.get('reason', 'No rationale provided.')}")
            uncertainty = item.get("uncertainty")
            if uncertainty:
                lines.append(f"- Open uncertainty: {uncertainty}")
            lines.append("")

            lines.append("#### Candidate MindSpeed Paths")
            lines.append("")
            lines.extend(render_path_table(item.get("candidate_paths", [])))
            lines.append("")

            proposed_edit = item.get("proposed_edit")
            if proposed_edit:
                lines.append("#### Proposed Change")
                lines.append("")
                lines.append(proposed_edit)
                lines.append("")

            upstream_evidence = item.get("upstream_evidence", [])
            if upstream_evidence:
                lines.append("#### Upstream Evidence")
                lines.append("")
                for evidence in upstream_evidence:
                    lines.append(f"- {evidence}")
                lines.append("")

            upstream_commits = listify(item.get("commits"))
            primary_commit = item.get("primary_commit")
            if primary_commit and primary_commit not in upstream_commits:
                upstream_commits.insert(0, str(primary_commit))
            if upstream_commits:
                lines.append("#### Upstream Reference Commits")
                lines.append("")
                for commit in upstream_commits:
                    lines.append(f"- `{commit}`")
                lines.append("")

            upstream_changed_files = listify(item.get("upstream_changed_files"))
            if upstream_changed_files:
                lines.append("#### Upstream Changed Files")
                lines.append("")
                for path in upstream_changed_files:
                    lines.append(f"- `{path}`")
                lines.append("")

            implementation_units = list_of_dicts(item.get("implementation_units"))
            if implementation_units:
                lines.append("#### Upstream Implementation Units")
                lines.append("")
                for unit in implementation_units:
                    lines.append(f"- `{unit.get('name', 'Untitled unit')}`: {unit.get('summary', 'No summary provided.')}")
                lines.append("")

            implementation_targets = list_of_dicts(item.get("implementation_targets"))
            if implementation_targets:
                lines.append("#### Local Implementation Targets")
                lines.append("")
                for target in implementation_targets:
                    lines.append(
                        f"- `{target.get('name', 'Untitled target')}` -> {target.get('required_change', 'No required change provided.')}"
                    )
                lines.append("")

            if item_manifest:
                manifest_entry = item_manifest[idx - 1]
                lines.append("#### Artifact Files")
                lines.append("")
                lines.append(f"- Patch folder: `{Path(manifest_entry['folder']).name}`")
                lines.append(f"- Full patch: `{manifest_entry['full_patch_file']}`")
                lines.append(f"- Patch entry: `{manifest_entry['patch_file']}`")
                lines.append(f"- Patch series dir: `{manifest_entry['patch_series_dir']}`")
                lines.append(f"- Notes: `{manifest_entry['readme_file']}`")
                lines.append(f"- Checklist: `{manifest_entry['checklist_file']}`")
                lines.append(f"- Package manifest: `{manifest_entry['package_manifest_file']}`")
                lines.append(f"- Upstream reference: `{manifest_entry['upstream_reference_file']}`")
                lines.append("")

            lines.extend(render_code_section(item))

    lines.append("## Recommended Next Actions")
    lines.append("")
    lines.append("- Review branch alignment and confirm the target branch pair before applying any code changes.")
    lines.append("- Prioritize high-confidence items for implementation and leave exploratory items in report-only mode.")
    lines.append("- Run static checks after manual refinement. Do not treat this document as proof of training correctness.")
    lines.append("")

    report_notes = data.get("report_notes", [])
    if report_notes:
        lines.append("## Notes")
        lines.append("")
        for note in report_notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def build_integrated_report_zh(data: dict[str, Any], report_title: str, item_manifest: list[dict[str, str]] | None = None) -> str:
    lines: list[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("> 本报告基于 impact report 生成，属于第一版迁移分析交付物。")
    lines.append("> 它用于审阅和继续开发，不代表已经完成运行时验证。")
    lines.append("")
    lines.append("## 执行摘要")
    lines.append("")
    lines.append("本报告将分支对齐信息、上游变更影响、迁移建议以及候选代码片段整合为一份可审阅的 Markdown 文档，便于开发人员快速评估是否需要在 MindSpeed 中跟进适配。")
    lines.append("")
    lines.append("## 迁移上下文")
    lines.append("")
    lines.append("| 字段 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| MindSpeed 分支 | `{data.get('mindspeed_branch', 'unknown')}` |")
    lines.append(f"| Megatron 基线分支 | `{data.get('megatron_base_branch', 'unknown')}` |")
    lines.append(f"| Megatron 目标分支 | `{data.get('megatron_target_branch', 'unknown')}` |")
    lines.append(f"| 对齐模式 | `{data.get('mode', 'unknown')}` |")
    lines.append("")

    items = data.get("items", [])
    lines.append("## 核心发现")
    lines.append("")
    lines.append("| 变更事项 | 状态 | 置信度 |")
    lines.append("|---|---|---|")
    if not items:
        lines.append("| 未提供迁移影响项 | `n/a` | `n/a` |")
    else:
        for item in items:
            lines.append(
                f"| {item.get('event_title', '未命名事项')} | `{item.get('status', 'unknown')}` | `{format_confidence(item.get('confidence'))}` |"
            )
    lines.append("")

    lines.append("## 详细迁移分析")
    lines.append("")
    if not items:
        lines.append("未提供可分析的迁移影响项。")
        lines.append("")
    else:
        for idx, item in enumerate(items, start=1):
            lines.append(f"### {idx}. {item.get('event_title', '未命名事项')}")
            lines.append("")
            lines.append("#### 评估结论")
            lines.append("")
            lines.append(f"- 状态：`{item.get('status', 'unknown')}`")
            lines.append(f"- 置信度：`{format_confidence(item.get('confidence'))}`")
            lines.append(f"- 影响说明：{item.get('reason', '未提供说明。')}")
            uncertainty = item.get("uncertainty")
            if uncertainty:
                lines.append(f"- 不确定点：{uncertainty}")
            lines.append("")

            lines.append("#### 候选 MindSpeed 适配点")
            lines.append("")
            lines.extend(render_path_table_zh(item.get("candidate_paths", [])))
            lines.append("")

            proposed_edit = item.get("proposed_edit")
            if proposed_edit:
                lines.append("#### 建议改动")
                lines.append("")
                lines.append(proposed_edit)
                lines.append("")

            upstream_evidence = item.get("upstream_evidence", [])
            if upstream_evidence:
                lines.append("#### 上游依据")
                lines.append("")
                for evidence in upstream_evidence:
                    lines.append(f"- {evidence}")
                lines.append("")

            upstream_commits = listify(item.get("commits"))
            primary_commit = item.get("primary_commit")
            if primary_commit and primary_commit not in upstream_commits:
                upstream_commits.insert(0, str(primary_commit))
            if upstream_commits:
                lines.append("#### 上游参考提交")
                lines.append("")
                for commit in upstream_commits:
                    lines.append(f"- `{commit}`")
                lines.append("")

            upstream_changed_files = listify(item.get("upstream_changed_files"))
            if upstream_changed_files:
                lines.append("#### 上游改动文件")
                lines.append("")
                for path in upstream_changed_files:
                    lines.append(f"- `{path}`")
                lines.append("")

            implementation_units = list_of_dicts(item.get("implementation_units"))
            if implementation_units:
                lines.append("#### 上游实现单元")
                lines.append("")
                for unit in implementation_units:
                    lines.append(f"- `{unit.get('name', '未命名单元')}`：{unit.get('summary', '未提供说明。')}")
                lines.append("")

            implementation_targets = list_of_dicts(item.get("implementation_targets"))
            if implementation_targets:
                lines.append("#### 本地实现目标")
                lines.append("")
                for target in implementation_targets:
                    lines.append(
                        f"- `{target.get('name', '未命名目标')}` -> {target.get('required_change', '未提供改动说明。')}"
                    )
                lines.append("")

            if item_manifest:
                manifest_entry = item_manifest[idx - 1]
                lines.append("#### 交付文件")
                lines.append("")
                lines.append(f"- 特性目录：`{Path(manifest_entry['folder']).name}`")
                lines.append(f"- 完整补丁：`{manifest_entry['full_patch_file']}`")
                lines.append(f"- Patch 入口：`{manifest_entry['patch_file']}`")
                lines.append(f"- 拆分补丁目录：`{manifest_entry['patch_series_dir']}`")
                lines.append(f"- 说明文件：`{manifest_entry['readme_file']}`")
                lines.append(f"- 检查单：`{manifest_entry['checklist_file']}`")
                lines.append(f"- 清单文件：`{manifest_entry['package_manifest_file']}`")
                lines.append(f"- 上游参考：`{manifest_entry['upstream_reference_file']}`")
                lines.append("")

            patch_snippet = item.get("patch_snippet") or item.get("code_snippet")
            implementation_notes = item.get("implementation_notes")
            validation_checks = item.get("validation_checks", [])

            if patch_snippet:
                lines.append("#### 候选代码")
                lines.append("")
                language = item.get("snippet_language", "diff")
                lines.extend(fenced_block(language, patch_snippet))

            if implementation_notes:
                lines.append("#### 实现说明")
                lines.append("")
                if isinstance(implementation_notes, list):
                    for note in implementation_notes:
                        lines.append(f"- {note}")
                else:
                    lines.append(f"- {implementation_notes}")
                lines.append("")

            if validation_checks:
                lines.append("#### 建议静态检查")
                lines.append("")
                for check in validation_checks:
                    lines.append(f"- {check}")
                lines.append("")

    lines.append("## 后续建议")
    lines.append("")
    lines.append("- 在实际落地代码前，先确认本次分析使用的 MindSpeed 与 Megatron 分支对是否符合你的目标环境。")
    lines.append("- 优先处理高置信度事项；探索性事项建议先做小范围验证，再决定是否进入正式迁移。")
    lines.append("- 完成代码修改后先做静态检查，不要将本报告视为训练正确性或性能正确性的证明。")
    lines.append("")

    report_notes = data.get("report_notes", [])
    if report_notes:
        lines.append("## 备注")
        lines.append("")
        for note in report_notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def build_report(data: dict[str, Any]) -> str:
    return build_integrated_report(data, "Megatron to MindSpeed Migration Report")


def build_patch_plan(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Candidate Patch Plan")
    lines.append("")
    lines.append("This file is a reviewable patch plan, not a guaranteed apply-ready diff.")
    lines.append("")
    for idx, item in enumerate(data.get("items", []), start=1):
        lines.append(f"## Item {idx}: {item.get('event_title', 'Untitled event')}")
        lines.append("")
        lines.append(f"Status: `{item.get('status', 'unknown')}`")
        lines.append(f"Confidence: `{item.get('confidence', 'unknown')}`")
        lines.append("")
        lines.append("Candidate paths:")
        for path in item.get("candidate_paths", []) or ["<no candidate path>"]:
            lines.append(f"- `{path}`")
        lines.append("")
        lines.append("Intended change:")
        lines.append(f"- {item.get('proposed_edit', 'Describe the local adaptation here.')}")
        lines.append("")
        lines.append("Upstream rationale:")
        lines.append(f"- {item.get('reason', 'No rationale provided.')}")
        lines.append("")
    return "\n".join(lines)


def build_patch_plan_zh(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 候选补丁计划")
    lines.append("")
    lines.append("本文件用于辅助审阅，不代表可直接应用的最终 diff。")
    lines.append("")
    for idx, item in enumerate(data.get("items", []), start=1):
        lines.append(f"## 事项 {idx}: {item.get('event_title', '未命名事项')}")
        lines.append("")
        lines.append(f"状态：`{item.get('status', 'unknown')}`")
        lines.append(f"置信度：`{format_confidence(item.get('confidence'))}`")
        lines.append("")
        lines.append("候选路径：")
        for path in item.get("candidate_paths", []) or ["<无候选路径>"]:
            lines.append(f"- `{path}`")
        lines.append("")
        lines.append("建议改动：")
        lines.append(f"- {item.get('proposed_edit', '请补充本地适配思路。')}")
        lines.append("")
        lines.append("上游依据：")
        lines.append(f"- {item.get('reason', '未提供说明。')}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impact-report", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--title")
    parser.add_argument("--locale", default="zh_CN")
    args = parser.parse_args()

    data = load_json(Path(args.impact_report))
    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    item_manifest = write_item_artifacts(outdir, data, args.locale)

    report_path = outdir / "migration_report.md"
    patch_path = outdir / "candidate_patch.md"
    impact_copy_path = outdir / "impact_report.json"
    manifest_path = outdir / "feature_manifest.json"

    if args.locale == "zh_CN":
        title = args.title or "Megatron 到 MindSpeed 迁移分析报告"
        report_path.write_text(build_integrated_report_zh(data, title, item_manifest))
        patch_path.write_text(build_patch_plan_zh(data))
    else:
        title = args.title or "Megatron to MindSpeed Migration Report"
        report_path.write_text(build_integrated_report(data, title, item_manifest))
        patch_path.write_text(build_patch_plan(data))
    impact_copy_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    manifest_path.write_text(json.dumps(item_manifest, indent=2, ensure_ascii=False))

    print(
        json.dumps(
            {
                "feature_manifest": str(manifest_path),
                "migration_report": str(report_path),
                "candidate_patch": str(patch_path),
                "impact_report_copy": str(impact_copy_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
