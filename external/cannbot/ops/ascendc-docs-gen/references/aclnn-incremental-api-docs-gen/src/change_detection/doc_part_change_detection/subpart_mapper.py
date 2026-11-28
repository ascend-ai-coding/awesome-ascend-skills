# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.panel import Panel
from rich.table import Table

from src.change_detection.doc_part_change_detection.docpart_file_extractor import (
    DocpartFileExtractor,
)
from src.change_detection.doc_part_change_detection.file_change_detector import (
    FileChangeDetector,
)
from src.generation_validation.config import format_doc_part, format_doc_parts
from src.logger import debug, error, info, warning


def _ranges_overlap(range1: Optional[Tuple[int, int]], range2: Optional[Tuple[int, int]]) -> bool:
    if range1 is None or range2 is None:
        return True
    return not (range1[1] < range2[0] or range2[1] < range1[0])


def _is_example_file(file_path: str) -> bool:
    parts = Path(file_path).parts
    return "examples" in parts


DOC_SUFFIXES = (".md", ".rst", ".txt", ".json", ".yaml", ".yml")
SIMILARITY_THRESHOLD = 0.8


def _compute_similarity(stem1: str, stem2: str) -> float:
    return SequenceMatcher(None, stem1, stem2).ratio()


class ChangedFileMapper:
    def __init__(
        self,
        changed_files: List[str],
        file_subpart_mapping: Dict[str, List],
        changed_lines: Dict = None,
    ):
        self.changed_files = changed_files
        self.file_subpart_mapping = file_subpart_mapping
        self.changed_lines = changed_lines or {}
        self.fine_grained = changed_lines is not None and len(changed_lines) > 0

    def _build_reverse_mapping(self) -> Dict:
        reverse_map = {}
        for subpart, files in self.file_subpart_mapping.items():
            for file_entry in files:
                if isinstance(file_entry, dict):
                    file_name = file_entry.get("file_name", "")
                    line_range = file_entry.get("line_range")
                else:
                    file_name = file_entry
                    line_range = None

                if file_name not in reverse_map:
                    reverse_map[file_name] = set()
                reverse_map[file_name].add((subpart, line_range))
        return reverse_map

    def map_changed_files_to_subparts(self) -> Dict[str, List[str]]:
        reverse_mapping = self._build_reverse_mapping()
        result = {}

        for changed_file in self.changed_files:
            file_name = changed_file.replace("\\", "/").split("/")[-1]
            file_changed_ranges = self.changed_lines.get(changed_file, [])

            matched_subparts = set()

            if file_name in reverse_mapping:
                for subpart, line_range in reverse_mapping[file_name]:
                    if self.fine_grained and file_changed_ranges:
                        if any(_ranges_overlap(cr, line_range) for cr in file_changed_ranges):
                            matched_subparts.add(subpart)
                    else:
                        matched_subparts.add(subpart)

            if not matched_subparts:
                for src_file, subparts_with_ranges in reverse_mapping.items():
                    if self._files_match(file_name, src_file):
                        for subpart, line_range in subparts_with_ranges:
                            if self.fine_grained and file_changed_ranges:
                                if any(_ranges_overlap(cr, line_range) for cr in file_changed_ranges):
                                    matched_subparts.add(subpart)
                            else:
                                matched_subparts.add(subpart)

            if matched_subparts:
                result[changed_file] = sorted(list(matched_subparts))

        return result

    @staticmethod
    def _files_match(file1: str, file2: str) -> bool:
        name1 = file1.replace(".h", "").replace(".cpp", "").replace(".py", "")
        name2 = file2.replace(".h", "").replace(".cpp", "").replace(".py", "")
        return name1 == name2

    def get_affected_subparts(self) -> Set[str]:
        mapping = self.map_changed_files_to_subparts()
        affected = set()
        for subparts in mapping.values():
            affected.update(subparts)
        return affected

    def get_changed_files_by_subpart(self) -> Dict[str, List[str]]:
        mapping = self.map_changed_files_to_subparts()
        by_subpart = {}

        for file_path, subparts in mapping.items():
            for subpart in subparts:
                if subpart not in by_subpart:
                    by_subpart[subpart] = []
                by_subpart[subpart].append(file_path)

        return by_subpart


def map_changed_files_to_subparts(changed_files: List[str], file_subpart_mapping: Dict[str, List]) -> Set[str]:
    mapper = ChangedFileMapper(changed_files, file_subpart_mapping)
    return mapper.get_affected_subparts()


def run_detection(
    evidence_dir: str,
    before_dir: str,
    after_dir: str,
    fine_grained: bool = False,
    enable_llm_filter: bool = False,
    console: Any = None,
    repo_before_dir: str = None,
    repo_after_dir: str = None,
):
    extractor = DocpartFileExtractor(evidence_dir, fine_grained)
    if fine_grained:
        file_subpart_mapping = extractor.extract_fine_grained_mappings()
    else:
        file_subpart_mapping = extractor.extract_mappings()

    detector = FileChangeDetector(before_dir, after_dir, fine_grained, enable_llm_filter)
    changes, changed_detail = detector.detect_changes()
    changed_files = [c["file"] for c in changes]

    changed_lines = {}
    if fine_grained:
        for change in changes:
            if change["type"] == "modified" and "changed_lines" in change:
                changed_lines[change["file"]] = change["changed_lines"]

    mapper = ChangedFileMapper(changed_files, file_subpart_mapping, changed_lines)
    affected_subparts = mapper.get_affected_subparts()
    by_subpart = mapper.get_changed_files_by_subpart()
    file_to_subparts = mapper.map_changed_files_to_subparts()
    mapped_files = list(file_to_subparts.keys())
    unmatched_files = [f for f in changed_files if f not in file_to_subparts and not f.endswith(".DS_Store")]

    example_files = [f for f in changed_files if _is_example_file(f)]
    if example_files and "example" not in affected_subparts:
        affected_subparts.add("example")
        by_subpart["example"] = example_files
        for f in example_files:
            if f in file_to_subparts:
                if "example" not in file_to_subparts[f]:
                    file_to_subparts[f].append("example")
            else:
                file_to_subparts[f] = ["example"]
                mapped_files.append(f)
        unmatched_files = [f for f in unmatched_files if not _is_example_file(f)]
        info(f"检测到 {len(example_files)} 个 example 文件变更，已注入 example doc_part")

    # 相似文件名兜底匹配：对未匹配的文件，通过计算 stem 相似度匹配到相近的文档
    reverse_mapping_full: Dict[str, Set[Tuple[str, Any]]] = {}
    for subpart, files in file_subpart_mapping.items():
        for entry in files:
            if isinstance(entry, dict):
                fn = entry.get("file_name", "")
                lr = entry.get("line_range")
            else:
                fn, lr = entry, None
            reverse_mapping_full.setdefault(fn, set()).add((subpart, lr))

    similarly_matched: Dict[str, List[str]] = {}
    similar_matched_files: Set[str] = set()
    unmatched_still: List[str] = []

    for uf in unmatched_files:
        if uf.endswith(DOC_SUFFIXES):
            unmatched_still.append(uf)
            continue

        uf_stem = Path(uf).stem
        best_stem = None
        best_subparts = None
        best_score = 0.0

        for known_basename, known_subparts in reverse_mapping_full.items():
            score = _compute_similarity(uf_stem, Path(known_basename).stem)
            if score >= SIMILARITY_THRESHOLD and score > best_score:
                best_score = score
                best_stem = known_basename
                best_subparts = known_subparts

        if best_stem is not None:
            adopted = sorted(s for s, _ in best_subparts)
            similarly_matched[uf] = adopted
            similar_matched_files.add(uf)
            info(f"  [相似匹配] {Path(uf).name} → {best_stem} " f"(ratio={best_score:.2f}), 继承 doc_parts: {adopted}")
        else:
            unmatched_still.append(uf)

    for f, dps in similarly_matched.items():
        file_to_subparts[f] = dps
        mapped_files.append(f)
        affected_subparts.update(dps)
        for dp in dps:
            by_subpart.setdefault(dp, []).append(f)

    unmatched_files = unmatched_still

    repo_other_mapping = extractor.get_repo_other_mapping()
    repo_other_changed = []
    # Backward-compat: old snapshots have operator_storage/ in paths; new ones use real repo
    # paths (relative_to succeeds first)
    snapshot_storage_marker = "/operator_storage/"

    if repo_before_dir and repo_after_dir and repo_other_mapping:
        repo_detector = FileChangeDetector(repo_before_dir, repo_after_dir, fine_grained, enable_llm_filter)
        repo_changes, repo_changed_detail = repo_detector.detect_changes()
        repo_changed_files = set(c["file"] for c in repo_changes)

        repo_before_path = Path(repo_before_dir).resolve()
        for abs_path, doc_parts in repo_other_mapping.items():
            abs_path_obj = Path(abs_path).resolve()
            try:
                rel_path = str(abs_path_obj.relative_to(repo_before_path))
            except ValueError:
                if snapshot_storage_marker in abs_path:
                    rel_path = abs_path.split(snapshot_storage_marker, 1)[1]
                else:
                    rel_path = None

            if rel_path and rel_path in repo_changed_files:
                change_type = next(
                    (c["type"] for c in repo_changes if c["file"] == rel_path),
                    "unknown",
                )
                real_path = str(repo_before_path / rel_path)
                repo_other_changed.append((real_path, rel_path, change_type, doc_parts))
                for dp in doc_parts:
                    affected_subparts.add(dp)
                    if dp not in by_subpart:
                        by_subpart[dp] = []
                    by_subpart[dp].append(real_path)
                changed_detail[real_path] = repo_changed_detail.get(rel_path, {})

    repo_other_change_map = {item[0]: item[2] for item in repo_other_changed}

    if console:
        repo_before_path = Path(repo_before_dir).resolve() if repo_before_dir else None
        repo_other_changed_rel = {item[1] for item in repo_other_changed}

        def _rel_display(raw_path):
            p = Path(raw_path)
            if not p.is_absolute():
                return raw_path
            if repo_before_path:
                try:
                    return str(p.relative_to(repo_before_path))
                except ValueError:
                    pass
            return p.name

        def _get_change_type(f, rel_f):
            ct = next((c["type"] for c in changes if c["file"] == f), None)
            if ct is None:
                ct = next((c["type"] for c in changes if c["file"] == rel_f), None)
            if ct is None:
                ct = repo_other_change_map.get(f, "unknown")
            return ct

        def _get_line_info(f, rel_f):
            if not fine_grained:
                return "-"
            cl = changed_lines.get(f) or changed_lines.get(rel_f)
            if not cl:
                return "-"
            if len(cl) <= 3:
                return ", ".join(f"{s}-{e}" for s, e in cl)
            return f"{len(cl)} 处"

        op_files = set()
        for subpart in affected_subparts:
            for f in by_subpart.get(subpart, []):
                if f not in repo_other_change_map:
                    op_files.add(f)

        if op_files:
            op_table = Table(show_header=True, header_style="bold cyan", show_lines=True, expand=True)
            op_table.add_column("文件", style="dim", no_wrap=True)
            op_table.add_column("类型", justify="center", no_wrap=True)
            op_table.add_column("行号", justify="center", no_wrap=True)
            op_table.add_column("影响文档")
            for f in sorted(op_files):
                rel_f = _rel_display(f)
                ct = _get_change_type(f, rel_f)
                li = _get_line_info(f, rel_f)
                dps = sorted(dp for dp in affected_subparts if f in by_subpart.get(dp, []))
                dp_str = ", ".join(format_doc_part(dp) for dp in dps)
                if f in similar_matched_files:
                    dp_str = f"(相似匹配) {dp_str}"
                op_table.add_row(rel_f, ct, li, dp_str)
            console.print(
                Panel(
                    op_table,
                    title=f"[green]算子内变更 ({len(op_files)} 文件 → {len(affected_subparts)} doc part)[/green]",
                    border_style="green",
                )
            )

        if unmatched_files:
            un_table = Table(show_header=True, header_style="bold", show_lines=True, expand=True)
            un_table.add_column("文件", style="dim", no_wrap=True)
            un_table.add_column("类型", justify="center", no_wrap=True)
            un_table.add_column("行号", justify="center", no_wrap=True)
            for f in sorted(unmatched_files):
                rel_f = _rel_display(f)
                ct = _get_change_type(f, rel_f)
                li = _get_line_info(f, rel_f)
                un_table.add_row(rel_f, ct, li)
            console.print(
                Panel(
                    un_table,
                    title=f"[yellow]未匹配到文档的变更 ({len(unmatched_files)} 文件)[/yellow]",
                    border_style="yellow",
                )
            )

        if repo_other_mapping:
            dep_table = Table(show_header=True, header_style="bold", show_lines=True, expand=True)
            dep_table.add_column("状态", width=10, justify="center")
            dep_table.add_column("文件", style="dim")
            dep_table.add_column("关联文档")

            changed_entries, unchanged_entries = [], []
            for abs_path, dps in sorted(repo_other_mapping.items()):
                abs_path_obj = Path(abs_path).resolve()
                try:
                    rel = str(abs_path_obj.relative_to(repo_before_path)) if repo_before_path else abs_path
                except ValueError:
                    if snapshot_storage_marker in abs_path:
                        rel = abs_path.split(snapshot_storage_marker, 1)[1]
                    else:
                        rel = abs_path_obj.name
                is_changed = rel in repo_other_changed_rel
                entry = (rel, format_doc_parts(dps), is_changed)
                if is_changed:
                    changed_entries.append(entry)
                else:
                    unchanged_entries.append(entry)

            for rel, dp_str, _ in changed_entries:
                dep_table.add_row("[red]changed[/red]", rel, dp_str)
            for rel, dp_str, _ in unchanged_entries:
                dep_table.add_row("[green]unchanged[/green]", rel, dp_str)

            n_changed, n_unchanged = len(changed_entries), len(unchanged_entries)
            summary_bits = []
            if n_changed:
                summary_bits.append(f"[red]{n_changed} changed[/red]")
            if n_unchanged:
                summary_bits.append(f"[green]{n_unchanged} unchanged[/green]")
            console.print(
                Panel(
                    dep_table,
                    title=f"[dim]算子外公共依赖 ({', '.join(summary_bits)})[/dim]",
                    border_style="dim",
                )
            )
    repo_other_changed_files = [item[0] for item in repo_other_changed]
    return (
        list(affected_subparts),
        changed_detail,
        mapped_files,
        unmatched_files,
        repo_other_changed_files,
    )


if __name__ == "__main__":
    from rich.console import Console

    from src.generation_validation import config

    evidence_dir = config.DATA_DIR / "outputs/snapshot/add/base/anchor"
    before_dir = config.DATA_DIR / "outputs/snapshot/operator_storage/add/base"
    after_dir = config.DATA_DIR / "test/ops-math-master/math/add_modified"
    repo_before_dir = config.DATA_DIR / "test/ops-math-master"
    repo_after_dir = config.DATA_DIR / "test/ops-math-master_modified"

    (
        affected_subparts,
        changed_detail,
        mapped_files,
        unmatched_files,
        repo_other_changed_files,
    ) = run_detection(
        str(evidence_dir),
        str(before_dir),
        str(after_dir),
        fine_grained=True,
        console=Console(),
        repo_before_dir=repo_before_dir,
        repo_after_dir=repo_after_dir,
    )
    print(f"\naffected_subparts ({len(affected_subparts)}):")
    for s in affected_subparts:
        print(f"  - {s}")
    print(f"\nmapped_files ({len(mapped_files)}):")
    for f in mapped_files:
        print(f"  - {f}")
    print(f"\nunmatched_files ({len(unmatched_files)}):")
    for f in unmatched_files:
        print(f"  - {f}")
    print(f"\nrepo_other_changed_files ({len(repo_other_changed_files)}):")
    for f in repo_other_changed_files:
        print(f"  - {f}")
