# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
import re
import hashlib
import difflib
from pathlib import Path
from typing import List, Tuple, Dict, Set, Any, Optional

from src.logger import info, debug, warning, error


class FileChangeDetector:
    def __init__(
        self,
        before_dir: str,
        after_dir: str,
        fine_grained: bool = False,
        enable_llm_filter: bool = False,
    ):
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)
        self.fine_grained = fine_grained
        self.enable_llm_filter = enable_llm_filter
        self._file_hashes: Dict[str, Tuple[str, str]] = {}

    def _calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5(usedforsecurity=False)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_all_files(self, base_dir: Path) -> Dict[str, Tuple[Path, str]]:
        files = {}
        if not base_dir.exists():
            return files
        for root, _, filenames in os.walk(base_dir):
            for filename in filenames:
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(base_dir)
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    files[str(rel_path)] = (file_path, file_hash)
                except (IOError, OSError):
                    files[str(rel_path)] = (file_path, "")
        return files

    def _extract_changed_lines(
        self, before_path: Path, after_path: Path
    ) -> List[Tuple[int, int]]:
        try:
            with open(before_path, "r", encoding="utf-8", errors="ignore") as f:
                before_lines = f.readlines()
            with open(after_path, "r", encoding="utf-8", errors="ignore") as f:
                after_lines = f.readlines()
        except (IOError, OSError):
            return []

        diff = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=str(before_path),
                tofile=str(after_path),
                lineterm="",
            )
        )

        changed_lines = []
        before_line_num = 0
        after_line_num = 0

        for line in diff:
            if line.startswith("@@"):
                match = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    before_line_num = int(match.group(1))
                    after_line_num = int(match.group(2))
            elif line.startswith("-") and not line.startswith("---"):
                changed_lines.append(before_line_num)
                before_line_num += 1
            elif line.startswith("+") and not line.startswith("+++"):
                changed_lines.append(after_line_num)
                after_line_num += 1
            elif not line.startswith("\\"):
                before_line_num += 1
                after_line_num += 1

        if not changed_lines:
            return []

        changed_lines.sort()
        ranges = []
        range_start = changed_lines[0]
        range_end = changed_lines[0]

        for line_num in changed_lines[1:]:
            if line_num == range_end + 1:
                range_end = line_num
            else:
                ranges.append((range_start, range_end))
                range_start = line_num
                range_end = line_num
        ranges.append((range_start, range_end))

        return ranges

    def _generate_diff_text(self, before_path: Path, after_path: Path) -> str:
        try:
            with open(before_path, "r", encoding="utf-8", errors="ignore") as f:
                before_lines = f.readlines()
            with open(after_path, "r", encoding="utf-8", errors="ignore") as f:
                after_lines = f.readlines()
        except (IOError, OSError):
            return ""

        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"{'before' if before_path else 'original'}",
                tofile=f"{'after' if after_path else 'modified'}",
                lineterm="\n",
            )
        )
        return "".join(diff_lines)

    def detect_changes(
        self, enable_llm_filter: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
        do_changed_detail = enable_llm_filter or self.enable_llm_filter
        before_files = self._get_all_files(self.before_dir)
        after_files = self._get_all_files(self.after_dir)

        changes = []
        changed_detail: Dict[str, Dict] = {}

        for file_path, (full_path, file_hash) in after_files.items():
            if file_path not in before_files:
                change_entry = {
                    "file": file_path,
                    "type": "added",
                    "full_path": str(full_path),
                }
                if do_changed_detail:
                    diff_text = self._generate_diff_text(Path(), full_path)
                    change_entry["changed_lines"] = []
                    changed_detail[file_path] = {
                        "type": "added",
                        "diff": diff_text,
                    }
                if self.fine_grained:
                    change_entry["changed_lines"] = []
                changes.append(change_entry)
            else:
                _, before_hash = before_files[file_path]
                if before_hash != file_hash:
                    change_entry = {
                        "file": file_path,
                        "type": "modified",
                        "full_path": str(full_path),
                    }
                    before_path = before_files[file_path][0]
                    after_path = full_path
                    if do_changed_detail:
                        diff_text = self._generate_diff_text(before_path, after_path)
                        changed_detail[file_path] = {
                            "type": "modified",
                            "diff": diff_text,
                        }
                    if self.fine_grained:
                        changed_lines = self._extract_changed_lines(
                            before_path, after_path
                        )
                        change_entry["changed_lines"] = changed_lines
                    changes.append(change_entry)

        for file_path, (full_path, _) in before_files.items():
            if file_path not in after_files:
                change_entry = {
                    "file": file_path,
                    "type": "deleted",
                    "full_path": str(full_path),
                }
                if do_changed_detail:
                    diff_text = self._generate_diff_text(full_path, Path())
                    change_entry["changed_lines"] = []
                    changed_detail[file_path] = {
                        "type": "deleted",
                        "diff": diff_text,
                    }
                if self.fine_grained:
                    change_entry["changed_lines"] = []
                changes.append(change_entry)

        return changes, changed_detail

    def get_changed_files(self) -> List[str]:
        changes, _ = self.detect_changes()
        return [change["file"] for change in changes]


def detect_file_changes(
    before_dir: str,
    after_dir: str,
    fine_grained: bool = False,
    enable_llm_filter: bool = False,
):
    detector = FileChangeDetector(
        before_dir, after_dir, fine_grained, enable_llm_filter
    )
    result = detector.get_changed_files()
    if enable_llm_filter:
        changes, changed_detail = detector.detect_changes(enable_llm_filter=True)
        return result, changed_detail
    return result


if __name__ == "__main__":
    from src.generation_validation import config

    before_dir = str(config.DATA_DIR / "abs")
    after_dir = str(config.DATA_DIR / "abs_modified")

    detector = FileChangeDetector(
        before_dir, after_dir, fine_grained=True, enable_llm_filter=True
    )
    changes, changed_detail = detector.detect_changes()

    info("变更检测结果:")
    info("-" * 50)
    for change in changes:
        info(f"[{change['type']}] {change['file']}")
        if "changed_lines" in change and change["changed_lines"]:
            for start, end in change["changed_lines"]:
                info(f"  行 {start}-{end} 变更")
    info("-" * 50)
    info(f"总计: {len(changes)} 个文件变更")
    info("\n详细变更内容 (changed_detail):")
    for fname, detail in changed_detail.items():
        info(f"\n  [{detail['type']}] {fname}")
        info(detail["diff"][:500] if len(detail["diff"]) > 500 else detail["diff"])
