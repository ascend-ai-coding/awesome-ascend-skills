# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

from src.logger import info, debug, warning, error


class FileEvidence:
    def __init__(self, file_name: str, line_range: Optional[Tuple[int, int]] = None):
        self.file_name = file_name
        self.line_range = line_range

    def __hash__(self):
        return hash((self.file_name, self.line_range))

    def __eq__(self, other):
        if not isinstance(other, FileEvidence):
            return False
        return self.file_name == other.file_name and self.line_range == other.line_range


class DocpartFileExtractor:
    def __init__(self, evidence_dir: str, fine_grained: bool = True):
        self.evidence_dir = Path(evidence_dir)
        self.fine_grained = fine_grained
        self._repo_other_mapping: Dict[str, Set[str]] = {}

    @staticmethod
    def _get_filename_from_path(file_path: str) -> str:
        return os.path.basename(file_path).replace("\\", "/").split("/")[-1]

    @staticmethod
    def _get_evidence_type(anchor: Dict) -> str:
        return anchor.get("anchor_information", {}).get("type", "unknown")

    def extract_mappings(self) -> Dict[str, List[str]]:
        mapping = {}
        self._repo_other_mapping = {}

        if not self.evidence_dir.exists():
            return mapping

        for file_path in self.evidence_dir.glob("*.json"):
            if file_path.name.startswith("."):
                continue
            part_mapping = self._parse_evidence_file(file_path)
            for doc_part, evidence_set in part_mapping.items():
                if doc_part not in mapping:
                    mapping[doc_part] = []
                for ev in sorted(evidence_set, key=lambda x: x.file_name):
                    mapping[doc_part].append(ev.file_name)

        return mapping

    def extract_fine_grained_mappings(self) -> Dict[str, List[Dict]]:
        mapping = {}
        self._repo_other_mapping = {}

        if not self.evidence_dir.exists():
            return mapping

        for file_path in self.evidence_dir.glob("*.json"):
            if file_path.name.startswith("."):
                continue
            part_mapping = self._parse_evidence_file(file_path)
            for doc_part, evidence_set in part_mapping.items():
                if doc_part not in mapping:
                    mapping[doc_part] = []
                for ev in sorted(
                    evidence_set, key=lambda x: (x.file_name, x.line_range or (0, 0))
                ):
                    entry = {"file_name": ev.file_name}
                    if ev.line_range is not None:
                        entry["line_range"] = ev.line_range
                    else:
                        entry["line_range"] = None
                    mapping[doc_part].append(entry)

        return mapping

    def get_subpart_file_table(self) -> Dict[str, List[str]]:
        return self.extract_mappings()

    def get_repo_other_mapping(self) -> Dict[str, List[str]]:
        return {
            path: sorted(doc_parts)
            for path, doc_parts in self._repo_other_mapping.items()
        }

    def _extract_evidence_from_anchor(
        self, anchor: Dict
    ) -> Tuple[str, Optional[Tuple[int, int]]]:
        note_type = self._get_evidence_type(anchor)
        evidence_list = anchor.get("evidence", [])

        for evidence_item in evidence_list:
            if "evidence_information" not in evidence_item:
                continue
            evidence_info = evidence_item["evidence_information"]
            if "file_path" not in evidence_info:
                continue

            file_name = self._get_filename_from_path(evidence_info["file_path"])

            if note_type in ("file", "function"):
                return file_name, None
            elif note_type in ("note", "section"):
                start_line = evidence_info.get("start_line")
                end_line = evidence_info.get("end_line")
                if start_line is not None and end_line is not None:
                    try:
                        return file_name, (int(start_line), int(end_line))
                    except (ValueError, TypeError):
                        pass

        if evidence_list:
            evidence_info = evidence_list[0].get("evidence_information", {})
            if "file_path" in evidence_info:
                file_name = self._get_filename_from_path(evidence_info["file_path"])
                return file_name, None

        return "", None

    def _parse_evidence_file(self, file_path: Path) -> Dict[str, Set[FileEvidence]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

        doc_info = data.get("doc", {})
        doc_part = doc_info.get("doc_part", "unknown")

        anchor_tuples = data.get("anchor_tuple", [])
        evidence_set = set()

        for anchor in anchor_tuples:
            file_name, line_range = self._extract_evidence_from_anchor(anchor)
            if file_name:
                evidence_set.add(FileEvidence(file_name, line_range))

            anchor_info = anchor.get("anchor_information", {})
            if anchor_info.get("dep_class") == "repo_other":
                for ev in anchor.get("evidence", []):
                    ev_info = ev.get("evidence_information", {})
                    repo_file_path = ev_info.get("file_path", "")
                    if repo_file_path:
                        if repo_file_path not in self._repo_other_mapping:
                            self._repo_other_mapping[repo_file_path] = set()
                        self._repo_other_mapping[repo_file_path].add(doc_part)

        return {doc_part: evidence_set}


def extract_evidence_mappings(evidence_dir: str, fine_grained: bool = False) -> Dict:
    extractor = DocpartFileExtractor(evidence_dir, fine_grained)
    if fine_grained:
        return extractor.extract_fine_grained_mappings()
    return extractor.extract_mappings()


if __name__ == "__main__":
    from src.generation_validation import config

    extractor = DocpartFileExtractor(
        evidence_dir="./scripts/cann-document-generation/data/outputs/snapshot/abs/base/anchor",
        fine_grained=True,
    )

    if extractor.fine_grained:
        mappings = extractor.extract_fine_grained_mappings()
        print("Subpart 与文件对应关系表 (细粒度):")
        print("-" * 50)
        for subpart, files in sorted(mappings.items()):
            print(f"\n[{subpart}]")
            for file_info in files:
                if file_info["line_range"]:
                    print(
                        f"  - {file_info['file_name']} (行 {file_info['line_range'][0]}-{file_info['line_range'][1]})"
                    )
                else:
                    print(f"  - {file_info['file_name']} (全文)")
        todo_list = [subpart for subpart, files in mappings.items() if files]
        print(todo_list)
    else:
        mappings = extractor.extract_mappings()
        print("Subpart 与文件对应关系表:")
        print("-" * 50)
        for subpart, files in sorted(mappings.items()):
            print(f"\n[{subpart}]")
            for file in files:
                print(f"  - {file}")
    print("-" * 50)
