# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Path configuration for anchor construction pipeline."""

import re
from pathlib import Path
from typing import Optional

from src.snapshot_construction.pub_ref import PubRefConfig


class PathConfig:
    """Centralized path configuration for anchor construction.

    Args:
        operator_path: Path to operator source directory
        out_path: Base output directory for all generated artifacts
        md_file: Optional markdown source file path (e.g., aclnnAddV3&aclnnInplaceAddV3.md).
            When set, md_stem derives a sanitized stem for output path construction.
        data_root: Optional shared directory for operator-level data
            (doc_json, code_map, tree_id, original_docs_dir, operator_note_dir).
            When set, these attributes resolve under data_root instead of out_path,
            enabling multiple per-md outputs to share a single data store.
            When None (default), all attributes derive from out_path (pre-refactor behavior).
    """

    # Subdirectory names (relative to out_path)
    DOC_JSON_DIR: str = "doc_json"
    CODE_MAP_DIR: str = "code_map"
    TREE_ID_DIR: str = "tree_id"
    ANCHOR_DIR: str = "anchor"
    ORIGINAL_DOCS_DIR: str = "organized_docs_with_descriptions/原始文档"
    OPERATOR_NOTE_DIR: str = "organized_docs_with_descriptions/算子说明"
    TEMP_DIR: str = "temp_func_build"

    # Fixed output filenames
    SUB_FUNCTION_HASH_FILE: str = "sub_function_hash.json"
    CONSTRAINT_FILE: str = "constraint.json"
    SUPPORT_FILE: str = "support.json"
    SUMMARY_FILE: str = "summary.json"
    FUNCTION_PROTO_FILE: str = "function_proto.json"

    _operator_path: Path
    _out_path: Path
    _md_file: Optional[str]
    _data_root: Optional[Path]

    def __init__(
        self,
        operator_path: str | Path,
        out_path: str | Path,
        md_file: Optional[str] = None,
        data_root: Optional[str | Path] = None,
    ):
        self._operator_path = Path(operator_path)
        self._out_path = Path(out_path)
        self._md_file = md_file
        self._data_root = Path(data_root) if data_root else None

    @property
    def _data_base(self) -> Path:
        """Base directory for operator-level data.

        Returns data_root when set, otherwise falls back to out_path
        (pre-refactor behavior).
        """
        return self._data_root if self._data_root is not None else self._out_path

    @property
    def operator_path(self) -> Path:
        """Operator source directory."""
        return self._operator_path

    @property
    def out_path(self) -> Path:
        """Base output directory."""
        return self._out_path

    @property
    def md_file(self) -> Optional[str]:
        """Markdown source file path (if set)."""
        return self._md_file

    @md_file.setter
    def md_file(self, value: Optional[str]) -> None:
        self._md_file = value

    @property
    def md_stem(self) -> Optional[str]:
        """Stem derived from md_file filename (original name preserved verbatim).

        No sanitization — preserves '&' (multi-API) and leading '[...]' prefix
        markers (e.g. [new], [deprecated]) so select_doc_file's
        f.stem == selected_md_stem match stays consistent with docs/ filenames.
        Examples:
            aclnnAddV3&aclnnInplaceAddV3.md -> aclnnAddV3&aclnnInplaceAddV3
            [new]aclnnAbs.md -> [new]aclnnAbs

        Returns None if md_file is not set.
        """
        if self._md_file is None:
            return None
        stem = Path(self._md_file).stem
        return stem

    @property
    def doc_json(self) -> Path:
        """Interface documentation JSON directory."""
        return self._data_base / self.DOC_JSON_DIR

    @property
    def code_map(self) -> Path:
        """Code map output directory."""
        return self._data_base / self.CODE_MAP_DIR

    @property
    def anchor_dir(self) -> Path:
        """Anchor output directory."""
        return self._out_path / self.ANCHOR_DIR

    @property
    def tree_id(self) -> Path:
        """Tree ID output directory."""
        return self._data_base / self.TREE_ID_DIR

    @property
    def original_docs_dir(self) -> Path:
        """Original documents directory (relative to tree_id)."""
        return self._data_base / self.TREE_ID_DIR / self.ORIGINAL_DOCS_DIR

    @property
    def operator_note_dir(self) -> Path:
        """Operator note directory (relative to tree_id)."""
        return self._data_base / self.TREE_ID_DIR / self.OPERATOR_NOTE_DIR

    @property
    def temp_dir(self) -> Path:
        """Temporary build directory."""
        return self._out_path / self.TEMP_DIR

    @staticmethod
    def get_doc_parts() -> list[str]:
        """Get document parts from PubRefConfig."""
        return PubRefConfig.DOC_PARTS
