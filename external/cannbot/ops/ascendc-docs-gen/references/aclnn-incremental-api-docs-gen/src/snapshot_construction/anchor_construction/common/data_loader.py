# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Unified data loading for anchor construction pipeline."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.snapshot_construction.anchor_construction.common.config import (
    CODE_MAP_NODE_TYPES,
    CODE_MAP_EDGE_TYPES,
)
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data loader for anchor construction.

    Consolidates all data loading functions across anchor_construction module.
    All methods handle FileNotFoundError gracefully by returning empty values.
    """

    def __init__(self, path_config: PathConfig):
        """Initialize DataLoader with PathConfig instance.

        Args:
            path_config: PathConfig instance containing all path configurations
        """
        self._path_config = path_config

    def load_code_map(self, file_name: str) -> dict[str, Any]:
        """Load code map for a specific file.

        Args:
            file_name: Source file name (e.g., "aclnn_abs.h")

        Returns:
            Dict with 'nodes' (list of node IDs) and 'edges' (list of edge dicts).
            Returns {nodes: [], edges: []} if file not found.
        """
        map_file = self._path_config.code_map / "files" / f"{file_name}_map.json"

        if not map_file.exists():
            logger.warning(f"Code map file not found: {map_file}")
            return {"nodes": [], "edges": []}

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Filter nodes: case-insensitive type matching, ID-only format
            filtered_nodes = [
                node["id"]
                for node in data.get("nodes", [])
                if isinstance(node.get("type"), str)
                and node["type"].lower() in CODE_MAP_NODE_TYPES
                and "id" in node
            ]

            # Filter edges: case-insensitive type matching
            filtered_edges = [
                {
                    "source": edge.get("source", ""),
                    "type": edge.get("type", ""),
                    "target": edge.get("target", ""),
                }
                for edge in data.get("edges", [])
                if isinstance(edge.get("type"), str)
                and edge["type"].lower() in CODE_MAP_EDGE_TYPES
            ]

            logger.debug(
                f"Loaded code map: {map_file} -> nodes={len(filtered_nodes)}, edges={len(filtered_edges)}"
            )
            return {"nodes": filtered_nodes, "edges": filtered_edges}

        except Exception as e:
            logger.warning(f"Error loading code map {map_file}: {e}")
            return {"nodes": [], "edges": []}

    def load_functions_json(self) -> dict[str, Any]:
        """Load functions.json from code_map directory.

        Returns:
            Dict with function data. Returns {} if file not found.
        """
        func_file = self._path_config.code_map / "functions.json"

        if not func_file.exists():
            logger.warning(f"Functions JSON file not found: {func_file}")
            return {}

        try:
            with open(func_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded functions JSON: {func_file}")
            return data
        except Exception as e:
            logger.warning(f"Error loading functions JSON {func_file}: {e}")
            return {}

    def load_file_func(self, file_name: str) -> str:
        """Load file function description from original docs.

        Args:
            file_name: Source file name (e.g., "aclnn_abs.h")

        Returns:
            Extracted "代码功能" section content. Returns "" if not found.
        """
        # Extract basename without extension
        basename = Path(file_name).stem

        # Glob for matching files in original_docs_dir
        if not self._path_config.original_docs_dir.exists():
            logger.warning(
                f"Original docs directory not found: {self._path_config.original_docs_dir}"
            )
            return ""

        # Try to find file matching pattern: {basename}_desc.md or similar
        candidates = list(self._path_config.original_docs_dir.glob(f"*{basename}*.md"))

        if not candidates:
            logger.warning(f"No matching original doc found for: {file_name}")
            return ""

        # Use first match
        doc_file = candidates[0]

        try:
            content = self.read_file_content(doc_file)

            # Extract "#### 代码功能" section
            pattern = r"####\s*代码功能\s*\n(.*?)(?=\n####|\Z)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                return match.group(1).strip()

            logger.warning(f"No '代码功能' section found in: {doc_file}")
            return ""

        except Exception as e:
            logger.warning(f"Error loading file function from {doc_file}: {e}")
            return ""

    def load_tree_id(self) -> str:
        """Load operator note content from tree_id directory.

        Returns:
            Extracted "代码整体目录及说明" section content. Returns "" if not found.
        """
        if not self._path_config.operator_note_dir.exists():
            logger.warning(
                f"Operator note directory not found: {self._path_config.operator_note_dir}"
            )
            return ""

        # Glob for *算子说明.md files
        candidates = list(self._path_config.operator_note_dir.glob("*算子说明.md"))

        if not candidates:
            logger.warning(
                f"No operator note file found in: {self._path_config.operator_note_dir}"
            )
            return ""

        # Use first match
        note_file = candidates[0]

        try:
            content = self.read_file_content(note_file)

            # Extract "## 二、代码整体目录及说明" section
            pattern = r"## 二、代码整体目录及说明\s*\n(.*)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                return match.group(1).strip()

            logger.warning(f"No '代码整体目录及说明' section found in: {note_file}")
            return ""

        except Exception as e:
            logger.warning(f"Error loading tree_id from {note_file}: {e}")
            return ""

    def load_support_matrix(self) -> dict[str, str]:
        """Load support matrix from all doc_json files.

        Returns:
            Dict mapping doc_name to support_matrix content.
            Returns {} if directory not found.
        """
        if not self._path_config.doc_json.exists():
            logger.warning(
                f"Doc JSON directory not found: {self._path_config.doc_json}"
            )
            return {}

        mapping = {}

        for json_file in self._path_config.doc_json.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract support.support_matrix
                content = data.get("support", {}).get("support_matrix", "")
                doc_name = json_file.stem + ".md"  # Convert to .md name
                mapping[doc_name] = content

            except Exception as e:
                logger.warning(f"Error loading support matrix from {json_file}: {e}")
                continue

        logger.debug(f"Loaded support matrix for {len(mapping)} documents")
        return mapping

    def read_file_content(self, file_path: Path) -> str:
        """Read file content as string.

        Args:
            file_path: Path to file

        Returns:
            File content as string. Returns "" if file not found.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""

    def read_lines(self, file_path: Path) -> list[str]:
        """Read file lines as list.

        Args:
            file_path: Path to file

        Returns:
            List of lines. Returns [] if file not found.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.readlines()
        except Exception as e:
            logger.warning(f"Error reading lines from {file_path}: {e}")
            return []
