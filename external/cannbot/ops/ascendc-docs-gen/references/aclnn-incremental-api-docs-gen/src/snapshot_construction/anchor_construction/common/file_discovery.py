# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Unified file discovery functions for anchor construction.

Consolidates all file search/discovery patterns from:
- extract_files.py: find_operator_files()
- extract_functions.py: find_aclnn_cpp_files()
- extract_notes.py: find_header_files()
- extract_section.py: find_aclnn_cpp_files(), find_def_h_file()
- extract_new_0331.py: find_all_aclnn_docs()
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def find_op_api_files(operator_path: Path) -> list[tuple[Path, str]]:
    """
    Find .h and .cpp files in op_api/ directory.

    Args:
        operator_path: Path to the operator directory

    Returns:
        List of tuples (file_path, source_pattern) where source_pattern describes
        the file pattern that matched (e.g., "aclnn_*.h", "aclnn_*.cpp")
    """
    files_with_source = []

    op_api_dir = operator_path / "op_api"
    if not op_api_dir.exists():
        op_api_dir = operator_path / "op_host" / "op_api"

    if op_api_dir.exists():
        for f in op_api_dir.glob("aclnn_*.h"):
            files_with_source.append((f, "aclnn_*.h"))
        for f in op_api_dir.glob("aclnn_*.cpp"):
            files_with_source.append((f, "aclnn_*.cpp"))

    logger.debug(f"Found {len(files_with_source)} op_api files in {operator_path}")
    return files_with_source


def find_operator_files(operator_path: Path) -> list[tuple[Path, str]]:
    """
    Find all operator source files in the operator directory.

    Uses a pattern list: each entry is (search_dir, glob_pattern, source_tag).
    rglob each pattern, collect results, dedup by Path.
    """
    op_api_dir = operator_path / "op_api"
    if not op_api_dir.exists():
        op_api_dir = operator_path / "op_host" / "op_api"
    op_host_dir = operator_path / "op_host"

    patterns = [
        (op_api_dir, "aclnn_*.h", "aclnn_*.h"),
        (op_api_dir, "aclnn_*.cpp", "aclnn_*.cpp"),
        (operator_path / "op_graph", "*_proto.h", "*_proto.h"),
        (op_host_dir, "*def*.cpp", "*def*.cpp"),
        (op_host_dir, "*def*.h", "*def*.h"),
        (op_host_dir, "*tiling*.cpp", "*tiling*.cpp"),
        (op_host_dir, "*tiling*.h", "*tiling*.h"),
        (operator_path / "op_tiling", "*.cpp", "op_tiling/*.cpp"),
        (operator_path / "op_tiling", "*.h", "op_tiling/*.h"),
        (op_host_dir / "op_tiling", "*.cpp", "op_tiling/*.cpp"),
        (op_host_dir / "op_tiling", "*.h", "op_tiling/*.h"),
        (operator_path / "op_kernel", "*.cpp", "*.cpp"),
        (operator_path / "op_kernel", "*.h", "*.h"),
    ]

    files_with_source = []
    for search_dir, pattern, tag in patterns:
        if search_dir.exists():
            for f in search_dir.rglob(pattern):
                files_with_source.append((f, tag))

    seen = set()
    unique_files = []
    for f, source in files_with_source:
        if f not in seen:
            seen.add(f)
            unique_files.append((f, source))

    logger.debug(f"Found {len(unique_files)} operator files in {operator_path}")
    return unique_files


def find_aclnn_cpp_files(operator_path: Path) -> list[Path]:
    """
    Find all aclnn_*.cpp files in operator directory recursively.

    Args:
        operator_path: Path to the operator directory

    Returns:
        List of paths to aclnn_*.cpp files
    """
    if not operator_path.exists():
        logger.warning(f"Path {operator_path} does not exist")
        return []

    cpp_files = []

    for root, dirs, files in os.walk(operator_path):
        for file in files:
            if file.startswith("aclnn_") and file.endswith(".cpp"):
                cpp_files.append(Path(root) / file)

    logger.debug(f"Found {len(cpp_files)} aclnn_*.cpp files in {operator_path}")
    return cpp_files


def find_header_files(operator_path: Path) -> list[Path]:
    """
    Find header files in the operator directory.

    Searches for .h files in op_api directories or files starting with 'aclnn_'.

    Args:
        operator_path: Path to the operator directory

    Returns:
        List of paths to header files
    """
    if not operator_path.exists():
        logger.warning(f"Path {operator_path} does not exist")
        return []

    header_files = []

    for root, _, files in os.walk(operator_path):
        for file in files:
            if file.endswith(".h"):
                if "op_api" in root or file.startswith("aclnn_"):
                    header_files.append(Path(root) / file)

    logger.debug(f"Found {len(header_files)} header files in {operator_path}")
    return header_files


def find_def_h_file(operator_path: Path) -> Optional[Path]:
    """
    Find def.h file in operator directory.

    Args:
        operator_path: Path to the operator directory

    Returns:
        Path to def.h file if found, None otherwise
    """
    if not operator_path.exists():
        logger.warning(f"Path {operator_path} does not exist")
        return None

    for root, dirs, files in os.walk(operator_path):
        if "def.h" in files:
            result = Path(root) / "def.h"
            logger.debug(f"Found def.h at {result}")
            return result

    logger.debug(f"No def.h found in {operator_path}")
    return None


def find_all_aclnn_docs(operator_path: Path) -> list[Path]:
    """
    Find all aclnn*.md documentation files in operator directory.

    Args:
        operator_path: Path to the operator directory

    Returns:
        List of paths to aclnn*.md files
    """
    if not operator_path.exists():
        logger.warning(f"Path {operator_path} does not exist")
        return []

    results = []

    for dirpath, _, filenames in os.walk(operator_path):
        for fname in filenames:
            if fname.startswith("aclnn") and fname.endswith(".md"):
                full_path = Path(dirpath) / fname
                results.append(full_path)
                logger.debug(f"Found doc file: {full_path}")

    if not results:
        logger.warning(f"No aclnn*.md files found in {operator_path}")

    return results
