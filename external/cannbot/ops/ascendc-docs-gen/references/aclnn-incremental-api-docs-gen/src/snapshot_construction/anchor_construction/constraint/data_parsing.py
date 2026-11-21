# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Constraint evidence extraction module."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.snapshot_construction.anchor_construction.common.anchor_builder import (
    build_file_anchor,
    build_section_anchor,
)
from src.snapshot_construction.anchor_construction.common.config import (
    ANCHOR_TYPE_FILE,
    ANCHOR_TYPE_SECTION,
)
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.pub_ref import PubRefConfig

logger = logging.getLogger(__name__)


def _get_code_block(lines: list[str], index: int) -> tuple[str, int, int]:
    """Extract code block around a line index.

    Args:
        lines: List of file lines
        index: Target line index (0-based)

    Returns:
        Tuple of (code block content, start_line, end_line) with 1-based line numbers
    """
    left, right = index - 1, index + 1

    while left >= 0 and lines[left].strip():
        left -= 1

    while right < len(lines) and lines[right].strip():
        right += 1

    start = left + 1
    block_lines = lines[start:right]
    content = "".join(block_lines)
    start_line = left + 2  # Convert 0-based slice start to 1-based line number
    end_line = right + 1  # Convert 0-based slice end to 1-based line number
    return content, start_line, end_line


def _get_ops_desc_map(data_loader: DataLoader) -> dict[str, str]:
    """Build mapping from file path to ops description.

    Args:
        data_loader: DataLoader instance

    Returns:
        Dict mapping file paths to their descriptions
    """
    result = {}

    if not data_loader._path_config.original_docs_dir.exists():
        logger.warning(f"Original docs directory not found: {data_loader._path_config.original_docs_dir}")
        return result

    for doc_file in data_loader._path_config.original_docs_dir.glob("*.md"):
        try:
            content = data_loader.read_file_content(doc_file)

            path_match = re.search(r"当前文件路径:(.*?)- 所属目录:", content, re.DOTALL)
            if not path_match:
                continue
            path = path_match.group(1).strip().strip("`")

            desc_match = re.search(r"#### 代码功能(.*?)#### 关键结构", content, re.DOTALL)
            if desc_match:
                result[path] = desc_match.group(1).strip()

        except Exception as e:
            logger.warning(f"Error processing {doc_file}: {e}")
            continue

    return result


def get_constraint_evidence(
    path_config: PathConfig,
    doc_part_control: Optional[list[str]] = None,
    source_files: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Extract constraint evidence from operator source code.

    Args:
        path_config: PathConfig instance for path resolution (required)
        doc_part_control: Optional list of doc parts to process
        source_files: Optional list of source file paths/names to filter processing.
            When provided, only files matching by filename or path suffix are processed.
            When None, all files are processed (backward compatible).

    Returns:
        Dict with constraint evidence data

    Raises:
        ValueError: If doc_part type is invalid
    """
    if not isinstance(path_config, PathConfig):
        raise TypeError(f"path_config must be a PathConfig instance, got {type(path_config).__name__}")
    if doc_part_control is not None:
        valid_parts = PathConfig.get_doc_parts()
        for part in doc_part_control:
            if part not in valid_parts:
                raise ValueError(f"doc_part 类型错误: '{part}' 不是有效的文档部分")

    data_loader = DataLoader(path_config)

    path_to_ops_desc = _get_ops_desc_map(data_loader)

    tree_desc = data_loader.load_tree_id()

    file_counter = 0
    section_counter = 0

    file_map: dict[str, Path] = {}
    first_md_file = None
    for md_file in path_config.operator_path.glob("**/*.md"):
        if md_file.name.startswith("acl"):
            file_map[md_file.name] = path_config.operator_path
            if first_md_file is None:
                first_md_file = md_file

    result: dict[str, Any] = {
        "doc": {
            "doc_path": str(first_md_file.parent) if first_md_file else "",
            "doc_name": first_md_file.name if first_md_file else "",
            "doc_part": "constraint",
            "md_format": [],
            "sub_part": {"constrain_clarify": [{"content": "", "evidence_id": []}]},
        },
        "anchor_tuple": [],
    }

    all_anchors: list[dict[str, Any]] = []

    source_filenames: set[str] = set()
    source_paths: set[str] = set()
    if source_files is not None:
        source_filenames = {Path(sf).name for sf in source_files}
        source_paths = {sf.replace("\\", "/") for sf in source_files}

    for name, path in file_map.items():
        result["anchor_tuple"] = []
        name_stem = name.replace(".md", "")

        doc_json_file = path_config.doc_json / f"{name_stem}.json"
        if doc_json_file.exists():
            try:
                with open(doc_json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result["doc"]["md_format"] = [data.get("constraint", {}).get("origin_txt", "")]
                    result["doc"]["sub_part"]["constrain_clarify"][0]["content"] = data.get("constraint", {}).get(
                        "constrain_clarify", ""
                    )
            except Exception as e:
                logger.warning(f"Error loading {doc_json_file}: {e}")

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            file_path_str = str(file_path).replace("\\", "/")
            filename = file_path.name

            if source_files is not None:
                if filename not in source_filenames and not any(file_path_str.endswith(p) for p in source_paths):
                    continue

            def create_file_anchor(fp: str, fn: str) -> dict[str, Any]:
                nonlocal file_counter
                file_counter += 1

                ops_desc = ""
                for key, value in path_to_ops_desc.items():
                    if key in fp:
                        ops_desc = value
                        break

                code_map = data_loader.load_code_map(fn)
                file_content = data_loader.read_file_content(Path(fp))

                anchor_id = f"anchor_file_{file_counter:03d}"
                evidence_id = f"ev_file_{file_counter:03d}"

                return build_file_anchor(
                    anchor_id=anchor_id,
                    file_path=fp,
                    file_name=fn,
                    code_map=code_map,
                    file_content=file_content,
                    file_func=ops_desc,
                    tree_id=tree_desc,
                    gene_deter="暂空",
                    gene_target=[],
                    evidence_id=evidence_id,
                    note="以下为file型标准结构",
                )

            def create_section_anchor(fp: str, code_block: str, start_line: int, end_line: int) -> dict[str, Any]:
                nonlocal section_counter
                section_counter += 1

                anchor_id = f"anchor_section_{section_counter:03d}"
                evidence_id = f"ev_section_{section_counter:03d}"

                return build_section_anchor(
                    anchor_id=anchor_id,
                    file_path=fp,
                    content=code_block,
                    start_line=start_line,
                    end_line=end_line,
                    token_hash=code_block,
                    gene_deter="暂空",
                    gene_target=[],
                    evidence_id=evidence_id,
                    note="以下为section型标准结构",
                )

            if re.search(r"op_api/aclnn_[^/]*?\.h", file_path_str):
                file_anchor = create_file_anchor(file_path_str, filename)
                file_anchor["evidence"][0]["gene_rule"]["target"] = [
                    ["deter_describe"],
                    ["constrain_model_scene"],
                    ["constrain_model_scene_clarify"],
                ]
                result["anchor_tuple"].append(file_anchor)

            if re.search(r"op_host/[^/]*?def\.cpp", file_path_str):
                file_anchor = create_file_anchor(file_path_str, filename)
                file_anchor["evidence"][0]["gene_rule"]["target"] = [
                    ["constrain_model_scene"],
                    ["constrain_model_scene_clarify"],
                ]
                result["anchor_tuple"].append(file_anchor)

            if re.search(r"op_host/[^/]*?tiling.*?\.cpp", file_path_str):
                file_anchor = create_file_anchor(file_path_str, filename)
                file_anchor["evidence"][0]["gene_rule"]["target"] = [
                    ["constrain_model_scene"],
                    ["constrain_model_scene_clarify"],
                ]
                result["anchor_tuple"].append(file_anchor)

                lines = data_loader.read_lines(Path(file_path_str))
                for index, line in enumerate(lines):
                    if "curShortSocName == platform_ascendc::SocVersion::" in line or re.search(
                        r".*?(switch|if).*?(SocVersion::|socVersion)", line
                    ):
                        code_block, s_line, e_line = _get_code_block(lines, index)
                        section_anchor = create_section_anchor(file_path_str, code_block, s_line, e_line)
                        section_anchor["evidence"][0]["gene_rule"]["target"] = [
                            ["constrain_model_scene"],
                            ["constrain_model_scene_clarify"],
                        ]
                        result["anchor_tuple"].append(section_anchor)

            if re.search(r"op_kernel/.+?\.(cpp|h)", file_path_str):
                file_anchor = create_file_anchor(file_path_str, filename)
                file_anchor["evidence"][0]["gene_rule"]["target"] = [
                    ["constrain_model_scene"],
                    ["constrain_model_scene_clarify"],
                ]
                result["anchor_tuple"].append(file_anchor)

                lines = data_loader.read_lines(Path(file_path_str))
                for index, line in enumerate(lines):
                    # __NPU_ARCH__ pattern
                    if re.search(r"#\s*if.*__NPU_ARCH__\s*==\s*\w+", line):
                        code_block, s_line, e_line = _get_code_block(lines, index)
                        section_anchor = create_section_anchor(file_path_str, code_block, s_line, e_line)
                        section_anchor["evidence"][0]["gene_rule"]["target"] = [
                            ["constrain_model_scene"],
                            ["constrain_model_scene_clarify"],
                        ]
                        result["anchor_tuple"].append(section_anchor)

                    # __CCE_AICORE__ pattern
                    if re.search(r"#\s*(if|elif).*__CCE_AICORE__\s*==\s*\w+", line):
                        code_block, s_line, e_line = _get_code_block(lines, index)
                        section_anchor = create_section_anchor(file_path_str, code_block, s_line, e_line)
                        section_anchor["evidence"][0]["gene_rule"]["target"] = [
                            ["constrain_model_scene"],
                            ["constrain_model_scene_clarify"],
                        ]
                        result["anchor_tuple"].append(section_anchor)

        all_anchors.extend(result["anchor_tuple"])

    result["anchor_tuple"] = all_anchors
    # fmt: off
    result["doc"]["sub_part"]["constrain_clarify"][0]["evidence_id"] = [
        x["evidence"][0]["evidence_id"]
        for x in all_anchors
    ]
    # fmt: on

    path_config.anchor_dir.mkdir(parents=True, exist_ok=True)
    output_file = path_config.anchor_dir / PathConfig.CONSTRAINT_FILE

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    logger.info(f"Constraint evidence written to: {output_file}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 4:
        operator_path = sys.argv[1]
        out_path = sys.argv[2]
        pc = PathConfig(operator_path, out_path)
        result = get_constraint_evidence(pc)
        print(f"Generated {len(result.get('anchor_tuple', []))} anchors")
    else:
        print("Usage: python data_parsing.py <operator_path> <out_path>")
