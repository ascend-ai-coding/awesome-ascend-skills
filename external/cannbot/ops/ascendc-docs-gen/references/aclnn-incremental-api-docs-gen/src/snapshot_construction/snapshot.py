# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console

from src.snapshot_construction import cann_doc_struct
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.constraint import (
    get_constraint_evidence,
)
from src.snapshot_construction.anchor_construction.summary_function_proto import (
    main as summary_function_proto_main,
)
from src.snapshot_construction.anchor_construction.support import extract_evidence
from src.snapshot_construction.build_tree_id import run_pipeline
from src.snapshot_construction.code_graph import single_code_map
from src.snapshot_construction.anchor_construction.sub_function import (
    run_pipe_line_single,
)
from src.snapshot_construction.anchor_construction.postprocess.expand_file_deps import (
    expand_file_deps,
)
from src.snapshot_construction.anchor_construction.postprocess.version_filter import (
    filter_version_mismatch_anchors,
)
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.pub_ref import PubRefConfig
from src.logger import info, debug, warning, error


def prepare_operator_data(
    operator_path: str,
    out_path: str,
    file_dir_list: Optional[List[str]] = None,
    old_tree_id_path: Optional[str] = None,
) -> None:
    """Prepare operator-level data (doc_json, code_map, tree_id).

    Extracted from load_snapshot to enable dedup: call once, then call
    load_snapshot(skip_data_prep=True) multiple times for per-md outputs.

    Args:
        operator_path: Path to operator source directory
        out_path: Base output directory for all generated artifacts
        file_dir_list: Optional file filter for incremental tree_id generation
        old_tree_id_path: Optional old tree_id path for incremental generation
    """
    console = Console()
    path_config = PathConfig(operator_path, out_path)

    # infe_3 接口文档提取 具体实现在 cann_doc_struct.py 中
    cann_doc_struct.main(str(path_config.operator_path), str(path_config.doc_json))

    # infe_4 code_map 提取
    console.print("      > 步骤 1: 建立代码函数级关联关系")
    single_code_map.main([str(path_config.operator_path)], str(path_config.code_map))

    # infe_5 算子 id 生成
    console.print("      > 步骤 2: 构建算子功能与结构画像")
    run_pipeline.main(
        path_config.operator_path,
        path_config.tree_id,
        file_filter=file_dir_list,
        old_tree_id_path=Path(old_tree_id_path) if old_tree_id_path else None,
    )


def load_snapshot(
    operator_path: str,
    out_path: str,
    ops_root: str,
    doc_part_control: Optional[List[str]] = None,
    file_dir_list: Optional[List[str]] = None,
    old_tree_id_path: Optional[str] = None,
    md_file: Optional[str] = None,
    source_files: Optional[list[str]] = None,
    *,
    data_root: Optional[str] = None,
    skip_data_prep: bool = False,
) -> List[Dict[str, Any]]:
    """Build snapshot for operator.

    Args:
        operator_path: Path to operator source directory
        out_path: Base output directory for all generated artifacts
        ops_root: Real repo root directory (e.g., the checkout root of the ops repo)
        doc_part_control: List of doc parts to build. If None, builds all 5 parts.
        file_dir_list: Optional file filter for incremental tree_id generation
        old_tree_id_path: Optional old tree_id path for incremental generation
        md_file: Optional markdown source file path for md_stem derivation.
            When None, scans full operator_path (backward compatible).
        source_files: Optional list of source files to filter anchors.
            When None, builders scan full operator_path (backward compatible).
            Filter logic implemented in Task 6/8.
        data_root: Optional shared directory for operator-level data
            (doc_json, code_map, tree_id). When set, PathConfig resolves these
            under data_root for reading, enabling dedup with prepare_operator_data.
            When None (default), all attributes derive from out_path.
        skip_data_prep: When True, skip operator data preparation (cann_doc_struct,
            single_code_map, run_pipeline). Use with prepare_operator_data for dedup.
            Default False (backward compatible — prep runs inside load_snapshot).

    Returns:
        List of anchor statistics per doc_part
    """
    console = Console()
    pc = PathConfig(operator_path, out_path, md_file=md_file, data_root=data_root)

    # Default to all 5 doc_parts if not specified
    if doc_part_control is None:
        doc_part_control = list(PubRefConfig.DOC_PARTS)

    # Operator data preparation (extracted to prepare_operator_data for dedup)
    if not skip_data_prep:
        prepare_operator_data(operator_path, out_path, file_dir_list, old_tree_id_path)

    # infe_6 anchor 构建
    console.print("      > 步骤 3: 提取文档模块对应代码")

    if "sub_function_hash" in doc_part_control:
        info("sub_function_hash")
        run_pipe_line_single(pc, source_files=source_files)

    if "constraint" in doc_part_control:
        info("constraint")
        get_constraint_evidence(
            pc, doc_part_control=doc_part_control, source_files=source_files
        )

    if "function_proto" in doc_part_control or "summary" in doc_part_control:
        info("function_proto")
        debug(str(pc.operator_path))
        debug(str(pc.out_path))
        debug(str(pc.anchor_dir))
        summary_function_proto_main(
            pc, doc_part_control=doc_part_control, source_files=source_files
        )

    if "support" in doc_part_control:
        info("support")
        extract_evidence(
            pc, doc_part_control=doc_part_control, source_files=source_files
        )

    # infe_7 file anchor 依赖拓展（传递闭包展开 ①② 依赖文件为新 anchor）
    console.print("      > 步骤 4: 抽取代码文件级依赖关系")
    _data_loader = DataLoader(pc)
    expand_file_deps(
        ops_root, operator_path, str(pc.anchor_dir), data_loader=_data_loader
    )

    if pc.md_stem:
        console.print("      > 步骤 5: 过滤版本不匹配的 anchor")
        _removed = filter_version_mismatch_anchors(str(pc.anchor_dir), pc.md_stem)
        if _removed:
            info(f"version_filter: removed {_removed} anchors (md_stem={pc.md_stem})")

    return statistic_anchor_info(str(pc.anchor_dir))


def statistic_anchor_info(anchor_path: str) -> List[Dict[str, Any]]:
    """Collect anchor statistics from output directory.

    Args:
        anchor_path: Path to anchor output directory

    Returns:
        List of statistics per doc_part
    """
    anchor_suffix_map = {
        PathConfig.SUB_FUNCTION_HASH_FILE: "sub_function_hash",
        PathConfig.CONSTRAINT_FILE: "constraint",
        PathConfig.SUPPORT_FILE: "support",
        PathConfig.SUMMARY_FILE: "summary",
        PathConfig.FUNCTION_PROTO_FILE: "function_proto",
    }

    statistic_info = []
    for suffix, part in anchor_suffix_map.items():
        json_files = [f for f in os.listdir(anchor_path) if f.endswith(suffix)]
        if len(json_files) != 1:
            info(f"anchor输出目录异常，异常对象： anchor_file:{json_files}")
            continue
        _info = {
            "doc_part": part,
            "file": [],
            "function": [],
            "section": [],
            "note": [],
        }
        with open(os.path.join(anchor_path, json_files[0]), "r", encoding="utf-8") as f:
            anchor = json.load(f)

        for anchor_tuple_item in anchor["anchor_tuple"]:
            _info[anchor_tuple_item["anchor_information"]["type"]].append(
                anchor_tuple_item["anchor_information"]["level1"]["file_path"]
            )
        statistic_info.append(_info)
    return statistic_info


if __name__ == "__main__":
    operator_path = "../../data/ops-math/math/abs"
    data_path = "../../out"
    ops_root = "../../data/ops-math"

    load_snapshot(operator_path, data_path, ops_root)
