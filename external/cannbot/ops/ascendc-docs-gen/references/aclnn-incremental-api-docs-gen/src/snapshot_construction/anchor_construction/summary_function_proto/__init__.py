# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Summary and function_proto anchor construction entry point.

Delegates to AnchorPipeline for unified anchor construction.
Accepts a PathConfig instance (required) instead of (operator_path, out_path).
"""

from typing import Optional

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline


def main(
    path_config: PathConfig,
    doc_part_control: Optional[list[str]] = None,
    source_files: Optional[list[str]] = None,
) -> None:
    """Extract summary and function_proto evidence from operator code.

    Args:
        path_config: PathConfig instance for path resolution (required)
        doc_part_control: Optional list of doc parts to process.
            If None, processes both "summary" and "function_proto".
        source_files: Optional list of source file paths/names to filter anchors.
            When None, all discovered files are processed (backward compatible).

    Note:
        summary and function_proto share extraction but output different gene_rules.
        AnchorPipeline handles this coupling internally.
    """
    # Default to both summary and function_proto if not specified
    if doc_part_control is None:
        doc_part_control = ["summary", "function_proto"]

    pipeline = AnchorPipeline(
        path_config,
        doc_part_control=doc_part_control,
        source_files=source_files,
    )
    pipeline.run()
