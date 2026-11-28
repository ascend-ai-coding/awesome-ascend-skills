# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Sub_function anchor construction entry point.

Delegates to AnchorPipeline for unified anchor construction.
Accepts a PathConfig instance (required) instead of (operator_path, out_path).
"""

from typing import Optional

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline


def run_pipe_line_single(
    path_config: PathConfig,
    source_files: Optional[list[str]] = None,
) -> None:
    """Execute the full sub_function processing pipeline.

    Args:
        path_config: PathConfig instance for path resolution (required)
        source_files: Optional list of source file paths/names to filter anchors.
            When None, all discovered files are processed (backward compatible).

    Note:
        Original implementation returns None (writes files to disk).
        This wrapper maintains backward compatibility.
    """
    pipeline = AnchorPipeline(
        path_config,
        doc_part_control=["sub_function_hash"],
        source_files=source_files,
    )
    pipeline.run()
