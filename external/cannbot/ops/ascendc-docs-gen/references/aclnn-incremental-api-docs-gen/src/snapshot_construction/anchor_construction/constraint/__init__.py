# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Constraint anchor construction entry point.

Delegates to AnchorPipeline for unified anchor construction.
Accepts a PathConfig instance (required) instead of (operator_path, out_path).
"""

from typing import Any, Optional

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline


def get_constraint_evidence(
    path_config: PathConfig,
    doc_part_control: Optional[list[str]] = None,
    source_files: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Extract constraint evidence from operator source code.

    Args:
        path_config: PathConfig instance for path resolution (required)
        doc_part_control: Optional list of doc parts (unused, for API compatibility)
        source_files: Optional list of source file paths/names to filter anchors.
            When None, all discovered files are processed (backward compatible).

    Returns:
        Dict with constraint evidence data
    """
    pipeline = AnchorPipeline(
        path_config,
        doc_part_control=["constraint"],
        source_files=source_files,
    )
    results = pipeline.run()
    return results["constraint"]
