# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Support anchor construction entry point.

Delegates to AnchorPipeline for unified anchor construction.
Accepts a PathConfig instance (required) instead of (operator_path, out_path).
"""

from typing import Optional

from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig


def extract_evidence(
    path_config: PathConfig,
    doc_part_control: Optional[list[str]] = None,
    source_files: Optional[list[str]] = None,
) -> list[str]:
    """Extract support evidence from operator code.

    Args:
        path_config: PathConfig instance for path resolution (required)
        doc_part_control: Optional list of doc parts (unused, for API compatibility)
        source_files: Optional list of source file paths/names to filter anchors.
            When None, all discovered files are processed (backward compatible).

    Returns:
        List of output file paths (backward compatibility with original signature)
    """
    pipeline = AnchorPipeline(
        path_config,
        doc_part_control=["support"],
        source_files=source_files,
    )
    pipeline.run()

    # Return output file path for backward compatibility
    # Original implementation returns list of file paths
    output_file = path_config.anchor_dir / PathConfig.SUPPORT_FILE
    return [str(output_file)]
