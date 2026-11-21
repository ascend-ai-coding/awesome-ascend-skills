# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for AnchorPipeline path_config injection (Task 3: snapshot-data-prep-dedup).

Verifies that AnchorPipeline receives a required PathConfig instance
(removing internal PathConfig construction from operator_path/out_path).
"""

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline

# =============================================================================
# Tests
# =============================================================================


class TestPathConfigInjection:
    """AnchorPipeline must accept a required PathConfig instance."""

    def test_path_config_injected(self):
        """PathConfig passed to AnchorPipeline is stored as-is and propagated
        to operator_path, out_path, and DataLoader.
        """
        pc = PathConfig("/tmp/op", "/tmp/out", data_root="/tmp/shared")
        pipeline = AnchorPipeline(pc, doc_part_control=["constraint"])

        # path_config stored by identity (same object, no internal reconstruction)
        assert pipeline.path_config is pc
        # operator_path / out_path derived from path_config
        assert pipeline.operator_path == pc.operator_path
        assert pipeline.out_path == pc.out_path
        # DataLoader receives the same path_config instance
        assert pipeline.data_loader._path_config is pc

    def test_rejects_positional_operator_path(self):
        """Passing positional operator_path/out_path (old signature) must
        fail — path_config is now the first required positional argument.
        """
        with pytest.raises((TypeError, AttributeError)):
            AnchorPipeline("/tmp/op", "/tmp/out")
