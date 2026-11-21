# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for get_constraint_evidence path_config injection (Task 5).

Verifies that get_constraint_evidence accepts a required PathConfig instance
(injected dependency) instead of constructing one internally from
(operator_path, out_path) string args.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.constraint.data_parsing import (
    get_constraint_evidence,
)


class TestPathConfigInjection:
    """Verify path_config is injected, not constructed internally."""

    def test_path_config_injected(self, tmp_path: Path):
        """Injected PathConfig is used; PathConfig.__init__ not re-invoked.

        Constructs a PathConfig with data_root, calls get_constraint_evidence
        with the instance, and asserts the constructor is not called again
        inside the function (proving injection replaces internal construction).
        """
        op = tmp_path / "op"
        op.mkdir()
        out = tmp_path / "out"
        shared = tmp_path / "shared"
        pc = PathConfig(op, out, data_root=shared)

        # Wrap PathConfig.__init__ to detect any internal construction.
        # pc is already built before entering the patch context, so the spy
        # only fires if get_constraint_evidence constructs a new PathConfig.
        with patch.object(
            PathConfig, "__init__", wraps=PathConfig.__init__
        ) as init_spy:
            result = get_constraint_evidence(pc, doc_part_control=["constraint"])
            init_spy.assert_not_called()

        assert isinstance(result, dict)
        # Output file should land under the injected pc's anchor_dir
        assert (pc.anchor_dir / PathConfig.CONSTRAINT_FILE).exists()

    def test_source_files_forwarded(self, tmp_path: Path):
        """source_files kwarg is accepted without TypeError."""
        op = tmp_path / "op"
        op.mkdir()
        out = tmp_path / "out"
        pc = PathConfig(op, out)

        # Should not raise TypeError when source_files is provided
        result = get_constraint_evidence(pc, source_files=["a.cpp"])
        assert isinstance(result, dict)

    def test_rejects_positional_operator_path(self):
        """Passing two positional string args (old call style) raises TypeError.

        Old signature was (operator_path, out_path, ...); new signature takes
        a single PathConfig as the first positional arg. Calling with two
        strings must fail fast with TypeError.
        """
        with pytest.raises(TypeError):
            get_constraint_evidence("/tmp/op", "/tmp/out")
