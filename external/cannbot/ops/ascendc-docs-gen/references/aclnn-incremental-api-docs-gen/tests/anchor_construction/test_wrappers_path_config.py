# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for 4 anchor_construction wrappers path_config injection (Task 4).

Verifies that the 4 entry-point wrappers (get_constraint_evidence,
extract_evidence, run_pipe_line_single, summary_function_proto.main)
accept a required PathConfig instance as the first positional arg and
forward it to AnchorPipeline, instead of the old (operator_path, out_path)
string pair.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.constraint import (
    get_constraint_evidence,
)
from src.snapshot_construction.anchor_construction.support import extract_evidence
from src.snapshot_construction.anchor_construction.sub_function import (
    run_pipe_line_single,
)
from src.snapshot_construction.anchor_construction.summary_function_proto import main

import src.snapshot_construction.anchor_construction.constraint as constraint_mod
import src.snapshot_construction.anchor_construction.support as support_mod
import src.snapshot_construction.anchor_construction.sub_function as sub_function_mod
import src.snapshot_construction.anchor_construction.summary_function_proto as sfp_mod


# (wrapper_fn, module_where_AnchorPipeline_lives, display_name)
WRAPPERS = [
    (get_constraint_evidence, constraint_mod, "get_constraint_evidence"),
    (extract_evidence, support_mod, "extract_evidence"),
    (run_pipe_line_single, sub_function_mod, "run_pipe_line_single"),
    (main, sfp_mod, "main"),
]


class TestWrappersForwardPathConfig:
    """Verify all 4 wrappers forward path_config to AnchorPipeline."""

    @pytest.mark.parametrize("fn,mod,name", WRAPPERS)
    def test_all_wrappers_forward_path_config(self, fn, mod, name, tmp_path: Path):
        """Each wrapper forwards the injected path_config to AnchorPipeline.

        Mocks AnchorPipeline in the wrapper's module, calls the wrapper
        with a PathConfig instance, and asserts the mock received the
        same instance (identity check) as the first positional arg.
        """
        op = tmp_path / "op"
        op.mkdir()
        out = tmp_path / "out"
        pc = PathConfig(op, out)

        with patch.object(mod, "AnchorPipeline") as mock_ap:
            fn(pc)
            mock_ap.assert_called_once()
            call_args = mock_ap.call_args
            assert call_args.args[0] is pc, (
                f"{name} did not forward path_config as first positional arg; "
                f"got {call_args.args[0]!r}"
            )

    def test_constraint_forwards_source_files(self, tmp_path: Path):
        """get_constraint_evidence forwards source_files kwarg to AnchorPipeline."""
        op = tmp_path / "op"
        op.mkdir()
        out = tmp_path / "out"
        pc = PathConfig(op, out)

        with patch.object(constraint_mod, "AnchorPipeline") as mock_ap:
            get_constraint_evidence(pc, source_files=["a.cpp"])
            mock_ap.assert_called_once()
            call_kwargs = mock_ap.call_args.kwargs
            assert call_kwargs.get("source_files") == ["a.cpp"], (
                f"constraint wrapper did not forward source_files; "
                f"got {call_kwargs.get('source_files')!r}"
            )


class TestRejectOldPositionalArgs:
    """Verify old (operator_path, out_path) positional call style is rejected."""

    @pytest.mark.parametrize("fn,mod,name", WRAPPERS)
    def test_reject_old_positional_args(self, fn, mod, name):
        """Calling with 2 positional string args (old style) must fail.

        Old signature was (operator_path, out_path, ...); new signature
        takes a single PathConfig as first positional arg. Calling with
        two strings must fail fast (TypeError or AttributeError when
        AnchorPipeline tries to access path_config.operator_path on a str).
        """
        with pytest.raises((TypeError, AttributeError)):
            fn("/tmp/op", "/tmp/out")
