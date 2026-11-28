# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Validate that all 4 anchor_construction entry points are importable with correct signatures."""

import inspect

import pytest


def test_import_get_constraint_evidence():
    """get_constraint_evidence must be importable from constraint package."""
    from src.snapshot_construction.anchor_construction.constraint import (
        get_constraint_evidence,
    )

    assert callable(get_constraint_evidence)
    sig = inspect.signature(get_constraint_evidence)
    assert "path_config" in sig.parameters


def test_import_extract_evidence():
    """extract_evidence must be importable from support package."""
    from src.snapshot_construction.anchor_construction.support import extract_evidence

    assert callable(extract_evidence)
    sig = inspect.signature(extract_evidence)
    assert "path_config" in sig.parameters


def test_import_main_doc_process():
    """main must be importable from summary_function_proto package."""
    from src.snapshot_construction.anchor_construction.summary_function_proto import (
        main,
    )

    assert callable(main)
    sig = inspect.signature(main)
    assert "path_config" in sig.parameters


def test_import_run_pipe_line_single():
    """run_pipe_line_single must be importable from sub_function package."""
    from src.snapshot_construction.anchor_construction.sub_function import (
        run_pipe_line_single,
    )

    assert callable(run_pipe_line_single)
    sig = inspect.signature(run_pipe_line_single)
    assert "path_config" in sig.parameters
