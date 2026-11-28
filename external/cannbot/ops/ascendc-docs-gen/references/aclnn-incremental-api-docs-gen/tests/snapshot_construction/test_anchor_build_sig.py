# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for snapshot.load_snapshot signature adaptation (Task 7: snapshot-data-prep-dedup).

The load_snapshot implementation lives in src/snapshot_construction/snapshot.py
(the former copy in change_detection/anchor_build.py was removed — rebuild now
imports it from there). Verifies that the 4 builder calls inside load_snapshot
receive the internally constructed PathConfig instance (identity-preserving)
instead of the old positional (operator_path, out_path) pair. expand_file_deps
and the load_snapshot signature itself remain unchanged.
"""

from unittest.mock import patch

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.snapshot import load_snapshot

# =============================================================================
# Tests
# =============================================================================


class TestLoadSnapshotSignature:
    """load_snapshot must forward path_config (not operator_path/out_path)
    to the 4 builder calls.
    """

    @pytest.fixture
    def mocks(self):
        """Patch all external collaborators of load_snapshot.

        4 builders (asserted): run_pipe_line_single, get_constraint_evidence,
        summary_function_proto_main, extract_evidence.
        Plus: cann_doc_struct, single_code_map, run_pipeline (I/O no-ops),
        expand_file_deps, statistic_anchor_info.
        """
        with (
            patch("src.snapshot_construction.snapshot.cann_doc_struct") as cann,
            patch("src.snapshot_construction.snapshot.single_code_map") as scm,
            patch("src.snapshot_construction.snapshot.run_pipeline") as rp,
            patch("src.snapshot_construction.snapshot.run_pipe_line_single") as rpls,
            patch("src.snapshot_construction.snapshot.get_constraint_evidence") as gce,
            patch("src.snapshot_construction.snapshot.summary_function_proto_main") as sfp,
            patch("src.snapshot_construction.snapshot.extract_evidence") as ee,
            patch("src.snapshot_construction.snapshot.expand_file_deps") as efd,
            patch(
                "src.snapshot_construction.snapshot.statistic_anchor_info",
                return_value=[{"doc_part": "stub"}],
            ) as sai,
        ):
            yield {
                "cann": cann,
                "scm": scm,
                "rp": rp,
                "rpls": rpls,
                "gce": gce,
                "sfp": sfp,
                "ee": ee,
                "efd": efd,
                "sai": sai,
            }

    def test_builders_receive_same_path_config_instance(self, mocks):
        """All 4 builders must be called with the SAME PathConfig instance
        (constructed once at load_snapshot L64) as first positional arg.
        """
        result = load_snapshot(
            "/tmp/op",
            "/tmp/out",
            "/tmp/ops_root",
            doc_part_control=[
                "constraint",
                "sub_function_hash",
                "function_proto",
                "summary",
                "support",
            ],
        )

        # Each builder called once
        assert mocks["rpls"].called
        assert mocks["gce"].called
        assert mocks["sfp"].called
        assert mocks["ee"].called

        # First positional arg of each builder is a PathConfig instance
        rpls_pc = mocks["rpls"].call_args.args[0]
        gce_pc = mocks["gce"].call_args.args[0]
        sfp_pc = mocks["sfp"].call_args.args[0]
        ee_pc = mocks["ee"].call_args.args[0]

        assert isinstance(
            rpls_pc, PathConfig
        ), f"run_pipe_line_single got {type(rpls_pc).__name__}, expected PathConfig"
        assert isinstance(
            gce_pc, PathConfig
        ), f"get_constraint_evidence got {type(gce_pc).__name__}, expected PathConfig"
        assert isinstance(
            sfp_pc, PathConfig
        ), f"summary_function_proto_main got {type(sfp_pc).__name__}, expected PathConfig"
        assert isinstance(ee_pc, PathConfig), f"extract_evidence got {type(ee_pc).__name__}, expected PathConfig"

        # All 4 receive the SAME instance (identity check)
        assert rpls_pc is gce_pc
        assert gce_pc is sfp_pc
        assert sfp_pc is ee_pc

        # doc_part_control forwarded to the 3 builders that accept it
        ctrl = [
            "constraint",
            "sub_function_hash",
            "function_proto",
            "summary",
            "support",
        ]
        assert mocks["gce"].call_args.kwargs.get("doc_part_control") == ctrl
        assert mocks["sfp"].call_args.kwargs.get("doc_part_control") == ctrl
        assert mocks["ee"].call_args.kwargs.get("doc_part_control") == ctrl

        # Return value is statistic_anchor_info's return value
        assert result == [{"doc_part": "stub"}]
        mocks["sai"].assert_called_once()

    def test_expand_file_deps_unchanged(self, mocks):
        """expand_file_deps call signature must remain unchanged
        (still receives _ops_root, _op_name, anchor_dir str, data_loader).
        """
        load_snapshot(
            "/tmp/op",
            "/tmp/out",
            "/tmp/ops_root",
            doc_part_control=["support"],
        )

        assert mocks["efd"].called
        # 3 positional args + data_loader kwarg
        args = mocks["efd"].call_args.args
        assert len(args) == 3
        # data_loader keyword present
        assert "data_loader" in mocks["efd"].call_args.kwargs
