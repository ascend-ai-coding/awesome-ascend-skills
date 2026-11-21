# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for snapshot_construction.snapshot module — data prep dedup (Task 6).

Tests cover:
- prepare_operator_data: standalone data preparation (cann_doc_struct,
  single_code_map, run_pipeline) extracted from load_snapshot.
- load_snapshot(skip_data_prep=True): skips data prep, enabling dedup
  (call prepare_operator_data once, then load_snapshot multiple times).
- load_snapshot(data_root=X): forwards data_root to PathConfig for
  shared operator-level data reading.
- load_snapshot: 4 builders receive the same path_config instance.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.snapshot_construction.snapshot import load_snapshot, prepare_operator_data


@pytest.fixture
def mock_all():
    """Patch all load_snapshot + prepare_operator_data dependencies."""
    with (
        patch("src.snapshot_construction.snapshot.cann_doc_struct") as cann_doc,
        patch("src.snapshot_construction.snapshot.single_code_map") as code_map,
        patch("src.snapshot_construction.snapshot.run_pipeline") as run_pipeline,
        patch(
            "src.snapshot_construction.snapshot.run_pipe_line_single"
        ) as run_pipe_single,
        patch(
            "src.snapshot_construction.snapshot.get_constraint_evidence"
        ) as get_constraint,
        patch(
            "src.snapshot_construction.snapshot.summary_function_proto_main"
        ) as summary_fp,
        patch("src.snapshot_construction.snapshot.extract_evidence") as extract_ev,
        patch("src.snapshot_construction.snapshot.expand_file_deps") as expand_deps,
        patch("src.snapshot_construction.snapshot.DataLoader") as data_loader,
        patch("src.snapshot_construction.snapshot.statistic_anchor_info") as stat_info,
    ):
        stat_info.return_value = []
        yield {
            "cann_doc_struct": cann_doc,
            "single_code_map": code_map,
            "run_pipeline": run_pipeline,
            "run_pipe_line_single": run_pipe_single,
            "get_constraint_evidence": get_constraint,
            "summary_function_proto_main": summary_fp,
            "extract_evidence": extract_ev,
            "expand_file_deps": expand_deps,
            "DataLoader": data_loader,
            "statistic_anchor_info": stat_info,
        }


class TestPrepareOperatorDataDedup:
    """Tests for prepare_operator_data + skip_data_prep dedup behavior."""

    def test_prep_runs_once_under_skip(self, mock_all, tmp_path):
        """多次 load_snapshot(skip_data_prep=True) 时 prep 只跑一次（核心回归）。

        Call prepare_operator_data once explicitly, then load_snapshot
        3 times with skip_data_prep=True. The three prep functions
        (cann_doc_struct.main, single_code_map.main, run_pipeline.main)
        should each have call_count==1 regardless of how many times
        load_snapshot is invoked.
        """
        op = str(tmp_path / "op")
        out = str(tmp_path / "out")
        out_md = str(tmp_path / "out_md")
        ops = str(tmp_path / "ops")

        # Call prepare_operator_data once explicitly
        prepare_operator_data(op, out)

        # Call load_snapshot 3 times with skip_data_prep=True
        for _ in range(3):
            load_snapshot(op, out_md, ops, skip_data_prep=True, doc_part_control=[])

        # Assert prep functions called exactly once (from prepare_operator_data,
        # not from load_snapshot which had skip_data_prep=True)
        assert mock_all["cann_doc_struct"].main.call_count == 1
        assert mock_all["single_code_map"].main.call_count == 1
        assert mock_all["run_pipeline"].main.call_count == 1

    def test_default_calls_prep(self, mock_all, tmp_path):
        """load_snapshot 默认行为（skip_data_prep=False）仍调 prep（向后兼容）。

        When skip_data_prep is not passed (default False), load_snapshot
        should call prepare_operator_data internally, invoking all three
        prep functions once.
        """
        op = str(tmp_path / "op")
        out = str(tmp_path / "out")
        ops = str(tmp_path / "ops")

        # Don't pass skip_data_prep — default is False
        load_snapshot(op, out, ops, doc_part_control=[])

        # Prep functions should be called once (default behavior)
        assert mock_all["cann_doc_struct"].main.call_count == 1
        assert mock_all["single_code_map"].main.call_count == 1
        assert mock_all["run_pipeline"].main.call_count == 1


class TestPathConfigInjection:
    """Tests for path_config injection into builders and data_root forwarding."""

    def test_builders_receive_same_path_config(self, mock_all, tmp_path):
        """4 builder 收到同一 path_config 实例。

        With doc_part_control=None (default all 5 parts), all 4 builders
        are called. Each should receive the same path_config instance
        (identity check via `is`), confirming no reconstruction inside
        load_snapshot.
        """
        op = str(tmp_path / "op")
        out = str(tmp_path / "out")
        ops = str(tmp_path / "ops")

        # doc_part_control=None → default all 5 parts → all 4 builders called
        load_snapshot(op, out, ops, doc_part_control=None)

        builders = [
            mock_all["run_pipe_line_single"],
            mock_all["get_constraint_evidence"],
            mock_all["summary_function_proto_main"],
            mock_all["extract_evidence"],
        ]

        # Capture the path_config from the first builder
        first_pc = builders[0].call_args.args[0]
        assert first_pc is not None

        # Assert all builders received the same path_config instance
        for builder in builders:
            assert builder.call_args.args[0] is first_pc

    def test_data_root_forwarded(self, mock_all, tmp_path):
        """data_root 透传到 PathConfig。

        load_snapshot(data_root="/shared") should construct a PathConfig
        whose _data_root == Path("/shared"), enabling operator-level data
        to be read from the shared directory.
        """
        op = str(tmp_path / "op")
        out = str(tmp_path / "out")
        ops = str(tmp_path / "ops")

        load_snapshot(
            op, out, ops, data_root="/shared", doc_part_control=["sub_function_hash"]
        )

        # The path_config passed to run_pipe_line_single should have
        # _data_root == Path("/shared")
        pc = mock_all["run_pipe_line_single"].call_args.args[0]
        assert pc._data_root == Path("/shared")
