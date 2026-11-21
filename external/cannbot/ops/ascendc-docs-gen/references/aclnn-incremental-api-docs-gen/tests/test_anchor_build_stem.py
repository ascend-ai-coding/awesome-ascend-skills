# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""TDD tests for snapshot.load_snapshot md_file/source_files threading (Task 4: C1).

The load_snapshot implementation lives in src/snapshot_construction/snapshot.py
(the former copy in change_detection/anchor_build.py was removed —
anchor_rebuild imports it from there). anchor_rebuild_cli no longer exists;
its md_file/source_files forwarding duty is carried by anchor_rebuild itself.

Test plan:
1. test_load_snapshot_threads_md_file — PathConfig receives md_file kwarg.
2. test_load_snapshot_threads_source_files — 4 builders receive source_files.
3. test_load_snapshot_md_file_none_backward_compat — md_file=None is safe.
4. test_anchor_rebuild_threads_md_file — anchor_rebuild forwards md_file.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.change_detection.anchor_build import anchor_rebuild
from src.snapshot_construction.snapshot import load_snapshot

# =============================================================================
# load_snapshot tests
# =============================================================================


class TestLoadSnapshotThreadsMdFile:
    """load_snapshot must forward md_file/data_root to PathConfig constructor
    and source_files to the 4 builder calls.
    """

    @pytest.fixture
    def mocks(self):
        """Patch all external collaborators of load_snapshot.

        PathConfig is patched to capture construction kwargs.
        All I/O functions (cann_doc_struct, single_code_map, run_pipeline,
        4 builders, expand_file_deps, DataLoader, statistic_anchor_info)
        are patched to no-ops.
        """
        with (
            patch("src.snapshot_construction.snapshot.PathConfig") as pc_mock,
            patch("src.snapshot_construction.snapshot.cann_doc_struct") as cann,
            patch("src.snapshot_construction.snapshot.single_code_map") as scm,
            patch("src.snapshot_construction.snapshot.run_pipeline") as rp,
            patch("src.snapshot_construction.snapshot.run_pipe_line_single") as rpls,
            patch("src.snapshot_construction.snapshot.get_constraint_evidence") as gce,
            patch("src.snapshot_construction.snapshot.summary_function_proto_main") as sfp,
            patch("src.snapshot_construction.snapshot.extract_evidence") as ee,
            patch("src.snapshot_construction.snapshot.expand_file_deps") as efd,
            patch("src.snapshot_construction.snapshot.DataLoader") as dl,
            patch(
                "src.snapshot_construction.snapshot.statistic_anchor_info",
                return_value=[{"doc_part": "stub"}],
            ) as sai,
        ):
            # Make PathConfig mock return a MagicMock with needed attributes
            # so str(path_config.anchor_dir) etc. don't blow up.
            pc_mock.return_value = MagicMock(
                operator_path="/tmp/op",
                out_path="/tmp/out",
                anchor_dir="/tmp/out/anchor",
                doc_json="/tmp/out/doc_json",
                code_map="/tmp/out/code_map",
                tree_id="/tmp/out/tree_id",
                md_stem=None,
            )
            yield {
                "pc": pc_mock,
                "cann": cann,
                "scm": scm,
                "rp": rp,
                "rpls": rpls,
                "gce": gce,
                "sfp": sfp,
                "ee": ee,
                "efd": efd,
                "dl": dl,
                "sai": sai,
            }

    def test_load_snapshot_threads_md_file(self, mocks):
        """md_file kwarg must reach PathConfig constructor."""
        load_snapshot(
            "/tmp/op",
            "/tmp/out",
            "/tmp/ops_root",
            doc_part_control=["sub_function_hash"],
            md_file="aclnnFakeOpV3.md",
            skip_data_prep=True,
        )
        assert mocks["pc"].called
        kwargs = mocks["pc"].call_args.kwargs
        assert kwargs.get("md_file") == "aclnnFakeOpV3.md"

    def test_load_snapshot_threads_source_files(self, mocks):
        """source_files kwarg must reach all 4 builders."""
        load_snapshot(
            "/tmp/op",
            "/tmp/out",
            "/tmp/ops_root",
            doc_part_control=[
                "sub_function_hash",
                "constraint",
                "function_proto",
                "summary",
                "support",
            ],
            source_files=["a.cpp"],
        )
        assert mocks["rpls"].call_args.kwargs.get("source_files") == ["a.cpp"]
        assert mocks["gce"].call_args.kwargs.get("source_files") == ["a.cpp"]
        assert mocks["sfp"].call_args.kwargs.get("source_files") == ["a.cpp"]
        assert mocks["ee"].call_args.kwargs.get("source_files") == ["a.cpp"]

    def test_load_snapshot_md_file_none_backward_compat(self, mocks):
        """md_file=None must not error; PathConfig.md_file kwarg is None."""
        load_snapshot(
            "/tmp/op",
            "/tmp/out",
            "/tmp/ops_root",
            doc_part_control=["sub_function_hash"],
            md_file=None,
            skip_data_prep=True,
        )
        kwargs = mocks["pc"].call_args.kwargs
        assert kwargs.get("md_file") is None


# =============================================================================
# anchor_rebuild tests
# =============================================================================


class TestAnchorRebuildThreadsMdFile:
    """anchor_rebuild must forward md_file+source_files to load_snapshot."""

    def test_anchor_rebuild_threads_md_file(self):
        """anchor_rebuild(..., md_file=...) must forward to load_snapshot."""
        with (
            patch(
                "src.change_detection.anchor_build.load_snapshot",
                return_value={"stub": True},
            ) as ls_mock,
            patch("shutil.copytree"),
            patch("shutil.rmtree"),
            patch("os.path.exists", return_value=False),
        ):
            anchor_rebuild(
                operator_name="aclnnFakeOp",
                operator_path="fake/op",
                change_operator_name="aclnnFakeOp",
                data_path="/tmp/data",
                filter_list=["sub_function_hash"],
                work_dir="/tmp/work",
                md_file="aclnnV3.md",
                source_files=["a.cpp"],
            )
        # load_snapshot received md_file and source_files
        kwargs = ls_mock.call_args.kwargs
        assert kwargs.get("md_file") == "aclnnV3.md"
        assert kwargs.get("source_files") == ["a.cpp"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
