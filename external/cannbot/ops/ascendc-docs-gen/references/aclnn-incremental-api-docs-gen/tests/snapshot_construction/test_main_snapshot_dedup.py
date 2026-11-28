# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for main_snapshot.main — data prep dedup (Task 8).

Tests cover:
- prepare_operator_data called once before the loop (when md_files non-empty)
- load_snapshot called with skip_data_prep=True + data_root=operator_out
- 0-md case: prepare skipped, load_snapshot skipped, result still printed
- 1-md case: prepare once, load_snapshot once
"""

import os

# Ensure env vars set before importing main_snapshot (module-level check
# at L13-18 raises Exception if INCREMENTAL_LLM_KEY/URL/MODEL missing).
os.environ.setdefault("INCREMENTAL_LLM_KEY", "test-key")
os.environ.setdefault("INCREMENTAL_LLM_URL", "test-url")
os.environ.setdefault("INCREMENTAL_LLM_MODEL", "test-model")

from pathlib import Path
from unittest.mock import patch

import src.main_snapshot as main_snapshot


class TestMainSnapshotDedup:
    """Tests for prepare_once + loop_skip dedup pattern in main()."""

    def test_prepare_once_loop_skip(self):
        """3 md: prepare_operator_data 调一次, load_snapshot 调三次(skip+data_root).

        Mock _discover_md_docs to return 3 md files. After main() runs:
        - prepare_operator_data called exactly once (before the loop)
        - load_snapshot called 3 times (once per md)
        - each load_snapshot call has data_root="out/abs" and skip_data_prep=True
        """
        md_files = [
            Path("/work/ops/math/abs/aclnnAbs.md"),
            Path("/work/ops/math/abs/aclnnAdd.md"),
            Path("/work/ops/math/abs/aclnnSub.md"),
        ]
        with (
            patch.object(main_snapshot, "_discover_md_docs", return_value=md_files),
            patch.object(main_snapshot, "prepare_operator_data") as prep,
            patch.object(main_snapshot, "load_snapshot", return_value={}) as load,
            patch.object(main_snapshot, "resolve_md_to_source_files", return_value=[]),
        ):
            main_snapshot.main("ops/math/abs", "out", "/work")

        # prepare called exactly once (before the loop)
        assert prep.call_count == 1
        # load_snapshot called once per md (3 times)
        assert load.call_count == 3
        # each call has data_root="out/abs" and skip_data_prep=True
        for call in load.call_args_list:
            assert call.kwargs["data_root"] == "out/abs"
            assert call.kwargs["skip_data_prep"] is True

    def test_zero_md_skips_prepare(self, capsys):
        """0 md: prepare 跳过, load_snapshot 跳过, ===snapshot-result=== 仍打印.

        Mock _discover_md_docs to return []. After main() runs:
        - prepare_operator_data not called (0 md → skip)
        - load_snapshot not called (loop body never executes)
        - ===snapshot-result=== still printed (empty echo_info)
        """
        with (
            patch.object(main_snapshot, "_discover_md_docs", return_value=[]),
            patch.object(main_snapshot, "prepare_operator_data") as prep,
            patch.object(main_snapshot, "load_snapshot") as load,
        ):
            main_snapshot.main("ops/math/abs", "out", "/work")

        assert prep.call_count == 0
        assert load.call_count == 0
        captured = capsys.readouterr().out
        assert "===snapshot-result===" in captured

    def test_single_md_works(self):
        """1 md: prepare 一次, load_snapshot 一次(skip_data_prep=True).

        Mock _discover_md_docs to return 1 md file. After main() runs:
        - prepare_operator_data called once
        - load_snapshot called once with skip_data_prep=True
        """
        md_files = [Path("/work/ops/math/abs/aclnnAbs.md")]
        with (
            patch.object(main_snapshot, "_discover_md_docs", return_value=md_files),
            patch.object(main_snapshot, "prepare_operator_data") as prep,
            patch.object(main_snapshot, "load_snapshot", return_value={}) as load,
            patch.object(main_snapshot, "resolve_md_to_source_files", return_value=[]),
        ):
            main_snapshot.main("ops/math/abs", "out", "/work")

        assert prep.call_count == 1
        assert load.call_count == 1
        assert load.call_args.kwargs["skip_data_prep"] is True
