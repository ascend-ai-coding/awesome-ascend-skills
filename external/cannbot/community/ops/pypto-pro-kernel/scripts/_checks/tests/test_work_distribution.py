# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from textwrap import dedent

import pytest

from _checks.cli import report_rules
from _checks.work_distribution import RULE, check


def probe(tmp_path, body, alias="pl"):
    path = tmp_path / "probe.py"
    path.write_text(f"import pypto_pro.language as {alias}\n" + dedent(body))
    return path


@pytest.mark.parametrize(
    "start", ["worker", "begin + worker", "worker + begin", "first"]
)
def test_group_restart_is_advisory_and_points_to_work_loop(tmp_path, start):
    path = probe(
        tmp_path,
        f"""
        def consume(tile):
            pass
        @pl.jit()
        def kernel(groups, workers, begin, count):
            worker = pl.get_block_idx()
            first = begin + worker
            for group in pl.range(groups):
                for tile in pl.range({start}, count, workers):
                    consume(tile)
    """,
    )
    findings = check(path)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule == RULE and finding.advisory
    assert "for tile" in path.read_text().splitlines()[finding.lineno - 1]
    assert report_rules([path])


def test_import_alias_and_direct_block_id(tmp_path):
    path = probe(
        tmp_path,
        """
        @native.jit
        def kernel():
            for group in native.range(8):
                for row in native.range(native.get_block_idx(), 16, native.get_block_num()):
                    consume(row)
    """,
        alias="native",
    )
    assert len(check(path)) == 1


def test_global_prefix_mapping_is_not_a_restart(tmp_path):
    path = probe(
        tmp_path,
        """
        @pl.jit()
        def kernel(groups, workers):
            worker = pl.get_block_idx()
            prefix = 0
            for group in pl.range(groups):
                first = (worker + workers - prefix % workers) % workers
                prefix += count
                for tile in pl.range(first, count, workers):
                    consume(tile)
    """,
    )
    assert not check(path)


@pytest.mark.parametrize(
    "setup,start",
    [
        ("worker = get_worker()", "worker"),
        ("worker = pl.get_block_idx()\n    worker = 0", "worker"),
        ("worker = pl.get_block_idx()\n    worker += workers", "worker"),
        ("worker = pl.get_block_idx()", "remap(worker, group)"),
        ("worker = pl.get_block_idx()", "worker * count"),
    ],
)
def test_unknown_or_rebound_mappings_are_not_guessed(tmp_path, setup, start):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n@pl.jit()\ndef kernel():\n    "
        + setup
        + "\n    for group in pl.range(8):\n"
        + f"        for row in pl.range({start}, count, workers):\n            consume(row)\n"
    )
    assert not check(path)


@pytest.mark.parametrize(
    "decorator", ["@pl.vector_function", "@pl.vector_function(mode='simt')", ""]
)
def test_only_jit_functions_are_scanned(tmp_path, decorator):
    path = probe(
        tmp_path,
        f"""
        {decorator}
        def kernel():
            for group in pl.range(8):
                for row in pl.range(pl.get_block_idx(), 16, 8):
                    consume(row)
    """,
    )
    assert not check(path)


def test_top_level_worker_loop_and_nested_helpers_do_not_trigger(tmp_path):
    path = probe(
        tmp_path,
        """
        @pl.jit()
        def kernel():
            for row in pl.range(pl.get_block_idx(), 16, 8):
                consume(row)
            def helper():
                for group in pl.range(8):
                    for row in pl.range(pl.get_block_idx(), 16, 8):
                        consume(row)
    """,
    )
    assert not check(path)


def test_shadowed_language_parameter_is_not_native(tmp_path):
    path = probe(
        tmp_path,
        """
        @pl.jit()
        def kernel(pl):
            for group in pl.range(8):
                for row in pl.range(pl.get_block_idx(), 16, 8):
                    consume(row)
    """,
    )
    assert not check(path)
