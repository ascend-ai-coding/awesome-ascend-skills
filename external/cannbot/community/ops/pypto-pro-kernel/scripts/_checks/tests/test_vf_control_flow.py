# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from textwrap import dedent

import pytest

from _checks.vf_control_flow import check
from _checks.cli import report_rules


def source(tmp_path, body):
    path = tmp_path / "probe.py"
    path.write_text("import pypto_pro.language as pl\n" + dedent(body))
    return path


def test_block_flags_inside_nested_loops_are_advisory(tmp_path):
    path = source(
        tmp_path,
        """
        def consume(dst):
            pass
        def finish(dst):
            pass
        @pl.vector_function
        def update(dst, n, first, last):
            for row in pl.range(n):
                for chunk in pl.range(8):
                    if not first:
                        consume(dst)
                    if last:
                        finish(dst)
    """,
    )
    findings = check(path)
    assert len(findings) == 2 and all(f.advisory for f in findings)
    assert report_rules([path])  # The rule must not prohibit legal runtime control.


def test_local_scalar_dependencies_and_loop_indices_are_followed(tmp_path):
    path = source(
        tmp_path,
        """
        @pl.vector_function
        def body(block, rows):
            final = block == 7
            for row in pl.range(rows):
                selected = row % 2
                if final or selected:
                    consume(row)
    """,
    )
    assert len(check(path)) == 1


def test_factory_constants_and_runtime_loop_bounds_are_not_branches(tmp_path):
    path = source(
        tmp_path,
        """
        def make_update(first):
            @pl.vector_function
            def body(dst, rows):
                for row in pl.range(rows):
                    if first:
                        initialize(dst)
                    else:
                        update(dst)
            return body
    """,
    )
    assert not check(path)


@pytest.mark.parametrize(
    "decorator",
    [
        "@pl.vector_function(mode='simt')",
        "@pl.vector_function(mode=unknown_mode)",
        "@pl.jit()",
    ],
)
def test_other_execution_modes_are_out_of_scope(tmp_path, decorator):
    path = source(
        tmp_path,
        f"""
        {decorator}
        def body(flag, rows):
            for row in pl.range(rows):
                if flag:
                    consume(row)
    """,
    )
    assert not check(path)


def test_dispatch_outside_the_vf_loop_is_not_reported(tmp_path):
    path = source(
        tmp_path,
        """
        @pl.vector_function
        def body(flag, rows):
            if flag:
                for row in pl.range(rows):
                    consume(row)
    """,
    )
    assert not check(path)


def test_nested_helpers_are_not_mistaken_for_the_enclosing_vf(tmp_path):
    path = source(
        tmp_path,
        """
        @pl.vector_function
        def body(flag, rows):
            def helper():
                for row in pl.range(rows):
                    if flag:
                        consume(row)
            consume(rows)
    """,
    )
    assert not check(path)


def test_ternary_and_while_control_are_visible(tmp_path):
    path = source(
        tmp_path,
        """
        @pl.vector_function
        def body(rows, flag):
            i = 0
            while i < rows:
                value = i if flag else 0
                i += 1
    """,
    )
    assert len(check(path)) == 2
