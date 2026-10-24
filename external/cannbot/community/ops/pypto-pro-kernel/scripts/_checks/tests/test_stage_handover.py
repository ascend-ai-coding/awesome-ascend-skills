# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Direct stage participation controls; no PyPTO installation is needed."""

import pytest

from _checks.stage_handover import RULE, check


def source(tmp_path, marker="", consumer_kind="vector", consumer="use(tile.current())"):
    path = tmp_path / "probe.py"
    path.write_text(f"""import pypto_pro.language as pl
def factory():
    @pl.pipeline.stage
    def produce(tile):
        use(tile.current())
    @pl.pipeline.stage
    def consume(tile):
        {consumer}
    @pl.jit(pipeline=pl.pipeline.PipelineConfig(preload=0))
    def kernel():
        shared = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=2{marker})
        for step in pl.range(2):
            with pl.section_cube():
                produce(shared)
            with pl.section_{consumer_kind}():
                consume(tile=shared)
    return kernel
""")
    return path


@pytest.mark.parametrize(
    "marker", ["", ", fwd_ids=None, bwd_ids=None", ", fwd_ids=[0, 1]"]
)
def test_unmarked_handover_needs_review(tmp_path, marker):
    findings = check(source(tmp_path, marker))
    assert len(findings) == 1
    assert findings[0].rule == RULE and findings[0].advisory
    assert findings[0].lineno == 11


def test_explicit_mark_and_local_groups_are_not_reported(tmp_path):
    assert not check(source(tmp_path, ", fwd_ids=[0, 1], bwd_ids=[0, 1]"))
    assert not check(source(tmp_path, ", fwd_ids=FWD, bwd_ids=BWD"))
    assert not check(source(tmp_path, consumer_kind="cube"))


def test_unused_stage_parameters_do_not_create_edges(tmp_path):
    assert not check(source(tmp_path, consumer="use(other)"))


def test_same_named_stages_in_other_factories_do_not_change_binding(tmp_path):
    path = source(tmp_path)
    path.write_text(
        path.read_text()
        + """
def unrelated():
    @pl.pipeline.stage
    def consume(tile):
        pass
"""
    )
    assert len(check(path)) == 1


def test_direct_group_alias_is_outside_this_prechecks_scope(tmp_path):
    path = source(tmp_path)
    path.write_text(
        path.read_text().replace("consume(tile=shared)", "consume(tile=alias)")
    )
    assert not check(path)


def test_nested_helpers_are_not_executed_as_stage_bodies(tmp_path):
    path = source(tmp_path, consumer="pass")
    path.write_text(
        path.read_text().replace(
            "        pass", "        def unused():\n            use(tile.current())"
        )
    )
    assert not check(path)
