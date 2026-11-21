# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bare-Tile findings require a source-bound handover proof, not a declaration."""

import pytest

from _checks import synchronisation
from _checks.paths import PYPTO_ROOT

RULE = synchronisation.RULE
TYPE = "pl.TileType(shape=[1,8],dtype=pl.DT_INT32,target_memory=pl.MemorySpace.Vec)"


def write(tmp_path, body, *, decorator="@pl.jit()", prefix="", claim=""):
    text = (
        "import pypto_pro.language as pl\n"
        + prefix
        + "\n"
        + decorator
        + "\ndef kernel(x,out):\n"
    )
    if claim:
        text += '    """' + claim + '"""\n'
    text += "    tile=pl.make_tile(" + TYPE + ",addr=0)\n" + body
    p = tmp_path / "candidate.py"
    p.write_text(text)
    return p


def state(path, **kwargs):
    findings = synchronisation.check(path, **kwargs)
    return (
        "error"
        if any(not f.advisory and not f.waived for f in findings)
        else "advisory"
        if findings
        else "none"
    )


COPY = "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n        pl.store(out,tile,[0,0])\n"


@pytest.mark.parametrize(
    "decorator",
    ["@pl.jit", "@pl.jit()", "@pl.jit(auto_mutex=True)", "@pl.jit(name='copy')"],
)
def test_adjacent_unprotected_handover_is_a_hard_finding(tmp_path, decorator):
    p = write(tmp_path, COPY, decorator=decorator)
    findings, scope = synchronisation.inspect(p)
    assert len(findings) == 1 and not findings[0].advisory
    assert (
        findings[0].lineno
        == p.read_text()
        .splitlines()
        .index("    tile=pl.make_tile(" + TYPE + ",addr=0)")
        + 1
    )
    assert scope["guard_verified"] and scope["proven_handover"] == 1
    assert len(scope["framework_sources"]) == 2


@pytest.mark.parametrize(
    "body",
    [
        "    pass\n",
        "    pl.set_validshape(tile,[1,8])\n",
        "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n",
        "    with pl.section_vector():\n        pl.store(out,tile,[0,0])\n",
        "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n        pl.load(tile,x,[1,0])\n",
    ],
)
def test_declaration_metadata_and_single_pipe_are_not_race_proofs(tmp_path, body):
    assert state(write(tmp_path, body)) == "none"


@pytest.mark.parametrize(
    "body",
    [
        "    consume(tile,x,out)\n",
        "    with pl.section_vector():\n        if flag:\n           "
        " pl.load(tile,x,[0,0])\n            pl.store(out,tile,[0,0])\n",
        "    with pl.section_vector():\n        for row in pl.range(rows):\n       "
        "     pl.load(tile,x,[row,0])\n            pl.store(out,tile,[row,0])\n",
        "    alias=tile\n" + COPY,
        "    tile=other\n" + COPY,
        "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n       "
        " pl.system.bar_all()\n        pl.store(out,tile,[0,0])\n",
        "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n       "
        " pl.system.sync_src(set_pipe=pl.PipeType.MTE2,wait_pipe=pl.PipeType.MTE3,event_id=0)\n"
        "        pl.store(out,tile,[0,0])\n",
        "    def helper():\n        consume(tile)\n    helper()\n",
        "    if flag:\n        return\n" + COPY,
        "    with pl.section_vector():\n        return\n       "
        " pl.load(tile,x,[0,0])\n        pl.store(out,tile,[0,0])\n",
        "    with pl.section_vector():\n        begin()\n       "
        " pl.load(tile,x,[0,0])\n        pl.store(out,tile,[0,0])\n        end()\n",
        "    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n       "
        " pl.store(wait_and_return(out),tile,[0,0])\n",
    ],
)
def test_incomplete_control_alias_or_manual_sync_evidence_is_advisory(tmp_path, body):
    assert state(write(tmp_path, body)) == "advisory"


@pytest.mark.parametrize(
    "decorator",
    [
        "@pl.jit(auto_mutex=FLAG)",
        "@pl.jit(**options)",
        "@pl.jit(pipeline=plan)",
        "@unknown\n@pl.jit()",
    ],
)
def test_unresolved_decorator_options_are_not_assumed_enabled(tmp_path, decorator):
    assert state(write(tmp_path, COPY, decorator=decorator)) == "advisory"


def test_explicit_manual_mode_is_outside_this_rule(tmp_path):
    assert state(write(tmp_path, COPY, decorator="@pl.jit(auto_mutex=False)")) == "none"


def test_distinct_buffer_uses_do_not_establish_a_handover(tmp_path):
    body = (
        "    other=pl.make_tile("
        + TYPE
        + ",addr=64)\n    with pl.section_vector():\n        pl.load(tile,x,[0,0])\n        pl.store(out,other,[0,0])\n"
    )
    assert state(write(tmp_path, body)) == "none"


def test_group_elsewhere_does_not_protect_a_bare_handover(tmp_path):
    body = (
        "    group=pl.make_tile_group(type="
        + TYPE
        + ",addrs=[64],mutex_ids=[0])\n"
        + COPY
    )
    assert state(write(tmp_path, body)) == "advisory"


@pytest.mark.parametrize(
    "prefix",
    [
        "pl=other\n",
        "pl.load=other\n",
        "import other as pl\n",
        "def pl(): pass\n",
        "class pl: pass\n",
        "del pl\n",
    ],
)
def test_shadowed_native_namespace_is_not_a_hard_proof(tmp_path, prefix):
    assert state(write(tmp_path, COPY, prefix=prefix)) == "advisory"


def test_missing_selected_source_leaves_handover_unverified(tmp_path):
    assert state(write(tmp_path, COPY), pypto_root=tmp_path / "missing") == "advisory"


@pytest.mark.parametrize("changed", ["role", "pipe"])
def test_source_guard_refreshes_after_a_role_or_pipe_change(tmp_path, changed):
    root = tmp_path / "framework"
    paths = [
        "python/pypto_pro/language/parser/_op_pipeline.py",
        "framework/src/interface/pypto_pro/backend/backend_cce_block_out_ops.cpp",
    ]
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PYPTO_ROOT / name).read_bytes())
    source = write(tmp_path, COPY)
    assert state(source, pypto_root=root) == "error"
    path = root / paths[0 if changed == "role" else 1]
    text = path.read_text()
    text = (
        text.replace('"load": ["W", "R"]', '"load": ["R", "R"]')
        if changed == "role"
        else text.replace(".set_pipe(ir::PipeType::MTE2)", ".set_pipe(ir::PipeType::V)")
    )
    path.write_text(text)
    assert state(source, pypto_root=root) == "advisory"


def test_unknown_tile_type_does_not_prove_vec_pipe_semantics(tmp_path):
    p = write(tmp_path, COPY)
    p.write_text(p.read_text().replace(TYPE, "tile_type"))
    assert state(p) == "advisory"


def test_counterexample_waiver_and_stale_claim_are_preserved(tmp_path):
    p = write(tmp_path, COPY, claim=f"反例：{RULE} —— missing handover is intentional")
    findings = synchronisation.check(p)
    assert len(findings) == 1 and findings[0].waived
    p = write(tmp_path, "    pass\n", claim=f"反例：{RULE} —— stale declaration")
    assert any(not f.waived and not f.advisory for f in synchronisation.check(p))
