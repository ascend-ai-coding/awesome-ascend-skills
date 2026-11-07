# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks import register_alias
from _checks.cli import report_rules


def probe(tmp_path, body, alias="vf"):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        f"from pypto_pro.language import Vf as {alias}\n"
        "@pl.vector_function\ndef body(tile, mask):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "update",
    [
        "b = vf.add(b, x, mask)",
        "b = vf.full(1.0, mask)",
        "for i in pl.range(3):\n    b = vf.add(b, x, mask)",
    ],
)
def test_mutated_alias_with_a_later_seed_read_is_advisory(tmp_path, update):
    path = probe(
        tmp_path,
        "a = vf.full(0.0, mask)\nb = a\n" + update + "\nvf.store_align(tile, a, mask)",
    )
    findings = register_alias.check(path)
    assert len(findings) == 1 and findings[0].advisory and "`b`" in findings[0].message


@pytest.mark.parametrize(
    "initialize", ["b = vf.full(0.0, mask)", "b = vf.move(a, mask)"]
)
def test_fresh_destinations_are_independent(tmp_path, initialize):
    path = probe(
        tmp_path,
        "a = vf.full(0.0, mask)\n"
        + initialize
        + "\nb = vf.add(b, x, mask)\nvf.store_align(tile, a, mask)",
    )
    assert register_alias.check(path) == []


def test_dead_original_name_is_not_reported(tmp_path):
    path = probe(
        tmp_path,
        "a = vf.load_align(tile)\nb = a\n"
        "b = vf.add(b, x, mask)\nvf.store_align(tile, b, mask)",
    )
    assert register_alias.check(path) == []


@pytest.mark.parametrize(
    "initialize", ["a = helper()", "a = vf.unknown_api()", "a = tile"]
)
def test_unknown_producers_are_not_assumed_to_be_registers(tmp_path, initialize):
    path = probe(
        tmp_path,
        initialize + "\nb = a\nb = vf.add(b, x, mask)\nvf.store_align(tile, a, mask)",
    )
    assert register_alias.check(path) == []


@pytest.mark.parametrize(
    "rebind",
    ["b = helper()", "b, c = helper()", "b: int = 3", "if flag:\n    b = helper()"],
)
def test_uncertain_rebindings_drop_the_alias(tmp_path, rebind):
    path = probe(
        tmp_path,
        "a = vf.full(0.0, mask)\nb = a\n"
        + rebind
        + "\nb = vf.add(b, x, mask)\nvf.store_align(tile, a, mask)",
    )
    assert register_alias.check(path) == []


def test_alias_chain_and_import_alias_do_not_fail_cli(tmp_path, capsys):
    path = probe(
        tmp_path,
        "x = ops.full(1.0, mask)\na = ops.full(0.0, mask)\nb = a\nc = b\n"
        "c = ops.add(c, x, mask)\nops.store_align(tile, a, mask)",
        alias="ops",
    )
    assert len(register_alias.check(path)) == 1
    assert report_rules([path])
    output = capsys.readouterr().out
    assert "[需核查]" in output and "0 处违规" in output and "1 处需核查提示" in output


@pytest.mark.parametrize("condition", ["if active:", "if active:\n    if selected:"])
def test_mutating_an_incoming_alias_inside_an_if_arm(tmp_path, condition):
    depth = condition.count("if ")
    indent = "    " * depth
    path = probe(
        tmp_path,
        "seed = vf.full(-1, mask)\nstate = seed\n"
        + condition
        + "\n"
        + indent
        + "state = vf.select(value, state, mask)\n"
        + indent
        + "vf.store_align(tile, seed, mask)",
    )
    assert len(register_alias.check(path)) == 1


def test_per_iteration_alias_updated_in_a_conditional_rank_loop(tmp_path):
    path = probe(
        tmp_path,
        "seed = vf.full(-1, mask)\n"
        "for block in pl.range(3):\n    state = seed\n"
        "    for rank in pl.range(4):\n        if rank < active:\n"
        "            state = vf.select(value, state, mask)\n"
        "            ids = vf.select(seed, ids, mask)",
    )
    assert len(register_alias.check(path)) == 1


def test_one_arm_does_not_supply_aliases_to_another_arm_or_join(tmp_path):
    path = probe(
        tmp_path,
        "seed = vf.full(-1, mask)\n"
        "if active:\n    state = seed\n"
        "else:\n    state = vf.full(0, mask)\n    vf.store_align(tile, seed, mask)\n"
        "state = vf.add(state, value, mask)\nvf.store_align(tile, seed, mask)",
    )
    assert register_alias.check(path) == []


def test_rebinding_inside_an_arm_removes_the_incoming_alias(tmp_path):
    path = probe(
        tmp_path,
        "seed = vf.full(-1, mask)\nstate = seed\n"
        "if active:\n    state = helper()\n    state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, seed, mask)",
    )
    assert register_alias.check(path) == []


def test_fresh_register_inside_loop_arm_is_independent(tmp_path):
    path = probe(
        tmp_path,
        "seed = vf.full(-1, mask)\n"
        "for block in pl.range(3):\n    state = vf.full(-1, mask)\n"
        "    if active:\n        state = vf.select(value, state, mask)\n"
        "        vf.store_align(tile, seed, mask)",
    )
    assert register_alias.check(path) == []
