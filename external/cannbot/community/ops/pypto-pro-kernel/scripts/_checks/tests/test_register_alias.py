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


@pytest.mark.parametrize("bound", ["3", "blocks"])
def test_persistent_seed_read_at_the_next_iteration_is_advisory(tmp_path, bound):
    path = probe(
        tmp_path,
        "seed = vf.full(1.0, mask)\n"
        f"for block in pl.range({bound}):\n    state = seed\n"
        "    for step in pl.range(6):\n        state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, state, mask)",
    )
    findings = register_alias.check(path)
    assert len(findings) == 1 and findings[0].advisory
    assert findings[0].lineno == 9


@pytest.mark.parametrize("bounds", ["0", "1", "7, 8", "0, 64, 64"])
def test_at_most_one_iteration_has_no_future_seed_read(tmp_path, bounds):
    path = probe(
        tmp_path,
        "seed = vf.full(1.0, mask)\n"
        f"for block in pl.range({bounds}):\n    state = seed\n"
        "    state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


def test_seed_reinitialized_each_iteration_does_not_need_preservation(tmp_path):
    path = probe(
        tmp_path,
        "for block in pl.range(3):\n"
        "    seed = vf.full(1.0, mask)\n    state = seed\n"
        "    for step in pl.range(6):\n        state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


def test_each_iteration_copies_into_an_independent_destination(tmp_path):
    path = probe(
        tmp_path,
        "seed = vf.full(1.0, mask)\n"
        "for block in pl.range(3):\n    state = vf.move(seed, mask)\n"
        "    state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


def test_unknown_seed_producer_stays_unclassified_across_iterations(tmp_path):
    path = probe(
        tmp_path,
        "seed = helper()\n"
        "for block in pl.range(3):\n    state = seed\n"
        "    state = vf.add(state, value, mask)\n"
        "    vf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


@pytest.mark.parametrize("loop", ["for i in pl.range(4):", "while active:"])
def test_existing_register_name_rebound_across_loop_is_reviewed(tmp_path, loop):
    path = probe(
        tmp_path,
        "state = vf.full(0.0, mask)\n" + loop + "\n"
        "    next_state = vf.add(state, value, mask)\n    state = next_state\n"
        "vf.store_align(tile, state, mask)",
    )
    findings = register_alias.check(path)
    assert len(findings) == 1 and findings[0].rule == register_alias.LOOP_RULE
    assert findings[0].advisory and findings[0].lineno == 8


def test_existing_register_updated_with_vf_move_is_not_rebound(tmp_path):
    path = probe(
        tmp_path,
        "state = vf.full(0.0, mask)\n"
        "for i in pl.range(4):\n    next_state = vf.add(state, value, mask)\n"
        "    state = vf.move(next_state, mask)\nvf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


def test_iteration_local_alias_does_not_create_a_loop_merge(tmp_path):
    path = probe(
        tmp_path,
        "for i in pl.range(4):\n"
        "    source = vf.load_align(tile)\n    local = source\n"
        "    local = vf.add(local, value, mask)\n    vf.store_align(tile, local, mask)",
    )
    assert register_alias.check(path) == []


def test_unknown_initial_binding_is_not_assumed_to_be_a_register_merge(tmp_path):
    path = probe(
        tmp_path,
        "state = helper()\n"
        "for i in pl.range(4):\n    next_state = vf.full(1.0, mask)\n"
        "    state = next_state\nvf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []


def test_plain_alias_after_a_loop_is_not_a_loop_carried_assignment(tmp_path):
    path = probe(
        tmp_path,
        "state = vf.full(0.0, mask)\n"
        "for i in pl.range(4):\n    next_state = vf.add(state, value, mask)\n"
        "state = next_state\nvf.store_align(tile, state, mask)",
    )
    assert register_alias.check(path) == []
