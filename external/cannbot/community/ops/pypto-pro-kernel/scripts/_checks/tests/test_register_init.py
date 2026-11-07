# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import subprocess
import sys

import pytest

from _checks import register_init
from _checks.paths import SKILL_ROOT


def probe(tmp_path, body, alias="vf", parameters="tile, mask, a, b, first"):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        f"from pypto_pro.language import Vf as {alias}\n"
        "@pl.vector_function\n"
        f"def body({parameters}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


@pytest.mark.parametrize("op", ["mul_dst_add", "mul_add_dst"])
@pytest.mark.parametrize("declaration", ["", "acc: pl.RegTensor\n"])
def test_fused_destination_needs_a_value_not_only_a_declaration(
    tmp_path, op, declaration
):
    path = probe(tmp_path, declaration + f"acc = vf.{op}(a, b, mask)")
    findings = register_init.check(path)
    assert len(findings) == 1 and findings[0].rule == register_init.RULE
    assert not findings[0].advisory and not findings[0].waived
    assert findings[0].lineno == 5 + bool(declaration)


@pytest.mark.parametrize("op", ["mul_dst_add", "mul_add_dst"])
@pytest.mark.parametrize(
    "seed", ["vf.load_align(tile)", "vf.full(0.0, mask)", "vf.move(a, mask)"]
)
def test_same_name_seeded_before_update_is_legal(tmp_path, op, seed):
    path = probe(tmp_path, f"acc = {seed}\nacc = vf.{op}(a, b, mask)")
    assert register_init.check(path) == []


def test_online_softmax_loading_a_different_name_does_not_seed_the_target(tmp_path):
    path = probe(
        tmp_path,
        "for row in pl.range(8):\n"
        "    if first:\n        vf.store_align(tile, b, mask)\n"
        "    else:\n        l_old = vf.load_align(tile, [row, 0])\n"
        "        l_new = vf.mul_dst_add(a, b, mask)\n"
        "        vf.store_align(tile, l_new, mask)",
    )
    findings = register_init.check(path)
    assert len(findings) == 1 and findings[0].lineno == 10
    corrected = path.read_text().replace("l_old =", "l_new =")
    path.write_text(corrected)
    assert register_init.check(path) == []


def test_a_later_straight_line_seed_does_not_initialize_the_first_update(tmp_path):
    path = probe(
        tmp_path, "acc = vf.mul_dst_add(a, b, mask)\nacc = vf.load_align(tile)"
    )
    assert len(register_init.check(path)) == 1


def test_one_if_arm_does_not_seed_its_sibling(tmp_path):
    path = probe(
        tmp_path,
        "if first:\n    acc = vf.load_align(tile)\n"
        "else:\n    acc = vf.mul_dst_add(a, b, mask)",
    )
    assert len(register_init.check(path)) == 1


@pytest.mark.parametrize("seed", ["acc = helper(tile)", "acc = a"])
def test_unknown_value_bindings_are_not_rejected(tmp_path, seed):
    path = probe(tmp_path, seed + "\nacc = vf.mul_dst_add(a, b, mask)")
    assert register_init.check(path) == []


def test_loop_carried_conditional_seed_remains_unknown(tmp_path):
    path = probe(
        tmp_path,
        "for row in pl.range(8):\n"
        "    if row == 0:\n        acc = vf.load_align(tile)\n"
        "    else:\n        acc = vf.mul_dst_add(a, b, mask)",
    )
    assert register_init.check(path) == []


def test_nested_helper_binding_does_not_initialize_outer_destination(tmp_path):
    path = probe(
        tmp_path,
        "def helper():\n    acc = vf.load_align(tile)\n"
        "acc = vf.mul_dst_add(a, b, mask)",
    )
    assert len(register_init.check(path)) == 1


def test_import_alias_is_recognized(tmp_path):
    path = probe(tmp_path, "acc = ops.mul_dst_add(a, b, mask)", alias="ops")
    assert len(register_init.check(path)) == 1


def test_scope_distinguishes_direct_targets_from_unclassified_calls(tmp_path):
    path = probe(
        tmp_path,
        "acc = vf.load_align(tile)\n"
        "acc = vf.mul_dst_add(a, b, mask)\n"
        "vf.mul_add_dst(a, b, mask)\n"
        "other, unused = vf.mul_dst_add(a, b, mask)",
    )
    findings, counts = register_init.inspect(path)
    assert findings == [] and counts == (1, 2)


def test_shadowed_namespace_is_counted_as_unclassified(tmp_path):
    path = probe(
        tmp_path, "acc = vf.mul_dst_add(a, b, mask)", parameters="vf, a, b, mask"
    )
    assert register_init.scope_counts(path) == (0, 1)


@pytest.mark.parametrize(
    "parameters,body",
    [
        ("tile, mask, a, b, acc", "acc = vf.mul_dst_add(a, b, mask)"),
        ("vf, a, b, mask", "acc = vf.mul_dst_add(a, b, mask)"),
        ("a, b, mask", "vf = helper()\nacc = vf.mul_dst_add(a, b, mask)"),
    ],
)
def test_parameters_and_shadowed_names_are_unknown(tmp_path, parameters, body):
    assert register_init.check(probe(tmp_path, body, parameters=parameters)) == []


def test_counterexample_is_visible_and_stale_waiver_fails(tmp_path):
    claim = f'"""反例：{register_init.RULE} —— static-only unseeded destination."""\n'
    path = probe(tmp_path, claim + "acc = vf.mul_dst_add(a, b, mask)")
    findings = register_init.check(path)
    assert len(findings) == 1 and findings[0].waived
    path = probe(
        tmp_path, claim + "acc = vf.full(0.0, mask)\nacc = vf.mul_dst_add(a, b, mask)"
    )
    findings = register_init.check(path)
    assert len(findings) == 1 and not findings[0].waived


@pytest.mark.parametrize("seed,status", [("", 1), ("acc = vf.load_align(tile)\n", 0)])
def test_public_cli_returns_nonzero_for_unseeded_fma_without_executing_source(
    tmp_path, seed, status
):
    path = probe(tmp_path, seed + "acc = vf.mul_dst_add(a, b, mask)")
    path.write_text(
        path.read_text() + "raise RuntimeError('must not execute checked source')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            str(SKILL_ROOT / "scripts/check.py"),
            "rules",
            str(path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert (register_init.RULE in result.stdout) == bool(status)
