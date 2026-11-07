# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks.vf_scope import check
from _checks.cli import report_rules


def probe(tmp_path, body, parameters="out"):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        "from pypto_pro.language import Vf as vf\n"
        "@pl.jit()\n"
        + f"def kernel({parameters}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "operation",
    [
        "vf.create_mask()",
        "vf.full(0, mask)",
        "vf.store_align(out, reg, mask)",
        "pl.Vf.load_align(out)",
    ],
)
def test_initialization_and_memory_ops_need_vf(tmp_path, operation):
    found = check(probe(tmp_path, "with pl.section_vector():\n    " + operation))
    assert len(found) == 1 and not found[0].advisory


def test_control_flow_does_not_create_vf_scope(tmp_path):
    assert (
        len(
            check(
                probe(
                    tmp_path,
                    "for i in pl.range(4):\n    if i > 0:\n        vf.full(0, mask)",
                )
            )
        )
        == 1
    )


def test_nested_vf_and_unresolved_helpers_are_not_rejected(tmp_path):
    path = probe(
        tmp_path,
        "@pl.vector_function\ndef initialize(out):\n    vf.full(0, mask)\n"
        "def helper(out):\n    vf.full(0, mask)\ninitialize(out)",
    )
    assert not check(path)


def test_aliases_are_resolved(tmp_path):
    path = probe(tmp_path, "vf.create_mask()")
    path.write_text(
        path.read_text()
        .replace("as pl", "as language")
        .replace("pl.", "language.")
        .replace("as vf", "as registers")
        .replace("vf.", "registers.")
    )
    assert len(check(path)) == 1


@pytest.mark.parametrize(
    "parameters,body",
    [
        ("out, vf", "vf.create_mask()"),
        ("out", "vf = custom\nvf.create_mask()"),
        ("out, pl", "pl.Vf.create_mask()"),
    ],
)
def test_shadowed_namespace_is_unknown(tmp_path, parameters, body):
    assert not check(probe(tmp_path, body, parameters))


def test_plain_function_is_not_a_jit_context(tmp_path):
    path = probe(tmp_path, "vf.full(0, mask)")
    path.write_text(path.read_text().replace("@pl.jit()\n", ""))
    assert not check(path)


def test_waiver_stays_visible_and_cli_rejects_unwaived(tmp_path, capsys):
    path = probe(tmp_path, "vf.full(0, mask)")
    assert not report_rules([path])
    assert "vf-operation-needs-vector-function" in capsys.readouterr().out
    path = probe(
        tmp_path,
        '"""反例：vf-operation-needs-vector-function —— parse-only diagnostic"""\n'
        "vf.full(0, mask)",
    )
    assert check(path)[0].waived


@pytest.mark.parametrize("statement", ["return", "return None", "return result"])
def test_simd_vf_returns_are_rejected(tmp_path, statement):
    path = probe(tmp_path, statement)
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    assert [item.rule for item in check(path)] == ["simd-vf-cannot-return"]


@pytest.mark.parametrize(
    "decorator",
    [
        '@pl.vector_function(mode="simt")',
        '@pl.vector_function(mode="simt", max_threads=128)',
        "@pl.vector_function(mode=selected_mode)",
        "",
    ],
)
def test_other_return_contexts_are_not_rejected(tmp_path, decorator):
    path = probe(tmp_path, "return result")
    path.write_text(path.read_text().replace("@pl.jit()", decorator))
    assert not check(path)


@pytest.mark.parametrize("expression", ["1.0 / 127.0", "8 / 2", "3.0 // 2", "-3.0 % 2"])
def test_literal_scalar_division_is_a_gate(tmp_path, expression):
    path = probe(tmp_path, "value = " + expression)
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    found = check(path)
    assert (
        len(found) == 1
        and found[0].rule == "vf-scalar-division"
        and not found[0].advisory
    )


def test_unknown_division_types_are_advisory_and_integer_ops_are_legal(tmp_path):
    path = probe(
        tmp_path,
        "a = 1.0 / width\nb = width // 64\nc = width % 64\nd = vf.div(x, y, mask)",
    )
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    found = check(path)
    assert len(found) == 1 and found[0].advisory


def test_nested_python_helper_return_is_not_a_vf_return(tmp_path):
    path = probe(tmp_path, "def helper():\n    return 1.0 / 127.0\nhelper()")
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    assert not check(path)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("vf.add(vf.load_align(out, 0), vf.load_align(out, 64), mask)", 2),
        ("vf.store_align(out, reg=vf.full(1.0, mask), mask=mask)", 1),
        ("consume(pair=vf.de_interleave(left, right))", 1),
        ("vf.add(left, right, vf.create_mask())", 0),
        ("vf.add(vf.bit_cast(left, dtype=pl.DT_FP32), right, mask)", 0),
    ],
)
def test_nested_vf_result_calls_follow_the_parser_table(tmp_path, expression, expected):
    path = probe(tmp_path, "value = " + expression)
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    found = check(path)
    assert len(found) == expected
    assert all(
        item.rule == "vf-result-nested-call" and not item.advisory for item in found
    )


def test_separate_assignments_and_import_aliases(tmp_path):
    path = probe(
        tmp_path, "reg = vf.load_align(out, 0)\nvalue = vf.add(reg, reg, mask)"
    )
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    assert not check(path)
    path.write_text(
        path.read_text()
        .replace("vf.add(reg, reg, mask)", "vf.add(vf.load_align(out, 0), reg, mask)")
        .replace("as vf", "as registers")
        .replace("vf.", "registers.")
    )
    assert len(check(path)) == 1


@pytest.mark.parametrize(
    "decorator,parameters",
    [
        ("@pl.vector_function", "out, vf"),
        ('@pl.vector_function(mode="simt")', "out"),
        ("", "out"),
    ],
)
def test_nested_calls_outside_known_simd_namespace_are_not_rejected(
    tmp_path, decorator, parameters
):
    path = probe(tmp_path, "consume(vf.full(1.0, mask))", parameters)
    path.write_text(path.read_text().replace("@pl.jit()", decorator))
    assert not check(path)


def test_nested_call_rule_uses_selected_source_and_guards_changed_parser(tmp_path):
    from _checks.paths import PYPTO_ROOT
    from _checks.vf_scope import _result_operations

    path = probe(tmp_path, "consume(vf.custom_result(1.0, mask))")
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    root = tmp_path / "framework"
    parser = root / "python/pypto_pro/language/parser/_call_parser.py"
    parser.parent.mkdir(parents=True)
    source = (PYPTO_ROOT / parser.relative_to(root)).read_text()
    parser.write_text(source.replace('"full": 1,', '"full": 1, "custom_result": 1,'))
    found = check(path, pypto_root=root)
    assert len(found) == 1 and not found[0].advisory
    assert str(parser) in found[0].message
    parser.write_text(
        parser.read_text().replace('op_name != "bit_cast"', 'op_name != "other"')
    )
    _result_operations.cache_clear()
    found = check(path, pypto_root=root)
    assert len(found) == 1 and found[0].advisory
    found = check(path, pypto_root=tmp_path / "missing")
    assert len(found) == 1 and found[0].advisory


def test_nested_call_waiver_is_visible(tmp_path):
    path = probe(
        tmp_path,
        '"""反例：vf-result-nested-call —— parse-only diagnostic"""\n'
        "consume(vf.full(1.0, mask))",
    )
    path.write_text(path.read_text().replace("@pl.jit()", "@pl.vector_function"))
    assert check(path)[0].waived
