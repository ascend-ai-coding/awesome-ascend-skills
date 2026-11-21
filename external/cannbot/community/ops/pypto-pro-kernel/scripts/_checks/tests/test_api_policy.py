# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Behavioral controls for namespace selection and Vector authoring policy."""

import pytest

from _checks.api_policy import (
    SORT_OPS,
    TILE_RULE,
    VF_RULE,
    VF_UNAVAILABLE,
    LOOP_RULE,
    check,
    inspect,
)


def write(tmp_path, code):
    path = tmp_path / "probe.py"
    path.write_text(code)
    return path


@pytest.mark.parametrize("decorator", ["@pl.jit", "@pl.jit(arch='a5')", "@compiled(arch='a5')"])
def test_native_python_range_is_diagnosed_without_rejecting_factory_loops(tmp_path, decorator):
    source = write(tmp_path,
        "import pypto_pro.language as pl\n"
        "from pypto_pro.language import jit as compiled\n"
        "def factory():\n"
        "    for value in range(2):\n"
        "        pass\n"
        f"    {decorator}\n"
        "    def kernel():\n"
        "        for value in range(2):\n"
        "            pass\n"
        "        for value in pl.range(2):\n"
        "            pass\n"
        "    return kernel\n"
        "def host_again():\n"
        "    for value in range(2):\n"
        "        pass\n")
    findings = check(source, pypto_root=tmp_path / "unavailable")
    assert [(finding.rule, finding.lineno) for finding in findings] == [(LOOP_RULE, 8)]


@pytest.mark.parametrize("iterator,waived", [("range", True), ("pl.range", False)])
def test_native_loop_counterexample_claim_requires_a_real_violation(tmp_path, iterator, waived):
    source = write(tmp_path,
        "import pypto_pro.language as pl\n"
        "@pl.jit\n"
        "def kernel():\n"
        '    """反例：native-loop-iterator —— Intentional parser rejection."""\n'
        f"    for value in {iterator}(2):\n"
        "        pass\n")
    findings = check(source, pypto_root=tmp_path / "unavailable")
    assert len(findings) == 1
    assert findings[0].rule == LOOP_RULE
    assert findings[0].waived is waived


@pytest.mark.parametrize(
    "operation",
    [
        "add",
        "rsqrt",
        "sum",
        "maximum",
        "cast",
        "gather",
        "transpose",
        "quant",
        "fill_index",
    ],
)
def test_tile_vector_compute_is_rejected(tmp_path, operation):
    findings = check(
        write(tmp_path, f"import pypto_pro.language as pl\npl.{operation}(out, x)\n")
    )
    assert [finding.rule for finding in findings] == [TILE_RULE]
    assert findings[0].lineno == 2
    assert not findings[0].advisory


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize(
    "operation", ["add", "sub", "mul", "div", "max", "min", "expdif"]
)
def test_backend_broadcast_spellings_cannot_bypass_policy(tmp_path, axis, operation):
    source = write(
        tmp_path,
        "import pypto_pro.language as pl\n"
        f"pl.{axis}_expand_{operation}(out, tile, vector)\n",
    )
    findings = check(source, pypto_root=tmp_path / "unavailable")
    assert [finding.rule for finding in findings] == [TILE_RULE]


@pytest.mark.parametrize(
    "operation",
    sorted(SORT_OPS)
    + [
        "load",
        "move",
        "store",
        "fillpad",
        "getval",
        "setval",
        "min",
        "max",
        "astype",
        "gemm",
        "gemm_mx",
        "matmul",
    ],
)
def test_sort_and_non_vector_names_are_not_blocked(tmp_path, operation):
    assert not check(
        write(tmp_path, f"import pypto_pro.language as pl\npl.{operation}(a, b)\n")
    )


@pytest.mark.parametrize(
    "code",
    [
        "import pypto_pro.language as language\nlanguage.rsqrt(out, x)",
        "from pypto_pro.language import rsqrt as reciprocal_root\nreciprocal_root(out, x)",
        "import pypto_pro.language as pl\noperation = pl.rsqrt\noperation(out, x)",
        "import pypto_pro.language\npypto_pro.language.rsqrt(out, x)",
    ],
)
def test_import_and_function_aliases_do_not_hide_tile_compute(tmp_path, code):
    assert check(write(tmp_path, code))[0].rule == TILE_RULE


def test_namespaces_and_shadowed_names_are_distinct(tmp_path):
    path = write(
        tmp_path,
        """import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
def body(x, mask):
    a = vf.sqrt(x, mask)
    b = vf.div(x, a, mask)
    return b
def unrelated(pl, vf):
    pl.rsqrt(x)
    vf.rsqrt(x)
""",
    )
    assert not check(path)


@pytest.mark.parametrize(
    "declaration,call",
    [
        ("from pypto_pro.language import Vf as registers", "registers.rsqrt(x, mask)"),
        ("import pypto_pro.language as pl", "pl.Vf.rsqrt(x, mask)"),
        (
            "from pypto_pro.language import Vf as vf\noperation = vf.rsqrt",
            "operation(x, mask)",
        ),
    ],
)
def test_invented_vf_names_fail_against_current_source(tmp_path, declaration, call):
    found = check(write(tmp_path, declaration + "\n" + call + "\n"))
    assert [finding.rule for finding in found] == [VF_RULE]
    assert "Python=" in found[0].message and "C++=" in found[0].message


def test_vf_argument_annotation_identifies_the_namespace(tmp_path):
    path = write(
        tmp_path,
        """import pypto_pro.language as pl
@pl.vector_function
def body(registers: pl.Vf, x, mask):
    registers.rsqrt(x, mask)
""",
    )
    assert check(path)[0].rule == VF_RULE


def test_missing_source_reports_unverified_calls_without_guessing(tmp_path):
    path = write(
        tmp_path, "from pypto_pro.language import Vf as vf\nvf.full(1)\nvf.sqrt(x, m)\n"
    )
    findings, counts = inspect(path, pypto_root=tmp_path / "absent")
    assert len(findings) == 1 and findings[0].rule == VF_UNAVAILABLE
    assert findings[0].advisory
    assert counts == {"tile": 0, "vf": 2, "unverified": 2}


def test_source_extension_changes_membership_without_executing_source(tmp_path):
    stub = tmp_path / "python/pypto_pro/language/_vf_api.py"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        "raise RuntimeError('do not import')\nclass Vf:\n"
        "    def full(): pass\n    def load_align(): pass\n    def new_register_op(): pass\n"
    )
    registry = tmp_path / "framework/src/interface/ir/op/vf_ops.cpp"
    registry.parent.mkdir(parents=True)
    registry.write_text('REGISTER_OP("vf.full")\nREGISTER_OP("vf.load_align")\n')
    source = write(
        tmp_path, "from pypto_pro.language import Vf as vf\nvf.new_register_op()\n"
    )
    assert not check(source, pypto_root=tmp_path)


def test_source_declared_tile_operation_is_subject_to_policy(tmp_path):
    stub = tmp_path / "python/pypto_pro/language/_api.py"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        "raise RuntimeError('do not import')\ndef new_tile_op(out: Tile, src: Tile): pass\n"
    )
    source = write(
        tmp_path, "import pypto_pro.language as pl\npl.new_tile_op(out, src)\n"
    )
    assert check(source, pypto_root=tmp_path)[0].rule == TILE_RULE


def test_policy_cannot_be_waived_by_a_teaching_docstring(tmp_path):
    source = write(
        tmp_path,
        """import pypto_pro.language as pl
def bad():
    '''反例：tile-vector-compute-forbidden —— teaching exception'''
    pl.add(out, a, b)
""",
    )
    assert not check(source)[0].waived


@pytest.fixture
def macro_registry(tmp_path):
    stub = tmp_path / "python/pypto_pro/language/_vf_api.py"
    stub.parent.mkdir(parents=True)
    stub.write_text(
        "raise RuntimeError('source inspection must not import')\nclass Vf:\n"
        "    def full(): pass\n    def load_align(): pass\n"
    )
    registry = tmp_path / "framework/src/interface/ir/op/vf_ops.cpp"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        '#include "ir/reg_op.h"\nREGISTER_OP(VF_FULL)\n'
        'REGISTER_OP(VF_LOAD_ALIGN)\nREGISTER_OP(VF_CPP_ONLY)\n'
        '// REGISTER_OP(IGNORED_COMMENT)\n'
    )
    header = tmp_path / "framework/include/ir/reg_op.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        '#define VF_FULL "vf.full"\n#define VF_LOAD_ALIGN "vf.load_align"\n'
        '#define VF_CPP_ONLY "vf.cpp_only" // registered outside Python stub\n'
    )
    return tmp_path, registry, header


@pytest.mark.parametrize("operation,expected", [("cpp_only", []), ("subs", [VF_RULE])])
def test_macro_registry_checks_real_membership(macro_registry, operation, expected):
    root, _, _ = macro_registry
    source = write(root, f"from pypto_pro.language import Vf as vf\nvf.{operation}()\n")
    findings, counts = inspect(source, pypto_root=root)
    assert [item.rule for item in findings] == expected
    assert counts["unverified"] == 0
    if findings:
        assert "C++ macros=" in findings[0].message


@pytest.mark.parametrize("argument", ["VF_UNKNOWN", "MAKE_OP(VF_FULL)"])
def test_unresolved_macro_registry_does_not_claim_absence(macro_registry, argument):
    root, registry, _ = macro_registry
    registry.write_text(registry.read_text() + f"REGISTER_OP({argument})\n")
    source = write(root, "from pypto_pro.language import Vf as vf\nvf.subs()\n")
    findings, counts = inspect(source, pypto_root=root)
    assert [item.rule for item in findings] == [VF_UNAVAILABLE]
    assert findings[0].advisory and counts["unverified"] == 1
    assert "unresolved registrations=" in findings[0].message


@pytest.mark.parametrize("header_state", ["missing", "ambiguous"])
def test_incomplete_macro_definitions_remain_unverified(macro_registry, header_state):
    root, _, header = macro_registry
    if header_state == "missing":
        header.unlink()
    else:
        header.write_text(header.read_text() + '#define VF_CPP_ONLY "vf.other"\n')
    source = write(root, "from pypto_pro.language import Vf as vf\nvf.subs()\n")
    findings = check(source, pypto_root=root)
    assert [item.rule for item in findings] == [VF_UNAVAILABLE]
