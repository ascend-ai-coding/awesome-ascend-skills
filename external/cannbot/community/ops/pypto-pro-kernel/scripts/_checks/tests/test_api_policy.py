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
    check,
    inspect,
)


def write(tmp_path, code):
    path = tmp_path / "probe.py"
    path.write_text(code)
    return path


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
