# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks import register_dtype
from _checks.cli import report_rules


def probe(tmp_path, body, *, module="", params="tile, mask"):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        "from pypto_pro.language import Vf as vf\n"
        + module
        + f"\n@pl.vector_function\ndef body({params}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "first",
    ["full(0, mask", "arange(0", "unpack(hist", "astype(hist, mask", "bit_cast(hist"],
)
def test_explicit_unsigned_to_signed_destination_reuse(tmp_path, first):
    path = probe(
        tmp_path,
        f"r = vf.{first}, dtype=U32)\nr = vf.bit_cast(r, dtype=I32)",
        module="U32 = pl.DT_UINT32\nI32 = pl.DT_INT32\n",
    )
    found = register_dtype.check(path)
    assert len(found) == 1
    assert not found[0].advisory
    assert "DT_UINT32" in found[0].message and "DT_INT32" in found[0].message


def test_new_name_for_conversion_is_valid(tmp_path):
    path = probe(
        tmp_path,
        "u = vf.unpack(hist, dtype=pl.DT_UINT32)\n"
        "i = vf.bit_cast(u, dtype=pl.DT_INT32)",
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize("target", ["pl.DT_FP16", "pl.DT_BF16"])
@pytest.mark.parametrize("layout", ["", ", layout=pl.CastLayout.ZERO"])
def test_half_width_zero_cast_raw_store_is_advisory(tmp_path, target, layout):
    path = probe(
        tmp_path,
        "p32 = vf.create_mask(dtype=pl.DT_FP32)\n"
        "for row in pl.range(4):\n"
        f"    half = vf.astype(value, p32, dtype={target}{layout})\n"
        "    vf.store_align(tile, half, mask, [row, 0])",
        params="tile, mask, value",
    )
    found = register_dtype.check(path)
    assert len(found) == 1 and found[0].rule == register_dtype.NARROW_STORE_RULE
    assert found[0].advisory
    assert report_rules([path])


@pytest.mark.parametrize(
    "finish",
    [
        "vf.store_align(tile, half, p32, 0, dist=pl.StoreDist.PACK_B32)",
        "packed, spare = vf.de_interleave(half, half)\nvf.store_align(tile, packed, mask, 0)",
        "vf.store_align(tile, half, mask, 0, dist=chosen_dist)",
    ],
)
def test_packed_reordered_and_unknown_store_forms_remain_legal(tmp_path, finish):
    path = probe(
        tmp_path,
        "p32 = vf.create_mask(dtype=pl.DT_FP32)\n"
        "half = vf.astype(value, p32, dtype=pl.DT_FP16)\n" + finish,
    )
    assert not register_dtype.check(path)


@pytest.mark.parametrize("dtype", ["pl.DT_FP16", "unknown_dtype"])
def test_predicate_without_known_32_bit_width_is_not_inferred(tmp_path, dtype):
    path = probe(
        tmp_path,
        f"p = vf.create_mask(dtype={dtype})\n"
        "half = vf.astype(value, p, dtype=pl.DT_FP16)\n"
        "vf.store_align(tile, half, mask, 0)",
    )
    assert not register_dtype.check(path)


def test_narrow_store_aliases_and_unknown_rebindings(tmp_path):
    path = probe(
        tmp_path,
        "p = vf.update_mask(64, dtype=F32)\n"
        "half = vf.astype(value, p, dtype=F16)\n"
        "vf.store_align(tile, half, mask, 0)",
        module="F32 = pl.DT_FP32\nF16 = pl.DT_FP16\n",
    )
    path.write_text(
        path.read_text().replace("as vf", "as registers").replace("vf.", "registers.")
    )
    assert len(register_dtype.check(path)) == 1
    path.write_text(
        path.read_text().replace(
            "    registers.store_align", "    half = other\n    registers.store_align"
        )
    )
    assert not register_dtype.check(path)


def test_intentional_raw_container_probe_can_be_waived(tmp_path):
    path = probe(
        tmp_path,
        '"""反例：vf-narrow-store-layout —— inspect physical cast container"""\n'
        "p32 = vf.create_mask(dtype=pl.DT_FP32)\n"
        "half = vf.astype(value, p32, dtype=pl.DT_FP16)\n"
        "vf.store_align(tile, half, mask, 0)",
    )
    assert register_dtype.check(path)[0].waived


@pytest.mark.parametrize(
    "setup,params",
    [
        ("", "tile, mask, index"),
        ("index = pl.max(count - 1, 0)\n", "tile, mask, count"),
    ],
)
def test_scalar_self_broadcast_is_an_advisory(tmp_path, setup, params):
    path = probe(
        tmp_path,
        setup + "index = vf.full(index, mask, dtype=pl.DT_UINT32)",
        params=params,
    )
    findings = register_dtype.check(path)
    assert len(findings) == 1
    assert findings[0].rule == register_dtype.BROADCAST_RULE
    assert findings[0].advisory
    assert findings[0].lineno == 6 + bool(setup)


def test_fresh_broadcast_destination_preserves_scalar_binding(tmp_path):
    path = probe(
        tmp_path,
        "indices = vf.full(index, mask, dtype=pl.DT_UINT32)",
        params="tile, mask, index",
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize(
    "setup",
    [
        "index = vf.load_align(tile, [0, 0])\n",
        "source = vf.load_align(tile, [0, 0])\nindex = source\n",
        "index = vf.full(3, mask, dtype=pl.DT_UINT32)\n",
    ],
)
def test_initialized_register_self_broadcast_is_legal(tmp_path, setup):
    path = probe(tmp_path, setup + "index = vf.full(index, mask, dtype=pl.DT_UINT32)")
    assert register_dtype.check(path) == []


def test_broadcast_rule_does_not_infer_helper_return_types(tmp_path):
    path = probe(
        tmp_path, "index = helper()\nindex = vf.full(index, mask, dtype=pl.DT_UINT32)"
    )
    findings = register_dtype.check(path)
    assert len(findings) == 1 and findings[0].advisory


def test_broadcast_rule_respects_shadowed_namespace(tmp_path):
    path = probe(
        tmp_path,
        "index = vf.full(index, mask, dtype=pl.DT_UINT32)",
        params="tile, mask, index, vf",
    )
    assert register_dtype.check(path) == []


def test_alias_and_literal_identify_the_same_type(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=I32)\nr = vf.full(1, dtype=pl.DT_INT32)",
        module="I32 = pl.DT_INT32\n",
    )
    assert register_dtype.check(path) == []


def test_error_does_not_change_the_original_destination_type(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=pl.DT_FP32)\n"
        "r = vf.astype(r, mask, dtype=pl.DT_INT32)\n"
        "r = vf.full(1, dtype=pl.DT_FP32)",
    )
    assert len(register_dtype.check(path)) == 1


def test_ordinary_alias_carries_the_view_type(tmp_path):
    path = probe(
        tmp_path,
        "a = vf.full(0, dtype=pl.DT_FP32)\nb = a\n"
        "b = vf.astype(b, mask, dtype=pl.DT_INT32)",
    )
    assert len(register_dtype.check(path)) == 1


def test_ordinary_rebinding_to_another_register_is_not_a_typed_write(tmp_path):
    path = probe(
        tmp_path,
        "a = vf.full(0, dtype=pl.DT_FP32)\n"
        "b = vf.full(0, dtype=pl.DT_INT32)\na = b\n"
        "a = vf.full(1, dtype=pl.DT_INT32)",
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize(
    "prefix", ["D = helper()", "D = pl.DT_FP32 if flag else pl.DT_INT32"]
)
def test_unknown_local_dtype_is_not_inferred(tmp_path, prefix):
    path = probe(
        tmp_path,
        prefix + "\nr = vf.full(0, dtype=D)\nr = vf.full(0, dtype=pl.DT_INT32)",
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize(
    "rebind", ["D = pl.DT_INT32\n", "if flag:\n    D = pl.DT_INT32\n"]
)
def test_reassigned_module_alias_is_unknown(tmp_path, rebind):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=D)\nr = vf.full(1, dtype=pl.DT_INT32)",
        module="D = pl.DT_FP32\n" + rebind,
    )
    assert register_dtype.check(path) == []


def test_parameter_shadows_module_alias(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=D)\nr = vf.full(1, dtype=pl.DT_INT32)",
        module="D = pl.DT_FP32\n",
        params="tile, mask, D",
    )
    assert register_dtype.check(path) == []


def test_closure_binding_is_not_replaced_with_module_alias(tmp_path):
    path = tmp_path / "closure.py"
    path.write_text(
        "import pypto_pro.language as pl\nfrom pypto_pro.language import Vf as vf\n"
        "D = pl.DT_FP32\ndef factory(D):\n"
        "    @pl.vector_function\n    def body(tile):\n"
        "        r = vf.full(0, dtype=D)\n        r = vf.full(1, dtype=pl.DT_INT32)\n"
        "    return body\n"
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize(
    "wrapper", ["if flag:", "for i in pl.range(2):", "with scope:"]
)
def test_checks_straight_line_code_inside_blocks(tmp_path, wrapper):
    path = probe(
        tmp_path,
        wrapper + "\n    r = vf.full(0, dtype=pl.DT_UINT32)\n"
        "    r = vf.bit_cast(r, dtype=pl.DT_INT32)",
    )
    assert len(register_dtype.check(path)) == 1


def test_no_type_inference_through_branch_joins(tmp_path):
    path = probe(
        tmp_path,
        "if flag:\n    r = vf.full(0, dtype=pl.DT_FP32)\n"
        "else:\n    r = vf.full(0, dtype=pl.DT_INT32)\n"
        "r = vf.full(1, dtype=pl.DT_INT32)",
    )
    assert register_dtype.check(path) == []


@pytest.mark.parametrize(
    "replacement",
    ["r = helper()", "r, other = helper()", "r = vf.unknown_api(dtype=pl.DT_INT32)"],
)
def test_unknown_rebinding_invalidates_type(tmp_path, replacement):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=pl.DT_FP32)\n"
        + replacement
        + "\nr = vf.full(1, dtype=pl.DT_INT32)",
    )
    assert register_dtype.check(path) == []


def test_predicate_granularity_is_not_a_data_register_type(tmp_path):
    path = probe(
        tmp_path,
        "p = vf.create_mask(dtype=pl.DT_FP32)\n"
        "p = vf.update_mask(3, dtype=pl.DT_UINT8)",
    )
    assert register_dtype.check(path) == []


def test_predicate_pack_unpack_dtype_keywords_do_not_change_cpp_type(tmp_path):
    path = probe(
        tmp_path,
        "mask = vf.create_mask(dtype=pl.DT_UINT32)\n"
        "p = vf.pack(mask, dtype=pl.DT_UINT16)\n"
        "p = vf.unpack(p, dtype=pl.DT_UINT32)",
    )
    assert register_dtype.check(path) == []


def test_pack_unpack_of_unknown_register_kind_is_not_rejected(tmp_path):
    path = probe(
        tmp_path,
        "p = vf.pack(mask, dtype=pl.DT_UINT16)\n"
        "alias = p\n"
        "alias = vf.unpack(alias, dtype=pl.DT_UINT32)",
    )
    assert register_dtype.check(path) == []


def test_pack_unpack_of_known_data_register_retains_destination_type(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(1, mask, dtype=pl.DT_UINT32)\n"
        "p = vf.pack(r, dtype=pl.DT_UINT16)\n"
        "p = vf.unpack(p, dtype=pl.DT_UINT32)",
    )
    assert len(register_dtype.check(path)) == 1


def test_bool_shared_carrier_is_outside_this_rule(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(True, dtype=pl.DT_BOOL)\nr = vf.bit_cast(r, dtype=pl.DT_UINT8)",
    )
    assert register_dtype.check(path) == []


def test_normal_python_function_is_outside_this_rule(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=pl.DT_UINT32)\nr = vf.bit_cast(r, dtype=pl.DT_INT32)",
    )
    path.write_text(path.read_text().replace("@pl.vector_function\n", ""))
    assert register_dtype.check(path) == []


def test_function_scopes_do_not_share_register_names(tmp_path):
    path = probe(tmp_path, "r = vf.full(0, dtype=pl.DT_UINT32)")
    path.write_text(
        path.read_text() + "\n@pl.vector_function\ndef other(tile):\n"
        "    r = vf.full(0, dtype=pl.DT_INT32)\n"
    )
    assert register_dtype.check(path) == []


def test_language_and_vf_import_aliases(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=pl.DT_UINT32)\nr = vf.bit_cast(r, dtype=pl.DT_INT32)",
    )
    path.write_text(
        path.read_text()
        .replace("as pl", "as lang")
        .replace("pl.", "lang.")
        .replace("as vf", "as ops")
        .replace("vf.", "ops.")
    )
    assert len(register_dtype.check(path)) == 1


def test_global_write_invalidates_a_dtype_alias(tmp_path):
    path = probe(
        tmp_path,
        "r = vf.full(0, dtype=D)\nr = vf.full(1, dtype=pl.DT_INT32)",
        module="D = pl.DT_FP32\ndef change():\n    global D\n    D = pl.DT_INT32\n",
    )
    assert register_dtype.check(path) == []


def test_imported_dtype_and_annotated_destination(tmp_path):
    path = probe(
        tmp_path,
        "r: object = vf.full(0, dtype=U32)\nr = vf.bit_cast(r, dtype=pl.DT_INT32)",
        module="from pypto_pro.language import DT_UINT32 as U32\n",
    )
    assert len(register_dtype.check(path)) == 1


def test_cli_fails_and_explicit_counterexample_can_be_waived(tmp_path, capsys):
    body = "r = vf.full(0, dtype=pl.DT_UINT32)\nr = vf.bit_cast(r, dtype=pl.DT_INT32)"
    path = probe(tmp_path, body)
    assert not report_rules([path])
    assert "vf-register-dtype-change" in capsys.readouterr().out
    path = probe(
        tmp_path,
        '"""反例：vf-register-dtype-change —— compile-only negative example"""\n'
        + body,
    )
    assert register_dtype.check(path)[0].waived
