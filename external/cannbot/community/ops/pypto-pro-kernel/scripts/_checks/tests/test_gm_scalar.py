# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks.gm_scalar import check
from _checks.cli import report_rules


def probe(tmp_path, body, annotation="pl.Tensor[[ROWS], pl.DT_FP32]"):
    path = tmp_path / "kernel.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        "@pl.jit()\n"
        + f"def kernel(out: {annotation}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


def test_direct_block_scalar_write_is_advisory(tmp_path):
    found = check(probe(tmp_path, "pl.setval(out, pl.get_block_idx(), 0.0)"))
    assert len(found) == 1 and found[0].advisory


def test_row_owned_side_output_with_branches(tmp_path):
    found = check(
        probe(
            tmp_path,
            "for row in pl.range(pl.get_block_idx(), ROWS, pl.get_block_num()):\n"
            "    if valid:\n        pl.setval(out, row, result)\n"
            "    else:\n        pl.setval(out, row, 0.0)",
        )
    )
    assert len(found) == 2


@pytest.mark.parametrize(
    "statement",
    [
        "pl.store(out, tile, [row])",
        "pl.setval(tile, row, 0.0)",
        "pl.setval(out, row * 16, 0.0)",
    ],
)
def test_other_transports_or_unknown_strides_are_not_claimed(tmp_path, statement):
    assert not check(
        probe(
            tmp_path,
            "for row in pl.range(pl.get_block_idx(), ROWS, pl.get_block_num()):\n    "
            + statement,
        )
    )


def test_tile_annotation_is_not_gm(tmp_path):
    assert not check(
        probe(tmp_path, "pl.setval(out, pl.get_block_idx(), 0.0)", annotation="pl.Tile")
    )


def test_rebound_output_is_unknown(tmp_path):
    assert not check(
        probe(tmp_path, "out = make_tile()\npl.setval(out, pl.get_block_idx(), 0.0)")
    )


def test_loop_index_rebound_inside_body_is_unknown(tmp_path):
    assert not check(
        probe(
            tmp_path,
            "for row in pl.range(pl.get_block_idx(), ROWS, pl.get_block_num()):\n"
            "    row = row * 16\n    pl.setval(out, row, 0.0)",
        )
    )


def test_index_from_ordinary_loop_is_unknown(tmp_path):
    assert not check(
        probe(tmp_path, "for row in pl.range(ROWS):\n    pl.setval(out, row, 0.0)")
    )


def test_keyword_api_arguments(tmp_path):
    assert (
        len(
            check(
                probe(
                    tmp_path,
                    "pl.setval(container=out, offset=pl.get_block_idx(), value=0.0)",
                )
            )
        )
        == 1
    )


def test_import_alias(tmp_path):
    path = probe(tmp_path, "pl.setval(out, pl.get_block_idx(), 0.0)")
    path.write_text(
        path.read_text().replace("as pl", "as language").replace("pl.", "language.")
    )
    assert len(check(path)) == 1


def test_nested_non_kernel_function_is_not_inherited(tmp_path):
    assert not check(
        probe(tmp_path, "def helper():\n    pl.setval(out, pl.get_block_idx(), 0.0)")
    )


def test_explicit_diagnostic_claim_is_visible(tmp_path):
    path = probe(
        tmp_path,
        '"""反例：gm-scalar-dense-output —— transport diagnostic"""\n'
        "pl.setval(out, pl.get_block_idx(), 0.0)",
    )
    assert check(path)[0].waived


def test_python_function_is_outside_the_rule(tmp_path):
    path = probe(tmp_path, "pl.setval(out, pl.get_block_idx(), 0.0)")
    path.write_text(path.read_text().replace("@pl.jit()\n", ""))
    assert not check(path)


def test_other_rank_and_incomplete_annotation_are_unknown(tmp_path):
    for annotation in ["pl.Tensor[[ROWS,1],pl.DT_FP32]", "pl.Tensor[()]"]:
        assert not check(
            probe(tmp_path, "pl.setval(out, pl.get_block_idx(), 0.0)", annotation)
        )


def test_advisory_does_not_fail_cli(tmp_path, capsys):
    path = probe(
        tmp_path,
        "pl.setval(out, pl.get_block_idx(), 0.0)",
        annotation="pl.Tensor[[128], pl.DT_FP32]",
    )
    path.write_text('"""写法：原始。Scalar write review case."""\n' + path.read_text())
    assert report_rules([path])
    output = capsys.readouterr().out
    assert "[需核查] gm-scalar-dense-output" in output
    assert "1 处需核查提示" in output
