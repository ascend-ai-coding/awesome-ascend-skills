# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Connected reviews preserve legal scalar consumers, aliases and unknowns."""

import pytest

from _checks import vf_dataflow

PREFIX = """import pypto_pro.language as pl
from pypto_pro.language import Vf as v
@pl.vector_function
def row(src, dst):
    m = v.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    x = v.load_align(src, [0,0])
    wide = v.astype(x, m, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    total = v.reduce_sum(wide, m)
"""


def inspect(tmp_path, body, prefix=PREFIX):
    path = tmp_path / "kernel.py"
    path.write_text(prefix + body)
    return vf_dataflow.inspect(path)


def test_connected_conversion_and_reduction_are_reviewed_together(tmp_path):
    findings, reviews, coverage = inspect(
        tmp_path, "    y = v.sub(wide, total, m)\n    z = v.mul(wide, total, m)\n"
    )
    assert {f.rule for f in findings} == {
        vf_dataflow.REDUCTION_RULE,
        vf_dataflow.CHAIN_RULE,
    }
    assert all(f.advisory for f in findings) and len(reviews) == 1
    assert {site["operation"] for site in reviews[0]["sites"]} >= {
        "load_align",
        "astype",
        "reduce_sum",
        "sub",
    }
    assert coverage["checked_consumers"] == 2


@pytest.mark.parametrize(
    "body",
    [
        "    total = v.full(total, m)\n    y = v.sub(wide, total, m)\n",
        "    one = v.update_mask(1, dtype=pl.DT_FP32)\n    y = v.sub(wide, total, one)\n",
        "    one = v.create_mask(pattern=pl.MaskPattern.VL1, dtype=pl.DT_FP32)\n    y = v.sub(wide, total, one)\n",
        "    empty = v.update_mask(0, dtype=pl.DT_FP32)\n    y = v.sub(wide, total, empty)\n",
        "    other = v.reduce_sum(wide, m)\n    combined = v.add(total, other, m)\n"
        "    combined = v.full(combined, m)\n    y = v.sub(wide, combined, m)\n",
        "    v.store_align(dst, total, m)\n",
    ],
)
def test_legal_consumers_do_not_receive_the_review(tmp_path, body):
    assert inspect(tmp_path, body)[0] == []


def test_unpk_path_has_only_the_reduction_review(tmp_path):
    prefix = PREFIX.replace(
        "v.load_align(src, [0,0])", "v.load_align(src, [0,0], dist=pl.LoadDist.UNPK)"
    )
    findings, _, _ = inspect(tmp_path, "    y = v.sub(wide, total, m)\n", prefix)
    assert [f.rule for f in findings] == [vf_dataflow.REDUCTION_RULE]


@pytest.mark.parametrize(
    "prefix",
    [
        PREFIX.replace(
            "v.reduce_sum(wide, m)", "v.reduce_sum(wide, m, datablock=True)"
        ),
        PREFIX.replace("v.reduce_sum(wide, m)", "v.reduce_sum(wide, m, **options)"),
        PREFIX.replace(
            "    total = v.reduce_sum(wide, m)",
            "    if flag:\n        total = v.reduce_sum(wide, m)",
        ),
        PREFIX.replace("    m =", "    v = src\n    m ="),
    ],
)
def test_datablock_control_flow_and_shadowing_remain_unknown(tmp_path, prefix):
    assert inspect(tmp_path, "    y = v.sub(wide, total, m)\n", prefix)[0] == []


def test_reduction_alias_copy_retains_lane_provenance(tmp_path):
    findings, _, _ = inspect(
        tmp_path,
        "    alias = total\n    copy = v.move(alias, m)\n    y = v.sub(wide, copy, m)\n",
    )
    assert vf_dataflow.REDUCTION_RULE in {f.rule for f in findings}


WIDENING_PREFIX = """import pypto_pro.language as pl
from pypto_pro.language import Vf as v
FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
@pl.vector_function
def row(src, dst, n, other):
    compute = v.update_mask(n, dtype=FP32)
    convert = v.update_mask(n, dtype=FP16)
    packed = v.load_align(src, [0,0], dist=pl.LoadDist.UNPK)
    wide = v.astype(packed, convert, dtype=FP32, layout=pl.CastLayout.ZERO)
"""


def widening(
    tmp_path, prefix=WIDENING_PREFIX, body="    out = v.muls(wide, 2.0, compute)\n"
):
    findings, reviews, coverage = inspect(tmp_path, body, prefix)
    return (
        [f for f in findings if f.rule == vf_dataflow.WIDENING_RULE],
        reviews,
        coverage,
    )


def test_same_dynamic_count_reviews_the_connected_consumer(tmp_path):
    findings, reviews, _ = widening(
        tmp_path,
        body="    alias = wide\n    out = v.muls(alias, 2.0, compute)\n    v.store_align(dst, wide, compute)\n",
    )
    assert len(findings) == 1 and findings[0].advisory
    assert reviews[0]["converted_lanes"] is None
    assert [s["operation"] for s in reviews[0]["sites"]] == [
        "load_align",
        "astype",
        "muls",
    ]


@pytest.mark.parametrize(
    "count,converted,needed",
    [(2, 1, 2), (16, 8, 16), (32, 16, 32), (64, 32, 64), (126, 63, 64)],
)
def test_known_counts_report_physical_undercoverage(tmp_path, count, converted, needed):
    prefix = WIDENING_PREFIX.replace("update_mask(n,", f"update_mask({count},")
    findings, reviews, _ = widening(tmp_path, prefix)
    assert len(findings) == 1
    assert (reviews[0]["converted_lanes"], reviews[0]["consumer_lanes"]) == (
        converted,
        needed,
    )


@pytest.mark.parametrize("count", [0, 1, 127, 128, 1000, -1, 4294967296, 4294967297])
def test_saturated_empty_single_lane_and_wrapped_counts_are_legal(tmp_path, count):
    prefix = WIDENING_PREFIX.replace("    compute =", f"    n = {count}\n    compute =")
    assert widening(tmp_path, prefix)[0] == []


@pytest.mark.parametrize(
    "change",
    [
        ("v.astype(packed, convert,", "v.astype(packed, compute,"),
        ("v.update_mask(n, dtype=FP16)", "v.update_mask(2*n, dtype=FP16)"),
        (
            "v.update_mask(n, dtype=FP16)",
            "v.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)",
        ),
        ("v.update_mask(n, dtype=FP16)", "v.update_mask(other, dtype=FP16)"),
        ("pl.LoadDist.UNPK", "pl.LoadDist.UNPK_B8"),
        ("pl.CastLayout.ZERO", "pl.CastLayout.ONE"),
        ("    convert =", "    n = other\n    convert ="),
        ("    convert =", "    n += 1\n    convert ="),
        ("    convert =", "    if flag:\n        n = other\n    convert ="),
        ("    convert =", "    FP16 = other\n    convert ="),
        ("    compute =", "    pl = other\n    compute ="),
        ("    wide =", "    if flag:\n        wide ="),
        ("v.load_align(src, [0,0], dist=pl.LoadDist.UNPK)", "helper(src)"),
        ("    compute =", "    v = other\n    compute ="),
    ],
)
def test_legal_or_unresolved_conversion_edges_are_not_rejected(tmp_path, change):
    assert widening(tmp_path, WIDENING_PREFIX.replace(*change))[0] == []


@pytest.mark.parametrize(
    "consumer",
    [
        "out = v.add(wide, wide, compute)",
        "out = v.reduce_sum(wide, compute)",
        "v.store_align(dst, wide, compute)",
        "out = v.exp(wide, preg=compute)",
    ],
)
def test_direct_lane_consumers_are_reviewed(tmp_path, consumer):
    assert len(widening(tmp_path, body="    " + consumer + "\n")[0]) == 1


def test_recast_or_broadcast_is_not_a_direct_unpk_edge(tmp_path):
    body = (
        "    again = v.astype(wide, convert, dtype=FP32, layout=pl.CastLayout.ZERO)\n  "
        "  out = v.muls(again, 2.0, compute)\n"
    )
    assert widening(tmp_path, body=body)[0] == []
    assert widening(tmp_path, body="    out = v.full(wide, compute)\n")[0] == []


def test_widening_review_routes_to_owner_without_blocking_the_runner(tmp_path):
    import json
    import subprocess
    import sys
    from _checks.paths import SKILL_ROOT

    path = tmp_path / "kernel.py"
    path.write_text(
        '"""写法：原始。Direct widening consumer."""\n'
        + WIDENING_PREFIX
        + "    out = v.muls(wide, 2.0, compute)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            "print('executed')",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 0 and report["command"]["started"]
    assert [row["rule"] for row in report["diagnostics"]] == [vf_dataflow.WIDENING_RULE]
    assert report["diagnostics"][0]["severity"] == "advisory"
    assert [row["id"] for row in report["context"]] == ["vf-fp16-widening"]
    assert report["dataflow_reviews"][0]["source_sha256"]
    assert report["validation"]["numerical"] == "not_established"


@pytest.mark.parametrize("parameter", ["FP16", "pl"])
def test_enclosing_parameter_shadowing_leaves_mask_granularity_unknown(
    tmp_path, parameter
):
    before, function = WIDENING_PREFIX.split("@pl.vector_function", 1)
    prefix = (
        before
        + f"def factory({parameter}):\n"
        + "\n".join(
            "    " + line for line in ("@pl.vector_function" + function).splitlines()
        )
        + "\n"
    )
    assert (
        widening(tmp_path, prefix, body="        out = v.muls(wide, 2.0, compute)\n")[0]
        == []
    )


def test_reimported_dtype_alias_is_not_a_known_mask_width(tmp_path):
    prefix = WIDENING_PREFIX.replace(
        "@pl.vector_function", "from other import FP16\n@pl.vector_function"
    )
    assert widening(tmp_path, prefix)[0] == []
