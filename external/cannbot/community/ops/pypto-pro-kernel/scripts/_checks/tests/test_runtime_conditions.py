# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Diagnostic bodies need their observed phase and an unambiguous source site."""

import pytest

from _checks import runtime_diagnostics


@pytest.mark.parametrize(
    "body,rule,topic",
    [
        (
            "Unsupported type annotation: FP32",
            "jit-annotation-type",
            "jit-annotation-types",
        ),
        (
            "Incomplete type annotation: pl.Scalar",
            "jit-annotation-type",
            "jit-annotation-types",
        ),
        ("Expected a native @pl.jit Pro kernel", "native-entry-kind", "jit-entry-kind"),
    ],
)
def test_framework_head_routes_actual_body_and_keeps_identity(body, rule, topic):
    message = f"[parser.py:8][PRO_PARSER]:ErrCode: F00001! Enum: ExternalError::INVALID_TYPE. {body}"
    found, context = runtime_diagnostics.from_output(
        f"RuntimeError: {message}\n  --> candidate.py:14:3\n"
    )
    assert found[0]["message"] == message and found[0]["rule"] == rule
    assert found[0]["evidence"] == {
        "module": "PRO_PARSER",
        "error_code": "F00001",
        "origin": "output_tail",
        "stage": "unknown",
    }
    assert found[0]["location"]["line"] == 14
    assert [item["id"] for item in context] == [topic]


@pytest.mark.parametrize("stage", ["argument_binding", "entry_import"])
@pytest.mark.parametrize(
    "body",
    [
        "Unsupported type annotation: FP32",
        "move: NZ destination uses align16 rows",
        "Expected a native @pl.jit Pro kernel",
    ],
)
def test_preparse_phase_does_not_infer_kernel_body_cause(stage, body):
    found, context = runtime_diagnostics.from_exception(RuntimeError(body), stage=stage)
    assert found[0]["rule"] == "native-stage-failed" and found[0]["message"] == body
    assert found[0]["evidence"]["stage"] == stage
    assert found[0]["evidence"]["exception"] == "RuntimeError"
    assert not context


@pytest.mark.parametrize(
    "body",
    [
        'not make_tile_group() depth is required when mutex_ids is "auto", None, or'
        " empty",
        "vf.add requires dst, src0, src1 to have the same type, got dst=DT_FP16"
        " src0=DT_FP16 src1=DT_FP32 (quoted control)",
        "host move: NZ destination uses align16 rows",
        "Expected a native @pl.jit Pro kernel (example text)",
    ],
)
def test_incidental_or_changed_diagnostic_body_is_not_a_proven_recipe(body):
    found, context = runtime_diagnostics.from_exception(
        RuntimeError(body), stage="native_parse"
    )
    assert found[0]["rule"] == "native-rejection" and not context


def site(*, line=8, api="move", src="Vec", dst="Vec", rows=49):
    return {
        "file": "candidate.py",
        "line": line,
        "api": api,
        "source": {"space": src, "layout": "ND"},
        "destination": {"space": dst, "layout": "NZ", "shape": [rows, 64]},
    }


def explain(sites, line=8):
    return runtime_diagnostics.from_output(
        f"RuntimeError: move: NZ destination uses align16 rows\n  --> candidate.py:{line}:0\n",
        transfers=sites,
    )


def test_pitch_diagnosis_requires_the_matched_vec_to_vec_non_aligned_descriptor():
    found, context = explain([site()])
    assert found[0]["rule"] == "ub-nz-move-pitch"
    assert found[0]["evidence"]["condition_status"] == "source_site_matched"
    assert [item["id"] for item in context] == ["ub-l1"]


def test_repeated_same_message_keeps_each_occurrences_source_location():
    text = "".join(
        f"RuntimeError: move: NZ destination uses align16 rows\n  --> candidate.py:{line}:0\n"
        for line in (8, 12)
    )
    found, _ = runtime_diagnostics.from_output(
        text, transfers=[site(line=8, dst="Mat"), site(line=12)]
    )
    assert [(item["location"]["line"], item["rule"]) for item in found] == [
        (8, "native-transfer-contract"),
        (12, "ub-nz-move-pitch"),
    ]


def test_store_offset_recipe_matches_the_actual_parser_diagnostic_only():
    message = (
        "vf.store_align (DATA_BLOCK_COPY) does not take an explicit offset: the vsstb "
        "instruction has no address operand, so the offset would be miscompiled into the block_stride field"
    )
    found, context = runtime_diagnostics.from_output("RuntimeError: " + message)
    assert found[0]["rule"] == "vf-nz-store-contract" and context[0]["id"] == "ub-l1"
    found, context = runtime_diagnostics.from_exception(
        RuntimeError("vf.store_align DATA_BLOCK_COPY unsupported dtype"),
        stage="native_parse",
    )
    assert found[0]["rule"] == "native-rejection" and not context


@pytest.mark.parametrize(
    "sites",
    [
        [],
        [site(line=9)],
        [site(rows=48)],
        [site(rows=None)],
        [site(dst="Mat")],
        [site(src="Acc")],
        [site(api="insert")],
        [site(), site(rows=48)],
    ],
)
def test_unknown_legal_other_path_or_ambiguous_sites_do_not_inherit_pitch_rejection(
    sites,
):
    found, _ = explain(sites)
    assert found[0]["rule"] == "native-transfer-contract"
    if len(sites) != 1 or sites[0]["line"] != 8 or sites[0]["api"] != "move":
        assert found[0]["evidence"]["condition_status"] == "unverified"
