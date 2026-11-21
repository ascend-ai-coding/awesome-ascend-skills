# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Scenario retrieval and runner feedback are source-bound and selective."""

import json
import subprocess
import sys

import pytest

from _checks import query, runtime_diagnostics, feedback
from _checks.paths import SKILL_ROOT

ENTRY = SKILL_ROOT / "scripts/check.py"


def test_source_identity_uses_current_runtime_entry_and_detects_missing_files(tmp_path):
    for name in feedback.FRAMEWORK_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# declaration fixture\n")
    assert not (tmp_path / "python/pypto_pro/language/parser/_op_pipeline.py").exists()
    assert feedback.identity(tmp_path)["selected_source"]["complete"]
    (tmp_path / "python/pypto_pro/runtime/jit.py").unlink()
    assert not feedback.identity(tmp_path)["selected_source"]["complete"]


@pytest.mark.parametrize(
    "text", ["UB to L1", "从UB向L1搬运", "把 VF 结果交给 Cube", "VF output to GEMM"]
)
def test_query_returns_owner_and_complete_native_example(text):
    result = query.report(text)
    assert result["status"] == "passed"
    assert result["context"][0]["id"] == "ub-l1"
    assert result["context"][0]["reference"]["sha256"]
    example = result["context"][0]["examples"][0]
    assert "pl.insert(" in example["text"] and "pl.gemm(" in example["text"]
    assert "set_cross_core" in example["text"] and "wait_cross_core" in example["text"]
    assert len(result["signatures"]) == 2
    assert result["validation"]["numerical"] == "not_established"


def test_exact_other_symbol_does_not_inject_ub_recipe():
    result = query.report("vf.add")
    assert result["signatures"] and not result["context"]


def test_padding_discovery_connects_the_protocol_without_claiming_a_fill():
    selected = query.report("fillpad")
    (context,) = selected["context"]
    assert context["id"] == "fillpad-execution"
    assert "three pieces" in context["text"] and "TilePad" in context["text"]
    assert selected["validation"]["numerical"] == "not_established"
    candidate = query.report("UB tile valid_shape fillpad")
    route = next(
        route
        for route in candidate["related_contracts"]
        if route["id"] == "fillpad-execution"
    )
    assert route["when"] and not candidate["context"]
    contract = next(
        row
        for row in candidate["knowledge"]["items"]
        if row["id"] == "contract:fillpad-execution"
    )
    assert contract["missing_facts"] and contract["reference"]["sha256"]
    setter = query.report("pl.set_validshape")
    assert not setter["context"]
    assert any(
        route["id"] == "fillpad-execution" for route in setter["related_contracts"]
    )


def test_simt_atomic_query_preserves_the_outer_entry_condition():
    result = query.report("pl.simt.atomic_add")
    route = next(
        route
        for route in result["related_contracts"]
        if route["id"] == "simt-entry-annotations"
    )
    assert route["when"] and route["reference"]["sha256"] and not result["context"]
    selected = query.report("SIMT outer wrapper")
    (context,) = selected["context"]
    assert context["id"] == "simt-entry-annotations"
    assert "outer" in context["text"] and "may be inferred" in context["text"]
    assert "simt_atomic/kernel.py" in context["examples"][0]["path"]
    assert selected["validation"]["numerical"] == "not_established"


@pytest.mark.parametrize(
    "text",
    [
        "Cannot resolve shape policy for tensor parameter 'weights'",
        "SIMT factory closure",
    ],
)
def test_simt_annotation_lookup_returns_captured_dimensions_without_rejecting_annotations(
    text,
):
    result = query.report(text)
    (context,) = result["context"]
    assert context["id"] == "simt-entry-annotations"
    assert "__closure__" in context["text"] and "def body(x, out)" in context["text"]
    assert "body also used the sizes" in context["text"]
    assert "omitted annotations as well as Tensor/Scalar annotations" in context["text"]
    contract = next(
        row
        for row in result["knowledge"]["items"]
        if row["id"] == "contract:simt-entry-annotations"
    )
    assert context["reference"]["sha256"] and contract["missing_facts"]
    assert result["validation"]["compilation"] == "not_established"


def test_general_simt_scalar_query_does_not_select_the_annotation_recipe():
    result = query.report("pl.simt.rint")
    assert "simt-entry-annotations" not in {row["id"] for row in result["context"]}


@pytest.mark.parametrize(
    "text",
    [
        "simt.launch Tensor argument must have a static shape",
        "SIMT physical shapes",
    ],
)
def test_simt_shape_query_distinguishes_physical_dimensions_from_runtime_traversal(
    text,
):
    result = query.report(text)
    (context,) = result["context"]
    assert context["id"] == "simt-physical-shapes"
    assert "pl.Tensor[[pl.STATIC]" in context["text"]
    assert "Runtime indexing, loop bounds, and logical valid counts" in context["text"]
    assert result["validation"]["compilation"] == "not_established"


def test_runtime_loop_query_does_not_select_a_static_physical_shape_recipe():
    result = query.report("pl.range")
    assert "simt-physical-shapes" not in {row["id"] for row in result["context"]}


def test_scalar_write_query_returns_a_conditional_gm_owner():
    result = query.report("pl.setval")
    route = next(
        row
        for row in result["related_contracts"]
        if row["id"] == "gm-scalar-write-ownership"
    )
    assert route["when"] and route["reference"]["sha256"]
    assert "gm-scalar-write-ownership" not in {row["id"] for row in result["context"]}
    selected = query.report("GM scalar writes")
    (context,) = selected["context"]
    assert context["id"] == "gm-scalar-write-ownership"
    assert "1 worker" in context["text"] and "pl.store" in context["text"]
    assert "exact hardware mechanism remains unproven" in context["text"]


def test_scalar_read_query_does_not_select_the_gm_write_owner():
    result = query.report("pl.getval")
    assert "gm-scalar-write-ownership" not in {
        row["id"] for row in result["related_contracts"]
    }


@pytest.mark.parametrize(
    "text", ["normalization clamp cancellation", "positive scale floor"]
)
def test_normalization_query_retains_the_nonzero_floor_branch_and_rounding_limit(text):
    result = query.report(text)
    (context,) = result["context"]
    assert context["id"] == "normalization-scale-floor"
    assert "max(m, tau / c)" in context["text"]
    assert "m == 0" in context["text"] and "rounded scale" in context["text"]
    assert "below, at and above the floor" in context["text"]
    assert context["reference"]["sha256"]
    assert result["validation"]["numerical"] == "not_established"


def test_scalar_floor_api_query_does_not_select_the_normalization_recipe():
    result = query.report("pl.simt.floor")
    assert "normalization-scale-floor" not in {row["id"] for row in result["context"]}


@pytest.mark.parametrize("text", ["SIMT large input slow", "kernel[None,1]"])
def test_single_worker_query_routes_coverage_and_conditional_tuning(text):
    result = query.report(text)
    (context,) = result["context"]
    assert context["id"] == "simt-worker-coverage"
    contract = next(
        row
        for row in result["knowledge"]["items"]
        if row["id"] == "contract:simt-worker-coverage"
    )
    assert contract["missing_facts"] and context["reference"]["sha256"]
    assert result["validation"]["performance"] == "not_established"


def test_phase_query_uses_the_existing_control_flow_owner():
    result = query.report("SIMT loop phase splitting")
    (context,) = result["context"]
    assert context["id"] == "simt-control-flow"
    assert context["reference"]["sha256"]
    assert result["validation"]["performance"] == "not_established"


@pytest.mark.parametrize("text", ["quantization off by one", "量化差1"])
def test_exact_quantization_query_returns_the_stage_diagnosis(text):
    result = query.report(text)
    (context,) = result["context"]
    assert context["id"] == "exact-quantization-boundary"
    contract = next(
        row
        for row in result["knowledge"]["items"]
        if row["id"] == "contract:exact-quantization-boundary"
    )
    assert contract["missing_facts"] and context["reference"]["sha256"]
    assert result["validation"]["numerical"] == "not_established"


def test_rounding_api_alone_does_not_select_an_exact_quantization_failure():
    result = query.report("pl.simt.rint")
    assert "exact-quantization-boundary" not in {row["id"] for row in result["context"]}


@pytest.mark.parametrize("symbol", ["pl.move", "pl.insert"])
def test_unbound_transfer_symbol_returns_conditional_routes_without_a_cube_recipe(
    symbol,
):
    result = query.report(symbol)
    assert result["status"] == "passed" and not result["context"]
    assert len(result["signatures"]) == 1
    assert "ub-l1" in {route["id"] for route in result["related_contracts"]}
    assert all(
        route["when"] and route["reference"]["sha256"]
        for route in result["related_contracts"]
    )


def test_only_a_matching_source_transfer_selects_the_handover_context(tmp_path):
    from .test_ub_l1 import source

    path = tmp_path / "candidate.py"
    path.write_text(source(src_layout="NZ", api="insert"))
    result = query.report("pl.insert", source=path)
    assert [c["id"] for c in result["context"]] == ["ub-l1"]
    assert result["context"][0]["matched_sites"][0]["file"] == str(path.resolve())
    assert len(result["signatures"]) == 2
    assert not query.report("pl.move", source=path)["context"]
    path.write_text(source(src_layout="NZ", dst_space="Vec", api="move"))
    assert not query.report("pl.move", source=path)["context"]


@pytest.mark.parametrize(
    "symbol,context_id",
    [
        ("vf.mul_add_dst", "vf-destination-state"),
        ("vf.mul_dst_add", "vf-destination-state"),
        ("vf.move", "vf-register-copies"),
        ("vf.reduce_sum", "vf-reduction-result"),
    ],
)
def test_state_and_reduction_queries_select_a_focused_owner(symbol, context_id):
    result = query.report(symbol)
    assert result["status"] == "passed"
    assert [c["id"] for c in result["context"]] == [context_id]
    assert result["signatures"] and result["missing_facts"]


def test_output_conversion_route_is_conditional_for_general_store_queries():
    result = query.report("pl.store")
    assert not result["context"]
    routes = {route["id"]: route for route in result["related_contracts"]}
    assert {"acc-final-output", "gm-plane-offsets"} <= routes.keys()
    assert all(
        route["when"] and route["reference"]["sha256"] for route in routes.values()
    )
    assert "selected_frontend_store" not in result
    selected = query.report("FP32 accumulator to FP16 GM")
    assert [c["id"] for c in selected["context"]] == ["acc-final-output"]
    table = selected["selected_frontend_store"]
    assert table["status"] == "observed_source" and table["reference"]["sha256"]
    assert ["NZ", "ND", "FP32", "FP16"] in table["rows"]
    assert ["NZ", "NZ", "FP32", "FP16"] in table["rows"]
    from pathlib import Path

    lines = Path(table["reference"]["path"]).read_text().splitlines()
    index = table["rows"].index(["NZ", "ND", "FP32", "FP16"])
    assert '"Acc", "GM"' in lines[table["row_lines"][index] - 1]
    assert "DataType.FP32, DataType.FP16" in lines[table["row_lines"][index] - 1]
    assert (
        table["reference"]["line"] == table["row_lines"][0] > table["declaration_line"]
    )
    assert selected["validation"]["compilation"] == "not_established"


def test_store_table_is_fresh_source_bound_and_never_imported(tmp_path):
    import hashlib
    import os

    path = tmp_path / "python/pypto_pro/ir/op/block_ops.py"
    path.parent.mkdir(parents=True)
    text = """raise RuntimeError("must not execute")
_A5_STORE_COMBOS = (("Acc", "GM", TensorLayout.NZ, TensorLayout.ND, DataType.FP32, DataType.FP16),)
_LAYOUT_DTYPE_COMBOS = {"store": {"a5": _A5_STORE_COMBOS}}
"""
    path.write_text(text)
    original = path.stat()
    first = query.report("acc-final-output", pypto_root=tmp_path)[
        "selected_frontend_store"
    ]
    assert first["rows"] == [["NZ", "ND", "FP32", "FP16"]]
    assert first["reference"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first["reference"]["line"] == 2
    path.write_text(text.replace("FP16", "BF16"))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = query.report("acc-final-output", pypto_root=tmp_path)[
        "selected_frontend_store"
    ]
    assert second["rows"] == [["NZ", "ND", "FP32", "BF16"]]
    assert second["reference"]["sha256"] != first["reference"]["sha256"]


@pytest.mark.parametrize(
    "mapping,observed",
    [
        ("NpuArch.DAV_3510: (_A5_STORE_COMBOS, _A5_STORE_QUANT_COMBOS)", True),
        ("NpuArch.DAV_2201: (_A5_STORE_COMBOS, _A5_STORE_QUANT_COMBOS)", False),
        ("NpuArch.DAV_3510: (_A5_STORE_QUANT_COMBOS, _A5_STORE_COMBOS)", False),
        ("NpuArch.DAV_3510: make_tables()", False),
        (
            '"a5": _A5_STORE_COMBOS, NpuArch.DAV_3510: (_A5_STORE_COMBOS, _A5_STORE_QUANT_COMBOS)',
            False,
        ),
    ],
)
def test_enum_architecture_store_pair_keeps_quantized_rows_separate(
    tmp_path, mapping, observed
):
    path = tmp_path / "python/pypto_pro/ir/op/block_ops.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        'raise RuntimeError("must not execute")\n'
        '_A5_STORE_COMBOS = (("Acc", "GM", TensorLayout.NZ, TensorLayout.ND, DataType.FP32, DataType.FP16),)\n'
        '_A5_STORE_QUANT_COMBOS = (("Acc", "GM", TensorLayout.NZ, TensorLayout.ND, DataType.FP32, DataType.INT8),)\n'
        '_LAYOUT_DTYPE_COMBOS = {"store": {' + mapping + "}}\n"
    )
    result = query.store_combinations(tmp_path)
    assert result["status"] == ("observed_source" if observed else "unverified")
    assert result["rows"] == ([["NZ", "ND", "FP32", "FP16"]] if observed else [])


@pytest.mark.parametrize(
    "declaration",
    [
        "",
        "_A5_STORE_COMBOS = build_table()",
        "_A5_STORE_COMBOS = (dynamic_row,)",
        '_A5_STORE_COMBOS = (("Acc", "GM", unknown.NZ, TensorLayout.ND, DataType.FP32, DataType.FP16),)',
        "_A5_STORE_COMBOS = ()\n_A5_STORE_COMBOS = ()",
        "_A5_STORE_COMBOS = ()\n_LAYOUT_DTYPE_COMBOS = {}",
    ],
)
def test_unknown_store_table_is_unverified_not_unsupported(tmp_path, declaration):
    path = tmp_path / "python/pypto_pro/ir/op/block_ops.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '_LAYOUT_DTYPE_COMBOS = {"store": {"a5": _A5_STORE_COMBOS}}\n' + declaration
    )
    result = query.report("acc-final-output", pypto_root=tmp_path)
    assert result["selected_frontend_store"]["status"] == "unverified"
    assert result["selected_frontend_store"]["rows"] == []
    assert result["validation"]["compilation"] == "not_established"


def test_store_table_bounds_rows_and_missing_source(tmp_path):
    assert (
        query.report("acc-final-output", pypto_root=tmp_path)[
            "selected_frontend_store"
        ]["status"]
        == "unverified"
    )
    path = tmp_path / "python/pypto_pro/ir/op/block_ops.py"
    path.parent.mkdir(parents=True)
    row = (
        '("Acc", "GM", TensorLayout.NZ, TensorLayout.ND, DataType.FP32, DataType.FP16),'
    )
    path.write_text(
        "_A5_STORE_COMBOS = ("
        + row * 35
        + ')\n_LAYOUT_DTYPE_COMBOS = {"store": {"a5": _A5_STORE_COMBOS}}'
    )
    table = query.report("acc-final-output", pypto_root=tmp_path)[
        "selected_frontend_store"
    ]
    assert len(table["rows"]) == 32 and table["omitted_rows"] == 3
    path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    assert query.store_combinations(tmp_path)["status"] == "unverified"


def test_store_table_source_drift_is_unverified(monkeypatch):
    original = query.store_combinations

    def changed(root):
        table = original(root)
        table["reference"]["sha256"] = "0" * 64
        return table

    monkeypatch.setattr(query, "store_combinations", changed)
    table = query.report("acc-final-output")["selected_frontend_store"]
    assert table["status"] == "unverified" and not table["rows"]


def test_cast_queries_offer_both_directions_without_selecting_a_recipe():
    result = query.report("vf.astype")
    assert not result["context"] and result["signatures"]
    assert {r["id"] for r in result["related_contracts"]} == {
        "vf-fp16-widening",
        "vf-fp32-narrowing",
    }
    generic = query.report("FP32 to FP16")
    assert not generic["context"]
    assert {r["id"] for r in generic["related_contracts"]} == {
        "acc-final-output",
        "vf-fp32-narrowing",
    }
    selected = query.report("VF FP32 to FP16")
    assert [c["id"] for c in selected["context"]] == ["vf-fp32-narrowing"]
    assert "PACK_B32" in selected["context"][0]["text"]
    assert "dense-register consumer" in selected["context"][0]["text"]
    assert "vf-fp32-narrowing" in {
        r["id"] for r in query.report("vf.store_align")["related_contracts"]
    }


@pytest.mark.parametrize(
    "text,owner",
    [
        ("runtime dispatch outside VF loop", "vf-loop-control"),
        (
            "Python bool closure specialization JIT if loop dispatch optimization",
            "vf-loop-control",
        ),
        ("work distribution", "grouped-ownership"),
        ("worker balance across groups", "grouped-ownership"),
        ("final FP16 output conversion", "acc-final-output"),
        ("NZ handover cursor store block_stride", "ub-l1"),
    ],
)
def test_scenario_discovery_provides_conditional_owner_routes(text, owner):
    result = query.report(text)
    assert result["status"] == "passed" and not result["context"]
    assert owner in {r["id"] for r in result["related_contracts"]}
    assert all(
        r["match"] == "discovery" and r["when"] for r in result["related_contracts"]
    )
    assert "do not establish" in result["missing_facts"][-1]
    selected = query.report(owner)
    assert owner in {c["id"] for c in selected["context"]}


def test_unambiguous_accumulation_and_vf_conversion_scenarios_stay_separate():
    result = query.report(
        "Cube GEMM FP32 accumulation FP16 output destination conversion"
    )
    assert [c["id"] for c in result["context"]] == ["acc-final-output"]
    generic = query.report("FP16 to FP32")
    assert not generic["context"] and [
        r["id"] for r in generic["related_contracts"]
    ] == ["vf-fp16-widening"]
    assert [c["id"] for c in query.report("VF FP16 to FP32")["context"]] == [
        "vf-fp16-widening"
    ]
    assert query.report("vf.not_a_real_cast")["status"] == "unverified"


def test_prefix_mask_query_returns_count_and_conversion_boundaries():
    result = query.report("vf.update_mask")
    assert [c["id"] for c in result["context"]] == ["vf-prefix-mask"]
    assert "uint32" in result["context"][0]["text"]
    assert "smaller logical tail bound" in result["context"][0]["text"]


def test_unqualified_move_name_does_not_select_a_memory_space_or_namespace():
    result = query.report("move")
    assert not result["context"] and not result["signatures"]
    assert {r["id"] for r in result["related_contracts"]} == {
        "ub-l1",
        "acc-final-output",
        "vf-register-copies",
    }
    assert not query.report("insert")["context"]


def test_context_bodies_do_not_include_adjacent_routing_metadata():
    result = query.report("vf.full")
    assert result["context"] and "kernel-query:" not in result["context"][0]["text"]


def test_failure_reviews_resolve_aliases_and_require_conversion_calls(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("""from pypto_pro.language import Vf as ops
def compute(src, dst, pred):
    values = ops.load_align(src, [0, 0])
    result = ops.astype(values, pred, dtype=32)
    ops.store_align(dst, result, pred, [0, 0])
""")
    routes, omitted = query.review_routes([path])
    assert {r["id"] for r in routes} == {"vf-fp16-widening", "vf-fp32-narrowing"}
    assert omitted == 0
    assert all(
        r["source_matches"] and r["when"] and r["reference"]["sha256"] for r in routes
    )
    assert all(
        site["source_sha256"] == feedback.file_identity(path)["sha256"]
        for r in routes
        for site in r["source_matches"]
    )
    path.write_text("""from pypto_pro.language import Vf as ops
def copy(src, dst, pred):
    values = ops.load_align(src, [0, 0])
    ops.store_align(dst, values, pred, [0, 0])
""")
    assert query.review_routes([path]) == ([], 0)


@pytest.mark.parametrize(
    "body",
    [
        "def f(ops, src, pred):\n    ops.reduce_sum(src, pred)\n",
        "ops = None\nops.reduce_sum(None, None)\n",
        "def f(src, pred):\n    ops.reduce_sum(src, pred)\n    ops = None\n",
    ],
)
def test_failure_reviews_do_not_treat_shadowed_names_as_native_calls(tmp_path, body):
    path = tmp_path / "candidate.py"
    path.write_text("from pypto_pro.language import Vf as ops\n" + body)
    assert query.review_routes([path]) == ([], 0)


def test_review_source_identity_covers_the_inspected_bytes(tmp_path, monkeypatch):
    from _checks import api_policy

    path = tmp_path / "candidate.py"
    path.write_text(
        "from pypto_pro.language import Vf as vf\nvf.reduce_sum(None, None)\n"
    )
    inspect = api_policy.inspect

    def inspect_then_change(source, **kwargs):
        result = inspect(source, **kwargs)
        source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(api_policy, "inspect", inspect_then_change)
    assert query.review_routes([path]) == ([], 0)


@pytest.mark.parametrize("exit_code", [0, 9])
def test_unclassified_failure_offers_reviews_without_inventing_a_diagnosis(
    tmp_path, exit_code
):
    path = tmp_path / "candidate.py"
    path.write_text("""import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
@pl.vector_function
def compute(src, dst, pred):
    values = vf.load_align(src, [0, 0])
    total = vf.reduce_sum(values, pred)
    vf.store_align(dst, total, pred, [0, 0])
""")
    proc = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({exit_code})",
        ],
        capture_output=True,
        text=True,
    )
    result = json.loads(proc.stdout)
    assert proc.returncode == exit_code == result["command"]["exit_code"]
    assert result["context"] == []
    assert result["validation"]["numerical"] == "not_established"
    if exit_code:
        assert [r["id"] for r in result["related_contracts"]] == ["vf-reduction-result"]
        assert "not a diagnosis" in result["contract_review_scope"]
        assert result["diagnostics"][-1]["rule"] == "command-failed"
    else:
        assert "related_contracts" not in result


def test_source_drift_suppresses_conditional_failure_reviews(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("""from pypto_pro.language import Vf as vf
def compute(src, pred):
    return vf.reduce_sum(src, pred)
""")
    command = f"from pathlib import Path; p=Path({str(path)!r}); p.write_text(p.read_text()+'\\n'); raise SystemExit(9)"
    proc = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
    )
    result = json.loads(proc.stdout)
    assert proc.returncode == 9
    assert "source-changed" in {d["rule"] for d in result["diagnostics"]}
    assert "related_contracts" not in result


@pytest.mark.parametrize(
    "line,rule",
    [
        (
            "unix_listener: cannot bind to path /restricted/control.sock: Operation not permitted",
            "execution-transport-failed",
        ),
        (
            "ssh: connect to host example.invalid port 22: Connection refused",
            "execution-transport-failed",
        ),
        (
            "user@example.invalid: Permission denied (publickey).",
            "execution-transport-failed",
        ),
        ("Host key verification failed.", "execution-transport-failed"),
        ("ModuleNotFoundError: No module named 'torch_npu'", "execution-import-failed"),
        (
            "ImportError: cannot import name 'unknown' from 'pypto_pro'",
            "execution-import-failed",
        ),
    ],
)
def test_reported_environment_boundaries_do_not_select_kernel_api_context(line, rule):
    found, contexts = runtime_diagnostics.from_output(line)
    assert [d["rule"] for d in found] == [rule]
    assert [c["id"] for c in contexts] == ["execution-environment"]
    assert not runtime_diagnostics.from_output(f"  {line}")[0]
    assert not runtime_diagnostics.from_output(f"print({line!r})")[0]


def test_transport_failure_preserves_exit_and_suppresses_source_reviews(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("""from pypto_pro.language import Vf as vf
def compute(src, pred):
    return vf.reduce_sum(src, pred)
""")
    program = (
        "print('unix_listener: cannot bind to path /restricted/control.sock: Operation"
        " not permitted'); raise SystemExit(255)"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            program,
        ],
        capture_output=True,
        text=True,
    )
    result = json.loads(proc.stdout)
    assert proc.returncode == result["command"]["exit_code"] == 255
    assert [c["id"] for c in result["context"]] == ["execution-environment"]
    assert "related_contracts" not in result
    assert result["validation"]["numerical"] == "not_established"


def test_source_feedback_uses_owner_declared_rule_routes(tmp_path):
    from _checks.cli import rules_result
    from _checks.paths import PYPTO_ROOT

    path = tmp_path / "candidate.py"
    path.write_text("""import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
@pl.vector_function
def compute(a, b, mask):
    result = vf.mul_add_dst(a, b, mask)
""")
    result = rules_result([path], pypto_root=PYPTO_ROOT)
    assert result["status"] == "failed"
    assert [c["id"] for c in result["context"]] == ["vf-destination-state"]


@pytest.mark.parametrize(
    "symbol,context_id",
    [("pl.make_tile_group", "tile-group-depth"), ("vf.full", "vf-scalar-dtype")],
)
def test_api_gap_queries_return_only_their_owner(symbol, context_id):
    result = query.report(symbol)
    assert result["status"] == "passed"
    assert [c["id"] for c in result["context"]] == [context_id]
    assert len(result["signatures"]) == 1
    assert result["signatures"][0]["signature"].startswith(symbol + "(")
    assert result["context"][0]["reference"]["sha256"]
    assert len(result["missing_facts"]) == 2
    assert result["validation"]["numerical"] == "not_established"


@pytest.mark.parametrize(
    "message,rule,context_id",
    [
        (
            'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
            "tile-group-depth-required",
            "tile-group-depth",
        ),
        (
            "vf.add requires dst, src0, src1 to have the same type, got dst=DT_FP16 src0=DT_FP16 src1=DT_FP32",
            "vf-add-operand-dtype",
            "vf-scalar-dtype",
        ),
    ],
)
def test_observed_api_failures_route_without_matching_quoted_source(
    message, rule, context_id
):
    log = (
        "RuntimeError: [source.cpp:8][PRO_CODEGEN]:ErrCode: F00001! Enum:"
        f" ExternalError::INVALID_TYPE. {message}\n  --> candidate.py:21:4\n"
    )
    found, context = runtime_diagnostics.from_output(log)
    assert [d["rule"] for d in found] == [rule]
    assert found[0]["location"] == {"file": "candidate.py", "line": 21, "column": 4}
    assert found[0]["evidence"]["error_code"] == "F00001"
    assert [c["id"] for c in context] == [context_id]
    assert runtime_diagnostics.from_output(
        f"  21 | raise RuntimeError({message!r})\nE       assert {message!r} in result"
    ) == ([], [])
    error = RuntimeError(message)
    diagnostics, context = runtime_diagnostics.from_exception(error, stage="parse")
    assert diagnostics[0]["rule"] == rule
    assert [c["id"] for c in context] == [context_id]


@pytest.mark.parametrize(
    "message",
    [
        "make_tile_group() depth must be a positive integer",
        "vf.add unsupported dtype",
        "vf.sub requires dst, src0, src1 to have the same type, got dst=DT_FP16 src0=DT_FP16 src1=DT_FP32",
    ],
)
def test_unmatched_api_errors_keep_original_diagnostic_without_guessed_recipe(message):
    error = RuntimeError(message)
    diagnostics, context = runtime_diagnostics.from_exception(error, stage="parse")
    assert diagnostics[0]["rule"] == "native-rejection"
    assert diagnostics[0]["message"] == message
    assert not context


def test_unknown_query_does_not_claim_api_absence():
    result = query.report("absenttopiczxq9")
    assert result["status"] == "unverified"
    assert not result["signatures"] and not result["context"]


def test_abbreviations_do_not_match_inside_unrelated_words():
    result = query.report("vector formatting")
    assert not result["context"]
    assert result["status"] == "unverified"


@pytest.mark.parametrize("text", ["L1 to UB", "从L1搬到UB", "Mat→Vec"])
def test_reverse_path_does_not_receive_the_forward_recipe(text):
    result = query.report(text)
    assert not result["context"] and result["status"] == "unverified"


def test_missing_companion_example_is_explicit(tmp_path):
    result = query.report("ub-l1", examples_root=tmp_path)
    assert result["context"][0]["example_unavailable"]
    assert not result["context"][0]["examples"]


def test_no_error_match_inside_source_excerpt():
    text = """  22 | raise RuntimeError("move: unsupported data path")
    print("Tensor parameter 'x' rank mismatch: expected 2, got 1")
E       assert "move: offset must contain exactly 2 elements" in message
RuntimeError: opaque device error 507015
"""
    found, context = runtime_diagnostics.from_output(text)
    assert not found and not context


def test_embedded_framework_head_is_not_a_diagnostic():
    text = '123 | assert "[block_ops.py:1][PRO_IR]:ErrCode: F00004! Enum: X. move: invalid shape" in message'
    assert runtime_diagnostics.from_output(text) == ([], [])


def test_separate_diagnostics_keep_their_own_locations():
    text = (
        "move: offset must contain exactly 2 elements\n  --> first.py:7:3\nmove: shape mismatch\n  --> second.py:11:5\n"
    )
    found, _ = runtime_diagnostics.from_output(text)
    assert [(d["location"]["file"], d["location"]["line"]) for d in found] == [
        ("first.py", 7),
        ("second.py", 11),
    ]


def test_framework_head_preserves_stage_code_and_location():
    text = (
        "[block_ops.py:1020][PRO_IR]:ErrCode: F00004! Enum:"
        " ExternalError::NOT_IMPLEMENTED_ERROR. move: NZ destination uses align16 rows\n  --> candidate.py:12:5\n"
    )
    found, context = runtime_diagnostics.from_output(text)
    assert found[0]["rule"] == "native-transfer-contract"
    assert found[0]["evidence"]["module"] == "PRO_IR"
    assert found[0]["evidence"]["error_code"] == "F00004"
    assert found[0]["location"]["line"] == 12
    assert found[0]["evidence"]["stage"] == "unknown"
    assert not context


def test_unknown_framework_error_retains_phase_without_irrelevant_recipe():
    text = (
        "[backend.cpp:4][PRO_CODEGEN]:ErrCode: F00001! Enum:"
        " ExternalError::INVALID_TYPE. vf.add operand types differ\n  --> candidate.py:10:3\n"
    )
    found, context = runtime_diagnostics.from_output(text)
    assert found[0]["rule"] == "framework-rejection"
    assert found[0]["evidence"]["module"] == "PRO_CODEGEN"
    assert not context


def test_pytest_wrapped_diagnostic_keeps_the_user_span():
    text = (
        "E RuntimeError: [backend.cpp:4][PRO_CODEGEN]:ErrCode: F00001! Enum:"
        " ExternalError::INVALID_TYPE. vf.add operand types differ\nE    --> candidate.py:10:3\n"
    )
    found, _ = runtime_diagnostics.from_output(text)
    assert found[0]["location"] == {"file": "candidate.py", "line": 10, "column": 3}


def run(tmp_path, command, source="value = 1\n", extra=()):
    path = tmp_path / "candidate.py"
    path.write_text(source)
    return subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(path),
            "--format",
            "json",
            *extra,
            "--",
            sys.executable,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_json_run_keeps_original_failure_and_routes_context(tmp_path):
    result = run(
        tmp_path, "print('move: NZ destination uses align16 rows'); raise SystemExit(7)"
    )
    assert result.returncode == 7
    report = json.loads(result.stdout)
    assert report["command"]["exit_code"] == 7
    assert "align16" in result.stderr
    assert report["diagnostics"][0]["rule"] == "native-transfer-contract"
    assert report["diagnostics"][0]["evidence"]["condition_status"] == "unverified"
    assert not report["context"]


def test_generic_transfer_error_needs_source_evidence_to_select_ub_context(tmp_path):
    file = str(tmp_path / "candidate.py")
    text = (
        "[block_ops.py:1174][PRO_IR]:ErrCode: F0000E! Enum:"
        " ExternalError::INVALID_ARGUMENT. offset must contain exactly 2 elements\n "
        f" --> {file}:6:5\n"
    )
    found, context = runtime_diagnostics.from_output(text)
    assert found[0]["rule"] == "native-transfer-contract" and not context
    sites = [
        {
            "file": file,
            "line": 5,
            "end_line": 7,
            "source": {"space": "Vec"},
            "destination": {"space": "Mat"},
        }
    ]
    found, context = runtime_diagnostics.from_output(text, transfers=sites)
    assert found[0]["evidence"]["transfer"] == sites[0]
    assert context[0]["id"] == "ub-l1"
    sites[0]["source"]["space"] = "Acc"
    assert not runtime_diagnostics.from_output(text, transfers=sites)[1]


def test_success_is_only_command_success(tmp_path):
    result = run(tmp_path, "print('FAIL: this is only printed text')")
    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["validation"]["numerical"] == "not_established"
    assert report["validation"]["performance"] == "not_established"


def test_changed_sources_invalidate_result(tmp_path):
    result = run(
        tmp_path,
        "from pathlib import Path; Path('"
        + str(tmp_path / "candidate.py")
        + "').write_text('value = 2')",
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["validation"]["numerical"] == "invalidated"


def test_static_failure_prevents_child_and_contains_context(tmp_path):
    from .test_ub_l1 import source

    result = run(
        tmp_path,
        "raise RuntimeError('child started')",
        source(dst_space="Vec", rows=33),
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert not report["command"]["started"]
    assert report["context"][0]["id"] == "ub-l1"


def test_query_cli_json_from_other_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ENTRY), "query", "UB to L1", "--format", "json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == "pypto-kernel-feedback/1"


def test_compact_output_keeps_evidence_and_can_expand_descriptors(tmp_path):
    from _checks.cli import rules_result
    from _checks.paths import PYPTO_ROOT
    from .test_ub_l1 import source

    path = tmp_path / "candidate.py"
    path.write_text(source())
    full = rules_result([path], pypto_root=PYPTO_ROOT)
    compact = feedback.presentation(full)
    assert compact["diagnostics"] == full["diagnostics"]
    assert compact["context"] == full["context"]
    assert compact["identity"]["sources"] == full["identity"]["sources"]
    assert compact["identity"]["selected_source"]["files_sha256"]
    assert "transfers" not in compact and full["transfers"]
    assert feedback.presentation(full, details=True) == full
