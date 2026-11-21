# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source hypotheses remain bounded, conditional and tied to measured bytes."""

import json
import subprocess
import sys
from textwrap import dedent

import pytest

from _checks import feedback, query, suggest
from _checks.paths import EXAMPLES_ROOT, PYPTO_ROOT, SKILL_ROOT


def source(tmp_path, body):
    path = tmp_path / "source.py"
    path.write_text("import pypto_pro.language as pl\n" + dedent(body))
    return path


VF = """
@pl.vector_function
def body(rows, first):
    for row in pl.range(rows):
        if first:
            consume(row)
"""
GROUP = """
@pl.jit
def kernel(groups, count):
    for group in pl.range(groups):
        for row in pl.range(pl.get_block_idx(), count, pl.get_block_num()):
            consume(row)
"""


@pytest.mark.parametrize(
    "body,pattern", [(VF, "vf-loop-runtime-control"), (GROUP, "nested-worker-restart")]
)
def test_existing_advisories_become_conditional_experiments(tmp_path, body, pattern):
    path = source(tmp_path, body)
    result = suggest.report([path])
    assert result["status"] == "passed" and len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["pattern"] == pattern and candidate["measured_cost"] is None
    assert candidate["prediction"]["latency_delta_us"] is None
    assert (
        candidate["conditions"] and candidate["reject_if"] and candidate["comparison"]
    )
    recipe = query.report(candidate["query"])
    assert recipe["context"][0]["reference"] == candidate["reference"]
    assert recipe["context"][0]["examples"][0]["text"]
    assert result["validation"]["performance"] == "not_established"


def test_real_handover_has_one_conversion_candidate():
    path = EXAMPLES_ROOT / "basic_api/datamove/ub_l1_handover/kernel.py"
    result = suggest.report([path])
    candidate = result["candidates"][0]
    assert candidate["pattern"] == "ub-l1-extra-layout-move"
    lines = path.read_text().splitlines()
    assert "pl.move" in lines[candidate["site"]["line"] - 1]
    assert "pl.insert" in lines[candidate["site"]["handover_line"] - 1]
    assert "active branch" in candidate["site"]["unknowns"]


def layout(tmp_path, *, source_layout="ND", consumer="nz", rebind="", space="Mat"):
    return source(
        tmp_path,
        f"""
        @pl.jit(auto_mutex=False)
        def kernel():
            nd = pl.make_tile(pl.TileType(shape=[32,128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.{source_layout}), addr=0)
            nz = pl.make_tile(pl.TileType(shape=[32,128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addr=8192)
            l1 = pl.make_tile(pl.TileType(shape=[32,128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.{space}, layout=pl.NZ), addr=0)
            pl.move(nz, nd)
            {rebind}
            pl.insert(l1, {consumer}, [0,0])
    """,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_layout": "NZ"},
        {"consumer": "other"},
        {"rebind": "nz = unknown()"},
        {"space": "Vec"},
    ],
)
def test_unrelated_or_unresolved_transfers_do_not_form_a_layout_chain(tmp_path, kwargs):
    assert not suggest.report([layout(tmp_path, **kwargs)])["candidates"]


def test_layout_chain_can_be_suggested_but_not_accepted_from_syntax(tmp_path):
    result = suggest.report([layout(tmp_path)])
    assert result["candidates"][0]["prediction"]["latency_delta_us"] is None
    assert result["validation"]["numerical"] == "not_established"


def test_no_match_is_not_an_optimization_ceiling(tmp_path):
    result = suggest.report([source(tmp_path, "x = 1\n")])
    assert not result["candidates"] and result["coverage"]["matched_sites"] == 0
    assert result["next_action"]


def test_candidate_count_is_bounded_with_explicit_omissions(tmp_path):
    path = source(
        tmp_path, "\n".join(VF.replace("def body", f"def body{i}") for i in range(8))
    )
    result = suggest.report([path])
    assert (
        len(result["candidates"]) == 3 and result["coverage"]["omitted_candidates"] == 5
    )
    assert len({c["id"] for c in result["candidates"]}) == 3
    with pytest.raises(ValueError):
        suggest.report([path], limit=6)


def costs(tmp_path, path):
    toolchain = tmp_path / "toolchain.json"
    toolchain.write_text('{"compiler":"fixture"}')
    sites = suggest.report([path])["candidates"]

    def duration(value):
        return {"median_us": value, "min_us": value - 1, "max_us": value + 1}

    data = {
        "schema": "pypto-kernel-costs/1",
        "sources": [feedback.file_identity(path)],
        "framework_sources": feedback.identity(PYPTO_ROOT)["selected_source"]["files"],
        "toolchain": feedback.file_identity(toolchain),
        "scope": "complete_call",
        "workload": {"rows": 32, "workers": 1},
        "profile": {
            "metric": "native_task_sum_us",
            **duration(30),
            "repeats": 5,
            "tasks_per_call": 2,
            "per_task": [
                {"position": 1, **duration(10)},
                {"position": 2, **duration(20)},
            ],
        },
        "sites": [
            {"file": str(path), "line": c["site"]["line"], "task_position": i + 1}
            for i, c in enumerate(sites)
        ],
    }
    output = tmp_path / "costs.json"
    output.write_text(json.dumps(data))
    return output, data


def test_matching_task_costs_prioritize_sites_without_inventing_exclusive_cost(
    tmp_path,
):
    path = source(tmp_path, VF + GROUP)
    measured, data = costs(tmp_path, path)
    result = suggest.report([path], measurements=measured)
    assert [c["measured_cost"]["task"]["position"] for c in result["candidates"]] == [
        2,
        1,
    ]
    assert result["candidates"][0]["measured_cost"]["complete_call_median_us"] == 30
    assert result["candidates"][0]["measured_cost"]["pipe_counters"] is None
    assert result["candidates"][0]["prediction"]["latency_delta_us"] is None


@pytest.mark.parametrize(
    "drift", ["source", "toolchain", "framework", "bad_profile", "bad_site"]
)
def test_stale_or_invalid_costs_are_explicit_and_never_rank_candidates(tmp_path, drift):
    path = source(tmp_path, VF + GROUP)
    measured, data = costs(tmp_path, path)
    if drift == "source":
        path.write_text(path.read_text() + "# changed\n")
    elif drift == "toolchain":
        (tmp_path / "toolchain.json").write_text("{}")
    elif drift == "framework":
        data["framework_sources"] = {}
    elif drift == "bad_profile":
        data["profile"]["per_task"].pop()
    else:
        data["sites"][0]["line"] = 9000
    measured.write_text(json.dumps(data))
    result = suggest.report([path], measurements=measured)
    assert result["status"] == "unverified" and result["diagnostics"]
    assert len(result["candidates"]) == 2 and all(
        c["measured_cost"] is None for c in result["candidates"]
    )


def test_cli_source_paths_follow_caller_and_emit_one_json_object(tmp_path):
    path = source(tmp_path, VF)
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "suggest",
            "--source",
            path.name,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["candidates"][0]["source"]["path"] == str(path)
