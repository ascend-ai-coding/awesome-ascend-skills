# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Complete program discovery, dependency roles and source-bound read-only refresh."""

import json
import os
from pathlib import Path

import pytest

from _checks import catalog, example_programs, feedback, host_io, query
from _checks.paths import EXAMPLES_ROOT


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def library(tmp_path, monkeypatch):
    root, refs = tmp_path / "examples", tmp_path / "references"
    for group, name in (
        ("basic_api", "chapter/minimal"),
        ("matmul", "product"),
        ("algorithms", "batched"),
    ):
        folder = root / group / name
        write(folder / "__init__.py", "")
        write(
            folder / "kernel.py",
            '''"""Independent rows and M tails; N aligned."""
import pypto_pro.language as dsl
from pypto_pro.language import Vf as vector
from ._kernels.compute import build
@dsl.jit
def kernel(x):
    return vector.add(x, x), dsl.system.bar_all(), dsl.simt.atomic_add(x)
raise RuntimeError("must not import any example")
''',
        )
        write(folder / "_kernels/__init__.py", "")
        write(
            folder / "_kernels/compute.py",
            """from pypto_pro.language import load as fetch
from .deeper import helper
def build(x):
    return fetch(x)
""",
        )
        write(folder / "_kernels/deeper.py", "def helper():\n    return 1\n")
        write(
            folder / "reference.py",
            '"""Independent FP64 formula."""\nimport torch\ndef reference(x):\n    return x.double()\n',
        )
        write(
            folder / "test.py",
            '''"""Only the declared input range; external evaluator 0.7.0 optional."""
import os as environment
import pytest
from .kernel import kernel
from .reference import reference
EXTERNAL = environment.environ.get("OPTIONAL_EVALUATOR_ROOT")
def test_sample(device):
    pytest.importorskip("torch_npu")
''',
        )
    write(root / "__init__.py", "")
    write(
        root / "conftest.py",
        'import os\nimport pytest\nSELECTED = os.environ["TILE_FWK_DEVICE_ID"]\n',
    )
    write(root / "pytest.ini", "[pytest]\npython_files = test.py\n")
    write(
        refs / "tooling/examples.md",
        """# Selection
| Mechanism | Example |
| --- | --- |
| Product | [product](../../examples/matmul/product/) |
| Batched transposed GEMM with M tails | [batched](../../examples/algorithms/batched/) |
""",
    )
    monkeypatch.setattr(query, "REFERENCES_ROOT", refs)
    return root, refs, tmp_path / "framework"


def discover(library, **kwargs):
    root, _, framework = library
    return query.report(
        examples_root=root, pypto_root=framework, kinds=["example"], **kwargs
    )["knowledge"]


@pytest.mark.parametrize(
    "text", ["AttnEpilogue", "attn_epilogue", "attn epilogue", "ATTN-EPILOGUE"]
)
def test_whole_program_names_accept_spelling_variants(text):
    row = {"title": "attn_epilogue", "aliases": []}
    assert example_programs.match(row, text, None, []) == ("example_name", 850)


@pytest.mark.parametrize(
    "text", ["EpilogueAttn", "AttnEpilogue extra", "pl.AttnEpilogue"]
)
def test_program_name_matching_does_not_rewrite_arbitrary_terms(text):
    row = {
        "title": "attn_epilogue",
        "aliases": [],
        "_meta": {},
        "_contracts": [],
        "_terms": "attn epilogue",
    }
    result = example_programs.match(row, text, None, [])
    assert result is None or result[0] != "example_name"


@pytest.mark.parametrize(
    "text,folder", [("AttnEpilogue", "attn_epilogue"), ("EngramGate", "engram_gate")]
)
def test_operator_names_resolve_complete_program_roles(text, folder):
    result = query.report(text, kinds=["example"], limit=1)
    (row,) = result["knowledge"]["items"]
    assert row["id"] == f"example:algorithms/{folder}/kernel.py"
    assert set(row["program"]["entries"]) == {"kernel", "reference", "test"}
    assert row["condition_sources"] and row["missing_facts"]
    (resolved,) = query.report(**row["resolve"])["knowledge"]["items"]
    assert (
        resolved["conditions"] and resolved["validation"]["status"] == "not_established"
    )


@pytest.mark.parametrize(
    "text,relative,dtype",
    [
        ("per-column statistics", "algorithms/column_statistics", "FP16"),
        ("row selector matrix product", "algorithms/indexed_batched_gemm", "FP32"),
        (
            "row softmax runtime prefix length empty rows physical tail padding",
            "algorithms/masked_softmax",
            "BF16",
        ),
        ("运行期前缀softmax", "algorithms/masked_softmax", "BF16"),
        (
            "L0B reuse across two matmul operations, explicit operand movement, and matmul chain",
            "matmul/l0_operand_reuse",
            "BF16",
        ),
        ("保留L0操作数", "matmul/l0_operand_reuse", "BF16"),
    ],
)
def test_program_scenario_labels_find_original_roles_and_keep_dtype_conditions(
    text, relative, dtype
):
    view = query.report(text, kinds=["example"])["knowledge"]
    row = view["items"][0]
    assert row["id"] == f"example:{relative}/kernel.py"
    assert set(row["program"]["entries"]) == {"kernel", "reference", "test"}
    config = host_io.policy(workspace=EXAMPLES_ROOT, examples_root=EXAMPLES_ROOT)
    for role, request in row["program"]["read_requests"].items():
        read = host_io.read_requests([request], config=config)
        assert read["items"][0]["sha256"] == row["program"]["entries"][role]["sha256"]
    assert row["validation"]["status"] == "not_established"
    filtered = query.report(text, kinds=["example"], filters={"source_dtype": dtype})[
        "knowledge"
    ]
    assert row["id"] not in {item["id"] for item in filtered["items"]}
    assert row["id"] in {item["id"] for item in filtered["rejected_conditions"]}


def test_long_program_miss_offers_pinned_recovery_without_promoting_a_match(capsys):
    from _checks import cli

    text = "gathered dot product matrix multiplication repeated row selector"
    result = query.report(text, kinds=["example"], limit=1)
    view = result["knowledge"]
    assert view["lookup"]["status"] == "retrieval_miss" and view["items"] == []
    assert view["page"]["total"] == 0 and view["page"]["next_cursor"] is None
    (row,) = view["recovery"]["items"]
    assert row["id"] == "example:algorithms/indexed_batched_gemm/kernel.py"
    assert (
        "selector" in row["matched_terms"]
        and "multiplication" in row["unmatched_terms"]
    )
    assert set(row["program"]["entries"]) == {"kernel", "reference", "test"}
    assert set(row["program"]["read_requests"]) == {"kernel", "reference", "test"}
    assert (
        row["evidence"]["native"] == row["evidence"]["numerical"] == "not_established"
    )
    original = query.report(**row["resolve"])["knowledge"]
    assert original["items"][0]["id"] == row["id"]
    expected = Path(row["program"]["read_requests"]["kernel"]["path"]).read_bytes()
    content, received = original["content"], bytearray()
    config = host_io.policy(workspace=EXAMPLES_ROOT, examples_root=EXAMPLES_ROOT)
    while True:
        assert content["status"] in {"partial", "complete"}
        previous_offset = len(received)
        for item in content["items"]:
            assert item["sha256"] == row["program"]["entries"]["kernel"]["sha256"]
            assert item["offset_bytes"] == len(received)
            received.extend(item["text"].encode("utf-8"))
        if content["status"] == "complete":
            break
        assert len(received) > previous_offset
        next_read = {
            key: value
            for key, value in content["next"].items()
            if key != "request_index"
        }
        assert next_read["offset_bytes"] == len(received) < len(expected)
        content = host_io.read_requests([next_read], config=config)
    assert bytes(received) == expected
    assert cli.main(["query", text, "--kind", "example", "--limit", "1"]) == 2
    output = capsys.readouterr().out
    assert (
        row["id"] in output
        and "retrieval_miss" in output
        and "Recovery program sources:" in output
    )


@pytest.mark.parametrize(
    "text,kinds",
    [
        (
            "VF indexed read and write for fp32 values with uint32 indices into UB Tile duplicate indices",
            ["example"],
        ),
        ("pl.missing_gather", ["example"]),
        ("unrelated pizza kernel", ["example"]),
        (
            "gathered dot product matrix multiplication repeated row selector",
            ["document"],
        ),
        ("gathered matrix product UB to L1 repeated selectors", ["example"]),
    ],
)
def test_program_recovery_keeps_explicit_api_direction_and_source_kind_boundaries(
    text, kinds
):
    assert "recovery" not in query.report(text, kinds=kinds)["knowledge"]


def test_program_recovery_preserves_exclusions_filters_and_source_refresh(library):
    root, refs, _ = library
    path = root / "algorithms/batched/kernel.py"
    before = path.read_text()
    metadata = {
        "facets": {"source_dtype": ["FP32"]},
        "exclude": [
            {
                "term_groups": [["unsupported tail"]],
                "reason": "This fixture excludes this source condition.",
            }
        ],
    }
    path.write_text(
        '"""Gather indexed matrix operands with a repeated row selector.\nexample-query: '
        + json.dumps(metadata)
        + '\n"""\n'
        + before[before.index("import "):]
    )
    text = "gathered dot product matrix multiplication repeated row selector"
    (row,) = discover(library, text=text)["recovery"]["items"]
    assert row["id"] == "example:algorithms/batched/kernel.py"
    (unknown,) = discover(library, text=text, filters={"layout": "NZ"})["recovery"][
        "items"
    ]
    assert unknown["applicability"]["unknown_filters"] == ["layout"]
    assert "recovery" not in discover(
        library, text=text, filters={"source_dtype": "BF16"}
    )
    assert "recovery" not in discover(library, text=text + " unsupported tail")
    path.write_text(before)
    refreshed = discover(library, text=text)
    assert (
        "recovery" not in refreshed
        and row["reference"]["sha256"] != feedback.file_identity(path)["sha256"]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "分段前缀和需要重置标记跨越寄存器和UB分块，保留局部状态与块间传递",
            "streaming_segmented_scan",
        ),
        (
            "计算很长输入行的均值和总体方差，包含大偏置与抵消，需要固定大小分块和尾部处理",
            "streaming_moments",
        ),
    ],
)
def test_chinese_scenario_recovery_exposes_original_condition_reads_without_accepting(
    text, expected
):
    view = query.report(text, kinds=["example"])["knowledge"]
    assert view["lookup"]["status"] == "retrieval_miss" and not view["items"]
    row = view["recovery"]["items"][0]
    assert row["id"] == f"example:algorithms/{expected}/kernel.py"
    assert row["missing_facts"] and row["evidence"]["performance"] == "not_established"
    for condition in row["condition_sources"]:
        request = condition["read"]
        report = host_io.read_requests(
            [request], config=host_io.policy(workspace=EXAMPLES_ROOT)
        )
        assert report["items"][0]["sha256"] == condition["reference"]["sha256"]
        assert report["items"][0]["start_line"] == condition["reference"]["line"]


@pytest.mark.parametrize(
    "text",
    ["请给我一个巧克力蛋糕配方，适合在新加坡的夏天烘焙", "FP32 123 456 789 01234"],
)
def test_recovery_does_not_rank_unrelated_chinese_or_incidental_numbers(text):
    assert "recovery" not in query.report(text, kinds=["example"])["knowledge"]


@pytest.mark.parametrize("namespace", ["VF SIMT", "pl.simt"])
def test_namespace_only_program_context_uses_owner_scenario_clues(namespace, tmp_path):
    text = "segmented softmax weighted sum ragged segments scan reduce " + namespace
    framework = tmp_path / "missing-framework"
    view = query.report(text, kinds=["example"], limit=1, pypto_root=framework)[
        "knowledge"
    ]
    assert view["lookup"]["status"] == "retrieval_miss" and view["items"] == []
    assert not view["coverage"]["api"]["complete"]
    assert view["coverage"]["api"]["errors"]
    (row,) = view["recovery"]["items"]
    assert row["id"] == "example:algorithms/csr_softmax_aggregate/kernel.py"
    assert row["condition_sources"] and row["facets"]["source_dtype"] == ["FP32"]
    assert row["evidence"]["native"] == "not_established"
    conflict = query.report(
        text,
        kinds=["example"],
        filters={"source_dtype": "BF16"},
        pypto_root=framework,
    )["knowledge"]
    assert row["id"] not in {
        r["id"] for r in conflict.get("recovery", {}).get("items", [])
    }
    backward = query.report(
        text + " backward", kinds=["example"], pypto_root=framework
    )["knowledge"]
    assert row["id"] not in {
        r["id"] for r in backward.get("recovery", {}).get("items", [])
    }


@pytest.mark.parametrize(
    "symbol",
    [
        "vf.reduce_sum",
        "pl.simt.atomic_add",
        "pl.simt.unknown_api",
        "pl.unknown_namespace",
    ],
)
def test_named_api_is_not_replaced_with_partial_program_clues(symbol, tmp_path):
    text = symbol + " segmented softmax weighted sum ragged segments"
    assert (
        "recovery"
        not in query.report(
            text, kinds=["example"], pypto_root=tmp_path / "missing-framework"
        )["knowledge"]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "VF Vector per-column reduction across rows for contiguous 2D FP32"
            " input; inspect suitable complete example and explicit buffers/stages",
            "column_statistics",
        ),
        (
            "stable ordered compaction using VF vector registers, prefix ranks and tail initialization",
            "stable_filter",
        ),
    ],
)
def test_long_scenario_identity_clues_survive_unmatched_implementation_words(
    text, expected
):
    view = query.report(text, kinds=["example"])["knowledge"]
    assert view["lookup"]["status"] == "retrieval_miss" and view["items"] == []
    (row,) = view["recovery"]["items"]
    assert row["id"] == f"example:algorithms/{expected}/kernel.py"
    assert row["evidence"]["native"] == "not_established"
    assert set(row["program"]["read_requests"]) == {"kernel", "reference", "test"}


@pytest.mark.parametrize(
    "text",
    [
        "UB Vec TileType capacity streaming tile",
        "VF UB Vec TileType capacity streaming tile",
        "tile buffers capacity address layout and slots",
    ],
)
def test_resource_words_alone_do_not_identify_an_algorithm(text):
    assert "recovery" not in query.report(text, kinds=["example"])["knowledge"]


def test_declared_scenario_clues_exclude_weaker_lexical_recovery_candidates():
    text = "segmented softmax weighted sum ragged segments scan reduce VF SIMT"
    rows = query.report(text, kinds=["example"], limit=3)["knowledge"]["recovery"][
        "items"
    ]
    assert [row["id"] for row in rows] == [
        "example:algorithms/csr_softmax_aggregate/kernel.py"
    ]


def test_missing_stdlib_inventory_returns_a_runtime_diagnostic(
    library, monkeypatch, capsys
):
    from _checks import cli

    root, _, framework = library
    monkeypatch.delattr(example_programs.sys, "stdlib_module_names")
    code = cli.main(
        [
            "--pypto-root",
            str(framework),
            "--examples-root",
            str(root),
            "query",
            "pl.DT_FP16",
            "--format",
            "json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 2 and result["status"] == "unverified"
    assert result["knowledge"]["lookup"]["reason"] == "example_runtime_unavailable"
    assert result["knowledge"]["lookup"]["knowledge_absence"] == "not_established"
    assert "Python 3.10+" in result["knowledge"]["lookup"]["message"]
    assert (
        query.collect(["api"], pypto_root=framework, examples_root=root)[
            "example_coverage"
        ]
        is None
    )


def test_full_program_reuses_calls_and_catalog_without_importing(library):
    view = discover(library, text="batched transposed GEMM with M tails")
    assert view["lookup"]["status"] == "matched"
    (row,) = view["items"]
    assert row["id"] == "example:algorithms/batched/kernel.py"
    assert row["match"]["reason"] == "example_source_terms"
    assert row["example_roles"] == ["algorithm"]
    assert set(row["program"]["entries"]) == {"kernel", "reference", "test"}
    assert row["selection"][0]["line"] == 5
    assert row["dependencies"]["external_modules"] == [
        "pypto_pro.language",
        "pytest",
        "torch",
        "torch_npu",
    ]
    assert row["dependencies"]["environment_variables"] == [
        "OPTIONAL_EVALUATOR_ROOT",
        "TILE_FWK_DEVICE_ID",
    ]
    assert row["validation"]["status"] == "not_established"
    resolved = discover(library, **row["resolve"])
    full = resolved["items"][0]["program"]
    paths = {Path(file["path"]).name for file in full["files"]}
    assert {"compute.py", "deeper.py", "conftest.py", "__init__.py"} <= paths
    assert full["dependencies"]["unresolved"] == []
    assert {item["role"] for item in full["source_statements"]} == {
        "kernel",
        "reference",
        "test",
    }
    assert "Independent FP64" in next(
        item["text"]
        for item in full["source_statements"]
        if item["role"] == "reference"
    )
    assert "must not import" in resolved["content"]["items"][0]["text"]
    assert set(row["evidence"].values()) == {"native_example_source", "not_established"}
    config = host_io.policy(workspace=library[0])
    for role in ("reference", "test"):
        read = host_io.read_requests([full["read_requests"][role]], config=config)
        assert read["items"][0]["sha256"] == row["program"]["entries"][role]["sha256"]


def test_program_navigation_words_preserve_computation_constraints(library):
    view = discover(
        library,
        text="find independent reference and tests for batched transposed GEMM with M tails",
    )
    assert [row["id"] for row in view["items"]] == [
        "example:algorithms/batched/kernel.py"
    ]
    unrelated = discover(
        library,
        text="find independent reference and tests for batched transposed GEMM backward",
    )
    assert not unrelated["items"]
    assert not discover(library, text="find independent reference and tests")["items"]


def test_text_cli_exposes_independent_program_entries(library, capsys):
    root, _, framework = library
    result = query.report(
        "example:algorithms/batched", examples_root=root, pypto_root=framework
    )
    feedback.print_result(result, json_output=False)
    output = capsys.readouterr().out
    assert "Program roles: algorithm" in output
    assert f"reference: {root}/algorithms/batched/reference.py" in output
    assert f"test: {root}/algorithms/batched/test.py" in output
    assert "must not import any example" in output


def test_quoted_directory_locators_roundtrip_without_private_helpers_becoming_programs(
    library,
):
    root, refs, _ = library
    old = root / "algorithms/batched"
    name = "批量.fp16 #%"
    old.rename(old.with_name(name))
    catalog_path = refs / "tooling/examples.md"
    from urllib.parse import quote

    encoded = quote(name)
    catalog_path.write_text(
        catalog_path.read_text().replace(
            "algorithms/batched/", "algorithms/" + encoded + "/"
        )
    )
    row = next(
        row
        for row in discover(library)["items"]
        if row["example_group"] == "algorithms"
    )
    assert row["id"] == "example:algorithms/" + encoded + "/kernel.py"
    assert discover(library, **row["resolve"])["content"]["items"]


def test_documentation_backlink_resolves_its_own_heading_not_the_kernel_anchor(library):
    root, refs, framework = library
    write(
        refs / "ownership.md",
        f"# Guidance\n\n## Program scope\n\n[factory]({root}/algorithms/batched/kernel.py#factory)\n",
    )
    (row,) = discover(library, text="example:algorithms/batched")["items"]
    relation = {
        "kind": "documentation",
        "target": "document:skill/ownership.md#program-scope",
    }
    assert relation in row["relations"]
    original = query.report(
        relation["target"], kinds=["document"], examples_root=root, pypto_root=framework
    )["knowledge"]
    assert original["lookup"]["status"] == "matched"
    assert "[factory]" in original["content"]["items"][0]["text"]


@pytest.mark.parametrize(
    "name",
    ["vf.add", "pl.Vf.add", "pl.load", "pl.system.bar_all", "pl.simt.atomic_add"],
)
def test_all_groups_and_transitive_support_calls_are_discoverable(library, name):
    rows = discover(library, text=name)["items"]
    assert len(rows) == 3
    assert {row["example_group"] for row in rows} == set(example_programs.GROUPS)
    assert all(row["match"]["reason"] == "example_api_call" for row in rows)
    assert {role for row in rows for role in row["example_roles"]} == {
        "minimal_api",
        "mechanism",
        "algorithm",
    }
    assert not discover(library, text="pl.DT_FP16")["items"]


def test_counterexamples_and_dependency_unknowns_remain_explicit(library):
    root, _, _ = library
    path = root / "basic_api/chapter/minimal/kernel.py"
    path.write_text(
        path.read_text().replace(
            "raise RuntimeError",
            '''
import importlib
def bad(x):
    """反例：api-contract —— teaching only; never run on board"""
    return dsl.insert(x, x)
def evaluator(name):
    return importlib.import_module(name)
raise RuntimeError''',
        )
    )
    (row,) = discover(library, text="pl.insert")["items"]
    assert "negative_control" in row["example_roles"]
    assert row["negative_controls"][0]["name"] == "bad"
    assert row["match"]["reason"] == "negative_control_api"
    resolved = discover(library, **row["resolve"])["items"][0]
    assert resolved["negative_controls"][0]["claims"] == {
        "api-contract": "teaching only; never run on board"
    }
    full = resolved["program"]
    assert full["dependencies"]["unresolved"][0]["reason"] == "dynamic_import"


def test_owner_semantics_keep_conditions_separate_from_source_calls(library):
    root, _, _ = library
    path = root / "algorithms/batched/kernel.py"
    meta = {
        "aliases": ["批量转置矩阵乘的M尾块"],
        "term_groups": [["batched"], ["transposed"], ["GEMM"]],
        "facets": {"source_dtype": ["FP16"]},
        "exclude": [
            {
                "term_groups": [["N tail", "N tails", "K tail", "K tails"]],
                "reason": "M tails only; N/K must align.",
            }
        ],
    }
    path.write_text(
        path.read_text().replace(
            'N aligned."""',
            "N aligned.\nexample-query: "
            + json.dumps(meta, ensure_ascii=False)
            + '\n"""',
        )
    )
    (row,) = discover(library, text="批量转置矩阵乘的M尾块")["items"]
    assert row["match"]["reason"] == "example_alias"
    assert (
        "example-query"
        not in discover(library, **row["resolve"])["items"][0]["conditions"][0]
    )
    assert discover(library, text="batched transposed GEMM N tails")["items"] == []
    assert discover(library, text="batched transposed GEMM N tails")[
        "rejected_conditions"
    ][0]["source_conditions"]
    rejected = discover(
        library, text="批量转置矩阵乘的M尾块", filters={"source_dtype": "FP32"}
    )
    assert not rejected["items"] and rejected["rejected_conditions"]
    (row,) = discover(
        library,
        text="批量转置矩阵乘的M尾块",
        filters={"source_dtype": "FP16", "layout": "NZ"},
    )["items"]
    assert row["applicability"]["matched_filters"] == {"source_dtype": "FP16"}
    assert row["applicability"]["unknown_filters"] == ["layout"]


def test_source_and_support_edits_invalidate_cursors_and_pinned_reads(library):
    page = discover(library, limit=1)
    cursor = page["page"]["next_cursor"]
    row = page["items"][0]
    path = Path(row["reference"]["path"])
    stamp = path.stat()
    path.write_text(path.read_text().replace("rows and", "rows plus"))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="stale"):
        discover(library, limit=1, cursor=cursor)
    assert discover(library, **row["resolve"])["lookup"]["status"] == "source_changed"
    current = discover(library, limit=1)
    full = discover(library, **current["items"][0]["resolve"])["items"][0]
    helper = next(
        file
        for file in full["program"]["files"]
        if Path(file["path"]).name == "deeper.py"
    )
    Path(helper["path"]).write_text("def helper():\n    return 2\n")
    with pytest.raises(ValueError, match="stale"):
        discover(library, limit=1, cursor=current["page"]["next_cursor"])
    changed = discover(library, text=full["id"])["items"][0]
    assert changed["program"]["program_sha256"] != full["program"]["program_sha256"]
    with pytest.raises(ValueError, match="Source changed"):
        host_io.read_requests(
            [helper["read"]], config=host_io.policy(workspace=library[0])
        )


@pytest.mark.parametrize("action", ["add", "rename", "remove"])
def test_inventory_and_catalog_refresh_together(library, action):
    root, refs, _ = library
    first = discover(library)
    folder = root / "algorithms/batched"
    catalog_path = refs / "tooling/examples.md"
    if action == "remove":
        import shutil

        shutil.rmtree(folder)
        catalog_path.write_text(
            "\n".join(
                line
                for line in catalog_path.read_text().splitlines()
                if "algorithms/batched/" not in line
            )
        )
    elif action == "rename":
        folder.rename(folder.with_name("renamed"))
        catalog_path.write_text(
            catalog_path.read_text().replace("algorithms/batched/", "algorithms/renamed/")
        )
    else:
        import shutil

        shutil.copytree(folder, folder.with_name("added"))
        catalog_path.write_text(
            catalog_path.read_text()
            + "| Added | [added](../../examples/algorithms/added/) |\n"
        )
    current = discover(library)
    assert current["corpus_sha256"] != first["corpus_sha256"]
    assert current["coverage"]["examples"]["complete"]
    assert current["page"]["total"] == {"add": 4, "rename": 3, "remove": 2}[action]
    assert all("batched/kernel.py" not in row["id"] for row in current["items"]) == (
        action != "add"
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_reference",
        "bad_metadata",
        "syntax",
        "escaped_kernel",
        "symlink_loop",
        "broken_import",
        "scan_limit",
    ],
)
def test_partial_sources_do_not_claim_a_complete_provider(
    library, problem, monkeypatch, tmp_path
):
    root, _, _ = library
    folder = root / "algorithms/batched"
    if problem == "missing_reference":
        (folder / "reference.py").unlink()
    elif problem == "bad_metadata":
        write(folder / "kernel.py", '"""example-query: {"unrecognized":true}"""\n')
    elif problem == "syntax":
        write(folder / "kernel.py", "invalid !!!")
    elif problem == "escaped_kernel":
        (folder / "kernel.py").unlink()
        (folder / "kernel.py").symlink_to(
            write(tmp_path / "outside.py", "DO_NOT_READ = True\n")
        )
    elif problem == "symlink_loop":
        (folder / "kernel.py").unlink()
        (folder / "kernel.py").symlink_to("kernel.py")
    elif problem == "broken_import":
        write(folder / "kernel.py", "from .missing import builder\n")
    else:
        monkeypatch.setitem(host_io.CEILINGS, "max_files", 2)
    view = discover(library)
    if problem == "broken_import":
        row = next(row for row in view["items"] if row["title"] == "batched")
        assert discover(library, **row["resolve"])["items"][0]["program"][
            "dependencies"
        ]["unresolved"]
    assert not view["coverage"]["examples"]["complete"]
    assert view["coverage"]["examples"]["errors"]
    assert view["lookup"]["knowledge_absence"] == "not_established"


@pytest.mark.parametrize(
    "identity",
    [
        "example:/etc/kernel.py",
        "example:algorithms/../secret/kernel.py",
        "example:algorithms/%2e%2e/secret",
        "example:algorithms/_kernels/kernel.py",
        "example:algorithms/batched/test.py",
        "example:unknown/batched",
        "example:algorithms/batched/kernel.py#part",
    ],
)
def test_invalid_locators_are_rejected(library, identity):
    with pytest.raises(ValueError):
        discover(library, text=identity)


def test_repository_query_connects_target_and_excludes_neighbor_conditions():
    (row,) = query.report("batched transposed GEMM with M tails", kinds=["example"])[
        "knowledge"
    ]["items"]
    assert row["id"] == "example:algorithms/batched_transposed_gemm/kernel.py"
    resolved = query.report(**row["resolve"])["knowledge"]["items"][0]
    assert "N 为 16" in resolved["conditions"][0]
    assert any(
        "n % 16" in guard["condition"] for guard in resolved["program"]["source_guards"]
    )
    assert (
        Path(row["program"]["entries"]["reference"]["path"])
        == EXAMPLES_ROOT / "algorithms/batched_transposed_gemm/reference.py"
    )
    assert query.report("batched transposed GEMM with N tails", kinds=["example"])[
        "knowledge"
    ]["rejected_conditions"]
    assert not query.report(
        "带M尾块的批量转置GEMM", kinds=["example"], filters={"source_dtype": "BF16"}
    )["knowledge"]["items"]


@pytest.mark.parametrize(
    "text, program, dtype, incompatible",
    [
        (
            "RMS normalization and GEMM with M tails",
            "rmsnorm_gemm_tail",
            "FP16",
            "RMS normalization and GEMM with N tails",
        ),
        (
            "CSR sparse softmax aggregation",
            "csr_softmax_aggregate",
            "FP32",
            "CSR sparse softmax aggregation backward",
        ),
        (
            "CSR softmax weighted sum",
            "csr_softmax_aggregate",
            "FP32",
            "CSR softmax weighted sum gradient",
        ),
    ],
)
def test_repository_scenario_aliases_preserve_original_roles_and_limits(
    text, program, dtype, incompatible
):
    identity = f"example:algorithms/{program}/kernel.py"
    view = query.report(
        text, kinds=["example"], filters={"source_dtype": dtype, "layout": "NZ"}
    )["knowledge"]
    row = next(row for row in view["items"] if row["id"] == identity)
    assert row["applicability"]["unknown_filters"] == ["layout"]
    resolved = query.report(**row["resolve"])["knowledge"]["items"][0]
    assert set(resolved["program"]["entries"]) == {"kernel", "reference", "test"}
    assert resolved["validation"]["status"] == "not_established"
    rejected = query.report(incompatible, kinds=["example"])["knowledge"]
    assert identity not in {item["id"] for item in rejected["items"]}
    assert any(
        item["id"] == identity and item["source_conditions"]
        for item in rejected["rejected_conditions"]
    )
    wrong_dtype = query.report(
        text, kinds=["example"], filters={"source_dtype": "BF16"}
    )["knowledge"]
    assert identity not in {item["id"] for item in wrong_dtype["items"]}


def test_contract_and_program_links_are_bidirectional():
    view = query.report("UB to L1")["knowledge"]
    rows = {row["id"]: row for row in view["items"]}
    program = rows["example:basic_api/datamove/ub_l1_handover/kernel.py"]
    assert {"kind": "contract", "target": "contract:ub-l1"} in program["relations"]
    assert any(
        relation["target"] == program["id"]
        for relation in rows["contract:ub-l1"]["relations"]
    )
    assert view["coverage"]["examples"]["groups"] == {
        group: sum(1 for _ in catalog.public_kernels(EXAMPLES_ROOT, (group,)))
        for group in example_programs.GROUPS
    }
