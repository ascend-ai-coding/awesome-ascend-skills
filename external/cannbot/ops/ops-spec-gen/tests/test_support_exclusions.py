# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Confirmed exclusions survive generation and cannot contradict support declarations."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from generate_spec import (
    GenInput,
    OutputSpec,
    TensorSpec,
    apply_unsupported_arg,
    build_supported_combinations_block,
    collect_input_dtypes,
    collect_unsupported,
    load_registries,
    parse_tensor_arg,
    render,
)
from validate_spec import (
    stage_1,
    stage_2,
    stage_7,
    load_registries as validation_registries,
)
from support_catalog import load_catalog, partition, universe

ROOT = Path(__file__).resolve().parent.parent
BASE = yaml.safe_load((ROOT / "examples/add/spec.yaml").read_text(encoding="utf-8"))


def findings(spec):
    return stage_2(spec, validation_registries())


def test_explicit_input_and_output_exclusions_survive_generation():
    inputs = [TensorSpec("x", ["float32"])]
    outputs = [OutputSpec("y")]
    reg = load_registries()
    apply_unsupported_arg(
        "input.x:int8,int32;output.y:complex64",
        inputs,
        outputs,
        "unsupported_dtypes",
        reg["dtypes"],
    )
    apply_unsupported_arg(
        "input.x:NZ;output.y:FRACTAL_Z",
        inputs,
        outputs,
        "unsupported_layouts",
        reg["layouts"],
        official=True,
    )
    spec = yaml.safe_load(
        render(
            GenInput(
                "neg",
                "Negation",
                "Elementwise",
                ["Elementwise"],
                inputs,
                outputs,
                broadcast_kind="none",
            )
        )
    )
    assert spec["inputs"][0]["unsupported_dtypes"] == ["int8", "int32"]
    assert spec["outputs"][0]["unsupported_dtypes"] == ["complex64"]
    assert spec["inputs"][0]["unsupported_layouts"] == ["FRACTAL_NZ"]
    assert spec["outputs"][0]["unsupported_layouts"] == ["FRACTAL_Z"]
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"


def test_unknown_scope_does_not_infer_exclusions():
    spec = yaml.safe_load(
        render(
            GenInput(
                "neg",
                "Negation",
                "Elementwise",
                ["Elementwise"],
                [TensorSpec("x", ["float32"])],
                ["y"],
            )
        )
    )
    for tensor in spec["inputs"] + spec["outputs"]:
        assert tensor["unsupported_dtypes"] == []
        assert tensor["unsupported_layouts"] == []


def test_supported_combinations_preserve_heterogeneous_index_dtype():
    block = build_supported_combinations_block(
        [TensorSpec("x", ["float32"]), TensorSpec("indices", ["int64"])],
        ["y"],
        "same_as_first_input",
    )

    rows = yaml.safe_load(block)
    assert rows[0] == {
        "inputs": {"x": "float32", "indices": "int64"},
        "outputs": {"y": "float32"},
    }


@pytest.mark.parametrize(
    "input_arg,expected_inputs,expected_dtype",
    [
        (
            "x:float16,float32;z:float32,float16",
            {"x": "float16", "z": "float32"},
            "float32",
        ),
        (
            "x:float32,float16;z:float16,float32",
            {"x": "float32", "z": "float16"},
            "float32",
        ),
        (
            "x:bfloat16,float16;z:float16,bfloat16",
            {"x": "bfloat16", "z": "float16"},
            "float32",
        ),
        (
            "x:float16;z:int16",
            {"x": "float16", "z": "int16"},
            "float32",
        ),
        (
            "x:int8;z:uint8",
            {"x": "int8", "z": "uint8"},
            "int16",
        ),
        (
            "x:int64;z:float32",
            {"x": "int64", "z": "float32"},
            "float64",
        ),
    ],
)
def test_cli_promotes_baseline_and_deduplicates_input_combinations(
    tmp_path, input_arg, expected_inputs, expected_dtype
):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--op-name",
            "add",
            "--category",
            "Broadcast",
            "--paradigms",
            "Elementwise,Broadcast",
            "--inputs",
            input_arg,
            "--outputs",
            "y",
            "--promotion",
            "numpy_promote",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    spec = yaml.safe_load((tmp_path / "spec.yaml").read_text(encoding="utf-8"))
    rows = spec["dtype_policy"]["supported_combinations"]
    matching = [row for row in rows if row["inputs"] == expected_inputs]
    assert len(matching) == 1
    assert matching[0]["outputs"] == {"y": expected_dtype}
    keys = [tuple(sorted(row["inputs"].items())) for row in rows]
    assert len(keys) == len(set(keys))
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"
    assert stage_7(spec).status == "PASS"
    assert set(spec["numerical_tolerance"]["per_dtype"]) == {
        dtype for row in rows for dtype in row["outputs"].values()
    }


@pytest.mark.parametrize(
    "dtype",
    [
        value
        for value in universe("dtypes")
        if value not in load_catalog()["non_tensor_dtypes"]
    ],
)
def test_same_dtype_combinations_preserve_official_types_without_promotion(dtype):
    rows = yaml.safe_load(
        build_supported_combinations_block(
            [TensorSpec("x", [dtype]), TensorSpec("z", [dtype])],
            ["y"],
            "numpy_promote",
        )
    )
    assert rows == [{"inputs": {"x": dtype, "z": dtype}, "outputs": {"y": dtype}}]


def test_cli_unknown_mixed_dtype_promotion_fails_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--op-name",
            "add",
            "--category",
            "Broadcast",
            "--paradigms",
            "Elementwise,Broadcast",
            "--inputs",
            "x:float6_e3m2;z:float32",
            "--outputs",
            "y",
            "--promotion",
            "numpy_promote",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "numpy_promote" in result.stderr and "float6_e3m2" in result.stderr
    assert not (tmp_path / "spec.yaml").exists()


@pytest.mark.parametrize("dtype", ["float6_e3m2", "float6_e2m3"])
def test_cli_preserves_same_special_dtype_and_covers_tolerance(tmp_path, dtype):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--complete-support-contract",
            "--op-name",
            "add",
            "--category",
            "Broadcast",
            "--paradigms",
            "Elementwise,Broadcast",
            "--inputs",
            f"x:{dtype};z:{dtype}",
            "--outputs",
            "y",
            "--promotion",
            "numpy_promote",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    spec = yaml.safe_load((tmp_path / "spec.yaml").read_text(encoding="utf-8"))
    assert spec["outputs"][0]["dtype_set"] == [dtype]
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"
    assert stage_7(spec).status == "PASS"


def test_cli_partial_exclusions_preserve_only_confirmed_assertions(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--op-name",
            "neg",
            "--category",
            "Elementwise",
            "--paradigms",
            "Elementwise",
            "--broadcast-kind",
            "none",
            "--inputs",
            "x:float32",
            "--outputs",
            "y",
            "--unsupported-dtypes",
            "input.x:int8;output.y:int8",
            "--unsupported-formats",
            "input.x:NCHW;output.y:NHWC",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    spec = yaml.safe_load((tmp_path / "spec.yaml").read_text(encoding="utf-8"))
    for tensor in spec["inputs"] + spec["outputs"]:
        for kind, positive, negative in (
            ("dtypes", "dtype_set", "unsupported_dtypes"),
            ("formats", "supported_layouts", "unsupported_layouts"),
        ):
            expected = ["int8"] if kind == "dtypes" else ["NCHW"] if tensor is spec["inputs"][0] else ["NHWC"]
            assert tensor[negative] == expected
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"


def test_numpy_promotion_excludes_axis_tensor_from_baseline():
    rows = yaml.safe_load(
        build_supported_combinations_block(
            [
                TensorSpec("x", ["float16", "float32"]),
                TensorSpec("z", ["float32", "float16"]),
                TensorSpec("axes", ["int64"]),
            ],
            ["y"],
            "numpy_promote",
            axis_source="input_tensor",
        )
    )
    assert rows[0] == {
        "inputs": {"x": "float16", "z": "float32", "axes": "int64"},
        "outputs": {"y": "float32"},
    }


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("side", ["inputs", "outputs"])
@pytest.mark.parametrize("value", [7, [{}], [[]], [None, {}, []]])
@pytest.mark.parametrize("stage_args", [[], ["--stage", "1", "--stage", "2"]])
def test_invalid_supported_layouts_returns_json_diagnostics(
    tmp_path, complete, side, value, stage_args
):
    spec = full_spec() if complete else copy.deepcopy(BASE)
    spec[side][0]["supported_layouts"] = value
    path = tmp_path / "invalid-supported-layouts.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/validate_spec.py"),
            str(path),
            "--json",
            *stage_args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    stages = {stage["stage_id"]: stage for stage in report["stages"]}
    assert stages[1]["status"] == "FAIL"
    assert any(
        "supported_layouts" in finding["field_path"]
        for finding in stages[1]["findings"]
    )
    assert stages[2]["status"] == "PASS"


@pytest.mark.parametrize("field", ["unsupported_dtypes", "unsupported_layouts"])
def test_invalid_exclusion_scalar_returns_json_diagnostics(tmp_path, field):
    spec = copy.deepcopy(BASE)
    spec["inputs"][0][field] = 7
    path = tmp_path / "invalid-exclusions.yaml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/validate_spec.py"),
            str(path),
            "--json",
            "--stage",
            "1",
            "--stage",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["stages"]
    assert any(
        item["stage_id"] == 1 and item["status"] == "FAIL" for item in report["stages"]
    )


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("inputs", "unsupported_dtypes", ["made_up"]),
        ("outputs", "unsupported_layouts", ["BSND"]),
        ("inputs", "unsupported_dtypes", ["int8", "int8"]),
        ("outputs", "unsupported_layouts", "NZ"),
    ],
)
def test_schema_rejects_invalid_exclusions(side, field, value):
    spec = copy.deepcopy(BASE)
    spec[side][0][field] = value
    assert stage_1(spec).status == "FAIL"


@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_dtype_conflict_in_supported_combination(side):
    spec = copy.deepcopy(BASE)
    spec[side][0]["unsupported_dtypes"] = ["float32"]
    result = findings(spec)
    assert result.status == "FAIL"
    assert any(
        f.rule_id == "support_exclusions.dtype_conflict" for f in result.findings
    )


def test_dtype_set_conflict_without_combination_coverage():
    spec = copy.deepcopy(BASE)
    spec["inputs"][0]["dtype_set"].append("int8")
    spec["inputs"][0]["unsupported_dtypes"] = ["int8"]
    assert any(
        f.rule_id == "support_exclusions.dtype_conflict"
        for f in findings(spec).findings
    )


@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_layout_conflict(side):
    spec = copy.deepcopy(BASE)
    spec[side][0]["unsupported_layouts"] = ["ND"]
    assert any(
        f.rule_id == "support_exclusions.layout_conflict"
        for f in findings(spec).findings
    )


def test_format_variant_conflicts_only_with_input_exclusion():
    spec = copy.deepcopy(BASE)
    spec["math_semantics"]["format_variants"] = [
        {"format": "NHWC", "rank": [4], "reduction_axes": [0, 1, 2]},
    ]
    spec["outputs"][0]["unsupported_layouts"] = ["NHWC"]
    assert not any(
        f.rule_id == "support_exclusions.layout_conflict"
        for f in findings(spec).findings
    )
    spec["inputs"][0]["unsupported_layouts"] = ["NHWC"]
    assert any(
        f.rule_id == "support_exclusions.layout_conflict"
        for f in findings(spec).findings
    )


@pytest.mark.parametrize(
    "raw",
    [
        "input.missing:int8",
        "x:int8",
        "input.x:nope",
        "input.x:",
        "input.x:int8,int8",
        "input.x:int8;input.x:int32",
    ],
)
def test_cli_exclusion_parser_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        apply_unsupported_arg(
            raw,
            [TensorSpec("x", ["float32"])],
            [OutputSpec("y")],
            "unsupported_dtypes",
            load_registries()["dtypes"],
        )


def test_interactive_exclusions_validate_and_preserve_answers(monkeypatch):
    answers = iter(["invalid", "DT_INT8,int32", "CSR", "NZ"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert collect_unsupported(load_registries()) == {
        "unsupported_dtypes": ["int8", "int32"],
        "unsupported_layouts": ["FRACTAL_NZ"],
    }


def test_interactive_input_dtypes_normalize_and_reprompt(monkeypatch, capsys):
    answers = iter(["typo", "DT_FLOAT,DT_FLOAT16"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert collect_input_dtypes() == ["float32", "float16"]
    assert "dtype_set 无效" in capsys.readouterr().out


def test_interactive_exclusions_reject_legacy_values_and_duplicate_aliases(
    monkeypatch, capsys
):
    answers = iter(["uint4", "DT_INT8", "CSR", "NZ,FRACTAL_NZ", "FORMAT_ND"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert collect_unsupported(load_registries()) == {
        "unsupported_dtypes": ["int8"],
        "unsupported_layouts": ["ND"],
    }
    output = capsys.readouterr().out
    assert output.count("有效值") == 3
    assert "'uint4'" not in output and "'CSR'" not in output


@pytest.mark.parametrize(
    "field,flag,value",
    [
        ("unsupported_layouts", "--unsupported-formats", "CSR"),
        ("unsupported_dtypes", "--unsupported-dtypes", "uint4"),
        ("supported_layouts", "--supported-formats", "CSR"),
    ],
)
def test_cli_rejects_values_outside_official_universe_at_input(field, flag, value):
    with pytest.raises(ValueError, match=flag):
        apply_unsupported_arg(
            f"input.x:{value}",
            [TensorSpec("x", ["float32"])],
            [],
            field,
            load_registries()["layouts" if "formats" in flag else "dtypes"],
            official=True,
            flag_name=flag,
        )


def test_parse_tensor_arg_normalizes_and_rejects_non_tensor_dtype():
    assert parse_tensor_arg("x:DT_FLOAT").dtype_set == ["float32"]
    with pytest.raises(ValueError, match="Non-tensor"):
        parse_tensor_arg("x:DT_MAX")


def test_legacy_exclusion_yaml_round_trips_null_literal_as_string():
    spec = yaml.safe_load(
        render(
            GenInput(
                "neg",
                "Negation",
                "Elementwise",
                ["Elementwise"],
                [TensorSpec("x", ["float32"], unsupported_layouts=["NULL"])],
                [OutputSpec("y", unsupported_layouts=["NULL"])],
                broadcast_kind="none",
            )
        )
    )
    assert spec["inputs"][0]["unsupported_layouts"] == ["NULL"]
    assert spec["outputs"][0]["unsupported_layouts"] == ["NULL"]


def test_legacy_positive_marker_is_rejected_even_without_chip_scope():
    spec = copy.deepcopy(BASE)
    spec["inputs"][0]["dtype_set"] = ["max"]
    assert stage_1(spec).status == "FAIL"
    assert any(
        f.rule_id == "support_partition.non_tensor" for f in findings(spec).findings
    )

    spec = copy.deepcopy(BASE)
    spec["inputs"][0]["layout"] = "ALL"
    assert stage_1(spec).status == "FAIL"
    assert any(
        f.rule_id == "support_partition.non_tensor" for f in findings(spec).findings
    )

    spec["inputs"][0]["layout"] = "ND"
    spec["inputs"][0]["supported_layouts"] = ["NULL"]
    assert any(
        f.rule_id == "support_partition.non_tensor" for f in findings(spec).findings
    )


def test_cli_refuses_conflicts_before_writing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--op-name",
            "neg",
            "--category",
            "Elementwise",
            "--paradigms",
            "Elementwise",
            "--inputs",
            "x:float32",
            "--outputs",
            "y",
            "--unsupported-dtypes",
            "input.x:float32",
            "--output-dir",
            str(tmp_path / "conflicting"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode != 0
    assert not (tmp_path / "conflicting/spec.yaml").exists()


def test_old_spec_remains_valid_without_exclusions():
    assert stage_1(BASE).status == "PASS"
    assert findings(BASE).status == "PASS"


def full_spec():
    spec = copy.deepcopy(BASE)
    spec["support_universe"] = load_catalog()["id"]
    rows = spec["dtype_policy"]["supported_combinations"]
    for side in ("inputs", "outputs"):
        for tensor in spec[side]:
            dtypes = tensor.get("dtype_set") or sorted(
                {row[side][tensor["name"]] for row in rows}
            )
            tensor["dtype_set"], tensor["unsupported_dtypes"] = partition(
                dtypes, [], "dtypes"
            )
            tensor["supported_layouts"], tensor["unsupported_layouts"] = partition(
                [tensor.get("layout", "ND")], [], "formats"
            )
    return spec


def test_catalog_exactly_matches_supplied_official_enums():
    catalog = load_catalog()
    dtype_doc = (ROOT / "references/datatype.md").read_text(encoding="utf-8")
    format_doc = (ROOT / "references/format.md").read_text(encoding="utf-8")
    dtype_body = dtype_doc.split("enum DataType {", 1)[1].split("};", 1)[0]
    format_body = format_doc.split("enum Format {", 1)[1].split("};", 1)[0]
    assert re.findall(r"^\s*(DT_\w+)\b", dtype_body, re.MULTILINE) == list(
        catalog["dtypes"]
    )
    assert (
        re.findall(r"^\s*FORMAT_(\w+)\b", format_body, re.MULTILINE)
        == catalog["formats"]
    )
    assert len(universe("dtypes")) == 42
    assert len(universe("formats")) == 57
    assert len(set(universe("dtypes"))) == 42
    assert len(set(universe("formats"))) == 57


def test_full_generation_partitions_every_interface_and_preserves_exclusions():
    spec = yaml.safe_load(
        render(
            GenInput(
                "neg",
                "Negation",
                "Elementwise",
                ["Elementwise"],
                [TensorSpec("x", ["float16", "float32"], unsupported_dtypes=["int8"])],
                [OutputSpec("y", supported_layouts=["NCHW"])],
                broadcast_kind="none",
                support_universe=load_catalog()["id"],
            )
        )
    )
    for tensor in spec["inputs"] + spec["outputs"]:
        for kind, positive, negative in (
            ("dtypes", "dtype_set", "unsupported_dtypes"),
            ("formats", "supported_layouts", "unsupported_layouts"),
        ):
            assert set(tensor[positive]).isdisjoint(tensor[negative])
            assert set(tensor[positive]) | set(tensor[negative]) == set(universe(kind))
    assert "string_ref" in spec["inputs"][0]["unsupported_dtypes"]
    assert "float6_e3m2" in spec["inputs"][0]["unsupported_dtypes"]
    assert spec["outputs"][0]["layout"] == "NCHW"
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"


@pytest.mark.parametrize("side", ["inputs", "outputs"])
@pytest.mark.parametrize("field", ["unsupported_dtypes", "unsupported_layouts"])
def test_missing_any_universe_member_is_rejected(side, field):
    spec = full_spec()
    spec[side][0][field].pop()
    assert stage_1(spec).status == "PASS"
    assert any(
        f.rule_id == "support_partition.coverage" for f in findings(spec).findings
    )


@pytest.mark.parametrize("side", ["inputs", "outputs"])
@pytest.mark.parametrize(
    "positive,negative",
    [("dtype_set", "unsupported_dtypes"), ("supported_layouts", "unsupported_layouts")],
)
def test_partition_overlap_is_rejected(side, positive, negative):
    spec = full_spec()
    spec[side][0][negative].append(spec[side][0][positive][0])
    assert any(
        f.rule_id == "support_partition.coverage" for f in findings(spec).findings
    )


@pytest.mark.parametrize(
    "field",
    ["dtype_set", "unsupported_dtypes", "supported_layouts", "unsupported_layouts"],
)
def test_schema_requires_both_halves_for_new_contract(field):
    spec = full_spec()
    del spec["outputs"][0][field]
    assert stage_1(spec).status == "FAIL"


def test_new_contract_allows_all_official_layout_members():
    spec = full_spec()
    exotic = "FRACTAL_NZ_C0_16"
    spec["inputs"][0]["layout"] = exotic
    assert stage_1(spec).status == "PASS"

    legacy = copy.deepcopy(BASE)
    legacy["inputs"][0]["layout"] = exotic
    assert stage_1(legacy).status == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("unsupported_dtypes", "uint4"),
        ("unsupported_layouts", "CSR"),
        ("unsupported_layouts", "NZ"),
    ],
)
def test_legacy_values_cannot_enter_official_partition(field, value):
    spec = full_spec()
    spec["inputs"][0][field].append(value)
    assert stage_1(spec).status == "FAIL"
    assert findings(spec).status == "FAIL"


def test_full_contract_checks_output_combinations_and_layout_variants():
    spec = full_spec()
    tensor = spec["outputs"][0]
    tensor["dtype_set"].append("int8")
    tensor["unsupported_dtypes"].remove("int8")
    assert any(
        f.rule_id == "support_partition.dtype_contract" for f in findings(spec).findings
    )
    spec = full_spec()
    spec["math_semantics"]["format_variants"] = [
        {"format": "NHWC", "rank": [4], "reduction_axes": [0, 1, 2]}
    ]
    assert any(
        f.rule_id == "support_partition.layout_contract"
        for f in findings(spec).findings
    )


@pytest.mark.parametrize(
    "kind,marker",
    [
        ("dtypes", "undefined"),
        ("dtypes", "max"),
        ("formats", "END"),
        ("formats", "ALL"),
    ],
)
def test_enum_markers_cannot_be_supported(kind, marker):
    with pytest.raises(ValueError, match="Non-tensor"):
        partition([marker], [], kind)
    spec = full_spec()
    positive, negative = (
        ("dtype_set", "unsupported_dtypes")
        if kind == "dtypes"
        else ("supported_layouts", "unsupported_layouts")
    )
    spec["inputs"][0][positive].append(marker)
    spec["inputs"][0][negative].remove(marker)
    assert any(
        f.rule_id == "support_partition.non_tensor" for f in findings(spec).findings
    )


@pytest.mark.parametrize(
    "kind,value", [("dtypes", "float6_e3m2"), ("formats", "FRACTAL_NZ_C0_16")]
)
def test_950_only_types_and_formats_require_correct_chip_scope(kind, value):
    spec = full_spec()
    for tensor in spec["inputs"] + spec["outputs"]:
        positive, negative = (
            ("dtype_set", "unsupported_dtypes")
            if kind == "dtypes"
            else ("supported_layouts", "unsupported_layouts")
        )
        tensor[positive], tensor[negative] = partition([value], [], kind)
        if kind == "formats":
            tensor["layout"] = value
    if kind == "dtypes":
        for row in spec["dtype_policy"]["supported_combinations"]:
            for side in ("inputs", "outputs"):
                row[side] = dict.fromkeys(row[side], value)
    spec["op"]["platform_constraints"]["supported_chips"] = ["Ascend910B"]
    assert any(
        f.rule_id == "support_partition.chip_scope" for f in findings(spec).findings
    )
    spec["op"]["platform_constraints"]["supported_chips"] = [
        "Ascend950PR",
        "Ascend950DT",
    ]
    assert findings(spec).status == "PASS"


def test_ir_input_restrictions_are_not_blanket_kernel_restrictions():
    spec = full_spec()
    tensor = spec["inputs"][0]
    tensor["supported_layouts"], tensor["unsupported_layouts"] = partition(
        ["NC1HWC0"], [], "formats"
    )
    tensor["layout"] = "NC1HWC0"
    assert "NC1HWC0" in load_catalog()["ir_graph_input_unsupported_formats"]
    assert findings(spec).status == "PASS"


def test_cli_normalizes_official_symbols_and_defaults_to_full_contract(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--complete-support-contract",
            "--op-name",
            "neg",
            "--category",
            "Elementwise",
            "--paradigms",
            "Elementwise",
            "--inputs",
            "x:DT_FLOAT",
            "--broadcast-kind",
            "none",
            "--outputs",
            "y",
            "--supported-formats",
            "input.x:FORMAT_FRACTAL_NZ;output.y:FORMAT_ND",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    spec = yaml.safe_load((tmp_path / "spec.yaml").read_text(encoding="utf-8"))
    assert spec["support_universe"] == load_catalog()["id"]
    assert spec["inputs"][0]["dtype_set"] == ["float32"]
    assert spec["inputs"][0]["supported_layouts"] == ["FRACTAL_NZ"]
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"


def test_unknown_universe_is_rejected():
    spec = full_spec()
    spec["support_universe"] = "unknown"
    assert stage_1(spec).status == "FAIL"
    assert any(
        f.rule_id == "support_partition.universe" for f in findings(spec).findings
    )


def test_official_aliases_are_one_member_and_duplicate_assertions_fail():
    supported, excluded = partition(["FORMAT_ND"], ["NZ"], "formats")
    assert supported == ["ND"]
    assert excluded.count("FRACTAL_NZ") == 1
    assert "NZ" not in excluded
    with pytest.raises(ValueError, match="unique"):
        partition(["ND"], ["NZ", "FRACTAL_NZ"], "formats")


def test_python_full_generation_includes_input_variants_without_mutation():
    gi = GenInput(
        "neg",
        "Negation",
        "Elementwise",
        ["Elementwise"],
        [TensorSpec("x", ["float32"])],
        ["y"],
        broadcast_kind="none",
        format_variants=[{"format": "NHWC", "rank": [4], "reduction_axes": [0, 1, 2]}],
        support_universe=load_catalog()["id"],
    )
    first = render(gi)
    assert render(gi) == first
    spec = yaml.safe_load(first)
    assert spec["inputs"][0]["supported_layouts"] == ["ND", "NHWC"]
    assert spec["outputs"][0]["supported_layouts"] == ["ND"]
    assert gi.inputs[0].supported_layouts == ["ND"]
    assert stage_1(spec).status == "PASS"
    assert findings(spec).status == "PASS"


def test_check_registry_sync_checks_restriction_subsets_and_chip_references():
    import check_registry_schema_sync as sync

    schema = json.loads(sync.SCHEMA_PATH.read_text(encoding="utf-8"))
    catalog = copy.deepcopy(load_catalog())
    chips = yaml.safe_load(
        (sync.REG_DIR / "chip_registry.yaml").read_text(encoding="utf-8")
    )
    assert sync._official_type_failures(schema, catalog, chips) == []

    catalog["non_tensor_dtypes"].append("not_a_dtype")
    catalog["restricted_formats"].append("not_a_format")
    catalog["ir_graph_input_unsupported_formats"].append("not_a_format")
    catalog["restricted_chips"].append("Ascend000")
    failures = sync._official_type_failures(schema, catalog, chips)
    assert any("non_tensor_dtypes" in failure for failure in failures)
    assert any("restricted_formats" in failure for failure in failures)
    assert any("ir_graph_input_unsupported_formats" in failure for failure in failures)
    assert any("restricted_chips" in failure for failure in failures)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("stage_args", [[], ["--stage", "2"]])
def test_malformed_format_variants_produce_json_without_traceback(
    tmp_path, complete, stage_args
):
    spec = full_spec() if complete else copy.deepcopy(BASE)
    spec["math_semantics"]["format_variants"] = [None, "bad", 42, [], True]
    assert findings(spec).status == "PASS"
    path = tmp_path / "malformed.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/validate_spec.py"),
            str(path),
            "--json",
            *stage_args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    stage2 = next(stage for stage in report["stages"] if stage["stage_id"] == 2)
    assert stage2["status"] == "PASS"
    if not stage_args:
        assert result.returncode == 1
        assert len(report["stages"]) == 12
        assert report["stages"][0]["status"] == "FAIL"


@pytest.mark.parametrize(
    "side,source",
    [
        ("inputs", "layout"),
        ("outputs", "layout"),
        ("inputs", "supported_layouts"),
        ("outputs", "supported_layouts"),
        ("inputs", "format_variants"),
    ],
)
@pytest.mark.parametrize(
    "positive,negative", [("NZ", "FRACTAL_NZ"), ("FRACTAL_NZ", "NZ")]
)
def test_legacy_format_alias_conflicts_cover_every_positive_source(
    side, positive, negative, source
):
    spec = copy.deepcopy(BASE)
    tensor = spec[side][0]
    tensor["unsupported_layouts"] = [negative]
    if source == "layout":
        tensor[source] = positive
    elif source == "supported_layouts":
        tensor[source] = [positive]
    else:
        spec["math_semantics"][source] = [
            {"format": positive, "rank": [4], "reduction_axes": [0]}
        ]
    assert any(
        f.rule_id == "support_exclusions.layout_conflict"
        for f in findings(spec).findings
    )


@pytest.mark.parametrize("side", ["inputs", "outputs"])
@pytest.mark.parametrize(
    "dtype", ["max", "undefined", "string", "string_ref", "resource", "variant", "dual"]
)
@pytest.mark.parametrize("source", ["dtype_set", "supported_combinations"])
def test_legacy_non_tensor_dtypes_cannot_enter_positive_declarations(
    side, dtype, source
):
    spec = copy.deepcopy(BASE)
    spec["op"].pop("platform_constraints", None)
    if source == "dtype_set":
        spec[side][0][source] = [dtype]
    else:
        name = spec[side][0]["name"]
        spec["dtype_policy"][source][0][side][name] = dtype
    assert stage_1(spec).status == "FAIL"
    assert any(
        f.rule_id == "support_partition.non_tensor" for f in findings(spec).findings
    )


@pytest.mark.parametrize(
    "code",
    [
        "universe",
        "required",
        "canonical",
        "coverage",
        "non_tensor",
        "chip_scope",
        "dtype_contract",
        "layout_contract",
    ],
)
def test_partition_diagnostics_have_chinese_messages_and_actionable_fixes(code):
    spec = full_spec()
    tensor = spec["outputs"][0]
    if code == "universe":
        spec["support_universe"] = "unknown"
    elif code == "required":
        del tensor["unsupported_dtypes"]
    elif code == "canonical":
        tensor["unsupported_layouts"].append("NZ")
    elif code == "coverage":
        tensor["unsupported_dtypes"].pop()
    elif code == "non_tensor":
        tensor["dtype_set"].append("max")
    elif code == "chip_scope":
        tensor["dtype_set"].append("float6_e3m2")
        spec["op"].pop("platform_constraints", None)
    elif code == "dtype_contract":
        tensor["dtype_set"].append("int8")
    else:
        tensor["layout"] = "NHWC"
    rule_id = f"support_partition.{code}"
    matches = [f for f in findings(spec).findings if f.rule_id == rule_id]
    assert matches
    for finding in matches:
        assert re.search(r"[\u4e00-\u9fff]", finding.message)
        assert finding.suggested_fix and re.search(
            r"[\u4e00-\u9fff]", finding.suggested_fix
        )


@pytest.mark.parametrize(
    "field",
    [
        "non_tensor_dtypes",
        "non_tensor_formats",
        "restricted_dtypes",
        "restricted_formats",
        "ir_graph_input_unsupported_formats",
        "restricted_chips",
    ],
)
def test_registry_rejects_unknown_members_in_each_restriction(
    field, monkeypatch, capsys
):
    import check_registry_schema_sync as sync

    catalog = copy.deepcopy(load_catalog())
    catalog[field].append("misspelled_member")
    monkeypatch.setattr(sync, "load_catalog", lambda schema: catalog)
    assert sync.main() == 1
    assert field in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--supported-formats", "CSR"),
        ("--unsupported-formats", "CSR"),
        ("--unsupported-dtypes", "uint4"),
    ],
)
def test_cli_invalid_official_values_report_flag_and_never_write(tmp_path, flag, value):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "scripts/generate_spec.py"),
            "--op-name",
            "neg",
            "--category",
            "Elementwise",
            "--paradigms",
            "Elementwise",
            "--inputs",
            "x:float32",
            flag,
            f"input.x:{value}",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 2
    error = result.stderr.split("error:", 1)[-1]
    assert flag in error and "schemas/op-spec.json" in error
    assert len(error) < 350
    assert "Outside official" not in error and "Traceback" not in result.stderr
    assert not (tmp_path / "spec.yaml").exists()


def test_mixed_interactive_cli_warns_and_validates_final_answers(
    tmp_path, monkeypatch, capsys
):
    import generate_spec as generator

    gi = GenInput(
        "neg",
        "Negation",
        "Elementwise",
        ["Elementwise"],
        [TensorSpec("x", ["float32"], unsupported_dtypes=["int8"])],
        [OutputSpec("y")],
        broadcast_kind="none",
    )
    monkeypatch.setattr(generator, "interactive_collect", lambda reg: gi)
    monkeypatch.setattr("builtins.input", lambda *_: "ND")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_spec.py",
            "--output-dir",
            str(tmp_path),
            "--unsupported-dtypes",
            "input.x:float32",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        generator.main()
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--unsupported-dtypes" in stderr and "覆盖向导" in stderr
    assert not (tmp_path / "spec.yaml").exists()


def test_interactive_dtype_alias_reaches_chip_narrowing_normalized(monkeypatch, capsys):
    import generate_spec as generator

    answers = iter(
        [
            "neg",
            "Negation",
            "Elementwise",
            "",
            "x",
            "typo",
            "DT_FLOAT",
            "tensor",
            "false",
            "",
            "",
            "",
            "y",
            "tensor",
            "false",
            "",
            "",
            "",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(generator, "collect_conditional_attributes", lambda: [])
    monkeypatch.setattr(generator, "collect_semantic_cases", lambda *args: [])
    monkeypatch.setattr(generator, "collect_layout_contract", lambda *args: None)
    # Let the interface prompts run, then resolve the final fixed choices.
    original_choice = generator.prompt_choice

    def choose(message, choices, default=""):
        if message in {
            "dtype_policy.promotion",
            "broadcast.kind",
            "determinism.accumulation_order",
        }:
            return default
        return original_choice(message, choices, default)

    monkeypatch.setattr(generator, "prompt_choice", choose)

    def select_chips(message, choices, defaults=None):
        assert "float32" in message and "DT_FLOAT" not in message
        assert choices == generator.narrow_chips_by_dtypes(
            generator.load_registries()["chips"], {"float32"}
        )
        return choices

    monkeypatch.setattr(generator, "prompt_multi_choice", select_chips)
    result = generator.interactive_collect(generator.load_registries())
    assert result.inputs[0].dtype_set == ["float32"]
    assert "没有芯片同时支持" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "field,flag,old,new",
    [
        ("supported_layouts", "--supported-formats", ["ND"], ["NCHW"]),
        ("unsupported_layouts", "--unsupported-formats", ["NHWC"], ["FRACTAL_Z"]),
        ("unsupported_dtypes", "--unsupported-dtypes", ["int8"], ["int32"]),
    ],
)
def test_cli_override_warns_only_when_interactive_answer_changes(
    field, flag, old, new, capsys
):
    tensor = TensorSpec("x", ["float32"])
    setattr(tensor, field, old)
    raw = "input.x:" + ",".join(new)
    apply_unsupported_arg(
        raw, [tensor], [], field, [], official=True, warn_override=True
    )
    assert getattr(tensor, field) == new
    warning = capsys.readouterr().err
    assert flag in warning and "覆盖向导" in warning
    apply_unsupported_arg(
        raw, [tensor], [], field, [], official=True, warn_override=True
    )
    assert capsys.readouterr().err == ""


def test_cheatsheet_top_level_fields_match_schema_and_keep_b9_in_section_b():
    schema = json.loads((ROOT / "schemas/op-spec.json").read_text(encoding="utf-8"))
    text = (ROOT / "references/spec-cheatsheet.md").read_text(encoding="utf-8")
    declared_count = int(re.search(r"## A\. 顶层 (\d+) 个字段", text).group(1))
    assert declared_count == len(schema["properties"])
    section_a = text.split("## A.", 1)[1].split("## B.", 1)[0]
    example = section_a.split("```yaml", 1)[1].split("```", 1)[0]
    assert set(yaml.safe_load(example)) == set(schema["properties"])
    section_b = text.split("## B.", 1)[1].split("## C.", 1)[0]
    assert "### B9. 支持与排除范围" in section_b
    before_b9 = section_b.split("### B8.", 1)[1].split("### B9.", 1)[0]
    assert not re.search(r"(?m)^---\s*$", before_b9)


@pytest.mark.parametrize(
    "kind,definition,removed,excluded_field",
    [
        ("dtypes", "official_dtype", "int8", "unsupported_dtypes"),
        ("formats", "official_format", "CHWN", "unsupported_layouts"),
    ],
)
def test_schema_universe_drives_generation_and_both_validation_stages(
    kind, definition, removed, excluded_field, tmp_path, monkeypatch
):
    import generate_spec as generator
    import support_catalog as catalog_module
    import validate_spec as validator

    schema = json.loads((ROOT / "schemas/op-spec.json").read_text(encoding="utf-8"))
    schema["$defs"][definition]["enum"].remove(removed)
    if kind == "dtypes":
        del schema["x-support-catalog"]["dtype_aliases"]["DT_INT8"]
    schema_path = tmp_path / "op-spec.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    catalog_module.load_schema.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(catalog_module, "SCHEMA_PATH", schema_path)
            loaded = catalog_module.load_schema()
            assert generator.load_schema() is validator._load_schema() is loaded
            assert universe(kind) == schema["$defs"][definition]["enum"]
            spec = yaml.safe_load(
                render(
                    GenInput(
                        "neg",
                        "Negation",
                        "Elementwise",
                        ["Elementwise"],
                        [TensorSpec("x", ["float32"])],
                        [OutputSpec("y")],
                        broadcast_kind="none",
                        support_universe=load_catalog()["id"],
                    )
                )
            )
            for tensor in spec["inputs"] + spec["outputs"]:
                assert removed not in tensor[excluded_field]
                assert tensor[excluded_field] == [
                    value
                    for value in universe(kind)
                    if value
                    not in tensor[
                        "dtype_set" if kind == "dtypes" else "supported_layouts"
                    ]
                ]
            assert stage_1(spec).status == "PASS"
            assert findings(spec).status == "PASS"
            spec["inputs"][0][excluded_field].append(removed)
            assert stage_1(spec).status == "FAIL"
            assert any(
                finding.rule_id == "support_partition.coverage"
                for finding in findings(spec).findings
            )
    finally:
        catalog_module.load_schema.cache_clear()


@pytest.mark.parametrize(
    "mutation", ["missing_alias", "unknown_target", "duplicate_target"]
)
def test_schema_consistency_rejects_invalid_dtype_alias_mappings(mutation):
    import check_registry_schema_sync as sync

    schema = json.loads(sync.SCHEMA_PATH.read_text(encoding="utf-8"))
    aliases = schema["x-support-catalog"]["dtype_aliases"]
    if mutation == "missing_alias":
        del aliases["DT_INT8"]
    elif mutation == "unknown_target":
        aliases["DT_INT8"] = "not_an_official_dtype"
    else:
        aliases["DT_INT8"] = aliases["DT_FLOAT"]
    chips = yaml.safe_load(
        (sync.REG_DIR / "chip_registry.yaml").read_text(encoding="utf-8")
    )
    failures = sync._official_type_failures(schema, load_catalog(schema), chips)
    assert any("dtype_aliases" in failure for failure in failures)


@pytest.mark.parametrize("definition", ["official_dtype", "official_format"])
@pytest.mark.parametrize("mutation", ["empty", "duplicate"])
def test_schema_consistency_rejects_empty_or_duplicate_universes(definition, mutation):
    import check_registry_schema_sync as sync

    schema = json.loads(sync.SCHEMA_PATH.read_text(encoding="utf-8"))
    values = schema["$defs"][definition]["enum"]
    if mutation == "empty":
        values.clear()
    else:
        values.append(values[0])
    chips = yaml.safe_load(
        (sync.REG_DIR / "chip_registry.yaml").read_text(encoding="utf-8")
    )
    failures = sync._official_type_failures(schema, load_catalog(schema), chips)
    assert any(f"{definition}.enum" in failure for failure in failures)
