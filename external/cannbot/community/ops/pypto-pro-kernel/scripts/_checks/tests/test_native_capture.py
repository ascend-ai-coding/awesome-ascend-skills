# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native parsing stays separate from execution and reports inspection gaps."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from _checks import native_capture as capture


def node(kind, **fields):
    result = type(kind, (), {})()
    result.__dict__.update(fields)
    return result


def program(source):
    def dimension(value):
        return node("ConstInt", value=value)

    def tile(space, major, minor):
        return node(
            "Var",
            type=node(
                "TileType",
                shape=[dimension(32), dimension(128)],
                dtype=SimpleNamespace(to_string=lambda: "float16"),
                memref=SimpleNamespace(memory_space=SimpleNamespace(name=space)),
                tile_view=SimpleNamespace(valid_shape=[dimension(32), dimension(128)]),
                hardware_info=SimpleNamespace(
                    blayout=SimpleNamespace(name=major),
                    slayout=SimpleNamespace(name=minor),
                ),
            ),
        )

    transfer = node(
        "Call",
        name="block.insert",
        args=[
            tile("Mat", "col_major", "row_major"),
            tile("Vec", "row_major", "none_box"),
        ],
        span=SimpleNamespace(filename=str(source), begin_line=9, begin_column=4),
    )
    return node(
        "Program",
        functions={
            "kernel": node(
                "Function",
                body=node(
                    "SeqStmts",
                    stmts=[
                        node(
                            "IfStmt",
                            condition=node("ConstBool", value=True),
                            then_body=node("EvalStmt", expr=transfer),
                            else_body=None,
                        )
                    ],
                ),
            )
        },
    )


def test_transfer_descriptors_keep_native_types_and_source_location(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("# supplied source\n")
    sources = {}
    facts, unknown = capture.inspect_program(program(source), "vector", sources)
    assert unknown == () and len(facts) == 1
    assert facts[0]["source"]["layout"] == "ND"
    assert facts[0]["destination"]["layout"] == "NZ"
    assert facts[0]["source"]["shape"] == [32, 128]
    assert facts[0]["location"] == {"file": str(source), "line": 9, "column": 4}
    assert next(iter(sources.values()))["available"]


def test_unknown_nodes_and_missing_fields_do_not_claim_complete_coverage(tmp_path):
    current = program(tmp_path / "entry.py")
    current.functions["kernel"].body.stmts.extend(
        [node("FutureStmt"), node("EvalStmt")]
    )
    facts, unknown = capture.inspect_program(current, "vector", {})
    assert len(facts) == 1
    assert unknown == ("EvalStmt.expr", "FutureStmt")


def runtime(
    monkeypatch, source, *, observe=True, prepare_error=False, active=("cube", "vector"), printer="python_print"
):
    events = []
    bound = object()

    def parse(section, *, bound_signature, observer):
        assert bound_signature is bound
        events.append("parse:" + section)
        value = program(source)
        if observe:
            observer(value, SimpleNamespace())
        return value, section in active

    definition = SimpleNamespace(
        _source_file=str(source), last_param_directions={}, parse_target_program=parse
    )

    def prepare(value, arch, **kwargs):
        assert (
            value is definition and arch == "a5" and kwargs["bound_signature"] is bound
        )
        assert Path(kwargs["out_dir"]).is_dir()
        events.append("prepare")
        if prepare_error:
            raise ValueError("invalid pipeline")

    jit = SimpleNamespace(
        _prepare_codegen_inputs=prepare,
        _extract_param_specs=lambda *a: [],
        get_current_arch=lambda: "a5",
    )
    ir = SimpleNamespace(
        SectionKind=SimpleNamespace(Cube="cube", Vector="vector"),
        **{printer: lambda value: "parsed target IR"},
    )
    monkeypatch.setattr(capture, "_runtime", lambda: (ir, jit, {}))
    kernel = SimpleNamespace(
        arch="a5",
        tilingkey_schema=None,
        _resolve_spec_identity=lambda *a: (None, None),
        _normalize_launch_args=lambda args, kw: args,
        _bind_runtime_args=lambda args: bound,
        to_kernel_def=lambda *a, **kw: definition,
    )
    return kernel, events


@pytest.mark.parametrize("printer", ["text_dump", "python_print"])
def test_pipeline_preparation_precedes_each_target_and_never_launches(
    tmp_path, monkeypatch, printer
):
    source = tmp_path / "kernel.py"
    source.write_text("# native entry\n")
    kernel, events = runtime(monkeypatch, source, printer=printer)
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "caller-value")
    result = capture.capture_ir(kernel, config=capture.CaptureConfig())
    assert events == ["prepare", "parse:cube", "parse:vector"]
    assert len(result.digest) == 64 and len(result.transfers) == 2
    assert result.unknown_nodes == ()
    assert os.environ["PYPTOPRO_JIT_ARCH"] == "caller-value"


def test_enum_architecture_uses_canonical_environment_and_restores_it(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "caller-value")
    arch = SimpleNamespace(name="DAV_3510")
    with capture._architecture(arch):
        assert os.environ["PYPTOPRO_JIT_ARCH"] == "3510"
    assert os.environ["PYPTOPRO_JIT_ARCH"] == "caller-value"


@pytest.mark.parametrize(
    "active,expected", [(("vector",), ["vector"]), ((), ["cube", "vector"])]
)
def test_transfer_inspection_uses_the_native_active_targets(
    tmp_path, monkeypatch, active, expected
):
    source = tmp_path / "kernel.py"
    source.write_text("# native entry\n")
    kernel, _ = runtime(monkeypatch, source, active=active)
    result = capture.capture_ir(kernel, config=capture.CaptureConfig())
    assert [item["target"] for item in result.transfers] == expected


@pytest.mark.parametrize(
    "mode,match", [("observer", "observer"), ("prepare", "pipeline")]
)
def test_failed_preparation_or_missing_observation_cannot_report_success(
    tmp_path, monkeypatch, mode, match
):
    source = tmp_path / "kernel.py"
    source.write_text("# native entry\n")
    kernel, events = runtime(
        monkeypatch, source, observe=mode != "observer", prepare_error=mode == "prepare"
    )
    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    with pytest.raises((RuntimeError, ValueError), match=match):
        capture.capture_ir(kernel, config=capture.CaptureConfig())
    assert "PYPTOPRO_JIT_ARCH" not in os.environ
    if mode == "prepare":
        assert events == ["prepare"]
