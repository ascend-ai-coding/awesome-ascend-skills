# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Example interfaces and host validation contracts work without a device."""

import ast
import importlib.util
import json
import subprocess
import sys
import textwrap
import types

import pytest

from _checks.paths import EXAMPLES_ROOT


@pytest.mark.parametrize(
    "folder,filename,specs,exports",
    [
        (
            "matmul/fundamental_usage/l1_layout",
            "kernel.py",
            {
                "_kernels.nn": ("K", "M", "N", "matmul_nn"),
                "_kernels.nt": ("matmul_nt",),
                "_kernels.tn": ("matmul_tn",),
                "_kernels.tt": ("matmul_tt",),
            },
            ("K", "M", "N", "matmul_nn", "matmul_nt", "matmul_tn", "matmul_tt"),
        ),
        (
            "matmul/fundamental_usage/l1_layout",
            "reference.py",
            {
                "_references.nn": ("reference_nn",),
                "_references.nt": ("reference_nt",),
                "_references.tn": ("reference_tn",),
                "_references.tt": ("reference_tt",),
            },
            ("reference_nn", "reference_nt", "reference_tn", "reference_tt"),
        ),
        (
            "matmul/tail_extent",
            "kernel.py",
            {
                "_kernels.k": (
                    "EM",
                    "EN",
                    "EK",
                    "K",
                    "M",
                    "N",
                    "OM",
                    "ON",
                    "OK",
                    "tail_k",
                ),
                "_kernels.m": ("tail_m",),
                "_kernels.n": ("tail_n",),
                "_kernels.mnk": ("tail_mnk",),
                "_kernels.odd": ("tail_odd",),
            },
            (
                "EM",
                "EN",
                "EK",
                "K",
                "M",
                "N",
                "OM",
                "ON",
                "OK",
                "tail_k",
                "tail_m",
                "tail_n",
                "tail_mnk",
                "tail_odd",
                "k",
                "m",
                "n",
            ),
        ),
        (
            "matmul/tail_extent",
            "reference.py",
            {
                "_references.k": ("reference_k",),
                "_references.m": ("reference_m",),
                "_references.n": ("reference_n",),
                "_references.mnk": ("reference_mnk",),
                "_references.odd": ("reference_odd",),
            },
            (
                "reference_k",
                "reference_m",
                "reference_n",
                "reference_mnk",
                "reference_odd",
            ),
        ),
    ],
)
def test_facades_reexport_their_implementation_entries(
    monkeypatch, folder, filename, specs, exports
):
    package = "_public_example_fixture"

    def module(name):
        existing = sys.modules.get(name)
        if existing is not None:
            return existing
        value = types.ModuleType(name)
        value.__path__ = []
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            parent, member = name.rsplit(".", 1)
            setattr(module(parent), member, value)
        return value

    module(package)
    sentinels = {}
    for suffix, names in specs.items():
        target = module(package + "." + suffix)
        sentinels[suffix.rsplit(".", 1)[-1]] = target
        for name in names:
            value = object()
            setattr(target, name, value)
            sentinels[name] = value
    path = EXAMPLES_ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(package + "." + path.stem, path)
    facade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facade)
    for name in exports:
        assert getattr(facade, name) is sentinels[name]
        assert name in facade.__all__


def test_attn_epilogue_keeps_the_public_input_keyword():
    for name, functions in [
        ("kernel.py", {"attn_epilogue"}),
        ("reference.py", {"inverse_rope", "attn_epilogue_golden_cpu"}),
    ]:
        tree = ast.parse(
            (EXAMPLES_ROOT / "algorithms/attn_epilogue" / name).read_text()
        )
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name in functions:
                assert function.args.args[0].arg == "o"


def test_row_top2_order_keeps_ties_stable_across_rows(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    path = EXAMPLES_ROOT / "algorithms/row_top2/reference.py"
    spec = importlib.util.spec_from_file_location("row_top2_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference._row_indices([5.0, 5.0, 1.0]) == [0, 1]
    assert reference._row_indices([1.0, 5.0, 5.0]) == [1, 2]


@pytest.mark.parametrize(
    "relative,name,arguments,message",
    [
        (
            "algorithms/mla_bounded/_kernels/fused.py",
            "make_fused",
            dict(batch=1, queries=256, keys=128, heads=16, nope=64, causal=True),
            "causal query count",
        ),
        (
            "algorithms/quant_lightning_indexer/_kernels/paged.py",
            "make_paged",
            dict(max_k=4096, page_size=64, k_page_stride=0, ks_page_stride=0),
            "paged layout",
        ),
        (
            "algorithms/quant_lightning_indexer/_kernels/paged.py",
            "make_paged",
            dict(max_k=128, page_size=128, k_page_stride=0, ks_page_stride=0),
            "partial top-k lists",
        ),
        (
            "algorithms/sparse_attention/_kernels/vector_ops.py",
            "make_softmax",
            dict(
                dense=True,
                half_rows=16,
                heads=16,
                key_block=128,
                key_rows=1024,
                query_rows=16,
                causal=False,
            ),
            "dense softmax",
        ),
        (
            "algorithms/sparse_attention/_kernels/vector_ops.py",
            "make_softmax",
            dict(
                dense=False,
                half_rows=16,
                heads=16,
                key_block=256,
                key_rows=1024,
                query_rows=16,
                causal=False,
            ),
            "indexed softmax",
        ),
    ],
)
def test_host_factory_rejects_invalid_inputs_with_python_optimization(
    relative, name, arguments, message
):
    path = EXAMPLES_ROOT / relative
    probe = textwrap.dedent("""\
        import ast
        import importlib.util
        import json
        from pathlib import Path
        import sys
        import types

        def register(name):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            return module

        def native_boundary(*args, **kwargs):
            raise RuntimeError("invalid inputs reached native code generation")

        root = register("pypto_pro")
        language = register("pypto_pro.language")
        root.language = language
        for dtype in ("DT_FP32", "DT_FP16", "DT_INT32", "DT_UINT32",
                      "DT_FP4E2M1", "DT_FP8E8M0", "DT_BF16", "Vf"):
            setattr(language, dtype, object())
        language.vector_function = native_boundary
        register("host_factory_probe")
        path = Path(sys.argv[1])
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.level:
                dependency = register("host_factory_probe." + node.module)
                for alias in node.names:
                    setattr(dependency, alias.name, object())
        spec = importlib.util.spec_from_file_location("host_factory_probe.target", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        try:
            getattr(module, sys.argv[2])(**json.loads(sys.argv[3]))
        except AssertionError as error:
            if sys.argv[4] not in str(error):
                raise
        else:
            raise RuntimeError("invalid factory inputs were accepted")
        """)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            probe,
            str(path),
            name,
            json.dumps(arguments),
            message,
        ],
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "relative,names",
    [
        ("algorithms/sparse_attention/kernel.py", {"prepare", "finish"}),
        ("algorithms/quant_sparse_lightning_indexer/kernel.py", {"qsli_kernel"}),
    ],
)
def test_native_tensor_slicing_retains_its_parser_syntax(relative, names):
    tree = ast.parse((EXAMPLES_ROOT / relative).read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    for function in functions:
        assert any(isinstance(node, ast.Slice) for node in ast.walk(function))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "slice"
            for node in ast.walk(function)
        )
