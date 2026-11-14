# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Upstream-aligned exact tests, independent CPU mathematics and contract boundaries."""

import ast
import importlib
import inspect
import os
from pathlib import Path
import sys

import pytest
import torch

# Support both direct execution and pytest from the repository root.
if __package__:
    from .reference import attn_epilogue_golden_cpu, inverse_rope, quantize_mx
    from .reference import assert_exact, assert_payload
else:
    from reference import attn_epilogue_golden_cpu, inverse_rope, quantize_mx
    from reference import assert_exact, assert_payload


def _implementation():
    return (
        importlib.import_module(".kernel", __package__)
        if __package__
        else importlib.import_module("kernel")
    )


SUPPLIED_CASES = [
    (1, 0, 1.0),
    (33, 0, 1.0),
    (72, 16, 1.0),
    (129, 0, 1.0),
    (193, 0, 1.0),
    (255, 0, 1.0),
    (72, 0, 0.0),
    (256, 0, 1e-37),
]
BOUNDARY_TOKENS = [
    2,
    15,
    16,
    17,
    31,
    32,
    63,
    64,
    65,
    127,
    128,
    192,
    256,
    257,
    272,
    511,
    512,
    513,
    769,
    1024,
    1793,
    2048,
    2049,
    4096,
    16384,
]


def input_fixture(t, scale=1.0):
    gen = torch.Generator().manual_seed(2026)
    return (torch.randn(t, 64, 512, generator=gen) * (0.3 * scale)).to(torch.bfloat16)


def rope_tables(t):
    j = torch.arange(0, 64, 2, dtype=torch.float32)
    angles = torch.outer(
        torch.arange(t, dtype=torch.float32), 1.0 / (10000.0 ** (j / 64))
    )
    return angles.sin().repeat_interleave(2, -1), angles.cos().repeat_interleave(2, -1)


def make_weights():
    import torch_npu

    gen = torch.Generator().manual_seed(2026)
    wa = (torch.randn(8, 1024, 4096, generator=gen) * 0.3).to(torch.bfloat16)
    wb = (torch.randn(5120, 8192, generator=gen) * 0.2).to(torch.bfloat16)
    aq, asc = torch_npu.npu_dynamic_mx_quant(wa.reshape(8192, 4096).npu(), dst_type=292)
    bq, bsc = torch_npu.npu_dynamic_mx_quant(wb.npu(), dst_type=292)
    e8 = torch.float8_e8m0fnu
    return (
        aq.reshape(8, 1024, 4096),
        bq,
        asc.view(e8).reshape(8, 1024, 64, 2),
        bsc.view(e8),
    )


def native_reference(attn_out, woa, wob, sa, sb, sin, cos):
    """Use the same native arithmetic and intermediate rounding as upstream."""
    import torch_npu

    t = attn_out.shape[0]
    rotated = inverse_rope(attn_out.cpu(), sin.cpu(), cos.cpu()).reshape(t, 8, 4096)
    aq, asc = torch_npu.npu_dynamic_mx_quant(rotated.npu(), dst_type=292)
    y = torch_npu.npu_transpose_quant_batchmatmul(
        aq,
        woa,
        15,
        x1_scale=asc.view(torch.float8_e8m0fnu),
        x2_scale=sa,
        group_sizes=[1, 1, 32],
        perm_x1=[1, 0, 2],
        perm_x2=[0, 2, 1],
        perm_y=[1, 0, 2],
    ).reshape(t, 8192)
    yq, ys = torch_npu.npu_dynamic_mx_quant(y, dst_type=292)
    out = torch_npu.npu_quant_matmul(
        yq,
        wob.t(),
        sb.transpose(0, 1),
        pertoken_scale=ys.view(torch.float8_e8m0fnu),
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )
    return (
        out.cpu(),
        aq.view(torch.uint8).cpu(),
        asc.view(torch.uint8).reshape(t, 8, 128).cpu(),
        yq.view(torch.uint8).cpu(),
        ys.view(torch.uint8).reshape(t, 256).cpu(),
    )


@pytest.fixture(scope="module")
def weights():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend 950 NPU required")
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    return make_weights()


def arguments(weights, t, scale=1.0):
    x = input_fixture(t, scale)
    sin, cos = rope_tables(t)
    return (x.npu(), *weights, sin.npu(), cos.npu())


def test_comparator_rejects_corruption():
    a = torch.ones((2, 3), dtype=torch.bfloat16)
    b = a.clone()
    b[1, 2] += 1
    with pytest.raises(AssertionError):
        assert_exact(a, b)
    with pytest.raises(AssertionError):
        assert_payload(
            torch.tensor([1, 2], dtype=torch.uint8),
            torch.tensor([1, 3], dtype=torch.uint8),
        )


@pytest.mark.parametrize("t,padding,scale", SUPPLIED_CASES)
def test_supplied_native_and_replay(weights, t, padding, scale):
    op = _implementation()
    args = arguments(weights, t, scale)
    expected = [native_reference(*args)]
    negative = (-args[0], *args[1:])
    expected.append(native_reference(*negative))
    m = (t + 15) // 16 * 16 + padding
    buffers = op._scratch(args[0], m)
    storage = torch.full(
        (m * 5120 + 64,), 19.0, dtype=torch.bfloat16, device=args[0].device
    )
    output = storage[32:-32].view(m, 5120)
    op._launch(*args, buffers, output)
    torch.npu.synchronize()
    assert_exact(output[:t], expected[0][0])
    assert_payload(
        buffers[0].view(8, m, 4096)[:, :t],
        expected[0][1].permute(1, 0, 2),
        "activation bytes",
    )
    assert_payload(
        buffers[1].view(8, m, 128)[:, :t],
        expected[0][2].permute(1, 0, 2),
        "activation scales",
    )
    assert_payload(buffers[2][:t], expected[0][3], "Y bytes")
    assert_payload(buffers[3][:t], expected[0][4], "Y scales")
    assert bool((storage[:32] == 19).all()) and bool((storage[-32:] == 19).all())
    assert bool((output[t:] == 19).all())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        snapshots = []
        for _ in range(2):
            args[0].neg_()
            op._launch(*args, buffers, output)
            snapshots.append(output[:t].clone())
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        for i, snapshot in enumerate(snapshots):
            assert_exact(snapshot, expected[(i + 1) % 2][0], "alternating replay")


@pytest.mark.parametrize("t", [1, 72, 256, 257, 2048])
def test_public_tp1(weights, t):
    op = _implementation()
    args = arguments(weights, t)
    expected = [native_reference(*args)[0], native_reference(-args[0], *args[1:])[0]]
    out = op.attn_epilogue(*args)
    assert (
        out.shape == (t, 5120) and out.dtype == torch.bfloat16 and out.is_contiguous()
    )
    ids = {
        key: tuple(v.data_ptr() for v in values) for key, values in op._SCRATCH.items()
    }
    args[0].neg_()
    other = op.attn_epilogue(*args)
    torch.npu.synchronize()
    assert out.data_ptr() != other.data_ptr()
    assert_exact(out, expected[0])
    assert_exact(other, expected[1])
    assert ids == {
        key: tuple(v.data_ptr() for v in values) for key, values in op._SCRATCH.items()
    }
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        args[0].neg_()
        first = op.attn_epilogue(*args)
        args[0].neg_()
        second = op.attn_epilogue(*args)
    assert first.data_ptr() != second.data_ptr()
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        assert_exact(first, expected[0])
        assert_exact(second, expected[1])


@pytest.mark.parametrize("t", BOUNDARY_TOKENS)
def test_token_boundaries(weights, t):
    attn_epilogue = _implementation().attn_epilogue
    args = arguments(weights, t)
    out = attn_epilogue(*args)
    assert_exact(out, native_reference(*args)[0])


def test_dynamic_token_range(weights):
    attn_epilogue = _implementation().attn_epilogue
    for t in range(1, 257):
        args = arguments(weights, t)
        assert_exact(attn_epilogue(*args), native_reference(*args)[0], f"runtime T={t}")


def test_primitive_only_implementation():
    op = _implementation()
    tree = ast.parse(inspect.getsource(op))
    accesses = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pl"
    }
    assert not accesses.intersection(
        {"gemm", "gemm_mx", "mem_space", "allocate_tile_group"}
    )
    assert {"make_tile_group", "matmul_mx", "matmul_mx_acc"} <= accesses


def test_bounded_tiling_profiles():
    from pypto_pro.runtime.tilingkey import TilingKeySchema

    op = _implementation()
    schema = TilingKeySchema(op.EpilogueKey)
    profiles = set(schema.enumerate_valid())
    assert len(profiles) == 6
    combined = set()
    for cores in (28, 32):
        selected = {
            tuple(op._tiling_key(t, cores)[name] for name in schema.field_names())
            for t in [*range(1, 257), *BOUNDARY_TOKENS]
        }
        assert len(selected) == 3
        assert selected <= profiles
        combined.update(selected)
    assert combined == profiles


def test_public_validation(weights):
    op = _implementation()
    import torch_npu

    args = arguments(weights, 72)
    assert list(inspect.signature(op.attn_epilogue).parameters) == [
        "attn_out",
        "woa",
        "wob",
        "descale_woa",
        "descale_wob",
        "rope_sin",
        "rope_cos",
    ]
    faults = [
        (0, args[0].reshape(576, 4096)),
        (1, weights[0].reshape(8192, 4096)),
        (3, weights[2].view(torch.uint8)),
        (6, args[6].cpu()),
        (6, args[6][:, ::2]),
        (6, args[6].to(torch.float16)),
        (6, torch.empty((72, 128), device=args[0].device)[:, ::2]),
        (2, torch_npu.npu_format_cast(weights[1], 29)),
    ]
    for index, value in faults:
        bad = list(args)
        bad[index] = value
        with pytest.raises((ValueError, TypeError)):
            op.attn_epilogue(*bad)
    bad = list(args)
    bad[0] = torch.empty((0, 64, 512), dtype=torch.bfloat16, device=args[0].device)
    with pytest.raises(ValueError):
        op.attn_epilogue(*bad)


def test_cpu_quantization_matches_native(weights):
    import torch_npu

    for scale in (0.0, 1e-37, 1.0, 1000.0):
        x = input_fixture(1, scale).reshape(8, 4096)
        q, s = quantize_mx(x)
        nq, ns = torch_npu.npu_dynamic_mx_quant(x.npu(), dst_type=292)
        assert_payload(q, nq.cpu())
        assert_payload(s, ns.cpu())


def test_independent_cpu_projection(weights):
    attn_epilogue = _implementation().attn_epilogue
    x = ((torch.arange(32768).float() % 127 - 63) / 128).reshape(1, 64, 512).bfloat16()
    sin, cos = rope_tables(1)
    codes = torch.tensor([1.0, -1.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    wa = torch.zeros((8, 1024, 4096), dtype=torch.uint8)
    rows = torch.arange(1024)
    for group in range(8):
        wa[group, rows, (rows * 3 + group * 17) % 4096] = codes[group % 2]
    wb = torch.zeros((5120, 8192), dtype=torch.uint8)
    rows = torch.arange(5120)
    wb[rows, (rows * 37 + 13) % 8192] = codes[rows % 2]
    wa = wa.view(torch.float8_e4m3fn)
    wb = wb.view(torch.float8_e4m3fn)
    sa = torch.full((8, 1024, 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sb = torch.full((5120, 128, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    args = (x, wa, wb, sa, sb, sin, cos)
    expected = attn_epilogue_golden_cpu(*args)
    assert_exact(
        attn_epilogue(*(v.npu() for v in args)), expected, "independent CPU projection"
    )


if __name__ == "__main__":
    torch.set_num_threads(8)
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
