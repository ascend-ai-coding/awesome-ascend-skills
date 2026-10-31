# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""形状、布局、因果坐标、缓冲复用和独立 V 的板端回归。

参考使用独立 head-major FP64 数学定义及预先给定的矩阵链误差预算，
不复制候选的概率舍入、分块或固定参考值。检查输入不变、NaN 毒化和输出守卫。
QK/PV 沿物理连续轴折叠，归一化保留四维步长并沿同一 query 的 head 轴分组；
普通 launch 边界完成 GM 阶段依赖，VF 中间值的写回/重读使用整块屏障。
变更 query 长度、两种布局及 workers=1/7，覆盖因果坐标和多轮缓冲复用。

性能比较需计入整个调用的 native task 总和，FP16 三阶段、BF16 四阶段，
并按实际可用 Cube/Vector 数选择 workers。全矩阵 GM 中间量和独立阶段有固定成本；
该组织不保证小 batch decode 优于融合实现。基线应使用 baseline_perf_us，
不可把理论硬件下界 t_hw_us 当作基线。
"""

import pytest
import torch

from .kernel import (
    make_kernels,
    mla,
    make_fused,
    mla_fused,
    make_decode_kernel,
    mla_decode,
)
from .reference import reference_and_bound


def make_inputs(shape, dtype, layout, seed=42):
    b, q, n, h, dn, dr = shape
    qshape = (b, q, h) if layout == "BSND" else (b, h, q)
    kshape = (b, n, 1) if layout == "BSND" else (b, 1, n)
    rng = torch.Generator().manual_seed(seed)
    return [
        torch.rand((*prefix, dim), generator=rng).mul(2).sub(1).to(dtype)
        for prefix, dim in [
            (qshape, dn),
            (qshape, dr),
            (kshape, dn),
            (kshape, dr),
            (kshape, dn),
        ]
    ]


def guarded(shape, dtype, device, sentinel=19):
    count = 1
    for dim in shape:
        count *= dim
    backing = torch.full((count + 128,), sentinel, dtype=dtype, device=device)
    return backing[64:-64].view(shape), backing


def execute_guarded(inputs, layout, causal, workers):
    qn, qr, kn, kr, values = inputs
    b = qn.shape[0]
    q, h = (
        (qn.shape[1], qn.shape[2]) if layout == "BSND" else (qn.shape[2], qn.shape[1])
    )
    n = kn.shape[1 if layout == "BSND" else 2]
    dn, dr, rows = qn.shape[-1], qr.shape[-1], q * h
    kernels = make_kernels(
        b, q, n, h, dn, dr, str(qn.dtype).split(".")[-1], layout, causal, workers
    )
    scores, sg = guarded((b, rows, n), torch.float32, qn.device)
    probabilities, pg = guarded((b, rows, n), qn.dtype, qn.device)
    output, og = guarded(tuple(qn.shape), qn.dtype, qn.device)
    for value in (scores, probabilities, output):
        value.fill_(float("nan"))
    score, normalize, product, finalize = kernels
    score[None, workers](
        qn.view(b, rows, dn),
        qr.view(b, rows, dr),
        kn.view(b, n, dn),
        kr.view(b, n, dr),
        scores,
    )
    soft_shape = (b, h, q, n) if layout == "BNSD" else (b, q, h, n)
    normalize[None, 2 * workers](
        scores.view(soft_shape), probabilities.view(soft_shape)
    )
    if finalize is None:
        product[None, workers](
            probabilities, values.view(b, n, dn), output.view(b, rows, dn)
        )
    else:
        scaled, scaled_guard = guarded((b, rows, dn), torch.float32, qn.device)
        scaled.fill_(float("nan"))
        product[None, workers](probabilities, values.view(b, n, dn), scaled)
        finalize[None, 2 * workers](scaled.view(-1), output.view(-1))
        torch.npu.synchronize()
        assert (scaled_guard[:64] == 19).all() and (scaled_guard[-64:] == 19).all()
    torch.npu.synchronize()
    for backing in (sg, pg, og):
        assert (backing[:64] == 19).all()
        assert (backing[-64:] == 19).all()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(output).all()
    return output.cpu()


@pytest.mark.parametrize(
    "shape,causal",
    [
        ((2, 3, 128, 32, 192, 64), True),
        ((3, 5, 384, 16, 448, 128), False),
        ((1, 65, 256, 16, 128, 64), True),
        ((2, 2, 2048, 128, 512, 128), True),
    ],
)
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shapes_layouts_and_slots(shape, causal, layout, dtype, device):
    host = make_inputs(shape, dtype, layout)
    expected, bound = reference_and_bound(*host, layout, causal)
    inputs = [value.to(device) for value in host]
    for workers in (1, 7):
        for _ in range(2):
            got = execute_guarded(inputs, layout, causal, workers).double()
            assert ((got - expected).abs() <= bound).all()
        for actual, original in zip(inputs, host):
            assert torch.equal(actual.cpu(), original)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_last_key_visibility_and_wrapper(layout, dtype, device):
    host = make_inputs((2, 3, 128, 16, 64, 64), dtype, layout)
    for value in host:
        value.zero_()
    qn, qr, kn, kr, values = host
    signs = torch.where(torch.arange(64) % 2 == 0, 1.0, -1.0).to(dtype)
    if layout == "BSND":
        values[0, -1, 0] = signs
        values[1, -1, 0] = -signs / 2
    else:
        values[0, 0, -1] = signs
        values[1, 0, -1] = -signs / 2
    expected, _ = reference_and_bound(*host, layout, True)
    inputs = [value.to(device) for value in host]
    for _ in range(2):
        got = mla(*inputs, layout=layout, causal=True, workers=3)
        torch.npu.synchronize()
        assert got.is_contiguous()
        assert torch.equal(got.cpu(), expected.to(dtype))


@pytest.mark.parametrize(
    "kind",
    [
        "score_floor",
        "score_ceiling",
        "tiny",
        "subnormal",
        "tiny_mixed",
        "subnormal_mixed",
        "rope_only",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numerical_boundaries(kind, dtype, device):
    shape = (2, 3, 128, 16, 512, 128)
    host = make_inputs(shape, dtype, "BNSD")
    qn, qr, kn, kr, values = host
    if kind.startswith("score_"):
        sign = -1 if kind == "score_floor" else 1
        qn.fill_(1)
        qr.fill_(1)
        kn.fill_(sign)
        kr.fill_(sign)
        values.fill_(0.5)
    elif kind in ("tiny", "subnormal", "tiny_mixed", "subnormal_mixed"):
        qn.zero_()
        qr.zero_()
        magnitude = torch.finfo(dtype).tiny
        if kind.startswith("subnormal"):
            magnitude = torch.nextafter(
                torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype)
            ).item()
        values.fill_(magnitude)
        if kind.endswith("mixed"):
            pattern = torch.tensor([1, 3, -1, 0], dtype=torch.float64).repeat(32)
            signs = torch.where(torch.arange(512) % 2 == 0, 1.0, -1.0)
            values[:] = (pattern[None, None, :, None] * signs * magnitude).to(dtype)
    else:
        qn.zero_()
        for h in range(16):
            qr[:, h].fill_(1 if h % 2 == 0 else -1)
        kr[:, :, :64].fill_(1)
        kr[:, :, 64:].fill_(-1)
        values[:, :, :64].fill_(0.5)
        values[:, :, 64:].fill_(-0.5)
    expected, bound = reference_and_bound(*host, "BNSD", True)
    got = execute_guarded(
        [value.to(device) for value in host], "BNSD", True, 3
    ).double()
    assert ((got - expected).abs() <= bound).all()


def test_bf16_unscale_rounding(device):
    quantum = 2.0**-133
    fractions = torch.tensor(
        [0, 0.5, 1, 1.5, 2.5, 63.5, 64.5, 126.5, 127.5, 128], dtype=torch.float64
    )
    values = torch.cat(
        (
            fractions * quantum,
            torch.tensor([0.5, 0.5 + 2**-9, 1.0], dtype=torch.float64),
        )
    )
    values = torch.cat((values, -values))
    scaled = (values * 2**32).float()
    scaled = torch.cat(
        (
            scaled,
            torch.nextafter(scaled, torch.full_like(scaled, -float("inf"))),
            torch.nextafter(scaled, torch.full_like(scaled, float("inf"))),
        )
    )
    scaled = scaled.repeat((1024 + scaled.numel() - 1) // scaled.numel())[:1024]
    # CPU float64->BF16 conversion can round through FP32 near subnormal ties.
    # Select the nearest BF16 value in an independently enumerated FP64 grid.
    codes = torch.arange(0x4002, dtype=torch.int64)
    exponent = codes >> 7
    mantissa = (codes & 127).double() + torch.where(exponent > 0, 128.0, 0.0)
    powers = torch.where(exponent > 0, exponent - 134, torch.full_like(exponent, -133))
    grid = torch.ldexp(mantissa, powers)
    exact = (scaled.double() * 2**-32).abs()
    upper = torch.searchsorted(grid, exact)
    lower = (upper - 1).clamp_min(0)
    dl, du = exact - grid[lower], grid[upper] - exact
    take_upper = (du < dl) | ((du == dl) & ((upper & 1) == 0))
    encoded = torch.where(take_upper, upper, lower)
    sign = (scaled.view(torch.int32).to(torch.int64) >> 31) & 1
    expected = (encoded | (sign << 15)).to(torch.int16)
    finalize = make_kernels(1, 1, 128, 16, 64, 64, "bfloat16", "BSND", False, 3)[3]
    src = scaled.to(device)
    out = torch.empty(1024, dtype=torch.bfloat16, device=device)
    finalize[None, 6](src, out)
    torch.npu.synchronize()
    actual_bits = out.cpu().view(torch.int16)
    expected_bits = expected.cpu().view(torch.int16)
    bad = (actual_bits != expected_bits).nonzero().flatten()[:8].tolist()
    details = [
        (i, scaled[i].item(), actual_bits[i].item(), expected_bits[i].item())
        for i in bad
    ]
    assert torch.equal(actual_bits, expected_bits), (
        str(scaled.device),
        str(expected.device),
        details,
    )


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_query_full_visibility(layout, dtype, device):
    host = make_inputs((2, 1, 384, 48, 192, 64), dtype, layout)
    for value in host:
        value.zero_()
    values = host[-1]
    signs = torch.where(torch.arange(192) % 2 == 0, 1.0, -1.0).to(dtype)
    if layout == "BSND":
        values[0, -1, 0] = signs
        values[1, -1, 0] = -signs
    else:
        values[0, 0, -1] = signs
        values[1, 0, -1] = -signs
    expected, _ = reference_and_bound(*host, layout, False)
    inputs = [value.to(device) for value in host]
    for causal in (False, True):
        got = execute_guarded(inputs, layout, causal, 5)
        assert torch.equal(got, expected.to(dtype))


def execute_fused_guarded(host, layout, causal, workers, device):
    expected, bound = reference_and_bound(*host, layout, causal)
    inputs = [value.to(device) for value in host]
    b, d1, d2, dn = host[0].shape
    q, h = (d1, d2) if layout == "BSND" else (d2, d1)
    n = host[2].shape[1 if layout == "BSND" else 2]
    dr = host[1].shape[-1]
    output, guard = guarded((b, q * h, dn), torch.float16, device)
    args = (
        inputs[0].view(b, q * h, dn),
        inputs[1].view(b, q * h, dr),
        inputs[2].view(b, n, dn),
        inputs[3].view(b, n, dr),
        inputs[4].view(b, n, dn),
        output,
    )
    for preload in (0, 1):
        for _ in range(2):
            output.fill_(float("nan"))
            step = 128 if causal else q
            for start in range(0, q, step):
                end = min(start + step, q)
                prefix = ((n - q + end + 127) // 128) * 128 if causal else n
                kernel = make_fused(
                    b,
                    q,
                    n,
                    h,
                    dn,
                    dr,
                    "float16",
                    layout,
                    causal,
                    workers,
                    cap_keys=prefix,
                    query_start=start,
                    query_end=end,
                    preload=preload,
                )
                kernel[None, workers](*args)
            torch.npu.synchronize()
            got = output.view(host[0].shape).cpu().double()
            assert torch.isfinite(got).all()
            assert ((got - expected).abs() <= bound).all()
            assert (guard[:64] == 19).all() and (guard[-64:] == 19).all()
    for actual, original in zip(inputs, host):
        assert torch.equal(actual.cpu(), original)


@pytest.mark.parametrize(
    "shape,causal",
    [
        ((2, 32, 128, 32, 192, 64), True),
        ((1, 128, 256, 32, 448, 128), True),
        ((1, 256, 256, 16, 512, 64), True),
        ((2, 16, 128, 16, 64, 64), False),
        ((2, 48, 128, 80, 192, 64), True),
    ],
)
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("workers", [1, 7])
def test_fused_shapes_layouts_and_aliases(shape, causal, layout, workers, device):
    execute_fused_guarded(
        make_inputs(shape, torch.float16, layout), layout, causal, workers, device
    )


@pytest.mark.parametrize(
    "kind",
    [
        "floor",
        "ceiling",
        "tiny",
        "subnormal",
        "tiny_mixed",
        "subnormal_mixed",
        "rope_only",
        "last_key",
    ],
)
def test_fused_numerical_boundaries(kind, device):
    host = make_inputs((1, 16, 128, 16, 512, 128), torch.float16, "BNSD")
    qn, qr, kn, kr, values = host
    if kind in ("floor", "ceiling"):
        sign = -1 if kind == "floor" else 1
        qn.fill_(1)
        qr.fill_(1)
        kn.fill_(sign)
        kr.fill_(sign)
        values.fill_(0.5)
    elif kind == "rope_only":
        qn.zero_()
        qr.fill_(1)
        kr[:, :, :64].fill_(1)
        kr[:, :, 64:].fill_(-1)
        values[:, :, :64].fill_(0.5)
        values[:, :, 64:].fill_(-0.5)
    elif kind == "last_key":
        for value in host:
            value.zero_()
        values[:, :, -1].fill_(1)
    else:
        qn.zero_()
        qr.zero_()
        magnitude = 2.0**-24 if kind.startswith("subnormal") else 2.0**-14
        values.fill_(magnitude)
        if kind.endswith("mixed"):
            pattern = torch.tensor([1, 3, -1, 0]).repeat(32)
            signs = torch.where(torch.arange(512) % 2 == 0, 1, -1)
            values[:] = (pattern[None, None, :, None] * signs * magnitude).half()
    execute_fused_guarded(host, "BNSD", True, 3, device)


def test_fused_wrapper_and_dtype_scope(device):
    host = make_inputs((1, 128, 256, 32, 192, 64), torch.float16, "BSND")
    expected, bound = reference_and_bound(*host, "BSND", True)
    got = mla_fused(*(value.to(device) for value in host), causal=True, workers=3)
    torch.npu.synchronize()
    assert got.is_contiguous()
    assert ((got.cpu().double() - expected).abs() <= bound).all()
    with pytest.raises(ValueError, match="FP16 only"):
        make_fused(1, 128, 128, 32, 512, dtype_name="bfloat16")


def execute(host, layout, rows, workers, device):
    expected, bound = reference_and_bound(*host, layout, False)
    inputs = [value.to(device) for value in host]
    batch = host[0].shape[0]
    heads = host[0].shape[2 if layout == "BSND" else 1]
    keys = host[2].shape[1 if layout == "BSND" else 2]
    nope, rope = host[0].shape[-1], host[1].shape[-1]
    output, guard = guarded((batch, heads, nope), torch.float16, device)
    args = (
        inputs[0].view(batch, heads, nope),
        inputs[1].view(batch, heads, rope),
        inputs[2].view(batch, keys, nope),
        inputs[3].view(batch, keys, rope),
        inputs[4].view(batch, keys, nope),
        output,
    )
    serial = None
    for preload in (0, 1):
        kernel = make_decode_kernel(
            batch, keys, heads, nope, rope, rows=rows, preload=preload
        )
        for _ in range(2):
            output.fill_(float("nan"))
            kernel[None, workers](*args)
            torch.npu.synchronize()
            actual = output.view(host[0].shape).cpu()
            assert torch.isfinite(actual).all()
            assert ((actual.double() - expected).abs() <= bound).all()
            assert (guard[:64] == 19).all() and (guard[-64:] == 19).all()
            if serial is None:
                serial = actual.clone()
            else:
                assert torch.equal(actual, serial)
    for actual, original in zip(inputs, host):
        assert torch.equal(actual.cpu(), original)


@pytest.mark.parametrize(
    "shape,rows",
    [
        ((3, 1, 384, 64, 512, 64), 64),
        ((4, 1, 128, 128, 64, 64), 32),
        ((2, 1, 640, 80, 192, 128), 16),
        ((3, 1, 256, 64, 512, 128), 64),
        ((2, 1, 384, 96, 448, 64), 32),
        ((1, 1, 4096, 64, 512, 64), 64),
    ],
)
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("workers", [1, 3])
def test_decode_shapes_and_ring_phase(shape, rows, layout, workers, device):
    execute(make_inputs(shape, torch.float16, layout), layout, rows, workers, device)


@pytest.mark.parametrize(
    "kind",
    [
        "floor",
        "ceiling",
        "early_max",
        "late_max",
        "tiny",
        "subnormal",
        "cancellation",
        "last_key",
        "rope_only",
    ],
)
def test_decode_state_transitions(kind, device):
    host = make_inputs((3, 1, 384, 64, 512, 128), torch.float16, "BSND")
    qn, qr, kn, kr, values = host
    if kind in ("floor", "ceiling", "early_max", "late_max", "rope_only"):
        qn.fill_(0 if kind == "rope_only" else 1)
        qr.fill_(1)
        kn.fill_(-1)
        kr.fill_(-1)
        values.fill_(0.25)
        if kind == "ceiling":
            kn.fill_(1)
            kr.fill_(1)
        elif kind in ("early_max", "late_max", "rope_only"):
            selected = slice(0, 128) if kind == "early_max" else slice(256, 384)
            kn[:, selected].fill_(1)
            kr[:, selected].fill_(1)
            values[:, selected].fill_(0.75)
    else:
        qn.zero_()
        qr.zero_()
        if kind in ("tiny", "subnormal"):
            values.fill_(2.0 ** (-24 if kind == "subnormal" else -14))
        elif kind == "cancellation":
            values[:, ::2].fill_(1)
            values[:, 1::2].fill_(-1)
        else:
            values.zero_()
            for batch, key in enumerate((0, 383, 129)):
                values[batch, key].fill_((batch + 1) / 4)
    execute(host, "BSND", 32, 2, device)


def test_decode_wrapper_and_query_scope(device):
    host = make_inputs((1, 1, 384, 80, 192, 64), torch.float16, "BNSD")
    expected, bound = reference_and_bound(*host, "BNSD", False)
    actual = mla_decode(*(value.to(device) for value in host), layout="BNSD", workers=3)
    torch.npu.synchronize()
    assert ((actual.cpu().double() - expected).abs() <= bound).all()
    assert actual.is_contiguous()
    with pytest.raises(ValueError, match="single query"):
        mla_decode(
            *(
                value.expand(-1, -1, 2, -1) if i < 2 else value
                for i, value in enumerate(host)
            ),
            layout="BNSD",
        )
