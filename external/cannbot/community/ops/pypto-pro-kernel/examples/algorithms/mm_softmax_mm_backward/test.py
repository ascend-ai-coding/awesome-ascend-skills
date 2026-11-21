# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Non-square, nonzero reverse passes, independent reference checks and public guards.

Use examples/pytest.ini and its device fixture. Both input dtypes use the same
FP32-output criterion: abs(error) <= 2e-5 + 2e-4*abs(reference). The reference
never rounds the derived matrices to the Cube component dtype.
"""

import pytest
import torch

from .kernel import execute, make_kernels
from .onchip import execute_onchip, make_onchip
from .reference import reference, scalar_objective

ATOL, RTOL = 2e-5, 2e-4


def assert_gradients(actual, expected):
    for got, want in zip(actual, expected):
        assert got.dtype == torch.float32
        assert got.shape == want.shape
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.cpu().double(), want, atol=ATOL, rtol=RTOL)


def allocate(shape, dtype, device):
    m, k, n, d = shape
    outputs, backings = [], []
    for out_shape in ((m, k), (k, n), (n, d)):
        count = out_shape[0] * out_shape[1]
        backing = torch.full((count + 64,), 29, dtype=torch.float32, device=device)
        outputs.append(backing[32:-32].view(out_shape))
        backings.append(backing)
    workspace = [torch.empty((m, n), dtype=kind, device=device) for kind in (
        torch.float32, torch.float32, dtype, dtype, torch.float32, torch.float32, dtype, dtype,
    )]
    return outputs, workspace, backings


def inputs(shape, dtype, seed=71):
    m, k, n, d = shape
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(size, generator=generator).mul(2).sub(1).mul(scale).to(dtype) for size, scale in (
        ((m, k), 0.25), ((k, n), 0.25), ((n, d), 0.75), ((m, d), 0.75),
    )]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 128, 192, 256), (128, 64, 256, 192)])
def test_rectangular_gradients(shape, dtype, device):
    original = inputs(shape, dtype)
    _, expected = reference(*original)
    assert all(value.abs().amax() > 0.005 for value in expected)
    native = [value.to(device) for value in original]
    outputs, workspace, backings = allocate(shape, dtype, device)
    kernels = make_kernels(*shape, str(dtype).split(".")[-1])
    for seed in (71, 73):
        changed = inputs(shape, dtype, seed)
        _, expected = reference(*changed)
        for destination, source in zip(native, changed):
            destination.copy_(source)
        for value in outputs + workspace:
            value.fill_(float("nan"))
        execute(kernels, *native, outputs, workspace)
        torch.npu.synchronize()
        assert_gradients(outputs, expected)
        for backing in backings:
            assert (backing[:32].cpu() == 29).all()
            assert (backing[-32:].cpu() == 29).all()
        for got, want in zip(native, changed):
            assert torch.equal(got.cpu(), want)
        p, ds = workspace[1].cpu(), workspace[5].cpu()
        torch.testing.assert_close(p.sum(1), torch.ones(shape[0]), atol=3e-6, rtol=0)
        torch.testing.assert_close(ds.sum(1), torch.zeros(shape[0]), atol=3e-6, rtol=0)


def test_reference_directional_derivative():
    generator = torch.Generator().manual_seed(17)
    a, b, c, g = [torch.randn(size, generator=generator, dtype=torch.float64).mul(0.2) for size in (
        (3, 5), (5, 7), (7, 4), (3, 4),
    )]
    _, gradients = reference(a, b, c, g)
    directions = [torch.randn(value.shape, generator=generator, dtype=torch.float64) for value in (a, b, c)]
    step = 1e-5
    positive = scalar_objective(*(value + step * direction for value, direction in zip((a, b, c), directions)), g)
    negative = scalar_objective(*(value - step * direction for value, direction in zip((a, b, c), directions)), g)
    finite_difference = (positive - negative) / (2 * step)
    analytical = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
    torch.testing.assert_close(finite_difference, analytical, atol=1e-9, rtol=1e-7)


def test_comparison_rejects_corruption():
    expected = tuple(torch.ones((2, 3), dtype=torch.float64) for _ in range(3))
    corrupted = [value.float() for value in expected]
    corrupted[1][0, 1] += 0.01
    with pytest.raises(AssertionError):
        assert_gradients(corrupted, expected)


def test_bfloat16_coefficient_residual(device):
    shape, dtype = (64, 128, 192, 256), torch.bfloat16
    cpu = inputs(shape, dtype)
    cpu[0].zero_()
    cpu[1].zero_()
    cpu[3].fill_(1)
    _, expected = reference(*cpu)
    high_only = torch.full_like(expected[2], 64 * float(torch.tensor(1 / 192, dtype=dtype)))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(high_only, expected[2], atol=ATOL, rtol=RTOL)
    native = [value.to(device) for value in cpu]
    outputs, workspace, _ = allocate(shape, dtype, device)
    execute(make_kernels(*shape, "bfloat16"), *native, outputs, workspace)
    torch.npu.synchronize()
    assert_gradients(outputs, expected)


def test_constant_columns_cancel_score_gradient(device):
    shape, dtype = (64, 128, 192, 256), torch.bfloat16
    cpu = inputs(shape, dtype)
    cpu[2][:] = cpu[2][0]
    _, expected = reference(*cpu)
    assert expected[0].abs().amax() < 1e-12
    assert expected[1].abs().amax() < 1e-12
    assert expected[2].abs().amax() > 0.01
    outputs, workspace, _ = allocate(shape, dtype, device)
    execute(make_kernels(*shape, "bfloat16"), *(value.to(device) for value in cpu), outputs, workspace)
    torch.npu.synchronize()
    assert_gradients(outputs, expected)


def test_rejects_strided_public_input():
    a, b, c, upstream = inputs((64, 128, 192, 256), torch.float16)
    with pytest.raises(ValueError, match="contiguous"):
        execute(None, a.transpose(0, 1), b, c, upstream, (), ())


@pytest.mark.parametrize("shape", [(63, 128, 192, 256), (64, 0, 192, 256), (64, 128, 1088, 256)])
def test_rejects_unsupported_extents(shape):
    with pytest.raises(ValueError, match="multiples of 64"):
        make_kernels(*shape)



def _onchip_check(shape, dtype, device, query_rows=16, scenarios=None):
    m, _, n, _ = shape
    program = make_onchip(*shape, str(dtype).split(".")[-1], query_rows=query_rows)
    native = [value.to(device) for value in inputs(shape, dtype)]
    outputs, _, backings = allocate(shape, dtype, device)
    workspace = [torch.empty((m, n), dtype=dtype, device=device) for _ in range(4)]
    for cpu in scenarios or [inputs(shape, dtype, seed) for seed in (71, 73)]:
        expected = reference(*cpu)[1]
        for dst, src in zip(native, cpu):
            dst.copy_(src)
        for value in outputs + workspace:
            value.fill_(float("nan"))
        execute_onchip(program, *native, outputs, workspace)
        torch.npu.synchronize()
        assert_gradients(outputs, expected)
        for backing in backings:
            assert (backing[:32].cpu() == 29).all()
            assert (backing[-32:].cpu() == 29).all()
        for got, want in zip(native, cpu):
            assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 128, 192, 256), (128, 64, 256, 192)])
def test_onchip_rectangular_gradients(shape, dtype, device):
    _onchip_check(shape, dtype, device)


def test_onchip_row_tile_configuration(device):
    _onchip_check((64, 128, 192, 256), torch.bfloat16, device, query_rows=32)


def test_onchip_cancellation_and_coefficient_residual(device):
    shape, dtype = (64, 128, 192, 256), torch.bfloat16
    constant = inputs(shape, dtype, 79)
    constant[2][:] = constant[2][0]
    uniform = inputs(shape, dtype, 83)
    uniform[0].zero_()
    uniform[1].zero_()
    uniform[3].fill_(1)
    _onchip_check(shape, dtype, device, scenarios=[constant, uniform])


def test_onchip_rejects_larger_resident_axis():
    with pytest.raises(ValueError, match="in \\[64,256\\]"):
        make_onchip(64, 128, 512, 256)
