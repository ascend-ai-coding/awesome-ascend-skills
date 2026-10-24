# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立验证标量控制、VF 比较与行/列归约。

有限随机输入的 max/min 应逐元素精确相等；[16,64] 的非方阵区分归约轴。
行归约的 DN 紧凑列与 ND 补齐行均输出 [M,1]，比较公开结果而不等同物理布局。
解析反例只覆盖标量接口：pl.min 接受标量，pl.const 要求字面量。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidType, RuntimeFailure

from .kernel import (
    CLAMP_HIGH,
    CLAMP_LOW,
    K,
    LOOP_LIMIT,
    M,
    col_min_dim1,
    const_and_astype,
    const_of_a_named_constant,
    const_of_a_runtime_value,
    elementwise_tile_and_scalar,
    row_max_dim0,
    row_max_dim0_without_dn,
    scalar_bounds,
    scalar_min_on_tiles,
)
from .reference import (
    clamped,
    col_min_dim1 as ref_col_min,
    elementwise_max,
    make_input,
    row_max_dim0 as ref_row_max,
    the_two_dims_disagree,
)

BLOCK_COUNTS = [1, 8]
"""`blocks` 取 1 和 8：让 `pl.min(block_num, 4)` 两条分支都走到
（1 < 4 取 block_num，8 > 4 取 4）。只测一个的话另一条分支没验。"""


def _run_scalars(kernel, device, size, blocks=1):
    out = torch.full((size,), -1, dtype=torch.int64, device=device)
    kernel[None, blocks](out)
    torch.npu.synchronize()
    return [int(v) for v in out.cpu().numpy()]


def _run_elementwise(device):
    a = make_input(M, K)
    b = make_input(M, K, seed=11)
    o_pair = torch.zeros((M, K), dtype=torch.float32, device=device)
    o_clamped = torch.zeros((M, K), dtype=torch.float32, device=device)
    elementwise_tile_and_scalar[None, 1](
        torch.from_numpy(a).to(device),
        torch.from_numpy(b).to(device),
        o_pair,
        o_clamped,
    )
    torch.npu.synchronize()
    return a, b, o_pair.cpu().numpy(), o_clamped.cpu().numpy()


def _run_reduce(kernel, device, out_shape):
    a = make_input(M, K)
    out = torch.zeros(out_shape, dtype=torch.float32, device=device)
    kernel[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# 一、标量
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", BLOCK_COUNTS)
def test_scalar_min_and_max_pick_the_right_side(device, blocks):
    """验证 1。`min(n, 4)` 和 `max(n, 4)`，两个 `blocks` 让两条分支都走到。

    最后一个判据是恒等式 `min + max == n + 4`：**它不依赖具体核数**,
    所以就算这台机器的核数变了，这一条仍然有效。
    """
    num, lo, hi, total = _run_scalars(scalar_bounds, device, 4, blocks=blocks)
    assert num == blocks, f"get_block_num 读到 {num}，主机侧给的是 {blocks}"
    assert lo == min(num, LOOP_LIMIT), f"pl.min({num}, {LOOP_LIMIT}) = {lo}"
    assert hi == max(num, LOOP_LIMIT), f"pl.max({num}, {LOOP_LIMIT}) = {hi}"
    assert total == num + LOOP_LIMIT, (
        f"min + max = {total}，恒等式要求 {num + LOOP_LIMIT}"
    )


def test_const_takes_a_literal_and_astype_takes_both(device):
    """验证 2。同一个 7 走 `pl.const` 和 `pl.astype` 两条路，结果相同。

    **`astype` 吃字面量这件事是实测的**：docstring 只说它转 "runtime scalar expression"。
    """
    typed, converted, from_runtime, negative = _run_scalars(const_and_astype, device, 4)
    assert typed == 7, f"pl.const(7, I32) 读回 {typed}"
    assert converted == 7, (
        f"pl.astype(7, I32) 读回 {converted} —— astype 不吃字面量了？"
    )
    assert typed == converted, "两条路给出不同的 7"
    assert from_runtime == 1, (
        f"pl.astype(get_block_num(), I32) 读回 {from_runtime}，单 block 应为 1"
    )
    assert negative == -3, f"pl.const(-3, I64) 读回 {negative}"


# ---------------------------------------------------------------------------
# 二、逐元素
# ---------------------------------------------------------------------------


def test_elementwise_max_of_two_tiles(device):
    """验证 3 前半。逐位等于 `np.maximum`。

    **逐位相等而不是 allclose**：max 只是选一个已有的值，没有精度可言。
    """
    a, b, got, _ = _run_elementwise(device)
    np.testing.assert_array_equal(got, elementwise_max(a, b))


def test_a_clamp_is_maximum_then_minimum(device):
    """验证 3 后半。`maximum` 到下界再 `minimum` 到上界。

    上下界写反会变成 `[0.5, -0.5]`，那是个空区间，结果会全塌成一个值 —— 判据一定红。
    所以这条判据除了比参考，还单独断言结果落在界内：**两个断言查的是不同的错**
    （比参考查「算错了」，查界查「界本身传错了」）。

    kernel 里那两个数是**字面量**而不是 `pl.const(CLAMP_LOW, ...)`：常量名会被拒，见 §六。
    所以这条判据同时在守「kernel 里的字面量和这里的 `CLAMP_LOW` 没有走散」。
    """
    a, _, _, got = _run_elementwise(device)
    np.testing.assert_array_equal(got, clamped(a))
    assert got.min() >= np.float32(CLAMP_LOW), f"夹逼后最小值 {got.min()} 低于下界"
    assert got.max() <= np.float32(CLAMP_HIGH), f"夹逼后最大值 {got.max()} 高于上界"


# ---------------------------------------------------------------------------
# 三、归约
# ---------------------------------------------------------------------------


def test_dim0_keeps_the_first_axis(device):
    """验证 4。`dim=0` → `out[i, 0] = max_j(a[i, j])`，numpy 侧是 `axis=1`。

    **两个数字相反**，这是这一例最该记住的事。
    """
    a, got = _run_reduce(row_max_dim0, device, (M, 1))
    np.testing.assert_array_equal(got, ref_row_max(a))
    assert got.shape == (M, 1), got.shape


def test_dim1_keeps_the_second_axis(device):
    """验证 5。`dim=1` → `out[0, j] = min_i(a[i, j])`，numpy 侧是 `axis=0`。

    顺手断言两个 `dim` 的输出形状不同——**输入取 `16 × 64` 而不是方阵就是为了这个**:
    方阵上行列写反也会「对」。
    """
    a, got = _run_reduce(col_min_dim1, device, (1, K))
    np.testing.assert_array_equal(got, ref_col_min(a))
    assert got.shape == (1, K), got.shape
    assert the_two_dims_disagree(a), "两个 dim 的输出形状一样，这份输入没有区分力"


def test_row_output_layouts_preserve_the_result(device):
    a, with_dn = _run_reduce(row_max_dim0, device, (M, 1))
    a2, with_padded_nd = _run_reduce(row_max_dim0_without_dn, device, (M, 1))
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(with_dn, with_padded_nd)
    np.testing.assert_array_equal(with_dn, ref_row_max(a))


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


def test_scalar_min_on_tiles_is_refused():
    """验证 6 之一。`pl.min` 只吃标量。逐元素计算应在 VF 中使用 `vf.min`。

    消息说的是「表达式必须是 ScalarType 的 Var」——**它没点出该用哪个接口**,
    所以这条判据同时是一份「看到这个消息该往哪想」的记录。
    """
    with pytest.raises(RuntimeFailure) as raised:
        _parse(scalar_min_on_tiles)
    assert "ScalarType" in str(raised.value), str(raised.value)


def test_const_of_a_runtime_value_is_refused():
    """验证 6 之三。`pl.const` 只吃数字字面量，运行期标量要用 `pl.astype`。

    **两个接口的分界是单向的**：`astype` 能干 `const` 的活（验证 2 量过），反过来不行。
    """
    with pytest.raises(InvalidType) as raised:
        _parse(const_of_a_runtime_value)
    assert "must be a numeric literal" in str(raised.value), str(raised.value)


def test_const_of_a_named_constant_is_also_refused():
    """验证 6 之四，**比上一条更值得记**。`pl.const(CLAMP_LOW, FP32)` 也被拒,
    哪怕 `CLAMP_LOW = -0.5` 写在模块顶上。

    检查看的是 AST 节点类型（`ast.Constant`），不是值。消息和上一条一模一样——
    「must be a numeric literal」——所以看到它的时候分不出是哪种情况。
    这一条判据的作用就是把两种情况都记下来，见 §六。
    """
    with pytest.raises(InvalidType) as raised:
        _parse(const_of_a_named_constant)
    assert "must be a numeric literal" in str(raised.value), str(raised.value)
