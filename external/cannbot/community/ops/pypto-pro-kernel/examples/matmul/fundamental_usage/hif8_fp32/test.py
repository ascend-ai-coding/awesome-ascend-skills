# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""低精度 dtype 的验收判据：参考实现必须吃 kernel 真正吃到的那些值。

**解决什么.** HF8（`pl.DT_HF8`）一个字节一个元素，尾数只有 1–3 位。拿原始 fp32 当参考
去比，误差是百分之几量级，`rtol=1e-2` 过不去——于是很容易把阈值一路放宽到「能过」。
**但那个误差不是 kernel 算错，是参考实现吃错了输入。** 把量化搬到 host 侧、让参考吃**量化
后**的值，`rtol` 可以一直保持 1e-3。

[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 写着「不静默放宽比较阈值」。这个样例就是那条规则在低精度 dtype 上的具体形态，
而且两边都测：正确的参考过，错误的参考**必须不过**。

**这个数是多少.** 本样例的输入栅格上，用原始 fp32 当参考的相对误差**中位数 7.4%**
（2026-09-26 实测），正好是栅格相邻两值的相对间隔（0.067 .. 0.125）。也就是说：你想吸收进
容差里的那个误差，就是量化步长本身。最大值没有意义——`golden` 里有接近零的项，除出来会到
几百倍，所以这条测试用中位数，不用最大值。

**怎么绕开 130 行 codec.** HF8 是**渐变精度**（tapered）：指数域宽度随数值变化，尾数位跟着
从 3 位缩到 1 位，所以完整的 256 项表需要一套位解码器。框架 ST 里那份有 130 行，而且只存在
于那个测试文件里（没有出货模块）。这里只取表里三段**常规阶梯**——形状都是 `(1 + m/8) * 2^e`，
48 个字节的值可以直接从位布局写出来，推导已对着 ST 的解码器逐码验过（48/48）。细节和它的
局限见 [`reference.py`](reference.py)。

**覆盖边界要说清.** 这段栅格的相对间隔只在 0.067 .. 0.125 之间，完整的 HF8 栅格是
1/9 .. 1.0。所以「一个固定 rtol 不能表示『差一格』」这条在整张表上才成立。要在 HF8
**输出**上验收，用 ST 的 `_assert_hif8_codes_close`——比编码、按格数打分；本样例的输出是
FP32，用不到那套。

**操作数通路全程 HF8.** `pl.load` 和 `pl.move` 从不做类型转换，所以 GM → L1 → L0A/L0B 都是
HF8，只有累加器是 FP32。这也是结果能被紧容差比较的原因：K 方向 128 项的累加在 FP32 上做。
有一条测试直接在解析后的 IR 上按内存空间核对每块 tile 的 dtype。

**如何独立验证.** 五条：

1. **栅格本身**：48 个不同值、单调、吸附是幂等的（在栅格上的值不会被挪动）。
2. **IR 里每块 tile 的 dtype**：Mat / Left / Right 全 HF8，Acc 是 FP32。
3. **正确的参考在 `rtol=1e-3` 下过**。
4. **错误的参考（吃原始 fp32）在同一容差下不过**，而且差出来的量级就是量化步长。
   这条是这个样例的重点——它把「放宽阈值」这个动作的后果摆出来。
5. **两遍逐位相同**。

来源：``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_hf8_fp32.py``（2008 行、
20 个 kernel 工厂）。那里 `_make_operand` 的 docstring 把这条判据写得最清楚：「the golden
uses the decoded values, **which are exactly what the Cube consumes**」，`_IN_TOL["hf8"]`
因此可以和 fp32 一样紧（`rtol=1e-3`）。这里只取 `HF8 × HF8 → FP32` 一条链；per-tile 的
浮点 scale、transpose、atomic、多核 K 切分在那份 ST 和本库别的样例里。
"""

import re

import numpy as np
import pytest
import torch

from .kernel import K, M, N, matmul_hf8
from .reference import GRID_MAX, GRID_VALUES, matmul_reference, snap_to_grid

TOL = dict(rtol=1e-3, atol=1e-3)
_TILE_TYPE = re.compile(
    r"ir\.TileType\(\[[^\]]*\], ir\.(\w+), ir\.MemRef\(ir\.MemorySpace\.(\w+)"
)


def _operands(seed=0):
    """返回 ((a 字节, a 精确值), (b 字节, b 精确值), 原始 fp32 两份)。"""
    rng = np.random.default_rng(seed)
    raw_a = ((rng.random((M, K), dtype=np.float32) * 2 - 1) * GRID_MAX).astype(
        np.float32
    )
    raw_b = ((rng.random((K, N), dtype=np.float32) * 2 - 1) * GRID_MAX).astype(
        np.float32
    )
    return snap_to_grid(raw_a), snap_to_grid(raw_b), (raw_a, raw_b)


def _run(device, a_codes, b_codes):
    """HF8 在 torch 里没有 dtype，操作数以 uint8 裸码传进去。"""
    a = torch.from_numpy(np.ascontiguousarray(a_codes)).to(device)
    b = torch.from_numpy(np.ascontiguousarray(b_codes)).to(device)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)
    matmul_hf8(a, b, out)
    torch.npu.synchronize()
    return out.cpu()


def test_the_grid_is_sane():
    """先验栅格：后面几条全靠它。"""
    assert GRID_VALUES.size == 48
    assert np.unique(GRID_VALUES).size == 48, "48 个码应当对应 48 个不同的值"
    assert np.all(np.diff(GRID_VALUES) > 0), "栅格必须是排好序的，吸附靠这个"

    # 在栅格上的值不该被挪动——否则「精确可表示」这个前提就是假的。
    codes, snapped = snap_to_grid(GRID_VALUES)
    assert np.array_equal(snapped, GRID_VALUES)

    positive = GRID_VALUES[GRID_VALUES > 0].astype(np.float64)
    gaps = (positive[1:] - positive[:-1]) / positive[:-1]
    assert 0.06 < gaps.min() < gaps.max() < 0.13, (
        f"相邻相对间隔应当在 0.067..0.125，实际 {gaps.min():.4f}..{gaps.max():.4f}"
    )


def test_the_whole_operand_path_is_hf8():
    """`pl.load` / `pl.move` 不转换类型，所以只有累加器是 FP32。"""
    import pypto_pro as pp

    program, _ = matmul_hf8.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)
    text = pp.ir.python_print(program)

    by_space: dict[str, set[str]] = {}
    for dtype, space in _TILE_TYPE.findall(text):
        by_space.setdefault(space, set()).add(dtype)

    assert by_space == {
        "Mat": {"HF8"},
        "Left": {"HF8"},
        "Right": {"HF8"},
        "Acc": {"FP32"},
    }, f"IR 里的 (内存空间 -> dtype) 是 {by_space}"


def test_matches_the_reference(device):
    """参考吃量化后的值，容差就能和 fp32 一样紧。"""
    (a_codes, a_exact), (b_codes, b_exact), _ = _operands()

    out = _run(device, a_codes, b_codes)

    torch.testing.assert_close(out, matmul_reference(a_exact, b_exact), **TOL)


def test_the_raw_fp32_golden_does_not_pass(device):
    """同一次运行，换成吃原始 fp32 的参考——**必须不过**。

    这条是这个样例的重点：它把「精度不够所以放宽阈值」这个念头的来源摆出来。差出来的量级
    就是量化步长，不是 kernel 的误差。
    """
    (a_codes, a_exact), (b_codes, b_exact), (raw_a, raw_b) = _operands()

    out = _run(device, a_codes, b_codes)
    right = matmul_reference(a_exact, b_exact)
    wrong = matmul_reference(raw_a, raw_b)

    torch.testing.assert_close(out, right, **TOL)  # 正确的那份过
    with pytest.raises(AssertionError):  # 错误的那份不过
        torch.testing.assert_close(out, wrong, **TOL)

    # 而且差的量级就是量化步长（栅格相对间隔 0.067..0.125）。用中位数：golden 里有接近零
    # 的项，最大相对误差会被除法放大到几百倍，没有意义。
    rel = (right - wrong).abs() / wrong.abs().clamp_min(1e-6)
    median = float(rel.median())
    assert 0.03 < median < 0.20, f"相对误差中位数 {median:.4f}，不在量化步长的量级上"


def test_two_runs_agree_bit_for_bit(device):
    """同一份输入跑两遍，逐位相同。"""
    (a_codes, _), (b_codes, _), _ = _operands(seed=2)

    first = _run(device, a_codes, b_codes)
    second = _run(device, a_codes, b_codes)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
