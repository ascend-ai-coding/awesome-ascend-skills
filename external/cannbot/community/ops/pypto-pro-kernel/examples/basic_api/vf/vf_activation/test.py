# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""六个激活 / 融合乘加算子：`relu` `neg` `prelu` `leaky_relu` `mul_add_dst` `mul_dst_add`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py) 与 [`vf_arith`](../vf_arith/test.py)。

## 一、分界线：读不读 dst 的前值

| 算子 | 语义 | 读 dst？ |
| --- | --- | --- |
| `relu(src, preg)` | `max(0, src)` | 否 |
| `neg(src, preg)` | `-src` | 否 |
| `prelu(src, slope, preg)` | `src >= 0 ? src : src * slope[i]`——斜率是**寄存器** | 否 |
| `leaky_relu(src, alpha, preg)` | `src >= 0 ? src : src * alpha`——斜率是**标量** | 否 |
| `mul_add_dst(src0, src1, preg)` | `dst = src0*src1 + dst`（`vmula`，dst 当**加数**） | **是** |
| `mul_dst_add(src0, src1, preg)` | `dst = dst*src0 + src1`（`vmadd`，dst 当**被乘数**） | **是** |

后两个必须先给 dst 装值，和 `vf_arith` 里 `axpy` 同一条规则：**解析器按赋值目标的已有绑定
给它定型**，没有已有绑定就不成立。

## 二、`prelu` 和 `leaky_relu` 为什么是两个 API

不是写法差异，是斜率来源差异：`prelu` 逐元素取寄存器里的斜率，`leaky_relu` 全部 lane 共用
一个标量。本例故意让两者的斜率不同（`slope` 是 0.5/0.25 交替，`alpha` 固定 0.5），
`test_prelu_and_leaky_relu_differ_where_the_slopes_differ` 断言它们在斜率不同的位置**必须**
给出不同结果——否则这一例就没验到「逐元素 vs 标量」这件事。

## 三、两个融合算子给同一个数，这是特意安排的

`mul_add_dst(a, b)` 预装 dst=`c` → `a*b + c`；`mul_dst_add(b, c)` 预装 dst=`a` → `a*b + c`。
两条路径算同一个值，参考只给一个。**如果任一条的 dst 预装值写错，它就对不上**——这比
分别写两个参考更能抓住错误。

## 四、判据

`rtol=0, atol=0`，逐位相同。操作数是挑过的：`a` 跨 -32..31，斜率与 `b`/`c` 全取 2 的幂，
所以乘法只改阶码。不是「误差小」，是**没有误差**——换成非 2 的幂就必须改成带容差比，
那时这一例检验的东西就弱了。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import N, act_fused, act_unary

_UNARY_NAMES = ["relu", "neg", "prelu", "leaky_relu"]
_FUSED_NAMES = ["mul_add_dst", "mul_dst_add"]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _outs(device, count):
    return [
        torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(count)
    ]


def _run_unary(device):
    a, slope, _, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    act_unary(_dev(a, device), _dev(slope, device), *outs)
    torch.npu.synchronize()
    return a, slope, [tensor.cpu().numpy() for tensor in outs]


def test_the_four_pure_activations_match_the_reference(device):
    """四个只读 src 的算子，逐位比。"""
    a, slope, got = _run_unary(device)
    want = ref.unary(a, slope)
    for name, g, w in zip(_UNARY_NAMES, got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=0, err_msg=f"{name} 不一致")


def test_relu_zeroes_the_negative_half_and_keeps_the_rest(device):
    """防空测：`a` 必须真的跨正负，否则 relu 这一条什么也没验到。"""
    a, _, got = _run_unary(device)
    relu = got[0]
    assert (a < 0).any() and (a > 0).any(), "输入没有跨正负，relu 这一条是空的"
    assert (relu[a < 0] == 0).all(), "负半边应该全被清零"
    np.testing.assert_allclose(relu[a > 0], a[a > 0], rtol=0, atol=0)


def test_prelu_and_leaky_relu_differ_where_the_slopes_differ(device):
    """逐元素斜率 vs 标量斜率——这是两个 API 并存的唯一理由，必须验到。"""
    a, slope, got = _run_unary(device)
    prelu, leaky = got[2], got[3]

    differing = (a < 0) & (slope != ref.ALPHA)
    assert differing.any(), (
        f"没有「负值且斜率 != {ref.ALPHA}」的位置，这一条是空测——检查 make_operands"
    )
    assert (prelu[differing] != leaky[differing]).all(), (
        "斜率不同的负值位置上，prelu 和 leaky_relu 竟然给出相同结果"
    )

    same = (a < 0) & (slope == ref.ALPHA)
    if same.any():
        np.testing.assert_allclose(
            prelu[same], leaky[same], rtol=0, atol=0, err_msg="斜率相同的位置应该一致"
        )


def test_the_two_fused_ops_reach_the_same_value_by_different_routes(device):
    """`mul_add_dst` 与 `mul_dst_add` 都得到 `a*b + c`，走的却是不同的 dst 位置。

    两者都对 → dst 预装值两边都对。任一条错 → 它对不上参考。
    """
    a, _, b, c = ref.make_operands(N)
    outs = _outs(device, 2)
    act_fused(_dev(a, device), _dev(b, device), _dev(c, device), *outs)
    torch.npu.synchronize()
    got = [tensor.cpu().numpy() for tensor in outs]

    want = ref.fused(a, b, c)
    for name, g, w in zip(_FUSED_NAMES, got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=0, err_msg=f"{name} 不一致")

    # 防空测：如果 dst 没被读到，结果会是 a*b（少了 c），而 c 全非零
    assert (c != 0).all(), "c 有零元素，下面这条判别就不成立了"
    plain_product = (a.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
    assert not np.array_equal(got[0], plain_product), (
        "mul_add_dst 的结果等于 a*b，说明 dst 的前值没被读进来"
    )


@pytest.mark.parametrize("name", _UNARY_NAMES + _FUSED_NAMES)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 6 个算子每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
