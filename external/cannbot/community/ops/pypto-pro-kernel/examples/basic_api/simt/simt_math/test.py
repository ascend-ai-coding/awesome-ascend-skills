# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""SIMT 的标量数学：十个超越函数 + `fma` + `abs` / `min` / `max`。

**写法：原始。**

## 一、三条面算同一个数学函数，精度不保证一致

| 面 | 接口 | 一条语句算 |
| --- | --- | --- |
| block op | ``pl.exp(out, src)`` | 整个 tile |
| vf | ``vf.exp(reg)`` | 一个向量寄存器 |
| **SIMT** | ``pl.simt.exp(value)`` | **一个标量** |

三者是不同的硬件路径。**这一例只标定 SIMT 这一条的精度**，不声称另两条一样。

## 二、输入怎么选

| | 取值 | 给谁用 |
| --- | --- | --- |
| `a` | `0.25 .. 8.0` 严格正，跨过 1 | `sqrt` / `rsqrt` / `log` / `log2` / `log1p` |
| `b` | `-4 .. 4`，**含 0** | `exp` / `exp2` / `sin` / `cos` / `tanh` / `abs` / `min` / `max` |

`a` 跨过 1 是为了让 `log` 变号（只取大于 1 的值时，符号错了看不出来）。
`b` 里的 0 是**显式塞进去的**——`linspace(-4, 4, 32)` 点数是偶数，正好跨过 0 而不落在 0 上,
不含 0 的话 `sin(0)` / `tanh(0)` / `abs(0)` 一个边界都验不到。

`a` 不含 0：`rsqrt(0)` / `log(0)` 是 inf，那属于边界行为，在 ``simt_round_pred``。

## 三、容差是量出来的，而且分三组

| 组 | 算子 | 实测最大相对误差 | 容差 |
| --- | --- | ---: | --- |
| 逐位相等 | `fma` `abs` `min` `max` | **0** | `rtol=0, atol=0` |
| 约 1 ulp | `sqrt` `rsqrt` `exp` `exp2` `log` `log2` `log1p` `sin` `cos` | ≤ 1.18e-07 | `rtol=2e-07` |
| 约 3 ulp | `tanh` | 3.48e-07 | `rtol=5e-07` |

FP32 的 eps 是 1.19e-07，所以第二组就是"最后一位"。

**`sqrt` 不在逐位组里。** 我最初以为它会在（IEEE 754 要求 `sqrt` 精确舍入），实测 1.12e-07
——硬件上它和别的超越函数一样是近似实现。**这句话是被实测改过的。**

**`fma` 逐位相等本身是个结论**：参考用 float64 算 `a*b+a` 再转 float32，和硬件结果完全一致,
说明它只舍入一次。分开算（先乘后加，两次舍入）的话这一组过不了 `rtol=0`。

`atol` 一律 0，有依据：`sin(0)` 和 `tanh(0)` 硬件返回**精确的 0**，否则它们的相对误差会是天文
数字（分母为 0）。**这就是为什么输入里必须有 0**——它同时在验边界和在验 `atol=0` 能不能立住。

## 四、如何独立验证

十四个算子各一条判据，逐个用自己的容差比 CPU 参考（`float64` 算完转 `float32`——
**参考要比被测的更准**，否则两边都有误差时分不清是谁的）。

**四条 `rtol=0` 的判据是这一例最有价值的部分**：它们不给退化留余地。剩下十条只能说"在 1 ulp
内"，而"1 ulp 内"这个结论本身要靠那张实测表撑着，不是一句客套话。

## 五、这一例不讲什么

- **取整与谓词**——五种取整、`isnan` / `isinf` / `isfinite`、`cast` / `bitcast` 在
  ``simt_round_pred``。
- **整数算子**——`popcount` / `mul_hi` / `fmod` 也在 ``simt_round_pred``。
- **inf / nan 的传播**——输入刻意都在定义域内。边界那一套在 ``simt_round_pred``。
- **和 vf / block op 的精度对比**——**没做**。要做得三条面各写一遍同一个函数再逐个比,
  那是一个独立的题目（`vf_transcendental` 标定的是 vf 那一条）。**这是明说跳过。**

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 三节。
"""

import numpy as np
import pytest
import torch

from .kernel import ROWS, THREADS, simt_scalar_math
from .reference import EXACT, OPS, RTOL, expected, make_inputs


def _run(device):
    a, b = make_inputs()
    out = torch.zeros((ROWS, THREADS), dtype=torch.float32, device=device)
    simt_scalar_math[None, 1](
        torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out
    )
    torch.npu.synchronize()
    return out.cpu().numpy(), expected(a, b)


@pytest.mark.parametrize("name", OPS)
def test_each_scalar_op_matches_the_cpu_reference(device, name):
    """十四个算子各一条。**容差按算子分**，四个是 `rtol=0`。"""
    got, want = _run(device)
    row = OPS.index(name)
    np.testing.assert_allclose(
        got[row],
        want[row],
        rtol=RTOL[name],
        atol=0.0,
        err_msg=f"`pl.simt.{name}` 超出实测标定的容差 {RTOL[name]:g}",
    )


def test_the_four_exact_ops_are_bit_identical(device):
    """`fma` / `abs` / `min` / `max` **逐位相等**，这一条把它单独说一遍。

    上面那条参数化测试已经用 `rtol=0` 覆盖了它们，但那样读起来像"容差恰好是 0"。
    **逐位相等是一个结论，不是一个容差选择**——尤其 `fma`：它等于 float64 算完再舍入一次,
    说明硬件真的只舍入一次。
    """
    got, want = _run(device)
    for name in EXACT:
        row = OPS.index(name)
        np.testing.assert_array_equal(
            got[row], want[row], err_msg=f"`pl.simt.{name}` 不再逐位相等"
        )
