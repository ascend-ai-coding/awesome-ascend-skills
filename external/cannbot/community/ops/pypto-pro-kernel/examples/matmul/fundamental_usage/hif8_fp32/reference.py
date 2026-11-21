# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""HF8 的常规阶梯：值和字节都能从格式直接写出来，不需要整套 codec。

HF8（`pl.DT_HF8`）是**渐变精度**（tapered）：指数域的宽度随数值大小变化，尾数位跟着从
3 位缩到 1 位。所以它那张 256 项的值表需要一套位解码器——框架 ST 里那份有 130 行
（`test_matmul_hf8_fp32.py:1425` 起 `_hif8_decode_bitstring`），而且**只存在于那个测试
文件里**，没有出货模块（`tests/ut/pypto_pro/runtime/test_hif8_reference.py` 是用
`importlib` 按路径把它加载进来的）。

这里不照搬那 130 行，只用表里三段**常规阶梯**。这三段的形状都是 `(1 + m/8) * 2^e`，
位布局直接可读：

```
d=0001   bits  s 0 0 0 1 m m m    code = s*128 +  8      + m    value = (1+m/8) * 2^0
d=001    bits  s 0 0 1 e m m m    code = s*128 + 16 + 8e + m    value = (1+m/8) * 2^(e ? -1 : +1)
```

48 个字节、48 个不同的值，覆盖 ±0.5 .. ±3.75。**这个推导对着 ST 的解码器逐码验过**
（2026-09-26，48/48 相符），不是从格式文档猜的。

代价要说清楚：这段栅格里相邻值的相对间隔只在 0.0667 .. 0.1250 之间变化，而完整的 HF8
栅格是 1/9 .. 1.0（ST 在 `_hif8_rank_table` 的注释里记了这个范围——靠近下端时相邻两个码
直接翻倍）。所以**「一个固定的 rtol 不能表示『差一格』」这条结论在整张表上才成立**，
本样例只覆盖其中一段。要在 HF8 **输出**上做验收，用 ST 那套「比编码、按格数打分」
（`_assert_hif8_codes_close`），别用相对误差。
"""

import numpy as np


def _build_grid():
    """返回 (按值排好序的值, 对应的字节)。"""
    codes, values = [], []
    for sign_bit in (0, 1):
        sign = -1.0 if sign_bit else 1.0
        for m in range(8):
            mantissa = 1.0 + m / 8
            codes.append(sign_bit * 128 + 8 + m)
            values.append(mantissa * 1.0 * sign)
            codes.append(sign_bit * 128 + 16 + m)  # e=0 -> ×2
            values.append(mantissa * 2.0 * sign)
            codes.append(sign_bit * 128 + 24 + m)  # e=1 -> ×0.5
            values.append(mantissa * 0.5 * sign)
    order = np.argsort(values)
    return np.asarray(values, np.float32)[order], np.asarray(codes, np.uint8)[order]


GRID_VALUES, GRID_CODES = _build_grid()
GRID_MAX = float(GRID_VALUES[-1])


def snap_to_grid(x):
    """把任意 fp32 吸附到这段栅格，返回 `(字节, 精确值)`。

    **精确值就是 Cube 真正吃到的东西**，参考实现必须用它，不是用吸附前的原始 fp32。
    栅格是排好序的，所以「最近的一个」在两个相邻候选里选。
    """
    src = np.asarray(x, np.float32)
    flat = src.reshape(-1).astype(np.float64)
    upper = np.searchsorted(GRID_VALUES, flat).clip(0, GRID_VALUES.size - 1)
    lower = (upper - 1).clip(0)
    take_high = np.abs(GRID_VALUES[upper] - flat) < np.abs(flat - GRID_VALUES[lower])
    idx = np.where(take_high, upper, lower)
    return GRID_CODES[idx].reshape(src.shape), GRID_VALUES[idx].reshape(src.shape)


def matmul_reference(a_values, b_values):
    """按定义算，FP32 累加——和 kernel 的累加器同精度。"""
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(a_values)).float()
        @ torch.from_numpy(np.ascontiguousarray(b_values)).float()
    )
