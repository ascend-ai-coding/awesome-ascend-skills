# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MXFP4 的两种 4 位格式：编码是**推出来的**，然后和官方 ST 的表核对。

FP4 一个元素只有 4 位，**两个逻辑元素挤在一个字节里**，所以操作数以 `uint8 [M, K//2]` 传进
kernel。这份文件负责三件事：把 16 个码的精确值算出来、按官方约定打包、给出一个整条和在 FP32 里
**精确可表示**的操作数集合（这样验收可以用 `rtol=0, atol=0`）。

**两种格式，同一个偏置.** 都是 `bias = 1`：

| 格式 | 位域 | 码 | 值 |
| --- | --- | --- | --- |
| `DT_FP4E2M1` | 1 符号 + 2 阶 + 1 尾 | `s*8 + e*2 + m` | `e=0`：`m/2`；`e>0`：`(1+m/2)·2^(e-1)` |
| `DT_FP4E1M2` | 1 符号 + 1 阶 + 2 尾 | `s*8 + e*4 + m` | `e=0`：`m/4`；`e=1`：`1+m/4` |

E2M1 四个阶码全部用于有限值，**没有 inf / nan**——4 位放不下那种奢侈。E1M2 同理。

推出来的两张表和官方 ST（`test_matmul_mx.py:885` 的 `_FP4_VALUES`）逐值相同，`test.py` 里有断言。
ST 另外给了两个单位码（`one_code = {"e2m1": 0x02, "e1m2": 0x04}`），也对得上：E2M1 的 `0x02` 是
`e=1,m=0` → 1.0，E1M2 的 `0x04` 同理。

**打包方向不能猜.** 官方 `_pack_fp4` 的原文是
`(codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)`——**偶数逻辑下标进低半字节**，沿
**最后一维**配对。写反了会得到一个转置过半字节的操作数，逐位比会立刻发现。

**为什么这组操作数能逐位验.** 取 E2M1 的 14 个非零码（值都是 0.5 的倍数、绝对值 ≤ 6）配
`{0.5, 1, 2}` 的 scale：

- 一个乘积项是 `0.25` 的倍数，绝对值 ≤ 36；配上两边的 scale 之后是 `2^-4` 的倍数、≤ 144
- K=128 的和 ≤ 18432，而 `18432 / 2^-4 = 294912 < 2^24`

所以**整条和在 FP32 里精确可表示**，参考在 float64 上算、和 kernel 的 FP32 累加顺序不同也应当逐
位相同。和 [`../fundamental_usage/matmul_mx/reference.py`](../fundamental_usage/matmul_mx/reference.py) 是同一个论证，只是换了阶梯。

scale 的打包直接复用那份文件，不重造。
"""

import numpy as np

from ..fundamental_usage.matmul_mx.reference import (
    GROUP_SIZE,
    E8M0_CHOICES,
    pack_scale_a,
    pack_scale_b,
)


def _e2m1_values():
    """1 符号 + 2 阶 + 1 尾，bias=1。16 个码按码序排。"""
    values = []
    for sign_bit in (0, 1):
        sign = -1.0 if sign_bit else 1.0
        for exponent in range(4):
            for mantissa in range(2):
                if exponent == 0:
                    magnitude = mantissa / 2  # 次正规
                else:
                    magnitude = (1.0 + mantissa / 2) * 2.0 ** (exponent - 1)
                values.append(sign * magnitude)
    return np.asarray(values, dtype=np.float32)


def _e1m2_values():
    """1 符号 + 1 阶 + 2 尾，bias=1。"""
    values = []
    for sign_bit in (0, 1):
        sign = -1.0 if sign_bit else 1.0
        for exponent in range(2):
            for mantissa in range(4):
                if exponent == 0:
                    magnitude = mantissa / 4
                else:
                    magnitude = 1.0 + mantissa / 4
                values.append(sign * magnitude)
    return np.asarray(values, dtype=np.float32)


E2M1_VALUES = _e2m1_values()
E1M2_VALUES = _e1m2_values()

VALUES = {"e2m1": E2M1_VALUES, "e1m2": E1M2_VALUES}

# 非零码：0 和 -0 排除掉，这样每个乘积项都非零，错了更容易露出来。
NONZERO = {
    name: np.asarray([c for c in range(16) if table[c] != 0], dtype=np.uint8)
    for name, table in VALUES.items()
}
NONZERO_E2M1 = NONZERO["e2m1"]


def pack_fp4(codes):
    """逻辑码 `[..., L]` -> 存储字节 `[..., L//2]`。偶数下标进低半字节。

    方向照官方 `_pack_fp4`，不是自己定的。
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2:
        raise ValueError(f"FP4 打包要求最后一维是偶数，得到 {codes.shape[-1]}")
    low = codes[..., 0::2] & 0x0F
    high = (codes[..., 1::2] & 0x0F) << 4
    return np.ascontiguousarray((low | high).astype(np.uint8))


def unpack_fp4(packed, logical_shape):
    """`pack_fp4` 的逆，测试里用来自证打包方向。"""
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(tuple(logical_shape), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    return codes


def _codes_and_values(shape, seed, fmt):
    rng = np.random.default_rng(seed)
    pool = NONZERO[fmt]
    codes = pool[rng.integers(0, pool.size, size=shape)]
    return codes, VALUES[fmt][codes]


def _scales(shape, seed):
    rng = np.random.default_rng(seed)
    e8m0 = np.asarray(E8M0_CHOICES, np.uint8)[
        rng.integers(0, len(E8M0_CHOICES), size=shape)
    ]
    return e8m0, np.exp2(e8m0.astype(np.int16) - 127).astype(np.float32)


def make_fp4_operands(m, k, n, seed=0, a_format="e2m1", b_format="e2m1"):
    """喂给 kernel 的**打包字节**，和参考要用的精确值。

    两侧的格式可以不同（官方 ST 测了四种配对）。归约轴两侧都是 K：A 的 K 是最后一维，B 的 K 是第
    一维——而 FP4 的打包沿**存储的最后一维**，所以 A 沿 K 配对、B 沿 N 配对。打包方向和组轴是两件
    独立的事。

    混合配对 `e2m1 × e1m2` 的和同样精确可表示：乘积项是 `2^-5` 的倍数、绝对值 ≤ 42（`6 × 1.75`
    再配两边的 scale），K=64 的和 ≤ 2688，`2688 / 2^-5 = 86016 < 2^24`。
    """
    a_codes, a_mantissa = _codes_and_values((m, k), seed, a_format)
    b_codes, b_mantissa = _codes_and_values((k, n), seed + 1, b_format)
    a_e8m0, a_scale = _scales((m, k // GROUP_SIZE), seed + 2)
    b_e8m0, b_scale = _scales((k // GROUP_SIZE, n), seed + 3)

    a_values = a_mantissa * np.repeat(a_scale, GROUP_SIZE, axis=1)
    b_values = b_mantissa * np.repeat(b_scale, GROUP_SIZE, axis=0)
    return dict(
        a_packed=pack_fp4(a_codes),
        b_packed=pack_fp4(b_codes),
        scale_a=pack_scale_a(a_e8m0),
        scale_b=pack_scale_b(b_e8m0),
        a_codes=a_codes,
        b_codes=b_codes,
        a_values=a_values.astype(np.float32),
        b_values=b_values.astype(np.float32),
    )
