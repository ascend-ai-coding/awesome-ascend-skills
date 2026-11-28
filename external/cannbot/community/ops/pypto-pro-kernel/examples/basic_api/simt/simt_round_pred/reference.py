# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""取整、谓词、整数算子的 CPU 参考，以及输入怎么挑。

**所有参考函数都返回一维数组**（内部 `ravel()`），输入是 `[1, THREADS]` 还是 `[THREADS]` 都一样。
统一形状是为了让判据里不出现 `[0]` 这种下标——那种下标错了不报错，只是比错了东西。

**为什么可以 `rtol=0, atol=0` 全程比。** 这一例一个近似算子都没有：取整、位计数、
高半乘、位重解释都是精确定义的整数／离散运算，`fmod` 也是精确的（商向零截断之后做一次减法）。
"""

import numpy as np

THREADS = 32
FMOD_DIVISOR = 2.0

# 前 16 个是**设计过的**边界：五种取整的区别全在 `.5` 和负数上。
# 后 16 个是普通值，用来确认普通情况没被特殊处理带坏。
BOUNDARY = [
    -2.5,
    -1.5,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.5,
    2.5,
    3.5,
    -1.7,
    -1.2,
    1.2,
    1.7,
    np.nan,
    np.inf,
    -np.inf,
]

# popcount 的输入：0 位、1 位、连续位、单独的高位、全 1。
POPCOUNT_INPUTS = [
    0,
    1,
    2,
    3,
    7,
    8,
    15,
    255,
    256,
    0x8000,
    0xFFFF,
    0x10000,
    0x80000000,
    0xFFFFFFFF,
    0xAAAAAAAA,
    0x55555555,
]

# mul_hi 的输入：平方之后必须溢出 32 位才验得到"高半"。
# 65536^2 = 2^32 正好溢出一位；0x40000000^2 = 2^60。**含负数**：有符号的高半和无符号不同。
MUL_HI_INPUTS = [
    0,
    1,
    3,
    256,
    65536,
    65537,
    0x10000000,
    0x40000000,
    -1,
    -3,
    -256,
    -65536,
    -65537,
    0x7FFFFFFF,
    -0x7FFFFFFF,
    123456789,
]


def make_inputs(threads=THREADS):
    """三个输入行。后半段用普通值填满，**不留 0**（0 会让好几个算子的结果重合）。"""
    f = np.zeros(threads, dtype=np.float32)
    f[: len(BOUNDARY)] = BOUNDARY
    f[slice(len(BOUNDARY), None)] = np.linspace(
        0.125, 6.125, threads - len(BOUNDARY), dtype=np.float32
    )

    u = np.zeros(threads, dtype=np.uint32)
    u[: len(POPCOUNT_INPUTS)] = POPCOUNT_INPUTS
    u[slice(len(POPCOUNT_INPUTS), None)] = np.arange(
        1, threads - len(POPCOUNT_INPUTS) + 1, dtype=np.uint32
    )

    i = np.zeros(threads, dtype=np.int32)
    i[: len(MUL_HI_INPUTS)] = MUL_HI_INPUTS
    i[slice(len(MUL_HI_INPUTS), None)] = np.arange(
        1, threads - len(MUL_HI_INPUTS) + 1, dtype=np.int32
    )

    return f.reshape(1, threads), u.reshape(1, threads), i.reshape(1, threads)


def finite_mask(f):
    """哪些位置是有限值。`cast(nan)` / `cast(inf)` 的结果没有定义，**那几列不进 cast 的判据**。"""
    return np.isfinite(f.ravel())


def rint_half_to_even(f):
    """`rint`：`.5` 就近取偶（numpy 的 `rint` 就是这个语义）。"""
    return np.rint(f.astype(np.float64).ravel()).astype(np.float32)


def round_half_away(f):
    """`round`：`.5` 远离零。**numpy 的 `np.round` 不是这个语义**（它就近取偶），所以自己算。"""
    x = f.astype(np.float64).ravel()
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.float32)


def popcount(u):
    return np.array([bin(int(v)).count("1") for v in u.ravel()], dtype=np.int32)


def mul_hi_i32(i):
    """有符号 32×32 乘法的高 32 位。"""
    prod = i.ravel().astype(np.int64) * i.ravel().astype(np.int64)
    return (prod >> 32).astype(np.int32)


def bitcast_to_i32(f):
    return f.ravel().view(np.int32)


def fmod_trunc(f, divisor=FMOD_DIVISOR):
    """商向零截断的余数，符号随被除数。`np.fmod` 正是这个语义（`np.mod` 不是）。

    `errstate` 是必要的：输入里有 `nan` 和 `±inf`，numpy 会为它们发
    `RuntimeWarning: invalid value encountered in fmod`。**这不是问题**（`fmod(nan, 2)` 就该是
    `nan`，硬件也给 `nan`），但一条留在测试输出里的 warning 会变成长期噪声。
    压掉的是**已知且期望的**那一类，不是"把警告关掉"。
    """
    with np.errstate(invalid="ignore"):
        return np.fmod(f.astype(np.float64).ravel(), divisor).astype(np.float32)
