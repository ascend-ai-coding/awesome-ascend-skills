# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考实现与操作数构造。

判据一律 `rtol=0, atol=0`：直方图是计数，`mem_bar` 那一条的算术是 `a + a` 与 `+a`，
输入取整数，全程精确。
"""

import numpy as np

N_BINS = 128


_BASE = np.array([0, 32, 128, 192], dtype=np.int64)


def make_u8_data(n=256):
    """周期 4 循环 `{0, 32, 128, 192}`，每个值出现 `n/4 = 64` 次。

    这组数换过一次——**第一版 `(i % 8) * 32` 让 BIN0 和 BIN1 给出完全相同的直方图**，
    防空测把它抓住了。值 `v` 和 `v+128` 同时存在时两个半区的桶号分布一模一样，
    于是 `bin_type` 有没有起作用根本看不出来。

    现在这组同时满足四条：

    - **两个半区的直方图形状不同**：`BIN0` 命中桶 `{0, 32}`，`BIN1` 命中桶 `{0, 64}`。
    - **高低值在寄存器里交错**（周期 4），所以「按值分桶」和「按 lane 分半区」两种读法
      给出不同结果——官方文档那句「统计 src 中 index 为 [0-127] 范围内」有歧义，
      这组数能把它判开。
    - **计数不是 1**：每个值 64 次。全 1 的分布下「计数」和「出现过」看不出区别。
    - **桶号好算**：值 `v` 在 `BIN0` 落 `dst[v]`，在 `BIN1` 落 `dst[v-128]`。
    """
    return _BASE[np.arange(n, dtype=np.int64) % 4].astype(np.uint8).reshape(1, n)


def frequency(data, bin_type=0):
    """`dst[n]` = 值 `n + 128*bin_type` 出现几次。返回 int16（host 侧的载体）。"""
    flat = data.reshape(-1).astype(np.int64)
    lo = 128 * bin_type
    out = np.zeros(N_BINS, dtype=np.int16)
    for v in flat:
        if lo <= v < lo + N_BINS:
            out[v - lo] += 1
    return out.reshape(1, N_BINS)


def accumulate(data, bin_type=0):
    """`dst[n] = |{v : v <= n + 128*bin_type}|`——**累计是对整个 0..255 值域做的**。

    `bin_type` 只决定你看哪 128 个桶，不缩小累计的范围。实测（值 `{0,32,128,192}`
    各 64 次）：

    | | `dst[0]` | `dst[64]` |
    | --- | --- | --- |
    | `BIN0` | 64（值 ≤ 0） | 128（值 ≤ 64） |
    | `BIN1` | **192**（值 ≤ 128） | **256**（值 ≤ 192） |

    换句话说 **`BIN1` 的累计从 `BIN0` 结束的地方接着往上走**，两半拼成一条 0..255 的
    连续累计分布。这正是 `lightning_indexer` 用它在整个字节域上找分位阈值的原因。

    **第一版参考写成 `cumsum(frequency(data, bin_type))` 是错的**，只在 `BIN0` 下凑巧
    相等（那一半里「值域内的前缀和」和「全局前缀和」一样）。官方文档那句
    「第 n 个元素表示 src 中从 0 到 n 的所有数值**在对应区间中**出现的总频率」
    读起来像限制在区间内——实测不是。
    """
    flat = data.reshape(-1).astype(np.int64)
    lo = 128 * bin_type
    return np.array(
        [(flat <= lo + n).sum() for n in range(N_BINS)], dtype=np.int16
    ).reshape(1, N_BINS)


def frequency_masked_below_128(data, bin_type=0):
    """只统计 `< 128` 的值。`BIN1` 那一半因此全 0。"""
    flat = data.reshape(-1).astype(np.int64)
    return frequency(flat[flat < 128].reshape(1, -1), bin_type)


def count_below_128(data):
    """输入里 `< 128` 的元素个数——掩码那一条的总计数期望值。"""
    return int((data.reshape(-1).astype(np.int64) < 128).sum())


def make_fp32_data(n):
    """`1.0 .. n`。值 = 1-based 下标，`mem_bar` 与 `clear_spr` 两条共用。"""
    return (np.arange(n, dtype=np.float32) + 1.0).reshape(1, n)


def triple(a):
    """`mem_bar_raw` 的期望结果：先写 `2a`，读回来再加 `a` = `3a`。"""
    return a * 3.0
