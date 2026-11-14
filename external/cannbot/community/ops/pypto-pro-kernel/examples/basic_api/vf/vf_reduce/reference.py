# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四个归约的参考。**`reduce_max` / `reduce_min` 的 lane 1 装的是极值下标**，不是 0。

## lane 1 的下标是整数位模式

`reduce_max` 把最大值写 `dst[0]`、把**第一个**最大值的下标写 `dst[1]`——写的是整数，而寄存器
是 FP32，所以读出来是个非规格化的小浮点数（下标 63 读成 `8.83e-44`）。要当下标看就得按
位重解释：`tensor.view(torch.int32)`。

这是官方文档 `reduction/reduce_max.md` / `reduce_min.md` 的「返回值说明」写明的：

> 归约结果写入第一个元素dst[0]，索引写入dst[1]。
> - 当存在多个最大值时，将第一个最大值的索引保存在dst[1]中。
> - 如果输入数据存在nan，将该数据类型的nan写入dst[0]，并将第一个nan的索引保存在dst[1]中。

**而 `_vf_api.py` 的 docstring 写的是 "The remaining lanes are zeroed"，那句是错的。**
`reduce_sum` 没有下标，它的 docstring 是对的。

## 为什么这一例能 `rtol=0, atol=0`

不是因为「误差小」，也**不是**因为归约顺序碰巧对上——官方文档 `reduce_sum.md` 写明：

> 指令内累加顺序采用二叉树累加方式，结果具有确定性。

顺序是**定的**。所以即使换成非整数输入，只要参考也按二叉树两两合并，仍然可以逐位比；
用 `numpy.sum` 的顺序累加才会差最后一位。本例的操作数是 1..64（全整数、和 ≤ 2080 < 2^24），
两种顺序都精确，所以直接用 `numpy` 求和即可——**但这是操作数的性质，不是通用许可**。
"""

import numpy as np

DATABLOCK_LANES = 8


def make_operands(n):
    """1..n。全正、全整数、和远小于 2^24，任何累加顺序都精确。"""
    return np.arange(1, n + 1, dtype=np.float32).reshape(1, n)


def make_tie_operands(n):
    """最大值出现两次（下标 10 和 20），最小值出现两次（下标 5 和 30）。

    用来验「多个极值时取第一个的下标」。
    """
    a = np.arange(1, n + 1, dtype=np.float32)
    a[10] = 1000.0
    a[20] = 1000.0
    a[5] = -7.0
    a[30] = -7.0
    return a.reshape(1, n)


def make_nan_operands(n):
    """下标 12 和 25 是 NaN。用来验「有 NaN 时 dst[0] 是 NaN、dst[1] 是第一个 NaN 的下标」。"""
    a = np.arange(1, n + 1, dtype=np.float32)
    a[12] = np.nan
    a[25] = np.nan
    return a.reshape(1, n)


def index_as_float(index):
    """把整数下标按位重解释成 FP32——这就是设备写进 lane 1 的那个值。"""
    return np.array([index], dtype=np.int32).view(np.float32)[0]


def _lane0_only(value, n):
    out = np.zeros((1, n), dtype=np.float32)
    out[0, 0] = np.float32(value)
    return out


def _value_and_index(value, index, n):
    """lane 0 = 极值，lane 1 = 下标的位模式，其余为 0。"""
    out = np.zeros((1, n), dtype=np.float32)
    out[0, 0] = np.float32(value)
    out[0, 1] = index_as_float(index)
    return out


def whole(a):
    """整寄存器归约。顺序与 kernel 的输出顺序一致：sum / max / min / pair。

    `sum` 只有 lane 0；`max` / `min` 有 lane 0 和 lane 1。
    """
    n = a.shape[1]
    flat = a.reshape(-1)
    a64 = flat.astype(np.float64)

    pair = np.zeros((1, n), dtype=np.float32)
    halves = (a64[0::2] + a64[1::2]).astype(np.float32)
    pair[0, : len(halves)] = halves

    return [
        _lane0_only(a64.sum(), n),
        _value_and_index(flat.max(), int(flat.argmax()), n),
        _value_and_index(flat.min(), int(flat.argmin()), n),
        pair,
    ]


def by_block(a, lanes=DATABLOCK_LANES):
    """组内归约：每 `lanes` 个元素一组，结果压在 lane 0..(n/lanes - 1)，其余为 0。

    **组内归约没有下标**——实测 `datablock=True` 的 lane 1 是第二组的极值，不是下标。
    所以这三个输出的形状和 `reduce_sum` 一样干净。
    """
    n = a.shape[1]
    a64 = a.astype(np.float64).reshape(-1)
    groups = a64.reshape(-1, lanes)
    out = []
    for values in (groups.sum(axis=1), groups.max(axis=1), groups.min(axis=1)):
        buf = np.zeros((1, n), dtype=np.float32)
        buf[0, : len(values)] = values.astype(np.float32)
        out.append(buf)
    return out
