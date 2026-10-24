# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""十个算子的参考。全部 `rtol=0, atol=0`——重排是搬位置，位宽重组是搬位，转换用挑过的值。

**操作数是挑过的**：

- 重排组用 `a = 1..64`、`b = 101..164`——两组值域不重叠，交织/解交织后每个位置来自哪个源
  一眼能认出来。用同一组值就分不清 `dst0` 是 a 的偶位还是 b 的偶位。
- 位宽组用 `a[i] = 0x00020003 + i * 0x00010001`（int32）——高半字和低半字都不为零、且各不相同,
  所以 `pack` 取低 16 位、`unpack` 选哪一半，都能从结果读出来。
- 转换组用 `1.5, 2.5, 3.5, ...`（半整数）——`astype` 与 `truncate` 的取整行为在这些值上
  才有区别可看；整数输入的话两者看起来一样。
"""

import numpy as np

HALF = 32


def make_shuffle_operands(n):
    """`a = 1..n`，`b = 101..100+n`。值域不重叠，便于认出来源。"""
    a = np.arange(1, n + 1, dtype=np.float32).reshape(1, n)
    b = (np.arange(1, n + 1) + 100).astype(np.float32).reshape(1, n)
    return a, b


def interleave(a, b):
    """实测排布：`dst0 = [a0, b0, a1, b1, ...]`（前 n 个），`dst1` 接着后 n 个。

    交织后总长 2n，拆成两个长度 n 的寄存器。
    """
    n = a.shape[1]
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    woven = np.empty(2 * n, dtype=np.float32)
    woven[0::2] = flat_a
    woven[1::2] = flat_b
    return woven[:n].reshape(1, n), woven[n:].reshape(1, n)


def de_interleave(a, b):
    """实测排布：把 `(a, b)` 当成一条 2n 的流，`dst0` 取偶位、`dst1` 取奇位。"""
    n = a.shape[1]
    stream = np.concatenate([a.reshape(-1), b.reshape(-1)])
    return stream[0::2][:n].reshape(1, n), stream[1::2][:n].reshape(1, n)


def squeeze(a, threshold=32.0):
    """有效元素依次搬到低端，其余置零。"""
    n = a.shape[1]
    flat = a.reshape(-1)
    picked = flat[flat > threshold]
    out = np.zeros(n, dtype=np.float32)
    out[: len(picked)] = picked
    return out.reshape(1, n)


def unsqueeze(a, threshold=32.0):
    """掩码的**前缀和**：每个 lane 的值 = 它前面有多少个有效 lane。

    实测确认的是官方文档的**正文**（「dst 首位为 0，mask[i] 为 1 时 dst[i] = dst[i-1] + 1」），
    **不是**同一页上的公式（那个公式写成 0/1，与正文矛盾）。
    """
    n = a.shape[1]
    active = (a.reshape(-1) > threshold).astype(np.int32)
    out = np.zeros(n, dtype=np.int32)
    running = 0
    for i in range(n):
        out[i] = running
        running += int(active[i])
    return out.reshape(1, n)


def make_regroup_operands(n):
    """高半字与低半字都不为零且各不相同，这样 pack/unpack 的选择能被读出来。"""
    raw = (np.arange(n, dtype=np.int64) * 0x00010001 + 0x00020003).astype(np.int32)
    return raw.reshape(1, n)


def pack(a, part, n=None):
    """取每个 int32 的**低 16 位**，依次塞进 dst 的低半或高半，其余为 0。

    读回来按 int32 看：相邻两个 uint16 拼成一个 int32（小端，低位在前）。
    """
    n = n or a.shape[1]
    lows = (a.reshape(-1) & 0xFFFF).astype(np.uint32)
    packed = np.zeros(n, dtype=np.uint32)
    pairs = lows[0::2] | (lows[1::2] << 16)  # n/2 个 int32
    if part == "LOWER":
        packed[: len(pairs)] = pairs
    else:
        packed[slice(n - len(pairs), None)] = pairs
    return packed.astype(np.int32).reshape(1, n)


def unpack(a, part, n=None):
    """取 src 的低半或高半（各 n/2 个 int32），每个符号扩展成 int64。

    读回来按 int32 看：每个 int64 占两个 int32 槽位，低位在前。正数的高位是 0。
    """
    n = n or a.shape[1]
    flat = a.reshape(-1)
    chosen = flat[: n // 2] if part == "LOWER" else flat[slice(n // 2, None)]
    wide = chosen.astype(np.int64)
    return wide.view(np.int32)[:n].reshape(1, n)


def make_convert_operands(n):
    """半整数：`1.5, 2.5, ...`。整数输入下 astype 与 truncate 看不出区别。"""
    return (np.arange(n, dtype=np.float32) + 1.5).reshape(1, n)


def astype_to_int32(a):
    """`vcvt` 的默认取整：`VFRoundMode.CAST_RINT`——**就近、平局取偶**。

    实测 `1.5 2.5 3.5 4.5 → 2 2 4 4`。`np.rint` 用的是同一条规则，所以直接拿它当参考。
    **不是向零取整**，和 `truncate` 不同——写这一版时我先按截断写，被这条测出来了。
    """
    return np.rint(a.astype(np.float64)).astype(np.int32)


def astype_toward_zero_to_int32(a):
    """`round_mode=VFRoundMode.CAST_TRUNC` 的结果：向零取整，和 `truncate` 同规则。"""
    return np.trunc(a.astype(np.float64)).astype(np.int32)


def bit_cast_to_int32(a):
    """位型原样重解释。`1.5` 是 `0x3FC00000`。"""
    return a.view(np.int32)


def truncate(a):
    """向零取整，**结果仍是 FP32**。"""
    return np.trunc(a.astype(np.float64)).astype(np.float32)


def muls_cast(a, scalar=4.0):
    """先乘标量再转 **FP16**（`dtype` 唯一合法值），样例再 `astype` 回 FP32 落盘。

    `1.5 * 4 = 6.0`。选 2 的幂做标量、半整数做输入，乘积全是 ≤ 258 的整数——FP16 能
    精确表示到 2048，所以来回一趟不丢位，可以 `atol=0` 比。
    """
    return (a.astype(np.float64) * scalar).astype(np.float16).astype(np.float32)
