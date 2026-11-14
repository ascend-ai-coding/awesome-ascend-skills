# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""十个谓词 API 的参考。比较结果按 0/1 表示，全部 `rtol=0, atol=0`。

**操作数是挑过的**，为了让六个比较算子的结果**两两不同**：

- `a = 1..64`，`b = 32` 全体。于是 `eq` 只有 1 个真、`ne` 有 63 个、`lt` 有 31 个、
  `gt` 有 32 个、`le` 有 32 个、`ge` 有 33 个——六个计数各不相同，任何两个算子搞混都会被抓到。
- 阈值 `LO=16` / `HI=48` / 标量 `32` 都是 2 的幂或整数，比较本身没有浮点歧义。

`select(ones, zeros, mask)` 把掩码变成 1.0 / 0.0，所以参考就是布尔数组转 float32。
"""

import numpy as np

LO, HI, SCALAR = 16.0, 48.0, 32.0


def make_operands(n):
    """`a = 1..n`，`b` 全是 32。"""
    a = np.arange(1, n + 1, dtype=np.float32).reshape(1, n)
    b = np.full((1, n), SCALAR, dtype=np.float32)
    return a, b


def _as01(flags):
    return flags.astype(np.float32)


def compare(a, b):
    """六个比较，顺序与 kernel 的输出顺序一致：eq / ne / lt / gt / le / ge。"""
    return [
        _as01(a == b),
        _as01(a != b),
        _as01(a < b),
        _as01(a > b),
        _as01(a <= b),
        _as01(a >= b),
    ]


def expected_true_counts(a, b):
    """六个算子各自的真值个数——用来断言它们两两不同，防空测。"""
    return [int(f.sum()) for f in compare(a, b)]


def combine(a):
    """链式取交集 / 换源取反 / 标量比较 / 数据按位取反。

    - `band` = `(a > LO) and (a < HI)`——kernel 侧是把第一个掩码当第二个比较的 preg
    - `inverted` = `not band`——kernel 侧是 `select` 换源
    - `scalar` = `a >= SCALAR`
    - `not_data` 是 **FP32 位型按位取反**，所以参考要在 int32 视图上算
    """
    band = (a > LO) & (a < HI)
    scalar = a >= SCALAR
    not_data = (~a.view(np.int32)).view(np.float32)
    return [_as01(band), _as01(~band), _as01(scalar), not_data]


def mask_from_spr_expected(n, mask_low=0xFFFFFFFF):
    """`pl.set_vec_mask(0, mask_low)` 之后，第 i 位控制第 i 个 lane。"""
    bits = np.array([(mask_low >> i) & 1 for i in range(n)], dtype=np.float32)
    return bits.reshape(1, n)


def make_bit_pattern(n):
    """造一个位型：前 4 个 int32 lane 分别给不同的模式，其余置零。

    FP32 下 `mask_gen_with_reg_tensor` 的 DataBlock 是 **8 字节 = 2 个 lane**：
    `offset=0` 取 lane 0..1 的 64 位，`offset=1` 取 lane 2..3 的 64 位。
    """
    raw = np.zeros(n, dtype=np.int32)
    raw[0] = 0x55555555  # 0101...
    raw[1] = 0x00000000
    raw[2] = 0x000000FF  # 低 8 位全 1
    raw[3] = 0x00000000
    return raw.view(np.float32).reshape(1, n), raw


def mask_from_bits_expected(raw, n, offset):
    """第 `offset` 个 8 字节块的 64 位，低位在前，每位控制一个 lane。"""
    lo = int(np.uint32(raw[offset * 2]))
    hi = int(np.uint32(raw[offset * 2 + 1]))
    value = lo | (hi << 32)
    bits = np.array([(value >> i) & 1 for i in range(n)], dtype=np.float32)
    return bits.reshape(1, n)
