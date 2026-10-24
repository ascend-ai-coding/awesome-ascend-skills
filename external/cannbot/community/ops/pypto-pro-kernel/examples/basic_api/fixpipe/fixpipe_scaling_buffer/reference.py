# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考、每列的 scale，以及 Scaling Tile 的 host 侧编码。

编码是这一例的关键：Scaling Tile 的元素是 **INT64**，装的是位域而不是数值。
`encode_scales` 把它逐位写出来，不调 `torch_npu.npu_trans_quant_param`——
**能自己算的就自己算，这样每一位的含义都在代码里**。
"""

import struct

import numpy as np

#: bit46 = signed-int8 标志。官方 ST `test_scale_per_channel_advanced.py:32`
#: 里叫 `FP_SCALE_SIGNED_INT8_FLAG`。它的实际效果见 test.py §三（实测）。
SIGNED_INT8_FLAG = 1 << 46

#: INT8 目的的饱和区间。实测值，见 `fixpipe_quant_matrix/reference.py`。
INT8_RANGE = (-128, 127)


def encode_scales(scales, *, signed_int8_flag=True):
    """每个 scale 编成 `(1 << 46) | float32_bits(scale)`，返回 INT64 数组。

    `signed_int8_flag=False` 就只留低 32 位——用来量那一位到底管什么。
    """
    out = np.empty(len(scales), dtype=np.int64)
    for i, s in enumerate(scales):
        bits = struct.unpack("!I", struct.pack("!f", float(s)))[0]
        out[i] = (SIGNED_INT8_FLAG | bits) if signed_int8_flag else bits
    return out.reshape(1, -1)


def column_scales(n, *, exponents=(-2, -1, 0, 1, 2)):
    """每列一个 scale，取 2 的幂：`2 ** exponents[j % len(exponents)]`。

    **必须是 2 的幂**：这样 `scale * acc` 在 FP32/FP16 里逐位精确，判据能写
    `atol=0`。`exponents` 循环长度 5 与 N=64 互素，所以没有哪两列是
    「同一列的复制」——`test_the_columns_really_differ` 把这一点钉成测试。
    """
    return np.array(
        [2.0 ** exponents[j % len(exponents)] for j in range(n)], dtype=np.float64
    )


def make_operands(m, k, n, seed=0):
    """FP16 小整数，`a @ b` 里负数约占四成（INT8 那格靠负数验饱和的下界）。"""
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(m, k)).astype(np.float16)
    b = rng.integers(-2, 2, size=(k, n)).astype(np.float16)
    return a, b


def accumulate(a, b):
    """cube 的累加结果，用 float64 算。"""
    return a.astype(np.float64) @ b.astype(np.float64)


def scaled_per_column(acc, scales):
    """期望：第 j 列整列乘 `scales[j]`。"""
    return acc * scales[np.newaxis, :]


def quantized_to_int8(acc, scales):
    """INT8 的期望：逐列缩放 → 半偶取整 → 饱和。

    取整与饱和的规则抄自 `fixpipe_quant_matrix` 的实测结果，不是猜的。
    """
    lo, hi = INT8_RANGE
    return np.clip(np.rint(scaled_per_column(acc, scales)), lo, hi)
