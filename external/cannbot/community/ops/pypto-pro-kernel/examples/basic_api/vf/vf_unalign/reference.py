# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三条链的参考。

**squeeze 那条链的语义是确定的**（与全仓唯一用它的 ST 一致）：两轮筛选的结果在输出里
**紧密排布、按顺序追加**，中间没有空洞。这条参考可以直接写。

**load / store 两条链的推进语义在写这份参考时还没实测**——`stride` 是「本次读写的长度」还是
「下次的起点偏移」，文档没写清。所以这两条的测试先断言**关系**（哪些位置必须搬到、
两次写的内容必须不同），实测之后再把确切值写进来。这样第一次上板的失败信息就是量出来的结果,
而不是白跑一轮。

判据 `rtol=0, atol=0`：搬运是逐字节的，操作数是 1..64 的整数，没有浮点误差。
"""

import numpy as np

CHUNK = 16
CHUNK_BYTES = CHUNK * 4
HI_THRESHOLD = 48.0
LO_THRESHOLD = 16.0


def make_operands(n):
    """`1..n`。整数、各不相同——搬到哪个位置一眼能认出来。"""
    return np.arange(1, n + 1, dtype=np.float32).reshape(1, n)


def squeeze_chain(a, hi=HI_THRESHOLD, lo=LO_THRESHOLD):
    """两轮筛选紧密追加：先 `> hi` 的，再 `<= lo` 的。

    输入 1..64、hi=48、lo=16 时：
    - 第一轮命中 49..64（16 个）
    - 第二轮命中 1..16（16 个）
    - 输出前 32 个是 `49..64, 1..16`，其余保持目标 Tile 的原值（这里初始化成 0）
    """
    n = a.shape[1]
    flat = a.reshape(-1)
    first = flat[flat > hi]
    second = flat[flat <= lo]
    out = np.zeros(n, dtype=np.float32)
    packed = np.concatenate([first, second])
    out[: len(packed)] = packed
    return out.reshape(1, n), len(first), len(second)
