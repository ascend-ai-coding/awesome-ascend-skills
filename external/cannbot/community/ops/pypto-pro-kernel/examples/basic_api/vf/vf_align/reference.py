# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""各分布模式该产出什么。全部是整数搬运，没有算术，所以一律逐位比。

**每条都是实测反推的，不是从文档推的。** 文档说得对，但「FP32 下哪些可用」它没讲，而位宽用错是
静默错。下面注释里的「实测」指 2026-09-27 在 Ascend950PR 上的观察。
"""

import numpy as np

VL_FP32 = 64
SENTINEL = -1.0


def make_input(n):
    """`1..n`，每个值一眼能看出来自哪个下标。"""
    return np.arange(1, n + 1, dtype=np.float32).reshape(1, n)


def load_norm(x, n):
    """逐元素：前 VL 个原样，其余是目的地原值（这里测试只比前 VL 个）。"""
    out = np.zeros((1, n), dtype=np.float32)
    out[0, :VL_FP32] = x[0, :VL_FP32]
    return out


def load_brc(x, n):
    """广播：lane 0 铺满整个寄存器。"""
    out = np.zeros((1, n), dtype=np.float32)
    out[0, :VL_FP32] = x[0, 0]
    return out


def load_unpk(x, n):
    """解包：读 VL/2 个元素，零填展开成 VL 个 —— `1,0,2,0,3,0,...`。"""
    out = np.zeros((1, n), dtype=np.float32)
    out[0, 0:VL_FP32:2] = x[0, : VL_FP32 // 2]
    return out


def load_blk(x, n):
    """块复制：一个 32B DataBlock（FP32 下 8 个元素）铺满整个寄存器。"""
    out = np.zeros((1, n), dtype=np.float32)
    block = x[0, :8]
    out[0, :VL_FP32] = np.tile(block, VL_FP32 // 8)
    return out


def load_dintlv(x, n):
    """双搬入：读 2*VL，偶数下标进一个寄存器、奇数进另一个。"""
    even = np.zeros((1, n), dtype=np.float32)
    odd = np.zeros((1, n), dtype=np.float32)
    even[0, slice(None, VL_FP32)] = x[0, slice(0, VL_FP32 * 2, 2)]
    odd[0, slice(None, VL_FP32)] = x[0, slice(1, VL_FP32 * 2, 2)]
    return even, odd


def store_norm(x, n, valid=32, sentinel=SENTINEL):
    """按 preg 写：选中的写入，未选中的**保留 Tile 原值**。"""
    out = np.full((1, n), sentinel, dtype=np.float32)
    out[0, :valid] = x[0, :valid]
    return out


def store_first_element(x, n, sentinel=SENTINEL):
    """只写 lane 0。"""
    out = np.full((1, n), sentinel, dtype=np.float32)
    out[0, 0] = x[0, 0]
    return out


def store_intlv(x, n):
    """双搬出：`[a0, b0, a1, b1, ...]`，a 是前 VL 个、b 是后 VL 个。"""
    a = x[0, :VL_FP32]
    b = x[0, slice(VL_FP32, VL_FP32 * 2)]
    out = np.empty((1, n), dtype=np.float32)
    out[0, 0::2] = a
    out[0, 1::2] = b
    return out


def load_datablock(x, n, block_stride=2, block_elems=8):
    """非连续：输出第 k 个 block 来自源第 `k * block_stride` 个 block，一条指令 8 个 block。"""
    out = np.zeros((1, n), dtype=np.float32)
    for k in range(VL_FP32 // block_elems):
        src_start = k * block_stride * block_elems
        out[0, slice(k * block_elems, (k + 1) * block_elems)] = x[
            0, slice(src_start, src_start + block_elems)
        ]
    return out


def load_ds_on_fp32(n):
    """**陷阱**：`LoadDist.DS` 用在 32 位上不报错，给全零。

    枚举里只有 `DS` / `DS_B8` / `DS_B16`，没有 `DS_B32`；`US` 同理。这个参考记录的是**当前的
    错误行为**，不是正确语义——它存在是为了让行为一旦改变（改成报错或改成可用）测试立刻失败。
    """
    return np.zeros((1, n), dtype=np.float32)
