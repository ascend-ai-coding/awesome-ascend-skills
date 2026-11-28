# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考实现与操作数构造。

判据一律 `rtol=0, atol=0`：gather / scatter / arange 都是整数下标搬运，
`muls` 的标量是整数、输入是整数，没有一步引入浮点误差。

操作数的选法：

- 数据是 `1.0 .. 64.0`（或 `1..128`），**元素值等于它的 1-based 下标**。
  这样输出里每个数直接告诉你它来自哪——调试时不用回头对索引表。
- 索引一律是**倒序**或**明显的置换**，不用恒等映射：恒等映射下
  「索引没被用上」和「索引对了」给出同一个结果，验不到东西。
"""

import numpy as np


def make_data(n):
    """`1.0 .. n`。值 = 1-based 下标，输出自带溯源信息。"""
    return (np.arange(n, dtype=np.float32) + 1.0).reshape(1, n)


def make_reverse_index(n):
    """倒序置换：`[n-1, n-2, ..., 0]`。逐元素 gather / scatter 共用。"""
    return (np.arange(n - 1, -1, -1, dtype=np.uint32)).reshape(1, n)


def make_block_index(n, block_bytes=32, itemsize=4):
    """DATA_BLOCK_LOAD 的索引：**单位是字节**，把 8 个 32B 块整体倒序。

    有意义的只有前 `n * itemsize // block_bytes` 个 lane，其余置 0
    （0 号块一定在范围内，不会越界）。
    """
    blocks = n * itemsize // block_bytes
    idx = np.zeros(n, dtype=np.uint32)
    idx[:blocks] = np.arange(blocks - 1, -1, -1, dtype=np.uint32) * block_bytes
    return idx.reshape(1, n)


def make_duplicate_index(n):
    """重复索引：全取 0 号元素之外，前 8 个 lane 轮流取 0 / 1。

    `gather` 下这是广播，有定义；`scatter` 下重复值「仅保留其中一个」，无定义。
    """
    idx = np.zeros(n, dtype=np.uint32)
    idx[:8] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint32)
    return idx.reshape(1, n)


def make_out_of_range_index(n):
    """越界索引：`[n, n+1, ..., 2n-1]`，reg→reg 形式下应当回卷成 `[0, 1, ..., n-1]`。"""
    return (np.arange(n, 2 * n, dtype=np.uint32)).reshape(1, n)


def gather(data, index):
    """`dst[i] = src[index[i]]`。两种形式、重复索引、回卷都用这一个式子。"""
    flat = data.reshape(-1)
    return flat[index.reshape(-1) % flat.size].reshape(data.shape)


def gather_datablock(data, index, block_bytes=32):
    """按 32B 块 gather：`dst` 的第 j 块 = `src` 中字节偏移 `index[j]` 起的那一块。"""
    per_block = block_bytes // data.dtype.itemsize
    blocks = data.size // per_block
    flat = data.reshape(-1)
    out = np.empty_like(flat)
    for j in range(blocks):
        start = int(index.reshape(-1)[j]) // data.dtype.itemsize
        out[slice(j * per_block, (j + 1) * per_block)] = flat[
            slice(start, start + per_block)
        ]
    return out.reshape(data.shape)


def scatter(data, index):
    """`dst[index[i]] = src[i]`。索引唯一时才有定义——本例只用置换索引调它。"""
    flat = data.reshape(-1)
    out = np.zeros_like(flat)
    out[index.reshape(-1)] = flat
    return out.reshape(data.shape)


def arange_inc(n, start=0):
    return (np.arange(n, dtype=np.int32) + start).reshape(1, n)


def arange_dec(n, start=100):
    """`start` 是**最后一个** lane 的值：`dst[i] = start + (n - 1 - i)`。

    实测（`start=100`、n=64）：`163, 162, ..., 100`。官方正文「以传入的 start 为最终值」
    是对的，`_vf_api.py` 的公式 `dst[i] = start - i` 是错的。见 kernel.py §四。
    """
    return (start + (n - 1) - np.arange(n, dtype=np.int32)).reshape(1, n)
