# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""视图读出来该是原张量的哪些元素。

**为什么可以 `rtol=0, atol=0` 比。** 视图不做算术，它只改「读哪些字节、怎么解释」。
所以每个参考都是 numpy 的一次切片或一次 `view()`——**两边做的是同一件事**，容差没有意义。

参考值刻意写成 `x[0::2]` 这种切片形式而不是逐元素构造：**切片本身就是"步长"这个概念**，
读参考的人一眼能看出被验证的是什么。
"""

import numpy as np

SENTINEL = np.float32(-999.0)


def indexed_rows(raw, indices):
    """Select complete rows of 16-bit storage, independently of word grouping."""
    return raw[np.arange(raw.shape[0])[:, None], indices]


def make_input(rows, cols):
    """`arange + 1`：元素的值就是它的下标 +1，读错行一眼能看出错了几行。不含 0。"""
    return (np.arange(rows * cols, dtype=np.float32) + 1.0).reshape(rows, cols)


def every_other_row(x, start, count):
    """从 `start` 行起、每隔两行取 `count` 行。步长 `2 * COLS` 的视图读到的就是它。"""
    return x[start::2][:count]


def as_int32(x):
    """同一批字节按 INT32 解释。`view` 而不是 `astype`——**前者不改字节，后者改**。

    这个区别是判据的全部：`astype` 会把 `1.0` 变成 `1`，`view` 会把它变成 `1065353216`
    （FP32 的位模式）。视图的 `dtype=` 做的是后者。
    """
    return x.view(np.int32)


def overlapped(x, rows, cols, overlap_rows, overlap_start, sentinel=SENTINEL):
    """两个视图都写过之后，**后写的赢**。只在两条 store 之间排过序时才成立。

    不排序的版本重叠区不确定（实测 20 次里 14 次 B、6 次 A），所以那一条判据不用这个函数，
    用下面的 `overlapped_certain_parts`。

    没被任何视图碰到的尾部保持哨兵——**哨兵是判"B 没写过头"的唯一手段**。
    """
    out = np.full((rows, cols), sentinel, dtype=np.float32)
    out[0:overlap_rows] = x[0:overlap_rows]
    out[slice(overlap_start, overlap_start + overlap_rows)] = x[
        slice(overlap_rows, overlap_rows * 2)
    ]
    return out


def overlapped_certain_parts(x, overlap_rows, overlap_start):
    """不排序时**确定**的那两段，加尾部的哨兵段由调用方自己断言。

    视图 B 把 `x[overlap_rows + k]` 写到 GM 第 `overlap_start + k` 行（`k` 从 0 起），所以：

    | GM 行 | 谁写的 | 内容 |
    | --- | --- | --- |
    | `0 .. overlap_start-1` | 只有 A | `x[0 : overlap_start]` |
    | `overlap_start .. overlap_rows-1` | **两个都写**（不确定） | 不进判据 |
    | `overlap_rows .. overlap_start+overlap_rows-1` | 只有 B | `x[2*overlap_rows-overlap_start : 2*overlap_rows]` |

    返回 `[(GM 行切片, 期望值, 说明), ...]`，**重叠区不在里面**——它的值取决于两条 store 落地的
    先后，写进判据会得到一条偶发失败的测试。
    """
    a_only = slice(0, overlap_start)
    b_only = slice(overlap_rows, overlap_start + overlap_rows)
    return [
        (a_only, x[0:overlap_start], "只有视图 A 覆盖"),
        (
            b_only,
            x[slice(2 * overlap_rows - overlap_start, 2 * overlap_rows)],
            "只有视图 B 覆盖",
        ),
    ]
