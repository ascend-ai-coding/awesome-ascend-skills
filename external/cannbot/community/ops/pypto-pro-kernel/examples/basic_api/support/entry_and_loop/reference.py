# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""循环与流水的参考。不 import 编译器。

流水那一组的参考特别重要：**`preload` 换了结果必须一样**，所以参考只有一份,
四个 `preload` 都对它。参考自己不含任何流水的概念——这正是要验的点。
"""

from __future__ import annotations

import numpy as np

SENTINEL = np.float32(-999.0)
"""带步长那一组的哨兵。**没被循环碰到的块必须仍是哨兵**，
不填哨兵就只能证明「碰到的对」，证不到「没碰的没动」。"""


def make_input(rows: int, cols: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def make_input_fp16(rows: int, cols: int, seed: int = 5) -> np.ndarray:
    """流水那一组走 FP16（跨核 buffer 是 L1 的 ZN 布局，照 ST 用 FP16）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float16)


def scaled_by_block_index(
    src: np.ndarray,
    tile_rows: int,
    blocks: int,
    step: int = 1,
    base: int = 1,
    untouched: np.ndarray | None = None,
) -> np.ndarray:
    """第 k 块乘 `k + base`。`step > 1` 时没走到的块取 `untouched` 的对应行。

    `base` 参数把两组统一起来：简单循环乘 `k + 1`，流水那组乘 `k + 2`
    （ST 里 `scale = ki + 2`）。**一个函数两组用**，所以「哪一块乘了几」只写一次。
    """
    out = src.copy() if untouched is None else untouched.copy()
    for k in range(0, blocks, step):
        lo, hi = k * tile_rows, (k + 1) * tile_rows
        out[lo:hi] = src[lo:hi] * src.dtype.type(k + base)
    return out


def all_equal(arrays: list[np.ndarray]) -> bool:
    """一组数组是否两两逐位相同。流水那一组的判据就是这个。

    **逐位而不是 allclose**：`preload` 改的是指令的排布，不是算术,
    所以不同 `preload` 之间不该有任何数值差。有差就说明语义被改了。
    """
    return all(np.array_equal(item, arrays[0]) for item in arrays)
