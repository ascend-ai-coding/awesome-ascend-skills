# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""GM↔UB 最小往返：两套偏移的分工。

**这是 datamove 章的第一例。** 前面 21 例都假定「数据已经在 UB 里」；这一例回答它是怎么到的。

**写法：原始。** 搬运面没有语法糖（`pl.mem_space()` 管的是资源声明，不是搬运）。

## 一、两套偏移，单位不同

| 接口 | 偏移单位 | 取 `[1, 1]` 号块 |
| --- | --- | --- |
| ``pl.load`` / ``pl.store`` | **元素** | ``[8, 16]``（自己乘） |
| ``pl.load_tile`` / ``pl.store_tile`` | **tile 块** | ``[1, 1]``（框架乘） |

`load_tile` 的 docstring 写得很清楚：「Offsets are in tile-block units, internally multiplied
by tile shape」。所以这两套**不是新旧关系，是单位不同**。

## 二、块索引什么时候不够用

块索引只能表达「第 n 个整块的起点」。张量形状切不齐的时候，最后一块的起点不是任何块索引的乘积
——那里必须用绝对偏移，再配 ``pl.set_validshape`` 把有效区收窄（收窄那一套在 ``tile_view``，
填充那一套在 ``dma_pad``）。**所以两套都得会。**

这一例的张量刻意取成 `16 × 32`，正好 2×2 个 `8 × 16` 的块——切得齐，两套都能写，才能逐位比。

## 三、为什么中间什么都不算

三个 kernel 都是 GM → UB → GM，一个 tile，零运算。**不算东西是判据设计的一部分**：一旦中间做了
运算，对不上的时候就分不清是偏移错了、片上拷贝错了，还是同步错了。这一例只想回答偏移。

## 四、哨兵

`out` 先被主机填成 `-1.0`，前两个 kernel 只写一个块。**没有哨兵就只能证明那一块对，证明不了
别的块没被碰**——而偏移算错最典型的表现正是写到了邻块。

## 五、如何独立验证

1. ``roundtrip_absolute``：`[1, 1]` 号块原样写回，其余三块保持 `-1.0`。
2. ``roundtrip_block_index``：与验证 1 **逐位相同**。这条把「块索引 = 索引 × 形状」钉住——
   如果框架的乘法错了，两者会差一整块。
3. ``roundtrip_all_blocks``：四块全写，整张等于输入，一个哨兵都不剩。

第 1 条单独不够：它只证明一种写法自洽。**第 2 条才是这一例的主题**，第 3 条才排除「只有
`[1, 1]` 这一个索引碰巧对」。

## 六、这一例不讲什么

- **有效区与尾块**——``pl.set_validshape`` 在 ``tile_view``。
- **填充区**——``pl.fillpad`` 与 ``TilePad`` 在 ``dma_pad``。
- **`order=` 的轴映射**——`load_tile` / `store_tile` 都有这个参数，它改的是 tile 维到张量轴的
  对应；那是布局的事，在 ``layout_table``。
- **同步**——这里全靠 `auto_mutex=True`。手工 `bar_` / `sync_` 在 ``move_sync``。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import torch

from .kernel import (
    BLOCKS,
    COLS,
    FULL_COLS,
    FULL_ROWS,
    ROWS,
    roundtrip_absolute,
    roundtrip_all_blocks,
    roundtrip_block_index,
)
from .reference import SENTINEL, block_written, make_input


def _dev(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _run(kernel, device):
    x = make_input(FULL_ROWS, FULL_COLS)
    out = torch.full(
        (FULL_ROWS, FULL_COLS), SENTINEL, device=device, dtype=torch.float32
    )
    kernel(_dev(x, device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def test_absolute_offsets_move_one_block(device):
    """验证 1。元素坐标 `[8, 16]` 取 `[1, 1]` 号块，其余三块必须还是哨兵。"""
    got, x = _run(roundtrip_absolute, device)
    np.testing.assert_array_equal(got, block_written(x, 1, 1, ROWS, COLS))


def test_block_index_lands_on_the_same_block(device):
    """验证 2，这一例的主题。两套偏移逐位相同。

    如果框架把块索引乘错了（比如按元素坐标当块索引用），`load_tile(tile, x, [1, 1])` 会落在
    `[1, 1]` 这个元素上而不是 `[8, 16]`，两者会差一整块——不是差几个元素，所以这条判据很钝，
    钝得刚好：它不会因为边界上的一两个元素而误报。
    """
    absolute, _ = _run(roundtrip_absolute, device)
    block_index, _ = _run(roundtrip_block_index, device)
    np.testing.assert_array_equal(absolute, block_index)


def test_every_block_is_reachable_by_index(device):
    """验证 3。四个块全搬，整张等于输入。

    防的是「只有 `[1, 1]` 碰巧对」：`[0, 0]` 在两套写法下都是 `[0, 0]`，光看它分不出单位；
    `[0, 1]` 和 `[1, 0]` 还能查出行列写反。
    """
    got, x = _run(roundtrip_all_blocks, device)
    np.testing.assert_array_equal(got, x)
    assert not (got == SENTINEL).any(), "还有哨兵没被覆盖，说明有块没搬到"
    assert BLOCKS * BLOCKS == 4
