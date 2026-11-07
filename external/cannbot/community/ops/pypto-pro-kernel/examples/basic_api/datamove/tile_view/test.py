# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""tile 的三层形状：物理形状、有效区、重解释后的视图。

**写法：原始。**

## 一、三层，不是三选一

| | 谁定 | 什么时候 | 改数据吗 |
| --- | --- | --- | --- |
| **物理形状** | ``TileType(shape=...)`` | 声明期，编译期常量 | 决定占多少 buffer |
| **有效区** | ``pl.set_validshape(tile, [r, c])`` | 运行期 | 不改，只限定搬多少 |
| **重解释的视图** | ``pl.reinterpret(tile, shape=/dtype=/layout=)`` | 声明期 | **不生成任何指令** |

## 二、`pl.reinterpret` 是视图声明，不是转换

官方 docstring 原话：「a compile-time view declaration, not a conversion — no
load/move/store/cast instruction is generated」。它在**原地址、原大小**重新声明一个 handle：

- 继承 memory space、fractal、pad、compact、mutex 映射；
- **不继承 `valid_shape`**——新 handle 从动态有效区开始（等于 `-1`）；
- 原 handle 不变。

**"不继承 valid_shape" 是个陷阱**，验证 5 专门钉它：`reinterpret` 读起来像"只改元数据"，
很容易以为收窄会跟着走。

## 三、GM↔UB 的 `set_validshape` 限定搬运量，load / store 两侧都生效（实测）

同一个 `8 × 16` 的 tile，收窄到 `4 × 8`，数 GM 上还剩几个哨兵（共 128 个位置）：

| 写法 | 哨兵剩 | 即写了 |
| --- | ---: | ---: |
| 不收窄 | 0 | 128 |
| 收窄放在 `pl.store` 前 | **96** | 32 = 4 × 8 |
| 收窄放在 `pl.load` 前 | **96** | 32 |
| 先收窄、再 `reinterpret` 成同形状 | **0** | 128（收窄丢了） |

**第 2、3 行的输出一样，但过程不同**：一个是"整块装进来、只搬一部分出去"，另一个是"只装了一部分
进来"。区别在 UB 里，GM 上看不出来——所以这两条判据放在一起只能说明"两侧都生效"，
**说明不了哪一侧生效了**。要区分得看 IR 或者故意在 UB 里留脏数据。这一点写清楚，免得有人
以为验证 2 和验证 3 是互相独立的证据。

## 四、整组操作

`pl.set_validshape` 和 `pl.reinterpret` **都接受 `make_tile_group`**：

- `set_validshape(group, [r, c])`：给组里**每个** tile 都设（docstring 原话「valid_shape is set
  on all tiles in the group」）。
- `reinterpret(group, shape=...)`：返回一个新组，**mutex 映射保留**。

保留 mutex 不是细节：新组丢了 mutex id 的话，`auto_mutex=True` 对它就失效，**而那种失效是静默
的**（`references/diagnostics/validation-traps.md` #7）。验证 7 **直接比 IR**，不靠"结果算对了"来间接推断。

**`.next()` 的轮转游标在原组和新组之间是共享的**（实测）：`group.next()` 拿槽 0 之后，
`wide.next()` 拿的是槽 **1**，IR 里一个是 `_bufidx_0` 一个是 `_bufidx_1`。这一条是踩出来的——
我第一版让 `wide.next()` 去搬 `group.next()` 装的数据，上板拿到「每 16 个元素前 8 个对、
后 8 个是 0」，那 8 个对的正是上一个 kernel 写进槽 1 的 `4 × 8` 残留。**又是 UB 残留**
（`references/verification/board-runs.md` §五点十一）。

## 五、如何独立验证

1. ``whole_tile``：不收窄不重解释，128 个元素全被写，一个哨兵不剩。
2. ``narrow_before_store``：只有左上角 `4 × 8` 被写，剩 96 个哨兵。
3. ``narrow_before_load``：同上（见 §三 的说明：这不是独立的第二份证据）。
4. ``reinterpret_to_another_shape``：按 `8 × 16` 装、按 `4 × 32` 搬出，等于 `x.reshape(4, 32)`。
5. ``reinterpret_drops_the_valid_shape``：收窄后再 `reinterpret`，**128 个全被写**——收窄丢了。
6. ``narrow_a_whole_group``：一次调用收窄整组，第二个槽搬出去的也只有 `4 × 8`。
7. ``reinterpret_a_whole_group``：**比 IR，不上板**。新组的 tile 形状变成 `[4, 32]`，
   而锁用的还是原来那个 mutex_ids tuple——映射没丢。
8. `reinterpret` footprint 超出原 buffer 被解析期拒。
9. `reinterpret` 改 dtype 不给 shape 被拒。
10. `reinterpret` 什么都不给被拒。
11. VF 用坐标互异的矩阵检验真实字节地址：ZN 标签与新 shape 不产生物理转置。
12. FIX 用先写 `A@B`、再经窄视图写 `2*(A@B)` 区分实际覆盖范围；真实 backing 始终容纳两侧的完整搬运，另外检查 GM 前后 guard 和变更输入后的重复调用。

**验证 1 是其余判据的对照**：没有它，"剩 96 个哨兵"这个数就没有基准，因为"少搬了"和"根本没搬"
在哨兵计数上长得很像。验证 5 的价值全在和验证 2 的对比上——同样先收窄，一个剩 96、一个剩 0。

## 六、这一例不讲什么

- **`reinterpret` 改 dtype 那一面**——含 5 个进不了搬运通道的 dtype 怎么绕，在
  ``support/dtype_carriers``。这一例只讲 shape。
- **收窄与 `compact` 的互动**——那是 matmul 路径上的事，而且"元素错位且不报错"，
  完整规则在 [`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #4。
  **这一例刻意不碰**：`compact` 的规则按"哪根轴被收窄"分情况，混进来会把这一例的主题冲掉。
- **填充区**——`pad=` 与 `pl.fillpad` 在 ``dma_pad``。
- **GM 侧的视图**——`pl.make_tensor` 的步长视图在 ``strided_view``。

## 七、`reinterpret` 的基址对齐检查从 Vec 这一侧摸不到

docstring 说「the base address must be aligned to the new element size (both checked at parse
time)」。想构造反例：`addr=0x0004` 的 FP32 tile 重解释成 INT64（8 字节）。**构造不出来**——
tile 创建期先拦下了：

```
Tile address 0x00004 (4) is not 32-byte aligned for memory space Vec
```

`block_ops.py:3511`。Vec 的 tile 地址必须 32 字节对齐，而 32 是所有元素宽度（最大 8 字节）的
倍数，**所以任何合法的 Vec tile 地址天然满足新 dtype 的对齐要求**。

这条不是"检查不存在"，是"从这个方向摸不到"——别的 memory space 或别的 tile 构造路径可能摸得到。
**记下来是因为「没覆盖」和「不需要覆盖」不是一回事**，而且这类可达性问题上本仓栽过
（`references/verification/evidence.md` 防法第 13 条：说是缺口之前先问它摸不摸得到）。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidShape, InvalidType, OutOfRange

from .kernel import load_reinterpreted_layout, make_receive_view
from .kernel import (
    COLS,
    ROWS,
    VALID_COLS,
    VALID_ROWS,
    WIDE_COLS,
    WIDE_ROWS,
    narrow_a_whole_group,
    narrow_before_load,
    narrow_before_store,
    reinterpret_a_whole_group,
    reinterpret_beyond_the_buffer,
    reinterpret_drops_the_valid_shape,
    reinterpret_dtype_without_shape,
    reinterpret_to_another_shape,
    reinterpret_with_nothing_to_override,
    whole_tile,
)
from .reference import (
    SENTINEL,
    accumulated_product,
    make_input,
    reshaped,
    valid_region_only,
    vf_view_window,
)


def _run(kernel, device, out_shape=(ROWS, COLS)):
    x = make_input(ROWS, COLS)
    out = torch.full(out_shape, float(SENTINEL), device=device, dtype=torch.float32)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def _parse(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)
    return pp.ir.python_print(program)


def test_a_tile_without_narrowing_moves_everything(device):
    """验证 1，其余判据的对照。128 个位置全被写。

    **这条不能省**：没有它，"剩 96 个哨兵"就没有基准——"少搬了"和"根本没搬"在哨兵计数上
    长得很像。
    """
    got, x = _run(whole_tile, device)
    np.testing.assert_array_equal(got, x)
    assert not (got == SENTINEL).any()


@pytest.mark.parametrize("kernel", [narrow_before_store, narrow_before_load])
def test_narrowing_limits_how_much_is_moved(device, kernel):
    """验证 2 / 3。收窄到 `4 × 8`，GM 上只有左上角被写，剩 96 个哨兵。

    两个 kernel 参数化在一起是**有意的**，因为它们在 GM 上的结果完全相同（§三）：一个是
    "整块装进来、只搬一部分出去"，另一个是"只装了一部分进来"，区别在 UB 里。
    **把它们写成两条独立的判据会让人以为拿到了两份证据。**
    """
    got, x = _run(kernel, device)
    np.testing.assert_array_equal(got, valid_region_only(x, VALID_ROWS, VALID_COLS))
    assert int((got == SENTINEL).sum()) == ROWS * COLS - VALID_ROWS * VALID_COLS


def test_reinterpret_changes_the_shape_without_moving_data(device):
    """验证 4。按 `8 × 16` 装、按 `4 × 32` 搬出，等于 `x.reshape(4, 32)`。

    行优先下这两个形状覆盖同一串字节，**逐位相等就是"没生成搬运指令"的可观测后果**。
    如果 `reinterpret` 真的搬了数据，结果只会更"整"，不会正好等于 reshape。
    """
    got, x = _run(
        reinterpret_to_another_shape, device, out_shape=(WIDE_ROWS, WIDE_COLS)
    )
    np.testing.assert_array_equal(got, reshaped(x, WIDE_ROWS, WIDE_COLS))


def test_reinterpret_does_not_inherit_the_valid_shape(device):
    """验证 5，这一例的陷阱。先收窄到 `4 × 8`，再 `reinterpret` 成同样的形状 → **收窄丢了**。

    判据是"一个哨兵都不剩"。**它的全部价值在和验证 2 的对比上**：同样先收窄，那边剩 96 个哨兵，
    这边剩 0 个。单看这一条会以为它只是又一次整块搬运。
    """
    got, x = _run(reinterpret_drops_the_valid_shape, device)
    np.testing.assert_array_equal(got, x)
    assert not (got == SENTINEL).any(), (
        "还有哨兵，说明 reinterpret 把 valid_shape 带过来了——docstring 说它不该带"
    )


def test_narrowing_a_group_narrows_every_slot(device):
    """验证 6。一次 `set_validshape(group, ...)`，组里两个槽都被收窄。

    判据落在**第二个**槽搬出去的结果上：如果只有第一个槽被设了，第二个槽会整块搬出，
    哨兵会剩 0 个而不是 96 个。
    """
    got, x = _run(narrow_a_whole_group, device)
    np.testing.assert_array_equal(got, valid_region_only(x, VALID_ROWS, VALID_COLS))


def test_reinterpreting_a_group_keeps_the_mutex_mapping():
    """验证 7，**不上板**：直接比 IR。

    两条断言对应两件事：新组的形状真的变了（`[4, 32]`），而锁引用的还是**原来那个**
    mutex_ids tuple（`_tg_group_mutex_ids`）——映射没丢。

    为什么不用数值判据：`.next()` 的游标在原组和新组之间共享，`wide.next()` 拿到的是没装过
    数据的槽 1，上板读到的是 UB 残留。**数值判据在这里只会证明残留长什么样。**
    """
    text = _parse(reinterpret_a_whole_group)
    assert "make_tile([4, 32]" in text, "新组的形状没变成 [4, 32]"
    assert "_tg_group_mutex_ids" in text, "锁没有引用原组的 mutex_ids tuple —— 映射丢了"
    assert "mutex_lock_dyn" in text and "mutex_unlock_dyn" in text


def test_a_reinterpret_beyond_the_buffer_is_refused():
    """验证 8。`16 × 16` FP32 = 1024 字节 > 原 buffer 的 512。"""
    with pytest.raises(OutOfRange) as raised:
        _parse(reinterpret_beyond_the_buffer)
    assert "exceeds the original buffer size" in str(raised.value), str(raised.value)


def test_changing_the_dtype_requires_a_shape():
    """验证 9。**元素数要重新说一遍**，不让框架替你猜。"""
    with pytest.raises(InvalidShape) as raised:
        _parse(reinterpret_dtype_without_shape)
    assert "'shape' is required when 'dtype' changes" in str(raised.value), str(
        raised.value
    )


def test_an_empty_reinterpret_is_refused():
    """验证 10。三个都不给时被拒，而不是返回一个一样的 handle。

    这一条看着琐碎，但它把"`reinterpret` 是个声明"这件事钉住了：一个什么都不改的声明没有意义，
    所以它是错误而不是恒等操作。
    """
    with pytest.raises(InvalidType) as raised:
        _parse(reinterpret_with_nothing_to_override)
    assert "at least one of dtype/shape/layout" in str(raised.value), str(raised.value)


def test_vf_layout_view_preserves_bytes_without_transposing(device):
    x = make_input(16, 64)
    out = torch.full((1, 64), float(SENTINEL), dtype=torch.float32, device=device)
    load_reinterpreted_layout[None, 1](torch.from_numpy(x).to(device), out)
    got = out.cpu().numpy()
    np.testing.assert_array_equal(got, vf_view_window(x))
    assert not np.array_equal(got[0, :16], x[:, 2])


@pytest.mark.parametrize("view_rows", [1, 7, 16])
def test_fix_transfer_extent_is_owned_by_source(device, view_rows):
    kernel = make_receive_view(view_rows)
    for seed in (17, 29):
        rng = np.random.default_rng(seed)
        a = rng.integers(-2, 3, size=(32, 16)).astype(np.float16)
        b = rng.integers(-2, 3, size=(16, 32)).astype(np.float16)
        backing = torch.full(
            (36, 32), float(SENTINEL), dtype=torch.float32, device=device
        )
        kernel[None, 1](
            torch.from_numpy(a).to(device),
            torch.from_numpy(b).to(device),
            backing[2:34],
        )
        got = backing.cpu().numpy()
        expected = accumulated_product(a, b)
        np.testing.assert_array_equal(got[2:34], expected)
        np.testing.assert_array_equal(got[:2], SENTINEL)
        np.testing.assert_array_equal(got[34:], SENTINEL)
        if view_rows < 16:
            # 窄视图之外也被第二次 FIX 改写；先前的 A@B 不能冒充正确结果。
            assert np.any(expected.reshape(2, 16, 32)[:, view_rows:] != 0)
