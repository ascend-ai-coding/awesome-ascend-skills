# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""填充区：`TilePad` 定填什么，`pl.fillpad` 定什么时候填，`set_validshape` 定填哪儿。

**写法：原始。**

## 一、三件事，三个地方

| | 写在哪 | 管什么 |
| --- | --- | --- |
| ``pad=pl.TilePad.zero / max / min / null`` | **目的 `TileType` 上**，声明期 | 填成什么值 |
| ``pl.set_validshape(src, [r, c])`` | 运行期，作用在**源** tile 上 | 哪一块是有效区 |
| ``pl.fillpad(dst, src, mode=...)`` | 一条指令 | 把有效区之外真的写掉 |

**`TilePad` 只是声明。** 不调 `fillpad`，填充区里就是上一次运行留在 UB 里的残留——验证 6
专门钉这一点，因为这是「有效区对、外面是脏数据」这类静默错误的来源。

## 二、`FillPadMode` 三个值：差别在目的，不在收窄方式

| | `dst` 的形状 | `dst` 的地址 |
| --- | --- | --- |
| ``NORMAL``（缺省） | 与 `src` 同形 | 与 `src` **不同** |
| ``EXPAND`` | **比 `src` 大** | 与 `src` 不同 |
| ``INPLACE`` | 与 `src` 同形 | 与 `src` **同址** |

三个模式**收窄源的方式完全一样**，都是 `set_validshape`。我最初把 `EXPAND` 理解成「源 tile
的形状更小、不用 `set_validshape`」，那样写源没有有效区边界，`fillpad` 拿不到该从哪儿开始填，
结果是残留。ST `test_pro_fillpad.py::fillpad_expand_dynamic_cce_kernel` 里源是 `8 × 8`
带 `set_validshape(src, [5, 7])`，目的是 `8 × 16`——**变的是目的**。

## 三、为什么用 INT32

`max` / `min` 在整型上是明确的 `2147483647` / `-2147483648`，一眼能认出是填充值。FP32 的极值
读出来是 `3.4e38`，和"算错了"长得太像——**判据要能让人一眼分清「填对了」和「算坏了」**。

输入取 `1..64` 而不是 `0..63`：**不含 0**，这样「填充区是 0」不会和有效区的数据撞上。

## 四、`TilePad` 就是字面意思

`zero`→`0`、`max`→`2147483647`、`min`→`-2147483648`。三个值实测与声明一致，冷进程、
热进程、任意调用顺序都一致。**这一句平淡的话是这一例最贵的一行**，见 §七。

## 五、如何独立验证

1. ``fillpad_zero``：`5 × 7` 有效区保留，其余是 `0`。
2. ``fillpad_max``：其余是 `2147483647`。
3. ``fillpad_min``：其余是 `-2147483648`。
4. ``fillpad_expand``：目的 `8 × 16`，有效区仍是 `5 × 7`，其余（含多出来的 8 列）是 `0`。
5. ``fillpad_inplace``：`src` 与 `dst` 同址，结果与验证 1 **逐位相同**。
6. ``pad_declared_but_not_filled``：**没有数值判据**，只断言它和验证 1 的输出**不同**。
   为此那个 kernel 先把整块 UB 写满已知非零数据——否则残留碰巧是 0 就什么都证明不了。
7. ``pad=null`` 配 ``fillpad`` 必须被**解析期**拒，消息里点名规则而不是错误码。

第 1 条单独不够。**第 2、3 条才排除「根本没填」**（填充区是 0 和「没动过的零页」分不开），
第 6 条才排除「`fillpad` 是空操作」，第 5 条才排除「同址退化成了 `NORMAL` 的两块内存」。

## 六、这一例不讲什么

- **有效区本身的用法**——`set_validshape` 在这里只是配角，它的完整语义（含读侧行为、
  tile_group 整组设置）在 ``tile_view``。
- **GM 侧的 padding**——`pl.load` / `pl.store` 对不齐的张量做的事不在这里。
- **`null` 填什么**——它配 `fillpad` 根本不让用（验证 7）。`pad=null` 有意义的场合是 `pl.load`
  那条 TLOAD 路径，那不在这一例里。
- **同步机制本身**——这一例被迫手工插同步（§七），但 `bar_*` / `sync_*` 的完整语义在
  ``move_sync``。

## 七、这一例真正查出来的东西：`auto_mutex=True` 不管裸 `make_tile`

**我在这一例上报了一个不存在的框架缺陷，然后撤回。** 过程值得写下来，因为省下的不是这一次的
时间，是下一次的。

最初每个 kernel 都是 `@pl.jit(auto_mutex=True)` 加裸 `pl.make_tile`，没有手工同步。量出来的
填充值与声明的名字**错位一位**：`zero`→`-2147483648`、`max`→`0`、`min`→`2147483647`，
三个值正好成一个循环。我验了四次，两种进程形态，字面与工厂两种写法，都一致；pypto 侧逐层查过
全对，连生成的 `kernel.cpp` 里都是 `PadValue::Zero`，ISA 头的 `PadValueMap<int32_t,
PadValue::Zero>` 也是 0。于是写成了「根因在 ISA 头的映射与硬件行为之间」的缺陷报告。

**是删掉两个无关 kernel 之后结果变了，才触发可重复性检查。** 同一个 pad 单独跑一个进程给的是
正确的 `0`；换调用顺序给的是另一个答案。根因：

> `auto_mutex=True` 靠 `pl.make_tile_group` 的 `mutex_ids` 推依赖。用裸 `pl.make_tile` 时
> 没有 mutex id 可推，于是 MTE2 的 load、V 的 fill、MTE3 的 store 之间**一条同步都不会插**，
> 而且不报任何错。

那三个"稳定"的错位值是残留：前一个样例在同一块 UB 地址留下的字节，被当成了这一次的填充值。
换成 ST 的写法（`@pl.jit()` + 手工 `sync_src` / `sync_dst`）之后，三个值全部等于声明值，
任意顺序都一致。缺陷报告已删除。

留下三样东西：

1. **`check.py` 的第五条规则** `auto-mutex-needs-tile-group`——`auto_mutex=True` 的 kernel
   里出现裸 `pl.make_tile` 而没有 `make_tile_group`，静态就能判。文档纠正不了这类错误，
   因为它的症状会伪装成别的东西（按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 的判据，这该落在工具里）。
2. **`references/verification/board-runs.md` §五点十一**：UB 不是零初始化的，"数值不重要"这句话在你
   把数值打印出来的那一刻就不再成立。
3. **`references/verification/evidence.md` #16**：四次一致的测量不构成可重复性证据——四次里如果每次
   的前置状态都一样，那是同一次测量做了四遍。

## 八、`EXPAND` / `INPLACE` 曾经也是这个坑

它们一度被记成「未覆盖的缺口，两次尝试都拿到 UB 残留，原因未定」。原因现在清楚了，而且是两层
叠在一起：少同步（§七），外加我把 `EXPAND` 的语义理解错了（§二）。两层都修掉之后它们就通了,
所以这一例不再有缺口。

路上量到一件确实的事，与本题无关但省下次的时间：**tile 的列数受 32 字节对齐约束。**
`4 × 4` INT32 = 16 字节会被 tile 构造期拒（`_validate_tile_layout_alignments`）：
`BLayout::RowMajor` 要求 `Cols * sizeof(DType) % 32 == 0`。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import PyptoProError

from .kernel import (
    COLS,
    ROWS,
    VALID_COLS,
    VALID_ROWS,
    WIDE_COLS,
    fillpad_expand,
    fillpad_inplace,
    fillpad_max,
    fillpad_min,
    fillpad_null,
    fillpad_zero,
    pad_declared_but_not_filled,
)
from .reference import PAD_VALUE, expanded, make_input, padded


def _run(kernel, device, out_cols=COLS):
    x = make_input(ROWS, COLS)
    out = torch.zeros((ROWS, out_cols), device=device, dtype=torch.int32)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


@pytest.mark.parametrize(
    "kernel, pad_name",
    [
        (fillpad_zero, "zero"),
        (fillpad_max, "max"),
        (fillpad_min, "min"),
    ],
)
def test_normal_mode_fills_outside_the_valid_shape(device, kernel, pad_name):
    """验证 1 / 2 / 3。`5 × 7` 有效区保留，其余是这个 `TilePad` 字面上的值。

    三个 pad 一起参数化不是为了省行数：只测 `zero` 的话，「填充区是 0」和「那块内存从来没被
    写过」分不开——`max` / `min` 才把「真的填了」钉住。
    """
    got, x = _run(kernel, device)
    np.testing.assert_array_equal(
        got, padded(x, VALID_ROWS, VALID_COLS, PAD_VALUE[pad_name])
    )


def test_expand_mode_fills_a_wider_destination(device):
    """验证 4。目的 `8 × 16`，有效区仍按 `set_validshape` 收窄，其余全 `0`。

    这条同时钉住 §二 那件事：`EXPAND` 改的是目的的形状，收窄源的方式和 `NORMAL` 一样。
    如果把它写成「源 tile 更小、不用 set_validshape」，有效区会变成源的整块 `8 × 8`，
    判据会在第 6、7 列上红。
    """
    got, x = _run(fillpad_expand, device, out_cols=WIDE_COLS)
    np.testing.assert_array_equal(
        got, expanded(x, VALID_ROWS, VALID_COLS, ROWS, WIDE_COLS, PAD_VALUE["zero"])
    )


def test_inplace_mode_matches_normal_mode(device):
    """验证 5。同址原地填，结果与 `NORMAL` 逐位相同。

    比的是两个 kernel 的输出而不是比一张 golden：golden 相同只能说明两者都对，
    **逐位相同才排除「同址退化成了两块内存各填一次」**——那种情况下 `INPLACE` 也会"对"，
    但省不下那块 UB，而省 UB 正是用它的唯一理由。
    """
    inplace, _ = _run(fillpad_inplace, device)
    normal, _ = _run(fillpad_zero, device)
    np.testing.assert_array_equal(inplace, normal)


def test_declaring_pad_without_calling_fillpad_changes_nothing(device):
    """验证 6。`TilePad` 只是声明——不调 `fillpad`，填充区就是残留。

    **没有数值判据**：残留不可预测。能断言的是它和填过的结果**不同**，而这条断言之所以成立，
    是因为那个 kernel 先把整块 UB 写满了 `1..64`——不先写的话残留碰巧是 0 就会和 `zero` 撞上，
    这条测试会以错误的理由通过。
    """
    unfilled, x = _run(pad_declared_but_not_filled, device)
    filled, _ = _run(fillpad_zero, device)
    assert not np.array_equal(unfilled, filled), (
        "不调 fillpad 却拿到了和填过一样的结果——要么 fillpad 是空操作，"
        "要么那块 UB 的预置数据没生效"
    )
    # 有效区两边必须一致：fillpad 不该碰有效区
    np.testing.assert_array_equal(
        unfilled[:VALID_ROWS, :VALID_COLS], filled[:VALID_ROWS, :VALID_COLS]
    )


def test_pad_null_is_refused_when_fillpad_is_called():
    """验证 7。`pad=null` 配 `pl.fillpad` **被解析期拒**，不需要卡。

    ```
    block.fillpad: out.hardware_info.pad must not be TilePad.null
    ```

    `out_memory.cpp:66`。**这不是缺陷**：ISA 侧有对应的 `static_assert`
    （`pto/npu/a5/TFillPad.hpp:118`「Fix: TFillPad dst vecTile pad value must not be Null!」），
    两层一致，而且 pypto 这一层拒得更早、消息带源码行。

    判据匹配**规则文本**而不是错误码——匹配错误码会让判据变成同义反复（[证据判断](../../../../references/verification/evidence.md) #14）。
    """
    with pytest.raises(PyptoProError) as raised:
        fillpad_null.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)
    assert "must not be TilePad.null" in str(raised.value), str(raised.value)
