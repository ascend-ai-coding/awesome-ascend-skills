# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""GM 步长视图（`pl.make_tensor`）与指针算术（`pl.make_ptr` / `pl.addptr`）。

**写法：原始。**

## 一、三个 API 的分工

| | 做什么 | 单位 |
| --- | --- | --- |
| ``pl.make_ptr(tensor, dtype=)`` | 取出裸指针，可顺便重解释元素类型 | — |
| ``pl.addptr(ptr, n)`` | 指针前移 `n` 个**元素** | 元素 |
| ``pl.make_tensor(src, shape, stride, dtype=)`` | 造视图：自己的形状、步长、可选 dtype | 步长按**元素** |

`make_tensor` 的第一个实参既可以是裸指针，也可以直接是一个 `pl.Tensor`。

## 二、这一例的主题：偏移也走步长

**视图的步长对通过它的每一次访问都生效，`pl.load` / `pl.store` 的偏移也不例外。**

想表达「从第 i 行开始、之后每隔 s 个元素取一行」：

| 写法 | 实际读到 |
| --- | --- |
| ``pl.load(tile, view, [i, 0])`` | 第 `i * s` 个元素起——**偏移被乘上了步长** |
| ``pl.make_tensor(pl.addptr(pl.make_ptr(x), i * COLS), ..., [s, 1])`` | 第 `i` 行起，行距 `s` |

验证 2 和验证 3 是同一个步长下的这两种写法，读到的行不同（第 1、3、5 … vs 第 2、4、6 …）。
**两条一起才说明这条规则**，单看任何一条都会以为自己理解了。

ST 的注释把这件事写在了源头（`test_strided_gather_view.py:105-108`）：

> The view is positioned by pointer rather than by a load offset: its stride is the *pair*
> spacing, so an offset through the view would step by s, not by one line.

## 三、为什么不是 `pl.set_stride`

同一份 ST 的模块 docstring 记了这套接口替掉了什么：旧的 `pl.set_stride` **就地改写**一个张量
的描述符，于是

1. 偏移算术仍然按**原来的**步长做；
2. 效果泄漏到之后对该张量的每一次访问；
3. 只需要一种布局变体时，也会写坏张量本身的声明。

视图是独立的张量，三个问题都没有。**这一段不是历史掌故**：它解释了为什么"改张量"这个更直觉的
做法被放弃了，省得有人再去找它。

## 四、如何独立验证

1. ``view_every_other_row``：步长 `2 * COLS`，读到第 0、2、4 … 14 行，等于 `x[0::2]`。
2. ``view_positioned_by_pointer``：`addptr` 移一行，读到第 1、3、5 … 15 行，等于 `x[1::2]`。
3. ``offset_goes_through_the_stride``：`pl.load(..., [1, 0])` 读到第 2、4、6 … 行，
   **不是**第 1 行起——等于 `x[2::2][:7]`。
4. ``view_reinterprets_the_dtype``：`dtype=I32` 的视图读出 FP32 的**位模式**，
   等于 `x.view(np.int32)`。
5. ``overlapping_writes``：两个重叠的视图各写一次。**重叠区不进判据**（结果不确定，见 §六），
   只断言 A 独占段、B 独占段和尾部哨兵。
6. ``overlapping_writes_ordered``：两条 store 之间加一条 `pl.system.bar_mte3()`，
   重叠区变确定——后写的赢。
7. ``same_pipe_flag_is_refused``：想用 flag 串两条 MTE3 的 store，`set_pipe == wait_pipe` 被拒。

**第 1 条单独不够**：步长 `2 * COLS` 且从第 0 行起，这一条并不能区分「步长生效了」和
「形状恰好只读了前 8 行的一半」。第 2、3 条才把步长和偏移的关系钉住。

步长刻意取 `2 * COLS` 而不是 `COLS`：**步长等于行宽时视图和原张量一样**，判据什么都证明不了。

## 五、`dtype=` 是重解释，不是转换

判据用 numpy 的 `x.view(np.int32)` 而不是 `x.astype(np.int32)`——这两个差得很远：

| | `1.0` 变成 |
| --- | --- |
| `view`（重解释字节） | `1065353216` |
| `astype`（数值转换） | `1` |

视图的 `dtype=` 做的是前者。**判据选错函数的话，这条测试会以错误的理由失败**，然后人会去怀疑
框架。

## 六、重叠写：没有别名检查，而且**同一条流水内也不保证程序序**

两个视图指向同一块 GM 这件事框架看不见，写重叠了不报也不警告。

我最初写的是「两条 `pl.store` 都在 MTE3 上，同一条流水内按程序序，所以结果确定」。**那是错的。**
实测 20 次（4 个进程各 5 次）：

| | 重叠区归谁 |
| --- | --- |
| 两条 store 之间什么都不加 | **14 次 B（后写）、6 次 A（先写）**，同一进程里连着跑也会变 |
| 中间加一条 `pl.system.bar_mte3()` | **20 次全是 B**，冷进程、热进程、换顺序都一致 |

所以：

- 验证 5 的判据**只断言重叠区之外**。把不确定的东西写进判据会得到一条偶发失败的测试，
  而偶发失败的测试比没有测试更糟——它会训练人去重跑。
- 验证 6 才是"后写的赢"这句话成立的前提。

**这是 `bar_*` 在本仓唯一出现的地方，而且是它的正当用法**：要的是"流水内不并行"，
不是"让 A 等 B"。flag 在这里用不上——`sync_src` / `sync_dst` 要求 `set_pipe != wait_pipe`,
MTE3→MTE3 解析期直接拒（验证 7 钉住这一点，免得"只能用 barrier"停留在一句说法）。

跨流水定序为什么不能用 `bar_*`，见 [`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #8。
两件事不矛盾：**`bar_*` 排空一条流水，flag 建立两条流水之间的先后**。这一节是前者的用例，
那一条是拿前者去干后者的活。

## 七、这一例不讲什么

- **tile 侧的视图**——`pl.make_tile` 的逻辑形状 vs 物理形状、有效区在 ``tile_view``。
- **`pl.Ptr` 作为 kernel 形参**——那条路在 `examples/matmul/tilingkey_specialize`（`tiling_key`
  的字段不是形参，所以形参只能是 `pl.Ptr`）。
- **越界的视图**——形状 × 步长超出原张量时会读到张量之外。**没测**：读越界的 GM 拿到什么取决于
  设备内存里恰好有什么，没有判据可写，而且它和 ``init_and_insert`` 里那条越界 `init_output`
  是同一类风险。**这是明说跳过。**

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import pytest
import torch

from .kernel import (
    COLS,
    HALF_ROWS,
    OFFSET_ROWS,
    OVERLAP_ROWS,
    OVERLAP_START,
    PTR_OFFSET_ROWS,
    ROWS,
    offset_goes_through_the_stride,
    overlapping_writes,
    overlapping_writes_ordered,
    same_pipe_flag_is_refused,
    view_every_other_row,
    view_positioned_by_pointer,
    view_reinterprets_the_dtype,
)
from .kernel import make_word_gather
from .reference import indexed_rows
from .reference import (
    SENTINEL,
    as_int32,
    every_other_row,
    make_input,
    overlapped,
    overlapped_certain_parts,
)


def _run(kernel, device, out_rows=HALF_ROWS, dtype=torch.float32):
    x = make_input(ROWS, COLS)
    out = torch.full((out_rows, COLS), float(SENTINEL), device=device, dtype=dtype)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def test_a_strided_view_gathers_every_other_row(device):
    """验证 1。步长 `2 * COLS` → 第 0、2、4 … 14 行。

    这条单独不够：从第 0 行起、取 8 行，「步长生效了」和「形状恰好读了前 8 行」都能解释它。
    下面两条才把步长钉住。
    """
    got, x = _run(view_every_other_row, device)
    np.testing.assert_array_equal(got, every_other_row(x, 0, HALF_ROWS))


def test_addptr_moves_the_base_by_elements(device):
    """验证 2。`addptr(ptr, 1 * COLS)` → 第 1、3、5 … 15 行。

    `addptr` 的单位是**元素**，所以"一行"要写成 `1 * COLS`。如果它是按行或按字节，
    这条会落在完全不同的地方。
    """
    got, x = _run(view_positioned_by_pointer, device)
    np.testing.assert_array_equal(got, every_other_row(x, PTR_OFFSET_ROWS, HALF_ROWS))


def test_a_load_offset_is_multiplied_by_the_view_stride(device):
    """验证 3，**这一例的主题**。`pl.load(..., [1, 0])` 落在第 2 行，不是第 1 行。

    和验证 2 对着看：同一个步长，`addptr` 移一行 → 第 1、3、5 …；load 偏移 1 → 第 2、4、6 …。
    **两条的差就是这条规则本身**。如果偏移不走步长，这条会等于验证 2 的结果。
    """
    got, x = _run(offset_goes_through_the_stride, device, out_rows=OFFSET_ROWS)
    np.testing.assert_array_equal(got, every_other_row(x, 2, OFFSET_ROWS))


def test_the_dtype_argument_reinterprets_the_bytes(device):
    """验证 4。`dtype=I32` 的视图读出 FP32 的位模式。

    判据用 `view` 而不是 `astype`（§五）：前者把 `1.0` 读成 `1065353216`，后者读成 `1`。
    选错的话这条会以错误的理由失败，然后人会去怀疑框架。
    """
    got, x = _run(view_reinterprets_the_dtype, device, dtype=torch.int32)
    np.testing.assert_array_equal(got, as_int32(x)[:HALF_ROWS])


def test_both_overlapping_views_land_but_the_overlap_is_undecided(device):
    """验证 5。两条 store 都落了地，但**重叠区归谁不确定**，所以它不进判据。

    实测 20 次：14 次重叠区是 B 的、6 次是 A 的（§六）。断言的是三段确定的东西——
    A 独占段、B 独占段、尾部哨兵。**尾部那一段少不了**：没有它就证不了视图 B 没写出自己的形状。
    """
    got, x = _run(overlapping_writes, device, out_rows=ROWS)
    for rows, expected, why in overlapped_certain_parts(x, OVERLAP_ROWS, OVERLAP_START):
        np.testing.assert_array_equal(got[rows], expected, err_msg=why)
    assert (got[slice(OVERLAP_START + OVERLAP_ROWS, None)] == SENTINEL).all(), (
        "尾部的哨兵被覆盖了，说明视图 B 写出了它自己的形状"
    )


def test_a_pipe_barrier_makes_the_later_write_win(device):
    """验证 6。两条 store 之间一条 `pl.system.bar_mte3()`，重叠区变确定。

    和验证 5 对着看：**同一个 kernel，只多了一条 barrier**，不确定的那一段变确定了。
    这是 `bar_*` 的正当用法——流水内不并行，不是跨流水定序。
    """
    got, x = _run(overlapping_writes_ordered, device, out_rows=ROWS)
    np.testing.assert_array_equal(
        got, overlapped(x, ROWS, COLS, OVERLAP_ROWS, OVERLAP_START)
    )


def test_a_flag_between_two_mte3_stores_is_refused():
    """验证 7。`set_pipe == wait_pipe` 被解析期拒，所以这里只能用 barrier。

    留着这条是为了让「只能用 `bar_mte3()`」有凭据而不是一句说法。
    """
    import pypto_pro as pp
    from pypto_pro._errors import InvalidOperation

    with pytest.raises(InvalidOperation) as raised:
        same_pipe_flag_is_refused.to_kernel_def().parse_target_program(
            pp.ir.SectionKind.Vector
        )
    assert "set_pipe and wait_pipe must differ" in str(raised.value), str(raised.value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("word_bits", [32, 64])
@pytest.mark.parametrize("shape", [(2, 17, 23, 24), (3, 67, 39, 40)])
def test_indexed_words_preserve_bytes_and_guards(shape, dtype, word_bits, device):
    batch, source_rows, selected_rows, width = shape
    generator = torch.Generator().manual_seed(203)
    raw = torch.randint(
        -32768,
        32768,
        (batch, source_rows, width),
        generator=generator,
        dtype=torch.int16,
    )
    # Include signed zero, infinities and distinct NaN encodings explicitly.
    raw.flatten()[:8] = torch.tensor(
        [0, -32768, 31744, 32257, 32341, 32640, 32641, -1], dtype=torch.int16
    )
    indices = torch.randint(
        source_rows, (batch, selected_rows), generator=generator, dtype=torch.int32
    )
    indices[:, :3] = torch.tensor([source_rows - 1, 0, 0], dtype=torch.int32)
    expected = torch.from_numpy(indexed_rows(raw.numpy(), indices.numpy()))
    guard = 4  # Eight-byte offset: word-aligned, but not a 32-byte aligned base.
    source_backing = torch.full(
        (raw.numel() + 2 * guard,), 12345, dtype=torch.int16, device=device
    )
    source_backing[guard:-guard].copy_(raw.flatten().to(device))
    source = source_backing[guard:-guard].view(dtype).reshape(raw.shape)
    output_backing = torch.full(
        (batch * selected_rows * width + 2 * guard,),
        12345,
        dtype=torch.int16,
        device=device,
    )
    output = (
        output_backing[guard:-guard].view(dtype).reshape(batch, selected_rows, width)
    )
    assert source.is_contiguous() and output.is_contiguous()
    assert (
        source.data_ptr() % (word_bits // 8)
        == output.data_ptr() % (word_bits // 8)
        == 0
    )
    kernel = make_word_gather(*shape, str(dtype).split(".")[-1], word_bits, 128)
    device_indices = indices.to(device)
    for workers in (1, 3):
        for sentinel in (12345, -23456):
            output_backing.fill_(sentinel)
            kernel[None, workers](source, device_indices, output)
            torch.npu.synchronize()
            assert torch.equal(output.cpu().view(torch.int16), expected)
            assert (output_backing[:guard].cpu() == sentinel).all()
            assert (output_backing[-guard:].cpu() == sentinel).all()
            assert torch.equal(source.cpu().view(torch.int16), raw)
            assert (source_backing[:guard].cpu() == 12345).all()
            assert (source_backing[-guard:].cpu() == 12345).all()
            assert torch.equal(device_indices.cpu(), indices)


@pytest.mark.parametrize("word_bits,width", [(32, 7), (64, 6)])
def test_partial_words_are_rejected_before_compilation(word_bits, width):
    with pytest.raises(ValueError, match="whole number of words"):
        make_word_gather(2, 17, 23, width, word_bits=word_bits)
