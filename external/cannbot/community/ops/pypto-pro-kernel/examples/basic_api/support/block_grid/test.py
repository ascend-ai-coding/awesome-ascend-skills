# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""核／子核的编号与总数，以及它和 SIMT 那套编号的关系。

**写法：原始。**

## 一、四个 API

| API | 是什么 | 实测范围 |
| --- | --- | --- |
| ``pl.get_block_idx()`` | 当前 AI 核（block）的号 | `0 .. block_num - 1` |
| ``pl.get_block_num()`` | **实际**启动的 block 数 | 主机侧给的 n；不给启动括号时 56 |
| ``pl.get_subblock_idx()`` | 子核号 | 纯 vector kernel 恒 0；混合 kernel 0 / 1 |
| ``pl.get_subblock_num()`` | 每核子核数 | 纯段 kernel **1**；混合 kernel **2** |

四个都只能在 `@pl.jit` 体里调，进不了 SIMT 函数（验证 6）。

## 二、如何独立验证

1. ``make_vector_reporter(STRIDE)``：`block_idx` 恰好是 `0..n-1` 的排列，每个 block 看到的
   `block_num` 都等于 n。n 取 1 / 4 / 8 / 16。
2. 不带启动括号：`block_num > 1`（这台机器 56）。
3. ``cube_only_reporter`` 与纯 vector 版：`subblock_num == 1`、`subblock_idx == 0`。
4. ``mixed_reporter``：`subblock_num == 2`，且 `subblock_idx == block_idx % 2`。
5. ``both_numbering_schemes``（单 block）：`pl.get_block_idx() == pl.simt.block_idx().x`,
   `pl.get_block_num() == pl.simt.grid_dim().x`。
6. ``kernel_api_inside_a_simt_function``：解析期拒。
7. 紧凑跨距那一组：留下来的数据自洽（**不断言留下几个**，见 §五）。

验证 1 用「是排列」而不是「每个值都在范围内」：**后者放过重号和缺号**,
而重号恰好是最可能出的错（两个 block 拿到同一个编号）。

## 三、`subblock_num` 取决于 kernel 有没有 cube 段（实测）

| kernel | `subblock_num` | `subblock_idx`（4 个 block） |
| --- | ---: | --- |
| 只有 `section_vector` | 1 | 0, 0, 0, 0 |
| 只有 `section_cube` | 1 | 0, 0, 0, 0 |
| **两段都有** | **2** | **0, 1, 0, 1** |

docstring 原话是「Returns 1 on AIC binaries, `get_subblockdim()` on AIV binaries」。
**实测印证了它，但从原话读不出「加一段 cube 就会变」**——那句话讲的是二进制类型,
而「这个 kernel 编成哪种二进制」取决于它有哪些 section。

后果：**任何用 `subblock_idx` 分工的代码，在 kernel 多出一个 cube 段的那天行为就变了。**
纯 vector 时 `subblock_idx` 恒 0，所以「按子核分两半」的代码在纯 vector kernel 里
是让所有 block 都走同一半——而且不报错。

混合 kernel 里 `block_idx` 仍然是 `0..n-1`（n 是主机侧给的数），子核号是 `block_idx % 2`。
所以那 n 个 block 是 **AIV worker**，两个共享一个 AIC。

## 四、两套编号报的是同一个数

| | kernel 侧 | SIMT 侧 |
| --- | --- | --- |
| 块号 | `pl.get_block_idx()` | `pl.simt.block_idx().x` |
| 网格 | `pl.get_block_num()` | `pl.simt.grid_dim().x` |

实测（`blocks=1` 和 `blocks=4`）两侧**完全一致**。

**这一条纠正了 ``simt/simt_launch`` §六 里的一句话**：那里写的是「两套编号不是一回事」,
实测它们报同一个数。真正的区别是**能在哪儿调**——`pl.get_block_idx()` 在 SIMT 体里
被解析期拒（验证 6），`pl.simt.block_idx()` 是 SIMT 侧的入口。作用域不同，语义相同。

验证 5 只在 `blocks=1` 时判：多 block 时 SIMT 那条链的 `pl.store` 都写 `[0, 0]`,
读回来是谁最后写赢（``simt_launch`` §四 量过这个坑）。

## 五、GM 标量写回的间距对照

这里的写入均为 `pl.setval` 到 GM。间距与 dcci 是本诊断的自变量，不是其他 kernel
的张量布局规定，更不是 DMA 输出必须填充的依据。当前事实与适用范围只在
[GM 标量小写入](../../../../references/api/support.md#7-small-gm-scalar-writes-from-multiple-blocks)
维护；本测试保留以下判据，竞争路径不固定要求某个失败数量或赢家编号。

| 情形 | 判据 |
| --- | --- |
| 跨距 64 B，`blocks` = 1/4/8/16 | **严格**：编号是排列、四个量全对 |
| 跨距 16 B，`blocks` = 1 | **严格**：没有竞争，数据必须完整 |
| 跨距 16 B，`blocks` = 8 | **只断言留下来的数据自洽**，不断言留下几个 |

最后一行按 `references/verification/board-runs.md` §五点十：判据的严格程度不该超过被测量的
可重复性。个数记在文档里，不写进 assert——10 次里出过一次 3，写 `== 2` 会红对一半。

**这一组同时补上了 ``datamove/mutex_dcci`` §五 的空缺**：那里说 `dcci` 的效果
「要在另一个观察者看得见差别的场景里才有意义，那需要跨核……的构造，不是这一章的题目」。
这就是那个构造，而结论是**在这个构造里 `dcci` 也没有可观测的效果**。

## 六、这一例不讲什么

- **SIMT 的线程编号**——`thread_idx` / `block_dim` / `linear_thread_idx` 在
  ``simt/simt_launch``。这里只用 SIMT 的**块**号做对照。
- **按核分工的实际写法**——把工作按 `block_idx` 切开、用 `block_num` 做跨距,
  是 `examples/matmul/` 那些样例的题目。这一例只钉编号本身的性质。
- **`get_block_num()` 小于主机侧请求的情形**——docstring 说 stream 的核数限制会让它变小,
  **没测**：要设 stream 的核数限制，而 `pypto/` 的 `core_limits/` ST 组当前 18 条全红
  （见 `references/diagnostics/regressions.md` §三）。**这是明说跳过。**
- **`subblock_idx` 真的对应两个物理 AIV 吗**——只量到「混合 kernel 里它按 block 交替」,
  没有独立手段确认那两个号对应不同的物理单元。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 四节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import NotSupported

from .kernel import (
    N,
    QUANTITIES,
    ROW_BLOCK_IDX,
    ROW_BLOCK_NUM,
    ROW_SUBBLOCK_IDX,
    ROW_SUBBLOCK_NUM,
    SIMT_ROW_BLOCK_IDX,
    SIMT_ROW_GRID_DIM,
    SIMT_THREADS,
    SLOTS,
    STRIDE,
    TIGHT_STRIDE,
    both_numbering_schemes,
    cube_only_reporter,
    kernel_api_inside_a_simt_function,
    make_tight_reporter_with_dcci,
    make_vector_reporter,
    mixed_reporter,
)
from .reference import (
    MIXED_SECTION_SUBBLOCKS,
    PURE_SECTION_SUBBLOCKS,
    is_a_permutation,
    is_self_consistent,
    reported_rows,
    surviving_blocks,
)

BLOCK_COUNTS = [1, 4, 8, 16]
"""测哪些 block 数。**1 是必须的**（很多判据只在单 block 时可判），16 是为了让
「跨距够宽时全留」这条有分量——8 个和 16 个都全留比只测 8 个强。"""


def _run_reporter(kernel, device, blocks, stride):
    out = torch.full((SLOTS * stride,), -1, dtype=torch.int32, device=device)
    if blocks is None:
        kernel(out)
    else:
        kernel[None, blocks](out)
    torch.npu.synchronize()
    return out.cpu().numpy()


def _run_mixed(device, blocks):
    a = np.arange(N, dtype=np.float32).reshape(1, N)
    out = torch.zeros((1, N), dtype=torch.float32, device=device)
    ids = torch.full((SLOTS * STRIDE,), -1, dtype=torch.int32, device=device)
    mixed_reporter[None, blocks](torch.from_numpy(a).to(device), out, ids)
    torch.npu.synchronize()
    return ids.cpu().numpy(), out.cpu().numpy(), a


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# 一、编号的结构性质
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", BLOCK_COUNTS)
def test_the_block_indices_are_a_permutation(device, blocks):
    """验证 1。`block_idx` 恰好是 `0..n-1` 的一个排列，每个 block 的 `block_num` 都是 n。

    **「是排列」比「都在范围内」强**：后者放过重号和缺号，而重号正是最可能出的错
    （两个 block 拿到同一个号）。
    """
    flat = _run_reporter(make_vector_reporter(STRIDE), device, blocks, STRIDE)
    rows = reported_rows(flat, STRIDE, blocks, QUANTITIES)
    assert is_a_permutation(rows[:, ROW_BLOCK_IDX], blocks), (
        f"blocks={blocks}: block_idx 读到 {rows[:, ROW_BLOCK_IDX].tolist()}，不是 0..{blocks - 1} 的排列"
    )
    np.testing.assert_array_equal(rows[:, ROW_BLOCK_NUM], np.full(blocks, blocks))


def test_without_a_launch_bracket_the_grid_is_the_whole_card(device):
    """验证 2。不带启动括号 → `block_num > 1`（这台机器 56）。

    **只断言 `> 1`**：核数是机器相关的，基线数字会漂
    （`references/verification/board-runs.md` §六）。和 ``simt_launch`` 那条判据同一个写法。
    """
    flat = _run_reporter(make_vector_reporter(STRIDE), device, None, STRIDE)
    rows = flat.reshape(-1, STRIDE)
    written = [block for block in range(SLOTS) if rows[block][ROW_BLOCK_IDX] != -1]
    assert written, "一个 block 都没写成功"
    nums = {int(rows[block][ROW_BLOCK_NUM]) for block in written}
    assert len(nums) == 1, f"不同 block 看到的 block_num 不一致：{nums}"
    assert nums.pop() > 1, "不带启动括号应该解成当前 stream 的全量核数"


# ---------------------------------------------------------------------------
# 二、subblock 取决于 section 组成
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", [1, 4])
def test_a_single_section_kernel_has_one_subblock(device, blocks):
    """验证 3。纯 vector 和纯 cube 两个 kernel 都是 `subblock_num == 1`、`subblock_idx == 0`。

    两种单段各测一次：如果只测 vector，「1」可能被读成「vector 段就是 1」,
    而实际是「单段就是 1」。
    """
    for label, kernel in (
        ("vector", make_vector_reporter(STRIDE)),
        ("cube", cube_only_reporter),
    ):
        rows = reported_rows(
            _run_reporter(kernel, device, blocks, STRIDE), STRIDE, blocks, QUANTITIES
        )
        np.testing.assert_array_equal(
            rows[:, ROW_SUBBLOCK_NUM],
            np.full(blocks, PURE_SECTION_SUBBLOCKS),
            err_msg=f"{label} 单段 kernel 的 subblock_num",
        )
        np.testing.assert_array_equal(
            rows[:, ROW_SUBBLOCK_IDX],
            np.zeros(blocks, dtype=np.int32),
            err_msg=f"{label} 单段 kernel 的 subblock_idx",
        )


@pytest.mark.parametrize("blocks", [1, 4])
def test_a_mixed_kernel_has_two_subblocks(device, blocks):
    """验证 4，**这一例的第一个核心结论**。两段都有时 `subblock_num == 2`,
    而 `subblock_idx == block_idx % 2`。

    判据先确认这个 kernel 真在干活（vector 段那一趟搬运对得上），**再**看编号——
    否则「读到 2」也可能只是编译期的巧合。
    """
    ids, out, a = _run_mixed(device, blocks)
    np.testing.assert_array_equal(
        out, a, err_msg="混合 kernel 的 vector 段搬运本身错了"
    )
    rows = reported_rows(ids, STRIDE, blocks, QUANTITIES)
    np.testing.assert_array_equal(
        rows[:, ROW_SUBBLOCK_NUM], np.full(blocks, MIXED_SECTION_SUBBLOCKS)
    )
    np.testing.assert_array_equal(
        rows[:, ROW_SUBBLOCK_IDX],
        np.arange(blocks, dtype=np.int32) % MIXED_SECTION_SUBBLOCKS,
    )


def test_the_two_section_shapes_disagree_on_the_subblock_count(device):
    """验证 3 与 4 的**对比本身**：同一台机器、同一次会话，单段读 1、混合读 2。

    分开测两个形状各自对得上是一回事；**把「它们不同」写成一条判据是另一回事**——
    这一条才是那句「加一段 cube 就会变」的判据。
    """
    pure = reported_rows(
        _run_reporter(make_vector_reporter(STRIDE), device, 1, STRIDE),
        STRIDE,
        1,
        QUANTITIES,
    )
    mixed = reported_rows(_run_mixed(device, 1)[0], STRIDE, 1, QUANTITIES)
    assert int(pure[0][ROW_SUBBLOCK_NUM]) != int(mixed[0][ROW_SUBBLOCK_NUM]), (
        f"单段读到 {pure[0][ROW_SUBBLOCK_NUM]}、混合读到 {mixed[0][ROW_SUBBLOCK_NUM]} —— "
        f"实测应当分别是 {PURE_SECTION_SUBBLOCKS} 和 {MIXED_SECTION_SUBBLOCKS}"
    )


# ---------------------------------------------------------------------------
# 三、两套编号
# ---------------------------------------------------------------------------


def test_the_kernel_and_simt_numbering_agree(device):
    """验证 5。`pl.get_block_idx() == pl.simt.block_idx().x`,
    `pl.get_block_num() == pl.simt.grid_dim().x`。

    **只在单 block 时判**：多 block 时 SIMT 那条链的 `pl.store` 都写 `[0, 0]`。
    这条纠正了 ``simt_launch`` §六 的说法——两套 API 报同一个数，区别在能在哪儿调。
    """
    simt_out = torch.full((2, SIMT_THREADS), -1, dtype=torch.int32, device=device)
    kernel_out = torch.full((SLOTS * STRIDE,), -1, dtype=torch.int32, device=device)
    both_numbering_schemes[None, 1](simt_out, kernel_out)
    torch.npu.synchronize()
    simt = simt_out.cpu().numpy()
    kernel = kernel_out.cpu().numpy().reshape(-1, STRIDE)[0]

    assert (simt[SIMT_ROW_BLOCK_IDX] == simt[SIMT_ROW_BLOCK_IDX][0]).all(), (
        f"同一个 block 里各线程看到的 simt block_idx 不一致：{simt[SIMT_ROW_BLOCK_IDX].tolist()}"
    )
    assert int(simt[SIMT_ROW_BLOCK_IDX][0]) == int(kernel[ROW_BLOCK_IDX]), (
        f"simt block_idx={simt[SIMT_ROW_BLOCK_IDX][0]}，kernel get_block_idx="
        f"{kernel[ROW_BLOCK_IDX]}"
    )
    assert int(simt[SIMT_ROW_GRID_DIM][0]) == int(kernel[ROW_BLOCK_NUM]), (
        f"simt grid_dim={simt[SIMT_ROW_GRID_DIM][0]}，kernel get_block_num="
        f"{kernel[ROW_BLOCK_NUM]}"
    )


def test_a_kernel_scope_id_api_is_refused_inside_a_simt_function():
    """验证 6。`pl.get_block_idx()` 进不了 SIMT 体，即使它和 `pl.simt.block_idx()` 报同一个数。

    **这就是两套 API 的真正区别**：作用域，不是语义。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(kernel_api_inside_a_simt_function)
    assert "is not supported inside a SIMT function" in str(raised.value), str(
        raised.value
    )


# ---------------------------------------------------------------------------
# 四、往 GM 写自己的结果
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", BLOCK_COUNTS)
def test_a_64_byte_stride_keeps_every_blocks_data(device, blocks):
    """验证 7 前半。跨距 64 字节时每个 block 的四个量都完整留下。

    这是**严格判据**：实测 8 个和 16 个 block 各连跑 5 次全中。
    """
    flat = _run_reporter(make_vector_reporter(STRIDE), device, blocks, STRIDE)
    assert surviving_blocks(flat, STRIDE, blocks) == list(range(blocks)), (
        f"blocks={blocks}、跨距 {STRIDE * 4} 字节时仍有 block 的数据丢了："
        f"留下 {surviving_blocks(flat, STRIDE, blocks)}"
    )
    rows = reported_rows(flat, STRIDE, blocks, QUANTITIES)
    for block in range(blocks):
        assert is_self_consistent(rows[block], block, blocks, PURE_SECTION_SUBBLOCKS), (
            f"block {block} 那一段不自洽：{rows[block].tolist()}"
        )


def test_a_tight_stride_is_safe_when_only_one_block_runs(device):
    """验证 7 中段。跨距 16 字节但只有一个 block —— **没有竞争，数据必须完整**。

    这条把「跨距不够」和「寻址写错了」分开：寻址错的话单 block 也会错。
    """
    flat = _run_reporter(make_vector_reporter(TIGHT_STRIDE), device, 1, TIGHT_STRIDE)
    rows = reported_rows(flat, TIGHT_STRIDE, 1, QUANTITIES)
    assert is_self_consistent(rows[0], 0, 1, PURE_SECTION_SUBBLOCKS), rows[0].tolist()


def test_a_tight_stride_loses_data_and_dcci_does_not_help(device):
    """验证 7 后半。跨距 16 字节、8 个 block：**只断言留下来的数据自洽，不断言留下几个。**

    10 次观测留下 `2, 3, 2, 2, 2, 2, 2, 2, 2, 2` 个，具体是哪几个每次不同。
    按 `references/verification/board-runs.md` §五点十，判据不能比被测量更严——
    写 `== 2` 会在出 3 的那次红，而那次红什么也没说明。

    **`dcci` 版一起测**：实测它救不回来。两个 kernel 的判据相同,
    差别记在文档里（§五）——这是「加了 dcci 不会把对的弄坏」的形状,
    和 ``mutex_dcci`` 验证 4 同一个道理。
    """
    for label, kernel in (
        ("无 dcci", make_vector_reporter(TIGHT_STRIDE)),
        ("有 dcci", make_tight_reporter_with_dcci()),
    ):
        flat = _run_reporter(kernel, device, 8, TIGHT_STRIDE)
        rows = flat.reshape(-1, TIGHT_STRIDE)
        survived = surviving_blocks(flat, TIGHT_STRIDE, 8)
        assert survived, f"{label}: 一个 block 的数据都没留下，那连自洽都无从谈起"
        for block in survived:
            assert int(rows[block][ROW_BLOCK_NUM]) == 8, (
                f"{label}: block {block} 留下的是拼接数据——block_num 读到 "
                f"{rows[block][ROW_BLOCK_NUM]}，应为 8"
            )
