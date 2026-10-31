# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""``vf.load_align`` / ``vf.store_align``：两个 API 名，一整个模式面。

**写法：原始。** `vf.*` 没有语法糖。前置样例是 [`vf_entry`](../vf_entry/test.py)。

## 一、为什么这一例的分量不是「2 个 API」

`LoadDist` 有 **23** 个取值，`StoreDist` 有 **18** 个（`__members__` 取；`dir()` 会多算 `name` /
`value` 两个描述符，别用它数），再加 `DataCopyMode` 的非连续搬运、双搬入
（一条指令读 2*VL 拆成两个寄存器）、双搬出（两个寄存器交错写 2*VL）。多数取值是同一机制的位宽变体
（`_B8` / `_B16` / `_B32`），所以这一例按**机制**组织。

这也是为什么 ascriptor 那边看起来「有我们没有的接口」：它的 `ub_to_reg` / `load_cont` /
`load_interleave` / `store_cont` / `store_interleave` 全是这两个 API 的模式。设计文档 §八点四 里
记了这个认错的过程。

## 二、实测能用的机制（FP32）

每一条都是 2026-09-27 在 A5 上逐个试出来的：

| 机制 | `dist=` | 结果 |
| --- | --- | --- |
| 逐元素 | `NORM`（默认） | `1,2,3,...,64` |
| 广播 | `BRC` | `1,1,1,...` —— lane 0 铺满 |
| 解包 | `UNPK` | `1,0,2,0,3,0,...` —— 读 VL/2 个，零填展开 |
| 块复制 | `BLK` | `1..8,1..8,...` —— 一个 32B block 铺满 |
| **双搬入** | `DINTLV_B32` | 返回**两个**寄存器：`1,3,5,...` 和 `2,4,6,...` |
| 只写首元素 | `FIRST_ELEMENT`（store） | 只有 lane 0 被写 |
| **双搬出** | `INTLV_B32`（store） | `1,65,2,66,...` —— 要传**两个** src |
| **非连续** | `data_copy_mode=DATA_BLOCK_COPY` | 隔 `block_stride` 个 block 取一个 |

## 三、三条实测出来的坑

### 3.1 `US` / `DS` 在 32 位下静默给错

枚举里只有 `US` / `US_B8` / `US_B16` 和 `DS` / `DS_B8` / `DS_B16`——**没有 `*_B32`**。也就是说
上采样/下采样只在 8 位和 16 位宽存在。

在 FP32 上用 `DS`：**不报错，拿到一块零**。验证 6 把这件事钉成一条会失败的测试——哪天行为变了
（改成报错也好，改成可用也好），它会立刻告诉我们。

**这是本例最值得记的一条**：其余机制用错了会给出看得出来的怪结果，这一条给出的是「看起来像没算」
的零，最容易被当成别处的问题。

### 3.2 `block_stride` 必须作关键字传

`vf.load_align` 的 codegen **按参数个数分派**（`backend_cce_vf_ops.cpp:1016`）：四个参数的形式是
**双搬入**（两个 dst + ptr + offset）。把 `block_stride` 当位置参数传会凑成四个，撞上那条路径并报：

```
vf.load_align 4-arg (de-interleave) form does not support data_copy_mode
```

注意官方文档说 `block_stride`「可作位置参数（第 4 个）」——那是对 `store_align` 说的。load 侧受
双搬入形式挤占。**这条是实测踩出来的，文档读不出来。**

### 3.3 DataBlock 模式改变两个参数位置的含义

- `offset` 的位置换成一个**选择用的 mask_reg**：某个 DataBlock 在 mask 里对应的 32 bit 有任意一位
  为 1 就搬；全 0 则不搬，**且目的位置不更新，即使 Tile 越界也不报错**。
- 第 4 个位置变成 `block_stride`（单位是 block 而不是元素）。

「越界不报错」那半句值得留意：它把一个本该是错误的形状变成了静默跳过。

## 四、两个掩码，作用在不同层

这一例和 `vf_arith` 放在一起容易混：

| | 掩码管什么 | 未选中的位置 |
| --- | --- | --- |
| 算子的 `preg`（`vf_arith`） | 哪些车道参与计算 | 目的**寄存器**里置 0（`MergeMode.ZEROING`） |
| `store_align` 的 `preg`（本例） | 哪些元素写进 Tile | **Tile 保留原值**，不被触碰 |

验证 4 用哨兵值把后者变成可观测的。

## 五、如何独立验证

1. 四种 load 分布（NORM / BRC / UNPK / BLK）逐个与参考逐位相等。
2. 双搬入返回的两个寄存器分别等于奇偶抽取。
3. 双搬出等于两个寄存器交错。
4. `store_align` 的 `NORM` 只写 preg 选中的 32 个，其余保持哨兵；`FIRST_ELEMENT` 只写 lane 0。
5. DataBlock 模式在 `block_stride=2` 下等于隔块抽取——**用 stride=2 而不是 1**，因为 stride=1
   等于连续，测不出非连续。
6. `DS` 在 FP32 下给全零（钉住当前的错误行为）。
7. 把 `block_stride` 当位置参数传，必须被拒且消息里点名 "4-arg"。

第 5 条那个「stride 不能取 1」和第 6 条的「钉错误行为」是这一例里两个刻意的选择，不是随手写的。

## 六、这一例不讲什么

- **非对齐搬运**在 ``vf_unalign``：`load_align` 要求偏移对齐，变长紧密输出是另一套机制
  （见设计文档 §九）。
- `dist=` 的**位宽绑定**机制在 [`vf_align_widths`](../vf_align_widths/test.py)
  （2026-09-29 板证）：UNPK 保序宽化（FP16 源→FP32 的标准入口）、PACK_B32
  窄化直出、粗名 PACK 的位截断陷阱。**寄存器到寄存器**的同名算子（`vf.pack` /
  `vf.interleave` 本体等十个）在 `vf_lane_layout`——同一批词，两个层。
- `post_update=True`（搬完地址自动累进）和 `create_addr_reg`（地址寄存器）在 ``vf_unalign`` /
  ``vf_gather_scatter``。
- `E2B` 也在 ``vf_align_widths``：本例 FP32 下观察到的「每元素重复 8 次」当时被解读
  为「不是它的语义」——后来 FP16 板证（每元素重复 16 次）证明**块广播就是它的
  语义**（element-to-block），官方文档「16→32 扩展」的描述是错的（open-items
  上报候选）。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §八。
"""

import numpy as np
import pytest
import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

from . import reference as ref
from .kernel import (
    N,
    VL_FP32,
    align_datablock,
    align_load_dintlv,
    align_load_dist,
    align_store_dist,
    align_store_intlv,
    align_width_trap,
)

FP32 = pl.DT_FP32


def _dev(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _outs(device, count, fill=0.0):
    return [
        torch.full([1, N], fill, device=device, dtype=torch.float32)
        for _ in range(count)
    ]


def test_the_four_load_distributions_match_the_reference(device):
    """验证 1。NORM / BRC / UNPK / BLK。

    只比前 VL 个元素：分布模式作用在一个寄存器上，VL 之后的位置这一条不关心。
    """
    x = ref.make_input(N)
    outs = _outs(device, 4)
    align_load_dist(_dev(x, device), *outs)
    torch.npu.synchronize()

    want = [
        ref.load_norm(x, N),
        ref.load_brc(x, N),
        ref.load_unpk(x, N),
        ref.load_blk(x, N),
    ]
    names = ("NORM", "BRC", "UNPK", "BLK")
    for name, got, expected in zip(names, outs, want):
        np.testing.assert_array_equal(
            got.cpu().numpy()[0, :VL_FP32],
            expected[0, :VL_FP32],
            err_msg=f"LoadDist.{name}",
        )


def test_the_de_interleaving_load_returns_two_registers(device):
    """验证 2。一条指令读 2*VL，奇偶分别进两个寄存器。"""
    x = ref.make_input(N)
    even_t, odd_t = _outs(device, 2)
    align_load_dintlv(_dev(x, device), even_t, odd_t)
    torch.npu.synchronize()

    want_even, want_odd = ref.load_dintlv(x, N)
    np.testing.assert_array_equal(
        even_t.cpu().numpy()[0, :VL_FP32], want_even[0, :VL_FP32]
    )
    np.testing.assert_array_equal(
        odd_t.cpu().numpy()[0, :VL_FP32], want_odd[0, :VL_FP32]
    )


def test_the_interleaving_store_writes_two_registers_alternately(device):
    """验证 3。双搬出：写满 2*VL。这一条比整块，因为交错的结果占满两个 VL。"""
    x = ref.make_input(N)
    out = _outs(device, 1)[0]
    align_store_intlv(_dev(x, device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), ref.store_intlv(x, N))


def test_the_store_mask_leaves_unselected_positions_untouched(device):
    """验证 4。`NORM` 只写 preg 选中的 32 个；`FIRST_ELEMENT` 只写 lane 0。

    两个输出先被填成哨兵再 load 进 UB，所以「未选中的位置保留原值」是可观测的——这正是
    `store_align` 的掩码与算子的 `MergeMode.ZEROING` 的区别。
    """
    x = ref.make_input(N)
    o_norm, o_first = _outs(device, 2, fill=ref.SENTINEL)
    align_store_dist(_dev(x, device), o_norm, o_first)
    torch.npu.synchronize()

    np.testing.assert_array_equal(
        o_norm.cpu().numpy(), ref.store_norm(x, N), err_msg="NORM"
    )
    np.testing.assert_array_equal(
        o_first.cpu().numpy(), ref.store_first_element(x, N), err_msg="FIRST_ELEMENT"
    )


def test_the_datablock_mode_skips_blocks_by_stride(device):
    """验证 5。`block_stride=2`：输出第 k 个 block 来自源第 2k 个 block。

    **stride 取 2 不是 1。** stride=1 等于连续搬运，结果和 NORM 一样，测不出非连续。
    """
    x = ref.make_input(N)
    out = _outs(device, 1)[0]
    align_datablock(_dev(x, device), out)
    torch.npu.synchronize()

    want = ref.load_datablock(x, N)
    np.testing.assert_array_equal(out.cpu().numpy()[0, :VL_FP32], want[0, :VL_FP32])
    # 和连续搬运真的不同，否则这条测试是空的
    assert not np.array_equal(want[0, :VL_FP32], x[0, :VL_FP32]), "stride 没起作用"


def test_downsample_on_32_bit_silently_gives_zeros(device):
    """验证 6。钉住一个**当前的错误行为**：`LoadDist.DS` 在 FP32 上不报错，给全零。

    枚举里没有 `DS_B32`——上采样/下采样只在 8 位和 16 位宽存在。这条测试不是在说「零是对的」，
    是在说「现在是零」：行为一旦变成报错或变成可用，它会立刻失败，那时回来改。
    """
    x = ref.make_input(N)
    out = _outs(device, 1)[0]
    align_width_trap(_dev(x, device), out)
    torch.npu.synchronize()

    got = out.cpu().numpy()[0, :VL_FP32]
    np.testing.assert_array_equal(got, ref.load_ds_on_fp32(N)[0, :VL_FP32])
    # 说清「静默」：没有任何异常抛出，而结果显然不是下采样该有的样子
    assert not np.array_equal(got, x[0, slice(0, VL_FP32 * 2, 2)][:VL_FP32]), (
        "如果这里相等，说明 DS 其实生效了"
    )


def test_block_stride_as_a_positional_argument_is_refused():
    """验证 7。`block_stride` 位置传会撞上双搬入形式。

    不需要卡——这是 codegen 期的检查，但 codegen 要走到编译。所以这里只断言「被拒」，
    用解析 + 编译触发。
    """

    @pl.vector_function
    def _bad(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        select = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        # block_stride 当第 3 个位置参数 -> 凑成四参数 -> 被当成双搬入
        r = vf.load_align(
            src, select, 2, data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY
        )
        vf.store_align(dst, r, preg, 0)

    @pl.jit(auto_mutex=True, name="align_bad_stride")
    def bad(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
        tt = pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)
        ga = pl.make_tile_group(type=tt, addrs=[0], mutex_ids=[0])
        go = pl.make_tile_group(type=tt, addrs=[512], mutex_ids=[1])
        with pl.section_vector():
            ta = ga.next()
            to = go.next()
            pl.load(ta, a, [0, 0])
            _bad(ta, to)
            pl.store(out, to, [0, 0])

    x = ref.make_input(N)
    with pytest.raises(RuntimeError) as raised:
        bad(torch.from_numpy(x), torch.zeros([1, N]))

    text = str(raised.value)
    assert "4-arg" in text or "de-interleave" in text or "INVALID_ARGUMENT" in text, (
        text
    )
