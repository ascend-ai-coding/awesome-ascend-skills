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

Contiguous, interleaved and distributed transfers are modes of the native
load/store API families. Inspect `dist` and copy-mode parameters before
concluding that a separate transfer API is missing.

## 二、搬运机制

以下机制均由本例在 A5 上逐项验证：

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
| MaskReg 位采样 | `DS` | 512bit 位图抽取偶数位，写出32字节 |

## 三、三条实测出来的坑

### 3.1 通用 `DS` 采样的是 MaskReg 位图

通用 `US` / `DS` 选择 MaskReg 搬入路径，采样的是原始位图，不是浮点元素。MaskReg 写出只有32字节；不能把其余未写的 UB 字节当成完整 FP32 输出，也不能假设它们为零。

本例只验证通用 `DS` 的 MaskReg 路径：原始位图逐字节对照独立位运算参考，写出之外的字节保留哨兵；不把位图解释为 FP32 数值，也不推断其他采样枚举的可用性。

### 3.2 `block_stride` 必须作关键字传

`vf.load_align` 的 `block_stride`、`repeat_stride`、`data_copy_mode` 等可选参数是 keyword-only。
把 `block_stride` 当位置参数传会在参数检查阶段以 `ValueError` 拒绝，消息指出
`vf.load_align() too many positional arguments`。双搬入通过 `dist=DINTLV_B32` 选择，不能把
可选的位置参数当成普通搬入的步长。

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
6. 通用 `DS` 的位图采样通过独立参考、尾部哨兵及输出守卫。
7. 把 `block_stride` 当位置参数传，必须被参数检查拒绝并指出位置参数过多。

第 5 条使用非单位 stride，第 6 条同时验证写入内容和未写范围，分别排除空测与未初始化数据。

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
    align_load_mask_ds,
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


@pytest.mark.parametrize("pattern", ["ramp", "alternating"])
def test_mask_downsample_preserves_bits_and_the_unwritten_tail(device, pattern):
    """通用 DS 按bit采样，MaskReg写出之外的输出和守卫不得改变。"""
    if pattern == "ramp":
        source = np.arange(N, dtype=np.uint8).reshape(1, N)
    else:
        source = np.resize(
            np.array([0x00, 0xFF, 0xAA, 0x55, 0xF0, 0x0F, 0xCC, 0x33], dtype=np.uint8),
            (1, N),
        )
    x = _dev(source, device)
    backing = torch.full((N + 128,), 0xE5, dtype=torch.uint8, device=device)
    out = backing[64:-64].view(1, N)
    for sentinel in (0xE5, 0x5A):
        out.fill_(sentinel)
        align_load_mask_ds(x, out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), ref.load_mask_ds(source, sentinel))
        np.testing.assert_array_equal(backing[:64].cpu().numpy(), 0xE5)
        np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 0xE5)
    np.testing.assert_array_equal(x.cpu().numpy(), source)


def test_block_stride_as_a_positional_argument_is_refused():
    """验证 7。参数检查拒绝用位置参数传 keyword-only 的 `block_stride`。"""

    @pl.vector_function
    def _bad(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        select = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        # block_stride 必须显式命名；这里故意传成位置参数。
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
    with pytest.raises(ValueError) as raised:
        bad(torch.from_numpy(x), torch.zeros([1, N]))

    text = str(raised.value)
    assert "vf.load_align() too many positional arguments" in text, text
    assert "block_stride" in text and "passed by keyword" in text, text
