# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四个接口：`gather` `scatter` `arange` `create_addr_reg`。

**写法：原始。** 前置样例是 [`vf_align`](../vf_align/test.py)（对齐搬运的基本形状）与
[`vf_lane_layout`](../vf_lane_layout/test.py)（`squeeze` 那一族的 lane 重排）。

机制的分层在 [`kernel.py`](kernel.py) 的模块 docstring 里——四个接口各自「谁给地址、
地址什么粒度」。**这里不重复，只演示和验证。**

## 一、这一例真正要钉的四件事

| # | 事实 | 错了会怎样 |
| --- | --- | --- |
| 1 | `gather` 按 `src` 类型分发两种形式，`preg` 一个必填一个不能传 | 签名报错，**不静默** |
| 2 | `DATA_BLOCK_LOAD` 的索引单位是**字节**且要 32B 对齐 | 静默给错结果 |
| 3 | `scatter` 的索引必须唯一，`gather` 允许重复 | 静默丢数据 |
| 4 | reg→reg 的越界索引回卷 `i % 64`，Tile→reg **没有这个承诺** | 越界读 UB |

第 2、3、4 条都是「文档里写着、运行期不查」，所以值得有测试。

## 二、`arange` 没有 step 参数

步长固定 `±1`，这是硬件特性（`vci` 指令）。要 `start + i*step` 就在后面接一条
`vf.muls`。**找不到 step 参数不是漏读。**

`DECREASE_ORDER` 下 **`start` 是最后一个 lane 的值**：`dst[i] = start + (VL-1-i)`。
两处文档对不上（正文说「最终值」，`_vf_api.py` 的公式说 `dst[i] = start - i`），
实测判官方正文赢——**公式看着更精确，反而是错的那一份**。已在 pypto 侧改掉 docstring。

## 二点五、kernel 体里调内联辅助函数只能用位置参数

`_tile_type(cols=2 * N)` 被解析期拒（`_call_parser.py:1438`：「only supports plain
positional arguments」）。`_tile_type(FP32, 1, 2 * N)` 才行。这条和
[`vf_arith`](../vf_arith/test.py) 记的「`pl.const` 第一个参数必须是字面量」、
「不能用列表推导」是同一类——**kernel 体是 DSL，不是 Python**。

## 三、`create_addr_reg` 的两个先例都是单轮 smoke

`vf_api` ST 的 kernel 81 / 92 都写 `pl.range(0, 1, 1)`，验的是「能编译」。
**单轮看不出偏移会不会累加**，所以本例跑两轮、输入两个寄存器长。

## 四、AddrReg 非对齐搬运的解析探针

`test_an_addr_reg_reaches_store_unalign_at_parse_time` 只检查组合是否进入 IR，
不宣称已验证硬件写入范围或数值。已有实现与文档冲突统一见 `references/api/vf.md`。

## 五、判据

`rtol=0, atol=0`。全是整数下标搬运，没有一步引入浮点误差。
索引一律用倒序或明显置换，**不用恒等映射**——恒等下「索引没被用上」和「索引对了」
给出同一个结果。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import pypto_pro as pp

from . import reference as ref
from .kernel import (
    BLOCKS,
    N,
    addr_reg_into_store_unalign,
    addr_reg_walk,
    arange_three_ways,
    gather_both_forms,
    gather_datablock,
    gather_duplicate_index,
    gather_reg_out_of_range,
    scatter,
)

_ALL_APIS = ["gather", "scatter", "arange", "create_addr_reg"]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _out(device, cols=N, dtype=torch.float32):
    return torch.zeros((1, cols), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# gather 的两种形式
# ---------------------------------------------------------------------------


def test_both_gather_forms_agree(device):
    """同一个倒序索引，Tile→reg 和 reg→reg 给出同一个结果。

    两种形式都对 + 互相一致，才说明「按 `src` 类型分发」这件事是透明的。
    """
    a, idx = ref.make_data(N), ref.make_reverse_index(N)
    o_tile, o_reg = _out(device), _out(device)
    gather_both_forms(_dev(a, device), _dev(idx, device), o_tile, o_reg)
    torch.npu.synchronize()

    want = ref.gather(a, idx)
    np.testing.assert_allclose(
        o_tile.cpu().numpy(), want, rtol=0, atol=0, err_msg="Tile→reg gather 不一致"
    )
    np.testing.assert_allclose(
        o_reg.cpu().numpy(), want, rtol=0, atol=0, err_msg="reg→reg gather 不一致"
    )


def test_the_gather_index_is_actually_used(device):
    """防空测：倒序索引下输出必须是倒序，不能等于输入。

    恒等映射验不到这件事——这就是参考里索引一律用置换的原因。
    """
    a, idx = ref.make_data(N), ref.make_reverse_index(N)
    o_tile, o_reg = _out(device), _out(device)
    gather_both_forms(_dev(a, device), _dev(idx, device), o_tile, o_reg)
    torch.npu.synchronize()

    got = o_tile.cpu().numpy().reshape(-1)
    assert got[0] == float(N), f"倒序后第 0 个应该是 {N}，实际 {got[0]}——索引没被用上"
    assert not np.array_equal(got, a.reshape(-1)), "输出等于输入，索引没起作用"


# ---------------------------------------------------------------------------
# DATA_BLOCK_LOAD
# ---------------------------------------------------------------------------


def test_datablock_gather_moves_whole_32_byte_blocks(device):
    """8 个 32B 块整体倒序：块内顺序不变，块之间倒过来。

    这个判据同时排除两种错法：索引没被用上（输出=输入），以及按元素而不是按块
    （输出会是整体倒序 `64..1`，而不是 `57..64, 49..56, ...`）。
    """
    a, idx = ref.make_data(N), ref.make_block_index(N)
    out = _out(device)
    gather_datablock(_dev(a, device), _dev(idx, device), out)
    torch.npu.synchronize()

    want = ref.gather_datablock(a, idx)
    np.testing.assert_allclose(
        out.cpu().numpy(), want, rtol=0, atol=0, err_msg="DATA_BLOCK_LOAD gather 不一致"
    )

    got = out.cpu().numpy().reshape(-1)
    per_block = N // BLOCKS
    assert got[0] == float(N - per_block + 1), (
        f"第一块应该是原来的最后一块（从 {N - per_block + 1} 开始），实际 {got[0]}"
    )
    # 块内顺序保持：第一块应该是连续递增的
    first_block = got[:per_block]
    assert np.array_equal(
        first_block, np.arange(N - per_block + 1, N + 1, dtype=np.float32)
    ), (
        f"块内顺序应当保持不变，实际 {first_block.tolist()}——可能是按元素而不是按块 gather"
    )


# ---------------------------------------------------------------------------
# scatter，以及重复索引的不对称
# ---------------------------------------------------------------------------


def test_scatter_writes_to_the_indexed_positions(device):
    """`scatter` 用倒序（唯一）索引：`dst[index[i]] = src[i]`。"""
    a, idx = ref.make_data(N), ref.make_reverse_index(N)
    out = _out(device)
    scatter(_dev(a, device), _dev(idx, device), out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        out.cpu().numpy(), ref.scatter(a, idx), rtol=0, atol=0, err_msg="scatter 不一致"
    )


def test_gather_accepts_duplicate_indices_as_a_broadcast(device):
    """**重复索引在 `gather` 下是广播，有定义。**

    同一份索引在 `scatter` 下无定义（「仅保留其中一个」，保留哪个没规定），
    所以这一对只验 gather 那一半——**给 scatter 写一个断言等于编造一个契约**。
    """
    a, idx = ref.make_data(N), ref.make_duplicate_index(N)
    out = _out(device)
    gather_duplicate_index(_dev(a, device), _dev(idx, device), out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        out.cpu().numpy(),
        ref.gather(a, idx),
        rtol=0,
        atol=0,
        err_msg="重复索引下 gather 的结果不是广播",
    )
    got = out.cpu().numpy().reshape(-1)
    assert got[0] == got[2] == 1.0 and got[1] == got[3] == 2.0, (
        f"前四个应该是 1,2,1,2（广播），实际 {got[:4].tolist()}"
    )


def test_reg_to_reg_gather_wraps_out_of_range_indices(device):
    """reg→reg 形式：索引 `64..127` 回卷成 `0..63`，输出等于输入。

    **这个承诺只对 reg→reg 有效。** Tile→reg 那边文档写的是「必须在 Tile 有效地址
    范围内」，没有回卷承诺，所以本例不去试 Tile→reg 的越界。
    """
    a, idx = ref.make_data(N), ref.make_out_of_range_index(N)
    out = _out(device)
    gather_reg_out_of_range(_dev(a, device), _dev(idx, device), out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        out.cpu().numpy(), a, rtol=0, atol=0, err_msg="越界索引没有按 i % 64 回卷"
    )


# ---------------------------------------------------------------------------
# arange
# ---------------------------------------------------------------------------


def test_arange_increase_order_is_start_plus_i(device):
    """默认方向：`dst[i] = start + i`，步长 1。"""
    outs = [_out(device, dtype=torch.int32) for _ in range(3)]
    arange_three_ways(*outs)
    torch.npu.synchronize()

    np.testing.assert_array_equal(
        outs[0].cpu().numpy(), ref.arange_inc(N), err_msg="arange 递增不一致"
    )


def test_arange_decrease_order_ends_at_start(device):
    """`DECREASE_ORDER`：**`start` 是最后一个 lane 的值**，`dst[i] = start + (VL-1-i)`。

    两处文档对不上，实测判的是官方正文赢：

    | 来源 | 说法 | 实测 |
    | --- | --- | --- |
    | 官方正文 | 「以传入的 start 为**最终值**」 | ✓ |
    | `_vf_api.py` docstring | `dst[i] = start - i` | ✗ |

    `start=100`、VL=64 → `163, 162, ..., 100`。**公式看着更精确，反而是错的那一份**——
    已在 pypto 侧改掉 docstring。
    """
    outs = [_out(device, dtype=torch.int32) for _ in range(3)]
    arange_three_ways(*outs)
    torch.npu.synchronize()

    got = outs[1].cpu().numpy()
    np.testing.assert_array_equal(
        got, ref.arange_dec(N), err_msg="arange 递减的排布不是 start + (VL-1-i)"
    )
    flat = got.reshape(-1)
    assert flat[-1] == 100, f"最后一个 lane 应该正好是 start=100，实际 {flat[-1]}"
    assert flat[0] == 100 + N - 1, f"第 0 个 lane 应该是 {100 + N - 1}，实际 {flat[0]}"


def test_a_non_unit_step_needs_a_following_muls(device):
    """步长 4 = `arange` + `muls`。**`arange` 自己没有 step 参数。**"""
    outs = [_out(device, dtype=torch.int32) for _ in range(3)]
    arange_three_ways(*outs)
    torch.npu.synchronize()

    inc, step4 = outs[0].cpu().numpy(), outs[2].cpu().numpy()
    np.testing.assert_array_equal(
        step4, inc * 4, err_msg="arange + muls 没给出步长 4 的序列"
    )
    assert step4.reshape(-1)[1] == 4, (
        f"步长 4 的第 1 项应该是 4，实际 {step4.reshape(-1)[1]}"
    )


# ---------------------------------------------------------------------------
# create_addr_reg
# ---------------------------------------------------------------------------


def test_the_addr_reg_offset_accumulates_across_iterations(device):
    """**这一例的重点**：两轮循环，偏移每轮加 64 个元素，输入输出逐位相同。

    输入 128 个元素（两个寄存器长）。如果偏移没累加，第 1 轮会重搬 `[0:64]`，
    输出后半段就是前半段的副本——下面第二条断言抓的就是这个。
    """
    a = ref.make_data(2 * N)
    out = _out(device, cols=2 * N)
    addr_reg_walk(_dev(a, device), out)
    torch.npu.synchronize()

    got = out.cpu().numpy()
    np.testing.assert_allclose(
        got, a, rtol=0, atol=0, err_msg="两轮搬完之后输出应当等于输入"
    )
    flat = got.reshape(-1)
    assert not np.array_equal(flat[:N], flat[N:]), (
        "输出后半段等于前半段——偏移没有跨轮累加，两轮搬的是同一段"
    )


def test_an_addr_reg_reaches_store_unalign_at_parse_time():
    """**只解析，不执行。** AddrReg 当 `store_unalign` 的 `stride`。

    官方 store_unalign 页面和后端已有该路径；本测试只检查解析层接受。
    不上板，因此没有实际写入范围和数值结论；未测不等于写入不可预测。
    """
    program, _ = addr_reg_into_store_unalign.to_kernel_def().parse_target_program(
        pp.ir.SectionKind.Vector
    )
    text = pp.ir.python_print(program)

    assert "@vf.create_addr_reg(" in text, "IR 里没有 create_addr_reg"
    assert "@vf.store_unalign(" in text, "IR 里没有 store_unalign"


# ---------------------------------------------------------------------------
# 覆盖检查
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 4 个接口每个都得在 kernel 源码里出现。"""
    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
