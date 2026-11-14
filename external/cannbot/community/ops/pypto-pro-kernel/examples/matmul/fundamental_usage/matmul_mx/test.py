# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MX 矩阵乘：一条约束只被验了槽 0，双缓冲下错了静默。

**解决什么.** MX（microscaling）把每 32 个连续元素（沿归约轴）配一个 **E8M0** 指数字节，
元素本身存成 **FP8** 尾数。硬件的 `mad_mx` **不把 scale 当显式操作数**——它靠 SFractal 布局
隐式找到，也就是靠**地址**。于是有了一条别处没有的约束，而且它是这条路上最容易错的东西。

**五条约束，谁查什么.**

| # | 约束 | 谁查 | 错了会怎样 |
| --- | --- | --- | --- |
| 1 | plane tile 地址=配对 L0 地址`>>4`，**逐槽** | `_check_mx_scale_tile` 只验槽0；`_check_mx_plane_slots` 逐槽验 | 修复前：**静默算错** |
| 2 | plane 形状 = `[M, K/32]` / `[K/32, N]` | `_check_mx_scale_tile`，无条件查 | 报错 |
| 3 | plane 的 dtype 是 E8M0、空间是 ScaleLeft / ScaleRight | 同上，无条件查 | 报错 |
| 4 | `K % 64 == 0` | `_check_mx_operands`（`block_ops.py:4437`） | 报错 |
| 5 | 布局固定：L1 scale 与 plane 在 A 侧 ZZ、B 侧 NN | `block_ops.py` 那张表 | 报错 |

**只有第 1 条曾经会静默，而且原因不是「检查缺失」。** `_check_mx_scale_tile` 把地址也查了，
推导和上面写的一致；但一个从多槽 group 取出的操作数，它的 `.type.memref.addr` 是**槽 0** 的
地址，于是检查比的是 `0x0 >> 4 == 0x0`——**验了基址，放过槽间距**。实测（隔离 `block_ops`
单独面对时）：基址写错报 `InvalidVal`，**槽间距写错静默通过**。标量 `addrs` 的语义保证基址
一定对、槽间距按自己的 size 递推，所以它命中的正是这个盲区。

（那条 *"Skip auto-allocated tiles or non-constant addresses"* 的注释确有其事，但实测在
`ut/pypto_pro` 2458 个用例里命中 0 次，不是本案的原因。扫描见
[diagnostics/check-coverage.md](../../../../references/diagnostics/check-coverage.md) §二.4。）

**那条地址规则有个按 dtype 变脸的陷阱.** plane tile 自己只有 128 字节，而它要的槽间距是
**数据 tile 字节数 >> 4**：

| 数据 dtype | 数据/块 | plane/块 | `addrs=<标量>` 给出的槽 1 | 硬件要的 | 结果 |
| --- | --- | --- | --- | --- | --- |
| FP8 | 4096 | 128 | `0x80` | `0x100` | **静默错，差一倍** |
| FP4 | 2048 | 128 | `0x80` | `0x80` | 恰好对 |

`make_tile_group(addrs=<标量>)` 的语义是「从这里开始按**自己的** tile 大小连续摆」，所以在
plane 组上给标量：FP8 错、FP4 对。**同一个写法在一个 dtype 上静默错、在另一个上正确。**
本样例因此把 plane 的地址写成显式地址表，并有一条测试把「标量形式会给出什么」算出来和正确值
比，直接把这个陷阱钉在纸面上。

**「静默」是实测的**（2026-09-26，同一份操作数、同一块卡，两个槽且基址正确，正落在盲区）：
正确地址下与参考**逐位相等**；
把 plane 换成标量 `addrs` 后 `max|diff|` 138.6（输出幅度上限 252.25），16384 个元素里 11143
个不对，**没有任何报错或警告**。症状不是「掉了一半」那种好认的形态——曾经猜错的那一半 scale
会变成零，一比就否掉了（错的输出离「只算第一个 K tile」的结果更远）。**它就是一个看起来合理
的错数。**

**两个槽是刻意的.** 一个槽时 plane 地址恒为 0，上面那条规则看不出来，也测不到。

**数值可以按逐位相等验.** 操作数取 E4M3 的「单位阶梯」（`E=7`，值 `±(1+M/8)`，16 个码全部
精确可表示）配 `{0.5, 1, 2}` 的 scale，于是每个乘积项是 `2^-8` 的倍数、绝对值不超过 14，
K=128 的和不超过约 1800，而 `1800 * 256 < 2^24`——**整条和在 FP32 里精确可表示**。所以参考
在 float64 上算、和 kernel 的 FP32 累加**顺序不同也应当逐位相同**，一点容差都不用给。
编码约定怎么核的见 [`reference.py`](reference.py)。

**如何独立验证.** 六条，前四条不需要卡：

1. **逐槽核对 plane 地址 == 数据地址 `>> 4`**。框架已经在 parser 侧查了这条，这里仍然自己查
   一遍——样例钉住的是不变量，不是某个框架版本的行为。
2. **算出标量 `addrs` 会给什么**，确认它和正确值不同——把陷阱变成一条会失败的断言，而不是
   一句注释。
3. **plane 形状是数据 tile 的函数**，从 IR 里读出来比。
4. **布局是 ZZ / NN**，同样从 IR 读。
5. **与参考逐位相等**（`rtol=0, atol=0`）。
6. **两遍逐位相同。**

来源：`pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_mx.py`（1607 行，覆盖
FP4 打包、batched、insert、转置、多核 atomic 等）。这里只取 **MXFP8 e4m3 × e4m3、精确形状、
多轮 K** 一条链——其余变体在那份 ST 里。

设计稿：[api/gemm-mx.md](../../../../references/api/gemm-mx.md)。本样例是它
§9.2 的第 2 步：先把不变量用手写 `pl.matmul_mx` 钉住，再谈语法糖。
"""

import re

import numpy as np
import torch

from .kernel import (
    GROUP_SIZE,
    K,
    M,
    N,
    PAIR,
    SCALE_K,
    TILE,
    _L0_BYTES,
    _PLANE_SLOT,
    matmul_mx,
)
from .reference import LADDER_VALUES, make_operands, matmul_reference

# ir.TileType([64, 64], ir.FP8E4M3FN, ir.MemRef(ir.MemorySpace.Left, 0, 4096), ...
_TILE = re.compile(
    r"_tg_(\w+?)_tiles_0:.*?ir\.TileType\(\[([^\]]*)\], ir\.(\w+), "
    r"ir\.MemRef\(ir\.MemorySpace\.(\w+), (\d+), (\d+)\)"
)
_CELL = re.compile(r"ir\.MemRef\(ir\.MemorySpace\.(\w+), (\d+), (\d+)\)")


def _declared_groups() -> dict:
    """从解析后的 IR 读每个 tile group 的 (空间, 形状, dtype, 每槽地址, 每块字节)。"""
    import pypto_pro as pp

    program, _ = matmul_mx.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)
    text = pp.ir.python_print(program)
    groups = {}
    for line in text.splitlines():
        stripped = line.strip()
        head = _TILE.search(stripped)
        if head is None:
            continue
        name, shape, dtype, space, _, size = head.groups()
        groups[name] = dict(
            space=space,
            dtype=dtype,
            size=int(size),
            shape=[int(v) for v in shape.split(",")],
            addrs=[int(a) for _, a, _ in _CELL.findall(stripped)],
        )
    return groups


def _run(device, ops):
    def dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    out = torch.zeros([M, N], device=device, dtype=torch.float32)
    matmul_mx(
        dev(ops["a_codes"]),
        dev(ops["b_codes"]),
        dev(ops["scale_a"]),
        dev(ops["scale_b"]),
        out,
    )
    torch.npu.synchronize()
    return out.cpu()


def test_the_plane_addresses_pair_with_their_data_tiles():
    """约束 1，逐槽。

    框架在 op 上只看到槽 0 的地址，所以修复前这条在双缓冲下失效；`a5c48ce63` 起 parser 侧
    逐槽补上了。这里仍然自己查一遍——否则这个样例本身就可能是个静默错的例子，而且它要在
    老环境里也能自证。
    """
    groups = _declared_groups()

    for data, plane in (("l0a", "plane_a"), ("l0b", "plane_b")):
        want = [addr >> 4 for addr in groups[data]["addrs"]]
        got = groups[plane]["addrs"]
        assert got == want, (
            f"{plane} 的槽地址应当是 {data} 的 >> 4：期望 {[hex(a) for a in want]}，"
            f"实际 {[hex(a) for a in got]}"
        )
        assert len(got) == 2, "两个槽才能看出槽间距那条规则"


def test_the_scalar_addrs_form_would_be_wrong_for_fp8():
    """把那个按 dtype 变脸的陷阱算出来，而不是写句注释。

    `addrs=<标量>` 让框架按 plane **自己**的 tile 大小递推，而硬件要的是**数据** tile 大小
    `>> 4`。FP8 下两者差一倍；FP4 下恰好相等，所以同一个写法只在一半的 dtype 上是对的。
    """
    groups = _declared_groups()
    plane_tile_bytes = groups["plane_a"]["size"]
    data_tile_bytes = groups["l0a"]["size"]

    assert data_tile_bytes == _L0_BYTES == TILE * TILE  # FP8 一字节一个元素
    assert plane_tile_bytes == TILE * SCALE_K  # E8M0 一字节一个组
    needed = data_tile_bytes >> 4
    assert needed == _PLANE_SLOT
    assert plane_tile_bytes != needed, (
        "本样例存在的理由之一就是这两个数不同；相等的话（FP4）标量形式恰好也对"
    )
    assert needed == 2 * plane_tile_bytes, "FP8 下差的正好是一倍"


def test_the_plane_shapes_are_derived_from_the_data_tiles():
    """约束 2。plane 的形状不是自由选的：每 32 个元素一个组。"""
    groups = _declared_groups()

    assert groups["l0a"]["shape"] == [TILE, TILE]
    assert groups["plane_a"]["shape"] == [TILE, TILE // GROUP_SIZE]
    assert groups["l0b"]["shape"] == [TILE, TILE]
    assert groups["plane_b"]["shape"] == [TILE // GROUP_SIZE, TILE]


def test_the_scale_layouts_are_zz_and_nn():
    """约束 5。A 侧 ZZ、B 侧 NN，L1 和 plane 都是。"""
    groups = _declared_groups()

    assert groups["plane_a"]["space"] == "ScaleLeft"
    assert groups["plane_b"]["space"] == "ScaleRight"
    for name in ("sa_l1", "sb_l1", "plane_a", "plane_b"):
        assert groups[name]["dtype"] == "FP8E8M0", f"{name} 应当是 E8M0"
    for name in ("l0a", "l0b", "a_l1", "b_l1"):
        assert groups[name]["dtype"] == "FP8E4M3FN", f"{name} 应当是 FP8"


def test_the_operands_are_consistent():
    """参考自己的自查：scale 的组数要和 K 对得上，阶梯值要精确。

    错了会让下面那条逐位断言拿一个错的 golden 去比。
    """
    ops = make_operands(M, K, N)

    assert ops["scale_a"].shape == (M, K // PAIR, 2)
    assert ops["scale_b"].shape == (K // PAIR, N, 2)
    assert ops["scale_a"].shape[1] * PAIR == K
    assert K % PAIR == 0, "约束 4：K 必须是 64 的倍数"
    # 阶梯：16 个不同值，单调，且都是 1/8 的倍数（3 位尾数，精确）
    assert LADDER_VALUES.size == 16
    assert np.unique(LADDER_VALUES).size == 16
    assert np.all(np.diff(LADDER_VALUES) > 0)
    assert np.all(np.abs(LADDER_VALUES * 8 - np.rint(LADDER_VALUES * 8)) == 0)


def test_matches_the_reference_bit_for_bit(device):
    """逐位相等，不给容差——这组操作数下整条和在 FP32 里精确可表示。

    比给一个松容差强得多：地址配对错了会让某些组乘错 2 倍，那是个**大**偏差；但形状或
    布局错了可能只差最后一个组，松容差有可能放过去。
    """
    ops = make_operands(M, K, N)

    out = _run(device, ops)

    torch.testing.assert_close(
        out,
        torch.from_numpy(matmul_reference(ops["a_values"], ops["b_values"])),
        rtol=0,
        atol=0,
    )


def test_two_runs_agree_bit_for_bit(device):
    """同一份输入跑两遍。"""
    ops = make_operands(M, K, N, seed=7)

    first = _run(device, ops)
    second = _run(device, ops)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
