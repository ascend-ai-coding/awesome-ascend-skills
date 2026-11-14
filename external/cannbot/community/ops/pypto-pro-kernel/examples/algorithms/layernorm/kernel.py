# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。多核逐行 LayerNorm（无仿射），有偏（总体）方差。

out[i, :] = (x[i, :] - mean(x[i, :])) / sqrt(var(x[i, :]) + eps)

## 物理访问

x: [512, 200] FP32，GM，row-major。512 行按 `pl.get_block_idx()` 均分给 16 个 AIV 核
（`kernel[None, NUM_CORES]` 启动），每核 32 行，一次性整块装 UB —— 单核只做一次
load → 算 → store，不构成流水，不需要双缓冲（"最简形式优先"）。

200 列不是 64-lane 寄存器的整数倍。UB 内工作 tile 开到 256 列（4 × 64）：
先用一个与 GM 形状一致的窄 tile（32 × 200）整体 `pl.load`，再 `pl.fillpad(..., mode=EXPAND)`
把第 4 个 64-lane 块的后 56 个 lane 垫零。**垫零只是让 tile 内容有定义、方便调试，不是
正确性的唯一保障**：两次 `reduce_sum`（求和、求平方和）在 tail 块上都换成
`vf.update_mask(8, ...)`，只累计真实的 8 个 lane，因此对 pad 里到底是什么值不敏感。
之所以两次都要掩码而不能只用零哨兵：第二次归约是在减均值之后做的，
`0 - mean != 0`，零哨兵到这一步已经失效（见
`references/patterns/vf-row-reduce.md` 的「Failure signatures」）。

## 为何这样同步

两个 tile 各自一个 `make_tile_group`、单槽（`mutex_ids=[0]` / `[1]`），`auto_mutex=True`
按声明顺序从这两个 id 推 load → fillpad → vf 计算 → store 之间的同步。两块地址
（0x0000 起 0x6400 字节；0x8000 起 0x8000 字节）不重叠，所以可以给不同的 mutex_id
（`overlapping-groups-need-shared-mutex` 只管地址重叠的槽）。

第一手来源：
`references/patterns/vf-row-reduce.md`
（行归约 → 标量链 → 广播这一形态的样板，RMSNorm 掩码尾块做法）；
`examples/basic_api/vf/vf_softmax`（reduce_max/reduce_sum + `vf.full` 广播链）；
`examples/basic_api/datamove/dma_pad`（`pl.fillpad(..., mode=EXPAND)` 用法）；
`examples/basic_api/vf/vf_entry`、`vf_mask_state`（`vf.update_mask(count, dtype=)` 尾块计数掩码）；
`pypto/python/tests/st/pypto_pro/frontend/gather/test_gather_block_variables.py`
（`get_block_idx()` 多核切分、`kernel[None, NUM_CORES]` 启动）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

ROWS = 512
COLS = 200
NUM_CORES = 16
ROWS_PER_CORE = ROWS // NUM_CORES  # 32，均分，见 vf-row-reduce.md「均分优先」

VL = 64  # 一个 FP32 寄存器 = 256 字节 = 64 lane
N_BLOCKS = 4  # ceil(200 / 64)
PADDED_COLS = N_BLOCKS * VL  # 256
BLOCK_OFFSETS = [i * VL for i in range(N_BLOCKS)]  # [0, 64, 128, 192]
TAIL_VALID = COLS - BLOCK_OFFSETS[-1]  # 200 - 192 = 8

EPS = 1e-5
INV_COLS = 1.0 / COLS

_VA_RAW = 0x0000  # 32 * 200 * 4 = 0x6400 字节
_VA_WORK = 0x8000  # 32 * 256 * 4 = 0x8000 字节，与上面不重叠


def _raw_type():
    """与 GM 形状一致的窄 tile：装真实的 200 列，valid_shape 运行期再定。"""
    return pl.TileType(
        shape=[ROWS_PER_CORE, COLS],
        dtype=FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pl.TilePad.null,
    )


def _work_type():
    """256 列的工作 tile，pad=zero 只是给未使用 lane 一个有定义的值。"""
    return pl.TileType(
        shape=[ROWS_PER_CORE, PADDED_COLS],
        dtype=FP32,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )


@pl.vector_function
def _layernorm_rows(tile):
    """逐行：两次归约（sum、sum-of-squares）+ 标量链 + 广播应用，原地写回。

    每行 4 个 64-lane 寄存器块；第 4 块只有前 `TAIL_VALID` 个 lane 是真实数据，
    两次 `reduce_sum` 都在这一块上用 `tail_mask` 而不是 `all_lanes`。
    每个 vf 结果都先命名再复用——vf 调用不能嵌套当另一个调用的实参。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    tail_mask = vf.update_mask(TAIL_VALID, dtype=FP32)

    for r in pl.range(ROWS_PER_CORE):
        b0 = vf.load_align(tile, [r, BLOCK_OFFSETS[0]])
        b1 = vf.load_align(tile, [r, BLOCK_OFFSETS[1]])
        b2 = vf.load_align(tile, [r, BLOCK_OFFSETS[2]])
        b3 = vf.load_align(tile, [r, BLOCK_OFFSETS[3]])

        # 均值：4 块各自 reduce_sum（结果在 lane 0，其余清零），标量相加，乘 1/COLS，广播。
        s0 = vf.reduce_sum(b0, all_lanes)
        s1 = vf.reduce_sum(b1, all_lanes)
        s2 = vf.reduce_sum(b2, all_lanes)
        s3 = vf.reduce_sum(b3, tail_mask)
        sum01 = vf.add(s0, s1, all_lanes)
        sum23 = vf.add(s2, s3, all_lanes)
        total = vf.add(sum01, sum23, all_lanes)
        mean = vf.muls(total, INV_COLS, all_lanes)
        mean_bcast = vf.full(mean, all_lanes)  # Tensor 模式，pos 默认 LOWEST = lane 0

        c0 = vf.sub(b0, mean_bcast, all_lanes)
        c1 = vf.sub(b1, mean_bcast, all_lanes)
        c2 = vf.sub(b2, mean_bcast, all_lanes)
        c3 = vf.sub(b3, mean_bcast, all_lanes)

        # 方差（有偏）：中心化后的平方和 / COLS，同样用 tail_mask 收第 4 块。
        sq0 = vf.mul(c0, c0, all_lanes)
        sq1 = vf.mul(c1, c1, all_lanes)
        sq2 = vf.mul(c2, c2, all_lanes)
        sq3 = vf.mul(c3, c3, all_lanes)
        v0 = vf.reduce_sum(sq0, all_lanes)
        v1 = vf.reduce_sum(sq1, all_lanes)
        v2 = vf.reduce_sum(sq2, all_lanes)
        v3 = vf.reduce_sum(sq3, tail_mask)
        vsum01 = vf.add(v0, v1, all_lanes)
        vsum23 = vf.add(v2, v3, all_lanes)
        vtotal = vf.add(vsum01, vsum23, all_lanes)
        var = vf.muls(vtotal, INV_COLS, all_lanes)
        var_eps = vf.adds(var, EPS, all_lanes)
        std = vf.sqrt(var_eps, all_lanes)
        std_bcast = vf.full(std, all_lanes)

        o0 = vf.div(c0, std_bcast, all_lanes)
        o1 = vf.div(c1, std_bcast, all_lanes)
        o2 = vf.div(c2, std_bcast, all_lanes)
        o3 = vf.div(c3, std_bcast, all_lanes)

        vf.store_align(tile, o0, all_lanes, [r, BLOCK_OFFSETS[0]])
        vf.store_align(tile, o1, all_lanes, [r, BLOCK_OFFSETS[1]])
        vf.store_align(tile, o2, all_lanes, [r, BLOCK_OFFSETS[2]])
        vf.store_align(tile, o3, all_lanes, [r, BLOCK_OFFSETS[3]])


@pl.jit(auto_mutex=True)
def layernorm_kernel(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    raw_group = pl.make_tile_group(type=_raw_type(), addrs=_VA_RAW, mutex_ids=[0])
    work_group = pl.make_tile_group(type=_work_type(), addrs=_VA_WORK, mutex_ids=[1])
    with pl.section_vector():
        raw = raw_group.current()
        work = work_group.current()
        base = pl.get_block_idx() * ROWS_PER_CORE  # 第 vidx 核处理第 [base, base+32) 行
        pl.load(raw, x, [base, 0])
        pl.set_validshape(raw, [ROWS_PER_CORE, COLS])
        pl.fillpad(work, raw, mode=pl.FillPadMode.EXPAND)
        _layernorm_rows(work)
        pl.set_validshape(work, [ROWS_PER_CORE, COLS])  # 只写回真实的 200 列
        pl.store(out, work, [base, 0])
