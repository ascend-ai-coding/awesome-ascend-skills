# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。固定 UB 工作区的长行 masked softmax 与 logsumexp。

契约：连续 FP32 x[M,N]，有限输入位于 [-1020,1020]；INT32 lengths[M]，M/N 为正且 0 <= lengths[r] <= N。
输出 FP32 y[M,N] 与 lse[M,1]；前缀归一化，失效后缀严格为零，空行 lse=-inf。
验证输入为有限 FP32：随机 [-20,20]、逐行 ±1000 偏移及结构化有限分布。

每行由一个 Vector worker 持有。4096 元素是 GM 搬运块，VF 在其中遍历 64-lane 寄存器块。
第一遍在线合并 max/denominator，第二遍重读输入后写概率，最后把 sum 槽复用为 LSE。
数据 tile 占 16384B@0；两个统计槽各占 256B@0x5000/0x5100，总地址高水位 0x5200。
跨 VF 调用的状态在 UB 内，auto_mutex 保护单槽复用；动态有效区限制 GM 尾块和紧密 LSE 写出。

来源：gpt-6-luna/max 独立实现，入库保留 kernel 计算逻辑；独立数学参考和契约测试在同目录。
上板边界：PyPTO 行为版本 872df763f、CANN 9.2.0 / BiSheng 15.0.5、Ascend950PR。
已验 (M,N)=(5,65),(3,4097),(3,131073)，含非整除多核分行；未声称此块大小对所有形状最优。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
VL = 64
CHUNK = 4096
_INITIAL_MAX = -3.0e38


@lru_cache(maxsize=None)
def make_kernel(m_rows: int, n_cols: int):
    """Build a shape-specialized kernel with bounded UB use."""

    @pl.vector_function
    def init_stats(max_tile, sum_tile):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        max_value = vf.full(_INITIAL_MAX, all_lanes)
        zero = vf.full(0.0, all_lanes)
        vf.store_align(max_tile, max_value, all_lanes, [0, 0])
        vf.store_align(sum_tile, zero, all_lanes, [0, 0])

    @pl.vector_function
    def update_stats(tile, max_tile, sum_tile, length, chunk_start):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_lane = vf.update_mask(1, dtype=FP32)
        zero = vf.full(0.0, all_lanes)

        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        for inner in pl.range(0, CHUNK, VL):
            col = chunk_start + inner
            active = length - col
            if active > VL:
                active = VL
            if active < 0:
                active = 0
            prefix = vf.update_mask(active, dtype=FP32)
            values = vf.load_align(tile, [0, inner])

            max_broadcast = vf.full(max_reg, all_lanes)
            safe_values = vf.select(values, max_broadcast, prefix)
            block_max = vf.reduce_max(safe_values, all_lanes)
            next_max = vf.max(max_reg, block_max, first_lane)
            next_max_broadcast = vf.full(next_max, all_lanes)

            shifted = vf.sub(safe_values, next_max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            masked_exponents = vf.select(exponents, zero, prefix)
            block_sum = vf.reduce_sum(masked_exponents, all_lanes)

            old_delta = vf.sub(max_reg, next_max, first_lane)
            old_scale = vf.exp(old_delta, first_lane)
            old_sum = vf.mul(sum_reg, old_scale, first_lane)
            next_sum = vf.add(old_sum, block_sum, first_lane)

            max_reg = vf.move(next_max, all_lanes)
            sum_reg = vf.move(next_sum, all_lanes)

        vf.store_align(max_tile, max_reg, first_lane, [0, 0])
        vf.store_align(sum_tile, sum_reg, first_lane, [0, 0])

    @pl.vector_function
    def emit_chunk(tile, max_tile, sum_tile, length, chunk_start, valid_cols):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        ones = vf.full(1.0, all_lanes)
        zero = vf.full(0.0, all_lanes)

        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        max_broadcast = vf.full(max_reg, all_lanes)
        sum_broadcast = vf.full(sum_reg, all_lanes)
        denominator = vf.max(sum_broadcast, ones, all_lanes)

        for inner in pl.range(0, CHUNK, VL):
            col = chunk_start + inner
            output_active = valid_cols - inner
            if output_active > VL:
                output_active = VL
            if output_active < 0:
                output_active = 0
            active = length - col
            if active > output_active:
                active = output_active
            if active < 0:
                active = 0
            write_mask = vf.update_mask(output_active, dtype=FP32)
            prefix = vf.update_mask(active, dtype=FP32)

            values = vf.load_align(tile, [0, inner])
            safe_values = vf.select(values, max_broadcast, prefix)
            shifted = vf.sub(safe_values, max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            masked_exponents = vf.select(exponents, zero, prefix)
            result = vf.div(masked_exponents, denominator, all_lanes)
            vf.store_align(tile, result, write_mask, [0, inner])

    @pl.vector_function
    def write_lse(max_tile, sum_tile):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_lane = vf.update_mask(1, dtype=FP32)
        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        max_broadcast = vf.full(max_reg, all_lanes)
        sum_broadcast = vf.full(sum_reg, all_lanes)
        log_sum = vf.log(sum_broadcast, all_lanes)
        lse = vf.add(max_broadcast, log_sum, all_lanes)
        vf.store_align(sum_tile, lse, first_lane, [0, 0])

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[m_rows, n_cols], FP32],
        lengths: pl.Tensor[[m_rows], INT32],
        out: pl.Tensor[[m_rows, n_cols], FP32],
        lse: pl.Tensor[[m_rows, 1], FP32],
    ):
        data_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, CHUNK],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=pl.TilePad.zero,
            ),
            addrs=[0x0000],
            mutex_ids=[0],
        )
        stats_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, VL],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=[0x5000, 0x5100],
            mutex_ids=[1, 2],
        )

        with pl.section_vector():
            data_tile = data_group.current()
            max_tile = stats_group[0]
            sum_tile = stats_group[1]
            for row in pl.range(pl.get_block_idx(), m_rows, pl.get_block_num()):
                length = pl.getval(lengths, row)
                pl.set_validshape(max_tile, [1, VL])
                pl.set_validshape(sum_tile, [1, VL])
                init_stats(max_tile, sum_tile)

                for chunk in pl.range(0, n_cols, CHUNK):
                    if chunk < length:
                        valid_cols = n_cols - chunk
                        if valid_cols > CHUNK:
                            valid_cols = CHUNK
                        pl.set_validshape(data_tile, [1, valid_cols])
                        pl.load(data_tile, x, [row, chunk])
                        update_stats(data_tile, max_tile, sum_tile, length, chunk)

                for chunk in pl.range(0, n_cols, CHUNK):
                    valid_cols = n_cols - chunk
                    if valid_cols > CHUNK:
                        valid_cols = CHUNK
                    pl.set_validshape(data_tile, [1, valid_cols])
                    pl.load(data_tile, x, [row, chunk])
                    emit_chunk(data_tile, max_tile, sum_tile, length, chunk, valid_cols)
                    pl.store(out, data_tile, [row, chunk])

                write_lse(max_tile, sum_tile)
                pl.set_validshape(sum_tile, [1, 1])
                pl.store(lse, sum_tile, [row, 0])

    return kernel


def run(x, lengths, out, lse, workers=1):
    """Launch the native kernel. All numerical work stays on the device."""
    m_rows, n_cols = x.shape
    make_kernel(m_rows, n_cols)[None, workers](x, lengths, out, lse)
