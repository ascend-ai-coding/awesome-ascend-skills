# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。带运行期有效长度的逐行 VF softmax。

契约：FP32 x[M, N]，INT32 lengths[M]，0 <= lengths[i] <= N。
仅有效前缀参与归一化，失效后缀和空行精确为零；本样例验证有限输入 [-8, 8]。
已验宽度 64、130、257；M 动态，按实际 vector worker 数做 grid-stride 行分配。

物理访问：每个 worker 一个 [1, ceil(N/64)*64] FP32 UB tile，动态有效区控制 GM 搬运。
三遍 VF 分别求最大、指数和及输出；最终一遍才原地写回。前缀掩码独立于物理 padding。
同步：单槽 tile group 与 auto_mutex 排序 load、VF、store；每行只有一个 worker 写入。

验证边界：行为版本 872df763f，CANN 9.2.0，Ascend950PR。
容差与边界由 test.py 固定，不代表性能已优化。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
VL = 64
_NEG = -3.0e38


def make_masked_softmax(n_cols):
    """Build a dynamic-row-count kernel specialized only for the column count."""
    blocks = (n_cols + VL - 1) // VL
    padded_cols = blocks * VL

    @pl.vector_function
    def softmax_row(tile, length):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_lane = vf.update_mask(1, dtype=FP32)
        neg_inf = vf.full(_NEG, all_lanes)
        zero = vf.full(0.0, all_lanes)
        one = vf.full(1.0, all_lanes)
        row_max = vf.full(_NEG, all_lanes)

        for block in pl.range(0, blocks, 1):
            offset = block * VL
            active = length - offset
            if active > VL:
                active = VL
            if active < 0:
                active = 0
            prefix = vf.update_mask(active, dtype=FP32)
            values = vf.load_align(tile, [0, offset])
            safe_values = vf.select(values, neg_inf, prefix)
            block_max = vf.reduce_max(safe_values, all_lanes)
            row_max = vf.max(row_max, block_max, first_lane)

        max_broadcast = vf.full(row_max, all_lanes)
        row_sum = vf.full(0.0, all_lanes)
        for block in pl.range(0, blocks, 1):
            offset = block * VL
            active = length - offset
            if active > VL:
                active = VL
            if active < 0:
                active = 0
            prefix = vf.update_mask(active, dtype=FP32)
            values = vf.load_align(tile, [0, offset])
            safe_values = vf.select(values, max_broadcast, prefix)
            shifted = vf.sub(safe_values, max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            masked_exponents = vf.select(exponents, zero, prefix)
            block_sum = vf.reduce_sum(masked_exponents, all_lanes)
            row_sum = vf.add(row_sum, block_sum, first_lane)

        denominator = vf.max(row_sum, one, first_lane)
        denominator_broadcast = vf.full(denominator, all_lanes)
        for block in pl.range(0, blocks, 1):
            offset = block * VL
            active = length - offset
            if active > VL:
                active = VL
            if active < 0:
                active = 0
            prefix = vf.update_mask(active, dtype=FP32)
            values = vf.load_align(tile, [0, offset])
            safe_values = vf.select(values, max_broadcast, prefix)
            shifted = vf.sub(safe_values, max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            masked_exponents = vf.select(exponents, zero, prefix)
            result = vf.div(masked_exponents, denominator_broadcast, all_lanes)
            vf.store_align(tile, result, all_lanes, [0, offset])

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, n_cols], FP32],
        lengths: pl.Tensor[[pl.DYNAMIC], INT32],
        out: pl.Tensor[[pl.DYNAMIC, n_cols], FP32],
    ):
        tile_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_cols],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=pl.TilePad.zero,
            ),
            addrs=[0x0000],
            mutex_ids=[0],
        )
        with pl.section_vector():
            tile = tile_group.current()
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                pl.set_validshape(tile, [1, n_cols])
                pl.load(tile, x, [row, 0])
                length = pl.getval(lengths, row)
                softmax_row(tile, length)
                pl.set_validshape(tile, [1, n_cols])
                pl.store(out, tile, [row, 0])

    return kernel
