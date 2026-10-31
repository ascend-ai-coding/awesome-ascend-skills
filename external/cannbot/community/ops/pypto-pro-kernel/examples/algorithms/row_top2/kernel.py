# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 稳定逐行 Top-2，输出输入原始 FP32 位模式。

契约：连续有限 FP32 x[M,N]，N>=2；数值降序、平局按原始下标升序，两个下标不同。
+0/-0 数值相等，但输出保留所选源的符号位。Host 仅选择形状与 worker 数。
每个 lane 保留两份独立的值/索引状态，再跨 lane 合并；最终从获胜来源选择数值。
全行常驻 UB，输入占 ceil(N/64)*256 字节，四个输出槽各占256字节；扩大形状前检查总容量。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR。test.py 固定精确下标、位模式及守卫判据。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
LANES = 64
NEG_INF = float("-inf")
INDEX_SENTINEL = 2147483647


@lru_cache(maxsize=None)
def _make_kernel(cols: int):
    padded_cols = ((cols + LANES - 1) // LANES) * LANES
    input_bytes = padded_cols * 4

    @pl.vector_function
    def _top2_row(input_tile, value1_tile, index1_tile, value2_tile, index2_tile):
        float_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
        negative_infinity = vf.full(NEG_INF, float_all, dtype=FP32)
        best_values = vf.full(NEG_INF, float_all, dtype=FP32)
        second_values = vf.full(NEG_INF, float_all, dtype=FP32)
        no_index = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)
        best_indices = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)
        second_indices = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)

        for col in pl.range(0, cols, LANES):
            active = pl.min(LANES, cols - col)
            active_mask = vf.update_mask(active, dtype=FP32)
            current_values = vf.load_align(input_tile, [0, col])
            current_indices = vf.arange(col, dtype=INT32)

            is_new_best = vf.gt(current_values, best_values, active_mask)
            is_new_second = vf.gt(current_values, second_values, active_mask)

            candidate_second_values = vf.select(
                current_values, second_values, is_new_second
            )
            candidate_second_indices = vf.select(
                current_indices, second_indices, is_new_second
            )
            second_values = vf.select(best_values, candidate_second_values, is_new_best)
            second_indices = vf.select(
                best_indices, candidate_second_indices, is_new_best
            )
            best_values = vf.select(current_values, best_values, is_new_best)
            best_indices = vf.select(current_indices, best_indices, is_new_best)

        global_best = vf.reduce_max(best_values, float_all)
        global_best_broadcast = vf.full(global_best, float_all, dtype=FP32)
        best_ties = vf.eq(best_values, global_best_broadcast, float_all)
        best_index_candidates = vf.select(best_indices, no_index, best_ties)
        global_best_index = vf.reduce_min(best_index_candidates, int_all)
        global_best_index_broadcast = vf.full(global_best_index, int_all, dtype=INT32)
        is_global_best = vf.eq(best_indices, global_best_index_broadcast, int_all)
        selected_best_values = vf.select(best_values, negative_infinity, is_global_best)
        selected_best = vf.reduce_max(selected_best_values, float_all)

        second_candidates = vf.select(second_values, best_values, is_global_best)
        second_index_candidates = vf.select(
            second_indices, best_indices, is_global_best
        )
        global_second = vf.reduce_max(second_candidates, float_all)
        global_second_broadcast = vf.full(global_second, float_all, dtype=FP32)
        second_ties = vf.eq(second_candidates, global_second_broadcast, float_all)
        second_index_ties = vf.select(second_index_candidates, no_index, second_ties)
        global_second_index = vf.reduce_min(second_index_ties, int_all)
        global_second_index_broadcast = vf.full(
            global_second_index, int_all, dtype=INT32
        )
        is_global_second = vf.eq(
            second_index_candidates, global_second_index_broadcast, int_all
        )
        selected_second_values = vf.select(
            second_candidates, negative_infinity, is_global_second
        )
        selected_second = vf.reduce_max(selected_second_values, float_all)

        vf.store_align(value1_tile, selected_best, float_all, [0, 0])
        vf.store_align(index1_tile, global_best_index, int_all, [0, 0])
        vf.store_align(value2_tile, selected_second, float_all, [0, 0])
        vf.store_align(index2_tile, global_second_index, int_all, [0, 0])

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        values: pl.Tensor[[pl.DYNAMIC, 2], FP32],
        indices: pl.Tensor[[pl.DYNAMIC, 2], INT32],
    ):
        input_type = pl.TileType(
            shape=[1, padded_cols],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, cols],
        )
        value_type = pl.TileType(
            shape=[1, LANES],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, 1],
        )
        index_type = pl.TileType(
            shape=[1, LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, 1],
        )
        input_group = pl.make_tile_group(type=input_type, addrs=[0], mutex_ids=[0])
        value1_group = pl.make_tile_group(
            type=value_type, addrs=[input_bytes], mutex_ids=[1]
        )
        index1_group = pl.make_tile_group(
            type=index_type, addrs=[input_bytes + 256], mutex_ids=[2]
        )
        value2_group = pl.make_tile_group(
            type=value_type, addrs=[input_bytes + 512], mutex_ids=[3]
        )
        index2_group = pl.make_tile_group(
            type=index_type, addrs=[input_bytes + 768], mutex_ids=[4]
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                input_tile = input_group.next()
                value1_tile = value1_group.next()
                index1_tile = index1_group.next()
                value2_tile = value2_group.next()
                index2_tile = index2_group.next()
                pl.load(input_tile, x, [row, 0])
                _top2_row(
                    input_tile,
                    value1_tile,
                    index1_tile,
                    value2_tile,
                    index2_tile,
                )
                pl.store(values, value1_tile, [row, 0])
                pl.store(indices, index1_tile, [row, 0])
                pl.store(values, value2_tile, [row, 1])
                pl.store(indices, index2_tile, [row, 1])

    return kernel


def run(x, values, indices, workers=1):
    """Launch the native Top-2 kernel; all numeric work stays on the NPU."""
    if workers < 1:
        raise ValueError("workers must be positive")
    cols = int(x.shape[1])
    _make_kernel(cols)[None, workers](x, values, indices)
