# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 逐行最大值与首个最大值下标，紧密 DMA 输出。

契约：有限 FP32 输入在 [-100, 100]，FP32 values[M] 与 INT32 indices[M] 紧密输出。
已验 M=1/17/65、N=63/130/257；行独立并按 worker 数循环分配。相等最大值选择最小列索引。
计算保持值与原始索引对应；先逐 lane 汇集分块候选，再归约最大值和合格候选的最小索引。
物理访问：输入行填齐到 64-lane，两个结果 tile 各 [1,64]，完整 VF store 的写入宽度与声明一致；GM store 只读有效的 [1,1]。
同步：三个互不重叠单槽组由 auto_mutex 排序。结果走 UB→GM DMA，避免多核相邻 GM 标量写的丢写路径。
来源：2026-10-06 gpt-6-luna/max skill 试验。入库时仅把输出 tile 物理宽度从16增至64以匹配原来的全寄存器 store，并整理说明、包导入和测试；算法不变。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR；运行 test.py 检查数值与输出守卫。未测硬件性能。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
LANES = 64


def make_row_argmax(cols: int, workers: int):
    """Build a dynamic-row kernel specialized for one column count."""
    padded_cols = ((cols + LANES - 1) // LANES) * LANES
    full_cols = (cols // LANES) * LANES
    tail_cols = cols - full_cols

    @pl.vector_function
    def _argmax_row(input_tile, value_tile, index_tile):
        all_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
        best_values = vf.full(-1.0e30, all_mask, dtype=FP32)
        best_indices = vf.arange(0, dtype=INT32)

        for col in pl.range(0, full_cols, LANES):
            current = vf.load_align(input_tile, [0, col])
            greater = vf.gt(current, best_values, all_mask)
            best_values = vf.select(current, best_values, greater)
            current_indices = vf.arange(col, dtype=INT32)
            best_indices = vf.select(current_indices, best_indices, greater)

        tail_mask = vf.update_mask(tail_cols, dtype=FP32)
        tail_values = vf.load_align(input_tile, [0, full_cols])
        tail_greater = vf.gt(tail_values, best_values, tail_mask)
        best_values = vf.select(tail_values, best_values, tail_greater)
        tail_indices = vf.arange(full_cols, dtype=INT32)
        best_indices = vf.select(tail_indices, best_indices, tail_greater)

        global_max = vf.reduce_max(best_values, all_mask)
        global_max_broadcast = vf.full(global_max, all_mask)
        maxima = vf.eq(best_values, global_max_broadcast, all_mask)
        no_index = vf.full(cols, int_mask, dtype=INT32)
        candidate_indices = vf.select(best_indices, no_index, maxima)
        first_index = vf.reduce_min(candidate_indices, int_mask)

        vf.store_align(value_tile, global_max, all_mask, [0, 0])
        vf.store_align(index_tile, first_index, int_mask, [0, 0])

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        values: pl.Tensor[[pl.DYNAMIC], FP32],
        indices: pl.Tensor[[pl.DYNAMIC], INT32],
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
        value_group = pl.make_tile_group(type=value_type, addrs=[1536], mutex_ids=[1])
        index_group = pl.make_tile_group(type=index_type, addrs=[1792], mutex_ids=[2])

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], workers):
                input_tile = input_group.next()
                value_tile = value_group.next()
                index_tile = index_group.next()
                pl.load(input_tile, x, [row, 0])
                _argmax_row(input_tile, value_tile, index_tile)
                pl.store(values, value_tile, [row])
                pl.store(indices, index_tile, [row])

    return kernel
