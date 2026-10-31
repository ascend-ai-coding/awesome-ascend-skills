# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。稳定行 Top-K，候选状态跨 VF 输入块保存在 UB。

支持连续有限 FP32 x[M,N]，1<=M<=64、1<=N<=131073、1<=K<=min(16,N)。
按值降序、原列号升序选择；返回 INT32 下标与原始 FP32 payload 位模式。
每 lane 保留 K 个候选，最终选择在独立 VF 调用间推进；输入更新循环用
VV_ALL 明确排序 UB 状态的访存。块大小 2048，K=16 时 UB 高水位 16896B。

该示例展示有界状态与同步，最终逐项扫描候选；完整计时包含所有归并调用。
行为版本 872df763f、CANN 9.2.0、Ascend950PR 上验证。当前数值范围内索引
转 FP32 比较精确，INT32 哨兵转换后仍大于任意合法索引。
库内测试逐位检查稳定参考，包含不同 worker、重复调用与输出守卫。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
LANES = 64
BLOCK = 2048
NEG_INF = float("-inf")
INDEX_SENTINEL = 2147483647


@lru_cache(maxsize=None)
def _make_kernel(cols: int, k: int):
    local_bytes = k * LANES * 4
    input_bytes = BLOCK * 4
    local_values_addr = input_bytes
    local_indices_addr = local_values_addr + local_bytes
    output_values_addr = local_indices_addr + local_bytes
    output_indices_addr = output_values_addr + 256

    @pl.vector_function
    def _insert_chunk(
        input_tile, local_values_tile, local_indices_tile, base, valid_cols
    ):
        float_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)

        for offset in pl.range(0, valid_cols, LANES):
            vf.mem_bar(mode=pl.MemBarMode.VV_ALL)
            active = pl.min(LANES, valid_cols - offset)
            active_mask = vf.update_mask(active, dtype=FP32)
            current_values = vf.load_align(input_tile, [0, offset])
            current_indices = vf.arange(base + offset, dtype=INT32)
            carry_values = current_values
            carry_indices = current_indices

            for rank in pl.range(k):
                prior_values = vf.load_align(local_values_tile, [rank, 0])
                prior_indices = vf.load_align(local_indices_tile, [rank, 0])
                higher_value = vf.gt(carry_values, prior_values, active_mask)
                equal_value = vf.eq(carry_values, prior_values, active_mask)
                earlier_index = vf.lt(carry_indices, prior_indices, active_mask)
                earlier_tie = vf.and_(equal_value, earlier_index, active_mask)
                is_better = vf.or_(higher_value, earlier_tie, active_mask)
                is_better = vf.and_(is_better, active_mask, active_mask)

                next_values = vf.select(carry_values, prior_values, is_better)
                next_indices = vf.select(carry_indices, prior_indices, is_better)
                carry_values = vf.select(prior_values, carry_values, is_better)
                carry_indices = vf.select(prior_indices, carry_indices, is_better)

                vf.store_align(local_values_tile, next_values, float_all, [rank, 0])
                vf.store_align(local_indices_tile, next_indices, int_all, [rank, 0])

    @pl.vector_function
    def _initialize(local_values_tile, local_indices_tile):
        float_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
        negative_infinity = vf.full(NEG_INF, float_all, dtype=FP32)
        no_index = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)
        for rank in pl.range(k):
            vf.store_align(local_values_tile, negative_infinity, float_all, [rank, 0])
            vf.store_align(local_indices_tile, no_index, int_all, [rank, 0])

    @pl.vector_function
    def _merge_one(
        local_values_tile,
        local_indices_tile,
        output_values_tile,
        output_indices_tile,
        output_rank,
    ):
        float_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
        uint_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        negative_infinity = vf.full(NEG_INF, float_all, dtype=FP32)
        no_index = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)
        output_lanes = vf.arange(0, dtype=INT32)
        output_indices = vf.load_align(output_indices_tile, [0, 0])
        winner_value = vf.full(NEG_INF, float_all, dtype=FP32)
        winner_index = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)

        for local_rank in pl.range(k):
            candidate_values = vf.load_align(local_values_tile, [local_rank, 0])
            candidate_indices = vf.load_align(local_indices_tile, [local_rank, 0])
            excluded = vf.eq(candidate_indices, no_index, int_all)
            for previous_rank in pl.range(output_rank):
                previous_slot = vf.full(previous_rank, uint_all, dtype=pl.DT_UINT32)
                previous_index = vf.gather(output_indices, previous_slot)
                is_previous = vf.eq(candidate_indices, previous_index, int_all)
                excluded = vf.or_(excluded, is_previous, int_all)
            eligible_indices = vf.select(no_index, candidate_indices, excluded)
            available_values = vf.select(negative_infinity, candidate_values, excluded)

            rank_max = vf.reduce_max(available_values, float_all)
            rank_max_broadcast = vf.full(rank_max, float_all, dtype=FP32)
            rank_ties = vf.eq(available_values, rank_max_broadcast, float_all)
            tied_indices = vf.select(eligible_indices, no_index, rank_ties)
            rank_index = vf.reduce_min(tied_indices, int_all)
            rank_index_broadcast = vf.full(rank_index, int_all, dtype=INT32)

            selected_lane = vf.eq(candidate_indices, rank_index_broadcast, int_all)
            exact_rank_values = vf.select(
                available_values, negative_infinity, selected_lane
            )
            rank_value = vf.reduce_max(exact_rank_values, float_all)
            rank_value_broadcast = vf.full(rank_value, float_all, dtype=FP32)

            rank_index_fp32 = vf.astype(rank_index_broadcast, int_all, dtype=FP32)
            winner_index_fp32 = vf.astype(winner_index, int_all, dtype=FP32)
            higher_value = vf.gt(rank_value_broadcast, winner_value, float_all)
            equal_value = vf.eq(rank_value_broadcast, winner_value, float_all)
            earlier_index = vf.lt(rank_index_fp32, winner_index_fp32, float_all)
            earlier_tie = vf.and_(equal_value, earlier_index, float_all)
            is_better = vf.or_(higher_value, earlier_tie, float_all)
            winner_value = vf.select(rank_value_broadcast, winner_value, is_better)
            winner_index = vf.select(rank_index_broadcast, winner_index, is_better)

        output_rank_broadcast = vf.full(output_rank, int_all, dtype=INT32)
        output_slot = vf.eq(output_lanes, output_rank_broadcast, int_all)
        vf.store_align(output_values_tile, winner_value, output_slot, [0, 0])
        vf.store_align(output_indices_tile, winner_index, output_slot, [0, 0])

        for local_rank in pl.range(k):
            candidate_values = vf.load_align(local_values_tile, [local_rank, 0])
            candidate_indices = vf.load_align(local_indices_tile, [local_rank, 0])
            selected_lane = vf.eq(candidate_indices, winner_index, int_all)
            candidate_values = vf.select(
                negative_infinity, candidate_values, selected_lane
            )
            candidate_indices = vf.select(no_index, candidate_indices, selected_lane)
            vf.store_align(
                local_values_tile, candidate_values, float_all, [local_rank, 0]
            )
            vf.store_align(
                local_indices_tile, candidate_indices, int_all, [local_rank, 0]
            )

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        values: pl.Tensor[[pl.DYNAMIC, k], FP32],
        indices: pl.Tensor[[pl.DYNAMIC, k], INT32],
    ):
        input_type = pl.TileType(
            shape=[1, BLOCK],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        )
        local_values_type = pl.TileType(
            shape=[k, LANES],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[k, LANES],
        )
        local_indices_type = pl.TileType(
            shape=[k, LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[k, LANES],
        )
        output_values_type = pl.TileType(
            shape=[1, LANES],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, k],
        )
        output_indices_type = pl.TileType(
            shape=[1, LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, k],
        )
        input_group = pl.make_tile_group(type=input_type, addrs=[0], mutex_ids=[0])
        local_values_group = pl.make_tile_group(
            type=local_values_type, addrs=[local_values_addr], mutex_ids=[1]
        )
        local_indices_group = pl.make_tile_group(
            type=local_indices_type, addrs=[local_indices_addr], mutex_ids=[2]
        )
        output_values_group = pl.make_tile_group(
            type=output_values_type, addrs=[output_values_addr], mutex_ids=[3]
        )
        output_indices_group = pl.make_tile_group(
            type=output_indices_type, addrs=[output_indices_addr], mutex_ids=[4]
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                input_tile = input_group.next()
                local_values_tile = local_values_group.next()
                local_indices_tile = local_indices_group.next()
                output_values_tile = output_values_group.next()
                output_indices_tile = output_indices_group.next()
                _initialize(local_values_tile, local_indices_tile)

                for base in pl.range(0, cols, BLOCK):
                    valid_cols = pl.min(BLOCK, cols - base)
                    pl.set_validshape(input_tile, [1, valid_cols])
                    pl.load(input_tile, x, [row, base])
                    _insert_chunk(
                        input_tile,
                        local_values_tile,
                        local_indices_tile,
                        base,
                        valid_cols,
                    )

                for output_rank in pl.range(k):
                    _merge_one(
                        local_values_tile,
                        local_indices_tile,
                        output_values_tile,
                        output_indices_tile,
                        output_rank,
                    )
                pl.store(values, output_values_tile, [row, 0])
                pl.store(indices, output_indices_tile, [row, 0])

    return kernel


def run(x, values, indices, k, workers=1):
    """执行稳定行 Top-K；Host 只检查元数据并选择编译形状。"""
    rows, cols = int(x.shape[0]), int(x.shape[1])
    k = int(k)
    workers = int(workers)
    if not 1 <= rows <= 64 or not 1 <= cols <= 131073:
        raise ValueError("expected 1<=M<=64 and 1<=N<=131073")
    if not 1 <= k <= min(16, cols):
        raise ValueError("expected 1<=k<=min(16,N)")
    if workers < 1:
        raise ValueError("workers must be positive")
    if tuple(values.shape) != (rows, k) or tuple(indices.shape) != (rows, k):
        raise ValueError("output shapes must both be (M,k)")
    _make_kernel(cols, k)[None, workers](x, values, indices)
