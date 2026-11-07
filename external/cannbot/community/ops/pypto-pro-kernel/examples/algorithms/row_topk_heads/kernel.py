# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。按完整总序维护局部候选，并以各列表首项归并稳定 Top-K。

契约：连续有限FP32 x[M,N]，1<=M<=32、1<=N<=65537、1<=K<=min(N,256)。
输出values[M,K]保留输入位模式，INT32 indices[M,K]按值降序、原列号升序。
每个worker拥有完整的行；单行只有一个有效owner，增加workers不拆分其计算。
输入按2048元素搬入；64个lane各保留K项，插入、置换、剪枝和归并都使用同一总序。
首项归并只推进获胜列表的游标；耗尽列表不能再胜出，gather地址始终在分配范围内。
游标在独立UB tile中更新，VV_ALL保护VF循环回边的读写；原生auto_mutex排序各tile。
Vec高水位为8192+2*K*256+256+2*ceil(K/64)*256 B，K=256时为141568 B。
无GM scratch；一个原生task包含候选构建、完整归并和稠密输出写回。
该示例展示归并表示的变化；局部插入仍随N*K增长，单行并行分区不在此实现中。
验证工具链：PyPTO行为基线872df763f、CANN 9.2.0、Ascend950PR、
BiSheng 15.0.5、torch/torch_npu 2.10.0；当前验收绑定实际源码哈希。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
UINT32 = pl.DT_UINT32
LANES = 64
BLOCK = 2048
NEG_INF = float("-inf")
INDEX_SENTINEL = 2147483647


@lru_cache(maxsize=None)
def _make_kernel(cols: int, k: int):
    local_bytes = k * LANES * 4
    input_bytes = BLOCK * 4
    padded_k = ((k + LANES - 1) // LANES) * LANES
    local_values_addr = input_bytes
    local_indices_addr = local_values_addr + local_bytes
    cursor_addr = local_indices_addr + local_bytes
    cursor_bytes = LANES * 4
    output_values_addr = cursor_addr + cursor_bytes
    output_indices_addr = output_values_addr + padded_k * 4

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
                tie_better = vf.and_(equal_value, earlier_index, active_mask)
                is_better = vf.or_(higher_value, tie_better, active_mask)
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
    def _merge_heads(
        local_values_tile,
        local_indices_tile,
        output_values_tile,
        output_indices_tile,
        cursor_tile,
    ):
        float_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
        uint_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=UINT32)
        negative_infinity = vf.full(NEG_INF, float_all, dtype=FP32)
        no_index = vf.full(INDEX_SENTINEL, int_all, dtype=INT32)
        lane_indices = vf.arange(0, dtype=INT32)
        lane_indices_u32 = vf.astype(lane_indices, int_all, dtype=UINT32)
        cursor_zero = vf.full(0, int_all, dtype=INT32)
        vf.store_align(cursor_tile, cursor_zero, int_all, [0, 0])
        one_cursor = vf.full(1, int_all, dtype=INT32)
        last_cursor = vf.full(k - 1, int_all, dtype=INT32)

        head_values = vf.gather(local_values_tile, lane_indices_u32, float_all)
        head_indices = vf.gather(local_indices_tile, lane_indices_u32, int_all)

        for output_rank in pl.range(k):
            vf.mem_bar(mode=pl.MemBarMode.VV_ALL)
            cursor = vf.load_align(cursor_tile, [0, 0])
            max_value = vf.reduce_max(head_values, float_all)
            max_value_lanes = vf.full(max_value, float_all, dtype=FP32)
            ties = vf.eq(head_values, max_value_lanes, float_all)
            tied_indices = vf.select(head_indices, no_index, ties)
            winner_index = vf.reduce_min(tied_indices, int_all)
            winner_index_lanes = vf.full(winner_index, int_all, dtype=INT32)
            winner_lane_mask = vf.eq(head_indices, winner_index_lanes, int_all)

            lane_candidates = vf.select(lane_indices, no_index, winner_lane_mask)
            winner_lane = vf.reduce_min(lane_candidates, int_all)
            winner_lane_lanes = vf.full(winner_lane, int_all, dtype=INT32)
            winner_lane_u32 = vf.astype(winner_lane_lanes, int_all, dtype=UINT32)
            winner_value = vf.gather(head_values, winner_lane_u32)

            output_lane = vf.full(output_rank % LANES, int_all, dtype=INT32)
            output_mask = vf.eq(lane_indices, output_lane, int_all)
            output_base = (output_rank // LANES) * LANES
            vf.store_align(
                output_values_tile, winner_value, output_mask, [0, output_base]
            )
            vf.store_align(
                output_indices_tile,
                winner_index_lanes,
                output_mask,
                [0, output_base],
            )

            has_next = vf.lt(cursor, last_cursor, int_all)
            advance = vf.and_(winner_lane_mask, has_next, int_all)
            cursor_plus_one = vf.add(cursor, one_cursor, int_all)
            next_cursor = vf.select(cursor_plus_one, cursor, advance)
            next_cursor_u32 = vf.astype(next_cursor, int_all, dtype=UINT32)
            next_offsets = vf.muls(next_cursor_u32, LANES, uint_all)
            next_indices = vf.add(next_offsets, lane_indices_u32, uint_all)
            next_values = vf.gather(local_values_tile, next_indices, advance)
            next_payload_indices = vf.gather(local_indices_tile, next_indices, advance)
            refreshed_values = vf.select(next_values, head_values, advance)
            refreshed_indices = vf.select(next_payload_indices, head_indices, advance)
            exhausted = vf.eq(cursor, last_cursor, int_all)
            exhausted = vf.and_(winner_lane_mask, exhausted, int_all)
            head_values = vf.select(negative_infinity, refreshed_values, exhausted)
            head_indices = vf.select(no_index, refreshed_indices, exhausted)
            vf.store_align(cursor_tile, next_cursor, int_all, [0, 0])

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
        cursor_type = pl.TileType(
            shape=[1, LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, LANES],
        )
        output_values_type = pl.TileType(
            shape=[1, padded_k],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        )
        output_indices_type = pl.TileType(
            shape=[1, padded_k],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        )
        input_group = pl.make_tile_group(type=input_type, addrs=[0], mutex_ids=[0])
        local_values_group = pl.make_tile_group(
            type=local_values_type, addrs=[local_values_addr], mutex_ids=[1]
        )
        local_indices_group = pl.make_tile_group(
            type=local_indices_type, addrs=[local_indices_addr], mutex_ids=[2]
        )
        cursor_group = pl.make_tile_group(
            type=cursor_type, addrs=[cursor_addr], mutex_ids=[5]
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
                cursor_tile = cursor_group.next()
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

                _merge_heads(
                    local_values_tile,
                    local_indices_tile,
                    output_values_tile,
                    output_indices_tile,
                    cursor_tile,
                )
                pl.set_validshape(output_values_tile, [1, k])
                pl.set_validshape(output_indices_tile, [1, k])
                pl.store(values, output_values_tile, [row, 0])
                pl.store(indices, output_indices_tile, [row, 0])

    return kernel


def run(x, values, indices, k, workers=4):
    """Execute stable row Top-K; the host only validates shape metadata."""
    rows, cols = int(x.shape[0]), int(x.shape[1])
    k = int(k)
    workers = int(workers)
    if not 1 <= rows <= 32 or not 1 <= cols <= 65537:
        raise ValueError("expected 1<=M<=32 and 1<=N<=65537")
    if not 1 <= k <= min(256, cols):
        raise ValueError("expected 1<=k<=min(256,N)")
    if workers not in (1, 4, 16):
        raise ValueError("workers must be one of 1, 4, or 16")
    if tuple(values.shape) != (rows, k) or tuple(indices.shape) != (rows, k):
        raise ValueError("output shapes must both be (M,k)")
    _make_kernel(cols, k)[None, workers](x, values, indices)
