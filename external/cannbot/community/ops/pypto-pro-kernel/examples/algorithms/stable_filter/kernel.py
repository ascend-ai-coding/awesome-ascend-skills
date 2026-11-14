# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。按行稳定保留正 FP32 值，并写回原始列号与数量。

契约：连续有限 FP32 x[M,N]，输入范围 [-3,3]；保留条件为 x > 0。values[M,N]
前 counts[row] 个元素按输入顺序保存原值，剩余元素写 +0；indices 对应原始列号，
剩余位置写 -1。纯 Vector worker 以行号步进分工。每行使用一组 FP32/INT32 Vec tile，
每 64 列计算谓词前缀排名，再 scatter 到紧凑输出位置。Host 仅检查张量元数据并发射。
全行驻留的 UB 预算为 3*ceil(N/64)*256+256 字节，扩大形状前检查容量。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
UINT32 = pl.DT_UINT32
LANES = 64


@pl.vector_function
def _filter_row(x_tile, values_tile, indices_tile, count_tile, cols):
    """Initialize fixed outputs, then scatter each positive lane to its stable rank."""
    fp_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    int_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    zero_values = vf.full(0.0, fp_mask, dtype=FP32)
    no_index = vf.full(-1, int_mask, dtype=INT32)

    for col in pl.range(0, cols, LANES):
        active = pl.min(LANES, cols - col)
        tail_fp = vf.update_mask(active, dtype=FP32)
        tail_int = vf.update_mask(active, dtype=INT32)
        vf.store_align(values_tile, zero_values, tail_fp, [0, col])
        vf.store_align(indices_tile, no_index, tail_int, [0, col])

    written = vf.full(0, int_mask, dtype=INT32)
    one = vf.full(1, int_mask, dtype=INT32)
    zero = vf.full(0, int_mask, dtype=INT32)
    for col in pl.range(0, cols, LANES):
        active = pl.min(LANES, cols - col)
        tail_fp = vf.update_mask(active, dtype=FP32)
        values = vf.load_align(x_tile, [0, col])
        positive = vf.gt(values, 0.0, tail_fp)

        local_rank = vf.unsqueeze(positive, dtype=INT32)
        base_rank = vf.full(written, int_mask, dtype=INT32, pos=pl.DuplicatePos.LOWEST)
        destination_i32 = vf.add(local_rank, base_rank, int_mask)
        destination_u32 = vf.astype(destination_i32, int_mask, dtype=UINT32)
        source_indices = vf.arange(col, dtype=INT32)

        vf.scatter(values_tile, values, destination_u32, positive)
        vf.scatter(indices_tile, source_indices, destination_u32, positive)

        selected = vf.select(one, zero, positive)
        chunk_count = vf.reduce_sum(selected, int_mask)
        written = vf.add(written, chunk_count, int_mask)

    vf.store_align(count_tile, written, int_mask, [0, 0])


@lru_cache(maxsize=None)
def _make_kernel(cols: int):
    """Build a kernel specialized for N; rows and Vector worker count stay dynamic."""
    if cols <= 0:
        raise ValueError("N must be positive")
    padded_cols = ((cols + LANES - 1) // LANES) * LANES
    row_bytes = padded_cols * 4

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        values: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        indices: pl.Tensor[[pl.DYNAMIC, cols], INT32],
        counts: pl.Tensor[[pl.DYNAMIC], INT32],
    ):
        input_type = pl.TileType(
            shape=[1, padded_cols],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, cols],
        )
        values_type = pl.TileType(
            shape=[1, padded_cols],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, cols],
        )
        indices_type = pl.TileType(
            shape=[1, padded_cols],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, cols],
        )
        count_type = pl.TileType(
            shape=[1, LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, 1],
        )
        input_group = pl.make_tile_group(type=input_type, addrs=[0], mutex_ids=[0])
        values_group = pl.make_tile_group(
            type=values_type, addrs=[row_bytes], mutex_ids=[1]
        )
        indices_group = pl.make_tile_group(
            type=indices_type, addrs=[2 * row_bytes], mutex_ids=[2]
        )
        count_group = pl.make_tile_group(
            type=count_type, addrs=[3 * row_bytes], mutex_ids=[3]
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                x_tile = input_group.next()
                values_tile = values_group.next()
                indices_tile = indices_group.next()
                count_tile = count_group.next()
                pl.load(x_tile, x, [row, 0])
                _filter_row(x_tile, values_tile, indices_tile, count_tile, cols)
                pl.store(values, values_tile, [row, 0])
                pl.store(indices, indices_tile, [row, 0])
                pl.store(counts, count_tile, [row])

    return kernel


def run(x, values, indices, counts, workers=1):
    """Launch the native kernel; host work is limited to tensor metadata."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if len(x.shape) != 2 or len(values.shape) != 2 or len(indices.shape) != 2:
        raise ValueError("x, values, and indices must be rank 2")
    rows, cols = map(int, x.shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("M and N must be positive")
    if tuple(values.shape) != (rows, cols) or tuple(indices.shape) != (rows, cols):
        raise ValueError("values and indices must match x shape")
    if tuple(counts.shape) != (rows,):
        raise ValueError("counts must have shape [M]")
    if str(x.dtype) != "torch.float32" or str(values.dtype) != "torch.float32":
        raise ValueError("x and values must be FP32")
    if str(indices.dtype) != "torch.int32" or str(counts.dtype) != "torch.int32":
        raise ValueError("indices and counts must be INT32")
    if not all(t.is_contiguous() for t in (x, values, indices, counts)):
        raise ValueError("all tensors must be contiguous")
    if any(t.device != x.device for t in (values, indices, counts)):
        raise ValueError("all tensors must be on the same device")
    _make_kernel(cols)[None, workers](x, values, indices, counts)
