# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。超出 UB 容量的一维 INT32 inclusive segmented scan。

契约：连续 x/heads/out[N]，1<=N<=1048576，x 在[-4,4]，heads 为0/1。
i=0或heads[i]=1时输出x[i]，否则累加前一输出；输入保持不变，输出精确INT32。
三次原生调用分别扫描独立块、计算块摘要的前缀、补入每块的外部carry。
块内寄存器carry与块外carry的有效范围不同；重置标记与数值共享lane边界。
第一、三阶段按逻辑块轮转分工；第二阶段由单worker串行处理至多86个摘要，
独占稠密carry输出。三次调用在同一Stream上排序，auto_mutex管理各阶段UB搬运。
默认块长12288，Vec高水位3*12288*4+2*64*4=147968 B；GM scratch为
ceil(N/12288)*516 B，最大44376 B。最坏累加绝对值4194304，不溢出INT32。
边界块显式清零无效lane，动态标量广播使用独立寄存器名，摘要保留末个有效值。
该三阶段方案保留并行块分工；少量worker下额外遍历和任务开销需要单独测量。
验证工具链：PyPTO行为基线872df763f、CANN 9.2.0、Ascend950PR、
BiSheng 15.0.5、torch/torch_npu 2.10.0；当前结果仍须绑定源文件哈希。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


INT32 = pl.DT_INT32
UINT32 = pl.DT_UINT32
VL = 64
CHUNK = 12288


@pl.vector_function
def _scan_local(x_tile, heads_tile, out_tile, tail_tile, has_head_tile, extent):
    """Write a zero-carry local scan and its summary; masked adds zero tail lanes."""
    int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    uint_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=UINT32)
    lane_ids = vf.arange(0, dtype=UINT32)
    index_mask = vf.full(VL - 1, uint_all, dtype=UINT32)
    zeros = vf.full(0, int_all, dtype=INT32)
    carry = vf.full(0, int_all, dtype=INT32)
    seen_head = vf.full(0, int_all, dtype=INT32)

    for col in pl.range(0, extent, VL):
        active = pl.min(VL, extent - col)
        active_mask = vf.update_mask(active, dtype=INT32)
        loaded_values = vf.load_align(x_tile, [0, col])
        loaded_heads = vf.load_align(heads_tile, [0, col])
        values = vf.adds(loaded_values, 0, active_mask)
        heads = vf.adds(loaded_heads, 0, active_mask)
        scanned = values
        reset = heads
        offset = 1
        for _ in pl.range(0, 6):
            wrapped = vf.adds(lane_ids, VL - offset, uint_all)
            wrapped = vf.and_(wrapped, index_mask, uint_all)
            left_sum = vf.gather(scanned, wrapped)
            left_reset = vf.gather(reset, wrapped)
            sum_with_left = vf.add(scanned, left_sum, int_all)
            no_reset = vf.eq(reset, zeros, int_all)
            sum_with_reset = vf.select(sum_with_left, scanned, no_reset)
            lower_lanes = vf.lt(lane_ids, offset, uint_all)
            scanned = vf.select(scanned, sum_with_reset, lower_lanes)
            reset_with_left = vf.max(reset, left_reset, int_all)
            reset = vf.select(reset, reset_with_left, lower_lanes)
            offset = offset * 2

        prefix_has_head = reset
        no_head_in_prefix = vf.eq(prefix_has_head, zeros, int_all)
        with_carry = vf.add(scanned, carry, int_all)
        result = vf.select(with_carry, scanned, no_head_in_prefix)
        vf.store_align(out_tile, result, active_mask, [0, col])

        last_index = vf.full(active - 1, uint_all, dtype=UINT32)
        carry = vf.gather(result, last_index)
        block_has_head = vf.gather(reset, last_index)
        seen_head = vf.max(seen_head, block_has_head, int_all)

    vf.store_align(tail_tile, carry, int_all, [0, 0])
    vf.store_align(has_head_tile, seen_head, int_all, [0, 0])


@pl.vector_function
def _apply_incoming_carry(out_tile, heads_tile, carry_value: pl.DT_INT64, extent):
    """Add the chunk carry only to elements before its first segment head."""
    int_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    uint_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=UINT32)
    lane_ids = vf.arange(0, dtype=UINT32)
    index_mask = vf.full(VL - 1, uint_all, dtype=UINT32)
    zeros = vf.full(0, int_all, dtype=INT32)
    carry = vf.full(carry_value, int_all, dtype=INT32)
    seen_head = vf.full(0, int_all, dtype=INT32)

    for col in pl.range(0, extent, VL):
        active = pl.min(VL, extent - col)
        active_mask = vf.update_mask(active, dtype=INT32)
        loaded_values = vf.load_align(out_tile, [0, col])
        loaded_heads = vf.load_align(heads_tile, [0, col])
        values = vf.adds(loaded_values, 0, active_mask)
        heads = vf.adds(loaded_heads, 0, active_mask)
        prefix_has_head = heads
        offset = 1
        for _ in pl.range(0, 6):
            wrapped = vf.adds(lane_ids, VL - offset, uint_all)
            wrapped = vf.and_(wrapped, index_mask, uint_all)
            left_head = vf.gather(prefix_has_head, wrapped)
            head_with_left = vf.max(prefix_has_head, left_head, int_all)
            lower_lanes = vf.lt(lane_ids, offset, uint_all)
            prefix_has_head = vf.select(prefix_has_head, head_with_left, lower_lanes)
            offset = offset * 2

        prefix_has_head = vf.max(seen_head, prefix_has_head, int_all)
        no_head_in_prefix = vf.eq(prefix_has_head, zeros, int_all)
        with_carry = vf.add(values, carry, int_all)
        result = vf.select(with_carry, values, no_head_in_prefix)
        vf.store_align(out_tile, result, active_mask, [0, col])
        last_index = vf.full(active - 1, uint_all, dtype=UINT32)
        seen_head = vf.gather(prefix_has_head, last_index)


@lru_cache(maxsize=None)
def _make_kernels(size: int):
    chunk_size = CHUNK
    if size < 1 or size > 1_048_576:
        raise ValueError("N must be in [1, 1048576]")
    chunks = (size + chunk_size - 1) // chunk_size
    heads_addr = chunk_size * 4
    out_addr = chunk_size * 8
    tail_addr = chunk_size * 12
    has_head_addr = tail_addr + VL * 4

    @pl.jit(auto_mutex=True)
    def local_scan(
        x: pl.Tensor[[size], INT32],
        heads: pl.Tensor[[size], INT32],
        out: pl.Tensor[[size], INT32],
        tails: pl.Tensor[[chunks, VL], INT32],
        has_heads: pl.Tensor[[chunks, VL], INT32],
    ):
        chunk_type = pl.TileType(
            shape=[1, chunk_size], dtype=INT32, target_memory=pl.MemorySpace.Vec
        )
        summary_type = pl.TileType(
            shape=[1, VL], dtype=INT32, target_memory=pl.MemorySpace.Vec
        )
        x_group = pl.make_tile_group(type=chunk_type, addrs=[0], mutex_ids=[0])
        heads_group = pl.make_tile_group(
            type=chunk_type, addrs=[heads_addr], mutex_ids=[1]
        )
        out_group = pl.make_tile_group(type=chunk_type, addrs=[out_addr], mutex_ids=[2])
        tail_group = pl.make_tile_group(
            type=summary_type, addrs=[tail_addr], mutex_ids=[3]
        )
        has_head_group = pl.make_tile_group(
            type=summary_type, addrs=[has_head_addr], mutex_ids=[4]
        )
        with pl.section_vector():
            for chunk in pl.range(pl.get_block_idx(), chunks, pl.get_block_num()):
                origin = chunk * chunk_size
                extent = pl.min(chunk_size, size - origin)
                x_tile = x_group.next()
                heads_tile = heads_group.next()
                out_tile = out_group.next()
                tail_tile = tail_group.next()
                has_head_tile = has_head_group.next()
                pl.set_validshape(x_tile, [1, extent])
                pl.set_validshape(heads_tile, [1, extent])
                pl.set_validshape(out_tile, [1, extent])
                pl.load(x_tile, x, [origin])
                pl.load(heads_tile, heads, [origin])
                _scan_local(
                    x_tile, heads_tile, out_tile, tail_tile, has_head_tile, extent
                )
                pl.store(out, out_tile, [origin])
                pl.store(tails, tail_tile, [chunk, 0])
                pl.store(has_heads, has_head_tile, [chunk, 0])

    @pl.jit(auto_mutex=True)
    def prefix_carries(
        tails: pl.Tensor[[chunks, VL], INT32],
        has_heads: pl.Tensor[[chunks, VL], INT32],
        carries: pl.Tensor[[chunks], INT32],
    ):
        with pl.section_vector():
            carry_value = 0
            for chunk in pl.range(chunks):
                pl.setval(carries, chunk, carry_value)
                chunk_has_head = pl.getval(has_heads, chunk * VL)
                chunk_tail = pl.getval(tails, chunk * VL)
                if chunk_has_head != 0:
                    carry_value = chunk_tail
                else:
                    carry_value = carry_value + chunk_tail

    @pl.jit(auto_mutex=True)
    def apply_carries(
        heads: pl.Tensor[[size], INT32],
        out: pl.Tensor[[size], INT32],
        carries: pl.Tensor[[chunks], INT32],
    ):
        chunk_type = pl.TileType(
            shape=[1, chunk_size], dtype=INT32, target_memory=pl.MemorySpace.Vec
        )
        heads_group = pl.make_tile_group(type=chunk_type, addrs=[0], mutex_ids=[0])
        out_group = pl.make_tile_group(
            type=chunk_type, addrs=[heads_addr], mutex_ids=[1]
        )
        with pl.section_vector():
            for chunk in pl.range(pl.get_block_idx(), chunks, pl.get_block_num()):
                origin = chunk * chunk_size
                extent = pl.min(chunk_size, size - origin)
                carry_value = pl.getval(carries, chunk)
                heads_tile = heads_group.next()
                out_tile = out_group.next()
                pl.set_validshape(heads_tile, [1, extent])
                pl.set_validshape(out_tile, [1, extent])
                pl.load(heads_tile, heads, [origin])
                pl.load(out_tile, out, [origin])
                _apply_incoming_carry(out_tile, heads_tile, carry_value, extent)
                pl.store(out, out_tile, [origin])

    return local_scan, prefix_carries, apply_carries, chunks


def run(x, heads, out, workers=4):
    """Launch three native tasks; Host only checks metadata and allocates scratch."""
    if workers not in (1, 4, 16):
        raise ValueError("workers must be 1, 4, or 16")
    if tuple(x.shape) != tuple(heads.shape) or tuple(x.shape) != tuple(out.shape):
        raise ValueError("x, heads, and out must have identical shapes")
    if len(x.shape) != 1 or not 1 <= int(x.shape[0]) <= 1_048_576:
        raise ValueError("x, heads, and out must be 1-D with N in [1, 1048576]")
    if (
        str(x.dtype) != "torch.int32"
        or str(heads.dtype) != "torch.int32"
        or str(out.dtype) != "torch.int32"
    ):
        raise TypeError("x, heads, and out must be INT32")
    if not x.is_contiguous() or not heads.is_contiguous() or not out.is_contiguous():
        raise ValueError("x, heads, and out must be contiguous")
    if x.device != heads.device or x.device != out.device:
        raise ValueError("x, heads, and out must use the same device")

    n = int(x.shape[0])
    local_scan, prefix_carries, apply_carries, chunks = _make_kernels(n)
    tails = x.new_empty((chunks, VL))
    has_heads = x.new_empty((chunks, VL))
    carries = x.new_empty((chunks,))
    local_scan[None, workers](x, heads, out, tails, has_heads)
    prefix_carries[None, 1](tails, has_heads, carries)
    apply_carries[None, workers](heads, out, carries)
