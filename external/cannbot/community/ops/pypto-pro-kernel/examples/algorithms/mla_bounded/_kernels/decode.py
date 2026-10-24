# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。单 query、独立 V 的分特征流式 attention。

FP16 连续 BSND/BNSD；Sq=1，B>0，H 为行块的倍数，行块可选16/32/64。
Skv为128的倍数且<=4096，Dn为64的倍数且<=512，Dr为64或128；输入有限且abs<=1。
QK分量在FP32累加，P按块窄化为FP16，最大值、分母和加权和始终为FP32。
不分配GM分数/概率矩阵。单query的右对齐因果范围也覆盖全部keys。

L1操作数、L0步长与Vector状态分别规划。两次乘法共享双槽L0；短特征使用保留
槽地址/mutex的窄形状别名。每行统计只占一个32字节块，经BRC_B32广播。
跨阶段状态按每个执行组的连续tick轮转，不能在奇数面板数的item边界重置槽号。
默认preload=1；preload=0保留相同运算顺序作为同步对照。

资源上限：Mat504KiB，Left/Right/Acc32/64/160KiB，Vec254.25KiB。
TileType/make_tile_group 显式指定物理地址，mutex_ids="auto" 管理核内依赖。
score/P/PV 的跨核事件分别为 0/1→2/3、4→5、6/7→8/9，槽深度为 2/1/2。
L1容量不足以双缓冲全部特征时退到单槽加载，不能把广播状态或槽数漏出预算。
无跨核key分区；选行块仍需比较完整调用、尾波和共享KV的并发读取。
"""

from functools import lru_cache
import math

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32


@pl.vector_function
def _stream_softmax(
    scores,
    alpha_state,
    sum_state,
    max_state,
    prev_sum,
    prev_max,
    nz_lo,
    first,
    scale,
    half_rows,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    mask16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    compact = vf.update_mask(8, dtype=FP32)
    for row in pl.range(half_rows):
        x0 = vf.load_align(scores, [row, 0])
        x1 = vf.load_align(scores, [row, 64])
        pair_max = vf.max(x0, x1, mask)
        scalar_max = vf.reduce_max(pair_max, mask)
        new_max = vf.full(scalar_max, mask)
        alpha = vf.full(1.0, mask, dtype=FP32)
        if not first:
            old_max = vf.load_align(prev_max, [row, 0], dist=pl.LoadDist.BRC_B32)
            new_max = vf.max(old_max, new_max, mask)
            delta = vf.sub(old_max, new_max, mask)
            delta = vf.muls(delta, scale, mask)
            alpha = vf.exp(delta, mask)
        vf.store_align(alpha_state, alpha, compact, [row, 0])
        vf.store_align(max_state, new_max, compact, [row, 0])
        x0 = vf.sub(x0, new_max, mask)
        x1 = vf.sub(x1, new_max, mask)
        x0 = vf.muls(x0, scale, mask)
        x1 = vf.muls(x1, scale, mask)
        p0 = vf.exp(x0, mask)
        p1 = vf.exp(x1, mask)
        lane_sum = vf.add(p0, p1, mask)
        scalar_sum = vf.reduce_sum(lane_sum, mask)
        new_sum = vf.full(scalar_sum, mask)
        if not first:
            old_sum = vf.load_align(prev_sum, [row, 0], dist=pl.LoadDist.BRC_B32)
            old_sum = vf.mul(old_sum, alpha, mask)
            new_sum = vf.add(old_sum, new_sum, mask)
        vf.store_align(sum_state, new_sum, compact, [row, 0])
        h0 = vf.astype(p0, mask, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
        h1 = vf.astype(p1, mask, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
        packed, _ = vf.de_interleave(h0, h1)
        vf.store_align(
            nz_lo,
            packed,
            mask16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=half_rows + 1,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def _stream_update(
    pv_values, alpha_state, running, first, half_rows, column_base, nope
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for row in pl.range(half_rows):
        alpha = vf.full(1.0, mask, dtype=FP32)
        if not first:
            alpha = vf.load_align(alpha_state, [row, 0], dist=pl.LoadDist.BRC_B32)
        for col in pl.range(nope // 64):
            column = column_base + col * 64
            pv = vf.load_align(pv_values, [row, col * 64])
            if first:
                value = pv
            else:
                value = vf.load_align(running, [row, column])
                value = vf.mul_dst_add(alpha, pv, mask)
            vf.store_align(running, value, mask, [row, column])


@pl.vector_function
def _stream_finalize(running, sum_state, out_tile, half_rows, output_cols, column_base):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    mask16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for row in pl.range(half_rows):
        denominator = vf.load_align(sum_state, [row, 0], dist=pl.LoadDist.BRC_B32)
        for col in pl.range(output_cols // 64):
            value = vf.load_align(running, [row, column_base + col * 64])
            normalized = vf.div(value, denominator, mask)
            narrow = vf.astype(
                normalized, mask, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
            )
            vf.store_align(
                out_tile, narrow, mask16, [row, col * 64], dist=pl.StoreDist.PACK_B32
            )


@pl.pipeline.stage
def _stream_qk_stage(
    part,
    batch_index,
    qn_group,
    qr_group,
    kn,
    kr,
    kv_group,
    kr_group,
    kn_transposed,
    kr_transposed,
    left,
    right,
    rope_left,
    rope_right,
    qk_acc,
    score_vec,
    rounds,
    tick,
    key_slots,
    main_step,
    rope_step,
):
    qn_tile = qn_group.current()
    qr_tile = qr_group.current()
    kn_tile = kv_group[tick % key_slots]
    kr_tile = kr_group[tick % key_slots]
    if part == 0 or key_slots == 1:
        pl.load(kn_tile, kn, [batch_index, part * 128, 0], order=[1, 2])
        pl.load(kr_tile, kr, [batch_index, part * 128, 0], order=[1, 2])
    if key_slots == 2 and part + 1 < rounds:
        kn_next = kv_group[(tick + 1) % key_slots]
        kr_next = kr_group[(tick + 1) % key_slots]
        pl.load(kn_next, kn, [batch_index, (part + 1) * 128, 0], order=[1, 2])
        pl.load(kr_next, kr, [batch_index, (part + 1) * 128, 0], order=[1, 2])
    acc = qk_acc.current()
    kn_rhs = kn_transposed[tick % key_slots]
    kr_rhs = kr_transposed[tick % key_slots]
    pl.gemm(acc, qn_tile, kn_rhs, l0a=left, l0b=right, splitk=main_step)
    pl.gemm(
        acc,
        qr_tile,
        kr_rhs,
        l0a=rope_left,
        l0b=rope_right,
        splitk=rope_step,
        init=False,
    )
    scores = score_vec.next()
    pl.move(scores, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def _stream_softmax_stage(
    part,
    score_vec,
    alpha_state,
    sum_state,
    max_state,
    nz_lo_group,
    probability_group,
    scale,
    half_rows,
    tick,
):
    scores = score_vec.next()
    alpha_tile = alpha_state[tick % 2]
    sum_tile = sum_state[tick % 2]
    max_tile = max_state[tick % 2]
    prev_sum = sum_state[(tick + 1) % 2]
    prev_max = max_state[(tick + 1) % 2]
    nz_lo = nz_lo_group.current()
    if part == 0:
        _stream_softmax(
            scores,
            alpha_tile,
            sum_tile,
            max_tile,
            prev_sum,
            prev_max,
            nz_lo,
            True,
            scale,
            half_rows,
        )
    else:
        _stream_softmax(
            scores,
            alpha_tile,
            sum_tile,
            max_tile,
            prev_sum,
            prev_max,
            nz_lo,
            False,
            scale,
            half_rows,
        )
    pl.set_validshape(nz_lo, [half_rows, 128])
    probabilities = probability_group.next()
    row_offset = pl.get_subblock_idx() * half_rows
    pl.insert(probabilities, nz_lo, [row_offset, 0])


@pl.pipeline.stage
def _stream_pv_stage(
    part,
    batch_index,
    values,
    value_tile_group,
    left,
    right,
    pv_acc_group,
    probability_group,
    pv_vec,
    product_step,
):
    probabilities = probability_group.next()
    values_tile = value_tile_group.current()
    pl.load(values_tile, values, [batch_index, part * 128, 0], order=[1, 2])
    acc = pv_acc_group.current()
    pl.gemm(acc, probabilities, values_tile, l0a=left, l0b=right, splitn=product_step)
    partial = pv_vec.next()
    pl.move(partial, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def _stream_update_stage(
    part,
    batch_index,
    output,
    pv_vec,
    alpha_state,
    sum_state,
    running_group,
    output_tile_group,
    half_rows,
    row_base,
    rounds,
    tick,
    nope,
    out_cols,
):
    partial = pv_vec.next()
    running = running_group.current()
    output_tile = output_tile_group.current()
    alpha_tile = alpha_state[tick % 2]
    sum_tile = sum_state[tick % 2]
    if part == 0:
        _stream_update(partial, alpha_tile, running, True, half_rows, 0, nope)
    else:
        _stream_update(partial, alpha_tile, running, False, half_rows, 0, nope)
    if part == rounds - 1:
        row_offset = row_base + pl.get_subblock_idx() * half_rows
        for column in pl.range(0, nope, out_cols):
            _stream_finalize(
                running, sum_tile, output_tile, half_rows, out_cols, column
            )
            pl.store(output, output_tile, [batch_index, row_offset, column])


@lru_cache(maxsize=128)
def make_decode_kernel(batch, keys, heads, nope, rope=64, *, rows=64, preload=1):
    """Return one native FP16 kernel; the caller supplies a legal worker count."""
    if batch < 1 or rows not in (16, 32, 64) or heads < rows or heads % rows:
        raise ValueError(
            "positive batch and complete 16/32/64-row head blocks are required"
        )
    if keys < 128 or keys > 4096 or keys % 128:
        raise ValueError("key length must be a multiple of 128 up to 4096")
    if nope < 64 or nope > 512 or nope % 64 or rope not in (64, 128):
        raise ValueError("Dn must be a multiple of 64 up to 512; Dr must be 64 or 128")
    if preload not in (0, 1):
        raise ValueError("this state/slot plan supports preload 0 or 1")
    dtype = pl.DT_FP16
    logical_rows = heads
    row_blocks = heads // rows
    rounds = keys // 128
    half_rows = rows // 2
    scale = 1 / math.sqrt(nope + rope)
    main_step = min(128, nope)
    while nope % main_step:
        main_step //= 2
    rope_step = min(128, rope)
    product_step = min(128, nope)
    while nope % product_step:
        product_step //= 2
    key_slots = 2
    mat_bytes = 2 * ((rows + key_slots * 128) * (nope + rope) + rows * 128 + 128 * nope)
    if mat_bytes > 512 * 1024:
        key_slots = 1
    out_cols = min(256, nope)
    while nope % out_cols:
        out_cols //= 2

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=f"mla_decode_b{batch}_h{heads}_k{keys}_d{nope}_r{rope}_m{rows}_p{preload}",
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        qn: pl.Tensor[[batch, logical_rows, nope], dtype],
        qr: pl.Tensor[[batch, logical_rows, rope], dtype],
        kn: pl.Tensor[[batch, keys, nope], dtype],
        kr: pl.Tensor[[batch, keys, rope], dtype],
        values: pl.Tensor[[batch, keys, nope], dtype],
        output: pl.Tensor[[batch, logical_rows, nope], dtype],
    ):
        qn_group_type = pl.TileType(
            shape=[rows, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qn_group = pl.make_tile_group(
            type=qn_group_type, addrs=0, mutex_ids="auto", depth=1
        )
        qr_group_type = pl.TileType(
            shape=[rows, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qr_group = pl.make_tile_group(
            type=qr_group_type, addrs=2 * nope * rows, mutex_ids="auto", depth=1
        )
        kv_group_type = pl.TileType(
            shape=[128, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        kv_group = pl.make_tile_group(
            type=kv_group_type,
            addrs=2 * nope * rows + 2 * rope * rows,
            mutex_ids="auto",
            depth=key_slots,
        )
        value_group_type = pl.TileType(
            shape=[128, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        value_group = pl.make_tile_group(
            type=value_group_type,
            addrs=2 * nope * rows + 2 * rope * rows + 256 * key_slots * nope,
            mutex_ids="auto",
            depth=1,
        )
        kr_group_type = pl.TileType(
            shape=[128, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        kr_group = pl.make_tile_group(
            type=kr_group_type,
            addrs=2 * nope * rows
            + 2 * rope * rows
            + 256 * key_slots * nope
            + 256 * nope,
            mutex_ids="auto",
            depth=key_slots,
        )
        left_type = pl.TileType(
            shape=[rows, 128], dtype=dtype, target_memory=pl.MemorySpace.Left, compact=1
        )
        left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids="auto", depth=2)
        pv_right_type = pl.TileType(
            shape=[128, 128], dtype=dtype, target_memory=pl.MemorySpace.Right, compact=1
        )
        pv_right = pl.make_tile_group(
            type=pv_right_type, addrs=0, mutex_ids="auto", depth=2
        )
        qk_acc_type = pl.TileType(
            shape=[rows, 128], dtype=FP32, target_memory=pl.MemorySpace.Acc
        )
        qk_acc = pl.make_tile_group(
            type=qk_acc_type, addrs=0, mutex_ids="auto", depth=1
        )
        pv_acc_type = pl.TileType(
            shape=[rows, nope], dtype=FP32, target_memory=pl.MemorySpace.Acc
        )
        pv_acc = pl.make_tile_group(
            type=pv_acc_type, addrs=512 * rows, mutex_ids="auto", depth=1
        )
        scores_group_type = pl.TileType(
            shape=[half_rows, 128],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.ND,
        )
        scores_group = pl.make_tile_group(
            type=scores_group_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        probabilities_group_type = pl.TileType(
            shape=[rows, 128],
            dtype=dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        probabilities_group = pl.make_tile_group(
            type=probabilities_group_type,
            addrs=2 * nope * rows
            + 2 * rope * rows
            + 256 * key_slots * nope
            + 256 * nope
            + 256 * key_slots * rope,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[4],
            bwd_ids=[5],
        )
        pv_group_type = pl.TileType(
            shape=[half_rows, nope],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.ND,
        )
        pv_group = pl.make_tile_group(
            type=pv_group_type,
            addrs=1024 * half_rows,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[6, 7],
            bwd_ids=[8, 9],
        )
        nz_lo_group_type = pl.TileType(
            shape=[half_rows + 1, 128],
            dtype=dtype,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        nz_lo_group = pl.make_tile_group(
            type=nz_lo_group_type,
            addrs=1024 * half_rows + 8 * half_rows * nope,
            mutex_ids="auto",
            depth=1,
        )
        alpha_state_type = pl.TileType(
            shape=[half_rows, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        alpha_state = pl.make_tile_group(
            type=alpha_state_type,
            addrs=1280 * half_rows + 8 * half_rows * nope + 256,
            mutex_ids="auto",
            depth=2,
        )
        sum_state_type = pl.TileType(
            shape=[half_rows, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        sum_state = pl.make_tile_group(
            type=sum_state_type,
            addrs=1344 * half_rows + 8 * half_rows * nope + 256,
            mutex_ids="auto",
            depth=2,
        )
        max_state_type = pl.TileType(
            shape=[half_rows, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        max_state = pl.make_tile_group(
            type=max_state_type,
            addrs=1408 * half_rows + 8 * half_rows * nope + 256,
            mutex_ids="auto",
            depth=2,
        )
        running_group_type = pl.TileType(
            shape=[half_rows, nope], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        running_group = pl.make_tile_group(
            type=running_group_type,
            addrs=1472 * half_rows + 8 * half_rows * nope + 256,
            mutex_ids="auto",
            depth=1,
        )
        output_tile_group_type = pl.TileType(
            shape=[half_rows, out_cols], dtype=dtype, target_memory=pl.MemorySpace.Vec
        )
        output_tile_group = pl.make_tile_group(
            type=output_tile_group_type,
            addrs=1472 * half_rows + 12 * half_rows * nope + 256,
            mutex_ids="auto",
            depth=1,
        )
        main_left = pl.reinterpret(left, shape=[rows, main_step])
        main_right = pl.reinterpret(pv_right, shape=[main_step, 128])
        product_right = pl.reinterpret(pv_right, shape=[128, product_step])
        rope_left = pl.reinterpret(left, shape=[rows, rope_step])
        rope_right = pl.reinterpret(pv_right, shape=[rope_step, 128])
        kn_transposed = pl.reinterpret(kv_group, shape=[nope, 128], layout=pl.ZN)
        kr_transposed = pl.reinterpret(kr_group, shape=[rope, 128], layout=pl.ZN)
        group = pl.get_block_idx() // pl.get_subblock_num()
        for work in pl.range(group, batch * row_blocks, pl.get_block_num()):
            batch_index = work // row_blocks
            row_base = work % row_blocks * rows
            with pl.section_cube():
                pl.load(
                    qn_group.current(), qn, [batch_index, row_base, 0], order=[1, 2]
                )
                pl.load(
                    qr_group.current(), qr, [batch_index, row_base, 0], order=[1, 2]
                )
            for part in pl.range(rounds):
                tick = (work // pl.get_block_num()) * rounds + part
                with pl.section_cube():
                    _stream_qk_stage(
                        part,
                        batch_index,
                        qn_group,
                        qr_group,
                        kn,
                        kr,
                        kv_group,
                        kr_group,
                        kn_transposed,
                        kr_transposed,
                        main_left,
                        main_right,
                        rope_left,
                        rope_right,
                        qk_acc,
                        scores_group,
                        rounds,
                        tick,
                        key_slots,
                        main_step,
                        rope_step,
                    )
                with pl.section_vector():
                    _stream_softmax_stage(
                        part,
                        scores_group,
                        alpha_state,
                        sum_state,
                        max_state,
                        nz_lo_group,
                        probabilities_group,
                        scale,
                        half_rows,
                        tick,
                    )
                with pl.section_cube():
                    _stream_pv_stage(
                        part,
                        batch_index,
                        values,
                        value_group,
                        left,
                        product_right,
                        pv_acc,
                        probabilities_group,
                        pv_group,
                        product_step,
                    )
                with pl.section_vector():
                    _stream_update_stage(
                        part,
                        batch_index,
                        output,
                        pv_group,
                        alpha_state,
                        sum_state,
                        running_group,
                        output_tile_group,
                        half_rows,
                        row_base,
                        rounds,
                        tick,
                        nope,
                        out_cols,
                    )

    return kernel


def mla_decode(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    values,
    *,
    layout="BSND",
    workers=28,
    rows=None,
    preload=1,
):
    """Metadata-only wrapper for single-query attention; V remains independent."""
    import torch

    if layout not in ("BSND", "BNSD") or q_nope.ndim != 4 or workers < 1:
        raise ValueError("rank-4 BSND/BNSD and a positive worker count are required")
    batch, d1, d2, nope = q_nope.shape
    queries, heads = (d1, d2) if layout == "BSND" else (d2, d1)
    if queries != 1:
        raise ValueError("this entry covers a single query")
    rope = q_rope.shape[-1]
    keys = k_nope.shape[1 if layout == "BSND" else 2]
    kshape = (batch, keys, 1, nope) if layout == "BSND" else (batch, 1, keys, nope)
    expected = [
        tuple(q_nope.shape[:-1]) + (rope,),
        kshape,
        kshape[:-1] + (rope,),
        kshape,
    ]
    tensors = (q_nope, q_rope, k_nope, k_rope, values)
    if [tuple(t.shape) for t in tensors[1:]] != expected:
        raise ValueError(
            "input shapes do not match single-query, single-KV-head attention"
        )
    if any(
        t.dtype != torch.float16 or t.device != q_nope.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("inputs must be contiguous FP16 tensors on the same device")
    if rows is None:
        rows = 64
        while rows > 16 and heads % rows:
            rows //= 2
    kernel = make_decode_kernel(
        batch, keys, heads, nope, rope, rows=rows, preload=preload
    )
    out = torch.empty_like(q_nope)
    kernel[None, workers](
        q_nope.view(batch, heads, nope),
        q_rope.view(batch, heads, rope),
        k_nope.view(batch, keys, nope),
        k_rope.view(batch, keys, rope),
        values.view(batch, keys, nope),
        out.view(batch, heads, nope),
    )
    return out
