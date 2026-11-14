# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。短归约轴的原生 MLA 片上融合对照。

支持 FP16、独立 V、连续 BSND/BNSD，输入有限且 abs<=1；Skv 为128或256，
Dn为64的倍数且<=512，Dr为64或128，H为16的倍数。BNSD因果路径的Sq须为16的倍数。
QK、整行归一化与PV在同一执行组内完成；Mat中的K缓冲在其最后一次读取后由V覆盖，
reinterpret保留地址和mutex身份。P使用原生NZ交接；preload=0为串行对照，默认1。
因果query区间按其最大可见key向上取整，多个launch覆盖不相交输出，计时须相加。

最大资源：Mat 496KiB，Left/Right/Acc 32/64/256KiB，Vec 160.5KiB。
物理地址由 TileType/make_tile_group 显式指定，mutex_ids="auto" 保留各槽的核内身份。
分数交接使用两槽事件 0/1→2/3，概率交接使用单槽事件 4→5。
BF16小值保护仍使用同目录kernel.py的实现；本融合入口明确只验收FP16。
"""

import math
from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

F32, F16 = pl.DT_FP32, pl.DT_FP16


@lru_cache(None)
def make_fused(
    batch,
    queries,
    keys,
    heads,
    nope,
    rope=64,
    dtype_name="float16",
    layout="BSND",
    causal=False,
    workers=28,
    rows=128,
    cap_keys=None,
    query_start=0,
    query_end=None,
    preload=1,
):
    if dtype_name != "float16":
        raise ValueError("this fused entry supports FP16 only")
    if batch < 1 or queries < 1 or heads < 16 or heads % 16 or workers < 1:
        raise ValueError("positive B/Sq and H divisible by 16 are required")
    if nope < 64 or nope > 512 or nope % 64 or rope not in (64, 128):
        raise ValueError("Dn must be a multiple of 64 up to 512; Dr must be 64 or 128")
    if layout not in ("BSND", "BNSD") or preload not in (0, 1):
        raise ValueError("BSND/BNSD and preload 0/1 are supported")
    if rows not in (16, 32, 64, 128):
        raise ValueError("row block must be 16/32/64/128")
    if causal and layout == "BNSD" and queries % 16:
        raise ValueError("BNSD causal Sq must be divisible by 16")
    dtype = F16
    flat_rows = queries * heads
    bnsd = layout == "BNSD"
    key_block = keys if cap_keys is None else cap_keys
    query_end = queries if query_end is None else query_end
    local_queries = query_end - query_start
    if keys not in (128, 256) or key_block not in (128, 256) or key_block > keys:
        raise ValueError("Skv and its resident prefix must be 128 or 256")
    if not 0 <= query_start < query_end <= queries:
        raise ValueError("invalid query interval")
    if causal and key_block < keys - queries + query_end:
        raise ValueError("resident prefix omits visible keys")
    if not causal and key_block != keys:
        raise ValueError("noncausal rows require all keys")
    assert not causal or queries <= keys
    while causal and rows > (queries if bnsd else heads):
        rows //= 2
    while (
        flat_rows % rows
        or local_queries * heads % rows
        or (causal and bnsd and (local_queries % rows or query_start % rows))
        or (causal and not bnsd and heads % (rows // 2))
        or 2 * (rows * (nope + rope) + key_block * (nope + rope) + rows * key_block)
        > 512 * 1024
    ):
        rows //= 2
    if rows < 16 or (causal and bnsd and (query_start % rows or local_queries % rows)):
        raise ValueError("query interval must contain complete native row blocks")
    assert flat_rows % rows == 0 and keys % key_block == 0
    assert key_block in (128, 256) and rows in (16, 32, 64, 128)
    half = rows // 2
    pitch = half + 1
    parts = 1
    row_blocks = local_queries * heads // rows
    width = min(256, nope)
    while nope % width:
        width -= 16
    scale = 1 / math.sqrt(nope + rope)
    shift = -math.sqrt(nope + rope)
    assert (
        2 * (rows * (nope + rope) + key_block * (nope + rope) + rows * key_block)
        <= 512 * 1024
    )

    @pl.vector_function
    def normalize(src, lo, hi, length0, row_step):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        mask16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        zero = vf.full(0.0, mask, dtype=F32)
        for row in pl.range(half):
            x0 = vf.load_align(src, [row, 0])
            x1 = vf.load_align(src, [row, 64])
            x0 = vf.muls(x0, scale, mask)
            x1 = vf.muls(x1, scale, mask)
            x0 = vf.adds(x0, shift, mask)
            x1 = vf.adds(x1, shift, mask)
            p0 = vf.exp(x0, mask)
            p1 = vf.exp(x1, mask)
            if causal:
                mask0 = vf.update_mask(pl.max(0, length0 + row * row_step), dtype=F32)
                mask1 = vf.update_mask(
                    pl.max(0, length0 + row * row_step - 64), dtype=F32
                )
                p0 = vf.select(p0, zero, mask0)
                p1 = vf.select(p1, zero, mask1)
            total = vf.add(p0, p1, mask)
            if key_block == 256:
                x2 = vf.load_align(src, [row, 128])
                x3 = vf.load_align(src, [row, 192])
                x2 = vf.muls(x2, scale, mask)
                x3 = vf.muls(x3, scale, mask)
                x2 = vf.adds(x2, shift, mask)
                x3 = vf.adds(x3, shift, mask)
                p2 = vf.exp(x2, mask)
                p3 = vf.exp(x3, mask)
                if causal:
                    mask2 = vf.update_mask(
                        pl.max(0, length0 + row * row_step - 128), dtype=F32
                    )
                    mask3 = vf.update_mask(
                        pl.max(0, length0 + row * row_step - 192), dtype=F32
                    )
                    p2 = vf.select(p2, zero, mask2)
                    p3 = vf.select(p3, zero, mask3)
                total_hi = vf.add(p2, p3, mask)
                total = vf.add(total, total_hi, mask)
            scalar_sum = vf.reduce_sum(total, mask)
            denominator = vf.full(scalar_sum, mask)
            p0 = vf.div(p0, denominator, mask)
            p1 = vf.div(p1, denominator, mask)
            h0 = vf.astype(p0, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
            h1 = vf.astype(p1, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
            packed, _ = vf.de_interleave(h0, h1)
            vf.store_align(
                lo,
                packed,
                mask16,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                block_stride=pitch,
                repeat_stride=1,
                post_update=True,
            )
            if key_block == 256:
                p2 = vf.div(p2, denominator, mask)
                p3 = vf.div(p3, denominator, mask)
                h2 = vf.astype(p2, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                h3 = vf.astype(p3, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                packed_hi, _ = vf.de_interleave(h2, h3)
                vf.store_align(
                    hi,
                    packed_hi,
                    mask16,
                    data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                    block_stride=pitch,
                    repeat_stride=1,
                    post_update=True,
                )

    @pl.pipeline.stage
    def qk(task, qn, qr, kn, kr, qng, qrg, kng, krg, lg, rg, ag, sg):
        b = task // (row_blocks * parts)
        m = task // parts % row_blocks * rows
        if bnsd:
            m = (m // local_queries) * queries + m % local_queries + query_start
        else:
            m = m + query_start * heads
        n = task % parts * key_block
        q0 = qng.current()
        q1 = qrg.current()
        k0 = kng.current()
        k1 = krg.current()
        acc = ag.current()
        pl.load(q0, qn, [b, m, 0], order=[1, 2])
        pl.load(q1, qr, [b, m, 0], order=[1, 2])
        pl.load(k0, kn, [b, n, 0], order=[1, 2])
        pl.load(k1, kr, [b, n, 0], order=[1, 2])
        pl.gemm(acc, q0, k0, l0a=lg, l0b=rg, trans_b=True, splitk=64)
        pl.gemm(acc, q1, k1, l0a=lg, l0b=rg, trans_b=True, splitk=64, init=False)
        dst = sg.next()
        pl.move(dst, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    @pl.pipeline.stage
    def soft(task, sg, lo_g, hi_g, pg):
        m = task // parts % row_blocks * rows
        if bnsd:
            m = (m // local_queries) * queries + m % local_queries + query_start
        else:
            m = m + query_start * heads
        m = m + pl.get_subblock_idx() * half
        src = sg.next()
        lo = lo_g.current()
        hi = hi_g.current()
        length0 = keys
        row_step = 0
        if causal:
            if bnsd:
                length0 = keys - queries + m % queries + 1
                row_step = 1
            else:
                length0 = keys - queries + m // heads + 1
        normalize(src, lo, hi, length0, row_step)
        pl.set_validshape(lo, [half, 128])
        p = pg.next()
        off = pl.get_subblock_idx() * half
        pl.insert(p, lo, [off, 0])
        if key_block == 256:
            pl.set_validshape(hi, [half, 128])
            pl.insert(p, hi, [off, 128])

    @pl.pipeline.stage
    def pv(task, pg, vg, lg, rg, ag, values, output):
        b = task // (row_blocks * parts)
        m = task // parts % row_blocks * rows
        if bnsd:
            m = (m // local_queries) * queries + m % local_queries + query_start
        else:
            m = m + query_start * heads
        part = task % parts
        p = pg.next()
        v = vg.current()
        acc = ag.current()
        for c in pl.range(nope // width):
            pl.load(v, values, [b, part * key_block, c * width], order=[1, 2])
            pl.gemm(acc, p, v, l0a=lg, l0b=rg, splitk=64)
            pl.store(output, acc, [b, m, c * width], order=[1, 2])

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=f"fused_mla_p{preload}_m{rows}_k{key_block}_qs{query_start}",
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def partial_kernel(
        qn: pl.Tensor[[batch, flat_rows, nope], dtype],
        qr: pl.Tensor[[batch, flat_rows, rope], dtype],
        kn: pl.Tensor[[batch, keys, nope], dtype],
        kr: pl.Tensor[[batch, keys, rope], dtype],
        values: pl.Tensor[[batch, keys, nope], dtype],
        output: pl.Tensor[[batch, flat_rows, nope], dtype],
    ):
        qng_type = pl.TileType(
            shape=[rows, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qng = pl.make_tile_group(type=qng_type, addrs=0, mutex_ids="auto", depth=1)
        qrg_type = pl.TileType(
            shape=[rows, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qrg = pl.make_tile_group(
            type=qrg_type, addrs=2 * nope * rows, mutex_ids="auto", depth=1
        )
        kng_type = pl.TileType(
            shape=[key_block, nope],
            dtype=dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        )
        kng = pl.make_tile_group(
            type=kng_type,
            addrs=2 * nope * rows + 2 * rope * rows,
            mutex_ids="auto",
            depth=1,
        )
        krg_type = pl.TileType(
            shape=[key_block, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        krg = pl.make_tile_group(
            type=krg_type,
            addrs=2 * nope * rows + 2 * rope * rows + 2 * key_block * nope,
            mutex_ids="auto",
            depth=1,
        )
        pg_type = pl.TileType(
            shape=[rows, key_block], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        pg = pl.make_tile_group(
            type=pg_type,
            addrs=2 * nope * rows
            + 2 * rope * rows
            + 2 * key_block * nope
            + 2 * key_block * rope,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[4],
            bwd_ids=[5],
        )
        ql_type = pl.TileType(
            shape=[rows, 64], dtype=dtype, target_memory=pl.MemorySpace.Left
        )
        ql = pl.make_tile_group(type=ql_type, addrs=0, mutex_ids="auto", depth=1)
        qr0_type = pl.TileType(
            shape=[64, key_block], dtype=dtype, target_memory=pl.MemorySpace.Right
        )
        qr0 = pl.make_tile_group(type=qr0_type, addrs=0, mutex_ids="auto", depth=1)
        vl_type = pl.TileType(
            shape=[rows, 64], dtype=dtype, target_memory=pl.MemorySpace.Left
        )
        vl = pl.make_tile_group(
            type=vl_type, addrs=128 * rows, mutex_ids="auto", depth=1
        )
        vr_type = pl.TileType(
            shape=[64, width], dtype=dtype, target_memory=pl.MemorySpace.Right
        )
        vr = pl.make_tile_group(
            type=vr_type, addrs=128 * key_block, mutex_ids="auto", depth=1
        )
        qa_type = pl.TileType(
            shape=[rows, key_block], dtype=F32, target_memory=pl.MemorySpace.Acc
        )
        qa = pl.make_tile_group(type=qa_type, addrs=0, mutex_ids="auto", depth=1)
        va_type = pl.TileType(
            shape=[rows, width], dtype=F32, target_memory=pl.MemorySpace.Acc
        )
        va = pl.make_tile_group(
            type=va_type, addrs=4 * key_block * rows, mutex_ids="auto", depth=1
        )
        sg_type = pl.TileType(
            shape=[half, key_block], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        sg = pl.make_tile_group(
            type=sg_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        lo_type = pl.TileType(
            shape=[pitch, 128],
            dtype=dtype,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        lo = pl.make_tile_group(
            type=lo_type, addrs=8 * half * key_block, mutex_ids="auto", depth=1
        )
        hi_type = pl.TileType(
            shape=[pitch, 128],
            dtype=dtype,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        hi = pl.make_tile_group(
            type=hi_type,
            addrs=8 * half * key_block + 256 * pitch,
            mutex_ids="auto",
            depth=1,
        )
        vg = pl.reinterpret(kng, shape=[key_block, width])
        group = pl.get_block_idx() // pl.get_subblock_num()
        for task in pl.range(group, batch * row_blocks * parts, pl.get_block_num()):
            with pl.section_cube():
                qk(task, qn, qr, kn, kr, qng, qrg, kng, krg, ql, qr0, qa, sg)
            with pl.section_vector():
                soft(task, sg, lo, hi, pg)
            with pl.section_cube():
                pv(task, pg, vg, vl, vr, va, values, output)

    return partial_kernel


def mla_fused(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    values,
    *,
    layout="BSND",
    causal=False,
    workers=28,
    preload=1,
):
    """Metadata-only wrapper for the bounded FP16 fused path."""
    import torch

    if layout not in ("BSND", "BNSD") or q_nope.ndim != 4:
        raise ValueError("rank-4 BSND/BNSD tensors are required")
    batch, dim1, dim2, nope = q_nope.shape
    queries, heads = (dim1, dim2) if layout == "BSND" else (dim2, dim1)
    keys = k_nope.shape[1 if layout == "BSND" else 2]
    rope = q_rope.shape[-1]
    key_shape = (batch, keys, 1, nope) if layout == "BSND" else (batch, 1, keys, nope)
    expected = [
        tuple(q_nope.shape[:-1]) + (rope,),
        key_shape,
        key_shape[:-1] + (rope,),
        key_shape,
    ]
    tensors = (q_nope, q_rope, k_nope, k_rope, values)
    if [tuple(t.shape) for t in tensors[1:]] != expected:
        raise ValueError("input shapes do not match the single-KV-head contract")
    if any(
        t.dtype != torch.float16 or t.device != q_nope.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("inputs must be contiguous FP16 tensors on the same device")
    out = torch.empty(tuple(q_nope.shape), dtype=q_nope.dtype, device=q_nope.device)
    flat = (
        q_nope.view(batch, queries * heads, nope),
        q_rope.view(batch, queries * heads, rope),
        k_nope.view(batch, keys, nope),
        k_rope.view(batch, keys, rope),
        values.view(batch, keys, nope),
        out.view(batch, queries * heads, nope),
    )
    step = 128 if causal else queries
    for start in range(0, queries, step):
        end = min(start + step, queries)
        prefix = ((keys - queries + end + 127) // 128) * 128 if causal else keys
        kernel = make_fused(
            batch,
            queries,
            keys,
            heads,
            nope,
            rope,
            "float16",
            layout,
            causal,
            workers,
            cap_keys=prefix,
            query_start=start,
            query_end=end,
            preload=preload,
        )
        kernel[None, workers](*flat)
    return out
