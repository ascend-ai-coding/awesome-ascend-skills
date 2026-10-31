# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。有限分数域、单 KV head 的原生 MLA。

Q/K 的 nope、rope 两部分在 FP32 Acc 中累加，默认 scale=1/sqrt(Dn+Dr)，
输出为 softmax(QK*scale)@V；V 是独立输入。支持连续 BSND/BNSD、FP16/BF16，
causal 使用右对齐前缀。FP16 用三个原生 launch 计算分数、概率和输出。
BF16 将系数精确放大 2^32，保留 PV 的 FP32 放大结果，再用一个 VF launch
通过整数位运算还原指数并按 nearest-even 写 BF16，保留正常数/次正规数边界。
直接在 Fixpipe 乘 2^-32 会在 FP32 下冲零；这里不放宽比较阈值。

契约：所有输入有限且 abs <= 1；B、Sq 为正；H 为 16 的正倍数；Skv 为 128
的正倍数且 <=2048；Dn 为 64 的正倍数且 <=512；Dr 为 64 或 128。
causal 要求 Sq<=Skv。固定参考 sqrt(Dn+Dr) 将指数参数约束在 [-2*sqrt(Dn+Dr),0]，
分母在 FP32 正常范围，概率归一化后再窄化。未知分数域应改用稳定最大值/在线归一化。

QK 保留 Q 复用相邻 key 面板，L0 双槽；VF 按整块搬运，整块写完指数和分母后
才执行一次 VST→VLD 屏障；PV 使用两个 K 面板槽提前发起下一次 GM 载入。
行/head 折叠只改变存储坐标，causal 仍从原 query 坐标导出。归一化按 query
分组、沿 head 轴读取一批行，BSND/BNSD 的 GM 步长由各自四维 view 保留。
全可见行省略前缀掩码；右对齐的单 query 因果行也属于此分支。

最大资源：QK Mat=2*(256+128)*640=480 KiB，Left/Right/Acc=64/32/128 KiB；
PV Mat=2*2*256*(128+256)=384 KiB，Left/Right/Acc=32/64/128 KiB；
VF Vec=16*(6*2048+4*64)=196 KiB；BF16 写回阶段 Vec 至多 96 KiB。
各阶段同时存活的块均独立计入。

默认块形由整除、容量和可用工作量决定，非逐 case 分派。workers 是调用方确认的
Cube 执行预算，Vector 使用两倍预算；不同设备/stream 应传入各自可用的预算。
少量任务时按 ceil(任务数/workers)*单块工作量估算最忙核；继续拆块若只增加
执行轮数而不降低该估算，就保留较大的复用块。这是候选启发式，仍需实际计时。
中间 GM 空间为 6*B*Sq*H*Skv 字节，BF16 另需 4*B*Sq*H*Dn 字节；
全矩阵空间不被允许时不适用此数据流。
"""

from functools import lru_cache
import math

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32


def _block_divisor(size, maximum):
    block = maximum
    while size % block:
        block //= 2
    return block


def _peak_work(items, work_per_item, workers):
    return ((items + workers - 1) // workers) * work_per_item


@lru_cache(maxsize=64)
def make_kernels(
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
):
    """Return score, normalization, product and optional BF16 finalization entries."""
    if min(batch, queries, heads, workers) < 1 or heads % 16:
        raise ValueError("positive B/Sq/workers and H divisible by 16 are required")
    if keys < 128 or keys > 2048 or keys % 128:
        raise ValueError("Skv must be a multiple of 128 in [128, 2048]")
    if not 64 <= nope <= 512 or nope % 64 or rope not in (64, 128):
        raise ValueError(
            "Dn must be a multiple of 64 in [64, 512]; Dr must be 64 or 128"
        )
    if layout not in ("BSND", "BNSD") or dtype_name not in ("float16", "bfloat16"):
        raise ValueError("supported layouts: BSND/BNSD; dtypes: float16/bfloat16")
    if causal and queries > keys:
        raise ValueError("right-aligned causal attention requires Sq <= Skv")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    bf16 = dtype_name == "bfloat16"
    coefficient_scale = 4294967296.0
    product_dtype = FP32 if bf16 else dtype
    bnsd = layout == "BNSD"
    dim1, dim2 = (heads, queries) if bnsd else (queries, heads)
    rows = queries * heads
    scale = 1 / math.sqrt(nope + rope)
    shift = -math.sqrt(nope + rope)
    score_m = _block_divisor(rows, 256)
    while batch * (rows // score_m) * (keys // 128) < workers and score_m > 16:
        items = batch * (rows // score_m) * (keys // 128)
        if _peak_work(2 * items, score_m // 2, workers) >= _peak_work(
            items, score_m, workers
        ):
            break
        score_m //= 2
    reuse = (
        2
        if keys % 256 == 0 and batch * (rows // score_m) * (keys // 256) >= workers
        else 1
    )
    score_groups = keys // (128 * reuse)
    vector_rows = 16
    while batch * rows // vector_rows < 2 * workers and vector_rows > 1:
        items = batch * rows // vector_rows
        if _peak_work(2 * items, vector_rows // 2, 2 * workers) >= _peak_work(
            items, vector_rows, 2 * workers
        ):
            break
        vector_rows //= 2
    product_m = _block_divisor(rows, 128)
    product_n = _block_divisor(nope, 256)
    while batch * (rows // product_m) * (nope // product_n) < workers:
        items = batch * (rows // product_m) * (nope // product_n)
        area = product_m * product_n
        if _peak_work(2 * items, area // 2, workers) >= _peak_work(
            items, area, workers
        ):
            break
        if product_n > product_m and product_n > 64:
            product_n //= 2
        elif product_m > 16:
            product_m //= 2
        elif product_n > 64:
            product_n //= 2
        else:
            break
    product_k = _block_divisor(keys, 256)
    panels = keys // product_k
    depth = 2 if panels > 1 else 1
    output_count = batch * rows * nope
    final_block = _block_divisor(output_count, 8192)

    @pl.jit(arch="a5", auto_mutex=True)
    def score_kernel(
        qn: pl.Tensor[[batch, rows, nope], dtype],
        qr: pl.Tensor[[batch, rows, rope], dtype],
        kn: pl.Tensor[[batch, keys, nope], dtype],
        kr: pl.Tensor[[batch, keys, rope], dtype],
        scores: pl.Tensor[[batch, rows, keys], FP32],
    ):
        qng_type = pl.TileType(
            shape=[score_m, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qng = pl.make_tile_group(type=qng_type, addrs=0, mutex_ids="auto", depth=1)
        qrg_type = pl.TileType(
            shape=[score_m, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        qrg = pl.make_tile_group(
            type=qrg_type, addrs=2 * nope * score_m, mutex_ids="auto", depth=1
        )
        kng_type = pl.TileType(
            shape=[128, nope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        kng = pl.make_tile_group(
            type=kng_type,
            addrs=2 * nope * score_m + 2 * rope * score_m,
            mutex_ids="auto",
            depth=1,
        )
        krg_type = pl.TileType(
            shape=[128, rope], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        krg = pl.make_tile_group(
            type=krg_type,
            addrs=2 * nope * score_m + 2 * rope * score_m + 256 * nope,
            mutex_ids="auto",
            depth=1,
        )
        left_type = pl.TileType(
            shape=[score_m, 64], dtype=dtype, target_memory=pl.MemorySpace.Left
        )
        left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids="auto", depth=2)
        right_type = pl.TileType(
            shape=[64, 128], dtype=dtype, target_memory=pl.MemorySpace.Right
        )
        right = pl.make_tile_group(type=right_type, addrs=0, mutex_ids="auto", depth=2)
        ag_type = pl.TileType(
            shape=[score_m, 128], dtype=FP32, target_memory=pl.MemorySpace.Acc
        )
        ag = pl.make_tile_group(type=ag_type, addrs=0, mutex_ids="auto", depth=1)
        with pl.section_cube():
            for task in pl.range(
                pl.get_block_idx(),
                batch * (rows // score_m) * score_groups,
                pl.get_block_num(),
            ):
                b = task // ((rows // score_m) * score_groups)
                m0 = (task // score_groups % (rows // score_m)) * score_m
                nbase = (task % score_groups) * reuse * 128
                a0 = qng.current()
                a1 = qrg.current()
                pl.load(a0, qn, [b, m0, 0], order=[1, 2])
                pl.load(a1, qr, [b, m0, 0], order=[1, 2])
                for part in pl.range(reuse):
                    n0 = nbase + part * 128
                    b0 = kng.current()
                    b1 = krg.current()
                    acc = ag.current()
                    pl.load(b0, kn, [b, n0, 0], order=[1, 2])
                    pl.load(b1, kr, [b, n0, 0], order=[1, 2])
                    pl.gemm(acc, a0, b0, l0a=left, l0b=right, trans_b=True, splitk=64)
                    pl.gemm(
                        acc,
                        a1,
                        b1,
                        l0a=left,
                        l0b=right,
                        trans_b=True,
                        splitk=64,
                        init=False,
                    )
                    pl.store(scores, acc, [b, m0, n0], order=[1, 2])

    @pl.vector_function
    def normalize_block(src, dst, totals, length):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        zero = vf.full(0.0, mask, dtype=FP32)
        for row in pl.range(vector_rows):
            lane_sum = vf.full(0.0, mask, dtype=FP32)
            for c in pl.range(keys // 64):
                x = vf.load_align(src, [row, c * 64])
                scaled = vf.muls(x, scale, mask)
                shifted = vf.adds(scaled, shift, mask)
                weight = vf.exp(shifted, mask)
                if causal and queries > 1:
                    prefix = vf.update_mask(pl.max(0, length - c * 64), dtype=FP32)
                    valid = vf.select(weight, zero, prefix)
                else:
                    valid = weight
                lane_sum = vf.add(lane_sum, valid, mask)
                vf.store_align(src, valid, mask, [row, c * 64])
            total = vf.reduce_sum(lane_sum, mask)
            denominator = vf.full(total, mask)
            vf.store_align(totals, denominator, mask, [row, 0])
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        for row in pl.range(vector_rows):
            denominator = vf.load_align(totals, [row, 0])
            for c in pl.range(keys // 64):
                weight = vf.load_align(src, [row, c * 64])
                p = vf.div(weight, denominator, mask)
                if bf16:
                    p = vf.muls(p, coefficient_scale, mask)
                narrow = vf.astype(p, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                vf.store_align(
                    dst, narrow, mask, [row, c * 64], dist=pl.StoreDist.PACK_B32
                )

    @pl.jit(arch="a5", auto_mutex=True)
    def normalize_kernel(
        scores: pl.Tensor[[batch, dim1, dim2, keys], FP32],
        probabilities: pl.Tensor[[batch, dim1, dim2, keys], dtype],
    ):
        sg_type = pl.TileType(
            shape=[vector_rows, keys], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        sg = pl.make_tile_group(type=sg_type, addrs=0, mutex_ids="auto", depth=1)
        pg_type = pl.TileType(
            shape=[vector_rows, keys], dtype=dtype, target_memory=pl.MemorySpace.Vec
        )
        pg = pl.make_tile_group(
            type=pg_type, addrs=4 * keys * vector_rows, mutex_ids="auto", depth=1
        )
        tg_type = pl.TileType(
            shape=[vector_rows, 64], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        tg = pl.make_tile_group(
            type=tg_type, addrs=6 * keys * vector_rows, mutex_ids="auto", depth=1
        )
        with pl.section_vector():
            for task in pl.range(
                pl.get_block_idx(), batch * rows // vector_rows, pl.get_block_num()
            ):
                b = task // (rows // vector_rows)
                q = task // (heads // vector_rows) % queries
                h = task % (heads // vector_rows) * vector_rows
                src = sg.current()
                dst = pg.current()
                totals = tg.current()
                if bnsd:
                    pl.load(src, scores, [b, h, q, 0], order=[1, 3])
                else:
                    pl.load(src, scores, [b, q, h, 0], order=[2, 3])
                length = keys
                if causal:
                    length = keys - queries + q + 1
                normalize_block(src, dst, totals, length)
                if bnsd:
                    pl.store(probabilities, dst, [b, h, q, 0], order=[1, 3])
                else:
                    pl.store(probabilities, dst, [b, q, h, 0], order=[2, 3])

    @pl.jit(arch="a5", auto_mutex=True)
    def product_kernel(
        probabilities: pl.Tensor[[batch, rows, keys], dtype],
        values: pl.Tensor[[batch, keys, nope], dtype],
        output: pl.Tensor[[batch, rows, nope], product_dtype],
    ):
        pg_type = pl.TileType(
            shape=[product_m, product_k], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        pg = pl.make_tile_group(type=pg_type, addrs=0, mutex_ids="auto", depth=depth)
        vg_type = pl.TileType(
            shape=[product_k, product_n], dtype=dtype, target_memory=pl.MemorySpace.Mat
        )
        vg = pl.make_tile_group(
            type=vg_type,
            addrs=2 * depth * product_k * product_m,
            mutex_ids="auto",
            depth=depth,
        )
        left_type = pl.TileType(
            shape=[product_m, 128], dtype=dtype, target_memory=pl.MemorySpace.Left
        )
        left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids="auto", depth=1)
        right_type = pl.TileType(
            shape=[128, product_n], dtype=dtype, target_memory=pl.MemorySpace.Right
        )
        right = pl.make_tile_group(type=right_type, addrs=0, mutex_ids="auto", depth=1)
        ag_type = pl.TileType(
            shape=[product_m, product_n], dtype=FP32, target_memory=pl.MemorySpace.Acc
        )
        ag = pl.make_tile_group(type=ag_type, addrs=0, mutex_ids="auto", depth=1)
        with pl.section_cube():
            for task in pl.range(
                pl.get_block_idx(),
                batch * (rows // product_m) * (nope // product_n),
                pl.get_block_num(),
            ):
                b = task // ((rows // product_m) * (nope // product_n))
                m0 = (task // (nope // product_n) % (rows // product_m)) * product_m
                d0 = task % (nope // product_n) * product_n
                p = pg[0]
                v = vg[0]
                acc = ag.current()
                pl.load(p, probabilities, [b, m0, 0], order=[1, 2])
                pl.load(v, values, [b, 0, d0], order=[1, 2])
                if panels > 1:
                    pn = pg[1]
                    vn = vg[1]
                    pl.load(pn, probabilities, [b, m0, product_k], order=[1, 2])
                    pl.load(vn, values, [b, product_k, d0], order=[1, 2])
                pl.gemm(acc, p, v, l0a=left, l0b=right, splitk=128)
                for k in pl.range(1, panels):
                    if k + 1 < panels:
                        slot_next = (k + 1) % 2
                        pn = pg[slot_next]
                        vn = vg[slot_next]
                        pl.load(
                            pn,
                            probabilities,
                            [b, m0, (k + 1) * product_k],
                            order=[1, 2],
                        )
                        pl.load(vn, values, [b, (k + 1) * product_k, d0], order=[1, 2])
                    slot = k % 2
                    p = pg[slot]
                    v = vg[slot]
                    pl.gemm(acc, p, v, l0a=left, l0b=right, splitk=128, init=False)
                pl.store(output, acc, [b, m0, d0], order=[1, 2])

    @pl.vector_function
    def unscale_bf16(src, dst):
        # For a finite FP32 value with biased exponent e and 24-bit mantissa M,
        # x/2^32 in BF16 uses shift=max(49-e,16), exponent=max(e-33,0).
        # e<25 rounds to signed zero. Clamp shifts before evaluating both paths.
        # Integer RNE uses (M + 2^(shift-1) - 1 + ((M>>shift)&1)) >> shift.
        # Pipeline bounds give abs(x)<=2^33; NaN/Inf encodings are not inputs.
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        one = vf.full(1, mask, dtype=pl.DT_UINT32)
        zero = vf.full(0, mask, dtype=pl.DT_UINT32)
        exp_mask = vf.full(255, mask, dtype=pl.DT_UINT32)
        mant_mask = vf.full(8388607, mask, dtype=pl.DT_UINT32)
        hidden = vf.full(8388608, mask, dtype=pl.DT_UINT32)
        sign_mask = vf.full(32768, mask, dtype=pl.DT_UINT32)
        c33 = vf.full(33, mask, dtype=pl.DT_UINT32)
        c49 = vf.full(49, mask, dtype=pl.DT_UINT32)
        c25 = vf.full(25, mask, dtype=pl.DT_UINT32)
        for c in pl.range(final_block // 64):
            x = vf.load_align(src, [0, c * 64])
            bits = vf.bit_cast(x, dtype=pl.DT_UINT32)
            high = vf.shift_right(bits, 23, mask)
            exponent = vf.and_(high, exp_mask, mask)
            low = vf.and_(bits, mant_mask, mask)
            mantissa = vf.or_(low, hidden, mask)
            clipped = vf.maxs(exponent, 25, mask)
            clipped = vf.mins(clipped, 33, mask)
            shift_bits = vf.sub(c49, clipped, mask)
            shift_count = vf.bit_cast(shift_bits, dtype=pl.DT_INT32)
            truncated = vf.shift_right(mantissa, shift_count, mask)
            parity = vf.and_(truncated, one, mask)
            half_shift_bits = vf.sub(shift_bits, one, mask)
            half_shift = vf.bit_cast(half_shift_bits, dtype=pl.DT_INT32)
            half = vf.shift_left(one, half_shift, mask)
            round_bias = vf.sub(half, one, mask)
            round_bias = vf.add(round_bias, parity, mask)
            rounded = vf.add(mantissa, round_bias, mask)
            fraction = vf.shift_right(rounded, shift_count, mask)
            too_small = vf.lt(exponent, c25, mask)
            fraction = vf.select(zero, fraction, too_small)
            normal_exponent = vf.maxs(exponent, 33, mask)
            normal_exponent = vf.sub(normal_exponent, c33, mask)
            normal_exponent = vf.shift_left(normal_exponent, 7, mask)
            magnitude = vf.add(normal_exponent, fraction, mask)
            upper = vf.shift_right(bits, 16, mask)
            sign = vf.and_(upper, sign_mask, mask)
            encoded = vf.or_(magnitude, sign, mask)
            narrow = vf.astype(
                encoded, mask, dtype=pl.DT_UINT16, layout=pl.CastLayout.ZERO
            )
            result = vf.bit_cast(narrow, dtype=pl.DT_BF16)
            vf.store_align(dst, result, mask, [0, c * 64], dist=pl.StoreDist.PACK_B32)

    @pl.jit(arch="a5", auto_mutex=True)
    def finalize_kernel(
        src: pl.Tensor[[output_count], FP32], dst: pl.Tensor[[output_count], pl.DT_BF16]
    ):
        sg_type = pl.TileType(
            shape=[1, final_block], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        sg = pl.make_tile_group(type=sg_type, addrs=0, mutex_ids="auto", depth=2)
        dg_type = pl.TileType(
            shape=[1, final_block], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        )
        dg = pl.make_tile_group(
            type=dg_type, addrs=8 * final_block, mutex_ids="auto", depth=2
        )
        with pl.section_vector():
            for task in pl.range(
                pl.get_block_idx(), output_count // final_block, pl.get_block_num()
            ):
                source = sg.next()
                target = dg.next()
                pl.load(source, src, [task * final_block])
                unscale_bf16(source, target)
                pl.store(dst, target, [task * final_block])

    return (
        score_kernel,
        normalize_kernel,
        product_kernel,
        finalize_kernel if bf16 else None,
    )


def mla(
    q_nope, q_rope, k_nope, k_rope, values, *, layout="BSND", causal=False, workers=28
):
    """Metadata-only wrapper; allocations and three FP16 or four BF16 launches."""
    import torch

    if layout not in ("BSND", "BNSD") or q_nope.ndim != 4:
        raise ValueError("rank-4 BSND/BNSD tensors are required")
    batch, dim1, dim2, nope = q_nope.shape
    queries, heads = (dim1, dim2) if layout == "BSND" else (dim2, dim1)
    keys = k_nope.shape[1 if layout == "BSND" else 2]
    rope = q_rope.shape[-1]
    k_shape = (batch, keys, 1, nope) if layout == "BSND" else (batch, 1, keys, nope)
    shapes = [
        tuple(q_nope.shape[:-1]) + (rope,),
        k_shape,
        k_shape[:-1] + (rope,),
        k_shape,
    ]
    tensors = (q_nope, q_rope, k_nope, k_rope, values)
    if [tuple(t.shape) for t in tensors[1:]] != shapes:
        raise ValueError(
            "input shapes must share B/heads and the two feature partitions"
        )
    if q_nope.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("FP16 or BF16 inputs are required")
    if any(
        t.dtype != q_nope.dtype or t.device != q_nope.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("inputs must be contiguous and share dtype/device")
    score, normalize, product, finalize = make_kernels(
        batch,
        queries,
        keys,
        heads,
        nope,
        rope,
        str(q_nope.dtype).split(".")[-1],
        layout,
        causal,
        workers,
    )
    rows = queries * heads
    scores = torch.empty((batch, rows, keys), dtype=torch.float32, device=q_nope.device)
    probabilities = torch.empty(
        (batch, rows, keys), dtype=q_nope.dtype, device=q_nope.device
    )
    output = torch.empty(tuple(q_nope.shape), dtype=q_nope.dtype, device=q_nope.device)
    score[None, workers](
        q_nope.view(batch, rows, nope),
        q_rope.view(batch, rows, rope),
        k_nope.view(batch, keys, nope),
        k_rope.view(batch, keys, rope),
        scores,
    )
    soft_shape = (
        (batch, heads, queries, keys)
        if layout == "BNSD"
        else (batch, queries, heads, keys)
    )
    normalize[None, 2 * workers](
        scores.view(soft_shape), probabilities.view(soft_shape)
    )
    if finalize is None:
        product[None, workers](
            probabilities,
            values.view(batch, keys, nope),
            output.view(batch, rows, nope),
        )
    else:
        scaled = torch.empty(
            (batch, rows, nope), dtype=torch.float32, device=q_nope.device
        )
        product[None, workers](probabilities, values.view(batch, keys, nope), scaled)
        finalize[None, 2 * workers](scaled.view(-1), output.view(-1))
    return output
