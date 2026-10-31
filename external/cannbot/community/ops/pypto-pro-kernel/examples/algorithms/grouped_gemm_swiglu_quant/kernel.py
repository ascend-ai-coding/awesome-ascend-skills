# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。分组 INT8 GEMM、VF SwiGLU 与逐行量化，保留分配、驻留和批处理对照。

契约：X[M,K]、W[G,K,N] 为 INT8；FP32 x_scale[M]、weight_scale[G,N] 为有限非负数；运行期 INT32 offsets[G+1] 单调，首0末M。输出 Y[M,N/2] INT8、
y_scale[M] FP32，另由调用者提供 mm[M,N] INT32 工作区。M=1..4096，G=1..32，K=16..8192 且为16的倍数，N=64..10240 且为32的倍数。输入整数积和不超过
8192*128*128，INT32可精确容纳；反量化和激活须有限。

计算：mm_g=X_g@W_g；z=FP32(mm)*x_scale*weight_scale；a=SiLU(z_left)*z_right；y_scale=max(max(abs(a)),FP32.tiny)/127；
Y=clip(rne(a/y_scale),-128,127)。零行的除数选1，输出仍为0，不构造溢出的倒数。

两次原生 launch：Cube 先写完整 mm，Vector 再消费；不使用跨核交接。global_work=True 将各组的 M/N tile 连续编号，Vector 按全局行号分配；False 每组重新从 worker 开始，是算术
及几何相同的性能对照。两条路径均只在本 worker 有行时加载 scale，空组和无任务 worker 不产生无消费者的预载。offsets 始终由设备读取，Host不读回分组内容。

每核资源：M块128、N块256，INT32 Acc单槽128 KiB；L0A/L0B各双槽，K步长<=128，分别<=32/64 KiB。L1的K驻留范围与L0步长独立，双槽A/B总量<=384 KiB（l1_k=512）；
pl.gemm(splitk=...)只在已加载的L1面板内分段。最后一个外层K面板显式传编译期有效K。UB每行物理宽度向64个FP32元素补齐；双槽总量为2*(5*row_width*4+y_width+256)，N上限时215552
B。所有DMA有效形状与VF尾掩码分别设置；VF内act写后读使用VST_VLD，下一行覆盖前使用VLD_VST。

reuse_l0=True把两个N块交给同一任务：两个独立Acc槽共享一次L0A搬运，使用gemm-desugar的保留L0操作数例外；默认M128/N256时Acc两槽共256 KiB，L1 K面板上限256，双槽A/B共320 KiB。
row_batch>1把独立行合并DMA和VF调用，先写全部行的激活和统计，再用一次VST_VLD进入量化。统计tile每行128个FP32元素分别存scale与截断到normal下限的amax；量化先形成a/amax，再乘127，不计算可
能溢出的1/y_scale。两条算术路径各自接受独立参考的固定判据，映射对照保持算术相同。quant_buffer_bytes统一计算批处理的UB容量。

本例教分配与存储层次，不把全局轮转或某个块大小当作普适最优。test.py保留不同分配、L0复用与行批处理的独立数值对照；配对性能使用skill的公共计时工具，驻留输入计时不等同于外部评测器的清缓存协议。
grouped_matmul_swiglu_quant接收Host已知的累计分组属性，以这些原生factory组合完整任务；设备offsets仍由各kernel读取，Host不读取设备内容。配置选择仅是指定工作负载族的起点，设备、形状或分布
变化后须重新测量。
"""

from functools import lru_cache

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

I8, I32, F32 = pl.DT_INT8, pl.DT_INT32, pl.DT_FP32
M_BLOCK, N_BLOCK, LANES = 128, 256, 64


@lru_cache(maxsize=None)
def gemm_kernel(
    groups,
    inner,
    cols,
    l1_k=256,
    l0_k=128,
    global_work=True,
    m_block=M_BLOCK,
    n_block=N_BLOCK,
    l1_depth=2,
):
    resident = min(l1_k, ((inner + 31) // 32) * 32)
    step = min(l0_k, resident)
    first_k = min(inner, resident)
    full_end = inner // resident * resident
    tail = inner % resident
    n_tiles = (cols + n_block - 1) // n_block

    @pl.jit(
        auto_mutex=True,
        name=f"grouped_gemm_{global_work}_{groups}_{inner}_{cols}_{m_block}_{n_block}_{resident}_{step}_{l1_depth}",
    )
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, inner], I8],
        weight: pl.Tensor[[groups, inner, cols], I8],
        offsets: pl.Tensor[[groups + 1], I32],
        mm: pl.Tensor[[pl.DYNAMIC, cols], I32],
    ):
        a_l1_type = pl.TileType(
            shape=[m_block, resident],
            dtype=I8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        a_l1 = pl.make_tile_group(
            type=a_l1_type, addrs=0, mutex_ids="auto", depth=l1_depth
        )
        b_l1_type = pl.TileType(
            shape=[resident, n_block],
            dtype=I8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        b_l1 = pl.make_tile_group(
            type=b_l1_type,
            addrs=l1_depth * m_block * resident,
            mutex_ids="auto",
            depth=l1_depth,
        )
        left_type = pl.TileType(
            shape=[m_block, step],
            dtype=I8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        )
        left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids="auto", depth=2)
        right_type = pl.TileType(
            shape=[step, n_block],
            dtype=I8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        )
        right = pl.make_tile_group(type=right_type, addrs=0, mutex_ids="auto", depth=2)
        acc_type = pl.TileType(
            shape=[m_block, n_block],
            dtype=I32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            compact=1,
        )
        acc = pl.make_tile_group(type=acc_type, addrs=0, mutex_ids="auto", depth=1)
        worker = pl.get_block_idx()
        workers = pl.get_block_num()
        with pl.section_cube():
            prefix = 0
            for group in pl.range(groups):
                begin = pl.getval(offsets, group)
                end = pl.getval(offsets, group + 1)
                count = ((end - begin + m_block - 1) // m_block) * n_tiles
                if global_work:
                    first = (worker + workers - prefix % workers) % workers
                else:
                    first = worker
                # Every worker advances the same prefix, including empty owners.
                prefix = prefix + count
                for tile in pl.range(first, count, workers):
                    row = begin + (tile // n_tiles) * m_block
                    col = (tile % n_tiles) * n_block
                    rows = pl.min(m_block, end - row)
                    width = pl.min(n_block, cols - col)
                    result = acc.next()
                    a = a_l1.next()
                    b = b_l1.next()
                    pl.set_validshape(a, [rows, first_k])
                    pl.set_validshape(b, [first_k, width])
                    pl.load(a, x, [row, 0])
                    pl.load(b, weight, [group, 0, col])
                    pl.gemm(
                        result,
                        a,
                        b,
                        l0a=left,
                        l0b=right,
                        m=rows,
                        n=width,
                        k=first_k,
                        splitk=step,
                    )
                    for k0 in pl.range(resident, full_end, resident):
                        a = a_l1.next()
                        b = b_l1.next()
                        pl.set_validshape(a, [rows, resident])
                        pl.set_validshape(b, [resident, width])
                        pl.load(a, x, [row, k0])
                        pl.load(b, weight, [group, k0, col])
                        pl.gemm(
                            result,
                            a,
                            b,
                            l0a=left,
                            l0b=right,
                            m=rows,
                            n=width,
                            k=resident,
                            splitk=step,
                            init=False,
                        )
                    if full_end > 0 and tail > 0:
                        a = a_l1.next()
                        b = b_l1.next()
                        pl.set_validshape(a, [rows, tail])
                        pl.set_validshape(b, [tail, width])
                        pl.load(a, x, [row, full_end])
                        pl.load(b, weight, [group, full_end, col])
                        pl.gemm(
                            result,
                            a,
                            b,
                            l0a=left,
                            l0b=right,
                            m=rows,
                            n=width,
                            k=tail,
                            splitk=step,
                            init=False,
                        )
                    pl.store(mm, result, [row, col])

    return kernel


@lru_cache(maxsize=None)
def quant_kernel(groups, cols, global_work=True):
    half = cols // 2
    row_width = (half + LANES - 1) // LANES * LANES
    y_width = (half + 31) // 32 * 32

    @pl.vector_function
    def quantize(ml, mr, wl, wr, act, yrow, scale, xs: pl.DT_FP32):
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        maximum = vf.full(0.0, full, dtype=F32)
        for col in pl.range(0, half, LANES):
            active = pl.min(LANES, half - col)
            mf = vf.update_mask(active, dtype=F32)
            mi = vf.update_mask(active, dtype=I32)
            il = vf.load_align(ml, [0, col])
            left = vf.astype(il, mi, dtype=F32)
            left = vf.muls(left, xs, mf)
            wleft = vf.load_align(wl, [0, col])
            left = vf.mul(left, wleft, mf)
            ir = vf.load_align(mr, [0, col])
            right = vf.astype(ir, mi, dtype=F32)
            right = vf.muls(right, xs, mf)
            wright = vf.load_align(wr, [0, col])
            right = vf.mul(right, wright, mf)
            negative = vf.neg(left, mf)
            exponential = vf.exp(negative, mf)
            denominator = vf.adds(exponential, 1.0, mf)
            silu = vf.div(left, denominator, mf)
            value = vf.mul(silu, right, mf)
            vf.store_align(act, value, mf, [0, col])
            absolute = vf.abs(value, mf)
            # A partial tail must not clear accumulated maxima in other lanes.
            maximum = vf.max(maximum, absolute, full)
        reduced = vf.reduce_max(maximum, full)
        row_max = vf.full(reduced, full)
        bounded = vf.maxs(row_max, pl.const(1.1754943508222875e-38, F32), full)
        y_scale = vf.muls(bounded, pl.const(0.007874015748031496, F32), full)
        first = vf.update_mask(1, dtype=F32)
        vf.store_align(scale, y_scale, first, [0, 0])
        zero = vf.full(0.0, full, dtype=F32)
        one = vf.full(1.0, full, dtype=F32)
        is_zero = vf.eq(row_max, zero, full)
        divisor = vf.select(one, y_scale, is_zero)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        for col in pl.range(0, half, LANES):
            active = pl.min(LANES, half - col)
            mf = vf.update_mask(active, dtype=F32)
            mi = vf.update_mask(active, dtype=I32)
            value = vf.load_align(act, [0, col])
            scaled = vf.div(value, divisor, mf)
            integer = vf.astype(
                scaled, mf, dtype=I32, round_mode=pl.VFRoundMode.CAST_RINT
            )
            low = vf.maxs(integer, pl.const(-128, I32), mi)
            clipped = vf.mins(low, pl.const(127, I32), mi)
            vf.store_align(yrow, clipped, mi, [0, col], dist=pl.StoreDist.PACK4_B32)

    @pl.jit(auto_mutex=True, name=f"grouped_quant_{global_work}_{groups}_{cols}")
    def kernel(
        mm: pl.Tensor[[pl.DYNAMIC, cols], I32],
        weight_scale: pl.Tensor[[groups, cols], F32],
        x_scale: pl.Tensor[[pl.DYNAMIC], F32],
        offsets: pl.Tensor[[groups + 1], I32],
        y: pl.Tensor[[pl.DYNAMIC, half], I8],
        y_scale: pl.Tensor[[pl.DYNAMIC], F32],
    ):
        wl_group_type = pl.TileType(
            shape=[1, row_width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        wl_group = pl.make_tile_group(
            type=wl_group_type, addrs=0, mutex_ids="auto", depth=2
        )
        wr_group_type = pl.TileType(
            shape=[1, row_width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        wr_group = pl.make_tile_group(
            type=wr_group_type, addrs=8 * row_width, mutex_ids="auto", depth=2
        )
        ml_group_type = pl.TileType(
            shape=[1, row_width], dtype=I32, target_memory=pl.MemorySpace.Vec
        )
        ml_group = pl.make_tile_group(
            type=ml_group_type, addrs=16 * row_width, mutex_ids="auto", depth=2
        )
        mr_group_type = pl.TileType(
            shape=[1, row_width], dtype=I32, target_memory=pl.MemorySpace.Vec
        )
        mr_group = pl.make_tile_group(
            type=mr_group_type, addrs=24 * row_width, mutex_ids="auto", depth=2
        )
        act_group_type = pl.TileType(
            shape=[1, row_width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        act_group = pl.make_tile_group(
            type=act_group_type, addrs=32 * row_width, mutex_ids="auto", depth=2
        )
        y_group_type = pl.TileType(
            shape=[1, y_width], dtype=I8, target_memory=pl.MemorySpace.Vec
        )
        y_group = pl.make_tile_group(
            type=y_group_type, addrs=40 * row_width, mutex_ids="auto", depth=2
        )
        scale_group_type = pl.TileType(
            shape=[1, LANES], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        scale_group = pl.make_tile_group(
            type=scale_group_type,
            addrs=40 * row_width + 2 * y_width,
            mutex_ids="auto",
            depth=2,
        )
        worker = pl.get_block_idx()
        workers = pl.get_block_num()
        with pl.section_vector():
            for group in pl.range(groups):
                begin = pl.getval(offsets, group)
                end = pl.getval(offsets, group + 1)
                if global_work:
                    first = begin + (worker + workers - begin % workers) % workers
                else:
                    first = begin + worker
                # Guard the actual consumer loop, not only global nonemptiness.
                if first < end:
                    wl = wl_group.next()
                    wr = wr_group.next()
                    pl.set_validshape(wl, [1, half])
                    pl.set_validshape(wr, [1, half])
                    pl.load(wl, weight_scale, [group, 0])
                    pl.load(wr, weight_scale, [group, half])
                    for row in pl.range(first, end, workers):
                        ml = ml_group.next()
                        mr = mr_group.next()
                        pl.set_validshape(ml, [1, half])
                        pl.set_validshape(mr, [1, half])
                        pl.load(ml, mm, [row, 0])
                        pl.load(mr, mm, [row, half])
                        act = act_group.next()
                        yrow = y_group.next()
                        scale = scale_group.next()
                        xs = pl.getval(x_scale, row)
                        quantize(ml, mr, wl, wr, act, yrow, scale, xs)
                        pl.set_validshape(yrow, [1, half])
                        pl.set_validshape(scale, [1, 1])
                        pl.store(y, yrow, [row, 0])
                        pl.store(y_scale, scale, [row])

    return kernel


def run(
    x,
    weight,
    weight_scale,
    x_scale,
    offsets,
    mm,
    y,
    y_scale,
    *,
    workers=1,
    vector_workers=1,
    global_work=True,
    l1_k=256,
    l0_k=128,
    reuse_l0=False,
    row_batch=1,
):
    """Validate metadata, specialize and launch. Runtime offsets stay on device."""
    rows, inner = x.shape
    groups, weight_inner, cols = weight.shape
    if not (
        1 <= rows <= 4096
        and 1 <= groups <= 32
        and weight_inner == inner
        and 16 <= inner <= 8192
        and inner % 16 == 0
        and 64 <= cols <= 10240
        and cols % 32 == 0
    ):
        raise ValueError("unsupported M/G/K/N; see the kernel contract")
    if l1_k not in (128, 256, 512) or l0_k not in (32, 64, 128) or l0_k > l1_k:
        raise ValueError(
            "expected l1_k in {128,256,512}, l0_k in {32,64,128}, l0_k <= l1_k"
        )
    if workers < 1 or vector_workers < 1 or not isinstance(global_work, bool):
        raise ValueError(
            "positive worker counts and a boolean global_work are required"
        )
    if not isinstance(reuse_l0, bool) or row_batch not in (1, 2, 4, 8, 16):
        raise ValueError("expected boolean reuse_l0 and row_batch in {1,2,4,8,16}")
    if reuse_l0 and l1_k > 256:
        raise ValueError("the paired N control supports L1 panels up to 256")
    if row_batch > 1 and quant_buffer_bytes(cols, row_batch) > 256 * 1024:
        raise ValueError("row batch and buffer slots exceed 256 KiB UB")
    tensors = (x, weight, weight_scale, x_scale, offsets, mm, y, y_scale)
    shapes = (
        (rows, inner),
        (groups, inner, cols),
        (groups, cols),
        (rows,),
        (groups + 1,),
        (rows, cols),
        (rows, cols // 2),
        (rows,),
    )
    dtypes = ("int8", "int8", "float32", "float32", "int32", "int32", "int8", "float32")
    for tensor, shape, dtype in zip(tensors, shapes, dtypes):
        if tuple(tensor.shape) != shape or str(tensor.dtype) != "torch." + dtype:
            raise ValueError(f"expected {shape} {dtype}")
        if not tensor.is_contiguous() or tensor.device != x.device:
            raise ValueError("all tensors must be contiguous on the same device")
    matrix = paired_gemm_kernel if reuse_l0 else gemm_kernel
    matrix(groups, inner, cols, l1_k, l0_k, global_work)[None, workers](
        x, weight, offsets, mm
    )
    if row_batch == 1:
        quant_kernel(groups, cols, global_work)[None, vector_workers](
            mm, weight_scale, x_scale, offsets, y, y_scale
        )
    else:
        batch_quant_kernel(groups, cols, row_batch, global_work)[None, vector_workers](
            mm, weight_scale, x_scale.view(1, rows), offsets, y, y_scale.view(rows, 1)
        )


@lru_cache(maxsize=None)
def paired_gemm_kernel(
    groups,
    inner,
    cols,
    l1_k=256,
    l0_k=128,
    global_work=True,
    m_block=M_BLOCK,
    n_block=N_BLOCK,
):
    """One L0A staging serves two independent N accumulators; never drop mutexes."""
    (resident, step) = (l1_k, l0_k)
    m_span = m_block
    n_span = 2 * n_block
    n_tiles = (cols + n_span - 1) // n_span
    resident = min(resident, (inner + 31) // 32 * 32)
    step = min(step, resident)

    @pl.jit(
        auto_mutex=True,
        name=f"grouped_pair_n_{groups}_{inner}_{cols}_{m_block}_{n_block}_{resident}_{step}_{global_work}",
    )
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, inner], I8],
        weight: pl.Tensor[[groups, inner, cols], I8],
        offsets: pl.Tensor[[groups + 1], I32],
        mm: pl.Tensor[[pl.DYNAMIC, cols], I32],
    ):
        ag_type = pl.TileType(
            shape=[m_span, resident],
            dtype=I8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        ag = pl.make_tile_group(type=ag_type, addrs=0, mutex_ids="auto", depth=2)
        bg_type = pl.TileType(
            shape=[resident, n_span],
            dtype=I8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        bg = pl.make_tile_group(
            type=bg_type, addrs=2 * m_span * resident, mutex_ids="auto", depth=2
        )
        left_type = pl.TileType(
            shape=[m_block, step],
            dtype=I8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        )
        left = pl.make_tile_group(type=left_type, addrs=0, mutex_ids="auto", depth=2)
        right_type = pl.TileType(
            shape=[step, n_block],
            dtype=I8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        )
        right = pl.make_tile_group(type=right_type, addrs=0, mutex_ids="auto", depth=2)
        accumulators_type = pl.TileType(
            shape=[m_block, n_block],
            dtype=I32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            compact=1,
        )
        accumulators = pl.make_tile_group(
            type=accumulators_type, addrs=0, mutex_ids="auto", depth=2
        )
        worker = pl.get_block_idx()
        workers = pl.get_block_num()
        with pl.section_cube():
            prefix = 0
            for g in pl.range(groups):
                begin = pl.getval(offsets, g)
                end = pl.getval(offsets, g + 1)
                count = (end - begin + m_span - 1) // m_span * n_tiles
                if global_work:
                    first = (worker + workers - prefix % workers) % workers
                else:
                    first = worker
                prefix = prefix + count
                for tile in pl.range(first, count, workers):
                    row = begin + tile // n_tiles * m_span
                    col = tile % n_tiles * n_span
                    total_m = pl.min(m_span, end - row)
                    total_n = pl.min(n_span, cols - col)
                    m0 = pl.min(m_block, total_m)
                    n0 = pl.min(n_block, total_n)
                    c0 = accumulators.next()
                    c1 = accumulators.next()
                    pl.set_validshape(c0, [m0, n0])
                    has_second = total_n > n_block
                    second = total_n - n_block
                    if has_second:
                        pl.set_validshape(c1, [m0, second])
                    for k0 in pl.range(0, inner, resident):
                        valid_k = pl.min(resident, inner - k0)
                        a = ag.next()
                        b = bg.next()
                        pl.set_validshape(a, [total_m, valid_k])
                        pl.set_validshape(b, [valid_k, total_n])
                        pl.load(a, x, [row, k0])
                        pl.load(b, weight, [g, k0, col])
                        for kk in pl.range(0, valid_k, step):
                            klen = pl.min(step, valid_k - kk)
                            al = left.next()
                            br = right.next()
                            pl.set_validshape(al, [m0, klen])
                            pl.set_validshape(br, [klen, n0])
                            pl.move(al, a, [0, kk])
                            pl.move(br, b, [kk, 0])
                            if k0 == 0 and kk == 0:
                                pl.matmul(c0, al, br)
                            else:
                                pl.matmul_acc(c0, c0, al, br)
                            if has_second:
                                br1 = right.next()
                                pl.set_validshape(br1, [klen, second])
                                pl.move(br1, b, [kk, n_block])
                                if k0 == 0 and kk == 0:
                                    pl.matmul(c1, al, br1)
                                else:
                                    pl.matmul_acc(c1, c1, al, br1)
                    pl.store(mm, c0, [row, col])
                    if has_second:
                        pl.store(mm, c1, [row, col + n_block])

    return kernel


def quant_buffer_bytes(cols, row_batch, depth=2):
    """Physical UB bytes, including both statistic fields and every slot."""
    half = cols // 2
    width = (half + 63) // 64 * 64
    y_width = (half + 31) // 32 * 32
    return 2 * width * 4 + depth * (
        3 * row_batch * width * 4 + row_batch * y_width + row_batch * 128 * 4 + 64 * 4
    )


@lru_cache(maxsize=None)
def batch_quant_kernel(groups, cols, row_batch=4, global_work=True):
    """Batch independent rows with a block-wide phase boundary and local guards."""
    depth = 2
    half = cols // 2
    width = (half + 63) // 64 * 64
    y_width = (half + 31) // 32 * 32
    if quant_buffer_bytes(cols, row_batch, depth) > 256 * 1024:
        raise ValueError("row batch and buffer slots exceed 256 KiB UB")

    @pl.vector_function
    def process(ml, mr, wl, wr, xs_tile, act, scales, yrow, rows: pl.DT_INT64):
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        for row in pl.range(rows):
            xs = vf.load_align(xs_tile, [0, row], dist=pl.LoadDist.BRC)
            maximum = vf.full(0.0, full, dtype=F32)
            for col in pl.range(0, half, 64):
                active = pl.min(64, half - col)
                mf = vf.update_mask(active, dtype=F32)
                mi = vf.update_mask(active, dtype=I32)
                il = vf.load_align(ml, [row, col])
                left = vf.astype(il, mi, dtype=F32)
                left = vf.mul(left, xs, mf)
                wleft = vf.load_align(wl, [0, col])
                left = vf.mul(left, wleft, mf)
                ir = vf.load_align(mr, [row, col])
                right = vf.astype(ir, mi, dtype=F32)
                right = vf.mul(right, xs, mf)
                wright = vf.load_align(wr, [0, col])
                right = vf.mul(right, wright, mf)
                negative = vf.neg(left, mf)
                exponent = vf.exp(negative, mf)
                denominator = vf.adds(exponent, 1.0, mf)
                silu = vf.div(left, denominator, mf)
                value = vf.mul(silu, right, mf)
                vf.store_align(act, value, mf, [row, col])
                absolute = vf.abs(value, mf)
                maximum = vf.max(maximum, absolute, full)
            reduced = vf.reduce_max(maximum, full)
            row_max = vf.full(reduced, full)
            bounded = vf.maxs(row_max, pl.const(1.1754943508222875e-38, F32), full)
            scale = vf.muls(bounded, pl.const(0.007874015748031496, F32), full)
            first = vf.update_mask(1, dtype=F32)
            vf.store_align(scales, scale, first, [row, 0])
            vf.store_align(scales, bounded, first, [row, 64])
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        one = vf.full(1.0, full, dtype=F32)
        for row in pl.range(rows):
            normalizer = vf.load_align(scales, [row, 64], dist=pl.LoadDist.BRC)
            inverse = vf.div(one, normalizer, full)
            for col in pl.range(0, half, 64):
                active = pl.min(64, half - col)
                mf = vf.update_mask(active, dtype=F32)
                mi = vf.update_mask(active, dtype=I32)
                value = vf.load_align(act, [row, col])
                normalized = vf.mul(value, inverse, mf)
                scaled = vf.muls(normalized, pl.const(127.0, F32), mf)
                integer = vf.astype(
                    scaled, mf, dtype=I32, round_mode=pl.VFRoundMode.CAST_RINT
                )
                low = vf.maxs(integer, pl.const(-128, I32), mi)
                clipped = vf.mins(low, pl.const(127, I32), mi)
                vf.store_align(
                    yrow, clipped, mi, [row, col], dist=pl.StoreDist.PACK4_B32
                )

    @pl.jit(
        auto_mutex=True,
        name=f"grouped_quant_batch_{groups}_{cols}_{row_batch}_{global_work}",
    )
    def kernel(
        mm: pl.Tensor[[pl.DYNAMIC, cols], I32],
        weight_scale: pl.Tensor[[groups, cols], F32],
        x_scale: pl.Tensor[[1, pl.DYNAMIC], F32],
        offsets: pl.Tensor[[groups + 1], I32],
        y: pl.Tensor[[pl.DYNAMIC, half], I8],
        y_scale: pl.Tensor[[pl.DYNAMIC, 1], F32],
    ):
        wl_group_type = pl.TileType(
            shape=[1, width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        wl_group = pl.make_tile_group(
            type=wl_group_type, addrs=0, mutex_ids="auto", depth=1
        )
        wr_group_type = pl.TileType(
            shape=[1, width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        wr_group = pl.make_tile_group(
            type=wr_group_type, addrs=4 * width, mutex_ids="auto", depth=1
        )
        ml_group_type = pl.TileType(
            shape=[row_batch, width], dtype=I32, target_memory=pl.MemorySpace.Vec
        )
        ml_group = pl.make_tile_group(
            type=ml_group_type, addrs=8 * width, mutex_ids="auto", depth=depth
        )
        mr_group_type = pl.TileType(
            shape=[row_batch, width], dtype=I32, target_memory=pl.MemorySpace.Vec
        )
        mr_group = pl.make_tile_group(
            type=mr_group_type,
            addrs=8 * width + 4 * depth * row_batch * width,
            mutex_ids="auto",
            depth=depth,
        )
        act_group_type = pl.TileType(
            shape=[row_batch, width], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        act_group = pl.make_tile_group(
            type=act_group_type,
            addrs=8 * width + 8 * depth * row_batch * width,
            mutex_ids="auto",
            depth=depth,
        )
        y_group_type = pl.TileType(
            shape=[row_batch, y_width], dtype=I8, target_memory=pl.MemorySpace.Vec
        )
        y_group = pl.make_tile_group(
            type=y_group_type,
            addrs=8 * width + 12 * depth * row_batch * width,
            mutex_ids="auto",
            depth=depth,
        )
        scale_group_type = pl.TileType(
            shape=[row_batch, 128], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        scale_group = pl.make_tile_group(
            type=scale_group_type,
            addrs=8 * width
            + 12 * depth * row_batch * width
            + depth * row_batch * y_width,
            mutex_ids="auto",
            depth=depth,
        )
        xs_group_type = pl.TileType(
            shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        xs_group = pl.make_tile_group(
            type=xs_group_type,
            addrs=8 * width
            + 12 * depth * row_batch * width
            + depth * row_batch * y_width
            + 512 * depth * row_batch,
            mutex_ids="auto",
            depth=depth,
        )
        worker = pl.get_block_idx()
        workers = pl.get_block_num()
        with pl.section_vector():
            prefix = 0
            for group in pl.range(groups):
                begin = pl.getval(offsets, group)
                end = pl.getval(offsets, group + 1)
                chunks = (end - begin + row_batch - 1) // row_batch
                if global_work:
                    first = (worker + workers - prefix % workers) % workers
                else:
                    first = worker
                prefix = prefix + chunks
                if first < chunks:
                    wl = wl_group.next()
                    wr = wr_group.next()
                    pl.set_validshape(wl, [1, half])
                    pl.set_validshape(wr, [1, half])
                    pl.load(wl, weight_scale, [group, 0])
                    pl.load(wr, weight_scale, [group, half])
                    for chunk in pl.range(first, chunks, workers):
                        row = begin + chunk * row_batch
                        rows = pl.min(row_batch, end - row)
                        ml = ml_group.next()
                        mr = mr_group.next()
                        xs = xs_group.next()
                        pl.set_validshape(ml, [rows, half])
                        pl.set_validshape(mr, [rows, half])
                        pl.set_validshape(xs, [1, rows])
                        pl.load(ml, mm, [row, 0])
                        pl.load(mr, mm, [row, half])
                        pl.load(xs, x_scale, [0, row])
                        act = act_group.next()
                        yr = y_group.next()
                        scale = scale_group.next()
                        process(ml, mr, wl, wr, xs, act, scale, yr, rows)
                        pl.set_validshape(yr, [rows, half])
                        pl.set_validshape(scale, [rows, 1])
                        pl.store(y, yr, [row, 0])
                        pl.store(y_scale, scale, [row, 0])

    return kernel


def configuration(rows, inner, cols, ends, cube_workers=28, vector_workers=56):
    counts = [ends[0]] + [b - a for a, b in zip(ends, ends[1:])]
    largest = max(counts)
    rounded = max(16, (largest + 15) // 16 * 16)
    wide_m = min(256 if largest <= 256 else 128, rounded)
    wide_tiles = sum((r + wide_m - 1) // wide_m for r in counts) * (cols // 256)
    if cols % 256 == 0 and wide_tiles >= 2 * cube_workers:
        mb, nb, step, depth = wide_m, 256, 128, 4
    else:
        mb, nb, step, depth = min(128, rounded), min(128, cols), 256, 2
    tiles = sum((r + mb - 1) // mb for r in counts) * ((cols + nb - 1) // nb)
    workers = max(1, min(cube_workers, tiles))
    # Short/narrow matrices retain the simple producer; whole 256-row groups
    # use one accumulator. Other wide outputs can retain L0A across two N tiles.
    if cols < 2048:
        mode, resident = "hierarchical", step
    elif mb == 256 or (largest <= 32 and len(ends) <= 8 and cols <= 4096):
        mode, resident, depth = "hierarchical", 512, 2
    else:
        mode, mb, nb, resident, step, depth = "paired_n", min(128, mb), 256, 256, 128, 2
    # Batch transport helps short rows, but an extra metadata DMA can lose on
    # regular wide rows. Keep the row control for that measured regime.
    if cols <= 1152:
        batch = 4
    elif (
        cols <= 4096
        and len(ends) <= 8
        and not (cols == 4096 and min(counts) == max(counts) == 128)
    ):
        batch = 2
    else:
        batch = 1
    return dict(
        mode=mode,
        m_block=mb,
        n_block=nb,
        l1_k=resident,
        l0_k=step,
        l1_depth=depth,
        workers=workers,
        vector_workers=min(vector_workers, rows),
        row_batch=batch,
    )


@lru_cache(maxsize=64)
def _offsets(ends, device):
    return torch.tensor((0,) + ends, dtype=torch.int32, device=device)


def grouped_matmul_swiglu_quant(x, weight, weight_scale, x_scale, group_list):
    if not isinstance(group_list, (list, tuple)):
        raise TypeError(
            "this adapter needs Host group boundaries; use kernel.run for device offsets"
        )
    rows, inner = x.shape
    groups, weight_inner, cols = weight.shape
    ends = tuple(int(v) for v in group_list)
    if (
        weight_inner != inner
        or cols % 2
        or len(ends) != groups
        or not ends
        or ends[-1] != rows
    ):
        raise ValueError("inconsistent shapes or group boundaries")
    if min(ends) < 0 or any(a > b for a, b in zip(ends, ends[1:])):
        raise ValueError("group boundaries must be nonnegative and monotone")
    if not x.is_contiguous():
        x = x.contiguous()
    plan = configuration(rows, inner, cols, ends)
    offsets = _offsets(ends, x.device)
    mm = torch.empty((rows, cols), dtype=torch.int32, device=x.device)
    y = torch.empty((rows, cols // 2), dtype=torch.int8, device=x.device)
    scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    if plan["mode"] == "paired_n":
        matrix = paired_gemm_kernel(
            groups,
            inner,
            cols,
            plan["l1_k"],
            plan["l0_k"],
            m_block=plan["m_block"],
            n_block=plan["n_block"],
        )
    else:
        matrix = gemm_kernel(
            groups,
            inner,
            cols,
            plan["l1_k"],
            plan["l0_k"],
            m_block=plan["m_block"],
            n_block=plan["n_block"],
            l1_depth=plan["l1_depth"],
        )
    matrix[None, plan["workers"]](x, weight, offsets, mm)
    if plan["row_batch"] == 1:
        quant_kernel(groups, cols)[None, plan["vector_workers"]](
            mm, weight_scale, x_scale, offsets, y, scale
        )
    else:
        batch_quant_kernel(groups, cols, plan["row_batch"])[
            None, plan["vector_workers"]
        ](mm, weight_scale, x_scale.view(1, rows), offsets, y, scale.view(rows, 1))
    return y, scale
