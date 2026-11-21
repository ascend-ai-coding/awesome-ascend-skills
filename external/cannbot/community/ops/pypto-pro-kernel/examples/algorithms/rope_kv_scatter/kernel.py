# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 相邻元素对 RoPE 与唯一槽位 KV 写入。

契约：连续FP16 Q[B,T,Hq,D]、K/V[B,T,Hkv,D]，值有限且在[-2,2]；FP32 cos/sin[P,R/2]为合法三角函数表，INT32 positions/slots[B,T]。B=1..4，T=1..513，
Hkv=1/2/4，Hq/Hkv=1/2/4，D=1..129，R为0..D内偶数，P/C>=1。positions合法，非负slots唯一且<C。前R维按相邻偶奇对旋转，其他维逐位保留；slot=-1仅跳过cache写入，Qout仍全部定
义。
分工：公开run使用512线程块、32-lane逻辑组。一个组负责一个(batch,token,head,输出类别)记录，lane按相邻元素对分担通道；Q输出与KV写入共用一次原生launch，但各自有独立的有效性判断。唯一slot及
head/pair分工保证所有写入唯一，无原子操作或跨线程共享状态。奇数D的最后一个未旋转元素单独检查边界。
R=0：使用不绑定空cos/sin张量的原生copy入口，以扁平元素域搬运Q和有效KV。正常旋转使用FP32计算后转换FP16；|a*c-b*s|和|a*s+b*c|至多4，输入范围内无溢出。未旋转分量和V直接传输FP16，不经过数值变换；
未写cache完全不写，含NaN payload、负零和次正规数的原始位模式必须保留。
资源与同步：SIMT直接访问GM，无显式UB/L1/GM scratch、共享存储或线程屏障；完整调用一个native task。Host只检查元数据并选形状特化，不读取positions/slots内容；位置与槽位内容的合法性由调用方契约
保证。配置选择同时考虑全有效及90%跳过两种负载，不根据Host读回的数据选择路径。
验证：独立FP64旋转参考再舍入FP16，旋转值atol=rtol=2e-3；未旋转部分、V、未写cache、输入和守卫按位模式比较。test.py覆盖workers=1/4、非零存储偏移、P=C=1、最大T/D、两种cache初值，以及同
一storage改变positions/slots后的保留区域。默认T/G来自该实例的邻域对照，其他负载和硬件仍须测量。工具链为PyPTO导出行为基线872df763f、CANN 9.2.0、BiSheng 15.0.5、
Ascend950PR、torch/torch_npu 2.10.0；当前验收仍须绑定源码哈希。
"""

from functools import lru_cache

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
I32 = pl.DT_INT32


@lru_cache(maxsize=None)
def _make_kernel(
    batch,
    tokens,
    q_heads,
    kv_heads,
    width,
    rotary,
    cache_size,
    position_count,
    threads,
    mode,
    group_width,
):
    """创建固定物理形状 specialization；mode 选择平铺、pair 或记录协作。"""
    q_items = batch * tokens * q_heads * width
    groups_per_block = threads // group_width
    if mode == "grouped":
        work_items = batch * tokens * (q_heads + kv_heads)
    else:
        unit_width = (width + 1) // 2 if mode == "paired" else width
        work_items = batch * tokens * (q_heads + kv_heads) * unit_width

    if rotary == 0:
        flat_items = q_items + batch * tokens * kv_heads * width

        @pl.vector_function(mode="simt", max_threads=threads)
        def copy_body(q, k, v, slots, qout, cache_k, cache_v):
            tid = pl.simt.linear_thread_idx()
            start = pl.simt.block_idx().x * threads + tid
            stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
            zero = pl.simt.cast(0, I32)
            for linear in pl.range(start, flat_items, stride):
                if linear < q_items:
                    d = linear % width
                    head = (linear // width) % q_heads
                    record = linear // (width * q_heads)
                    batch_i = record // tokens
                    token_i = record % tokens
                    qout[batch_i, token_i, head, d] = q[batch_i, token_i, head, d]
                else:
                    kv_linear = linear - q_items
                    d = kv_linear % width
                    head = (kv_linear // width) % kv_heads
                    record = kv_linear // (width * kv_heads)
                    batch_i = record // tokens
                    token_i = record % tokens
                    slot = slots[batch_i, token_i]
                    if slot >= zero:
                        cache_k[slot, head, d] = k[batch_i, token_i, head, d]
                        cache_v[slot, head, d] = v[batch_i, token_i, head, d]

        @pl.jit()
        def copy_kernel(
            q: pl.Tensor[[batch, tokens, q_heads, width], FP16],
            k: pl.Tensor[[batch, tokens, kv_heads, width], FP16],
            v: pl.Tensor[[batch, tokens, kv_heads, width], FP16],
            slots: pl.Tensor[[batch, tokens], I32],
            qout: pl.Tensor[[batch, tokens, q_heads, width], FP16],
            cache_k: pl.Tensor[[cache_size, kv_heads, width], FP16],
            cache_v: pl.Tensor[[cache_size, kv_heads, width], FP16],
        ):
            with pl.section_vector():
                copy_body[threads](q, k, v, slots, qout, cache_k, cache_v)

        return copy_kernel

    @pl.vector_function(mode="simt", max_threads=threads)
    def body(q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v):
        tid = pl.simt.linear_thread_idx()
        zero_i32 = pl.simt.cast(0, I32)

        if mode == "grouped":
            lane = tid % group_width
            group_start = pl.simt.block_idx().x * groups_per_block + tid // group_width
            group_stride = pl.simt.grid_dim().x * groups_per_block
            head_count = q_heads + kv_heads
            pair_count = (width + 1) // 2
            for item in pl.range(group_start, work_items, group_stride):
                head_code = item % head_count
                record = item // head_count
                batch_i = record // tokens
                token_i = record % tokens
                position = positions[batch_i, token_i]
                if head_code < q_heads:
                    head = head_code
                    for pair in pl.range(lane, pair_count, group_width):
                        d0 = pair * 2
                        d1 = d0 + 1
                        x0 = q[batch_i, token_i, head, d0]
                        if d0 < rotary:
                            c0 = pl.simt.cast(cos[position, pair], FP32)
                            s0 = pl.simt.cast(sin[position, pair], FP32)
                            a = pl.simt.cast(x0, FP32)
                            b0 = pl.simt.cast(q[batch_i, token_i, head, d1], FP32)
                            y0 = a * c0 - b0 * s0
                            y1 = a * s0 + b0 * c0
                            qout[batch_i, token_i, head, d0] = pl.simt.cast(y0, FP16)
                            if d1 < width:
                                qout[batch_i, token_i, head, d1] = pl.simt.cast(
                                    y1, FP16
                                )
                        else:
                            qout[batch_i, token_i, head, d0] = x0
                            if d1 < width:
                                qout[batch_i, token_i, head, d1] = q[
                                    batch_i, token_i, head, d1
                                ]
                else:
                    head = head_code - q_heads
                    slot = slots[batch_i, token_i]
                    if slot >= zero_i32:
                        for pair in pl.range(lane, pair_count, group_width):
                            d0 = pair * 2
                            d1 = d0 + 1
                            x0 = k[batch_i, token_i, head, d0]
                            if d0 < rotary:
                                c0 = pl.simt.cast(cos[position, pair], FP32)
                                s0 = pl.simt.cast(sin[position, pair], FP32)
                                a = pl.simt.cast(x0, FP32)
                                b0 = pl.simt.cast(k[batch_i, token_i, head, d1], FP32)
                                y0 = a * c0 - b0 * s0
                                y1 = a * s0 + b0 * c0
                                cache_k[slot, head, d0] = pl.simt.cast(y0, FP16)
                                if d1 < width:
                                    cache_k[slot, head, d1] = pl.simt.cast(y1, FP16)
                            else:
                                cache_k[slot, head, d0] = x0
                                if d1 < width:
                                    cache_k[slot, head, d1] = k[
                                        batch_i, token_i, head, d1
                                    ]
                            cache_v[slot, head, d0] = v[batch_i, token_i, head, d0]
                            if d1 < width:
                                cache_v[slot, head, d1] = v[batch_i, token_i, head, d1]
        else:
            start = pl.simt.block_idx().x * threads + tid
            stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
            for linear in pl.range(start, work_items, stride):
                if mode == "paired":
                    pair_count = (width + 1) // 2
                    q_work = batch * tokens * q_heads * pair_count
                    if linear < q_work:
                        pair = linear % pair_count
                        head = (linear // pair_count) % q_heads
                        record = linear // (pair_count * q_heads)
                        batch_i = record // tokens
                        token_i = record % tokens
                        d0 = pair * 2
                        d1 = d0 + 1
                        x0 = q[batch_i, token_i, head, d0]
                        if d0 < rotary:
                            position = positions[batch_i, token_i]
                            c0 = pl.simt.cast(cos[position, pair], FP32)
                            s0 = pl.simt.cast(sin[position, pair], FP32)
                            a = pl.simt.cast(x0, FP32)
                            b0 = pl.simt.cast(q[batch_i, token_i, head, d1], FP32)
                            y0 = a * c0 - b0 * s0
                            y1 = a * s0 + b0 * c0
                            qout[batch_i, token_i, head, d0] = pl.simt.cast(y0, FP16)
                            if d1 < width:
                                qout[batch_i, token_i, head, d1] = pl.simt.cast(
                                    y1, FP16
                                )
                        else:
                            qout[batch_i, token_i, head, d0] = x0
                            if d1 < width:
                                qout[batch_i, token_i, head, d1] = q[
                                    batch_i, token_i, head, d1
                                ]
                    else:
                        kv_linear = linear - q_work
                        pair = kv_linear % pair_count
                        head = (kv_linear // pair_count) % kv_heads
                        record = kv_linear // (pair_count * kv_heads)
                        batch_i = record // tokens
                        token_i = record % tokens
                        slot = slots[batch_i, token_i]
                        if slot >= zero_i32:
                            d0 = pair * 2
                            d1 = d0 + 1
                            x0 = k[batch_i, token_i, head, d0]
                            if d0 < rotary:
                                position = positions[batch_i, token_i]
                                c0 = pl.simt.cast(cos[position, pair], FP32)
                                s0 = pl.simt.cast(sin[position, pair], FP32)
                                a = pl.simt.cast(x0, FP32)
                                b0 = pl.simt.cast(k[batch_i, token_i, head, d1], FP32)
                                y0 = a * c0 - b0 * s0
                                y1 = a * s0 + b0 * c0
                                cache_k[slot, head, d0] = pl.simt.cast(y0, FP16)
                                if d1 < width:
                                    cache_k[slot, head, d1] = pl.simt.cast(y1, FP16)
                            else:
                                cache_k[slot, head, d0] = x0
                                if d1 < width:
                                    cache_k[slot, head, d1] = k[
                                        batch_i, token_i, head, d1
                                    ]
                            cache_v[slot, head, d0] = v[batch_i, token_i, head, d0]
                            if d1 < width:
                                cache_v[slot, head, d1] = v[batch_i, token_i, head, d1]
                else:
                    if linear < q_items:
                        d = linear % width
                        head = (linear // width) % q_heads
                        record = linear // (width * q_heads)
                        batch_i = record // tokens
                        token_i = record % tokens
                        x = q[batch_i, token_i, head, d]
                        if d < rotary:
                            pair = d // 2
                            d0 = pair * 2
                            d1 = d0 + 1
                            position = positions[batch_i, token_i]
                            c0 = pl.simt.cast(cos[position, pair], FP32)
                            s0 = pl.simt.cast(sin[position, pair], FP32)
                            a = pl.simt.cast(q[batch_i, token_i, head, d0], FP32)
                            b0 = pl.simt.cast(q[batch_i, token_i, head, d1], FP32)
                            if d % 2 == 0:
                                y = a * c0 - b0 * s0
                            else:
                                y = a * s0 + b0 * c0
                            qout[batch_i, token_i, head, d] = pl.simt.cast(y, FP16)
                        else:
                            qout[batch_i, token_i, head, d] = x
                    else:
                        kv_linear = linear - q_items
                        d = kv_linear % width
                        head = (kv_linear // width) % kv_heads
                        record = kv_linear // (width * kv_heads)
                        batch_i = record // tokens
                        token_i = record % tokens
                        slot = slots[batch_i, token_i]
                        if slot >= zero_i32:
                            if d < rotary:
                                pair = d // 2
                                d0 = pair * 2
                                d1 = d0 + 1
                                position = positions[batch_i, token_i]
                                c0 = pl.simt.cast(cos[position, pair], FP32)
                                s0 = pl.simt.cast(sin[position, pair], FP32)
                                a = pl.simt.cast(k[batch_i, token_i, head, d0], FP32)
                                b0 = pl.simt.cast(k[batch_i, token_i, head, d1], FP32)
                                if d % 2 == 0:
                                    y = a * c0 - b0 * s0
                                else:
                                    y = a * s0 + b0 * c0
                                cache_k[slot, head, d] = pl.simt.cast(y, FP16)
                            else:
                                cache_k[slot, head, d] = k[batch_i, token_i, head, d]
                            cache_v[slot, head, d] = v[batch_i, token_i, head, d]

    @pl.jit()
    def kernel(
        q: pl.Tensor[[batch, tokens, q_heads, width], FP16],
        k: pl.Tensor[[batch, tokens, kv_heads, width], FP16],
        v: pl.Tensor[[batch, tokens, kv_heads, width], FP16],
        cos: pl.Tensor[[position_count, rotary // 2], FP32],
        sin: pl.Tensor[[position_count, rotary // 2], FP32],
        positions: pl.Tensor[[batch, tokens], I32],
        slots: pl.Tensor[[batch, tokens], I32],
        qout: pl.Tensor[[batch, tokens, q_heads, width], FP16],
        cache_k: pl.Tensor[[cache_size, kv_heads, width], FP16],
        cache_v: pl.Tensor[[cache_size, kv_heads, width], FP16],
    ):
        with pl.section_vector():
            body[threads](q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v)

    return kernel


def _shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def _validate(q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v, workers):
    q_shape, k_shape, v_shape = _shape(q), _shape(k), _shape(v)
    cos_shape, sin_shape = _shape(cos), _shape(sin)
    position_shape, slot_shape = _shape(positions), _shape(slots)
    qout_shape, cache_k_shape, cache_v_shape = (
        _shape(qout),
        _shape(cache_k),
        _shape(cache_v),
    )
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("Q, K and V must have rank four")
    batch, tokens, q_heads, width = q_shape
    if k_shape[:2] != (batch, tokens) or v_shape != k_shape:
        raise ValueError("K and V shapes must agree on [B,T,Hkv,D]")
    kv_heads = k_shape[2]
    if qout_shape != q_shape:
        raise ValueError("Qout must have Q's shape")
    if len(cos_shape) != 2 or sin_shape != cos_shape or 2 * cos_shape[1] > width:
        raise ValueError("cos and sin must have shape [P,R/2] with even R <= D")
    rotary = 2 * cos_shape[1]
    cache_size = cache_k_shape[0] if len(cache_k_shape) == 3 else 0
    if cache_k_shape != (cache_size, kv_heads, width) or cache_v_shape != cache_k_shape:
        raise ValueError("cacheK and cacheV must have shape [C,Hkv,D]")
    if position_shape != (batch, tokens) or slot_shape != position_shape:
        raise ValueError("positions and slots must have shape [B,T]")
    if not (
        1 <= batch <= 4
        and 1 <= tokens <= 513
        and kv_heads in (1, 2, 4)
        and q_heads in (kv_heads, 2 * kv_heads, 4 * kv_heads)
        and 1 <= width <= 129
        and cos_shape[0] >= 1
        and cache_size >= 1
    ):
        raise ValueError("shape is outside the supported contract")
    if not all(
        str(x.dtype).endswith("float16") for x in (q, k, v, qout, cache_k, cache_v)
    ):
        raise TypeError("Q, K, V, Qout and caches must be FP16")
    if not all(str(x.dtype).endswith("float32") for x in (cos, sin)):
        raise TypeError("cos and sin must be FP32")
    if not all(str(x.dtype).endswith("int32") for x in (positions, slots)):
        raise TypeError("positions and slots must be INT32")
    tensors = (q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v)
    if not all(x.is_contiguous() for x in tensors):
        raise ValueError("all tensors must be contiguous")
    if len({str(x.device) for x in tensors}) != 1:
        raise ValueError("all tensors must be on the same device")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    return batch, tokens, q_heads, kv_heads, width, rotary, cache_size, cos_shape[0]


def _run_config(
    q,
    k,
    v,
    cos,
    sin,
    positions,
    slots,
    qout,
    cache_k,
    cache_v,
    workers,
    threads,
    mode,
    group_width=1,
):
    shape = _validate(
        q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v, workers
    )
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if mode == "grouped" and (
        type(group_width) is not int or group_width < 1 or threads % group_width
    ):
        raise ValueError("threads must be a positive multiple of group_width")
    kernel = _make_kernel(*shape, threads, mode, group_width)
    if shape[5] == 0:
        kernel[None, workers](q, k, v, slots, qout, cache_k, cache_v)
    else:
        kernel[None, workers](
            q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v
        )


def run(q, k, v, cos, sin, positions, slots, qout, cache_k, cache_v, workers=1):
    """生产配置：512-thread blocks、32-lane record groups，协作处理相邻 pair。"""
    _run_config(
        q,
        k,
        v,
        cos,
        sin,
        positions,
        slots,
        qout,
        cache_k,
        cache_v,
        workers,
        threads=512,
        mode="grouped",
        group_width=32,
    )
