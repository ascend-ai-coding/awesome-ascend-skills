# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。运行期页表 GQA decode，VF 页内点积、分块部分和与独立合并。

契约：连续FP16 Q[B,Hq,D]、K[P,S,Hkv,D]、V[P,S,Hkv,Dv]，值有限且在[-2,2]；INT32页表[B,L]/长度[B]；FP32 O[B,Hq,Dv]/LSE[B,Hq]。B=1..16，
Hkv=1/2/4，Hq/Hkv=1/2/4/8，S=16/32/64，D/Dv=64/128，L=1..256，P=1..4096，长度0..S*L。允许重复/共享物理页，未消费页表项可为-1；空序列输出精确0/-inf。
分工：一个producer任务负责(batch,kv_head,连续8页)，按运行期长度截断访问，组内query heads复用K/V页；每个query head的分子与分母只由该任务写入对应partial。第二个native kernel
按query head合并有效partial并最终归一化。Host只处理形状、分配和调度，不读取页表/长度内容。未产生的空partial不会被合并阶段读取。
物理资源：producer使用单槽UB，依次放置Q[G,D]、K[S,D]、V[S,Dv] FP16，页分子[1,Dv]、页分母[1,8]、组分子[G,Dv]、组分母[G,8] FP32；最大39712 B，mutex IDs 0..6。
merge使用三个[1,Dv]及三个[1,8] FP32单槽tile，最大1632 B，mutex IDs 0..5。GM scratch为[B,Hq,ceil(L/8),Dv+8] FP32，最大8912896 B；分母重复为8个元素，以
保持32B搬运宽度及partial行距对齐。完整调用为两个native task，scratch存活覆盖两个发射。
数值：|QK/sqrt(D)|<=4*sqrt(D)<=45.26，公共参考0下最小非空分母>=exp(-45.26)，最大分母<=16384*exp(45.26)，分子绝对值<=2倍该上界，均在FP32正常范围。不同partial可直接相
加，不需要在线max重标定。精度仍由原始FP64数学参考、固定atol=rtol=2e-3验收，不以范围证明替代误差验证。
同步与边界：auto_mutex管理MTE2/V/MTE3及tile复用；VF原位状态更新另用定向mem_bar覆盖RAW/WAR/WAW关系。页尾只消费有效key；O与LSE分别设置有效写回形状。两个阶段在同一Stream上依序启动，
producer和merge都有唯一输出所有者。
验证：test.py覆盖workers=1/4、物理守卫、原始输入字节不变、输出毒值、数值端点、首末页峰值、抵消及同地址页表/长度变化。默认8页与28 workers来自该实例的完整调用比较，不是通用最优配置。工具链为PyPTO导出行为基线
872df763f、CANN 9.2.0、BiSheng 15.0.5、Ascend950PR、torch/torch_npu 2.10.0；当前验收仍须绑定源码哈希。
"""

from functools import lru_cache
import math

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
NEG_INF = float("-inf")


@lru_cache(maxsize=None)
def _make_page_kernels(
    batch,
    query_heads,
    kv_heads,
    page_size,
    table_len,
    head_dim,
    value_dim,
    chunk_pages=1,
):
    """Build a page-chunk producer and per-query-head merge kernel."""
    group_size = query_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    chunk_count = (table_len + chunk_pages - 1) // chunk_pages

    @pl.vector_function
    def _page_summary(
        q_tile, k_tile, v_tile, partial_out, partial_den, query_row, key_count
    ):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        den8_mask = vf.update_mask(8, dtype=FP32)
        q0h = vf.load_align(q_tile, [query_row, 0], dist=pl.LoadDist.UNPK)
        q0 = vf.astype(q0h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        if head_dim > 64:
            q1h = vf.load_align(q_tile, [query_row, 64], dist=pl.LoadDist.UNPK)
            q1 = vf.astype(q1h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        den_sum = vf.full(0.0, vec_mask, dtype=FP32)
        out0 = vf.full(0.0, vec_mask, dtype=FP32)
        if value_dim > 64:
            out1 = vf.full(0.0, vec_mask, dtype=FP32)

        for key in pl.range(key_count):
            k0h = vf.load_align(k_tile, [key, 0], dist=pl.LoadDist.UNPK)
            k0 = vf.astype(k0h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            prod0 = vf.mul(q0, k0, vec_mask)
            dot0 = vf.reduce_sum(prod0, vec_mask)
            dot = vf.full(dot0, vec_mask)
            if head_dim > 64:
                k1h = vf.load_align(k_tile, [key, 64], dist=pl.LoadDist.UNPK)
                k1 = vf.astype(k1h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
                prod1 = vf.mul(q1, k1, vec_mask)
                dot1 = vf.reduce_sum(prod1, vec_mask)
                dot1_full = vf.full(dot1, vec_mask)
                dot = vf.add(dot, dot1_full, vec_mask)
            score = vf.muls(dot, scale, vec_mask)
            weight = vf.exp(score, vec_mask)
            den_sum = vf.add(den_sum, weight, vec_mask)

            v0h = vf.load_align(v_tile, [key, 0], dist=pl.LoadDist.UNPK)
            v0 = vf.astype(v0h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            weighted0 = vf.mul(weight, v0, vec_mask)
            out0 = vf.add(out0, weighted0, vec_mask)
            if value_dim > 64:
                v1h = vf.load_align(v_tile, [key, 64], dist=pl.LoadDist.UNPK)
                v1 = vf.astype(v1h, vec_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
                weighted1 = vf.mul(weight, v1, vec_mask)
                out1 = vf.add(out1, weighted1, vec_mask)

        # Previous page merging reads the reused UB output slot.
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        vf.mem_bar(mode=pl.MemBarMode.VST_VST)
        vf.store_align(partial_out, out0, vec_mask, [0, 0])
        if value_dim > 64:
            vf.store_align(partial_out, out1, vec_mask, [0, 64])
        vf.store_align(partial_den, den_sum, den8_mask, [0, 0])

    @pl.vector_function
    def _reset_chunk_state(run_out, run_den):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        den8_mask = vf.update_mask(8, dtype=FP32)
        zero = vf.full(0.0, vec_mask, dtype=FP32)
        for row in pl.range(group_size):
            for col in pl.range(0, value_dim, 64):
                vf.store_align(run_out, zero, vec_mask, [row, col])
            vf.store_align(run_den, zero, den8_mask, [row, 0])

    @pl.vector_function
    def _merge_chunk_page(page_out, page_den, run_out, run_den, row):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_mask = vf.update_mask(1, dtype=FP32)
        den8_mask = vf.update_mask(8, dtype=FP32)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        for col in pl.range(0, value_dim, 64):
            page_values = vf.load_align(page_out, [0, col])
            running = vf.load_align(run_out, [row, col])
            merged = vf.add(running, page_values, vec_mask)
            vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
            vf.store_align(run_out, merged, vec_mask, [row, col])
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        old_den_vec = vf.load_align(run_den, [row, 0])
        old_den = vf.reduce_sum(old_den_vec, first_mask)
        page_den_vec = vf.load_align(page_den, [0, 0])
        page_den_value = vf.reduce_sum(page_den_vec, first_mask)
        old_den_full = vf.full(old_den, vec_mask)
        page_den_full = vf.full(page_den_value, vec_mask)
        new_den = vf.add(old_den_full, page_den_full, vec_mask)
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        vf.store_align(run_den, new_den, den8_mask, [row, 0])

    @pl.jit(auto_mutex=True, name="paged_gqa_page_partials")
    def page_partials(
        q: pl.Tensor[[batch, query_heads, head_dim], FP16],
        k_cache: pl.Tensor[[pl.DYNAMIC, page_size, kv_heads, head_dim], FP16],
        v_cache: pl.Tensor[[pl.DYNAMIC, page_size, kv_heads, value_dim], FP16],
        block_table: pl.Tensor[[batch, table_len], INT32],
        lengths: pl.Tensor[[batch], INT32],
        partials: pl.Tensor[[batch, query_heads, chunk_count, value_dim + 8], FP32],
    ):
        with pl.section_vector():
            q_bytes = group_size * head_dim * 2
            k_bytes = page_size * head_dim * 2
            v_bytes = page_size * value_dim * 2
            out_bytes = value_dim * 4
            run_out_bytes = group_size * value_dim * 4

            q_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[group_size, head_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            k_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[page_size, head_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes],
                mutex_ids=[1],
            )
            v_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[page_size, value_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes + k_bytes],
                mutex_ids=[2],
            )
            out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, value_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[q_bytes + k_bytes + v_bytes],
                mutex_ids=[3],
            )
            den_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[q_bytes + k_bytes + v_bytes + out_bytes],
                mutex_ids=[4],
            )
            run_out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[group_size, value_dim],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes + k_bytes + v_bytes + out_bytes + 8 * 4],
                mutex_ids=[5],
            )
            run_den_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[group_size, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[q_bytes + k_bytes + v_bytes + out_bytes + 8 * 4 + run_out_bytes],
                mutex_ids=[6],
            )

            page_tasks = batch * kv_heads * chunk_count
            for task in pl.range(pl.get_block_idx(), page_tasks, pl.get_block_num()):
                batch_idx = task // (kv_heads * chunk_count)
                kv_head = (task // chunk_count) % kv_heads
                chunk_idx = task % chunk_count
                length = pl.getval(lengths, batch_idx)
                first_page = chunk_idx * chunk_pages
                page_start = first_page * page_size
                if page_start < length:
                    q_tile = q_group.next()
                    run_out = run_out_group.next()
                    run_den = run_den_group.next()
                    pl.set_validshape(q_tile, [group_size, head_dim])
                    pl.load(
                        q_tile, q, [batch_idx, kv_head * group_size, 0], order=[1, 2]
                    )
                    _reset_chunk_state(run_out, run_den)
                    remaining_pages = (length - page_start + page_size - 1) // page_size
                    pages_in_chunk = pl.min(chunk_pages, remaining_pages)
                    for page_offset in pl.range(pages_in_chunk):
                        page_slot = first_page + page_offset
                        active = pl.min(page_size, length - page_slot * page_size)
                        physical_page = pl.getval(
                            block_table, batch_idx * table_len + page_slot
                        )
                        k_tile = k_group.next()
                        v_tile = v_group.next()
                        pl.load(
                            k_tile,
                            k_cache,
                            [physical_page, 0, kv_head, 0],
                            order=[1, 3],
                        )
                        pl.load(
                            v_tile,
                            v_cache,
                            [physical_page, 0, kv_head, 0],
                            order=[1, 3],
                        )
                        for group_row in pl.range(group_size):
                            partial_out = out_group.next()
                            partial_den = den_group.next()
                            _page_summary(
                                q_tile,
                                k_tile,
                                v_tile,
                                partial_out,
                                partial_den,
                                group_row,
                                active,
                            )
                            _merge_chunk_page(
                                partial_out, partial_den, run_out, run_den, group_row
                            )
                    pl.set_validshape(run_out, [group_size, value_dim])
                    pl.set_validshape(run_den, [group_size, 8])
                    pl.store(
                        partials,
                        run_out,
                        [batch_idx, kv_head * group_size, chunk_idx, 0],
                        order=[1, 3],
                    )
                    pl.store(
                        partials,
                        run_den,
                        [batch_idx, kv_head * group_size, chunk_idx, value_dim],
                        order=[1, 3],
                    )

    @pl.vector_function
    def _reset_state(run_out, run_den):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_mask = vf.update_mask(1, dtype=FP32)
        zero = vf.full(0.0, vec_mask, dtype=FP32)
        for col in pl.range(0, value_dim, 64):
            vf.store_align(run_out, zero, vec_mask, [0, col])
        vf.store_align(run_den, zero, first_mask, [0, 0])

    @pl.vector_function
    def _merge_partial(partial_out, partial_den, run_out, run_den):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_mask = vf.update_mask(1, dtype=FP32)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        for col in pl.range(0, value_dim, 64):
            part = vf.load_align(partial_out, [0, col])
            running = vf.load_align(run_out, [0, col])
            merged = vf.add(running, part, vec_mask)
            vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
            vf.store_align(run_out, merged, vec_mask, [0, col])
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        old_den_vec = vf.load_align(run_den, [0, 0])
        old_den = vf.reduce_sum(old_den_vec, first_mask)
        partial_den_vec = vf.load_align(partial_den, [0, 0])
        partial_den_value = vf.reduce_sum(partial_den_vec, first_mask)
        old_den_full = vf.full(old_den, vec_mask)
        partial_den_full = vf.full(partial_den_value, vec_mask)
        new_den = vf.add(old_den_full, partial_den_full, vec_mask)
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
        vf.store_align(run_den, new_den, first_mask, [0, 0])

    @pl.vector_function
    def _finish(run_out, run_den, out_tile, lse_tile):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_mask = vf.update_mask(1, dtype=FP32)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
        den_vec = vf.load_align(run_den, [0, 0])
        den_scalar = vf.reduce_sum(den_vec, first_mask)
        den_full = vf.full(den_scalar, vec_mask)
        for col in pl.range(0, value_dim, 64):
            numerator = vf.load_align(run_out, [0, col])
            normalized = vf.div(numerator, den_full, vec_mask)
            vf.store_align(out_tile, normalized, vec_mask, [0, col])
        log_den = vf.log(den_full, vec_mask)
        vf.store_align(lse_tile, log_den, first_mask, [0, 0])
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)

    @pl.vector_function
    def _finish_empty(out_tile, lse_tile):
        vec_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_mask = vf.update_mask(1, dtype=FP32)
        zero = vf.full(0.0, vec_mask, dtype=FP32)
        neg_inf = vf.full(NEG_INF, first_mask, dtype=FP32)
        for col in pl.range(0, value_dim, 64):
            vf.store_align(out_tile, zero, vec_mask, [0, col])
        vf.store_align(lse_tile, neg_inf, first_mask, [0, 0])

    @pl.jit(auto_mutex=True, name="paged_gqa_partial_merge")
    def merge_partials(
        partials: pl.Tensor[[batch, query_heads, chunk_count, value_dim + 8], FP32],
        lengths: pl.Tensor[[batch], INT32],
        out: pl.Tensor[[batch, query_heads, value_dim], FP32],
        lse: pl.Tensor[[batch, query_heads], FP32],
    ):
        with pl.section_vector():
            page_out_bytes = value_dim * 4
            page_den_bytes = 8 * 4
            run_out_bytes = value_dim * 4
            run_den_bytes = 8 * 4
            page_out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, value_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            page_den_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[page_out_bytes],
                mutex_ids=[1],
            )
            run_out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, value_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[page_out_bytes + page_den_bytes],
                mutex_ids=[2],
            )
            run_den_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[page_out_bytes + page_den_bytes + run_out_bytes],
                mutex_ids=[3],
            )
            out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, value_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[page_out_bytes + page_den_bytes + run_out_bytes + run_den_bytes],
                mutex_ids=[4],
            )
            lse_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, 8], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[
                    page_out_bytes
                    + page_den_bytes
                    + run_out_bytes
                    + run_den_bytes
                    + value_dim * 4
                ],
                mutex_ids=[5],
            )

            for item in pl.range(
                pl.get_block_idx(), batch * query_heads, pl.get_block_num()
            ):
                batch_idx = item // query_heads
                query_head = item % query_heads
                length = pl.getval(lengths, batch_idx)
                page_out = page_out_group.next()
                page_den = page_den_group.next()
                run_out = run_out_group.next()
                run_den = run_den_group.next()
                out_tile = out_group.next()
                lse_tile = lse_group.next()
                if length > 0:
                    _reset_state(run_out, run_den)
                    page_count = (length + page_size - 1) // page_size
                    chunks = (page_count + chunk_pages - 1) // chunk_pages
                    for chunk_idx in pl.range(chunks):
                        pl.load(
                            page_out,
                            partials,
                            [batch_idx, query_head, chunk_idx, 0],
                            order=[2, 3],
                        )
                        pl.load(
                            page_den,
                            partials,
                            [batch_idx, query_head, chunk_idx, value_dim],
                            order=[2, 3],
                        )
                        _merge_partial(page_out, page_den, run_out, run_den)
                    _finish(run_out, run_den, out_tile, lse_tile)
                else:
                    _finish_empty(out_tile, lse_tile)
                pl.set_validshape(out_tile, [1, value_dim])
                pl.set_validshape(lse_tile, [1, 1])
                pl.store(out, out_tile, [batch_idx, query_head, 0], order=[1, 2])
                pl.store(lse, lse_tile, [batch_idx, query_head], order=[0, 1])

    return page_partials, merge_partials


def _validate_meta(q, k_cache, v_cache, block_table, lengths, out, lse, workers):
    batch, query_heads, head_dim = q.shape
    pages, page_size, kv_heads, key_dim = k_cache.shape
    v_pages, v_size, v_heads, value_dim = v_cache.shape
    table_batch, table_len = block_table.shape
    if not 1 <= batch <= 16:
        raise ValueError("B must be in [1, 16]")
    if kv_heads not in (1, 2, 4) or query_heads % kv_heads:
        raise ValueError("Hkv must be 1/2/4 and Hq must be divisible by Hkv")
    if query_heads // kv_heads not in (1, 2, 4, 8):
        raise ValueError("Hq/Hkv must be 1/2/4/8")
    if page_size not in (16, 32, 64):
        raise ValueError("S must be 16/32/64")
    if head_dim not in (64, 128) or value_dim not in (64, 128):
        raise ValueError("D and Dv must be 64 or 128")
    if not 1 <= pages <= 4096 or not 1 <= table_len <= 256:
        raise ValueError("P must be 1..4096 and L must be 1..256")
    if (
        key_dim != head_dim
        or v_pages != pages
        or v_size != page_size
        or v_heads != kv_heads
        or table_batch != batch
        or lengths.shape != (batch,)
        or out.shape != (batch, query_heads, value_dim)
        or lse.shape != (batch, query_heads)
    ):
        raise ValueError("input/output shapes do not match the paged GQA contract")
    if workers < 1:
        raise ValueError("workers must be positive")
    if (
        q.dtype != torch.float16
        or k_cache.dtype != torch.float16
        or v_cache.dtype != torch.float16
    ):
        raise TypeError("Q, Kcache and Vcache must be FP16")
    if block_table.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise TypeError("block_table and lengths must be INT32")
    if out.dtype != torch.float32 or lse.dtype != torch.float32:
        raise TypeError("O and LSE must be FP32")
    if any(
        t.device != q.device for t in (k_cache, v_cache, block_table, lengths, out, lse)
    ):
        raise ValueError("all tensors must be on the same device")
    return (
        batch,
        query_heads,
        kv_heads,
        page_size,
        block_table.shape[1],
        head_dim,
        value_dim,
    )


def _run_chunked(
    q, k_cache, v_cache, block_table, lengths, out, lse, workers, chunk_pages
):
    (batch, query_heads, kv_heads, page_size, table_len, head_dim, value_dim) = (
        _validate_meta(q, k_cache, v_cache, block_table, lengths, out, lse, workers)
    )
    chunk_count = (table_len + chunk_pages - 1) // chunk_pages
    partials = torch.empty(
        (batch, query_heads, chunk_count, value_dim + 8),
        dtype=torch.float32,
        device=q.device,
    )
    page_kernel, merge_kernel = _make_page_kernels(
        batch,
        query_heads,
        kv_heads,
        page_size,
        table_len,
        head_dim,
        value_dim,
        chunk_pages,
    )
    page_kernel[None, workers](q, k_cache, v_cache, block_table, lengths, partials)
    merge_kernel[None, workers](partials, lengths, out, lse)


def run(q, k_cache, v_cache, block_table, lengths, out, lse, workers=28):
    """Launch page-parallel paged GQA attention with eight-page partials."""
    _run_chunked(
        q, k_cache, v_cache, block_table, lengths, out, lse, workers, chunk_pages=8
    )
