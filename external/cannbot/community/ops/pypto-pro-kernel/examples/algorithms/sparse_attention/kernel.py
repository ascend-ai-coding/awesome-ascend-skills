# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。SparseFlashAttention 的 dense-mask / indexed-KV 两条完整路径。

example-query: {"aliases": ["sparse flash attention", "SparseFlashAttention", "SFA", "ROWPLUSONE", "ROWPLUSONE attention", "稀疏闪存注意力"], "term_groups": [["sparse flash attention", "SparseFlashAttention", "SFA", "稀疏闪存注意力"]], "facets": {"source_dtype": ["FP16"], "destination_dtype": ["FP16"]}, "exclude": [{"term_groups": [["independent v", "independent value", "独立V"]], "reason": "The native paths require V to equal the K prefix."}]}

契约为 CANN SparseFlashAttention：FP16，索引无重复且在序列范围内，V 是 K 的前缀。
Q/K 为有限值且幅值不超过 1，scale 有限且绝对值不超过 32；输入连续且 32 B 对齐。
Dense 路径将同一 KV head 的相邻 query/head 合为 M 行；indexed 路径的一位 owner
处理一条 query 的 GQA heads。两者均用 Cube 计算 QK/PV、FP32 online softmax
和输出累加、FP16 P；因果比较保留原始 key 坐标并右下角对齐，全屏蔽行输出零。

公开入口 sparse_attention 按元数据选择默认配置：BSND dense 为 M128/Bk256、
preload1/双缓冲；代表性 BNSD wide case 为 M64/Bk128、preload0/单缓冲。
make_kernel 只构建调用方指定的配置，通用默认值为 M64/Bk64、preload1/双缓冲，
不会执行公开入口的选择策略。适配预分配 output 时，应保留对应的完整计算路径和配置。

P 默认采用 ROWPLUSONE；aligned 和 padded 保留为布局对照。物理 pitch、资源预算、
跨核事件与阶段延迟的说明紧邻下面的声明。首块/末块用静态 VF 变体处理。
递推、稀疏坐标与比较原则见
[sparse recurrence](../../../references/patterns/online-softmax.md#sparse-keys-and-representative-implementation-comparisons)。
此实现覆盖上述两类规则分块，不声明通用尾块、独立 V、BF16、重复或越界索引支持。
目标为支持原生自动 mutex ID 的 PyPTO 与 Ascend950PR；测试保留原评测器的误差标准。
"""

from functools import lru_cache
from math import isfinite

import pypto_pro.language as pl
import torch

from ._kernels.vector_ops import make_softmax, make_update, make_dense_mask


@lru_cache(maxsize=None)
def make_kernel(
    batch,
    query_rows,
    key_rows,
    query_heads,
    kv_heads,
    key_dim,
    value_dim,
    selected_rows,
    layout,
    causal,
    dense,
    preload=1,
    rows=64,
    key_block=64,
    buffers=2,
    probability_layout="row_plus_one",
):
    """Build one explicit native configuration without metadata-based dispatch.

    The factory defaults are M64/Bk64, preload1 and two buffers. The public
    sparse_attention entry chooses its own geometry from the workload.
    """
    assert preload in (0, 1)
    assert buffers in (1, 2) and (preload == 0 or buffers == 2)
    assert probability_layout in ("aligned", "padded", "row_plus_one")
    assert rows >= 32 and rows % 32 == 0
    assert key_dim % 64 == value_dim % 64 == key_block % 64 == 0
    assert (
        0 < value_dim <= key_dim
        and 0 < selected_rows <= key_rows
        and selected_rows % 64 == 0
    )
    assert query_heads % kv_heads == 0
    heads = query_heads // kv_heads
    assert (dense and layout == "BSND" and rows % heads == 0) or (
        not dense and layout == "BNSD" and rows == heads
    )
    queries_per_item = rows // heads if dense else 1
    assert query_rows % queries_per_item == 0
    blocks = (key_rows if dense else selected_rows) // key_block
    assert blocks > 0 and (key_rows if dense else selected_rows) % key_block == 0
    items_per_head = query_rows // queries_per_item
    items = batch * kv_heads * items_per_head
    half_rows, half_keys = rows // 2, key_block // 2
    assert not dense or half_rows % heads == 0
    half_queries = half_rows // heads if dense else 1
    # QK steps through K64 views of the same slots used by PV. For a wide PV,
    # the full Bk and a bounded N strip keep L0B within capacity; Bk128/Dv512
    # uses K128/N128 while QK retains the original slot span through its K64 view.
    product_k = key_block if value_dim > 256 else 64
    local_k = max(64, product_k)
    strip = min(value_dim, 16384 // local_k)
    right_columns = max(key_block, strip)
    fp16, fp32, i32 = pl.DT_FP16, pl.DT_FP32, pl.DT_INT32
    qshape = (
        [batch, query_rows, query_heads, key_dim]
        if dense
        else [batch, query_heads, query_rows, key_dim]
    )
    kshape = (
        [batch, key_rows, kv_heads, key_dim]
        if dense
        else [batch, kv_heads, key_rows, key_dim]
    )
    ishape = (
        [batch, query_rows, kv_heads, selected_rows]
        if dense
        else [batch, kv_heads, query_rows, selected_rows]
    )
    oshape = qshape[:-1] + [value_dim]

    staging_rows = 16 if dense else half_keys
    mask_width = min(key_rows, 1024) if dense else 8
    mask_parts = mask_width // key_block if dense else 1
    index_rows = min(4, half_queries) if dense else 1
    query_pack_rows = max(16 // heads, 1) if dense else 1
    query_pack_cols = min(heads, 16) * key_dim if dense else 16
    assert not dense or 16 % heads == 0 or heads % 16 == 0
    # Each AIV owns half_rows, a multiple of 16. ROWPLUSONE uses physical
    # pitch half_rows+1 and compact=2; VF publication and TINSERT share this map.
    # padded keeps that pitch with compact=0; aligned has no extra physical row.
    probability_rows = half_rows + (probability_layout != "aligned")
    probability_compact = 2 if probability_layout == "row_plus_one" else 0

    # Byte offsets describe physical allocations, including every rotating slot.
    # Valid extents and reinterpretations below never change these slot pitches.
    l1_query = buffers * 2 * key_dim * (rows if dense else key_block)
    l1_key = l1_query + 2 * (16 * 16 if dense else rows * key_dim)
    l1_probability = l1_key + 2 * (2 * key_block * key_dim if dense else 16 * 16)
    l1_value = l1_probability + buffers * 2 * rows * key_block
    acc_product = 4 * rows * key_block

    ub_product = 4 * half_rows * key_block
    ub_staging = ub_product + 4 * half_rows * value_dim
    ub_indices = ub_staging + 2 * (1 if dense else buffers) * staging_rows * key_dim
    ub_membership = ub_indices + 4 * index_rows * (
        selected_rows if dense else key_block
    )
    ub_selectors = (
        ub_membership + 4 * (1 if dense else buffers) * half_queries * mask_width
    )
    ub_probability_low = ub_selectors + 4 * key_block
    ub_probability_high = ub_probability_low + 2 * probability_rows * (
        128 if dense else key_block
    )
    ub_maxima = ub_probability_high + 2 * (probability_rows if dense else 1) * (
        128 if dense else 16
    )
    ub_denominators = ub_maxima + 4 * half_rows
    ub_statistics = ub_denominators + 4 * half_rows
    ub_numerator = ub_statistics + 2 * 8 * half_rows
    ub_query = ub_numerator + 4 * half_rows * value_dim
    l1_end = l1_value + 2 * (16 * 16 if dense else buffers * key_block * value_dim)
    ub_end = ub_query + 2 * (16 * key_dim if dense else 16)
    # With preload1/two buffers, dense M128/Bk256/Dk192/Dv128/R512/GQA4
    # uses L1 427008 B and UB 252928 B. Indexed M64/Bk64/Dk576/Dv512
    # uses L1 369152 B and UB 218624 B, including alignment and rotating slots.
    assert l1_end <= 512 * 1024 and ub_end <= 256 * 1024
    assert 4 * rows * (key_block + value_dim) <= 256 * 1024
    assert 4 * rows * local_k <= 64 * 1024
    assert 4 * local_k * right_columns <= 64 * 1024

    # Input and P/V have two cross-core slots with preload1, while score and
    # product each have one. Forward/reverse events are 0/1→2/3, 4→5,
    # 6/7→8/9 and 10→11. Aliases inherit ownership; IDs within each core are auto.
    # preload0/one buffer shortens input/P/V rings and permits indexed Bk128,
    # trading fewer panels against overlap; compare complete calls.
    input_ready, input_free = list(range(buffers)), list(range(2, 2 + buffers))
    probability_ready = list(range(6, 6 + buffers))
    probability_free = list(range(8, 8 + buffers))
    # Dense uses DMA/ND→NZ query packing and VF-scattered window membership;
    # indexed uses per-row DMA for selected K/V. Neither path needs SIMT.
    clear_membership, write_membership = make_dense_mask(
        half_queries, mask_width, selected_rows, key_rows, query_rows, causal
    )

    # Running-max subtraction keeps valid exp inputs nonpositive and P in
    # [0,1], with a nonnegative denominator. Softmax packs P with de_interleave
    # at the NZ pitch above; the final update narrows FP32→FP16 with ZERO and
    # writes PACK_B32. First/last decisions select static VF bodies.
    online_softmax = make_softmax(
        dense,
        half_rows,
        heads,
        key_block,
        key_rows,
        query_rows,
        causal,
        probability_pitch=probability_rows,
    )
    update_numerator = make_update(half_rows, value_dim)
    single_block = blocks == 1

    @pl.pipeline.stage
    def prepare(
        item,
        block,
        tick,
        owner_step,
        query_matrix,
        key_fragments,
        index_flat,
        staging,
        staging_query,
        query_nd,
        query_rows_nd,
        staging_dma,
        gather_indices,
        membership,
        shared_input,
    ):
        target = shared_input.next()
        sub = pl.get_subblock_idx()
        if dense:
            if block < buffers:
                local = staging_query.current()
                nd = query_nd.current()
                packed_rows = query_rows_nd.current()
                b = item // (kv_heads * items_per_head)
                head = (item // items_per_head) % kv_heads
                qbase = (item % items_per_head) * queries_per_item + sub * half_queries
                for slab in pl.range(half_rows // 16):
                    qpos = qbase + slab * 16 // heads
                    first_head = head * heads + (slab * 16) % heads
                    pl.load(
                        packed_rows,
                        query_matrix,
                        [b * query_rows + qpos, first_head * key_dim],
                    )
                    pl.move(local, nd)
                    pl.insert(target, local, [sub * half_rows + slab * 16, 0])
        else:
            local = staging[tick % buffers]
            index_tile = gather_indices.current()
            pl.load(index_tile, index_flat, [item * selected_rows + block * key_block])
            packed = staging_dma[tick % buffers]
            b = item // (kv_heads * query_rows)
            head = (item // query_rows) % kv_heads
            for row in pl.range(half_keys):
                selected = pl.getval(index_tile, sub * half_keys + row)
                source_row = ((b * kv_heads + head) * key_rows + selected) * (
                    key_dim // 16
                )
                pl.load(
                    packed[:, row * 16: (row + 1) * 16],
                    key_fragments,
                    [source_row, 0],
                )
            pl.insert(target, local, [sub * half_keys, 0])

    @pl.pipeline.stage
    def score_product(
        item,
        block,
        tick,
        query,
        key,
        shared_input,
        transposed_input,
        query_local,
        key_local,
        key_transpose,
        left,
        right,
        accumulator,
        shared_score,
    ):
        incoming = shared_input.next()
        score = shared_score.next()
        b = item // (kv_heads * items_per_head)
        head = (item // items_per_head) % kv_heads
        qbase = (item % items_per_head) * queries_per_item
        if dense:
            keys = key_local[tick % 2]
            pl.load(keys, key, [b, block * key_block, head, 0], order=[1, 3])
            transposed = key_transpose[tick % 2]
            q = incoming
        else:
            loaded_query = query_local.current()
            if block == 0:
                pl.load(loaded_query, query, [b, head * heads, qbase, 0], order=[1, 3])
            q = loaded_query
            transposed = transposed_input.current()
        acc = accumulator.current()
        pl.gemm(acc, q, transposed, l0a=left, l0b=right, splitk=64)
        pl.move(score, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    @pl.pipeline.stage
    def normalize(
        item,
        block,
        tick,
        owner_step,
        scale,
        indices,
        indices_matrix,
        gather_indices,
        shared_score,
        membership,
        staging_values,
        selectors,
        p_low,
        p_high,
        maxima,
        denominators,
        statistics,
        shared_p,
        shared_v,
    ):
        score = shared_score.next()
        p = shared_p.next()
        low = p_low.current()
        high = p_high.current()
        m = maxima.current()
        denominator = denominators.current()
        stats = statistics[tick % 2]
        ix = selectors.current()
        mask = membership[0 if dense else owner_step % buffers]
        if dense:
            if block % mask_parts == 0:
                if block == 0:
                    clear_membership(mask)
                selections = gather_indices.current()
                b = item // (kv_heads * items_per_head)
                head = (item // items_per_head) % kv_heads
                qbase = (
                    item % items_per_head
                ) * queries_per_item + pl.get_subblock_idx() * half_queries
                for row0 in pl.range(0, half_queries, index_rows):
                    active = pl.min(index_rows, half_queries - row0)
                    pl.set_validshape(selections, [active, selected_rows])
                    pl.load(
                        selections,
                        indices_matrix,
                        [b * query_rows + qbase + row0, head * selected_rows],
                    )
                    write_membership(
                        selections,
                        mask,
                        (block // mask_parts) * mask_width,
                        qbase + row0,
                        active,
                        row0,
                        block // mask_parts + 1,
                    )
        else:
            pl.load(ix, indices, [item * selected_rows + block * key_block])
            values = shared_v.next()
            prefix = staging_values[tick % buffers]
            sub = pl.get_subblock_idx()
            pl.insert(values, prefix, [sub * half_keys, 0])
        qpos = item % items_per_head
        if block == 0:
            online_softmax(
                score, mask, ix, low, high, m, denominator, stats, block, qpos, scale, True
            )
        else:
            online_softmax(
                score, mask, ix, low, high, m, denominator, stats, block, qpos, scale, False
            )
        sub = pl.get_subblock_idx()
        pl.insert(p, low, [sub * half_rows, 0])
        if dense:
            pl.insert(p, high, [sub * half_rows, 128])

    @pl.pipeline.stage
    def weighted_product(
        tick, shared_p, shared_v, key_prefix, left, right, accumulator, shared_product
    ):
        p = shared_p.next()
        product = shared_product.next()
        if dense:
            values = key_prefix[tick % 2]
        else:
            values = shared_v.next()
        acc = accumulator.current()
        if value_dim <= 256:
            pl.gemm(acc, p, values, l0a=left, l0b=right, splitk=64)
        else:
            pl.gemm(acc, p, values, l0a=left, l0b=right, splitn=strip)
        pl.move(product, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    @pl.pipeline.stage
    def finish(
        item, block, tick, shared_product, statistics, numerator, output_local, output
    ):
        product = shared_product.next()
        stats = statistics[tick % 2]
        running = numerator.current()
        result = output_local.current()
        if block == 0:
            update_numerator(product, stats, running, result, True, single_block)
        elif block == blocks - 1:
            update_numerator(product, stats, running, result, False, True)
        else:
            update_numerator(product, stats, running, result, False, False)
        if block == blocks - 1:
            b = item // (kv_heads * items_per_head)
            head = (item // items_per_head) % kv_heads
            qbase = (item % items_per_head) * queries_per_item
            sub = pl.get_subblock_idx()
            if dense:
                for row in pl.range(half_queries):
                    pl.store(
                        output,
                        result[row * heads: (row + 1) * heads, :],
                        [b, qbase + sub * half_queries + row, head * heads, 0],
                        order=[2, 3],
                    )
            else:
                pl.store(
                    output,
                    result,
                    [b, head * heads + sub * half_rows, qbase, 0],
                    order=[1, 3],
                )

    name = (
        f"cann_sfa_{'dense' if dense else 'indexed'}_m{rows}_k{key_block}_p{preload}"
        f"_b{batch}s{query_rows}t{key_rows}h{query_heads}v{kv_heads}d{key_dim}o{value_dim}"
        f"r{selected_rows}c{int(causal)}_d{buffers}_{probability_layout}_expsub"
    )

    @pl.jit(
        arch="a5",
        name=name,
        auto_mutex=True,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        query: pl.Tensor[qshape, fp16],
        key: pl.Tensor[kshape, fp16],
        indices: pl.Tensor[ishape, i32],
        output: pl.Tensor[oshape, fp16],
        scale: pl.DT_FP32,
    ):
        query_matrix = pl.make_tensor(
            query, [batch * query_rows, query_heads * key_dim], dtype=fp16
        )
        key_fragments = pl.make_tensor(
            key, [batch * key_rows * kv_heads * (key_dim // 16), 16], dtype=fp16
        )
        indices_matrix = pl.make_tensor(
            indices, [batch * query_rows, kv_heads * selected_rows], dtype=i32
        )
        index_flat = pl.make_tensor(
            indices, [batch * query_rows * kv_heads * selected_rows], dtype=i32
        )
        shared_input_type = pl.TileType(
            shape=[rows if dense else key_block, key_dim],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        shared_input = pl.make_tile_group(
            type=shared_input_type,
            addrs=0,
            mutex_ids="auto",
            depth=buffers,
            fwd_ids=input_ready,
            bwd_ids=input_free,
        )
        query_local_type = pl.TileType(
            shape=[16 if dense else rows, 16 if dense else key_dim],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
        )
        query_local = pl.make_tile_group(
            type=query_local_type,
            addrs=l1_query,
            mutex_ids="auto",
            depth=1,
        )
        key_local_type = pl.TileType(
            shape=[key_block if dense else 16, key_dim if dense else 16],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
        )
        key_local = pl.make_tile_group(
            type=key_local_type,
            addrs=l1_key,
            mutex_ids="auto",
            depth=2 if dense else 1,
        )
        shared_p_type = pl.TileType(
            shape=[rows, key_block],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        shared_p = pl.make_tile_group(
            type=shared_p_type,
            addrs=l1_probability,
            mutex_ids="auto",
            depth=buffers,
            fwd_ids=probability_ready,
            bwd_ids=probability_free,
        )
        # Sparse V shares P's handover slots: normalize writes both on MTE3,
        # weighted_product reads both on MTE1, and P's generated forward/reverse
        # events bracket both lifetimes. Dense does not use this placeholder.
        shared_v_type = pl.TileType(
            shape=[16 if dense else key_block, 16 if dense else value_dim],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        shared_v = pl.make_tile_group(
            type=shared_v_type,
            addrs=l1_value,
            mutex_ids="auto",
            depth=1 if dense else buffers,
        )
        left_type = pl.TileType(
            shape=[rows, local_k],
            dtype=fp16,
            target_memory=pl.MemorySpace.Left,
        )
        left = pl.make_tile_group(
            type=left_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        right_type = pl.TileType(
            shape=[local_k, right_columns],
            dtype=fp16,
            target_memory=pl.MemorySpace.Right,
        )
        right = pl.make_tile_group(
            type=right_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        score_acc_type = pl.TileType(
            shape=[rows, key_block],
            dtype=fp32,
            target_memory=pl.MemorySpace.Acc,
        )
        score_acc = pl.make_tile_group(
            type=score_acc_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        product_acc_type = pl.TileType(
            shape=[rows, value_dim],
            dtype=fp32,
            target_memory=pl.MemorySpace.Acc,
        )
        product_acc = pl.make_tile_group(
            type=product_acc_type,
            addrs=acc_product,
            mutex_ids="auto",
            depth=1,
        )
        shared_score_type = pl.TileType(
            shape=[half_rows, key_block],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        shared_score = pl.make_tile_group(
            type=shared_score_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[4],
            bwd_ids=[5],
        )
        shared_product_type = pl.TileType(
            shape=[half_rows, value_dim],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        shared_product = pl.make_tile_group(
            type=shared_product_type,
            addrs=ub_product,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[10],
            bwd_ids=[11],
        )
        staging_type = pl.TileType(
            shape=[staging_rows, key_dim],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        staging = pl.make_tile_group(
            type=staging_type,
            addrs=ub_staging,
            mutex_ids="auto",
            depth=1 if dense else buffers,
        )
        gather_indices_type = pl.TileType(
            shape=[index_rows, selected_rows if dense else key_block],
            dtype=i32,
            target_memory=pl.MemorySpace.Vec,
        )
        gather_indices = pl.make_tile_group(
            type=gather_indices_type,
            addrs=ub_indices,
            mutex_ids="auto",
            depth=1,
        )
        membership_type = pl.TileType(
            shape=[half_queries, mask_width],
            dtype=i32,
            target_memory=pl.MemorySpace.Vec,
        )
        membership = pl.make_tile_group(
            type=membership_type,
            addrs=ub_membership,
            mutex_ids="auto",
            depth=1 if dense else buffers,
        )
        selectors_type = pl.TileType(
            shape=[1, key_block],
            dtype=i32,
            target_memory=pl.MemorySpace.Vec,
        )
        selectors = pl.make_tile_group(
            type=selectors_type,
            addrs=ub_selectors,
            mutex_ids="auto",
            depth=1,
        )
        # The VF cursor and TINSERT must use the same pitch. The padded control
        # uses exact declared rows; RowPlusOne uses align16(valid rows)+1.
        p_low_type = pl.TileType(
            shape=[probability_rows, 128 if dense else key_block],
            valid_shape=[half_rows, 128 if dense else key_block],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
            compact=probability_compact,
        )
        p_low = pl.make_tile_group(
            type=p_low_type,
            addrs=ub_probability_low,
            mutex_ids="auto",
            depth=1,
        )
        p_high_type = pl.TileType(
            shape=[probability_rows if dense else 1, 128 if dense else 16],
            valid_shape=[half_rows if dense else 1, 128 if dense else 16],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
            compact=probability_compact if dense else 0,
        )
        p_high = pl.make_tile_group(
            type=p_high_type,
            addrs=ub_probability_high,
            mutex_ids="auto",
            depth=1,
        )
        maxima_type = pl.TileType(
            shape=[1, half_rows],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        maxima = pl.make_tile_group(
            type=maxima_type,
            addrs=ub_maxima,
            mutex_ids="auto",
            depth=1,
        )
        denominators_type = pl.TileType(
            shape=[1, half_rows],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        denominators = pl.make_tile_group(
            type=denominators_type,
            addrs=ub_denominators,
            mutex_ids="auto",
            depth=1,
        )
        statistics_type = pl.TileType(
            shape=[2, half_rows],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        statistics = pl.make_tile_group(
            type=statistics_type,
            addrs=ub_statistics,
            mutex_ids="auto",
            depth=2,
        )
        numerator_type = pl.TileType(
            shape=[half_rows, value_dim],
            dtype=fp32,
            target_memory=pl.MemorySpace.Vec,
        )
        numerator = pl.make_tile_group(
            type=numerator_type,
            addrs=ub_numerator,
            mutex_ids="auto",
            depth=1,
        )
        # The dense-only scratch comes last so it does not shift the indexed path's buffers.
        query_nd_type = pl.TileType(
            shape=[16 if dense else 1, key_dim if dense else 16],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
        )
        query_nd = pl.make_tile_group(
            type=query_nd_type,
            addrs=ub_query,
            mutex_ids="auto",
            depth=1,
        )
        # The final update packs already-read numerator bytes into FP16 output.
        output_local = pl.reinterpret(
            numerator, shape=[half_rows, value_dim], dtype=fp16
        )
        transposed_input = pl.reinterpret(
            shared_input, shape=[key_dim, rows if dense else key_block], layout=pl.ZN
        )
        query_rows_nd = pl.reinterpret(
            query_nd, shape=[query_pack_rows, query_pack_cols], layout=pl.ND
        )
        staging_query = pl.reinterpret(
            staging, shape=[staging_rows, key_dim], layout=pl.NZ
        )
        staging_dma = pl.reinterpret(
            staging, shape=[key_dim // 16, staging_rows * 16], layout=pl.ND
        )
        staging_values = pl.reinterpret(
            staging, shape=[staging_rows, value_dim], layout=pl.NZ
        )
        key_transpose = pl.reinterpret(
            key_local,
            shape=[key_dim if dense else 16, key_block if dense else 16],
            layout=pl.ZN,
        )
        key_prefix = pl.reinterpret(
            key_local,
            shape=[key_block if dense else 16, value_dim if dense else 16],
            layout=pl.NZ,
        )
        right_qk = pl.reinterpret(right, shape=[64, key_block])
        right_pv = pl.reinterpret(right, shape=[product_k, strip])
        left_qk = pl.reinterpret(left, shape=[rows, 64])
        core = pl.get_block_idx() // pl.get_subblock_num()
        cores = pl.get_block_num()
        count = (items - core + cores - 1) // cores
        # Five V-C-V-C-V stages run over one flat tick loop across owners.
        # With preload1 their delays are 0/1/1/2/2. Local KV/alpha use two slots;
        # dense membership has one epoch-updated slot, cleared at each owner.
        for tick in pl.range(count * blocks):
            owner_step = tick // blocks
            item = core + owner_step * cores
            block = tick % blocks
            with pl.section_vector():
                prepare(
                    item,
                    block,
                    tick,
                    owner_step,
                    query_matrix,
                    key_fragments,
                    index_flat,
                    staging,
                    staging_query,
                    query_nd,
                    query_rows_nd,
                    staging_dma,
                    gather_indices,
                    membership,
                    shared_input,
                )
            with pl.section_cube():
                score_product(
                    item,
                    block,
                    tick,
                    query,
                    key,
                    shared_input,
                    transposed_input,
                    query_local,
                    key_local,
                    key_transpose,
                    left_qk,
                    right_qk,
                    score_acc,
                    shared_score,
                )
            with pl.section_vector():
                normalize(
                    item,
                    block,
                    tick,
                    owner_step,
                    scale,
                    index_flat,
                    indices_matrix,
                    gather_indices,
                    shared_score,
                    membership,
                    staging_values,
                    selectors,
                    p_low,
                    p_high,
                    maxima,
                    denominators,
                    statistics,
                    shared_p,
                    shared_v,
                )
            with pl.section_cube():
                weighted_product(
                    tick,
                    shared_p,
                    shared_v,
                    key_prefix,
                    left,
                    right_pv,
                    product_acc,
                    shared_product,
                )
            with pl.section_vector():
                finish(
                    item,
                    block,
                    tick,
                    shared_product,
                    statistics,
                    numerator,
                    output_local,
                    output,
                )

    return kernel


def sparse_attention(
    query,
    key,
    value,
    sparse_indices,
    scale,
    layout="BSND",
    causal=False,
    *,
    preload=None,
    workers=None,
    rows=None,
    key_block=None,
    buffers=None,
    probability_layout="row_plus_one",
):
    """Run prefix-V attention with metadata-selected defaults.

    BSND uses dense M128/Bk256 with preload1/two buffers. The representative
    BNSD wide shape selects M64/Bk128 with preload0/one buffer when all geometry
    controls are omitted. Explicit controls retain the fallback rules below;
    make_kernel itself does not apply this selector. The caller guarantees
    value == key[..., :Dv].
    """
    assert layout in ("BSND", "BNSD")
    assert query.dtype == key.dtype == value.dtype == torch.float16
    assert sparse_indices.dtype == torch.int32
    assert all(x.is_contiguous() for x in (query, key, value, sparse_indices))
    assert query.ndim == key.ndim == value.ndim == sparse_indices.ndim == 4
    assert query.device == key.device == value.device == sparse_indices.device
    if query.device.type != "npu" or any(
        x.data_ptr() % 32 for x in (query, key, value, sparse_indices)
    ):
        raise ValueError("expected 32-byte-aligned tensors on one NPU")
    if isinstance(scale, torch.Tensor):
        raise TypeError("scale must be a host number")
    scale = float(scale)
    if not isfinite(scale) or abs(scale) > 32:
        raise ValueError("scale must be finite with magnitude at most 32")
    sequence_axis, head_axis = (1, 2) if layout == "BSND" else (2, 1)
    b, sq, hq, dk = (
        query.shape[0],
        query.shape[sequence_axis],
        query.shape[head_axis],
        query.shape[-1],
    )
    sk, hkv, dv = key.shape[sequence_axis], key.shape[head_axis], value.shape[-1]
    selected = sparse_indices.shape[-1]
    assert (
        key.shape[0] == b and key.shape[-1] == dk and value.shape[:-1] == key.shape[:-1]
    )
    expected_indices = (
        (b, sq, hkv, selected) if layout == "BSND" else (b, hkv, sq, selected)
    )
    assert sparse_indices.shape == expected_indices
    dense = layout == "BSND"
    if all(x is None for x in (preload, rows, key_block, buffers)) and (
        b,
        sq,
        sk,
        hq,
        hkv,
        dk,
        dv,
        selected,
        layout,
        causal,
    ) == (16, 4, 8192, 64, 1, 576, 512, 512, "BNSD", True):
        preload, rows, key_block, buffers = 0, 64, 128, 1
    preload = 1 if preload is None else preload
    buffers = 2 if buffers is None else buffers
    rows = rows if rows is not None else (128 if dense else 64)
    block = key_block if key_block is not None else (256 if dense else 64)
    shape = tuple(query.shape[:-1]) + (dv,)
    output = torch.empty(shape, dtype=query.dtype, device=query.device)
    kernel = make_kernel(
        b,
        sq,
        sk,
        hq,
        hkv,
        dk,
        dv,
        selected,
        layout,
        causal,
        dense,
        preload,
        rows,
        block,
        buffers,
        probability_layout,
    )
    if workers is None:
        kernel(query, key, sparse_indices, output, float(scale))
    else:
        kernel[None, workers](query, key, sparse_indices, output, float(scale))
    return output
