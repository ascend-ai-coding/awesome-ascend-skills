# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。原生 pipeline 重写的 dense-mask / indexed-KV 注意力。

契约为 CANN SparseFlashAttention：FP16，索引无重复且在序列范围内，V 是 K 的前缀。
Dense 路径将同一 KV head 的相邻 query/head 合为 M 行；indexed 路径的一个 owner
是一条 query 的 M 个 GQA heads。两者都使用 FP32 online softmax 和输出累加，FP16 P。
因果比较保留原始 key 坐标，右下角对齐；全屏蔽行输出零。

五个 stage 为准备输入、QK、softmax/发布 P/V、PV、更新输出，V-C-V-C-V 交替。
P=1 时延迟为 0/1/1/2/2；第 1/3 条跨核边两槽，第 2/4 条边一槽。
同核 KV/alpha 交接两槽；dense membership 在 normalize 内按 key 窗口更新 epoch，
单槽复用，跨 owner 重新清零。单个平坦 tick 循环跨 owner 连续排布。

Dense M=128、Bk=256、Dk=192、Dv=128、R=512、GQA=4 时 L1 427008 B、UB 252928 B；
indexed M=64、Bk=64、Dk=576、Dv=512 时 L1 369152 B、UB 218624 B，均包含对齐。
Dense 用 DMA/ND→NZ 打包 Q、VF scatter 建立窗口 mask；indexed 用逐行 DMA，
两者均无 SIMT，UB 上限为 256 KiB。P 在 VF 中直接写入带填充行的 NZ 缓冲；
最后一块将 numerator 已读过的字节压紧成 FP16 输出，不增设输出缓冲。
共享 L0A 为 [M,64] 两槽，L0B 为 [64,256] 两槽；两个 GEMM 通过组视图复用。
QK/PV 累加器分别常驻，PV 按最多 256 列分条写入共享输出，避免撑满 L0B。
TileType/make_tile_group 显式指定物理地址，mutex_ids="auto" 分配核内 ID，别名继承 ID。
四条跨核边的交接/回收事件为 0/1→2/3、4→5、6/7→8/9、10→11，
由 pipeline 生成相应同步。默认 preload=1，preload=0 留作同步/调度对照。
首块/末块判断放在 stage 中，选择静态 VF 变体，避免在 VF 的行/列循环内反复执行运行时分支。

指数以运行最大值为基准，输入给 exp 的有效值非正，P 在 [0,1]；分母为非负指数之和。
输出 FP32→FP16 使用 ZERO 布局并以 PACK_B32 写回；P 经 de_interleave 后按 NZ 块间距存放。
测试与性能入口保留原评测器的误差标准。此实现覆盖这两类规则分块，不声明通用尾块、
独立 V、BF16、重复或越界索引支持。目标为支持原生自动 mutex ID 的 PyPTO 与 Ascend950PR。
"""

from functools import lru_cache

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
):
    assert preload in (0, 1)
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
    strip = min(value_dim, 256)
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
    clear_membership, write_membership = make_dense_mask(
        half_queries, mask_width, selected_rows, key_rows, query_rows, causal
    )

    online_softmax = make_softmax(
        dense, half_rows, heads, key_block, key_rows, query_rows, causal
    )
    online_softmax_first = make_softmax(
        dense,
        half_rows,
        heads,
        key_block,
        key_rows,
        query_rows,
        causal,
        first_block=True,
    )
    update_numerator = make_update(half_rows, value_dim)
    update_numerator_first = make_update(
        half_rows, value_dim, first_block=True, last_block=blocks == 1
    )
    update_numerator_last = make_update(half_rows, value_dim, last_block=True)

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
            if block < 2:
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
            local = staging[tick % 2]
            index_tile = gather_indices.current()
            pl.load(index_tile, index_flat, [item * selected_rows + block * key_block])
            packed = staging_dma[tick % 2]
            b = item // (kv_heads * query_rows)
            head = (item // query_rows) % kv_heads
            for row in pl.range(half_keys):
                selected = pl.getval(index_tile, sub * half_keys + row)
                source_row = ((b * kv_heads + head) * key_rows + selected) * (
                    key_dim // 16
                )
                pl.load(
                    packed[:, slice(row * 16, (row + 1) * 16)],
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
        mask = membership[0 if dense else owner_step % 2]
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
            prefix = staging_values[tick % 2]
            sub = pl.get_subblock_idx()
            pl.insert(values, prefix, [sub * half_keys, 0])
        qpos = item % items_per_head
        if block == 0:
            online_softmax_first(
                score, mask, ix, low, high, m, denominator, stats, block, qpos, scale
            )
        else:
            online_softmax(
                score, mask, ix, low, high, m, denominator, stats, block, qpos, scale
            )
        sub = pl.get_subblock_idx()
        pl.set_validshape(low, [half_rows, 128 if dense else key_block])
        pl.insert(p, low, [sub * half_rows, 0])
        if dense:
            pl.set_validshape(high, [half_rows, 128])
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
            update_numerator_first(product, stats, running, result)
        elif block == blocks - 1:
            update_numerator_last(product, stats, running, result)
        else:
            update_numerator(product, stats, running, result)
        if block == blocks - 1:
            b = item // (kv_heads * items_per_head)
            head = (item // items_per_head) % kv_heads
            qbase = (item % items_per_head) * queries_per_item
            sub = pl.get_subblock_idx()
            if dense:
                for row in pl.range(half_queries):
                    pl.store(
                        output,
                        result[slice(row * heads, (row + 1) * heads), :],
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
        f"r{selected_rows}c{int(causal)}"
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
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        query_local_type = pl.TileType(
            shape=[16 if dense else rows, 16 if dense else key_dim],
            dtype=fp16,
            target_memory=pl.MemorySpace.Mat,
        )
        query_local = pl.make_tile_group(
            type=query_local_type,
            addrs=4 * key_dim * (rows if dense else key_block),
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
            addrs=(
                4 * key_dim * (rows if dense else key_block)
                + 2 * (16 if dense else key_dim) * (16 if dense else rows)
            ),
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
            addrs=(
                4 * key_dim * (rows if dense else key_block)
                + 2 * (16 if dense else key_dim) * (16 if dense else rows)
                + 2
                * (2 if dense else 1)
                * (key_block if dense else 16)
                * (key_dim if dense else 16)
            ),
            mutex_ids="auto",
            depth=2,
            fwd_ids=[6, 7],
            bwd_ids=[8, 9],
        )
        # Sparse V shares P's two-slot handover: normalize writes both on MTE3,
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
            addrs=(
                4 * key_dim * (rows if dense else key_block)
                + 2 * (16 if dense else key_dim) * (16 if dense else rows)
                + 2
                * (2 if dense else 1)
                * (key_block if dense else 16)
                * (key_dim if dense else 16)
                + 4 * key_block * rows
            ),
            mutex_ids="auto",
            depth=1 if dense else 2,
        )
        left_type = pl.TileType(
            shape=[rows, 64],
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
            shape=[64, right_columns],
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
            addrs=4 * key_block * rows,
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
            addrs=4 * half_rows * key_block,
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
            addrs=4 * half_rows * key_block + 4 * half_rows * value_dim,
            mutex_ids="auto",
            depth=1 if dense else 2,
        )
        gather_indices_type = pl.TileType(
            shape=[index_rows, selected_rows if dense else key_block],
            dtype=i32,
            target_memory=pl.MemorySpace.Vec,
        )
        gather_indices = pl.make_tile_group(
            type=gather_indices_type,
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
            ),
            mutex_ids="auto",
            depth=1 if dense else 2,
        )
        selectors_type = pl.TileType(
            shape=[1, key_block],
            dtype=i32,
            target_memory=pl.MemorySpace.Vec,
        )
        selectors = pl.make_tile_group(
            type=selectors_type,
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
            ),
            mutex_ids="auto",
            depth=1,
        )
        p_low_type = pl.TileType(
            shape=[half_rows + 1, 128 if dense else key_block],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        p_low = pl.make_tile_group(
            type=p_low_type,
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
            ),
            mutex_ids="auto",
            depth=1,
        )
        p_high_type = pl.TileType(
            shape=[half_rows + 1 if dense else 1, 128 if dense else 16],
            dtype=fp16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        p_high = pl.make_tile_group(
            type=p_high_type,
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
                + 2 * (128 if dense else 16) * (half_rows + 1 if dense else 1)
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
                + 2 * (128 if dense else 16) * (half_rows + 1 if dense else 1)
                + 4 * half_rows
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
                + 2 * (128 if dense else 16) * (half_rows + 1 if dense else 1)
                + 8 * half_rows
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 4 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
                + 2 * (128 if dense else 16) * (half_rows + 1 if dense else 1)
                + 24 * half_rows
            ),
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
            addrs=(
                4 * half_rows * key_block
                + 8 * half_rows * value_dim
                + 2 * (1 if dense else 2) * key_dim * staging_rows
                + 4 * index_rows * (selected_rows if dense else key_block)
                + 4 * (1 if dense else 2) * half_queries * mask_width
                + 4 * key_block
                + 2 * (128 if dense else key_block) * half_rows
                + 2 * (128 if dense else key_block)
                + 2 * (128 if dense else 16) * (half_rows + 1 if dense else 1)
                + 24 * half_rows
            ),
            mutex_ids="auto",
            depth=1,
        )
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
        right_pv = pl.reinterpret(right, shape=[64, strip])
        core = pl.get_block_idx() // pl.get_subblock_num()
        cores = pl.get_block_num()
        count = (items - core + cores - 1) // cores
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
                    left,
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
    preload=1,
    workers=None,
    rows=None,
    key_block=None,
):
    """Run the CANN prefix-V contract; the caller guarantees value == key[..., :Dv]."""
    assert layout in ("BSND", "BNSD")
    assert query.dtype == key.dtype == value.dtype == torch.float16
    assert sparse_indices.dtype == torch.int32
    assert all(x.is_contiguous() for x in (query, key, value, sparse_indices))
    assert query.ndim == key.ndim == value.ndim == sparse_indices.ndim == 4
    assert query.device == key.device == value.device == sparse_indices.device
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
    )
    if workers is None:
        from pypto_pro.runtime.platform import get_platform_info

        workers = int(get_platform_info().core_num)
    kernel[None, workers](query, key, sparse_indices, output, float(scale))
    return output
