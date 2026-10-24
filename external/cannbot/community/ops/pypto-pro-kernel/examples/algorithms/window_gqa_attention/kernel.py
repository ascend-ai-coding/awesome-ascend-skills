# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 共享每行归一化状态，按查询块复用 K/V。

连续 Q[B,Hq,M,D]、K[B,Hkv,N,D]、V[B,Hkv,N,Dv] 为 FP16，
lengths[B] 与 positions[B,M] 为 INT32；O 与 LSE 为 FP32。
查询头 h 使用 h//(Hq/Hkv) 的 KV 头；分数为 dot(Q,K)/sqrt(D)。
位置 p 的有效 key 满足 0<=j<length 且 p-window<j<=p；空集合
精确输出 O=0、LSE=-inf。有效行输出 softmax 加权 V 和 logsumexp。

支持 B/Hkv 为1或2，Hq/Hkv 为1、2或4，M=1..33、N=1..513，
D 为16/32/64、Dv 为16/32/48/64，window=1..N；length 在0..N，
position 在-1..N-1，Q/K/V 有限且在[-2,2]。固定 atol=rtol=2e-3。
Host 只消费形状与 worker 元数据；输出不依赖原值，不分配 GM 中间结果。

每个查询块独占输出行，K/V 驻留 UB；每行沿有效区间在线更新 FP32
最大值、分母与完整 V 累加器。worker 足够少时整头复用，较多时切查询
行以增加独立工作项；这会重载 K/V，分块选择须随形状和设备重新测量。
LSE 的物理 UB 列宽按8元素补齐，与逻辑输出列数分开。

验证身份：行为基线 872df763f，CANN 9.2.0，Ascend950PR，torch_npu
2.10.0。独立 FP64 参考覆盖运行期窗口、分组头、首末峰值、空行、
退化/尾部尺寸、两种分工、毒化重跑、输入不变和偏移输出守卫。
"""

from functools import lru_cache
import math

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
NEG_INF = float("-inf")
NEG_BIG = -1.0e30


def _query_block_for_workers(batch, query_heads, query_len, workers):
    head_items = batch * query_heads
    if workers <= head_items:
        return query_len
    target_rows = (head_items * query_len + workers - 1) // workers
    block = 1 << (target_rows.bit_length() - 1)
    return min(query_len, block)


@lru_cache(maxsize=None)
def _make_kernel(
    batch, query_heads, kv_heads, query_len, key_len, head_dim, value_dim, query_block
):
    group_size = query_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    lse_cols = ((query_block + 7) // 8) * 8

    @pl.vector_function
    def _attend_row(q_tile, k_tile, v_tile, out_tile, lse_tile, row, begin, end):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_lane = vf.update_mask(1, dtype=FP32)
        q_mask = vf.update_mask(head_dim, dtype=FP32)
        v_mask = vf.update_mask(value_dim, dtype=FP32)

        q16 = vf.load_align(q_tile, [row, 0], dist=pl.LoadDist.UNPK)
        q32 = vf.astype(q16, q_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        max_reg = vf.full(NEG_BIG, all_lanes, dtype=FP32)
        sum_reg = vf.full(0.0, all_lanes, dtype=FP32)
        out_reg = vf.full(0.0, all_lanes, dtype=FP32)

        for key in pl.range(begin, end):
            k16 = vf.load_align(k_tile, [key, 0], dist=pl.LoadDist.UNPK)
            k32 = vf.astype(k16, q_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            product = vf.mul(q32, k32, q_mask)
            dot = vf.reduce_sum(product, q_mask)
            dot_reg = vf.full(dot, all_lanes)
            score = vf.muls(dot_reg, scale, all_lanes)

            next_max = vf.max(max_reg, score, all_lanes)
            old_delta = vf.sub(max_reg, next_max, all_lanes)
            alpha = vf.exp(old_delta, all_lanes)
            score_delta = vf.sub(score, next_max, all_lanes)
            weight = vf.exp(score_delta, all_lanes)

            sum_old = vf.mul(sum_reg, alpha, all_lanes)
            next_sum = vf.add(sum_old, weight, all_lanes)
            v16 = vf.load_align(v_tile, [key, 0], dist=pl.LoadDist.UNPK)
            v32 = vf.astype(v16, v_mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            out_old = vf.mul(out_reg, alpha, v_mask)
            weighted_v = vf.mul(weight, v32, v_mask)
            next_out = vf.add(out_old, weighted_v, v_mask)

            max_reg = vf.move(next_max, all_lanes)
            sum_reg = vf.move(next_sum, all_lanes)
            out_reg = vf.move(next_out, v_mask)

        if begin < end:
            normalized = vf.div(out_reg, sum_reg, v_mask)
            vf.store_align(out_tile, normalized, v_mask, [row, 0])
            log_sum = vf.log(sum_reg, all_lanes)
            lse_value = vf.add(max_reg, log_sum, all_lanes)
            vf.store_align(
                lse_tile,
                lse_value,
                first_lane,
                [0, row],
                dist=pl.StoreDist.FIRST_ELEMENT_B32,
            )
        else:
            zero = vf.full(0.0, v_mask, dtype=FP32)
            neg_inf = vf.full(NEG_INF, first_lane, dtype=FP32)
            vf.store_align(out_tile, zero, v_mask, [row, 0])
            vf.store_align(
                lse_tile,
                neg_inf,
                first_lane,
                [0, row],
                dist=pl.StoreDist.FIRST_ELEMENT_B32,
            )

    @pl.jit(auto_mutex=True, name=f"gqa_sliding_attention_qb{query_block}")
    def kernel(
        q: pl.Tensor[[batch, query_heads, query_len, head_dim], FP16],
        k: pl.Tensor[[batch, kv_heads, key_len, head_dim], FP16],
        v: pl.Tensor[[batch, kv_heads, key_len, value_dim], FP16],
        kv_lengths: pl.Tensor[[batch], INT32],
        query_positions: pl.Tensor[[batch, query_len], INT32],
        out: pl.Tensor[[batch, query_heads, query_len, value_dim], FP32],
        lse: pl.Tensor[[batch, query_heads, query_len], FP32],
        window: pl.DT_INT32,
    ):
        with pl.section_vector():
            q_bytes = query_block * head_dim * 2
            k_bytes = key_len * head_dim * 2
            v_bytes = key_len * value_dim * 2
            out_bytes = query_block * value_dim * 4
            q_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[query_block, head_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            k_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[key_len, head_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes],
                mutex_ids=[1],
            )
            v_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[key_len, value_dim],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes + k_bytes],
                mutex_ids=[2],
            )
            out_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[query_block, value_dim],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[q_bytes + k_bytes + v_bytes],
                mutex_ids=[3],
            )
            lse_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, lse_cols], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[q_bytes + k_bytes + v_bytes + out_bytes],
                mutex_ids=[4],
            )

            query_blocks = (query_len + query_block - 1) // query_block
            work_items = batch * query_heads * query_blocks
            for item in pl.range(pl.get_block_idx(), work_items, pl.get_block_num()):
                batch_idx = item // (query_heads * query_blocks)
                head_idx = (item // query_blocks) % query_heads
                query_begin = (item % query_blocks) * query_block
                active_rows = pl.min(query_block, query_len - query_begin)
                kv_head_idx = head_idx // group_size
                q_tile = q_group.next()
                k_tile = k_group.next()
                v_tile = v_group.next()
                out_tile = out_group.next()
                lse_tile = lse_group.next()
                pl.set_validshape(q_tile, [active_rows, head_dim])
                pl.load(q_tile, q, [batch_idx, head_idx, query_begin, 0], order=[2, 3])
                pl.load(k_tile, k, [batch_idx, kv_head_idx, 0, 0], order=[2, 3])
                pl.load(v_tile, v, [batch_idx, kv_head_idx, 0, 0], order=[2, 3])

                length = pl.getval(kv_lengths, batch_idx)
                for row in pl.range(active_rows):
                    position = pl.getval(
                        query_positions, batch_idx * query_len + query_begin + row
                    )
                    begin = position - window + 1
                    if begin < 0:
                        begin = 0
                    end = pl.min(length, position + 1)
                    end = pl.min(end, key_len)
                    _attend_row(
                        q_tile, k_tile, v_tile, out_tile, lse_tile, row, begin, end
                    )

                pl.set_validshape(out_tile, [active_rows, value_dim])
                pl.set_validshape(lse_tile, [1, active_rows])
                pl.store(
                    out, out_tile, [batch_idx, head_idx, query_begin, 0], order=[2, 3]
                )
                pl.store(
                    lse, lse_tile, [batch_idx, head_idx, query_begin], order=[1, 2]
                )

    return kernel


def run(q, k, v, kv_lengths, query_positions, out, lse, window, workers=1):
    """运行滑窗 GQA attention；Host 仅检查张量元数据并选择形状专用 kernel。"""
    batch, query_heads, query_len, head_dim = q.shape
    kv_batch, kv_heads, key_len, key_dim = k.shape
    v_batch, v_heads, value_len, value_dim = v.shape
    if (kv_batch, value_len, key_dim) != (batch, key_len, head_dim):
        raise ValueError("q、k、v 的 batch、序列与 head 维不匹配")
    if (
        v_heads != kv_heads
        or query_heads % kv_heads != 0
        or kv_lengths.shape != (batch,)
        or query_positions.shape != (batch, query_len)
        or out.shape != (batch, query_heads, query_len, value_dim)
        or lse.shape != (batch, query_heads, query_len)
    ):
        raise ValueError("输入或输出形状不符合 GQA 契约")
    if workers < 1:
        raise ValueError("workers 必须为正数")
    query_block = _query_block_for_workers(batch, query_heads, query_len, workers)
    kernel = _make_kernel(
        batch,
        query_heads,
        kv_heads,
        query_len,
        key_len,
        head_dim,
        value_dim,
        query_block,
    )
    kernel[None, workers](q, k, v, kv_lengths, query_positions, out, lse, window)
