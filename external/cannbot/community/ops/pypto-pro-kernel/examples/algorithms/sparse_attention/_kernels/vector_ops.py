# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VF normalization with direct NZ publication and one reciprocal per output row."""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

F32, F16, I32, U32 = pl.DT_FP32, pl.DT_FP16, pl.DT_INT32, pl.DT_UINT32


def make_softmax(
    dense, half_rows, heads, key_block, key_rows, query_rows, causal, first_block=False
):
    pitch = half_rows + 1
    if dense:
        assert key_block == 256
        mask_width = min(key_rows, 1024)
        mask_parts = mask_width // key_block

        @pl.vector_function
        def dense_softmax(
            score,
            membership,
            selectors,
            low,
            high,
            maxima,
            denominators,
            stats,
            block,
            query_position,
            scale,
        ):
            mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
            half_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F16)
            imask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
            mask_col = (block % mask_parts) * key_block
            marker = block // mask_parts + 1
            zero = vf.full(0.0, mask, dtype=F32)
            negative = vf.full(-1.0e30, mask, dtype=F32)
            for query in pl.range(half_rows // heads):
                b0 = vf.load_align(membership, [query, mask_col])
                b1 = vf.load_align(membership, [query, mask_col + 64])
                b2 = vf.load_align(membership, [query, mask_col + 128])
                b3 = vf.load_align(membership, [query, mask_col + 192])
                valid0 = vf.eq(b0, marker, imask)
                valid1 = vf.eq(b1, marker, imask)
                valid2 = vf.eq(b2, marker, imask)
                valid3 = vf.eq(b3, marker, imask)
                for head in pl.range(heads):
                    row = query * heads + head
                    s0 = vf.load_align(score, [row, 0])
                    s1 = vf.load_align(score, [row, 64])
                    s2 = vf.load_align(score, [row, 128])
                    s3 = vf.load_align(score, [row, 192])
                    s0 = vf.muls(s0, scale, mask)
                    s1 = vf.muls(s1, scale, mask)
                    s2 = vf.muls(s2, scale, mask)
                    s3 = vf.muls(s3, scale, mask)
                    s0 = vf.select(s0, negative, valid0)
                    s1 = vf.select(s1, negative, valid1)
                    s2 = vf.select(s2, negative, valid2)
                    s3 = vf.select(s3, negative, valid3)
                    max01 = vf.max(s0, s1, mask)
                    max23 = vf.max(s2, s3, mask)
                    max_all = vf.max(max01, max23, mask)
                    max_scalar = vf.reduce_max(max_all, mask)
                    block_max = vf.full(max_scalar, mask)
                    old_max = vf.full(-1.0e30, mask, dtype=F32)
                    old_sum = vf.full(0.0, mask, dtype=F32)
                    if not first_block:
                        old_max = vf.load_align(maxima, row, dist=pl.LoadDist.BRC_B32)
                        old_sum = vf.load_align(
                            denominators, row, dist=pl.LoadDist.BRC_B32
                        )
                    new_max = vf.max(old_max, block_max, mask)
                    delta = vf.sub(old_max, new_max, mask)
                    alpha = vf.exp(delta, mask)
                    s0 = vf.sub(s0, new_max, mask)
                    s1 = vf.sub(s1, new_max, mask)
                    s2 = vf.sub(s2, new_max, mask)
                    s3 = vf.sub(s3, new_max, mask)
                    p0 = vf.exp(s0, mask)
                    p1 = vf.exp(s1, mask)
                    p2 = vf.exp(s2, mask)
                    p3 = vf.exp(s3, mask)
                    p0 = vf.select(p0, zero, valid0)
                    p1 = vf.select(p1, zero, valid1)
                    p2 = vf.select(p2, zero, valid2)
                    p3 = vf.select(p3, zero, valid3)
                    sum01 = vf.add(p0, p1, mask)
                    sum23 = vf.add(p2, p3, mask)
                    sum_all = vf.add(sum01, sum23, mask)
                    sum_scalar = vf.reduce_sum(sum_all, mask)
                    sum_block = vf.full(sum_scalar, mask)
                    old_sum = vf.mul_dst_add(alpha, sum_block, mask)
                    vf.store_align(
                        maxima, new_max, mask, row, dist=pl.StoreDist.FIRST_ELEMENT
                    )
                    vf.store_align(
                        denominators,
                        old_sum,
                        mask,
                        row,
                        dist=pl.StoreDist.FIRST_ELEMENT,
                    )
                    vf.store_align(
                        stats, alpha, mask, row, dist=pl.StoreDist.FIRST_ELEMENT
                    )
                    vf.store_align(
                        stats,
                        old_sum,
                        mask,
                        half_rows + row,
                        dist=pl.StoreDist.FIRST_ELEMENT,
                    )
                    h0 = vf.astype(p0, mask, dtype=F16, layout=pl.CastLayout.ZERO)
                    h1 = vf.astype(p1, mask, dtype=F16, layout=pl.CastLayout.ZERO)
                    h2 = vf.astype(p2, mask, dtype=F16, layout=pl.CastLayout.ZERO)
                    h3 = vf.astype(p3, mask, dtype=F16, layout=pl.CastLayout.ZERO)
                    packed_low, unused_low = vf.de_interleave(h0, h1)
                    packed_high, unused_high = vf.de_interleave(h2, h3)
                    vf.store_align(
                        low,
                        packed_low,
                        half_mask,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                        block_stride=pitch,
                        repeat_stride=1,
                        post_update=True,
                    )
                    vf.store_align(
                        high,
                        packed_high,
                        half_mask,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                        block_stride=pitch,
                        repeat_stride=1,
                        post_update=True,
                    )

        return dense_softmax

    assert key_block in (64, 128)

    @pl.vector_function
    def indexed_softmax(
        score,
        membership,
        selectors,
        low,
        high,
        maxima,
        denominators,
        stats,
        block,
        query_position,
        scale,
    ):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        imask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
        half_mask = vf.update_mask(key_block, dtype=F16)
        index = vf.load_align(selectors, 0)
        valid = vf.ge(index, 0, imask)
        if causal:
            valid = vf.le(index, key_rows - query_rows + query_position, valid)
        if key_block == 128:
            index1 = vf.load_align(selectors, 64)
            valid1 = vf.ge(index1, 0, imask)
            if causal:
                valid1 = vf.le(index1, key_rows - query_rows + query_position, valid1)
        zero = vf.full(0.0, mask, dtype=F32)
        negative = vf.full(-1.0e30, mask, dtype=F32)
        for row in pl.range(half_rows):
            score_row = vf.load_align(score, [row, 0])
            scaled = vf.muls(score_row, scale, mask)
            masked = vf.select(scaled, negative, valid)
            max_lanes = vf.move(masked, mask)
            if key_block == 128:
                score1 = vf.load_align(score, [row, 64])
                scaled1 = vf.muls(score1, scale, mask)
                masked1 = vf.select(scaled1, negative, valid1)
                max_lanes = vf.max(masked, masked1, mask)
            max_scalar = vf.reduce_max(max_lanes, mask)
            block_max = vf.full(max_scalar, mask)
            old_max = vf.full(-1.0e30, mask, dtype=F32)
            old_sum = vf.full(0.0, mask, dtype=F32)
            if not first_block:
                old_max = vf.load_align(maxima, row, dist=pl.LoadDist.BRC_B32)
                old_sum = vf.load_align(denominators, row, dist=pl.LoadDist.BRC_B32)
            new_max = vf.max(old_max, block_max, mask)
            delta = vf.sub(old_max, new_max, mask)
            alpha = vf.exp(delta, mask)
            delta_score = vf.sub(masked, new_max, mask)
            exponent = vf.exp(delta_score, mask)
            exponent = vf.select(exponent, zero, valid)
            sum_lanes = vf.move(exponent, mask)
            if key_block == 128:
                delta1 = vf.sub(masked1, new_max, mask)
                exponent1 = vf.exp(delta1, mask)
                exponent1 = vf.select(exponent1, zero, valid1)
                sum_lanes = vf.add(exponent, exponent1, mask)
            sum_scalar = vf.reduce_sum(sum_lanes, mask)
            block_sum = vf.full(sum_scalar, mask)
            old_sum = vf.mul_dst_add(alpha, block_sum, mask)
            vf.store_align(maxima, new_max, mask, row, dist=pl.StoreDist.FIRST_ELEMENT)
            vf.store_align(
                denominators, old_sum, mask, row, dist=pl.StoreDist.FIRST_ELEMENT
            )
            vf.store_align(stats, alpha, mask, row, dist=pl.StoreDist.FIRST_ELEMENT)
            vf.store_align(
                stats, old_sum, mask, half_rows + row, dist=pl.StoreDist.FIRST_ELEMENT
            )
            half_row = vf.astype(exponent, mask, dtype=F16, layout=pl.CastLayout.ZERO)
            if key_block == 128:
                half_row1 = vf.astype(
                    exponent1, mask, dtype=F16, layout=pl.CastLayout.ZERO
                )
                packed, unused = vf.de_interleave(half_row, half_row1)
            else:
                packed, unused = vf.de_interleave(half_row, half_row)
            vf.store_align(
                low,
                packed,
                half_mask,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                block_stride=pitch,
                repeat_stride=1,
                post_update=True,
            )

    return indexed_softmax


def make_update(half_rows, value_dim, first_block=False, last_block=False):
    @pl.vector_function
    def update(product, stats, numerator, result):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        one = vf.full(1.0, mask, dtype=F32)
        for row in pl.range(half_rows):
            alpha = vf.load_align(stats, row, dist=pl.LoadDist.BRC_B32)
            inverse = vf.full(1.0, mask, dtype=F32)
            if last_block:
                total = vf.load_align(stats, half_rows + row, dist=pl.LoadDist.BRC_B32)
                denominator = vf.max(total, one, mask)
                inverse = vf.div(one, denominator, mask)
            for chunk in pl.range(value_dim // 64):
                partial = vf.load_align(product, [row, chunk * 64])
                accumulated = vf.full(0.0, mask, dtype=F32)
                if not first_block:
                    accumulated = vf.load_align(numerator, [row, chunk * 64])
                accumulated = vf.mul_dst_add(alpha, partial, mask)
                if last_block:
                    normalized = vf.mul(accumulated, inverse, mask)
                    narrow = vf.astype(
                        normalized, mask, dtype=F16, layout=pl.CastLayout.ZERO
                    )
                    vf.store_align(
                        result,
                        narrow,
                        mask,
                        [row, chunk * 64],
                        dist=pl.StoreDist.PACK_B32,
                    )
                else:
                    vf.store_align(numerator, accumulated, mask, [row, chunk * 64])
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    return update


def make_dense_mask(mask_rows, width, selected, key_rows, query_rows, causal):
    @pl.vector_function
    def clear(destination):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
        zero = vf.full(0, mask, dtype=I32)
        for row in pl.range(mask_rows):
            for chunk in pl.range(width // 64):
                vf.store_align(destination, zero, mask, [row, chunk * 64])
        vf.mem_bar(mode=pl.MemBarMode.VST_VST)

    @pl.vector_function
    def build(indices, destination, key0, query0, active, row0, marker):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
        zero = vf.full(0, mask, dtype=I32)
        epoch = vf.full(marker, mask, dtype=I32)
        for row in pl.range(active):
            for chunk in pl.range(selected // 64):
                index = vf.load_align(indices, [row, chunk * 64])
                column = vf.adds(index, -key0, mask)
                valid = vf.ge(column, 0, mask)
                upper = vf.lt(column, width, mask)
                valid = vf.and_(valid, upper, mask)
                if causal:
                    visible = vf.le(index, key_rows - query_rows + query0 + row, mask)
                    valid = vf.and_(valid, visible, mask)
                safe = vf.select(column, zero, valid)
                offsets = vf.bit_cast(safe, dtype=U32)
                vf.scatter(destination + (row0 + row) * width, epoch, offsets, valid)
            vf.mem_bar(mode=pl.MemBarMode.VST_VST)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    return clear, build
