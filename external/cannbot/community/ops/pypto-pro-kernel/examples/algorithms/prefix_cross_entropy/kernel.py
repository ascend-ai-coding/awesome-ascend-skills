# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Prefix cross entropy and logits gradient with label smoothing.

Contract: contiguous FP32 x[M,N] in [-80,80], INT32 labels/lengths[M],
FP32 loss[M] and grad[M,N], with 1<=M<=64, 1<=N<=262145, 0<=smoothing<=1.
A valid label lies in [0,length); label=-1 or length=0 ignores the whole row.
Ignored outputs and gradient suffixes are exactly zero. Inputs remain unchanged;
input/output buffers are distinct. Host work consists only of metadata and launch.

Each Vector worker owns complete rows. The first input pass carries max,
exponential denominator and logits sum; the second emits gradients. A compact
loss is formed in UB and drained with native DMA, including for ignored rows.
This permits tight output views without imposing cache-line-sized row groups.

A 16384-element input tile occupies 65536 B at UB 0; three 256 B state slots
start at 0x11000, 0x11100 and 0x11200 (high-water 0x11300). The logits-sum slot
becomes the loss after its final consumer. Auto-mutex protects each slot's reuse.
VF loops stop at the actual loaded/written extent, not always tile capacity.

Verified on behavior 872df763f, CANN 9.2.0 and Ascend950PR. Independent FP64
references use loss atol=rtol=2e-4 and gradient atol=2e-6, rtol=2e-4. Tests include
empty/ignored rows, extreme finite score gaps, smoothing endpoints/interior,
nonzero output storage offsets, parameter boundaries and poisoned repeated calls.
The test matrix is not an exhaustive sweep of the stated parameter domain.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
VL = 64
CHUNK = 16384
_INITIAL_MAX = -3.0e38


@lru_cache(maxsize=None)
def _make_kernel(rows: int, cols: int):
    @pl.vector_function
    def _init_stats(max_tile, sum_tile, logits_tile):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        max_values = vf.full(_INITIAL_MAX, all_lanes)
        zeros = vf.full(0.0, all_lanes)
        vf.store_align(max_tile, max_values, all_lanes, [0, 0])
        vf.store_align(sum_tile, zeros, all_lanes, [0, 0])
        vf.store_align(logits_tile, zeros, all_lanes, [0, 0])

    @pl.vector_function
    def _update_stats(
        tile, max_tile, sum_tile, logits_tile, length, chunk_start, valid_cols
    ):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_lane = vf.update_mask(1, dtype=FP32)
        zeros = vf.full(0.0, all_lanes)

        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        logits_reg = vf.load_align(logits_tile, [0, 0])
        for inner in pl.range(0, valid_cols, VL):
            col = chunk_start + inner
            active = length - col
            if active > VL:
                active = VL
            if active < 0:
                active = 0
            prefix = vf.update_mask(active, dtype=FP32)
            values = vf.load_align(tile, [0, inner])

            old_max_broadcast = vf.full(max_reg, all_lanes)
            safe_values = vf.select(values, old_max_broadcast, prefix)
            block_max = vf.reduce_max(safe_values, all_lanes)
            next_max = vf.max(max_reg, block_max, first_lane)
            next_max_broadcast = vf.full(next_max, all_lanes)

            shifted = vf.sub(safe_values, next_max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            active_exponents = vf.select(exponents, zeros, prefix)
            block_sum = vf.reduce_sum(active_exponents, all_lanes)

            old_delta = vf.sub(max_reg, next_max, first_lane)
            old_scale = vf.exp(old_delta, first_lane)
            scaled_sum = vf.mul(sum_reg, old_scale, first_lane)
            next_sum = vf.add(scaled_sum, block_sum, first_lane)

            active_values = vf.select(values, zeros, prefix)
            block_logits = vf.reduce_sum(active_values, all_lanes)
            next_logits = vf.add(logits_reg, block_logits, first_lane)

            max_reg = vf.move(next_max, all_lanes)
            sum_reg = vf.move(next_sum, all_lanes)
            logits_reg = vf.move(next_logits, all_lanes)

        vf.store_align(max_tile, max_reg, first_lane, [0, 0])
        vf.store_align(sum_tile, sum_reg, first_lane, [0, 0])
        vf.store_align(logits_tile, logits_reg, first_lane, [0, 0])

    @pl.vector_function
    def _write_loss(max_tile, sum_tile, logits_tile, length, target_logit, smoothing):
        first_lane = vf.update_mask(1, dtype=FP32)
        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        logits_reg = vf.load_align(logits_tile, [0, 0])
        length_reg = vf.full(length, first_lane, dtype=FP32)
        target_reg = vf.full(target_logit, first_lane, dtype=FP32)
        smoothing_reg = vf.full(smoothing, first_lane, dtype=FP32)
        one_reg = vf.full(1.0, first_lane)

        log_sum = vf.log(sum_reg, first_lane)
        log_z = vf.add(max_reg, log_sum, first_lane)
        target_weight = vf.sub(one_reg, smoothing_reg, first_lane)
        target_term = vf.mul(target_reg, target_weight, first_lane)
        mean_logits = vf.div(logits_reg, length_reg, first_lane)
        smooth_term = vf.mul(mean_logits, smoothing_reg, first_lane)
        loss_reg = vf.sub(log_z, target_term, first_lane)
        loss_reg = vf.sub(loss_reg, smooth_term, first_lane)
        vf.store_align(logits_tile, loss_reg, first_lane, [0, 0])

    @pl.vector_function
    def _emit_chunk(
        tile, max_tile, sum_tile, length, label, chunk_start, valid_cols, smoothing
    ):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        zeros = vf.full(0.0, all_lanes)
        ones = vf.full(1.0, all_lanes)
        max_reg = vf.load_align(max_tile, [0, 0])
        sum_reg = vf.load_align(sum_tile, [0, 0])
        max_broadcast = vf.full(max_reg, all_lanes)
        sum_broadcast = vf.full(sum_reg, all_lanes)
        denominator = vf.max(sum_broadcast, ones, all_lanes)

        smoothing_reg = vf.full(smoothing, all_lanes, dtype=FP32)
        length_reg = vf.full(length, all_lanes, dtype=FP32)
        uniform_weight = vf.div(smoothing_reg, length_reg, all_lanes)
        one_minus_smoothing = vf.sub(ones, smoothing_reg, all_lanes)
        lanes = vf.arange(0, dtype=INT32)
        label_reg = vf.full(label, all_lanes, dtype=INT32)

        for inner in pl.range(0, valid_cols, VL):
            output_active = valid_cols - inner
            if output_active > VL:
                output_active = VL
            if output_active < 0:
                output_active = 0
            active = length - chunk_start - inner
            if active > output_active:
                active = output_active
            if active < 0:
                active = 0
            write_mask = vf.update_mask(output_active, dtype=FP32)
            prefix = vf.update_mask(active, dtype=FP32)

            values = vf.load_align(tile, [0, inner])
            safe_values = vf.select(values, max_broadcast, prefix)
            shifted = vf.sub(safe_values, max_broadcast, all_lanes)
            exponents = vf.exp(shifted, all_lanes)
            active_exponents = vf.select(exponents, zeros, prefix)
            probabilities = vf.div(active_exponents, denominator, all_lanes)

            absolute_lanes = vf.adds(lanes, chunk_start + inner, all_lanes)
            is_label = vf.eq(absolute_lanes, label_reg, all_lanes)
            label_one = vf.select(ones, zeros, is_label)
            target_q = vf.mul(label_one, one_minus_smoothing, all_lanes)
            q = vf.add(target_q, uniform_weight, all_lanes)
            raw_grad = vf.sub(probabilities, q, all_lanes)
            grad_values = vf.select(raw_grad, zeros, prefix)
            vf.store_align(tile, grad_values, write_mask, [0, inner])

    @pl.vector_function
    def _zero_chunk(tile, valid_cols):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        zeros = vf.full(0.0, all_lanes)
        for inner in pl.range(0, valid_cols, VL):
            vf.store_align(tile, zeros, all_lanes, [0, inner])

    @pl.vector_function
    def _zero_loss(tile):
        first_lane = vf.update_mask(1, dtype=FP32)
        zeros = vf.full(0.0, first_lane)
        vf.store_align(tile, zeros, first_lane, [0, 0])

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[rows, cols], FP32],
        labels: pl.Tensor[[rows], INT32],
        lengths: pl.Tensor[[rows], INT32],
        loss: pl.Tensor[[rows], FP32],
        grad: pl.Tensor[[rows, cols], FP32],
        smoothing: pl.DT_FP32,
    ):
        data_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, CHUNK],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=pl.TilePad.zero,
            ),
            addrs=[0x0000],
            mutex_ids=[0],
        )
        stats_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, VL],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=[0x11000, 0x11100, 0x11200],
            mutex_ids=[1, 2, 3],
        )

        with pl.section_vector():
            data_tile = data_group.current()
            max_tile = stats_group[0]
            sum_tile = stats_group[1]
            logits_tile = stats_group[2]

            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                length = pl.getval(lengths, row)
                label = pl.getval(labels, row)
                pl.set_validshape(max_tile, [1, VL])
                pl.set_validshape(sum_tile, [1, VL])
                pl.set_validshape(logits_tile, [1, VL])

                if length == 0 or label < 0:
                    _zero_loss(logits_tile)
                    pl.set_validshape(logits_tile, [1, 1])
                    pl.store(loss, logits_tile, [row])
                    for chunk in pl.range(0, cols, CHUNK):
                        valid_cols = cols - chunk
                        if valid_cols > CHUNK:
                            valid_cols = CHUNK
                        pl.set_validshape(data_tile, [1, valid_cols])
                        _zero_chunk(data_tile, valid_cols)
                        pl.store(grad, data_tile, [row, chunk])
                else:
                    _init_stats(max_tile, sum_tile, logits_tile)
                    for chunk in pl.range(0, cols, CHUNK):
                        if chunk < length:
                            valid_cols = cols - chunk
                            if valid_cols > CHUNK:
                                valid_cols = CHUNK
                            pl.set_validshape(data_tile, [1, valid_cols])
                            pl.load(data_tile, x, [row, chunk])
                            _update_stats(
                                data_tile,
                                max_tile,
                                sum_tile,
                                logits_tile,
                                length,
                                chunk,
                                valid_cols,
                            )

                    target_logit = pl.getval(x, row * cols + label)
                    _write_loss(
                        max_tile, sum_tile, logits_tile, length, target_logit, smoothing
                    )
                    pl.set_validshape(logits_tile, [1, 1])
                    pl.store(loss, logits_tile, [row])

                    for chunk in pl.range(0, cols, CHUNK):
                        valid_cols = cols - chunk
                        if valid_cols > CHUNK:
                            valid_cols = CHUNK
                        pl.set_validshape(data_tile, [1, valid_cols])
                        if chunk < length:
                            pl.load(data_tile, x, [row, chunk])
                            _emit_chunk(
                                data_tile,
                                max_tile,
                                sum_tile,
                                length,
                                label,
                                chunk,
                                valid_cols,
                                smoothing,
                            )
                        else:
                            _zero_chunk(data_tile, valid_cols)
                        pl.store(grad, data_tile, [row, chunk])

    return kernel


def run(x, labels, lengths, loss, grad, smoothing, workers=1):
    """Launch the native kernel; the host handles only shape and launch data."""
    rows, cols = x.shape
    _make_kernel(rows, cols)[None, workers](x, labels, lengths, loss, grad, smoothing)
