# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ruff: noqa: F821
"""写法：混合（原生 Cube、VF 和显式流水同步）。

Each Cube and its two AIVs process one contiguous LI task range. Each AIV
owns 256 tokens in a 512-token panel; 544-byte records contain eight FP4
keys and their E8M0 scales. Two UB gather slots feed four private GM slots;
gather runs three panels ahead of QK consumption.
The Cube retains Q and its scale in L0 across a row's panels (explicit retained-operand
load/move/matmul_mx calls); K has two L1/L0 slots, and QK has two UB slots.
Each head FMA rounds to BF16, including signed weights. AIV0 selects radix
TopK and cross-core MIX synchronization precedes merges of split rows.

Ring-ready/free and QK-ready/free events pair producer completion with the
first consumer pipe and guard reuse; initial credits are drained at exit.
Paired DMA addresses are sorted and VF restores the original candidate order.
The maximum per-AIV UB span is 214528 bytes (TopK=8192, PA). L0A uses
2048 bytes, L0B uses 65536 bytes and Acc uses 65536 bytes; the highest L1
scale slots remain below 286720 bytes. Aliases use shared IDs or explicit
whole-slot locks and the drained merge boundary described at their declarations. The original LI/LD int32[1024] metadata
is consumed on device; runtime configuration uses native scalar arguments; the host handles shapes, allocation and
launch only.
"""

import functools
from typing import Optional

import pypto_pro.language as pl
import torch
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

U16 = pl.DT_UINT16
U32 = pl.DT_UINT32
TILE_M = 32
TILE_N = 512
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
HEADS_PER_GROUP = 4
SCALE_GROUP_SIZE = 32
SCALE_K = LOGICAL_D // SCALE_GROUP_SIZE
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
BLOCKS_PER_TILE = TILE_N // CANDIDATE_BLOCK_SIZE
BLOCKS_SHIFT = 6
ROWS_PER_STAGE = TILE_N // CANDIDATE_BLOCK_SIZE
BLOCKS_PER_AIV = BLOCKS_PER_TILE // 2
KEY_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * PACKED_D
SCALE_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * SCALE_K
PACKED_BLOCK_BYTES = KEY_BLOCK_BYTES + SCALE_BLOCK_BYTES
STAGE_DEPTH = 4
GATHER_AHEAD = 3
BUF_DEPTH = 2
STAGE_MASK = STAGE_DEPTH - 1
BUF_MASK = BUF_DEPTH - 1
QK_DEPTH = 2
QK_MASK = QK_DEPTH - 1
assert BLOCKS_PER_TILE == 1 << BLOCKS_SHIFT
assert STAGE_DEPTH & STAGE_MASK == 0 and BUF_DEPTH & BUF_MASK == 0
assert QK_DEPTH & QK_MASK == 0
assert 1 <= GATHER_AHEAD <= STAGE_DEPTH
MAX_CUBE_WORKERS = 32
MAX_TOKENS = CANDIDATE_CAPACITY * CANDIDATE_BLOCK_SIZE
HALF_TOKENS = TILE_N // 2
MAX_TILES = MAX_TOKENS // TILE_N
STAGE_TOKENS = MAX_CUBE_WORKERS * STAGE_DEPTH * TILE_N
STAGE_BLOCKS = STAGE_TOKENS // CANDIDATE_BLOCK_SIZE
assert STAGE_TOKENS * PACKED_D % KEY_BLOCK_BYTES == 0
KEY_WS_ROWS = MAX_CUBE_WORKERS * MAX_TILES * 2
BF16_NEG_INF_BITS = 65408
PA_TABLE_CACHE_ENTRIES = 2048
TABLE_COLS_PER_BURST = 8
MAX_ADDRESS32_BYTES = 1 << 31
PA_BLOCK_ALIGNMENT = 16
MAX_PA_BLOCK_SIZE = 1024
MAX_TOPK = 8192
IDX_CHUNK = 1024
PUB16_MIN = 16
MAX_CMP_RATIO = 128
USER_UB_BYTES = 216 * 1024
EV_K_READY = 0
EV_RING_FREE = EV_K_READY + STAGE_DEPTH
EV_QK_READY = EV_RING_FREE + STAGE_DEPTH
EV_JOIN = EV_QK_READY + QK_DEPTH
EV_JOINED = EV_JOIN + 1
EV_QK_FREE = 16 - QK_DEPTH
_AIC_RESERVED = (11, 12, 13)
_aic_ids = (
    [EV_K_READY + i for i in range(STAGE_DEPTH)]
    + [EV_RING_FREE + i for i in range(STAGE_DEPTH)]
    + [EV_QK_READY + i for i in range(QK_DEPTH)]
    + [EV_QK_FREE + i for i in range(QK_DEPTH)]
)
_aiv_only_ids = [EV_JOIN, EV_JOINED]
assert len(set(_aic_ids + _aiv_only_ids)) == len(_aic_ids) + len(_aiv_only_ids), (
    f"cross-core event ids collide: AIC {_aic_ids}, AIV-only {_aiv_only_ids}"
)
assert all((0 <= i <= 15 for i in _aic_ids + _aiv_only_ids)), "event id out of 0..15"
assert not set(_aic_ids) & set(_AIC_RESERVED), (
    f"AIC-visible ids {sorted(set(_aic_ids) & set(_AIC_RESERVED))} are reserved by sync_all(MIX); "
    f"STAGE_DEPTH={STAGE_DEPTH} QK_DEPTH={QK_DEPTH}"
)
assert not set(_aiv_only_ids) & {12, 13}, "AIV-side ids 12..13 are sync_all(MIX)'s"
SY_V_TO_MTE3 = 0
SY_MTE2_TO_V = 1
SY_V_TO_S = 2
SY_MTE3_TO_V = 3
SY_V_TO_MTE2 = 4
SY_S_TO_MTE2 = 5


def _ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pa_axis0_stride(tensor, name):
    strides = tuple((int(v) for v in tensor.stride()))
    expected = [1]
    for size in reversed(tuple((int(v) for v in tensor.shape[1:]))):
        expected.append(expected[-1] * size)
    inner = tuple(reversed(expected[:-1]))
    if strides[1:] != inner:
        raise ValueError(
            f"{name} may only be non-contiguous on axis 0; expected inner strides {inner}, got {strides[1:]}"
        )
    if strides[0] < expected[-1]:
        raise ValueError(f"{name} axis-0 rows must not overlap")
    return strides[0]


def _batch_consistency_enabled():
    try:
        import torch_npu

        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False


def _align_up(value, align):
    return _ceil_div(value, align) * align


METADATA_ELEMENTS = 1024
METADATA_FIELDS = 8
LI_RECORDS = 32
LD_RECORD_START = 36
LD_RECORD_LIMIT = 108
MERGE_SLOTS = 2 * MAX_CUBE_WORKERS
MERGE_SLOT_CAP = 32
CARRY_SLOTS = 2 * MAX_CUBE_WORKERS
WORKSPACE_SLOTS = MERGE_SLOTS + CARRY_SLOTS


@functools.lru_cache(maxsize=None)
def _ub_layout(topk, is_pa):
    out32_len = min(IDX_CHUNK, max(64, _align_up(topk, 64)))
    out16_len = max(128, _align_up(topk, 128))
    pub16 = max(PUB16_MIN, _align_up(topk, PUB16_MIN))
    off = {}
    cursor = [0]

    def alloc(name, nbytes, align=512):
        cursor[0] = _align_up(cursor[0], align)
        off[name] = cursor[0]
        cursor[0] += nbytes
        return off[name]

    slot_bytes = _align_up(BLOCKS_PER_AIV * PACKED_BLOCK_BYTES, 512)
    assert slot_bytes == BLOCKS_PER_AIV * CANDIDATE_BLOCK_SIZE * (PACKED_D + SCALE_K)
    alloc("ring", BUF_DEPTH * slot_bytes)
    alloc("qk", QK_DEPTH * TILE_M * HALF_TOKENS * 2)
    alloc("kscratch", QK_DEPTH * HALF_TOKENS * 2)
    alloc("key", MAX_TOKENS * 2)
    alloc("cand", CANDIDATE_CAPACITY * 4)
    alloc("cand_pf", CANDIDATE_CAPACITY * 4)
    alloc("phys", CANDIDATE_CAPACITY * 4)
    alloc("swap", CANDIDATE_CAPACITY * 4)
    if is_pa:
        alloc("table", PA_TABLE_CACHE_ENTRIES * 4)
    else:
        alloc("scale_pitch", CANDIDATE_BLOCK_SIZE * SCALE_BLOCK_BYTES)
    alloc("counts", CANDIDATE_CAPACITY * 4)
    alloc("total", 256)
    alloc("validm", HALF_TOKENS * 2)
    alloc("w_f32", 256)
    alloc("w_bf16", 256)
    alloc("hist_hi", 1024)
    alloc("hist_lo", 1024)
    alloc("sqz", 1024)
    alloc("idx_hi", 256)
    alloc("nk", 256)
    alloc("kth", 256)
    alloc("pos", (MAX_TOKENS + 2048) * 2)
    alloc("out_i32", out32_len * 4)
    alloc("out_u16", out16_len * 2)
    off["_total"] = cursor[0]
    off["_slot_bytes"] = slot_bytes
    off["_fits"] = cursor[0] <= USER_UB_BYTES
    off["_out32_len"] = out32_len
    off["_pub16"] = pub16
    off["_out16_len"] = out16_len
    return off


@pl.vector_function
def _weights_to_bf16_vf(w_bf16_tile, w_f32_tile):
    # FP32 lane i narrows into the low BF16 halfword of a 32-bit container;
    # pack produces a dense BF16 vector, and the 32-head mask limits the store.
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m16 = vf.update_mask(N1, dtype=pl.DT_UINT16)
    value = vf.load_align(w_f32_tile, 0)
    narrow = vf.astype(value, preg32, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
    narrow_u = vf.bit_cast(narrow, dtype=pl.DT_UINT32)
    packed = vf.pack(narrow_u, dtype=pl.DT_UINT16)
    vf.store_align(w_bf16_tile, packed, m16)
    vf.mem_bar()


@pl.vector_function
def _count_visible_vf(
    counts_tile,
    total_tile,
    cand_tile,
    valid_s2: pl.DT_INT64,
    cand_len: pl.DT_INT64,
    blocks: pl.DT_INT64,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    total = vf.full(0, preg, dtype=pl.DT_INT32)
    vs2 = vf.full(valid_s2, preg, dtype=pl.DT_INT32)
    clen = vf.full(cand_len, preg, dtype=pl.DT_INT32)
    zero_i = vf.full(0, preg, dtype=pl.DT_INT32)
    for chunk in pl.range(0, (blocks + 63) // 64):
        ids = vf.load_align(cand_tile, chunk * 64)
        scaled = vf.muls(ids, CANDIDATE_BLOCK_SIZE, preg)
        count = vf.sub(vs2, scaled, preg)
        count = vf.maxs(count, 0, preg)
        count = vf.mins(count, CANDIDATE_BLOCK_SIZE, preg)
        lane = vf.arange(chunk * 64, dtype=pl.DT_INT32)
        in_range = vf.lt(lane, clen, preg)
        count = vf.select(count, zero_i, in_range)
        vf.store_align(counts_tile + chunk * 64, count, preg)
        total = vf.add(total, count, preg)
    red = vf.reduce_sum(total, preg)
    vf.store_align(total_tile, red, preg)
    vf.mem_bar()


@pl.vector_function
def _valid_mask_vf(validm_tile, counts_tile, base_block: pl.DT_INT64):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    m16 = vf.update_mask(64, dtype=pl.DT_UINT16)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    seven = vf.full(CANDIDATE_BLOCK_SIZE - 1, preg32, dtype=pl.DT_UINT32)
    for half in pl.range(0, 4):
        lane = vf.arange(half * 64, dtype=pl.DT_UINT32)
        offset = vf.and_(lane, seven, preg32)
        counts = vf.load_align(
            counts_tile,
            base_block + half * 8,
            dist=pl.LoadDist.E2B_B32,
            dtype=pl.DT_UINT32,
        )
        inb = vf.lt(offset, counts, preg32)
        valid = vf.select(one32, zero32, inb)
        packed = vf.pack(valid, dtype=pl.DT_UINT16)
        vf.store_align(validm_tile + half * 64, packed, m16)
    vf.mem_bar()


@pl.vector_function
def _identity_u32_vf(tile, chunks: pl.DT_INT64):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lane = vf.arange(0, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, chunks):
        row = vf.adds(lane, chunk * 64, preg32)
        vf.store_align(tile + chunk * 64, row, preg32)
    vf.mem_bar()


@pl.vector_function
def _zero_u32_vf(tile, chunks: pl.DT_INT64):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, chunks):
        vf.store_align(tile + chunk * 64, zero32, preg32)
    vf.mem_bar()


@pl.vector_function
def _reduce_heads_vf(
    qk_tile,
    w_bf16_tile,
    key_tile,
    validm_tile,
    swap_tile,
    tile_idx: pl.DT_INT64,
    key_off: pl.DT_INT64,
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    acc0 = vf.full(0.0, preg16, dtype=pl.DT_BF16)
    acc1 = vf.full(0.0, preg16, dtype=pl.DT_BF16)
    for g in pl.range(0, N1 // HEADS_PER_GROUP):
        h = g * HEADS_PER_GROUP
        w0 = vf.load_align(w_bf16_tile, h, dist=pl.LoadDist.BRC_B16)
        qa0 = vf.load_align(qk_tile, h * HALF_TOKENS)
        qa1 = vf.load_align(qk_tile, h * HALF_TOKENS + 128)
        qa0 = vf.maxs(qa0, 0.0, preg16)
        qa1 = vf.maxs(qa1, 0.0, preg16)
        acc0 = vf.mul_add_dst(qa0, w0, preg16)
        acc1 = vf.mul_add_dst(qa1, w0, preg16)
        w1 = vf.load_align(w_bf16_tile, h + 1, dist=pl.LoadDist.BRC_B16)
        qb0 = vf.load_align(qk_tile, (h + 1) * HALF_TOKENS)
        qb1 = vf.load_align(qk_tile, (h + 1) * HALF_TOKENS + 128)
        qb0 = vf.maxs(qb0, 0.0, preg16)
        qb1 = vf.maxs(qb1, 0.0, preg16)
        acc0 = vf.mul_add_dst(qb0, w1, preg16)
        acc1 = vf.mul_add_dst(qb1, w1, preg16)
        w2 = vf.load_align(w_bf16_tile, h + 2, dist=pl.LoadDist.BRC_B16)
        qc0 = vf.load_align(qk_tile, (h + 2) * HALF_TOKENS)
        qc1 = vf.load_align(qk_tile, (h + 2) * HALF_TOKENS + 128)
        qc0 = vf.maxs(qc0, 0.0, preg16)
        qc1 = vf.maxs(qc1, 0.0, preg16)
        acc0 = vf.mul_add_dst(qc0, w2, preg16)
        acc1 = vf.mul_add_dst(qc1, w2, preg16)
        w3 = vf.load_align(w_bf16_tile, h + 3, dist=pl.LoadDist.BRC_B16)
        qd0 = vf.load_align(qk_tile, (h + 3) * HALF_TOKENS)
        qd1 = vf.load_align(qk_tile, (h + 3) * HALF_TOKENS + 128)
        qd0 = vf.maxs(qd0, 0.0, preg16)
        qd1 = vf.maxs(qd1, 0.0, preg16)
        acc0 = vf.mul_add_dst(qd0, w3, preg16)
        acc1 = vf.mul_add_dst(qd1, w3, preg16)
    word_lane = vf.arange(0, dtype=pl.DT_UINT32)
    local_block = vf.shift_right(word_lane, 2, preg32)
    block0 = vf.adds(local_block, tile_idx * BLOCKS_PER_AIV, preg32)
    block1 = vf.adds(block0, 16, preg32)
    sw0 = vf.gather(swap_tile, block0, preg32)
    sw1 = vf.gather(swap_tile, block1, preg32)
    perm0 = vf.xor(word_lane, sw0, preg32)
    perm1 = vf.xor(word_lane, sw1, preg32)
    acc0_u = vf.bit_cast(acc0, dtype=pl.DT_UINT32)
    acc1_u = vf.bit_cast(acc1, dtype=pl.DT_UINT32)
    acc0_p = vf.gather(acc0_u, perm0)
    acc1_p = vf.gather(acc1_u, perm1)
    acc0 = vf.bit_cast(acc0_p, dtype=pl.DT_BF16)
    acc1 = vf.bit_cast(acc1_p, dtype=pl.DT_BF16)
    sign = vf.full(32768, preg_u16, dtype=pl.DT_UINT16)
    allone = vf.full(65535, preg_u16, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg_u16, dtype=pl.DT_UINT16)
    one16 = vf.full(1, preg_u16, dtype=pl.DT_UINT16)
    bits0 = vf.bit_cast(acc0, dtype=pl.DT_UINT16)
    masked0 = vf.and_(bits0, sign, preg_u16)
    neg0 = vf.eq(masked0, sign, preg_u16)
    nkey0 = vf.xor(bits0, allone, preg_u16)
    pkey0 = vf.xor(bits0, sign, preg_u16)
    key0 = vf.select(nkey0, pkey0, neg0)
    v0 = vf.load_align(validm_tile, 0)
    keep0 = vf.eq(v0, one16, preg_u16)
    key0 = vf.select(key0, zero16, keep0)
    vf.store_align(key_tile + key_off, key0, preg_u16)
    bits1 = vf.bit_cast(acc1, dtype=pl.DT_UINT16)
    masked1 = vf.and_(bits1, sign, preg_u16)
    neg1 = vf.eq(masked1, sign, preg_u16)
    nkey1 = vf.xor(bits1, allone, preg_u16)
    pkey1 = vf.xor(bits1, sign, preg_u16)
    key1 = vf.select(nkey1, pkey1, neg1)
    v1 = vf.load_align(validm_tile, 128)
    keep1 = vf.eq(v1, one16, preg_u16)
    key1 = vf.select(key1, zero16, keep1)
    vf.store_align(key_tile + key_off + 128, key1, preg_u16)
    vf.mem_bar()


@pl.vector_function
def _zero_keys_vf(key_tile, begin_chunk: pl.DT_INT64, end_chunk: pl.DT_INT64):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(begin_chunk, end_chunk):
        vf.store_align(key_tile + chunk * 128, zero16, preg16)
    vf.mem_bar()


@pl.vector_function
def _hist_high_vf(hist_high_tile, key_tile, hist_loop: pl.DT_INT64):
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    cout0 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    cout1 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    for i in pl.range(0, hist_loop):
        low, high = vf.load_align(
            key_tile, i * 256, dist=pl.LoadDist.DINTLV_B8, dtype=pl.DT_UINT8
        )
        cout0 = vf.histograms(
            high, preg_b8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        cout1 = vf.histograms(
            high, preg_b8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
    c0e = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c0o = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    c1e = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c1o = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    vf.store_align(hist_high_tile, c0e, c0o, preg_b32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(
        hist_high_tile + 128, c1e, c1o, preg_b32, dist=pl.StoreDist.INTLV_B32
    )
    vf.mem_bar()


@pl.vector_function
def _find_high_and_hist_low_vf(
    hist_low_tile,
    idx_high_tile,
    hist_high_tile,
    score_u16_tile,
    sqz_idx_buf,
    nk_value_tile,
    bottom_k,
    hist_loop,
    base=0,
):
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.full(bottom_k, p32, dtype=U32)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_high_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    high = vf.full(minimum, p32)
    prev = vf.maxs(high, 1, p32)
    one = vf.full(1, p32, dtype=U32)
    prev = vf.sub(prev, one, p32)
    counts = vf.gather(hist_high_tile, prev, p32)
    zero = vf.full(0, p32, dtype=U32)
    has_prev = vf.gt(high, zero, p32)
    preceding = vf.select(counts, zero, has_prev)
    next_k = vf.sub(target, preceding, p32)
    repeated = vf.muls(high, 16843009, p32)
    vf.store_align(idx_high_tile, repeated, p32, 0)
    vf.store_align(nk_value_tile, next_k, p32, 0)
    high8 = vf.bit_cast(repeated, dtype=pl.DT_UINT8)
    count0 = vf.full(0, p16, dtype=U16)
    count1 = vf.full(0, p16, dtype=U16)
    for i in pl.range(hist_loop):
        low, hi = vf.load_align(
            score_u16_tile,
            base + i * 256,
            dist=pl.LoadDist.DINTLV_B8,
            dtype=pl.DT_UINT8,
        )
        match = vf.eq(hi, high8, p8)
        count0 = vf.histograms(
            low, match, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        count1 = vf.histograms(
            low, match, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
    e0 = vf.astype(count0, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o0 = vf.astype(count0, p16, layout=pl.CastLayout.ONE, dtype=U32)
    e1 = vf.astype(count1, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o1 = vf.astype(count1, p16, layout=pl.CastLayout.ONE, dtype=U32)
    vf.store_align(hist_low_tile, e0, o0, p32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(hist_low_tile + 128, e1, o1, p32, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def _find_kth_vf(
    kth_value_tile, hist_low_tile, idx_high_tile, sqz_idx_buf, nk_value_tile
):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.load_align(nk_value_tile, 0)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_low_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    low = vf.full(minimum, p32)
    high = vf.load_align(idx_high_tile, 0)
    mask = vf.full(255, p32, dtype=U32)
    high = vf.and_(high, mask, p32)
    high = vf.shift_left(high, 8, p32)
    key = vf.or_(high, low, p32)
    upper = vf.shift_left(key, 16, p32)
    both = vf.or_(key, upper, p32)
    key16 = vf.bit_cast(both, dtype=U16)
    vf.store_align(kth_value_tile, key16, p16, 0)


@pl.vector_function
def _collect_gt_eq_vf(pos_tile, key_tile, kth_tile, vf_loop: pl.DT_INT64):
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    vf.clear_spr()
    align_idx = vf.unalign_reg_for_store()
    kth = vf.load_align(kth_tile, 0, dist=pl.LoadDist.BRC_B16, dtype=pl.DT_UINT16)
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(i * 128, dtype=pl.DT_INT16)
        keys = vf.load_align(key_tile, i * 128)
        pout = vf.gt(keys, kth, preg_b16)
        sqz = vf.squeeze(idx_c, pout, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(pos_tile, sqz, align_idx)
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(i * 128, dtype=pl.DT_INT16)
        keys = vf.load_align(key_tile, i * 128)
        pout = vf.eq(keys, kth, preg_b16)
        sqz = vf.squeeze(idx_c, pout, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(pos_tile, sqz, align_idx)
    vf.squeeze_store_unalign_post(pos_tile, align_idx)
    vf.mem_bar()


@pl.vector_function
def _collect_valid_vf(pos_tile, key_tile, vf_loop: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    zero = vf.full(0, p, dtype=pl.DT_UINT16)
    vf.clear_spr()
    ar = vf.unalign_reg_for_store()
    for chunk in pl.range(vf_loop):
        positions = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        keys = vf.load_align(key_tile, chunk * 128)
        valid = vf.gt(keys, zero, p)
        packed = vf.squeeze(positions, valid, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(pos_tile, packed, ar)
    vf.squeeze_store_unalign_post(pos_tile, ar)
    vf.mem_bar()


@pl.vector_function
def _fill_invalid_vf(
    out_u16_tile, out_i32_tile, loops16: pl.DT_INT64, loops32: pl.DT_INT64
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    neg_inf = vf.full(BF16_NEG_INF_BITS, preg16, dtype=pl.DT_UINT16)
    neg_one = vf.full(-1, preg32, dtype=pl.DT_INT32)
    for c in pl.range(0, loops16):
        vf.store_align(out_u16_tile + c * 128, neg_inf, preg16)
    for c in pl.range(0, loops32):
        vf.store_align(out_i32_tile + c * 64, neg_one, preg32)
    vf.mem_bar()


@pl.vector_function
def _map_positions_vf(
    out_i32_tile,
    pos_tile,
    cand_tile,
    valid_count: pl.DT_INT64,
    idx_offset: pl.DT_INT64,
    base: pl.DT_INT64,
    chunk_base: pl.DT_INT64,
    loops: pl.DT_INT64,
):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    vcnt = vf.full(valid_count, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    seven = vf.full(CANDIDATE_BLOCK_SIZE - 1, preg32, dtype=pl.DT_UINT32)
    neg1 = vf.full(-1, preg32, dtype=pl.DT_INT32)
    for chunk in pl.range(0, loops):
        raw = vf.load_align(pos_tile, chunk_base + chunk * 64, dtype=pl.DT_UINT16)
        pos = vf.unpack(raw, part=pl.PackPart.LOWER, dtype=pl.DT_UINT32)
        lane = vf.arange(chunk_base + chunk * 64, dtype=pl.DT_UINT32)
        valid = vf.lt(lane, vcnt, preg32)
        pos = vf.select(pos, zero32, valid)
        pos = vf.adds(pos, base, preg32)
        block_pos = vf.shift_right(pos, 3, preg32)
        block_id = vf.gather(cand_tile, block_pos, preg32)
        delta = vf.and_(pos, seven, preg32)
        delta_i = vf.bit_cast(delta, dtype=pl.DT_INT32)
        scaled = vf.muls(block_id, CANDIDATE_BLOCK_SIZE, preg32)
        logical = vf.add(scaled, delta_i, preg32)
        logical = vf.adds(logical, idx_offset, preg32)
        out = vf.select(logical, neg1, valid)
        vf.store_align(out_i32_tile + chunk * 64, out, preg32)
    vf.mem_bar()


@pl.vector_function
def _map_values_vf(
    out_u16_tile, pos_tile, key_tile, valid_count: pl.DT_INT64, loops: pl.DT_INT64
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    sign = vf.full(32768, preg16, dtype=pl.DT_UINT16)
    allone = vf.full(65535, preg16, dtype=pl.DT_UINT16)
    neg_inf = vf.full(BF16_NEG_INF_BITS, preg16, dtype=pl.DT_UINT16)
    vcnt = vf.full(valid_count, preg16, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(0, loops):
        raw = vf.load_align(pos_tile, chunk * 128)
        lane = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        valid = vf.lt(lane, vcnt, preg16)
        pos = vf.select(raw, zero16, valid)
        sel = vf.gather(key_tile, pos, preg16)
        masked = vf.and_(sel, sign, preg16)
        positive = vf.eq(masked, sign, preg16)
        a = vf.xor(sel, sign, preg16)
        b = vf.xor(sel, allone, preg16)
        bits = vf.select(a, b, positive)
        bits = vf.select(bits, neg_inf, valid)
        vf.store_align(out_u16_tile + chunk * 128, bits, preg16)
    vf.mem_bar()


@pl.vector_function
def _keys_to_values_vf(out_u16_tile, valid_count: pl.DT_INT64, loops: pl.DT_INT64):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    sign = vf.full(32768, preg16, dtype=pl.DT_UINT16)
    allone = vf.full(65535, preg16, dtype=pl.DT_UINT16)
    neg_inf = vf.full(BF16_NEG_INF_BITS, preg16, dtype=pl.DT_UINT16)
    vcnt = vf.full(valid_count, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(0, loops):
        sel = vf.load_align(out_u16_tile, chunk * 128)
        lane = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        valid = vf.lt(lane, vcnt, preg16)
        masked = vf.and_(sel, sign, preg16)
        positive = vf.eq(masked, sign, preg16)
        a = vf.xor(sel, sign, preg16)
        b = vf.xor(sel, allone, preg16)
        bits = vf.select(a, b, positive)
        bits = vf.select(bits, neg_inf, valid)
        vf.store_align(out_u16_tile + chunk * 128, bits, preg16)
    vf.mem_bar()


@pl.vector_function
def _map_keys_vf(
    out_key_tile, pos_tile, key_tile, valid_count: pl.DT_INT64, loops: pl.DT_INT64
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg16, dtype=pl.DT_UINT16)
    vcnt = vf.full(valid_count, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(0, loops):
        raw = vf.load_align(pos_tile, chunk * 128)
        lane = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        valid = vf.lt(lane, vcnt, preg16)
        pos = vf.select(raw, zero16, valid)
        sel = vf.gather(key_tile, pos, preg16)
        out = vf.select(sel, zero16, valid)
        vf.store_align(out_key_tile + chunk * 128, out, preg16)
    vf.mem_bar()


@pl.vector_function
def _map_rawpos_vf(
    pos_tile, valid_count: pl.DT_INT64, base: pl.DT_INT64, loops: pl.DT_INT64
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg16, dtype=pl.DT_UINT16)
    vcnt = vf.full(valid_count, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(0, loops):
        raw = vf.load_align(pos_tile, chunk * 128)
        lane = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        valid = vf.lt(lane, vcnt, preg16)
        pos = vf.select(raw, zero16, valid)
        pos = vf.adds(pos, base, preg16)
        vf.store_align(pos_tile + chunk * 128, pos, preg16)
    vf.mem_bar()


@pl.vector_function
def _resolve_merge_pos_vf(
    pos_tile, mpos_tile, valid_count: pl.DT_INT64, loops: pl.DT_INT64
):
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    zero16 = vf.full(0, preg16, dtype=pl.DT_UINT16)
    vcnt = vf.full(valid_count, preg16, dtype=pl.DT_UINT16)
    for chunk in pl.range(0, loops):
        raw = vf.load_align(pos_tile, chunk * 128)
        lane = vf.arange(chunk * 128, dtype=pl.DT_UINT16)
        valid = vf.lt(lane, vcnt, preg16)
        pos = vf.select(raw, zero16, valid)
        resolved = vf.gather(mpos_tile, pos, preg16)
        vf.store_align(pos_tile + chunk * 128, resolved, preg16)
    vf.mem_bar()


@pl.vector_function
def _count_positive_vf(total_tile, key_tile, chunks: pl.DT_INT64):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    acc = vf.full(0, preg32, dtype=pl.DT_UINT32)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, chunks):
        raw = vf.load_align(key_tile, chunk * 128)
        low = vf.unpack(raw, part=pl.PackPart.LOWER, dtype=pl.DT_UINT32)
        high = vf.unpack(raw, part=pl.PackPart.UPPER, dtype=pl.DT_UINT32)
        hit_low = vf.gt(low, zero32, preg32)
        add_low = vf.select(one32, zero32, hit_low)
        acc = vf.add(acc, add_low, preg32)
        hit_high = vf.gt(high, zero32, preg32)
        add_high = vf.select(one32, zero32, hit_high)
        acc = vf.add(acc, add_high, preg32)
    red = vf.reduce_sum(acc, preg32)
    vf.store_align(total_tile, red, preg32)
    vf.mem_bar()


@pl.vector_function
def _split_page_general_vf(
    page,
    within,
    cand_u,
    recip_reg,
    one32,
    preg32,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    _recip_lo, page = vf.mull(cand_u, recip_reg, preg32)
    product = vf.muls(page, blocks_per_page, preg32)
    over = vf.gt(product, cand_u, preg32)
    page_dec = vf.sub(page, one32, preg32)
    page = vf.select(page_dec, page, over)
    folded = vf.muls(page, blocks_per_page, preg32)
    within = vf.sub(cand_u, folded, preg32)  # noqa: F841 - writes the caller register


@pl.vector_function
def _split_page_pow2_vf(
    page,
    within,
    cand_u,
    recip_reg,
    one32,
    preg32,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    page_mask_reg = vf.full(blocks_per_page - 1, preg32, dtype=pl.DT_UINT32)
    page = vf.shift_right(cand_u, page_param, preg32)  # noqa: F841 - writes the caller register
    within = vf.and_(cand_u, page_mask_reg, preg32)  # noqa: F841 - writes the caller register


def _prepare_addresses_scalar(
    phys_tile,
    candidates,
    block_table,
    row,
    batch_idx,
    valid_blocks: pl.DT_INT64,
    key_stride: pl.DT_INT64,
    tile_end: pl.DT_INT64,
    sub_base: pl.DT_INT64,
    blocks_per_page: pl.DT_INT64,
    has_table: pl.DT_INT64,
):
    for t in pl.range(0, tile_end):
        for i in pl.range(0, BLOCKS_PER_AIV):
            pos = pl.min(t * BLOCKS_PER_TILE + sub_base + i, valid_blocks - 1)
            cand = pl.max(0, candidates[row, pos])
            page = cand // blocks_per_page
            if has_table != 0:
                physical = block_table[batch_idx, page]
            else:
                physical = page
            pl.setval(
                phys_tile,
                t * BLOCKS_PER_AIV + i,
                physical * key_stride
                + (cand - page * blocks_per_page) * PACKED_BLOCK_BYTES,
            )


def _pair_load(slot, phys_tile, idx, key_ptr, pair_min, packed32):
    if packed32:
        packed = pl.getval(phys_tile, idx)
        a0 = packed & 4294967295
        a1 = packed >> 32 & 4294967295
    else:
        a0 = pl.getval(phys_tile, idx)
        a1 = pl.getval(phys_tile, idx + 1)
    gap = a1 - a0
    if pair_min != 0 and gap >= pair_min:
        view = pl.make_tensor(pl.addptr(key_ptr, a0), [2, PACKED_BLOCK_BYTES], [gap, 1])
        pl.load(slot, view, [0, 0])
    else:
        one0 = pl.make_tensor(
            pl.addptr(key_ptr, a0), [1, PACKED_BLOCK_BYTES], [PACKED_BLOCK_BYTES, 1]
        )
        pl.load(slot[0:1, 0:PACKED_BLOCK_BYTES], one0, [0, 0])
        one1 = pl.make_tensor(
            pl.addptr(key_ptr, a1), [1, PACKED_BLOCK_BYTES], [PACKED_BLOCK_BYTES, 1]
        )
        pl.load(slot[1:2, 0:PACKED_BLOCK_BYTES], one1, [0, 0])


def _gather_pa(pair, phys_tile, key_ptr, base_idx, slot_base, pair_min, packed32=False):
    h = base_idx >> 1 if packed32 else base_idx
    step = 1 if packed32 else 2
    _pair_load(
        pair[slot_base + 0], phys_tile, h + 0 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 1], phys_tile, h + 1 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 2], phys_tile, h + 2 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 3], phys_tile, h + 3 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 4], phys_tile, h + 4 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 5], phys_tile, h + 5 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 6], phys_tile, h + 6 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 7], phys_tile, h + 7 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 8], phys_tile, h + 8 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 9], phys_tile, h + 9 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 10], phys_tile, h + 10 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 11], phys_tile, h + 11 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 12], phys_tile, h + 12 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 13], phys_tile, h + 13 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 14], phys_tile, h + 14 * step, key_ptr, pair_min, packed32
    )
    _pair_load(
        pair[slot_base + 15], phys_tile, h + 15 * step, key_ptr, pair_min, packed32
    )


def _stage_tile_pa(
    pair,
    ring,
    phys_tile,
    key_ptr,
    stg_k512,
    stg_s32,
    tick,
    worker,
    sub_base,
    pair_min,
    ring_id0,
    ring_id1,
    packed32=False,
):
    stage = tick & STAGE_MASK
    dest_row = (worker * STAGE_DEPTH + stage) * ROWS_PER_STAGE + sub_base
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=EV_RING_FREE + stage)
    if tick & BUF_MASK == 0:
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=ring_id0)
        _gather_pa(
            pair, phys_tile, key_ptr, tick * BLOCKS_PER_AIV, 0, pair_min, packed32
        )
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=ring_id0)
        pl.store(stg_k512, ring[0][0:BLOCKS_PER_AIV, 0:KEY_BLOCK_BYTES], [dest_row, 0])
        pl.store(
            stg_s32,
            ring[0][0:BLOCKS_PER_AIV, KEY_BLOCK_BYTES:PACKED_BLOCK_BYTES],
            [dest_row, 0],
        )
    else:
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=ring_id1)
        _gather_pa(
            pair,
            phys_tile,
            key_ptr,
            tick * BLOCKS_PER_AIV,
            BLOCKS_PER_AIV // 2,
            pair_min,
            packed32,
        )
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=ring_id1)
        pl.store(stg_k512, ring[1][0:BLOCKS_PER_AIV, 0:KEY_BLOCK_BYTES], [dest_row, 0])
        pl.store(
            stg_s32,
            ring[1][0:BLOCKS_PER_AIV, KEY_BLOCK_BYTES:PACKED_BLOCK_BYTES],
            [dest_row, 0],
        )
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_K_READY + stage)


def _stage_tile_tnd(
    ring,
    ring_s,
    blk_k,
    blk_s,
    blk_sp,
    cand_tile,
    key_ptr,
    dk_ptr,
    stg_k64,
    stg_s32,
    stg_s_ptr,
    tick,
    worker,
    sub,
    sub_base,
    k_begin,
    k_len,
    valid_blocks,
    k_stride,
    s_stride,
    rows,
):
    buf = tick & BUF_MASK
    stage = tick & STAGE_MASK
    row_base = (worker * STAGE_DEPTH + stage) * TILE_N
    scale_row = (worker * STAGE_DEPTH + stage) * ROWS_PER_STAGE + sub_base
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=EV_RING_FREE + stage)
    if buf == 0:
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=31)
        if s_stride == SCALE_K:
            pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=29)
    else:
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=30)
        if s_stride == SCALE_K:
            pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=28)
    if buf == 0:
        _gather_tnd(
            blk_k,
            blk_s,
            blk_sp,
            cand_tile,
            key_ptr,
            dk_ptr,
            stg_s_ptr,
            k_begin,
            k_len,
            tick * BLOCKS_PER_TILE + sub_base,
            valid_blocks,
            0,
            scale_row,
            k_stride,
            s_stride,
            rows,
        )
    else:
        _gather_tnd(
            blk_k,
            blk_s,
            blk_sp,
            cand_tile,
            key_ptr,
            dk_ptr,
            stg_s_ptr,
            k_begin,
            k_len,
            tick * BLOCKS_PER_TILE + sub_base,
            valid_blocks,
            32,
            scale_row,
            k_stride,
            s_stride,
            rows,
        )

    if buf == 0:
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=31)
        if s_stride == SCALE_K:
            pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=29)
    else:
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=30)
        if s_stride == SCALE_K:
            pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=28)
    pl.store(stg_k64, ring[buf], [row_base + sub * HALF_TOKENS, 0])
    if s_stride == SCALE_K:
        pl.store(stg_s32, ring_s[buf], [scale_row, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_K_READY + stage)


def _gather_tnd_block(
    blk_k,
    blk_s,
    blk_sp,
    cand_tile,
    key_ptr,
    dk_ptr,
    stg_s_ptr,
    k_begin,
    batch_len,
    base_block,
    valid_blocks,
    slot_base,
    scale_row,
    k_stride,
    s_stride,
    rows,
    j,
):
    idx = pl.min(base_block + j, valid_blocks - 1)
    blk = pl.max(0, pl.getval(cand_tile, idx))
    count = pl.max(
        1, pl.min(CANDIDATE_BLOCK_SIZE, batch_len - blk * CANDIDATE_BLOCK_SIZE)
    )
    tok = pl.min(k_begin + blk * CANDIDATE_BLOCK_SIZE, rows - count)
    kview = pl.make_tensor(
        pl.addptr(key_ptr, tok * k_stride),
        [CANDIDATE_BLOCK_SIZE, PACKED_D],
        [k_stride, 1],
    )
    kt = blk_k[slot_base + j]
    pl.set_validshape(kt, [count, PACKED_D])
    pl.load(kt, kview, [0, 0])
    if s_stride == SCALE_K:
        sview = pl.make_tensor(
            pl.addptr(dk_ptr, tok * s_stride),
            [1, SCALE_BLOCK_BYTES],
            [SCALE_BLOCK_BYTES, 1],
        )
        st = blk_s[slot_base + j]
        pl.set_validshape(st, [1, count * SCALE_K])
        pl.load(st, sview, [0, 0])
    else:
        sview = pl.make_tensor(
            pl.addptr(dk_ptr, tok * s_stride),
            [CANDIDATE_BLOCK_SIZE, SCALE_K],
            [s_stride, 1],
        )
        pl.set_validshape(blk_sp, [count, SCALE_K])
        pl.load(blk_sp.current(), sview, [0, 0])
        dview = pl.make_tensor(
            pl.addptr(stg_s_ptr, (scale_row + j) * SCALE_BLOCK_BYTES),
            [CANDIDATE_BLOCK_SIZE, SCALE_K],
            [SCALE_K, 1],
        )
        pl.store(dview, blk_sp.current(), [0, 0])


def _gather_tnd_quad(
    blk_k,
    blk_s,
    blk_sp,
    cand_tile,
    key_ptr,
    dk_ptr,
    stg_s_ptr,
    k_begin,
    batch_len,
    base_block,
    valid_blocks,
    slot_base,
    scale_row,
    k_stride,
    s_stride,
    rows,
    base,
):
    _gather_tnd_block(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        base + 0,
    )
    _gather_tnd_block(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        base + 1,
    )
    _gather_tnd_block(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        base + 2,
    )
    _gather_tnd_block(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        base + 3,
    )


def _gather_tnd(
    blk_k,
    blk_s,
    blk_sp,
    cand_tile,
    key_ptr,
    dk_ptr,
    stg_s_ptr,
    k_begin,
    batch_len,
    base_block,
    valid_blocks,
    slot_base,
    scale_row,
    k_stride,
    s_stride,
    rows,
):
    # Constant tile indices avoid descriptor arrays and runtime loop overhead.
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        0,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        4,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        8,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        12,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        16,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        20,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        24,
    )
    _gather_tnd_quad(
        blk_k,
        blk_s,
        blk_sp,
        cand_tile,
        key_ptr,
        dk_ptr,
        stg_s_ptr,
        k_begin,
        batch_len,
        base_block,
        valid_blocks,
        slot_base,
        scale_row,
        k_stride,
        s_stride,
        rows,
        28,
    )


class QsliKey:
    WriteValues = TilingKeyField(bits=1, values=[0, 1])
    IsPA = TilingKeyField(bits=1, values=[0, 1])
    TopKCapacity = TilingKeyField(
        bits=4, values=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )
    TndStrideMode = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    PairMin = TilingKeyField(bits=1, values=[544, 0])
    # Uncached table, cached table with 32-bit offsets, cached table with 64-bit offsets.
    # 0: uncached scalar addresses; 1/2: cached table with 32/64-bit offsets.
    PAGatherMode = TilingKeyField(bits=2, values=[0, 1, 2])


_LAYOUTS = {0: _ub_layout(512, False), 1: _ub_layout(512, True)}
PAGE_PARAMS = tuple(
    (
        (n.bit_length() - 1 if n and (not n & n - 1) else (1 << 32) // n + 1)
        if n
        else 0
        for n in range(129)
    )
)
_RING_IDS = [31, 30]
_SCALE_IDS = [29, 28]
MX_KEY = 27
MX_PHYS = 26
_RING_ADDRS = [slot * 17408 for slot in range(BUF_DEPTH)]
_PAIR_ADDRS = [
    slot * 17408 + pair * 1088
    for slot in range(BUF_DEPTH)
    for pair in range(BLOCKS_PER_AIV // 2)
]
_TND_SCALE_ADDRS = [
    slot * 17408 + BLOCKS_PER_AIV * KEY_BLOCK_BYTES for slot in range(BUF_DEPTH)
]
_TND_KEY_BLOCK_ADDRS = [
    slot * 17408 + block * KEY_BLOCK_BYTES
    for slot in range(BUF_DEPTH)
    for block in range(BLOCKS_PER_AIV)
]
_TND_KEY_IDS = [
    _RING_IDS[slot] for slot in range(BUF_DEPTH) for _ in range(BLOCKS_PER_AIV)
]
_TND_SCALE_BLOCK_ADDRS = [
    slot * 17408 + BLOCKS_PER_AIV * KEY_BLOCK_BYTES + block * SCALE_BLOCK_BYTES
    for slot in range(BUF_DEPTH)
    for block in range(BLOCKS_PER_AIV)
]
_TND_SCALE_IDS = [
    _SCALE_IDS[slot] for slot in range(BUF_DEPTH) for _ in range(BLOCKS_PER_AIV)
]


@pl.vector_function
def _prepare_addresses32_general_vf(
    phys_tile,
    swap_tile,
    cand_tile,
    table_tile,
    valid_blocks: pl.DT_INT64,
    stride_lo: pl.DT_INT64,
    stride_hi: pl.DT_INT64,
    tile_end: pl.DT_INT64,
    sub_base: pl.DT_INT64,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lane = vf.arange(0, dtype=pl.DT_UINT32)
    mask31 = vf.full(BLOCKS_PER_AIV - 1, preg32, dtype=pl.DT_UINT32)
    in_half = vf.and_(lane, mask31, preg32)
    hi = vf.shift_right(lane, 5, preg32)
    next_tile = vf.muls(hi, BLOCKS_PER_TILE, preg32)
    position_offsets = vf.add(in_half, next_tile, preg32)
    stride_lo_reg = vf.full(stride_lo, preg32, dtype=pl.DT_UINT32)
    even_mask = vf.full(4294967294, preg32, dtype=pl.DT_UINT32)
    even_lane = vf.and_(lane, even_mask, preg32)
    odd_lane = vf.adds(even_lane, 1, preg32)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    recip_reg = vf.full(page_param, preg32, dtype=pl.DT_UINT32)
    page = vf.full(0, preg32, dtype=pl.DT_UINT32)
    within = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, tile_end, 2):
        base = chunk * BLOCKS_PER_TILE + sub_base
        logical = vf.adds(position_offsets, base, preg32)
        safe = vf.mins(logical, valid_blocks - 1, preg32)
        cand = vf.gather(cand_tile, safe, preg32)
        cand = vf.maxs(cand, 0, preg32)
        cand_u = vf.bit_cast(cand, dtype=pl.DT_UINT32)
        _split_page_general_vf(
            page, within, cand_u, recip_reg, one32, preg32, blocks_per_page, page_param
        )
        phys_page = vf.gather(table_tile, page, preg32)
        phys_u = vf.bit_cast(phys_page, dtype=pl.DT_UINT32)
        addr = vf.mul(phys_u, stride_lo_reg, preg32)
        byte_off = vf.muls(within, PACKED_BLOCK_BYTES, preg32)
        addr = vf.add(addr, byte_off, preg32)
        even_addr = vf.gather(addr, even_lane)
        odd_addr = vf.gather(addr, odd_lane)
        gt = vf.gt(even_addr, odd_addr, preg32)
        swap = vf.select(one32, zero32, gt)
        permutation = vf.xor(lane, swap, preg32)
        addr = vf.gather(addr, permutation)
        addr_i = vf.bit_cast(addr, dtype=pl.DT_INT32)
        swap4 = vf.muls(swap, 4, preg32)
        store_mask = vf.update_mask(
            (tile_end - chunk) * BLOCKS_PER_AIV, dtype=pl.DT_UINT32
        )
        vf.store_align(swap_tile + chunk * BLOCKS_PER_AIV, swap4, store_mask)
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV, addr_i, store_mask)
    vf.mem_bar()


@pl.vector_function
def _prepare_addresses32_pow2_vf(
    phys_tile,
    swap_tile,
    cand_tile,
    table_tile,
    valid_blocks: pl.DT_INT64,
    stride_lo: pl.DT_INT64,
    stride_hi: pl.DT_INT64,
    tile_end: pl.DT_INT64,
    sub_base: pl.DT_INT64,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lane = vf.arange(0, dtype=pl.DT_UINT32)
    mask31 = vf.full(BLOCKS_PER_AIV - 1, preg32, dtype=pl.DT_UINT32)
    in_half = vf.and_(lane, mask31, preg32)
    hi = vf.shift_right(lane, 5, preg32)
    next_tile = vf.muls(hi, BLOCKS_PER_TILE, preg32)
    position_offsets = vf.add(in_half, next_tile, preg32)
    stride_lo_reg = vf.full(stride_lo, preg32, dtype=pl.DT_UINT32)
    even_mask = vf.full(4294967294, preg32, dtype=pl.DT_UINT32)
    even_lane = vf.and_(lane, even_mask, preg32)
    odd_lane = vf.adds(even_lane, 1, preg32)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    recip_reg = vf.full(page_param, preg32, dtype=pl.DT_UINT32)
    page = vf.full(0, preg32, dtype=pl.DT_UINT32)
    within = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, tile_end, 2):
        base = chunk * BLOCKS_PER_TILE + sub_base
        logical = vf.adds(position_offsets, base, preg32)
        safe = vf.mins(logical, valid_blocks - 1, preg32)
        cand = vf.gather(cand_tile, safe, preg32)
        cand = vf.maxs(cand, 0, preg32)
        cand_u = vf.bit_cast(cand, dtype=pl.DT_UINT32)
        _split_page_pow2_vf(
            page, within, cand_u, recip_reg, one32, preg32, blocks_per_page, page_param
        )
        phys_page = vf.gather(table_tile, page, preg32)
        phys_u = vf.bit_cast(phys_page, dtype=pl.DT_UINT32)
        addr = vf.mul(phys_u, stride_lo_reg, preg32)
        byte_off = vf.muls(within, PACKED_BLOCK_BYTES, preg32)
        addr = vf.add(addr, byte_off, preg32)
        even_addr = vf.gather(addr, even_lane)
        odd_addr = vf.gather(addr, odd_lane)
        gt = vf.gt(even_addr, odd_addr, preg32)
        swap = vf.select(one32, zero32, gt)
        permutation = vf.xor(lane, swap, preg32)
        addr = vf.gather(addr, permutation)
        addr_i = vf.bit_cast(addr, dtype=pl.DT_INT32)
        swap4 = vf.muls(swap, 4, preg32)
        store_mask = vf.update_mask(
            (tile_end - chunk) * BLOCKS_PER_AIV, dtype=pl.DT_UINT32
        )
        vf.store_align(swap_tile + chunk * BLOCKS_PER_AIV, swap4, store_mask)
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV, addr_i, store_mask)
    vf.mem_bar()


@pl.vector_function
def _prepare_addresses64_general_vf(
    phys_tile,
    swap_tile,
    cand_tile,
    table_tile,
    valid_blocks: pl.DT_INT64,
    stride_lo: pl.DT_INT64,
    stride_hi: pl.DT_INT64,
    tile_end: pl.DT_INT64,
    sub_base: pl.DT_INT64,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lane = vf.arange(0, dtype=pl.DT_UINT32)
    mask31 = vf.full(BLOCKS_PER_AIV - 1, preg32, dtype=pl.DT_UINT32)
    in_half = vf.and_(lane, mask31, preg32)
    hi = vf.shift_right(lane, 5, preg32)
    next_tile = vf.muls(hi, BLOCKS_PER_TILE, preg32)
    position_offsets = vf.add(in_half, next_tile, preg32)
    stride_lo_reg = vf.full(stride_lo, preg32, dtype=pl.DT_UINT32)
    even_mask = vf.full(4294967294, preg32, dtype=pl.DT_UINT32)
    even_lane = vf.and_(lane, even_mask, preg32)
    odd_lane = vf.adds(even_lane, 1, preg32)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    no_carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    carry_out = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    stride_hi_reg = vf.full(stride_hi, preg32, dtype=pl.DT_UINT32)
    recip_reg = vf.full(page_param, preg32, dtype=pl.DT_UINT32)
    page = vf.full(0, preg32, dtype=pl.DT_UINT32)
    within = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, tile_end, 2):
        base = chunk * BLOCKS_PER_TILE + sub_base
        logical = vf.adds(position_offsets, base, preg32)
        safe = vf.mins(logical, valid_blocks - 1, preg32)
        cand = vf.gather(cand_tile, safe, preg32)
        cand = vf.maxs(cand, 0, preg32)
        cand_u = vf.bit_cast(cand, dtype=pl.DT_UINT32)
        _split_page_general_vf(
            page, within, cand_u, recip_reg, one32, preg32, blocks_per_page, page_param
        )
        phys_page = vf.gather(table_tile, page, preg32)
        phys_u = vf.bit_cast(phys_page, dtype=pl.DT_UINT32)
        low, high = vf.mull(phys_u, stride_lo_reg, preg32)
        byte_off = vf.muls(within, PACKED_BLOCK_BYTES, preg32)
        carry, low = vf.addc(low, byte_off, no_carry, preg32)
        hi_part = vf.mul(phys_u, stride_hi_reg, preg32)
        carry_out, high = vf.addc(high, hi_part, carry, preg32)
        even_low = vf.gather(low, even_lane)
        odd_low = vf.gather(low, odd_lane)
        even_high = vf.gather(high, even_lane)
        odd_high = vf.gather(high, odd_lane)
        low_gt_m = vf.gt(even_low, odd_low, preg32)
        low_gt = vf.select(one32, zero32, low_gt_m)
        high_gt_m = vf.gt(even_high, odd_high, preg32)
        high_gt = vf.select(one32, zero32, high_gt_m)
        high_eq = vf.eq(even_high, odd_high, preg32)
        swap = vf.select(low_gt, high_gt, high_eq)
        permutation = vf.xor(lane, swap, preg32)
        low = vf.gather(low, permutation)
        high = vf.gather(high, permutation)
        swap4 = vf.muls(swap, 4, preg32)
        swap_mask = vf.update_mask(
            (tile_end - chunk) * BLOCKS_PER_AIV, dtype=pl.DT_UINT32
        )
        vf.store_align(swap_tile + chunk * BLOCKS_PER_AIV, swap4, swap_mask)
        word0, word1 = vf.interleave(low, high)
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV * 2, word0, preg32)
        tail_mask = vf.update_mask(
            (tile_end - chunk - 1) * BLOCKS_PER_AIV * 2, dtype=pl.DT_UINT32
        )
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV * 2 + 64, word1, tail_mask)
    vf.mem_bar()


@pl.vector_function
def _prepare_addresses64_pow2_vf(
    phys_tile,
    swap_tile,
    cand_tile,
    table_tile,
    valid_blocks: pl.DT_INT64,
    stride_lo: pl.DT_INT64,
    stride_hi: pl.DT_INT64,
    tile_end: pl.DT_INT64,
    sub_base: pl.DT_INT64,
    blocks_per_page: pl.DT_INT64,
    page_param: pl.DT_INT64,
):
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lane = vf.arange(0, dtype=pl.DT_UINT32)
    mask31 = vf.full(BLOCKS_PER_AIV - 1, preg32, dtype=pl.DT_UINT32)
    in_half = vf.and_(lane, mask31, preg32)
    hi = vf.shift_right(lane, 5, preg32)
    next_tile = vf.muls(hi, BLOCKS_PER_TILE, preg32)
    position_offsets = vf.add(in_half, next_tile, preg32)
    stride_lo_reg = vf.full(stride_lo, preg32, dtype=pl.DT_UINT32)
    even_mask = vf.full(4294967294, preg32, dtype=pl.DT_UINT32)
    even_lane = vf.and_(lane, even_mask, preg32)
    odd_lane = vf.adds(even_lane, 1, preg32)
    one32 = vf.full(1, preg32, dtype=pl.DT_UINT32)
    zero32 = vf.full(0, preg32, dtype=pl.DT_UINT32)
    no_carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    carry_out = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT32)
    stride_hi_reg = vf.full(stride_hi, preg32, dtype=pl.DT_UINT32)
    recip_reg = vf.full(page_param, preg32, dtype=pl.DT_UINT32)
    page = vf.full(0, preg32, dtype=pl.DT_UINT32)
    within = vf.full(0, preg32, dtype=pl.DT_UINT32)
    for chunk in pl.range(0, tile_end, 2):
        base = chunk * BLOCKS_PER_TILE + sub_base
        logical = vf.adds(position_offsets, base, preg32)
        safe = vf.mins(logical, valid_blocks - 1, preg32)
        cand = vf.gather(cand_tile, safe, preg32)
        cand = vf.maxs(cand, 0, preg32)
        cand_u = vf.bit_cast(cand, dtype=pl.DT_UINT32)
        _split_page_pow2_vf(
            page, within, cand_u, recip_reg, one32, preg32, blocks_per_page, page_param
        )
        phys_page = vf.gather(table_tile, page, preg32)
        phys_u = vf.bit_cast(phys_page, dtype=pl.DT_UINT32)
        low, high = vf.mull(phys_u, stride_lo_reg, preg32)
        byte_off = vf.muls(within, PACKED_BLOCK_BYTES, preg32)
        carry, low = vf.addc(low, byte_off, no_carry, preg32)
        hi_part = vf.mul(phys_u, stride_hi_reg, preg32)
        carry_out, high = vf.addc(high, hi_part, carry, preg32)
        even_low = vf.gather(low, even_lane)
        odd_low = vf.gather(low, odd_lane)
        even_high = vf.gather(high, even_lane)
        odd_high = vf.gather(high, odd_lane)
        low_gt_m = vf.gt(even_low, odd_low, preg32)
        low_gt = vf.select(one32, zero32, low_gt_m)
        high_gt_m = vf.gt(even_high, odd_high, preg32)
        high_gt = vf.select(one32, zero32, high_gt_m)
        high_eq = vf.eq(even_high, odd_high, preg32)
        swap = vf.select(low_gt, high_gt, high_eq)
        permutation = vf.xor(lane, swap, preg32)
        low = vf.gather(low, permutation)
        high = vf.gather(high, permutation)
        swap4 = vf.muls(swap, 4, preg32)
        swap_mask = vf.update_mask(
            (tile_end - chunk) * BLOCKS_PER_AIV, dtype=pl.DT_UINT32
        )
        vf.store_align(swap_tile + chunk * BLOCKS_PER_AIV, swap4, swap_mask)
        word0, word1 = vf.interleave(low, high)
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV * 2, word0, preg32)
        tail_mask = vf.update_mask(
            (tile_end - chunk - 1) * BLOCKS_PER_AIV * 2, dtype=pl.DT_UINT32
        )
        vf.store_align(phys_tile + chunk * BLOCKS_PER_AIV * 2 + 64, word1, tail_mask)
    vf.mem_bar()


@pl.jit(arch="a5", auto_mutex=True, tiling_key=QsliKey)
def qsli_kernel(
    query: pl.Tensor[[pl.DYNAMIC, LOGICAL_D], pl.DT_FP4E2M1],
    query_scale: pl.Tensor[[pl.DYNAMIC, 2, 2], pl.DT_FP8E8M0],
    weights: pl.Tensor[[pl.DYNAMIC, N1], pl.DT_FP32],
    candidates: pl.Tensor[[pl.DYNAMIC, CANDIDATE_CAPACITY], pl.DT_INT32],
    cand_len: pl.Tensor[[pl.DYNAMIC, N2], pl.DT_INT32],
    key_pa: pl.Ptr[pl.DT_UINT8],
    key_tnd: pl.Ptr[pl.DT_UINT8],
    dk_tnd: pl.Ptr[pl.DT_UINT8],
    block_table: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    cu_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    cu_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    seq_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    seq_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    residual: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    idx_offset: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    sched: pl.Tensor[[METADATA_ELEMENTS], pl.DT_INT32],
    stg_k512: pl.Tensor[[pl.DYNAMIC, KEY_BLOCK_BYTES], pl.DT_UINT8],
    stg_k64: pl.Tensor[[pl.DYNAMIC, PACKED_D], pl.DT_UINT8],
    stg_k_fp4: pl.Tensor[[pl.DYNAMIC, LOGICAL_D], pl.DT_FP4E2M1],
    stg_s32: pl.Tensor[[pl.DYNAMIC, SCALE_BLOCK_BYTES], pl.DT_UINT8],
    stg_s_ptr: pl.Ptr[pl.DT_UINT8],
    stg_s_e8: pl.Tensor[[pl.DYNAMIC, 2, 2], pl.DT_FP8E8M0],
    key_ws: pl.Tensor[[pl.DYNAMIC, HALF_TOKENS], pl.DT_UINT16],
    out_idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out_val: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    part_key: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    part_pos: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    rt_batch: pl.DT_INT64,
    rt_key_stride: pl.DT_INT64,
    rt_table_width: pl.DT_INT64,
    rt_total_q: pl.DT_INT64,
    rt_table_cached: pl.DT_INT64,
    rt_tnd_k_stride: pl.DT_INT64,
    rt_tnd_s_stride: pl.DT_INT64,
    rt_tnd_rows: pl.DT_INT64,
    rt_addr32: pl.DT_INT64,
    rt_has_cu_q: pl.DT_INT64,
    rt_has_seq_q: pl.DT_INT64,
    rt_has_seq_k: pl.DT_INT64,
    rt_has_residual: pl.DT_INT64,
    rt_res_eq_used_q: pl.DT_INT64,
    rt_has_idx_off: pl.DT_INT64,
    rt_has_table: pl.DT_INT64,
    rt_k_capacity: pl.DT_INT64,
    rt_cmp_ratio: pl.DT_INT64,
    rt_blocks_per_page: pl.DT_INT64,
    rt_page_param: pl.DT_INT64,
    rt_topk: pl.DT_INT64,
):
    ring = pl.make_tile_group(
        type=pl.TileType(
            shape=[BLOCKS_PER_AIV, PACKED_BLOCK_BYTES]
            if IsPA
            else [HALF_TOKENS, PACKED_D],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_RING_ADDRS,
        mutex_ids=_RING_IDS,
    )
    if IsPA:
        # Each pair aliases one ring slot. _stage_tile_pa holds that slot's
        # MTE2 mutex around all pair loads; the ring's MTE3 stores use the same
        # ID. The individual pair descriptors intentionally carry no auto ID.
        pa_pair = pl.make_tile_group(
            type=pl.TileType(
                shape=[2, PACKED_BLOCK_BYTES],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=_PAIR_ADDRS,
            mutex_ids=None,
            depth=BUF_DEPTH * (BLOCKS_PER_AIV // 2),
        )
    else:
        ring_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCKS_PER_AIV, SCALE_BLOCK_BYTES],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=_TND_SCALE_ADDRS,
            mutex_ids=_SCALE_IDS,
        )
        # Block views share each ring slot's explicit MTE2 lock; the matching
        # whole-ring MTE3 store uses the same physical-slot mutex.
        tnd_k = pl.make_tile_group(
            type=pl.TileType(
                shape=[CANDIDATE_BLOCK_SIZE, PACKED_D],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=_TND_KEY_BLOCK_ADDRS,
            mutex_ids=None,
            depth=BUF_DEPTH * BLOCKS_PER_AIV,
        )
        tnd_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, SCALE_BLOCK_BYTES],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=_TND_SCALE_BLOCK_ADDRS,
            mutex_ids=None,
            depth=BUF_DEPTH * BLOCKS_PER_AIV,
        )
        scale_pitch = pl.make_tile_group(
            type=pl.TileType(
                shape=[CANDIDATE_BLOCK_SIZE, SCALE_BLOCK_BYTES],
                valid_shape=[-1, -1],
                dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=_LAYOUTS[IsPA]["scale_pitch"],
            mutex_ids=[21],
            depth=1,
        )
    qk = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, HALF_TOKENS],
            dtype=pl.DT_BF16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["qk"],
        mutex_ids=None,
        depth=QK_DEPTH,
    )
    kscratch = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, HALF_TOKENS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["kscratch"],
        mutex_ids=[0, 1],
        depth=QK_DEPTH,
    )
    key2d = pl.make_tile_group(
        type=pl.TileType(
            shape=[MAX_TILES * 2, HALF_TOKENS],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=_LAYOUTS[IsPA]["key"],
        mutex_ids=[MX_KEY],
        depth=1,
    )
    keyflat = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, MAX_TOKENS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["key"],
        mutex_ids=[MX_KEY],
        depth=1,
    )
    mkey = pl.make_tile_group(
        type=pl.TileType(
            shape=[
                max(
                    2,
                    min(MERGE_SLOT_CAP, MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16)),
                )
                - 1,
                (TopKCapacity + 15) // 16 * 16,
            ],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=_LAYOUTS[IsPA]["key"],
        mutex_ids=[MX_KEY],
    )
    mpos = pl.make_tile_group(
        type=pl.TileType(
            shape=[
                max(
                    2,
                    min(MERGE_SLOT_CAP, MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16)),
                )
                - 1,
                (TopKCapacity + 15) // 16 * 16,
            ],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=_LAYOUTS[IsPA]["ring"],
        mutex_ids=[_RING_IDS[0]],
    )
    mcarry_key = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, (TopKCapacity + 15) // 16 * 16],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["key"]
        + (
            max(2, min(MERGE_SLOT_CAP, MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16)))
            - 1
        )
        * ((TopKCapacity + 15) // 16 * 16)
        * 2,
        mutex_ids=[MX_KEY],
    )
    mcarry_pos = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, (TopKCapacity + 15) // 16 * 16],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["ring"]
        + (
            max(2, min(MERGE_SLOT_CAP, MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16)))
            - 1
        )
        * ((TopKCapacity + 15) // 16 * 16)
        * 2,
        mutex_ids=[_RING_IDS[0]],
    )
    # The merge reuses both gather slots only after all credits are drained,
    # bar_mte3 completes the publishers and sync_all(MIX) joins every producer.
    # This phase boundary protects aliases spanning the two ring mutex IDs.
    mpos_flat = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, MAX_TOKENS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["ring"],
        mutex_ids=[_RING_IDS[0]],
    )
    cand = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=_LAYOUTS[IsPA]["cand"],
        mutex_ids=[2],
        depth=1,
    )
    cand_pf = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=_LAYOUTS[IsPA]["cand_pf"],
        mutex_ids=[3],
        depth=1,
    )
    phys = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["phys"],
        mutex_ids=[MX_PHYS],
        depth=1,
    )
    phys64 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY // 2],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["phys"],
        mutex_ids=[MX_PHYS],
        depth=1,
    )
    swap = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["swap"],
        mutex_ids=None,
        depth=1,
    )
    if IsPA:
        table = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, PA_TABLE_CACHE_ENTRIES],
                dtype=pl.DT_INT32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=_LAYOUTS[IsPA]["table"],
            mutex_ids=[22],
            depth=1,
        )
    counts = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, CANDIDATE_CAPACITY],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["counts"],
        mutex_ids=None,
        depth=1,
    )
    total = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["total"],
        mutex_ids=[4],
        depth=1,
    )
    validm = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, HALF_TOKENS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["validm"],
        mutex_ids=None,
        depth=1,
    )
    w_f32 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["w_f32"],
        mutex_ids=[5],
        depth=1,
    )
    w_bf16 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["w_bf16"],
        mutex_ids=None,
        depth=1,
    )
    hist_hi = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["hist_hi"],
        mutex_ids=None,
        depth=1,
    )
    hist_lo = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["hist_lo"],
        mutex_ids=None,
        depth=1,
    )
    sqz = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["sqz"],
        mutex_ids=None,
        depth=1,
    )
    idx_hi = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["idx_hi"],
        mutex_ids=None,
        depth=1,
    )
    nk = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["nk"],
        mutex_ids=None,
        depth=1,
    )
    kth = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=_LAYOUTS[IsPA]["kth"],
        mutex_ids=None,
        depth=1,
    )
    pos = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, MAX_TOKENS + 2048],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["pos"],
        mutex_ids=[23],
        depth=1,
    )
    out_i32 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64))],
            dtype=pl.DT_INT32,
            valid_shape=[-1, -1],
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["out_i32"],
        mutex_ids=[6],
        depth=1,
    )
    out_u16 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, (TopKCapacity + 127) // 128 * 128],
            dtype=pl.DT_UINT16,
            valid_shape=[-1, -1],
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=_LAYOUTS[IsPA]["out_i32"]
        + (min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)) * 4 + 511)
        // 512
        * 512,
        mutex_ids=[7],
        depth=1,
    )
    q_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, LOGICAL_D],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0,
        mutex_ids=[8],
        depth=1,
    )
    qs_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, SCALE_K],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
            valid_shape=[-1, -1],
        ),
        addrs=4096,
        mutex_ids=[9],
        depth=1,
    )
    k_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[LOGICAL_D, TILE_N],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=8192,
        mutex_ids=[10, 11],
        depth=BUF_DEPTH,
    )
    ks_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE_N],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
            valid_shape=[-1, -1],
        ),
        addrs=270336,
        mutex_ids=[12, 13],
        depth=BUF_DEPTH,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, LOGICAL_D],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0,
        mutex_ids=[14],
        depth=1,
    )
    sa_l0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, SCALE_K],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0,
        mutex_ids=[15],
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[LOGICAL_D, TILE_N],
            dtype=pl.DT_FP4E2M1,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0,
        mutex_ids=[16, 17],
        depth=BUF_DEPTH,
    )
    sb_l0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE_N],
            dtype=pl.DT_FP8E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            layout=pl.NN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0,
        mutex_ids=[18, 19],
        depth=BUF_DEPTH,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_N],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0,
        mutex_ids=[20],
        depth=1,
    )
    with pl.section_cube():
        worker = pl.get_block_idx()
        batch_count = rt_batch
        total_q = rt_total_q
        k_capacity = rt_k_capacity
        has_cu_q = rt_has_cu_q
        has_seq_q = rt_has_seq_q
        has_seq_k = rt_has_seq_k
        has_residual = rt_has_residual
        res_eq_used_q = rt_res_eq_used_q
        for _slot in pl.range(0, STAGE_DEPTH):
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE2, event_id=EV_RING_FREE + _slot
            )
        pl.set_validshape(q_l1, [TILE_M, LOGICAL_D])
        pl.set_validshape(qs_l1, [TILE_M, SCALE_K])
        pl.set_validshape(k_l1, [LOGICAL_D, TILE_N])
        pl.set_validshape(ks_l1, [SCALE_K, TILE_N])
        pl.set_validshape(l0a, [TILE_M, LOGICAL_D])
        pl.set_validshape(sa_l0, [TILE_M, SCALE_K])
        pl.set_validshape(l0b, [LOGICAL_D, TILE_N])
        pl.set_validshape(sb_l0, [SCALE_K, TILE_N])
        pl.set_validshape(acc, [TILE_M, TILE_N])
        li = worker * METADATA_FIELDS
        li_on = pl.getval(sched, li)
        first_b = pl.getval(sched, li + 1)
        first_g = pl.getval(sched, li + 2)
        first_t = pl.getval(sched, li + 3)
        last_b = pl.getval(sched, li + 4)
        last_g = pl.getval(sched, li + 5)
        last_t = pl.getval(sched, li + 6)
        for b in pl.range(first_b * li_on, pl.min(last_b + 1, batch_count) * li_on):
            if has_cu_q != 0:
                q_begin = pl.getval(cu_q, b)
                q_end = pl.getval(cu_q, b + 1)
            else:
                q_begin = 0
                q_end = total_q
            if has_seq_q != 0:
                used_q = pl.getval(seq_q, b)
            else:
                used_q = q_end - q_begin
            if has_seq_k != 0:
                actual_k = pl.getval(seq_k, b)
            elif IsPA:
                actual_k = k_capacity
            else:
                actual_k = pl.getval(cu_k, b + 1) - pl.getval(cu_k, b)
            if has_residual != 0:
                res = pl.getval(residual, b)
            elif res_eq_used_q != 0:
                res = used_q
            else:
                res = 0
            at_first_b = 1 - pl.min(1, pl.max(b - first_b, first_b - b))
            at_last_b = 1 - pl.min(1, pl.max(b - last_b, last_b - b))
            rows = q_end - q_begin
            m_begin = first_g * at_first_b
            m_end = rows + (last_g + pl.min(1, last_t) - rows) * at_last_b
            for m in pl.range(m_begin, m_end):
                row = q_begin + m
                at_first_m = at_first_b * (
                    1 - pl.min(1, pl.max(m - first_g, first_g - m))
                )
                at_last_m = at_last_b * (1 - pl.min(1, pl.max(m - last_g, last_g - m)))
                raw = (actual_k * rt_cmp_ratio + res - used_q + m + 1) // rt_cmp_ratio
                live = pl.min(1, pl.max(0, used_q - m))
                visible = pl.min(actual_k, pl.max(0, raw)) * live
                blocks = pl.getval(cand_len, row)
                gate = pl.min(1, pl.max(0, visible))
                n_full = blocks * gate + BLOCKS_PER_TILE - 1 >> BLOCKS_SHIFT
                tile_begin = first_t * at_first_m
                tile_end = pl.min(n_full + (last_t - n_full) * at_last_m, n_full)
                if tile_end > tile_begin:
                    pl.load(q_l1.current(), query, [row * N1, 0])
                    pl.load(qs_l1.current(), query_scale, [row * N1, 0, 0])
                    pl.move(l0a.current(), q_l1.current())
                    pl.move(sa_l0.current(), qs_l1.current())
                for t in pl.range(tile_begin, tile_end):
                    buf = t & BUF_MASK
                    stage = t & STAGE_MASK
                    qslot = t & QK_MASK
                    row_base = (worker * STAGE_DEPTH + stage) * TILE_N
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE2, event_id=EV_K_READY + stage
                    )
                    pl.load(k_l1[buf], stg_k_fp4, [row_base, 0], order=[1, 0])
                    pl.load(ks_l1[buf], stg_s_e8, [row_base, 0, 0], order=[1, 0])
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.MTE2, event_id=EV_RING_FREE + stage
                    )
                    pl.move(l0b[buf], k_l1[buf])
                    pl.move(sb_l0[buf], ks_l1[buf])
                    pl.matmul_mx(
                        acc.current(),
                        l0a.current(),
                        l0b[buf],
                        sa_l0.current(),
                        sb_l0[buf],
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=EV_QK_FREE + qslot
                    )
                    pl.move(
                        qk[qslot],
                        acc.current(),
                        [0, 0],
                        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
                    )
                    pl.move(
                        qk[qslot],
                        acc.current(),
                        [0, HALF_TOKENS],
                        acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=EV_QK_READY + qslot
                    )
        for _qslot in pl.range(0, QK_DEPTH):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.FIX, event_id=EV_QK_FREE + _qslot
            )
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        sub_num = pl.get_subblock_num()
        worker = pl.get_block_idx() // sub_num
        batch_count = rt_batch
        key_stride = rt_key_stride
        key_stride_lo = key_stride & 4294967295
        key_stride_hi = key_stride >> 32
        table_width = rt_table_width
        sub = pl.get_subblock_idx()
        sub_base = sub * BLOCKS_PER_AIV
        total_q = rt_total_q
        table_cached = 1 if PAGatherMode != 0 else 0
        tnd_k_stride = rt_tnd_k_stride
        tnd_s_stride = rt_tnd_s_stride
        if TndStrideMode & 1:
            tnd_k_stride = PACKED_D
        if TndStrideMode & 2:
            tnd_s_stride = SCALE_K
        tnd_rows = rt_tnd_rows
        k_capacity = rt_k_capacity
        has_cu_q = rt_has_cu_q
        has_seq_q = rt_has_seq_q
        has_seq_k = rt_has_seq_k
        has_residual = rt_has_residual
        res_eq_used_q = rt_res_eq_used_q
        has_idx_off = rt_has_idx_off
        has_table = rt_has_table
        addr_w32 = 1 if PAGatherMode == 1 else 0
        for _qslot in pl.range(0, QK_DEPTH):
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_QK_FREE + _qslot)
        if not IsPA:
            _zero_u32_vf(swap.current(), CANDIDATE_CAPACITY // 64)
        li = worker * METADATA_FIELDS
        li_on = pl.getval(sched, li)
        first_b = pl.getval(sched, li + 1)
        first_g = pl.getval(sched, li + 2)
        first_t = pl.getval(sched, li + 3)
        last_b = pl.getval(sched, li + 4)
        last_g = pl.getval(sched, li + 5)
        last_t = pl.getval(sched, li + 6)
        ws_base = pl.getval(sched, li + 7)
        first_is_ld = pl.min(1, first_t)
        prefetched = 0
        for b in pl.range(first_b * li_on, pl.min(last_b + 1, batch_count) * li_on):
            if has_cu_q != 0:
                q_begin = pl.getval(cu_q, b)
                q_end = pl.getval(cu_q, b + 1)
            else:
                q_begin = 0
                q_end = total_q
            if not IsPA:
                k_begin = pl.getval(cu_k, b)
                k_len = pl.getval(cu_k, b + 1) - k_begin
            if has_seq_q != 0:
                used_q = pl.getval(seq_q, b)
            else:
                used_q = q_end - q_begin
            if has_seq_k != 0:
                actual_k = pl.getval(seq_k, b)
            elif IsPA:
                actual_k = k_capacity
            else:
                actual_k = pl.getval(cu_k, b + 1) - pl.getval(cu_k, b)
            if has_residual != 0:
                res = pl.getval(residual, b)
            elif res_eq_used_q != 0:
                res = used_q
            else:
                res = 0
            at_first_b = 1 - pl.min(1, pl.max(b - first_b, first_b - b))
            at_last_b = 1 - pl.min(1, pl.max(b - last_b, last_b - b))
            rows = q_end - q_begin
            m_begin = first_g * at_first_b
            m_end = rows + (last_g + pl.min(1, last_t) - rows) * at_last_b
            if IsPA:
                if table_cached != 0:
                    if has_table != 0:
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE2,
                            event_id=SY_V_TO_MTE2,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE2,
                            event_id=SY_V_TO_MTE2,
                        )
                        pl.set_validshape(table, [1, table_width])
                        pl.load(table.current(), block_table, [b, 0])
                    else:
                        pl.set_validshape(table, [1, PA_TABLE_CACHE_ENTRIES])
                        _identity_u32_vf(table.current(), PA_TABLE_CACHE_ENTRIES // 64)
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.S,
                            event_id=SY_V_TO_S,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.S,
                            event_id=SY_V_TO_S,
                        )
            for m in pl.range(m_begin, m_end):
                row = q_begin + m
                at_first_m = at_first_b * (
                    1 - pl.min(1, pl.max(m - first_g, first_g - m))
                )
                at_last_m = at_last_b * (1 - pl.min(1, pl.max(m - last_g, last_g - m)))
                raw = (actual_k * rt_cmp_ratio + res - used_q + m + 1) // rt_cmp_ratio
                live = pl.min(1, pl.max(0, used_q - m))
                visible = pl.min(actual_k, pl.max(0, raw)) * live
                blocks = pl.getval(cand_len, row)
                gate = pl.min(1, pl.max(0, visible))
                n_full = blocks * gate + BLOCKS_PER_TILE - 1 >> BLOCKS_SHIFT
                tile_begin = first_t * at_first_m
                tile_end = pl.min(n_full + (last_t - n_full) * at_last_m, n_full)
                n_tiles = pl.max(0, tile_end - tile_begin)
                row_ld = gate * pl.min(1, tile_begin + pl.max(0, n_full - tile_end))
                ld_slot = ws_base + first_is_ld * (1 - at_first_m)
                if has_idx_off != 0:
                    ioff = pl.getval(idx_offset, row)
                else:
                    ioff = 0
                valid_count = 0
                tokens = n_tiles * TILE_N
                if n_tiles > 0:
                    safe_blocks = pl.max(1, blocks)
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.MTE2,
                        event_id=SY_V_TO_MTE2,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.MTE2,
                        event_id=SY_V_TO_MTE2,
                    )
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.S,
                        wait_pipe=pl.PipeType.MTE2,
                        event_id=SY_S_TO_MTE2,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.S,
                        wait_pipe=pl.PipeType.MTE2,
                        event_id=SY_S_TO_MTE2,
                    )
                    cand_cols = pl.max(
                        BLOCKS_PER_TILE,
                        safe_blocks + BLOCKS_PER_TILE - 1
                        >> BLOCKS_SHIFT
                        << BLOCKS_SHIFT,
                    )
                    pl.set_validshape(cand, [1, cand_cols])
                    pl.load(cand.current(), candidates, [row, 0])
                    pl.load(w_f32.current(), weights, [row, 0])
                    if prefetched == 0:
                        pl.set_validshape(cand_pf, [1, cand_cols])
                        pl.load(cand_pf.current(), candidates, [row, 0])
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.MTE2,
                        wait_pipe=pl.PipeType.V,
                        event_id=SY_MTE2_TO_V,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.MTE2,
                        wait_pipe=pl.PipeType.V,
                        event_id=SY_MTE2_TO_V,
                    )
                    _weights_to_bf16_vf(w_bf16.current(), w_f32.current())
                    _count_visible_vf(
                        counts.current(),
                        total.current(),
                        cand.current(),
                        visible,
                        blocks,
                        safe_blocks,
                    )
                    if IsPA:
                        if prefetched == 0:
                            if table_cached != 0:
                                if rt_blocks_per_page & rt_blocks_per_page - 1 == 0:
                                    if addr_w32 != 0:
                                        _prepare_addresses32_pow2_vf(
                                            phys.current(),
                                            swap.current(),
                                            cand_pf.current(),
                                            table.current(),
                                            safe_blocks,
                                            key_stride_lo,
                                            key_stride_hi,
                                            tile_end,
                                            sub_base,
                                            rt_blocks_per_page,
                                            rt_page_param,
                                        )
                                    else:
                                        _prepare_addresses64_pow2_vf(
                                            phys.current(),
                                            swap.current(),
                                            cand_pf.current(),
                                            table.current(),
                                            safe_blocks,
                                            key_stride_lo,
                                            key_stride_hi,
                                            tile_end,
                                            sub_base,
                                            rt_blocks_per_page,
                                            rt_page_param,
                                        )
                                elif addr_w32 != 0:
                                    _prepare_addresses32_general_vf(
                                        phys.current(),
                                        swap.current(),
                                        cand_pf.current(),
                                        table.current(),
                                        safe_blocks,
                                        key_stride_lo,
                                        key_stride_hi,
                                        tile_end,
                                        sub_base,
                                        rt_blocks_per_page,
                                        rt_page_param,
                                    )
                                else:
                                    _prepare_addresses64_general_vf(
                                        phys.current(),
                                        swap.current(),
                                        cand_pf.current(),
                                        table.current(),
                                        safe_blocks,
                                        key_stride_lo,
                                        key_stride_hi,
                                        tile_end,
                                        sub_base,
                                        rt_blocks_per_page,
                                        rt_page_param,
                                    )
                            else:
                                _zero_u32_vf(swap.current(), CANDIDATE_CAPACITY // 64)
                                _prepare_addresses_scalar(
                                    phys64.current(),
                                    candidates,
                                    block_table,
                                    row,
                                    b,
                                    safe_blocks,
                                    key_stride,
                                    tile_end,
                                    sub_base,
                                    rt_blocks_per_page,
                                    has_table,
                                )
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.S,
                        event_id=SY_V_TO_S,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.S,
                        event_id=SY_V_TO_S,
                    )
                    valid_count = pl.getval(total.current(), 0)
                    # Enqueue independent gather work before waiting for QK.
                    # The extra iterations drain short and partial task ranges.
                    for tick in pl.range(tile_begin, tile_end + GATHER_AHEAD):
                        prev_qslot = tick - GATHER_AHEAD & QK_MASK
                        if tick < tile_end and tick >= tile_begin + prefetched:
                            if IsPA:
                                if addr_w32 != 0:
                                    _stage_tile_pa(
                                        pa_pair,
                                        ring,
                                        phys64.current(),
                                        key_pa,
                                        stg_k512,
                                        stg_s32,
                                        tick,
                                        worker,
                                        sub_base,
                                        PairMin,
                                        _RING_IDS[0],
                                        _RING_IDS[1],
                                        True,
                                    )
                                else:
                                    _stage_tile_pa(
                                        pa_pair,
                                        ring,
                                        phys64.current(),
                                        key_pa,
                                        stg_k512,
                                        stg_s32,
                                        tick,
                                        worker,
                                        sub_base,
                                        PairMin,
                                        _RING_IDS[0],
                                        _RING_IDS[1],
                                    )
                            else:
                                _stage_tile_tnd(
                                    ring,
                                    ring_s,
                                    tnd_k,
                                    tnd_s,
                                    scale_pitch,
                                    cand_pf.current(),
                                    key_tnd,
                                    dk_tnd,
                                    stg_k64,
                                    stg_s32,
                                    stg_s_ptr,
                                    tick,
                                    worker,
                                    sub,
                                    sub_base,
                                    k_begin,
                                    k_len,
                                    safe_blocks,
                                    tnd_k_stride,
                                    tnd_s_stride,
                                    tnd_rows,
                                )
                        if tick >= tile_begin + GATHER_AHEAD:
                            t = tick - GATHER_AHEAD
                            _valid_mask_vf(
                                validm.current(),
                                counts.current(),
                                t * BLOCKS_PER_TILE + sub_base,
                            )
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_QK_READY + prev_qslot
                            )
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.MTE3,
                                wait_pipe=pl.PipeType.V,
                                event_id=SY_MTE3_TO_V,
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.MTE3,
                                wait_pipe=pl.PipeType.V,
                                event_id=SY_MTE3_TO_V,
                            )
                            _reduce_heads_vf(
                                qk[prev_qslot],
                                w_bf16.current(),
                                kscratch[prev_qslot],
                                validm.current(),
                                swap.current(),
                                t,
                                0,
                            )
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_QK_FREE + prev_qslot
                            )
                            pl.system.sync_src(
                                set_pipe=pl.PipeType.V,
                                wait_pipe=pl.PipeType.MTE3,
                                event_id=SY_V_TO_MTE3,
                            )
                            pl.system.sync_dst(
                                set_pipe=pl.PipeType.V,
                                wait_pipe=pl.PipeType.MTE3,
                                event_id=SY_V_TO_MTE3,
                            )
                            pl.store(
                                key_ws,
                                kscratch[prev_qslot],
                                [
                                    worker * MAX_TILES * 2 + (t - tile_begin) * 2 + sub,
                                    0,
                                ],
                            )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.MTE3,
                        event_id=EV_JOIN,
                        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE2,
                        event_id=EV_JOIN,
                        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
                    )
                    if sub == 0 and n_tiles > 0:
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE2,
                            event_id=SY_V_TO_MTE2,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE2,
                            event_id=SY_V_TO_MTE2,
                        )
                        pl.set_validshape(key2d, [n_tiles * 2, HALF_TOKENS])
                        pl.load(key2d.current(), key_ws, [worker * MAX_TILES * 2, 0])
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.MTE2,
                            wait_pipe=pl.PipeType.V,
                            event_id=SY_MTE2_TO_V,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.MTE2,
                            wait_pipe=pl.PipeType.V,
                            event_id=SY_MTE2_TO_V,
                        )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.MTE2,
                        event_id=EV_JOINED,
                        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3,
                        event_id=EV_JOINED,
                        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
                    )
                    prefetched = 0
                    next_m = m + 1
                    if next_m < m_end:
                        if next_m < used_q:
                            next_row = row + 1
                            next_blocks = pl.getval(cand_len, next_row)
                            next_raw = (
                                actual_k * rt_cmp_ratio + res - used_q + next_m + 1
                            ) // rt_cmp_ratio
                            next_live = pl.min(1, pl.max(0, used_q - next_m))
                            next_vis = pl.min(actual_k, pl.max(0, next_raw)) * next_live
                            next_gate = pl.min(1, pl.max(0, next_vis))
                            next_full = (
                                next_blocks * next_gate + BLOCKS_PER_TILE - 1
                                >> BLOCKS_SHIFT
                            )
                            at_last_next = at_last_b * (
                                1 - pl.min(1, pl.max(next_m - last_g, last_g - next_m))
                            )
                            next_end = pl.min(
                                next_full + (last_t - next_full) * at_last_next,
                                next_full,
                            )
                            if next_end > 0:
                                next_safe = pl.max(1, next_blocks)
                                next_cols = pl.max(
                                    BLOCKS_PER_TILE,
                                    next_safe + BLOCKS_PER_TILE - 1
                                    >> BLOCKS_SHIFT
                                    << BLOCKS_SHIFT,
                                )
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.S,
                                    wait_pipe=pl.PipeType.MTE2,
                                    event_id=SY_S_TO_MTE2,
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.S,
                                    wait_pipe=pl.PipeType.MTE2,
                                    event_id=SY_S_TO_MTE2,
                                )
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.V,
                                    wait_pipe=pl.PipeType.MTE2,
                                    event_id=SY_V_TO_MTE2,
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.V,
                                    wait_pipe=pl.PipeType.MTE2,
                                    event_id=SY_V_TO_MTE2,
                                )
                                pl.set_validshape(cand_pf, [1, next_cols])
                                pl.load(cand_pf.current(), candidates, [next_row, 0])
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.MTE2,
                                    wait_pipe=pl.PipeType.V,
                                    event_id=SY_MTE2_TO_V,
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.MTE2,
                                    wait_pipe=pl.PipeType.V,
                                    event_id=SY_MTE2_TO_V,
                                )
                                if IsPA:
                                    if table_cached != 0:
                                        if (
                                            rt_blocks_per_page & rt_blocks_per_page - 1
                                            == 0
                                        ):
                                            if addr_w32 != 0:
                                                _prepare_addresses32_pow2_vf(
                                                    phys.current(),
                                                    swap.current(),
                                                    cand_pf.current(),
                                                    table.current(),
                                                    next_safe,
                                                    key_stride_lo,
                                                    key_stride_hi,
                                                    next_end,
                                                    sub_base,
                                                    rt_blocks_per_page,
                                                    rt_page_param,
                                                )
                                            else:
                                                _prepare_addresses64_pow2_vf(
                                                    phys.current(),
                                                    swap.current(),
                                                    cand_pf.current(),
                                                    table.current(),
                                                    next_safe,
                                                    key_stride_lo,
                                                    key_stride_hi,
                                                    next_end,
                                                    sub_base,
                                                    rt_blocks_per_page,
                                                    rt_page_param,
                                                )
                                        elif addr_w32 != 0:
                                            _prepare_addresses32_general_vf(
                                                phys.current(),
                                                swap.current(),
                                                cand_pf.current(),
                                                table.current(),
                                                next_safe,
                                                key_stride_lo,
                                                key_stride_hi,
                                                next_end,
                                                sub_base,
                                                rt_blocks_per_page,
                                                rt_page_param,
                                            )
                                        else:
                                            _prepare_addresses64_general_vf(
                                                phys.current(),
                                                swap.current(),
                                                cand_pf.current(),
                                                table.current(),
                                                next_safe,
                                                key_stride_lo,
                                                key_stride_hi,
                                                next_end,
                                                sub_base,
                                                rt_blocks_per_page,
                                                rt_page_param,
                                            )
                                    else:
                                        _zero_u32_vf(
                                            swap.current(), CANDIDATE_CAPACITY // 64
                                        )
                                        _prepare_addresses_scalar(
                                            phys64.current(),
                                            candidates,
                                            block_table,
                                            next_row,
                                            b,
                                            next_safe,
                                            key_stride,
                                            next_end,
                                            sub_base,
                                            rt_blocks_per_page,
                                            has_table,
                                        )
                                pl.system.sync_src(
                                    set_pipe=pl.PipeType.V,
                                    wait_pipe=pl.PipeType.S,
                                    event_id=SY_V_TO_S,
                                )
                                pl.system.sync_dst(
                                    set_pipe=pl.PipeType.V,
                                    wait_pipe=pl.PipeType.S,
                                    event_id=SY_V_TO_S,
                                )
                                prefetched = pl.min(STAGE_DEPTH, next_end)
                                for g in pl.range(0, prefetched):
                                    if IsPA:
                                        if addr_w32 != 0:
                                            _stage_tile_pa(
                                                pa_pair,
                                                ring,
                                                phys64.current(),
                                                key_pa,
                                                stg_k512,
                                                stg_s32,
                                                g,
                                                worker,
                                                sub_base,
                                                PairMin,
                                                _RING_IDS[0],
                                                _RING_IDS[1],
                                                True,
                                            )
                                        else:
                                            _stage_tile_pa(
                                                pa_pair,
                                                ring,
                                                phys64.current(),
                                                key_pa,
                                                stg_k512,
                                                stg_s32,
                                                g,
                                                worker,
                                                sub_base,
                                                PairMin,
                                                _RING_IDS[0],
                                                _RING_IDS[1],
                                            )
                                    else:
                                        _stage_tile_tnd(
                                            ring,
                                            ring_s,
                                            tnd_k,
                                            tnd_s,
                                            scale_pitch,
                                            cand_pf.current(),
                                            key_tnd,
                                            dk_tnd,
                                            stg_k64,
                                            stg_s32,
                                            stg_s_ptr,
                                            g,
                                            worker,
                                            sub,
                                            sub_base,
                                            k_begin,
                                            k_len,
                                            next_safe,
                                            tnd_k_stride,
                                            tnd_s_stride,
                                            tnd_rows,
                                        )
                else:
                    prefetched = 0
                # The two AIVs share empty rows in both output modes.
                if sub == 0 or n_tiles == 0:
                    produce = pl.min(1, valid_count) * pl.min(1, n_tiles)
                    if produce != 0:
                        scan = pl.max(tokens, (TopKCapacity + 255) // 256 * 256)
                        _zero_keys_vf(keyflat.current(), tokens // 128, scan // 128)
                        if row_ld == 0 and valid_count <= rt_topk:
                            _collect_valid_vf(
                                pos.current(), keyflat.current(), scan // 128
                            )
                        else:
                            bottom = pl.max(1, scan - rt_topk + 1)
                            hist_loop = scan // 256
                            _hist_high_vf(
                                hist_hi.current(), keyflat.current(), hist_loop
                            )
                            _find_high_and_hist_low_vf(
                                hist_lo.current(),
                                idx_hi.current(),
                                hist_hi.current(),
                                keyflat.current(),
                                sqz.current(),
                                nk.current(),
                                bottom,
                                hist_loop,
                            )
                            _find_kth_vf(
                                kth.current(),
                                hist_lo.current(),
                                idx_hi.current(),
                                sqz.current(),
                                nk.current(),
                            )
                            _collect_gt_eq_vf(
                                pos.current(),
                                keyflat.current(),
                                kth.current(),
                                scan // 128,
                            )
                        kept = pl.min(rt_topk, valid_count)
                        if row_ld != 0:
                            _map_keys_vf(
                                out_u16.current(),
                                pos.current(),
                                keyflat.current(),
                                kept,
                                (TopKCapacity + 127) // 128,
                            )
                            _map_rawpos_vf(
                                pos.current(),
                                kept,
                                tile_begin * TILE_N,
                                (TopKCapacity + 127) // 128,
                            )
                            pl.set_validshape(
                                out_u16, [1, (TopKCapacity + 15) // 16 * 16]
                            )
                            pl.store(part_key, out_u16.current(), [ld_slot, 0])
                            pl.store(
                                part_pos,
                                pos.current()[
                                    0:1, slice(0, (TopKCapacity + 15) // 16 * 16)
                                ],
                                [ld_slot, 0],
                            )
                        else:
                            if WriteValues:
                                _map_values_vf(
                                    out_u16.current(),
                                    pos.current(),
                                    keyflat.current(),
                                    kept,
                                    (TopKCapacity + 127) // 128,
                                )
                            pl.set_validshape(out_u16, [1, rt_topk])
                            if WriteValues:
                                pl.store(out_val, out_u16.current(), [row, 0])
                            for chunk in pl.range(
                                0,
                                (
                                    (rt_topk + 63) // 64 * 64
                                    + min(
                                        IDX_CHUNK,
                                        max(64, (TopKCapacity + 63) // 64 * 64),
                                    )
                                    - 1
                                )
                                // min(
                                    IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)
                                ),
                            ):
                                column = chunk * min(
                                    IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)
                                )
                                _map_positions_vf(
                                    out_i32.current(),
                                    pos.current(),
                                    cand.current(),
                                    kept,
                                    ioff,
                                    tile_begin * TILE_N,
                                    column,
                                    min(
                                        IDX_CHUNK,
                                        max(64, (TopKCapacity + 63) // 64 * 64),
                                    )
                                    // 64,
                                )
                                pl.set_validshape(
                                    out_i32,
                                    [
                                        1,
                                        pl.min(
                                            min(
                                                IDX_CHUNK,
                                                max(64, (TopKCapacity + 63) // 64 * 64),
                                            ),
                                            rt_topk - column,
                                        ),
                                    ],
                                )
                                pl.store(out_idx, out_i32.current(), [row, column])
                    elif row_ld == 0 and (n_tiles > 0 or (m - m_begin) % 2 == sub):
                        _fill_invalid_vf(
                            out_u16.current(),
                            out_i32.current(),
                            (TopKCapacity + 127) // 128,
                            min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64))
                            // 64,
                        )
                        pl.set_validshape(out_u16, [1, rt_topk])
                        if WriteValues:
                            pl.store(out_val, out_u16.current(), [row, 0])
                        for chunk in pl.range(
                            0,
                            (
                                (rt_topk + 63) // 64 * 64
                                + min(
                                    IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)
                                )
                                - 1
                            )
                            // min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)),
                        ):
                            column = chunk * min(
                                IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)
                            )
                            pl.set_validshape(
                                out_i32,
                                [
                                    1,
                                    pl.min(
                                        min(
                                            IDX_CHUNK,
                                            max(64, (TopKCapacity + 63) // 64 * 64),
                                        ),
                                        rt_topk - column,
                                    ),
                                ],
                            )
                            pl.store(out_idx, out_i32.current(), [row, column])
        for _slot in pl.range(0, STAGE_DEPTH):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE2, event_id=EV_RING_FREE + _slot
            )
        pl.system.bar_mte3()
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        ld = (LD_RECORD_START + pl.get_block_idx()) * METADATA_FIELDS
        if pl.getval(sched, ld) != 0:
            mb = pl.getval(sched, ld + 1)
            mg = pl.getval(sched, ld + 2)
            mbase = pl.getval(sched, ld + 3)
            parts = pl.getval(sched, ld + 4)
            if has_cu_q != 0:
                mrow = pl.getval(cu_q, mb) + mg
            else:
                mrow = mg
            if has_idx_off != 0:
                mioff = pl.getval(idx_offset, mrow)
            else:
                mioff = 0
            mblocks = pl.max(1, pl.getval(cand_len, mrow))
            mcand_cols = pl.max(
                BLOCKS_PER_TILE,
                mblocks + BLOCKS_PER_TILE - 1 >> BLOCKS_SHIFT << BLOCKS_SHIFT,
            )
            carry_slot = MERGE_SLOTS + pl.get_block_idx()
            for first in pl.range(
                0,
                parts,
                max(
                    2,
                    min(MERGE_SLOT_CAP, MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16)),
                )
                - 1,
            ):
                count = pl.min(
                    parts - first,
                    max(
                        2,
                        min(
                            MERGE_SLOT_CAP,
                            MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16),
                        ),
                    )
                    - 1,
                )
                carry = pl.min(1, first)
                more = pl.min(1, parts - first - count)
                held = (
                    count
                    + (
                        max(
                            2,
                            min(
                                MERGE_SLOT_CAP,
                                MAX_TOKENS // ((TopKCapacity + 15) // 16 * 16),
                            ),
                        )
                        - 1
                        + 1
                        - count
                    )
                    * carry
                ) * ((TopKCapacity + 15) // 16 * 16)
                mscan = pl.max(
                    (held + 255) // 256 * 256, (TopKCapacity + 255) // 256 * 256
                )
                _zero_keys_vf(keyflat.current(), 0, mscan // 128)
                pl.system.sync_src(
                    set_pipe=pl.PipeType.V,
                    wait_pipe=pl.PipeType.MTE2,
                    event_id=SY_V_TO_MTE2,
                )
                pl.system.sync_dst(
                    set_pipe=pl.PipeType.V,
                    wait_pipe=pl.PipeType.MTE2,
                    event_id=SY_V_TO_MTE2,
                )
                pl.set_validshape(mkey, [count, (TopKCapacity + 15) // 16 * 16])
                pl.set_validshape(mpos, [count, (TopKCapacity + 15) // 16 * 16])
                pl.load(mkey.current(), part_key, [mbase + first, 0])
                pl.load(mpos.current(), part_pos, [mbase + first, 0])
                if carry != 0:
                    pl.load(mcarry_key.current(), part_key, [carry_slot, 0])
                    pl.load(mcarry_pos.current(), part_pos, [carry_slot, 0])
                pl.set_validshape(cand, [1, mcand_cols])
                pl.load(cand.current(), candidates, [mrow, 0])
                pl.system.sync_src(
                    set_pipe=pl.PipeType.MTE2,
                    wait_pipe=pl.PipeType.V,
                    event_id=SY_MTE2_TO_V,
                )
                pl.system.sync_dst(
                    set_pipe=pl.PipeType.MTE2,
                    wait_pipe=pl.PipeType.V,
                    event_id=SY_MTE2_TO_V,
                )
                _count_positive_vf(total.current(), keyflat.current(), mscan // 128)
                pl.system.sync_src(
                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=SY_V_TO_S
                )
                pl.system.sync_dst(
                    set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=SY_V_TO_S
                )
                mvalid = pl.getval(total.current(), 0)
                if mvalid > 0:
                    mbottom = pl.max(1, mscan - rt_topk + 1)
                    mhist = mscan // 256
                    _hist_high_vf(hist_hi.current(), keyflat.current(), mhist)
                    _find_high_and_hist_low_vf(
                        hist_lo.current(),
                        idx_hi.current(),
                        hist_hi.current(),
                        keyflat.current(),
                        sqz.current(),
                        nk.current(),
                        mbottom,
                        mhist,
                    )
                    _find_kth_vf(
                        kth.current(),
                        hist_lo.current(),
                        idx_hi.current(),
                        sqz.current(),
                        nk.current(),
                    )
                    _collect_gt_eq_vf(
                        pos.current(), keyflat.current(), kth.current(), mscan // 128
                    )
                    mkept = pl.min(rt_topk, mvalid)
                    _map_keys_vf(
                        out_u16.current(),
                        pos.current(),
                        keyflat.current(),
                        mkept,
                        (TopKCapacity + 127) // 128,
                    )
                    _resolve_merge_pos_vf(
                        pos.current(),
                        mpos_flat.current(),
                        mkept,
                        (TopKCapacity + 127) // 128,
                    )
                    if more != 0:
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE3,
                            event_id=SY_V_TO_MTE3,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE3,
                            event_id=SY_V_TO_MTE3,
                        )
                        pl.set_validshape(out_u16, [1, (TopKCapacity + 15) // 16 * 16])
                        pl.store(part_key, out_u16.current(), [carry_slot, 0])
                        pl.store(
                            part_pos,
                            pos.current()[
                                0:1, slice(0, (TopKCapacity + 15) // 16 * 16)
                            ],
                            [carry_slot, 0],
                        )
                    if WriteValues:
                        _keys_to_values_vf(
                            out_u16.current(), mkept, (TopKCapacity + 127) // 128
                        )
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.MTE3,
                        event_id=SY_V_TO_MTE3,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.V,
                        wait_pipe=pl.PipeType.MTE3,
                        event_id=SY_V_TO_MTE3,
                    )
                    pl.set_validshape(out_u16, [1, rt_topk])
                    if WriteValues:
                        pl.store(out_val, out_u16.current(), [mrow, 0])
                    for chunk in pl.range(
                        0,
                        (
                            (rt_topk + 63) // 64 * 64
                            + min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64))
                            - 1
                        )
                        // min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)),
                    ):
                        column = chunk * min(
                            IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64)
                        )
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.MTE3,
                            wait_pipe=pl.PipeType.V,
                            event_id=SY_MTE3_TO_V,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.MTE3,
                            wait_pipe=pl.PipeType.V,
                            event_id=SY_MTE3_TO_V,
                        )
                        _map_positions_vf(
                            out_i32.current(),
                            pos.current(),
                            cand.current(),
                            mkept,
                            mioff,
                            0,
                            column,
                            min(IDX_CHUNK, max(64, (TopKCapacity + 63) // 64 * 64))
                            // 64,
                        )
                        pl.system.sync_src(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE3,
                            event_id=SY_V_TO_MTE3,
                        )
                        pl.system.sync_dst(
                            set_pipe=pl.PipeType.V,
                            wait_pipe=pl.PipeType.MTE3,
                            event_id=SY_V_TO_MTE3,
                        )
                        pl.set_validshape(
                            out_i32,
                            [
                                1,
                                pl.min(
                                    min(
                                        IDX_CHUNK,
                                        max(64, (TopKCapacity + 63) // 64 * 64),
                                    ),
                                    rt_topk - column,
                                ),
                            ],
                        )
                        pl.store(out_idx, out_i32.current(), [mrow, column])
                    pl.system.sync_src(
                        set_pipe=pl.PipeType.MTE3,
                        wait_pipe=pl.PipeType.V,
                        event_id=SY_MTE3_TO_V,
                    )
                    pl.system.sync_dst(
                        set_pipe=pl.PipeType.MTE3,
                        wait_pipe=pl.PipeType.V,
                        event_id=SY_MTE3_TO_V,
                    )


def _workspace(num_workers, topk, device):
    rows = num_workers * STAGE_DEPTH * TILE_N
    slots = (topk + 15) // 16 * 16
    return dict(
        stg_k=torch.empty(rows * PACKED_D, dtype=torch.uint8, device=device),
        stg_s=torch.empty(rows * SCALE_K, dtype=torch.uint8, device=device),
        key_ws=torch.empty(
            num_workers * MAX_TILES * 2, HALF_TOKENS, dtype=torch.uint16, device=device
        ),
        part_key=torch.empty(WORKSPACE_SLOTS, slots, dtype=torch.uint16, device=device),
        part_pos=torch.empty(WORKSPACE_SLOTS, slots, dtype=torch.uint16, device=device),
    )


_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)


def _validate_query_auxiliary(w, descale_q, query_rows):
    if tuple(w.shape) != (query_rows, N1) or w.dtype != torch.float32:
        raise ValueError("W must be FP32 with shape (T1,32)")
    if w.device.type != "npu":
        raise ValueError("W must be an NPU tensor")
    expected_q_scale = (query_rows, N1, LOGICAL_D // 64, 2)
    if descale_q is None or tuple(descale_q.shape) != expected_q_scale:
        raise ValueError("Descale_q is required with shape (T1,32,D/64,2)")
    if descale_q.dtype not in _SCALE_DTYPES or descale_q.device.type != "npu":
        raise TypeError("Descale_q must be an NPU E8M0/uint8 tensor")


def _validate_compute_attributes(
    candidate_block_size, topk, mask_mode, cmp_ratio, max_seqlen_q
):
    if int(candidate_block_size) != CANDIDATE_BLOCK_SIZE:
        raise ValueError("Candidate_block_size must be 8")
    if int(topk) < 1 or int(topk) > MAX_TOPK:
        raise ValueError(f"Topk must be in [1,{MAX_TOPK}]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("Mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > MAX_CMP_RATIO:
        raise ValueError(f"Cmp_ratio must be in [1,{MAX_CMP_RATIO}]")
    if int(max_seqlen_q) < -1:
        raise ValueError("Max_seqlen_q must be -1 or nonnegative")


def _validate_candidate_inputs(
    candidate_block_indices, candidate_block_length, query_rows
):
    if (
        candidate_block_indices.ndim != 3
        or tuple(candidate_block_indices.shape[:2]) != (query_rows, N2)
        or int(candidate_block_indices.shape[2]) != CANDIDATE_CAPACITY
    ):
        raise ValueError("Candidate_block_indices must have shape (T1,1,2048)")
    if (
        candidate_block_indices.dtype != torch.int32
        or candidate_block_indices.device.type != "npu"
    ):
        raise TypeError("Candidate_block_indices must be an NPU int32 tensor")
    if (
        candidate_block_length.dtype != torch.int32
        or candidate_block_length.device.type != "npu"
        or int(candidate_block_length.numel()) != query_rows * N2
    ):
        raise ValueError("Candidate_block_length must be NPU int32 with T1*N2 entries")


def _validate_pa_sequences(
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    query_rows,
    physical_k_blocks,
    pa_block_size,
    mask_mode,
    cmp_ratio,
):
    if cu_seqlens_q is None:
        batch = 1
    else:
        if (
            cu_seqlens_q.dtype != torch.int32
            or cu_seqlens_q.device.type != "npu"
            or cu_seqlens_q.ndim != 1
            or (int(cu_seqlens_q.numel()) < 2)
        ):
            raise ValueError("Cu_seqlens_q must be NPU int32 with shape (B+1,)")
        batch = int(cu_seqlens_q.numel()) - 1
    if seqused_q is not None and (
        tuple(seqused_q.shape) != (batch,)
        or seqused_q.dtype != torch.int32
        or seqused_q.device.type != "npu"
    ):
        raise ValueError("Seqused_q must be NPU int32 with shape (B,)")
    if block_table is None:
        if batch != 1:
            raise ValueError("Block_table is required when batch is greater than one")
        logical_k_capacity = physical_k_blocks * pa_block_size
    else:
        if (
            block_table.ndim != 2
            or int(block_table.shape[0]) != batch
            or block_table.dtype != torch.int32
            or (block_table.device.type != "npu")
        ):
            raise ValueError("Block_table must be NPU int32 with shape (B,max_blocks)")
        logical_k_capacity = int(block_table.shape[1]) * pa_block_size
    if seqused_k is not None and (
        tuple(seqused_k.shape) != (batch,)
        or seqused_k.dtype != torch.int32
        or seqused_k.device.type != "npu"
    ):
        raise ValueError("Seqused_k must be NPU int32 with shape (B,)")
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None or (
            tuple(cmp_residual_k.shape) != (batch,)
            or cmp_residual_k.dtype != torch.int32
            or cmp_residual_k.device.type != "npu"
        ):
            raise ValueError(
                "Cmp_residual_k must be NPU int32 with shape (B,) for causal compression"
            )
    elif cmp_residual_k is not None:
        raise ValueError(
            "Cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if output_idx_offset is not None and (
        tuple(output_idx_offset.shape) != (query_rows, N2)
        or output_idx_offset.dtype != torch.int32
        or output_idx_offset.device.type != "npu"
    ):
        raise ValueError("Output_idx_offset must be NPU int32 with shape (T1,1)")
    return (batch, logical_k_capacity)


def _validate_tnd_sequence(tensor, name, device, count=None):
    if tensor is None or tensor.dtype != torch.int32 or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if tensor.device != device or device.type != "npu" or (not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous on the input NPU device")
    if count is not None and tensor.numel() != count:
        raise ValueError(f"{name} must have {count} entries")


def _validate_tnd_geometry(
    lengths, cu_q, cu_k, used_q, used_k, residual, *, topk, mask_mode, cmp_ratio
):
    if (
        lengths.ndim != 2
        or lengths.shape[1] != 1
        or lengths.shape[0] <= 0
        or (lengths.dtype != torch.int32)
        or (lengths.device.type != "npu")
        or (not lengths.is_contiguous())
    ):
        raise ValueError(
            "Candidate_block_length must be contiguous NPU int32 (T1,1), T1>0"
        )
    device = lengths.device
    _validate_tnd_sequence(cu_q, "cu_seqlens_q", device)
    if cu_q.numel() < 2:
        raise ValueError("Cu_seqlens_q must contain at least two boundaries")
    batch = cu_q.numel() - 1
    _validate_tnd_sequence(cu_k, "cu_seqlens_k", device, batch + 1)
    for name, tensor in (("seqused_q", used_q), ("seqused_k", used_k)):
        if tensor is not None:
            _validate_tnd_sequence(tensor, name, device, batch)
    if (
        not 1 <= int(topk) <= MAX_TOPK
        or int(mask_mode) not in (0, 3)
        or (not 1 <= int(cmp_ratio) <= MAX_CMP_RATIO)
    ):
        raise ValueError("Invalid topk, mask_mode or cmp_ratio")
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        _validate_tnd_sequence(residual, "cmp_residual_k", device, batch)
    elif residual is not None:
        raise ValueError("Cmp_residual_k requires mask_mode=3 and cmp_ratio!=1")
    return batch


def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    descale_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Select unordered token TopK from packed MXFP4 candidate blocks.

    The caller supplies the current LI/LD metadata on the input device.
    Runtime sizes, page geometry and compression ratio remain scalar arguments.
    PA selects table-cache/address-width classes, and TopK selects a
    power-of-two scratch capacity.
    Invalid outputs are -1/-inf. Scores are omitted when return_value=False.
    """
    if metadata is None:
        raise ValueError(
            "metadata must contain the current LI/LD schedule (int32[1024])"
        )
    if int(quant_mode) != 1:
        raise ValueError("Quant_sparse_lightning_indexer supports quant_mode=1 only")
    if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
        raise ValueError("Current QSLI layouts must be TND Q with PA_BBND or TND K")
    is_pa = layout_k == "PA_BBND"
    if is_pa and descale_k is not None:
        raise ValueError(
            "Descale_k is only supported with TND K; PA_BBND stores scales in packed k"
        )
    if not is_pa and block_table is not None:
        raise ValueError("Block_table must be None for TND K")
    if q.device.type != "npu" or k.device.type != "npu":
        raise ValueError("Quant_sparse_lightning_indexer requires NPU q/k tensors")
    if q.dtype != torch.uint8 or k.dtype != torch.uint8:
        raise TypeError("Q/k storage must be uint8 packed MXFP4")
    if q.ndim != 3 or tuple(q.shape[1:]) != (N1, PACKED_D):
        raise ValueError("Q must have shape (T1,32,64)")
    device = q.device
    total_q = int(q.shape[0])
    for name, tensor in (
        ("k", k),
        ("w", w),
        ("descale_q", descale_q),
        ("candidate_block_indices", candidate_block_indices),
        ("candidate_block_length", candidate_block_length),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("seqused_q", seqused_q),
        ("seqused_k", seqused_k),
        ("cmp_residual_k", cmp_residual_k),
        ("block_table", block_table),
        ("output_idx_offset", output_idx_offset),
        ("metadata", metadata),
        ("descale_k", descale_k),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"{name} must share q's device")
        if (
            tensor is not None
            and name not in ("k", "descale_k")
            and not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous")
    if is_pa:
        if k.ndim != 3 or int(k.shape[2]) != PACKED_BLOCK_BYTES:
            raise ValueError(
                "K must have shape (block_num,block_size/8,544), each block K[512] followed by scale[32]"
            )
        blocks_per_page = int(k.shape[1])
        page_size = blocks_per_page * CANDIDATE_BLOCK_SIZE
        if (
            page_size <= 0
            or page_size > MAX_PA_BLOCK_SIZE
            or page_size % PA_BLOCK_ALIGNMENT != 0
        ):
            raise ValueError(
                f"PA block_size must be a multiple of {PA_BLOCK_ALIGNMENT} in (0, {MAX_PA_BLOCK_SIZE}]"
            )
        if total_q <= 0 or int(k.shape[0]) <= 0:
            raise ValueError("T1 and the physical K block count must be positive")
    else:
        if (
            total_q <= 0
            or k.ndim != 3
            or tuple(k.shape[1:]) != (N2, PACKED_D)
            or (int(k.shape[0]) <= 0)
        ):
            raise ValueError("TND k must have shape (T2,1,64), T1/T2>0")
        if descale_k is None or tuple(descale_k.shape) != (int(k.shape[0]), N2, 2, 2):
            raise ValueError("Descale_k is required with shape (T2,1,2,2)")
        if descale_k.dtype not in _SCALE_DTYPES:
            raise TypeError("Descale_k must contain E8M0 or uint8 bits")
        for _name, _tensor in (
            ("k", k),
            ("w", w),
            ("descale_q", descale_q),
            ("descale_k", descale_k),
            ("candidate_block_indices", candidate_block_indices),
            ("candidate_block_length", candidate_block_length),
        ):
            if _tensor.device != device:
                raise ValueError(f"{_name} must share q's device")
    _pa_axis0_stride(k, "k")
    for _name, _tensor in (
        ("q", q),
        ("w", w),
        ("descale_q", descale_q),
        ("candidate_block_indices", candidate_block_indices),
        ("candidate_block_length", candidate_block_length),
        ("block_table", block_table),
    ):
        if _tensor is not None and (not _tensor.is_contiguous()):
            raise ValueError(f"{_name} must be contiguous")
    _validate_query_auxiliary(w, descale_q, total_q)
    _validate_compute_attributes(
        candidate_block_size, topk, mask_mode, cmp_ratio, max_seqlen_q
    )
    _validate_candidate_inputs(candidate_block_indices, candidate_block_length, total_q)
    if is_pa:
        batch, logical_k_capacity = _validate_pa_sequences(
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            output_idx_offset,
            total_q,
            int(k.shape[0]),
            page_size,
            mask_mode,
            cmp_ratio,
        )
    else:
        batch = _validate_tnd_geometry(
            candidate_block_length,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            topk=topk,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )
        if output_idx_offset is not None and (
            tuple(output_idx_offset.shape) != (total_q, N2)
            or output_idx_offset.dtype != torch.int32
            or output_idx_offset.device != device
            or (not output_idx_offset.is_contiguous())
        ):
            raise ValueError(
                "Output_idx_offset must be contiguous int32 (T1,1) on q's device"
            )
        logical_k_capacity = int(k.shape[0])
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != METADATA_ELEMENTS
        or (not metadata.is_contiguous())
    ):
        raise ValueError(
            "Metadata must be contiguous int32 [1024] in LI/LD boundary format"
        )
    if metadata.device != device:
        raise ValueError("Metadata must be on the same device as q")
    stand_in = candidate_block_length.reshape(-1)
    has_cu_q = cu_seqlens_q is not None
    has_seq_q = seqused_q is not None
    has_seq_k = seqused_k is not None
    has_idx_off = output_idx_offset is not None
    query = q.reshape(total_q * N1, PACKED_D)
    query_scale = descale_q.reshape(total_q * N1, 2, 2)
    weights = w
    cand = candidate_block_indices.reshape(total_q, CANDIDATE_CAPACITY)
    cand_len = candidate_block_length.reshape(total_q, N2)
    if is_pa:
        key_stride = _pa_axis0_stride(k, "k")
        span = (int(k.shape[0]) - 1) * key_stride + blocks_per_page * PACKED_BLOCK_BYTES
        addr32 = span < MAX_ADDRESS32_BYTES
        key_flat = k.as_strided((span,), (1,))
        key_tnd = None
        dk_tnd = None
        tnd_k_stride = PACKED_D
        s_stride = SCALE_K
        k_rows = 1
        table = block_table
        has_table = table is not None
        table_width = int(table.shape[1]) if has_table else int(k.shape[0])
        table_cached = table_width <= PA_TABLE_CACHE_ENTRIES
        cu_k_dev = None
    else:
        blocks_per_page = 1
        key_stride = 0
        key_flat = None
        k_rows = int(k.shape[0])
        tnd_k_stride = _pa_axis0_stride(k, "k")
        s_stride = _pa_axis0_stride(descale_k, "descale_k")
        key_tnd = k.as_strided(((k_rows - 1) * tnd_k_stride + PACKED_D,), (1,))
        dk_span = (int(descale_k.shape[0]) - 1) * s_stride + SCALE_K
        dk_tnd = descale_k.view(torch.uint8).as_strided((dk_span,), (1,))
        table = None
        cu_k_dev = cu_seqlens_k
    res_eq_used_q = mask_mode == 0
    if res_eq_used_q:
        cmp_ratio = 1
    has_residual = not res_eq_used_q and cmp_residual_k is not None
    idx_off = output_idx_offset.reshape(-1) if has_idx_off else stand_in
    num_workers = min(32, get_platform_info().cube_core_num)
    topk_capacity = max(16, 1 << (int(topk) - 1).bit_length())
    ws = _workspace(num_workers, topk_capacity, device)
    stg_k, stg_s, key_ws = (ws["stg_k"], ws["stg_s"], ws["key_ws"])
    sched = metadata
    tiling_args = (
        batch,
        key_stride,
        table_width if is_pa else 0,
        total_q,
        int(table_cached) if is_pa else 1,
        tnd_k_stride,
        s_stride,
        k_rows,
        int(addr32) if is_pa else 0,
        int(has_cu_q),
        int(has_seq_q),
        int(has_seq_k),
        int(has_residual),
        int(res_eq_used_q),
        int(has_idx_off),
        int(has_table) if is_pa else 0,
        int(logical_k_capacity),
        int(cmp_ratio),
        int(blocks_per_page),
        PAGE_PARAMS[blocks_per_page],
        int(topk),
    )
    out_idx = torch.empty((total_q, topk), dtype=torch.int32, device=device)
    out_val = torch.empty((total_q, topk), dtype=torch.bfloat16, device=device)
    pair_min = 0 if _batch_consistency_enabled() else PACKED_BLOCK_BYTES
    key = dict(
        WriteValues=int(return_value),
        IsPA=int(is_pa),
        TopKCapacity=topk_capacity,
        PairMin=pair_min if is_pa else 0,
        PAGatherMode=(1 if addr32 else 2) if is_pa and table_cached else 0,
        TndStrideMode=(int(tnd_k_stride == PACKED_D) | (int(s_stride == SCALE_K) << 1))
        if not is_pa
        else 0,
    )
    qsli_kernel[None, num_workers, key](
        query,
        query_scale,
        weights,
        cand,
        cand_len,
        key_flat,
        key_tnd,
        dk_tnd,
        table,
        cu_seqlens_q if has_cu_q else stand_in,
        cu_k_dev,
        seqused_q if has_seq_q else stand_in,
        seqused_k if has_seq_k else stand_in,
        cmp_residual_k if has_residual else stand_in,
        idx_off,
        sched,
        stg_k.view(-1, KEY_BLOCK_BYTES),
        stg_k.view(-1, PACKED_D),
        stg_k.view(-1, PACKED_D),
        stg_s.view(-1, SCALE_BLOCK_BYTES),
        stg_s.view(-1) if not is_pa else None,
        stg_s.view(-1, 2, 2),
        key_ws,
        out_idx,
        out_val.view(torch.uint16),
        ws["part_key"],
        ws["part_pos"],
        *tiling_args,
    )
    indices = out_idx.reshape(total_q, 1, topk)
    values = out_val
    values = (
        values.reshape(total_q, 1, topk) if return_value else values.reshape(-1)[:0]
    )
    return (indices, values)
