# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 GQA decode with explicit resources, guarded prefetch and optional split-state merge.

Contract: q/O[B=16,HQ=64,1,D=128], K/V[B=16,HKV=8,S=4096,D=128], grouped by G=HQ/HKV=8. Inputs are contiguous finite
FP16; output is FP32. One new query sees all historical keys without a causal mask or dropout. Scores, running
maxima/sums and output accumulation use FP32; unnormalized probabilities narrow to FP16 for Cube. Scale is 1/sqrt(128).

Route A assigns 128 whole (batch,KV-head) items to 28 contiguous execution-group ranges and writes output directly.
Route B partitions 2048 KV-block units into 28 contiguous ranges; an item cut across a boundary stores front/back
unnormalized O, m and l in caller-provided ws[128,2,8,256] FP32 (2 MiB). After the pipeline drains, Vector bar_all
completes local writes and both Cube and Vector enter sync_all(MIX). The group owning an item's back portion reads both
records and merges with m=max(m0,m1), a_i=exp(scale*(m_i-m)), O=(a0*O0+a1*O1)/(a0*l0+a1*l1). Complete items bypass the
workspace. These routes demonstrate different ownership/synchronization mechanisms, without a current
relative-performance conclusion.

Both routes use qk -> softmax -> pv -> update over 256-key blocks, with a continuous tick across items. Standard score
rows hold query heads and lanes hold keys. Physical M=16 gives two eight-row AIV halves; only AIV0 owns the eight
logical query heads. Q loads use G valid rows, then restore the physical declaration before GEMM. AIV1 participates in
synchronization but does not insert P or store output; unused rows remain isolated from valid rows. Two [9,128] NZ tiles
pack P with eight valid rows and a padded physical pitch.

Q has two slots and prefetches the next item at loop entry. K/V have two slots indexed by tick%2, including across
split-item boundaries; their stages prefetch the next block or the next item's first block. Next-item and has_next
guards remain independent memory-safety requirements. Separate m/l/alpha rings have ST_SLOTS=PRELOAD+1=3; P has
P_SLOTS=2. All four entries (both routes, preload=2/0) retain the same relevant physical allocation and independent
kernel names.

Explicit byte ranges at PRELOAD=2 (ends exclusive):

| Space | Groups and ranges | Bytes/slot x depth |
| --- | --- | --- |
| Mat | q_g [0,0x2000) | 4096 x 2 |
| Mat | k_g [0x2000,0x22000), v_g [0x22000,0x42000) | 65536 x 2 each |
| Mat | shared_p [0x42000,0x46000) | 8192 x 2 |
| Left | l0a1 [0,0x1000), l0a2 [0x1000,0x2000) | 2048 x 2 each |
| Right | l0b1 [0,0x8000), l0b2 [0x8000,0xc000) | 32768 x 1 / 16384 x 1 |
| Acc | acc1_g [0,0x8000), acc2_g [0x8000,0xc000) | 16384 x 2 / 8192 x 2 |
| Vec | shared_s [0,0x4000), shared_pv [0x4000,0x6000) | 8192 x 2 / 4096 x 2 |
| Vec | nzlo_g [0x6000,0x7200), nzhi_g [0x7200,0x8400) | 2304 x 2 each |
| Vec | st_a [0x8400,0x9c00), st_l [0x9c00,0xb400), st_m [0xb400,0xcc00) | 2048 x 3 each |
| Vec | o_g [0xcc00,0xdc00) | 4096 x 1 |
| Vec | pw_g [0xdc00,0x11c00), Route B only | 8192 x 2 |

Mat uses 280 KiB, Left 8 KiB, Right/Acc 48 KiB each, and Vec 55 KiB (A) or 71 KiB (B). All groups use automatic
intra-core mutex IDs. shared_s uses forward [0,1], reverse [2,3]; shared_p uses forward [4,5], reverse [6,7]; shared_pv
uses forward [8,9], reverse [10,11]. Score/product edges are FIX -> V with V -> FIX release. P is MTE3 -> MTE1 with MTE1
-> MTE3 release. VF leaves the shared score/product inputs unchanged. Main stage delays are 0/1/2/3: the three
cross-core edges need two slots each and same-core state needs three. Changing geometry or depth requires replanning
these ranges and event arrays.

Reference math and the input-dependent error budget reuse flash_attention.reference per GQA group, with the assumptions
stated in its error_terms. Tests cover both routes, normal/extremal/growing and group-distinct inputs, one-hot
selection, partial launches, poisoned workspace, zero-tolerance preload/repeat consistency, bounded local storage,
per-route depths, prefetch guards and incorrect mathematical controls. Numerical consistency does not prove every
lifetime or the safety of an unused prefetch. No current performance claim is made.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

B, HQ, HKV, D = 16, 64, 8, 128  # q/O [B,HQ,1,D]; K/V [B,HKV,S,D]
G = HQ // HKV  # 8：每 KV 头共享的 q 头数（组比）
S = 4096  # KV 序列长
BKV = 256  # 一个 KV 块的 key 数（[256,128] FP16 = 64KB 整块连续 DMA）
NB = S // BKV  # 16：每 item 的 KV 块数
ITEMS = B * HKV  # 128 个 (b, h_kv) 对
UNITS = ITEMS * NB  # 2048 个 KV 块单元（route B 的切分粒度）
NUM_CORES = 28
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 方括号数 = 执行组数（1c+2v）
MP = 16  # 物理 m（分形对齐）；有效行 G=8
SK = 64  # 两个 gemm 共用的 splitk 步长
VL = 64  # FP32 寄存器 lane 数
NCH = BKV // VL  # 4：一行分数的寄存器块数
DCH = D // VL  # 2：一行 O 的寄存器块数
NZ_PITCH = G + 1  # 9：UB NZ 物理列高 = rows+1，奇数错开 bank
HALF_N = 128  # 一条 FP16 寄存器的列数；每 128 列一个 NZ tile
PRELOAD = 2  # Main schedule; explicit addresses and event arrays use this depth.
ST_SLOTS = PRELOAD + 1  # 同核 softmax→update：拍距 P，槽数 P+1
P_SLOTS = max(PRELOAD - 1, 0) + 1  # 跨核 v→c 返程：拍距 P-1，槽数 = 拍距+1
Q_SLOTS = 2  # Q 双槽：item 顶部预取下一 item
KV_SLOTS = 2  # K/V 流环：第 tick 拍用 tick%2，拍内预取 tick+1
SCALE = 0.08838834764831845  # 1/sqrt(128)
MAX_CORES = 28
WS_COLS = 4 * VL  # ws 记录行宽：O 0..127 / m 128..191 / l 192..255

# route A：(b, h_kv) 对连续段等量切分（编译期字面量）。
CUTS = tuple(ITEMS * c // NUM_CORES for c in range(MAX_CORES + 1))
# route B：KV 块单元连续段切分。
UCUTS = tuple(UNITS * c // NUM_CORES for c in range(MAX_CORES + 1))


# ---------------------------------------------------------------------------------------------
# VF：标准布局行 softmax（行 = q，lane = key）、O 更新、fd 两分部 merge
# ---------------------------------------------------------------------------------------------


@pl.vector_function
def _softmax_block(s_src, a_cur, l_prev, l_cur, m_prev, m_cur, nz_lo, nz_hi, first):
    """一个 KV 块的在线 softmax（本 AIV 半区 ``[G, BKV]`` FP32；AIV0 半区 = 全部有效行）。

    状态是三条 [G, VL] 广播行环（alpha / l / m），读上拍槽写本拍槽；``first``（item 内首块）
    不读旧状态。每行两遍：行最大（4 chunk 归约 + 广播合并），exp((s-m)*scale)、行和、FP16
    窄化（astype ZERO 成对 de_interleave）、vsstb 游标散写 row+1 pitch NZ（无显式 offset，
    mm_softmax_mm 配方；chunk 0/1 -> nz_lo，2/3 -> nz_hi）。VF 结果先命名再复用（parser 约束）。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    for r in pl.range(G):
        c0 = vf.load_align(s_src, [r, 0])
        c1 = vf.load_align(s_src, [r, VL])
        c2 = vf.load_align(s_src, [r, 2 * VL])
        c3 = vf.load_align(s_src, [r, 3 * VL])
        b0 = vf.max(c0, c1, preg)
        b1 = vf.max(c2, c3, preg)
        b2 = vf.max(b0, b1, preg)
        mr = vf.reduce_max(b2, preg)
        mblk = vf.full(mr, preg)
        if first:
            m_new = mblk
        else:
            m_old = vf.load_align(m_prev, [r, 0])
            m_new = vf.max(m_old, mblk, preg)
            dm = vf.sub(m_old, m_new, preg)
            dms = vf.muls(dm, SCALE, preg)
            alpha = vf.exp(dms, preg)
            vf.store_align(a_cur, alpha, preg, [r, 0])
        vf.store_align(m_cur, m_new, preg, [r, 0])
        d0 = vf.sub(c0, m_new, preg)
        d1 = vf.sub(c1, m_new, preg)
        d2 = vf.sub(c2, m_new, preg)
        d3 = vf.sub(c3, m_new, preg)
        e0 = vf.muls(d0, SCALE, preg)
        e1 = vf.muls(d1, SCALE, preg)
        e2 = vf.muls(d2, SCALE, preg)
        e3 = vf.muls(d3, SCALE, preg)
        p0 = vf.exp(e0, preg)
        p1 = vf.exp(e1, preg)
        p2 = vf.exp(e2, preg)
        p3 = vf.exp(e3, preg)
        s01 = vf.add(p0, p1, preg)
        s23 = vf.add(p2, p3, preg)
        s03 = vf.add(s01, s23, preg)
        rs = vf.reduce_sum(s03, preg)
        rsb = vf.full(rs, preg)
        if first:
            vf.store_align(l_cur, rsb, preg, [r, 0])
        else:
            l_new = vf.load_align(l_prev, [r, 0])
            l_new = vf.mul_dst_add(alpha, rsb, preg)  # l = l*alpha + rowsum(P)
            vf.store_align(l_cur, l_new, preg, [r, 0])
        h0 = vf.astype(p0, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        h1 = vf.astype(p1, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        lo_packed, _a = vf.de_interleave(h0, h1)
        vf.store_align(
            nz_lo,
            lo_packed,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )
        h2 = vf.astype(p2, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        h3 = vf.astype(p3, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        hi_packed, _b = vf.de_interleave(h2, h3)
        vf.store_align(
            nz_hi,
            hi_packed,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def _update_block(pv_src, a_cur, l_cur, o_run, first, last):
    """``O = alpha*O + PV``（首块 ``O = PV``）；``last`` 时融合 ``/ l`` 原位写回 ``o_run``。

    ``pv_src``：本 AIV 半区 ``[G, D]`` FP32（AIV0 = 有效行）。alpha/l 从本拍状态槽的广播行读。
    fd 的部分收尾（不除 l）走 ``last=0``，由 stage 把未归一的 ``o_run`` 与 m/l 记录存 ws。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(G):
        if not first:
            a = vf.load_align(a_cur, [r, 0])
        if last:
            lr = vf.load_align(l_cur, [r, 0])
        for c in pl.range(DCH):
            pv = vf.load_align(pv_src, [r, c * VL])
            if first:
                o_row = pv
            else:
                o_row = vf.load_align(o_run, [r, c * VL])
                o_row = vf.mul_dst_add(a, pv, preg)
            if last:
                oq = vf.div(o_row, lr, preg)
                vf.store_align(o_run, oq, preg, [r, c * VL])
            else:
                vf.store_align(o_run, o_row, preg, [r, c * VL])


@pl.vector_function
def _merge_parts(p0, p1, o_run):
    """fd phase 2：两分部 merge。``p0``/``p1``：``[G, 4*VL]`` 记录（O/m/l 广播行，见 docstring）。

    m = max(m0, m1)；a_i = exp(scale*(m_i - m))；O = (a0 O0 + a1 O1) / (a0 l0 + a1 l1)。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(G):
        m0 = vf.load_align(p0, [r, 2 * VL])
        m1 = vf.load_align(p1, [r, 2 * VL])
        mm = vf.max(m0, m1, preg)
        d0 = vf.sub(m0, mm, preg)
        d1 = vf.sub(m1, mm, preg)
        s0 = vf.muls(d0, SCALE, preg)
        s1 = vf.muls(d1, SCALE, preg)
        a0 = vf.exp(s0, preg)
        a1 = vf.exp(s1, preg)
        l0 = vf.load_align(p0, [r, 3 * VL])
        l1 = vf.load_align(p1, [r, 3 * VL])
        t0 = vf.mul(l0, a0, preg)
        t1 = vf.mul(l1, a1, preg)
        lm = vf.add(t0, t1, preg)
        for c in pl.range(DCH):
            o0 = vf.load_align(p0, [r, c * VL])
            o1 = vf.load_align(p1, [r, c * VL])
            u0 = vf.mul(o0, a0, preg)
            u1 = vf.mul(o1, a1, preg)
            us = vf.add(u0, u1, preg)
            oq = vf.div(us, lm, preg)
            vf.store_align(o_run, oq, preg, [r, c * VL])


# ---------------------------------------------------------------------------------------------
# B 层 stage（route A 与 fd 共形；fd 版多 segment 边界与部分收尾参数）
# ---------------------------------------------------------------------------------------------


@pl.pipeline.stage
def qk_block(
    j,
    j1,
    tick,
    b,
    h,
    qs,
    nb,
    nh,
    has_next,
    k,
    q_g,
    k_g,
    k_zn,
    l0a1,
    l0b1,
    acc1_g,
    shared_s,
):
    q_tile = q_g[qs]
    k_t = k_zn[
        tick % KV_SLOTS
    ]  # K_j^T（组级 ZN 转置视图 + 标量选槽，按连续拍号取奇偶）
    acc1 = acc1_g.next()
    pl.gemm(acc1, q_tile, k_t, l0a=l0a1, l0b=l0b1, splitk=SK)  # S = Q K_j^T，[16, BKV]
    s_slot = shared_s.next()
    pl.move(s_slot, acc1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    if j + 1 < j1:  # 块内预取：第 tick 拍发射第 tick+1 拍的 K
        nk_tile = k_g[(tick + 1) % KV_SLOTS]
        pl.load(nk_tile, k, [b, h, (j + 1) * BKV, 0], order=[2, 3])
    if has_next == 1:  # 跨 item 预取：末拍发射下一 item 的首块 K
        if j == j1 - 1:
            x_tile = k_g[(tick + 1) % KV_SLOTS]
            pl.load(x_tile, k, [nb, nh, 0, 0], order=[2, 3])


@pl.pipeline.stage
def softmax_block(j, j0, tick, shared_s, st_a, st_l, st_m, nzlo_g, nzhi_g, shared_p):
    s_slot = shared_s.next()
    a_cur = st_a[tick % ST_SLOTS]
    l_cur = st_l[tick % ST_SLOTS]
    l_prev = st_l[(tick + ST_SLOTS - 1) % ST_SLOTS]
    m_cur = st_m[tick % ST_SLOTS]
    m_prev = st_m[(tick + ST_SLOTS - 1) % ST_SLOTS]
    nz_lo = nzlo_g.next()
    nz_hi = nzhi_g.next()
    s0 = pl.get_subblock_idx()
    if j == j0:
        _softmax_block(s_slot, a_cur, l_prev, l_cur, m_prev, m_cur, nz_lo, nz_hi, 1)
    else:
        _softmax_block(s_slot, a_cur, l_prev, l_cur, m_prev, m_cur, nz_lo, nz_hi, 0)
    pl.set_validshape(nz_lo, [G, HALF_N])
    pl.set_validshape(nz_hi, [G, HALF_N])
    p_tile = shared_p.next()
    if s0 == 0:  # 只有 AIV0 半区是有效行；AIV1 的 nz 结果不进 P
        pl.insert(p_tile, nz_lo, [0, 0])
        pl.insert(p_tile, nz_hi, [0, HALF_N])


@pl.pipeline.stage
def pv_block(
    j, j1, tick, b, h, nb, nh, has_next, v, v_g, l0a2, l0b2, acc2_g, shared_p, shared_pv
):
    v_tile = v_g[tick % KV_SLOTS]
    p_tile = shared_p.next()
    acc2 = acc2_g.next()
    pl.gemm(
        acc2, p_tile, v_tile, l0a=l0a2, l0b=l0b2, splitk=SK
    )  # PV_j = P_j V_j，[16, D]
    pv_slot = shared_pv.next()
    pl.move(pv_slot, acc2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    if j + 1 < j1:
        nv_tile = v_g[(tick + 1) % KV_SLOTS]
        pl.load(nv_tile, v, [b, h, (j + 1) * BKV, 0], order=[2, 3])
    if has_next == 1:
        if j == j1 - 1:
            x_tile = v_g[(tick + 1) % KV_SLOTS]
            pl.load(x_tile, v, [nb, nh, 0, 0], order=[2, 3])


@pl.pipeline.stage
def update_block(j, j0, j1, tick, b, h, out, shared_pv, st_a, st_l, o_g):
    """route A 收尾：末块（j1-1）除 l 后 AIV0 存出。"""
    pv_slot = shared_pv.next()
    a_cur = st_a[tick % ST_SLOTS]
    l_cur = st_l[tick % ST_SLOTS]
    o_run = o_g.current()
    if j == j0:
        _update_block(pv_slot, a_cur, l_cur, o_run, 1, 0)
    else:
        if j == j1 - 1:
            _update_block(pv_slot, a_cur, l_cur, o_run, 0, 1)
        else:
            _update_block(pv_slot, a_cur, l_cur, o_run, 0, 0)
    if j == j1 - 1:
        if pl.get_subblock_idx() == 0:  # 只有 AIV0 持有有效行
            pl.store(out, o_run, [b, h * G, 0, 0], order=[1, 3])


@pl.pipeline.stage
def update_block_fd(
    j, j0, j1, pmode, it, tick, b, h, out, ws, shared_pv, st_a, st_l, st_m, o_g
):
    """fd 收尾：``pmode`` 0 = 完整 item（除 l 存 out）；1/2 = front/back 部分
    （不除 l，把 O/m/l 记录存 ``ws[it, pmode-1]``，phase 2 合并）。
    """
    pv_slot = shared_pv.next()
    a_cur = st_a[tick % ST_SLOTS]
    l_cur = st_l[tick % ST_SLOTS]
    o_run = o_g.current()
    # 完整 item 恒有 j0=0、j1=NB（每核 >= 73 单元，first==last 不可能），所以
    # 「首块即末块」只出现在部分 segment（pmode != 0），除法分支不会与 first 重叠。
    fin = 0
    if j == j1 - 1:
        if pmode == 0:
            fin = 1
    if j == j0:
        _update_block(pv_slot, a_cur, l_cur, o_run, 1, 0)
    else:
        if fin == 1:
            _update_block(pv_slot, a_cur, l_cur, o_run, 0, 1)
        else:
            _update_block(pv_slot, a_cur, l_cur, o_run, 0, 0)
    if j == j1 - 1:
        if pl.get_subblock_idx() == 0:
            if pmode == 0:
                pl.store(out, o_run, [b, h * G, 0, 0], order=[1, 3])
            else:
                part = pmode - 1
                m_rec = st_m[tick % ST_SLOTS]
                l_rec = st_l[tick % ST_SLOTS]
                pl.store(ws, o_run, [it, part, 0, 0], order=[2, 3])
                pl.store(ws, m_rec, [it, part, 0, 2 * VL], order=[2, 3])
                pl.store(ws, l_rec, [it, part, 0, 3 * VL], order=[2, 3])


def _make_kernel(preload: int, name: str):
    """Build a separately named entry with the route's fixed physical plan."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        q: pl.Tensor[[B, HQ, 1, D], FP16],
        k: pl.Tensor[[B, HKV, S, D], FP16],
        v: pl.Tensor[[B, HKV, S, D], FP16],
        out: pl.Tensor[[B, HQ, 1, D], FP32],
    ):
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        begin = 0
        end = 0
        if core_id == 0:
            begin = CUTS[0]
            end = CUTS[1]
        if core_id == 1:
            begin = CUTS[1]
            end = CUTS[2]
        if core_id == 2:
            begin = CUTS[2]
            end = CUTS[3]
        if core_id == 3:
            begin = CUTS[3]
            end = CUTS[4]
        if core_id == 4:
            begin = CUTS[4]
            end = CUTS[5]
        if core_id == 5:
            begin = CUTS[5]
            end = CUTS[6]
        if core_id == 6:
            begin = CUTS[6]
            end = CUTS[7]
        if core_id == 7:
            begin = CUTS[7]
            end = CUTS[8]
        if core_id == 8:
            begin = CUTS[8]
            end = CUTS[9]
        if core_id == 9:
            begin = CUTS[9]
            end = CUTS[10]
        if core_id == 10:
            begin = CUTS[10]
            end = CUTS[11]
        if core_id == 11:
            begin = CUTS[11]
            end = CUTS[12]
        if core_id == 12:
            begin = CUTS[12]
            end = CUTS[13]
        if core_id == 13:
            begin = CUTS[13]
            end = CUTS[14]
        if core_id == 14:
            begin = CUTS[14]
            end = CUTS[15]
        if core_id == 15:
            begin = CUTS[15]
            end = CUTS[16]
        if core_id == 16:
            begin = CUTS[16]
            end = CUTS[17]
        if core_id == 17:
            begin = CUTS[17]
            end = CUTS[18]
        if core_id == 18:
            begin = CUTS[18]
            end = CUTS[19]
        if core_id == 19:
            begin = CUTS[19]
            end = CUTS[20]
        if core_id == 20:
            begin = CUTS[20]
            end = CUTS[21]
        if core_id == 21:
            begin = CUTS[21]
            end = CUTS[22]
        if core_id == 22:
            begin = CUTS[22]
            end = CUTS[23]
        if core_id == 23:
            begin = CUTS[23]
            end = CUTS[24]
        if core_id == 24:
            begin = CUTS[24]
            end = CUTS[25]
        if core_id == 25:
            begin = CUTS[25]
            end = CUTS[26]
        if core_id == 26:
            begin = CUTS[26]
            end = CUTS[27]
        if core_id == 27:
            begin = CUTS[27]
            end = CUTS[28]

        q_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=Q_SLOTS,
        )
        k_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        v_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x22000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        l0a1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, BKV], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, BKV], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0a2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x1000,
            mutex_ids="auto",
            depth=2,
        )
        l0b2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, D], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=1,
        )
        acc2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, D], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=2,
        )
        shared_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, BKV],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        shared_p = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, BKV],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x42000,
            mutex_ids="auto",
            depth=P_SLOTS,
            fwd_ids=[4, 5],
            bwd_ids=[6, 7],
        )
        shared_pv = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, D], dtype=FP32, target_memory=pl.MemorySpace.Vec, layout=pl.ND
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[8, 9],
            bwd_ids=[10, 11],
        )
        nzlo_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x6000,
            mutex_ids="auto",
            depth=2,
        )
        nzhi_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x7200,
            mutex_ids="auto",
            depth=2,
        )
        st_a = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x8400,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        st_l = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x9C00,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        st_m = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0xB400,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        o_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, D], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0xCC00,
            mutex_ids="auto",
            depth=1,
        )

        # K 槽组的转置视图：gemm 的 b 操作数 K_j^T；trans_x 不接受运行期选槽 tile，
        # 组级 reinterpret 视图 + 标量下标是框架点名的出路（MHA q_zn 同款）。
        k_zn = pl.reinterpret(k_g, shape=[D, BKV], layout=pl.ZN)

        with pl.section_cube():
            # 核内首个 item 的 Q 与 K/V 块 0；此后每个 item 的首块由上一 item 末拍预取
            fb = begin // HKV
            fh = begin - fb * HKV
            pl.set_validshape(q_g[begin % Q_SLOTS], [G, D])
            pl.load(q_g[begin % Q_SLOTS], q, [fb, fh * G, 0, 0], order=[1, 3])
            pl.set_validshape(q_g[begin % Q_SLOTS], [MP, D])
            pl.load(k_g[0], k, [fb, fh, 0, 0], order=[2, 3])
            pl.load(v_g[0], v, [fb, fh, 0, 0], order=[2, 3])

        tick = 0
        for it in pl.range(begin, end):
            b = it // HKV
            h = it - b * HKV
            with pl.section_cube():
                # 下一 item 的 Q：循环体顶部发射（先于本 item 首个 stage 的跨核 wait），
                # 只搬 G 行（set_validshape 限定搬运量，不越界读 GM），随后恢复声明形。
                if it + 1 < end:
                    nxt = it + 1
                    pb = nxt // HKV
                    ph = nxt - pb * HKV
                    pl.set_validshape(q_g[nxt % Q_SLOTS], [G, D])
                    pl.load(q_g[nxt % Q_SLOTS], q, [pb, ph * G, 0, 0], order=[1, 3])
                    pl.set_validshape(q_g[nxt % Q_SLOTS], [MP, D])
            has_next = 0
            nb = 0
            nh = 0
            if it + 1 < end:
                has_next = 1
                nxt2 = it + 1
                nb = nxt2 // HKV
                nh = nxt2 - nb * HKV
            qs = it % Q_SLOTS
            for j in pl.range(NB):
                with pl.section_cube():
                    qk_block(
                        j,
                        NB,
                        tick,
                        b,
                        h,
                        qs,
                        nb,
                        nh,
                        has_next,
                        k,
                        q_g,
                        k_g,
                        k_zn,
                        l0a1,
                        l0b1,
                        acc1_g,
                        shared_s,
                    )
                with pl.section_vector():
                    softmax_block(
                        j, 0, tick, shared_s, st_a, st_l, st_m, nzlo_g, nzhi_g, shared_p
                    )
                with pl.section_cube():
                    pv_block(
                        j,
                        NB,
                        tick,
                        b,
                        h,
                        nb,
                        nh,
                        has_next,
                        v,
                        v_g,
                        l0a2,
                        l0b2,
                        acc2_g,
                        shared_p,
                        shared_pv,
                    )
                with pl.section_vector():
                    update_block(j, 0, NB, tick, b, h, out, shared_pv, st_a, st_l, o_g)
                tick = tick + 1

    return kernel


def _make_kernel_fd(preload: int, name: str):
    """Build a separately named entry with the route's fixed physical plan."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        q: pl.Tensor[[B, HQ, 1, D], FP16],
        k: pl.Tensor[[B, HKV, S, D], FP16],
        v: pl.Tensor[[B, HKV, S, D], FP16],
        ws: pl.Tensor[[ITEMS, 2, G, WS_COLS], FP32],
        out: pl.Tensor[[B, HQ, 1, D], FP32],
    ):
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        ub = 0
        ue = 0
        if core_id == 0:
            ub = UCUTS[0]
            ue = UCUTS[1]
        if core_id == 1:
            ub = UCUTS[1]
            ue = UCUTS[2]
        if core_id == 2:
            ub = UCUTS[2]
            ue = UCUTS[3]
        if core_id == 3:
            ub = UCUTS[3]
            ue = UCUTS[4]
        if core_id == 4:
            ub = UCUTS[4]
            ue = UCUTS[5]
        if core_id == 5:
            ub = UCUTS[5]
            ue = UCUTS[6]
        if core_id == 6:
            ub = UCUTS[6]
            ue = UCUTS[7]
        if core_id == 7:
            ub = UCUTS[7]
            ue = UCUTS[8]
        if core_id == 8:
            ub = UCUTS[8]
            ue = UCUTS[9]
        if core_id == 9:
            ub = UCUTS[9]
            ue = UCUTS[10]
        if core_id == 10:
            ub = UCUTS[10]
            ue = UCUTS[11]
        if core_id == 11:
            ub = UCUTS[11]
            ue = UCUTS[12]
        if core_id == 12:
            ub = UCUTS[12]
            ue = UCUTS[13]
        if core_id == 13:
            ub = UCUTS[13]
            ue = UCUTS[14]
        if core_id == 14:
            ub = UCUTS[14]
            ue = UCUTS[15]
        if core_id == 15:
            ub = UCUTS[15]
            ue = UCUTS[16]
        if core_id == 16:
            ub = UCUTS[16]
            ue = UCUTS[17]
        if core_id == 17:
            ub = UCUTS[17]
            ue = UCUTS[18]
        if core_id == 18:
            ub = UCUTS[18]
            ue = UCUTS[19]
        if core_id == 19:
            ub = UCUTS[19]
            ue = UCUTS[20]
        if core_id == 20:
            ub = UCUTS[20]
            ue = UCUTS[21]
        if core_id == 21:
            ub = UCUTS[21]
            ue = UCUTS[22]
        if core_id == 22:
            ub = UCUTS[22]
            ue = UCUTS[23]
        if core_id == 23:
            ub = UCUTS[23]
            ue = UCUTS[24]
        if core_id == 24:
            ub = UCUTS[24]
            ue = UCUTS[25]
        if core_id == 25:
            ub = UCUTS[25]
            ue = UCUTS[26]
        if core_id == 26:
            ub = UCUTS[26]
            ue = UCUTS[27]
        if core_id == 27:
            ub = UCUTS[27]
            ue = UCUTS[28]

        q_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=Q_SLOTS,
        )
        k_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        v_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x22000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        l0a1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, BKV], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, BKV], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0a2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x1000,
            mutex_ids="auto",
            depth=2,
        )
        l0b2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, D], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=1,
        )
        acc2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, D], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=2,
        )
        shared_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, BKV],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        shared_p = pl.make_tile_group(
            type=pl.TileType(
                shape=[MP, BKV],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x42000,
            mutex_ids="auto",
            depth=P_SLOTS,
            fwd_ids=[4, 5],
            bwd_ids=[6, 7],
        )
        shared_pv = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, D], dtype=FP32, target_memory=pl.MemorySpace.Vec, layout=pl.ND
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[8, 9],
            bwd_ids=[10, 11],
        )
        nzlo_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x6000,
            mutex_ids="auto",
            depth=2,
        )
        nzhi_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x7200,
            mutex_ids="auto",
            depth=2,
        )
        st_a = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x8400,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        st_l = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x9C00,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        st_m = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, VL], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0xB400,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        o_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, D], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0xCC00,
            mutex_ids="auto",
            depth=1,
        )
        pw_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[G, WS_COLS], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0xDC00,
            mutex_ids="auto",
            depth=2,
        )

        k_zn = pl.reinterpret(k_g, shape=[D, BKV], layout=pl.ZN)

        first_item = ub // NB
        last_item = (ue - 1) // NB
        jhead = ub - first_item * NB  # 核内首 item 的起始块（>0 即 back 部分）
        jtail = ue - last_item * NB  # 核内末 item 的终止块（<NB 即 front 部分）

        with pl.section_cube():
            fb = first_item // HKV
            fh = first_item - fb * HKV
            pl.set_validshape(q_g[first_item % Q_SLOTS], [G, D])
            pl.load(q_g[first_item % Q_SLOTS], q, [fb, fh * G, 0, 0], order=[1, 3])
            pl.set_validshape(q_g[first_item % Q_SLOTS], [MP, D])
            pl.load(k_g[0], k, [fb, fh, jhead * BKV, 0], order=[2, 3])
            pl.load(v_g[0], v, [fb, fh, jhead * BKV, 0], order=[2, 3])

        tick = 0
        for it in pl.range(first_item, last_item + 1):
            b = it // HKV
            h = it - b * HKV
            with pl.section_cube():
                if it + 1 <= last_item:
                    nxt = it + 1
                    pb = nxt // HKV
                    ph = nxt - pb * HKV
                    pl.set_validshape(q_g[nxt % Q_SLOTS], [G, D])
                    pl.load(q_g[nxt % Q_SLOTS], q, [pb, ph * G, 0, 0], order=[1, 3])
                    pl.set_validshape(q_g[nxt % Q_SLOTS], [MP, D])
            j0 = 0
            if it == first_item:
                j0 = jhead
            j1 = NB
            if it == last_item:
                j1 = jtail
            pmode = 0
            if it == first_item:
                if jhead != 0:
                    pmode = 2  # back 部分 -> ws part 1
            if it == last_item:
                if jtail != NB:
                    pmode = 1  # front 部分 -> ws part 0
            has_next = 0
            nb = 0
            nh = 0
            if it + 1 <= last_item:
                has_next = 1
                nxt2 = it + 1
                nb = nxt2 // HKV
                nh = nxt2 - nb * HKV
            qs = it % Q_SLOTS
            for j in pl.range(j0, j1):
                with pl.section_cube():
                    qk_block(
                        j,
                        j1,
                        tick,
                        b,
                        h,
                        qs,
                        nb,
                        nh,
                        has_next,
                        k,
                        q_g,
                        k_g,
                        k_zn,
                        l0a1,
                        l0b1,
                        acc1_g,
                        shared_s,
                    )
                with pl.section_vector():
                    softmax_block(
                        j,
                        j0,
                        tick,
                        shared_s,
                        st_a,
                        st_l,
                        st_m,
                        nzlo_g,
                        nzhi_g,
                        shared_p,
                    )
                with pl.section_cube():
                    pv_block(
                        j,
                        j1,
                        tick,
                        b,
                        h,
                        nb,
                        nh,
                        has_next,
                        v,
                        v_g,
                        l0a2,
                        l0b2,
                        acc2_g,
                        shared_p,
                        shared_pv,
                    )
                with pl.section_vector():
                    update_block_fd(
                        j,
                        j0,
                        j1,
                        pmode,
                        it,
                        tick,
                        b,
                        h,
                        out,
                        ws,
                        shared_pv,
                        st_a,
                        st_l,
                        st_m,
                        o_g,
                    )
                tick = tick + 1

        # phase 边界：vector 侧先沥干本核 MTE3（ws 部分状态落盘）再进全核屏障；
        # MIX 屏障必须 cube / vector 两侧都到达（框架 ST 的调用契约）。
        with pl.section_cube():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

        # phase 2：head 截断核对自己的首 item 做两分部 merge（连续段保证至多两分部）。
        with pl.section_vector():
            if jhead != 0:
                if pl.get_subblock_idx() == 0:
                    mb = first_item // HKV
                    mh = first_item - mb * HKV
                    pl.load(pw_g[0], ws, [first_item, 0, 0, 0], order=[2, 3])
                    pl.load(pw_g[1], ws, [first_item, 1, 0, 0], order=[2, 3])
                    _merge_parts(pw_g[0], pw_g[1], o_g.current())
                    pl.store(out, o_g.current(), [mb, mh * G, 0, 0], order=[1, 3])

    return kernel


decode_attention_gqa = _make_kernel(preload=PRELOAD, name="decode_attention_gqa")
decode_attention_gqa_p0 = _make_kernel(preload=0, name="decode_attention_gqa_p0")
decode_attention_gqa_fd = _make_kernel_fd(
    preload=PRELOAD, name="decode_attention_gqa_fd"
)
decode_attention_gqa_fd_p0 = _make_kernel_fd(
    preload=0, name="decode_attention_gqa_fd_p0"
)
