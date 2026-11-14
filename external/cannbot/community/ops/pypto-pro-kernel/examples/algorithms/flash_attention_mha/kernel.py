# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Causal MHA with owner caching, guarded prefetch and explicit physical resources.

Contract: O[b,i,h] = softmax(Q[b,i,h] @ K[b,:i+1,h].T / sqrt(128)) @ V[b,:i+1,h]. Q/K/V are contiguous FP16 BSND
[4,1024,32,128]; O has the same shape in FP32. There is no dropout. Ordinary inputs are finite; poison probes check only
query rows whose visible prefix remains finite. Scores, running state and O accumulation use FP32; unnormalized P
narrows to FP16 for Cube.

An item is (batch,head,query block), with 128 rows per block. There are 128 owners and eight items per owner. Host
weighted_partition assigns 1024 items to 28 contiguous execution-group ranges, minimizing maximum causal work and then
imbalance, K/V loads and range lengths. A range can cross owners or start inside an owner, so first-of-owner handling
includes the range's first item.

Each item runs qk -> softmax -> pv -> update over KV blocks 0..g. Transposed scores have key rows and query lanes, as in
flash_attention. A persistent [64,64] bias tile supplies the diagonal staircase; AIV0 skips the fully invisible key
half. The first block initializes running state, and the last block divides by l and stores the item's FP32 output. No
full K/V matrix or score row is held locally.

Q has two slots: the next item is prefetched at the current item's loop entry. Each K/V group has three owner-cache
slots and two streaming slots; later blocks use NK+(j-NK)%NSK. First-of-owner items reload their visible prefix, while
other items reuse the cached prefix. K/V prefetch the next block in their respective Cube stages, and prefetch a new
owner's block zero at the preceding owner's final block. it+1<end, has_next and j<g guard these reads independently of
whether their results affect numerical output.

The group-level Q ZN view permits dynamic slot selection of Q.T. Two [65,64] NZ tiles pack P.T; insert offsets [0,off]
and [KH,off] use NZ source coordinates to assemble the ZN shared P tile. Per-item scalar context is passed to stages;
shared groups and views remain loop-invariant.

Explicit byte ranges at PRELOAD=4 (ends exclusive):

| Space | Groups and ranges | Bytes/slot x depth |
| --- | --- | --- |
| Mat | q_g [0,0x10000) | 32768 x 2 |
| Mat | k_g [0x10000,0x38000), v_g [0x38000,0x60000) | 32768 x 5 each |
| Mat | shared_p [0x60000,0x80000) | 32768 x 4 |
| Left | l0a1 [0,0x8000), l0a2 [0x8000,0x10000) | 16384 x 2 each |
| Right | l0b1 [0,0x8000), l0b2 [0x8000,0x10000) | 16384 x 2 each |
| Acc | acc1_g [0,0x20000), acc2_g [0x20000,0x40000) | 65536 x 2 each |
| Vec | shared_s [0,0x10000), shared_pv [0x10000,0x20000) | 32768 x 2 each |
| Vec | ptm_g [0x20000,0x24100), ptb_g [0x24100,0x28200) | 8320 x 2 each |
| Vec | st_g [0x28200,0x29100) | 768 x 5 |
| Vec | tri_g [0x29100,0x2d100), o_g [0x2d100,0x35100) | 16384 x 1 / 32768 x 1 |

All groups use automatic intra-core mutex IDs. shared_s uses forward [0,1], reverse [2,3]; shared_p uses forward
[4,5,6,7], reverse [8,9,10,11]; shared_pv uses forward [12,13], reverse [14,15]. Score/product edges are FIX -> V with V
-> FIX release; P is MTE3 -> MTE1 with MTE1 -> MTE3 release. All 16 event IDs are occupied, one per slot and direction.
VF reads the shared score/product inputs without modifying them.

Stage delays are 0/1/4/5. Score/product handovers need two slots each; P needs four. The same-core m/l/alpha state ring
needs ST_SLOTS=PRELOAD+1=5, selected by a monotonically increasing tick across items. The preload=0 control retains
these groups and depths. Addresses and event arrays describe the fixed constants above and must be replanned when
geometry or depths change.

Mat, Left/Right and Acc reach their respective 512/64/256 KiB capacities. Vec uses 217344 bytes of 256 KiB. Tests use an
independent FP64 causal reference and the unchanged input-dependent budget in reference.error_terms, plus staircase,
first-row, poison, uniform-prefix, one-hot, partial-launch, owner-boundary and zero-tolerance consistency checks. Static
guards cover storage and prefetch bounds; nine incorrect mathematical controls exercise the comparator. Passing
consistency does not prove race freedom. This example makes no current performance claim.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

BATCH, SEQ, HEADS, D = 4, 1024, 32, 128  # BSND：Q/K/V/O 均为 [B, S, H, D]
NUM_CORES = 28
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 的方括号数是执行组数（1c+2v/组）
BQ = 128  # 一个 item 的 q 行数
HALF = BQ // 2  # 一个 AIV 的 q 行数 = 一条 FP32 寄存器的 lane 数
BKV = 128  # 一个 KV 块的 key 数
KH = BKV // 2
QB = SEQ // BQ  # 每个 (batch, head) 的 q 块数 = 8
OWNERS = BATCH * HEADS  # (batch, head) 对数 = 128
ITEMS = OWNERS * QB  # 工作 item = (batch, head, q 块) = 1024
SK = 64
VL = 64
DCH = D // VL
NZ_PITCH = KH + 1
PRELOAD = 4  # Main schedule; address ranges and events use this fixed depth.
ST_SLOTS = PRELOAD + 1  # 同核跨 stage 的状态环：拍距 PRELOAD，槽数 = 拍距 + 1
P_SLOTS = max(PRELOAD - 1, 0) + 1  # P 跨核交接槽：拍距 PRELOAD - 1，槽数 = 拍距 + 1
Q_SLOTS = 2  # Q 双槽：item i 顶部预取 item i+1 的 Q
NK = 3  # owner 缓存：每 head 的前 NK 个 K/V 块常驻 L1
NSK = 2  # 其余块的流式槽数
KV_SLOTS = NK + NSK
SCALE = 0.08838834764831845  # 1/sqrt(128)
NEG_BIG = -1.0e30
ITEM_COST = 0  # 每 item 固定开销（以 1/4 步为单位），切分权重的一项
MAX_CORES = 28  # kernel 里 if 链的路数


# ---------------------------------------------------------------------------------------------
# host：按因果 key 权重的连续切分
# ---------------------------------------------------------------------------------------------


def tile_needs_load(j, g, first_of_owner):
    """owner 缓存策略：item ``g`` 的第 ``j`` 个 K/V 块是否要从 GM 载入。

    前 ``NK`` 个块常驻；同一 head 的连续 item（g 递增）里，块 ``j < g`` 已由前一个 item 载入。
    ``first_of_owner`` = 该核上这个 head 的第一个 item（head 起点，或被核边界截断后的第一项）。
    """
    return j >= NK or first_of_owner or j >= g


@lru_cache(maxsize=None)
def weighted_partition(items, cores, item_cost=ITEM_COST):
    """把 ``items`` 个连续 item 切成 ``cores`` 段，返回长度 ``cores + 1`` 的切点表（host 算好，kernel 里是字面量）。

    item ``i`` 的 g = ``i % QB``，它看 ``g + 1`` 个 KV 块（因果），权重 = ``4 * (g + 1) + item_cost``。
    目标依次为：最大段权重最小（贪心可行性 + 二分）；再按字典序最小化（各段与均值差的平方和，K/V 载入次数，
    各段 item 数的平方和）。同 owner 的 item 在切点处尽量不被截断由第二项体现。
    """
    assert items >= cores
    weights = [4 * (i % QB + 1) + item_cost for i in range(items)]
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    total = prefix[-1]

    def cores_needed(cap):
        used, acc = 1, 0
        for w in weights:
            if acc + w > cap:
                used, acc = used + 1, w
            else:
                acc += w
        return used

    lo, hi = max(weights), total
    while lo < hi:
        mid = (lo + hi) // 2
        if cores_needed(mid) <= cores:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    # 单个 item 在「是 / 不是该核该 head 首项」两种情形下的 K/V 载入次数
    loads = {
        fo: [sum(tile_needs_load(j, g, fo) for j in range(g + 1)) for g in range(QB)]
        for fo in (True, False)
    }

    states = {0: ((0, 0, 0), (0,))}
    for used in range(1, cores + 1):
        remaining = cores - used
        nxt = {}
        for begin, (score, cuts) in states.items():
            n_loads = 0
            for end in range(begin + 1, items - remaining + 1):
                it = end - 1
                g = it % QB
                n_loads += loads[g == 0 or it == begin][g]
                work = prefix[end] - prefix[begin]
                if work > cap:
                    break
                if total - prefix[end] > remaining * cap:
                    continue
                cand = ((cores * work - total) ** 2, n_loads, (end - begin) ** 2)
                new = (tuple(a + b for a, b in zip(score, cand)), (*cuts, end))
                if end not in nxt or new < nxt[end]:
                    nxt[end] = new
        states = nxt
    return states[items][1]


def _padded_cuts(items, cores):
    cuts = weighted_partition(items, cores)
    return cuts + (items,) * (MAX_CORES + 1 - len(cuts))


CUTS = _padded_cuts(ITEMS, NUM_CORES)


# ---------------------------------------------------------------------------------------------
# vector 函数
# ---------------------------------------------------------------------------------------------


@pl.vector_function
def _init_tri(tri):
    """阶梯偏置 ``tri[a, l] = NEG_BIG if l < a else 0``（``[HALF, HALF]``）：一次性造好，对角块的掩码只需 ``x + tri[a]``。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    negbig = vf.full(NEG_BIG, preg, dtype=FP32)
    zero = vf.full(0.0, preg, dtype=FP32)
    for a in pl.range(HALF):
        mk = vf.update_mask(a, dtype=FP32)
        row = vf.select(negbig, zero, mk)
        vf.store_align(tri, row, preg, [a, 0])


@pl.vector_function
def _softmax_block(s_src, st_prev, st_cur, pt_main, pt_back, tri, first, diag):
    """一个 KV 块的在线 softmax（转置布局：行 = key，lane = q 行）。

    ``st_*`` 是 ``[3, HALF]`` 状态行：0 = alpha，1 = l，2 = m；读上一拍槽 ``st_prev``、写本拍槽 ``st_cur``。
    ``first``：本 item 的首块（不读旧状态，m = 块最大，l = 行和）。
    ``diag``：0 = 普通块；1 / 2 = 因果对角块在 AIV0 / AIV1 上的形态。本 AIV 的 q 是 ``[64 s, 64 s + 64)``，key 行 ``r`` 的被掩 lane
    是 ``l < r - 64 s``：AIV0（s=0）的 key 行 ``[0, 64)`` 是阶梯（``tri`` 偏置）、``[64, 128)`` 整行被掩（P = 0，跳过 exp）；AIV1（s=1）的
    key 行 ``[0, 64)`` 全可见、``[64, 128)`` 是阶梯。被掩位置加 NEG_BIG 使 exp 精确下溢为 0（api/vf.md 二点十五）。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=FP16)
    a0 = vf.full(NEG_BIG, preg, dtype=FP32)
    a1 = vf.full(NEG_BIG, preg, dtype=FP32)
    a2 = vf.full(NEG_BIG, preg, dtype=FP32)
    a3 = vf.full(NEG_BIG, preg, dtype=FP32)
    if diag == 1:
        for k in pl.range(KH // 4):
            x0 = vf.load_align(s_src, [4 * k, 0])
            x1 = vf.load_align(s_src, [4 * k + 1, 0])
            x2 = vf.load_align(s_src, [4 * k + 2, 0])
            x3 = vf.load_align(s_src, [4 * k + 3, 0])
            t0 = vf.load_align(tri, [4 * k, 0])
            t1 = vf.load_align(tri, [4 * k + 1, 0])
            t2 = vf.load_align(tri, [4 * k + 2, 0])
            t3 = vf.load_align(tri, [4 * k + 3, 0])
            x0 = vf.add(x0, t0, preg)
            x1 = vf.add(x1, t1, preg)
            x2 = vf.add(x2, t2, preg)
            x3 = vf.add(x3, t3, preg)
            a0 = vf.max(a0, x0, preg)
            a1 = vf.max(a1, x1, preg)
            a2 = vf.max(a2, x2, preg)
            a3 = vf.max(a3, x3, preg)
    else:
        for k in pl.range(KH // 4):
            x0 = vf.load_align(s_src, [4 * k, 0])
            x1 = vf.load_align(s_src, [4 * k + 1, 0])
            x2 = vf.load_align(s_src, [4 * k + 2, 0])
            x3 = vf.load_align(s_src, [4 * k + 3, 0])
            a0 = vf.max(a0, x0, preg)
            a1 = vf.max(a1, x1, preg)
            a2 = vf.max(a2, x2, preg)
            a3 = vf.max(a3, x3, preg)
        if diag == 2:
            for k in pl.range(KH // 4):
                x0 = vf.load_align(s_src, [KH + 4 * k, 0])
                x1 = vf.load_align(s_src, [KH + 4 * k + 1, 0])
                x2 = vf.load_align(s_src, [KH + 4 * k + 2, 0])
                x3 = vf.load_align(s_src, [KH + 4 * k + 3, 0])
                t0 = vf.load_align(tri, [4 * k, 0])
                t1 = vf.load_align(tri, [4 * k + 1, 0])
                t2 = vf.load_align(tri, [4 * k + 2, 0])
                t3 = vf.load_align(tri, [4 * k + 3, 0])
                x0 = vf.add(x0, t0, preg)
                x1 = vf.add(x1, t1, preg)
                x2 = vf.add(x2, t2, preg)
                x3 = vf.add(x3, t3, preg)
                a0 = vf.max(a0, x0, preg)
                a1 = vf.max(a1, x1, preg)
                a2 = vf.max(a2, x2, preg)
                a3 = vf.max(a3, x3, preg)
        else:
            for k in pl.range(KH // 4):
                x0 = vf.load_align(s_src, [KH + 4 * k, 0])
                x1 = vf.load_align(s_src, [KH + 4 * k + 1, 0])
                x2 = vf.load_align(s_src, [KH + 4 * k + 2, 0])
                x3 = vf.load_align(s_src, [KH + 4 * k + 3, 0])
                a0 = vf.max(a0, x0, preg)
                a1 = vf.max(a1, x1, preg)
                a2 = vf.max(a2, x2, preg)
                a3 = vf.max(a3, x3, preg)
    b0 = vf.max(a0, a1, preg)
    b1 = vf.max(a2, a3, preg)
    m_new = vf.max(b0, b1, preg)
    if not first:
        m_old = vf.load_align(st_prev, [2, 0])
        m_new = vf.max(m_old, m_new, preg)
        dm = vf.sub(m_old, m_new, preg)
        dms = vf.muls(dm, SCALE, preg)
        alpha = vf.exp(dms, preg)
        vf.store_align(st_cur, alpha, preg, [0, 0])
    vf.store_align(st_cur, m_new, preg, [2, 0])
    sa = vf.full(0.0, preg, dtype=FP32)
    sb = vf.full(0.0, preg, dtype=FP32)
    if diag == 1:
        zero_f32 = vf.full(0.0, preg, dtype=FP32)
        zero16 = vf.astype(zero_f32, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        dnz, _z = vf.de_interleave(zero16, zero16)
    for r in pl.range(KH):
        xa = vf.load_align(s_src, [r, 0])
        if diag == 1:
            ta = vf.load_align(tri, [r, 0])
            xa = vf.add(xa, ta, preg)
        else:
            xb = vf.load_align(s_src, [r + KH, 0])
            if diag == 2:
                tb = vf.load_align(tri, [r, 0])
                xb = vf.add(xb, tb, preg)
            db = vf.sub(xb, m_new, preg)
            eb = vf.muls(db, SCALE, preg)
            pb = vf.exp(eb, preg)
            sb = vf.add(sb, pb, preg)
            hb = vf.astype(pb, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
            dnb, _b = vf.de_interleave(hb, hb)
        da = vf.sub(xa, m_new, preg)
        ea = vf.muls(da, SCALE, preg)
        pa = vf.exp(ea, preg)
        sa = vf.add(sa, pa, preg)
        ha = vf.astype(pa, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        dna, _a = vf.de_interleave(ha, ha)
        vf.store_align(
            pt_main,
            dna,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )
        if diag == 1:
            vf.store_align(
                pt_back,
                dnz,
                preg16,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                block_stride=NZ_PITCH,
                repeat_stride=1,
                post_update=True,
            )
        else:
            vf.store_align(
                pt_back,
                dnb,
                preg16,
                data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
                block_stride=NZ_PITCH,
                repeat_stride=1,
                post_update=True,
            )
    stot = vf.add(sa, sb, preg)
    if first:
        vf.store_align(st_cur, stot, preg, [1, 0])
    else:
        l_new = vf.load_align(st_prev, [1, 0])
        l_new = vf.mul_dst_add(alpha, stot, preg)  # l = l * alpha + rowsum(P)
        vf.store_align(st_cur, l_new, preg, [1, 0])


@pl.vector_function
def _update_block(pv_src, st_cur, o_run, first, last):
    """``O = alpha * O + PV``（首块直接 ``O = PV``）；末块融合 ``/ l``，结果原位写回 ``o_run``，由调用方从 ``o_run`` 存出。

    原位写回省掉单独的 O 输出缓冲（UB 里 32KB）。下一个 item 的首块会覆盖 ``o_run``，与上一个 item 的 GM 存出之间的
    先后由 ``pl.jit(auto_mutex=True)`` 的核内 mutex 保证。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(HALF):
        if not first:
            a = vf.load_align(st_cur, r, dist=pl.LoadDist.BRC_B32)
        if last:
            lr = vf.load_align(st_cur, HALF + r, dist=pl.LoadDist.BRC_B32)
        for c in pl.range(DCH):
            pv = vf.load_align(pv_src, [r, c * VL])
            if first:
                o_row = pv
            else:
                o_row = vf.load_align(o_run, [r, c * VL])
                o_row = vf.mul_dst_add(a, pv, preg)
            if last:
                q = vf.div(o_row, lr, preg)
                vf.store_align(o_run, q, preg, [r, c * VL])
            else:
                vf.store_align(o_run, o_row, preg, [r, c * VL])


# ---------------------------------------------------------------------------------------------
# B 层 stage
# ---------------------------------------------------------------------------------------------


@pl.pipeline.stage
def qk_block(j, g, fo, has_next, b, h, qs, k, q_zn, k_g, l0a1, l0b1, acc1_g, shared_s):
    q_tile = q_zn[qs]  # Q_s^T（``q_zn`` 是 Q 槽组的转置视图，见 kernel 体）
    slot = j
    if j >= NK:
        slot = NK + (j - NK) % NSK
    k_tile = k_g[
        slot
    ]  # 块 j 已在槽里：首块由循环前 / 上一 item 预取，其余块由上一拍预取
    acc1 = acc1_g.next()
    pl.gemm(acc1, k_tile, q_tile, l0a=l0a1, l0b=l0b1, splitk=SK)  # S^T = K_j Q^T
    s_slot = shared_s.next()
    pl.move(s_slot, acc1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    if j < g:  # 预取本 item 的下一块（发射得比使用早一拍）
        nj = j + 1
        nslot = nj
        if nj >= NK:
            nslot = NK + (nj - NK) % NSK
        nload = 1
        if fo == 0:
            if nj < g:
                if nj < NK:
                    nload = 0
        if nload == 1:
            nk_tile = k_g[nslot]
            pl.load(nk_tile, k, [b, nj * BKV, h, 0], order=[1, 3])
    if has_next == 1:  # 下一个 item 是新 owner 的首个 item：预取它的块 0 进槽 0
        if g == QB - 1:
            if j == g:
                xh = h + 1
                xb = b
                if xh == HEADS:
                    xh = 0
                    xb = b + 1
                x_tile = k_g[0]
                pl.load(x_tile, k, [xb, 0, xh, 0], order=[1, 3])


@pl.pipeline.stage
def softmax_block(j, g, tick, shared_s, st_g, tri_g, ptm_g, ptb_g, shared_p):
    s_slot = shared_s.next()
    st_cur = st_g[tick % ST_SLOTS]
    st_prev = st_g[(tick + ST_SLOTS - 1) % ST_SLOTS]
    tri = tri_g.current()
    ptm = ptm_g.next()
    ptb = ptb_g.next()
    s0 = pl.get_subblock_idx()
    if j == g:
        if s0 == 0:
            if j == 0:
                _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 1, 1)
            else:
                _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 0, 1)
        else:
            if j == 0:
                _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 1, 2)
            else:
                _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 0, 2)
    else:
        if j == 0:
            _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 1, 0)
        else:
            _softmax_block(s_slot, st_prev, st_cur, ptm, ptb, tri, 0, 0)
    pl.set_validshape(ptm, [KH, HALF])
    pl.set_validshape(ptb, [KH, HALF])
    p_tile = shared_p.next()
    off = s0 * HALF
    pl.insert(p_tile, ptm, [0, off])
    pl.insert(p_tile, ptb, [KH, off])


@pl.pipeline.stage
def pv_block(j, g, fo, has_next, b, h, v, v_g, l0a2, l0b2, acc2_g, shared_p, shared_pv):
    slot = j
    if j >= NK:
        slot = NK + (j - NK) % NSK
    v_tile = v_g[slot]  # 块 j 已在槽里（预取规则同 qk_block）
    p_tile = shared_p.next()
    acc2 = acc2_g.next()
    pl.gemm(acc2, p_tile, v_tile, l0a=l0a2, l0b=l0b2, splitk=SK)  # PV_j = P_j V_j
    pv_slot = shared_pv.next()
    pl.move(pv_slot, acc2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    if j < g:
        nj = j + 1
        nslot = nj
        if nj >= NK:
            nslot = NK + (nj - NK) % NSK
        nload = 1
        if fo == 0:
            if nj < g:
                if nj < NK:
                    nload = 0
        if nload == 1:
            nv_tile = v_g[nslot]
            pl.load(nv_tile, v, [b, nj * BKV, h, 0], order=[1, 3])
    if has_next == 1:
        if g == QB - 1:
            if j == g:
                xh = h + 1
                xb = b
                if xh == HEADS:
                    xh = 0
                    xb = b + 1
                x_tile = v_g[0]
                pl.load(x_tile, v, [xb, 0, xh, 0], order=[1, 3])


@pl.pipeline.stage
def update_block(j, g, b, h, tick, out, shared_pv, st_g, o_g):
    pv_slot = shared_pv.next()
    st_cur = st_g[tick % ST_SLOTS]
    o_run = o_g.current()
    if j == 0:
        if g == 0:
            _update_block(pv_slot, st_cur, o_run, 1, 1)
        else:
            _update_block(pv_slot, st_cur, o_run, 1, 0)
    else:
        if j == g:
            _update_block(pv_slot, st_cur, o_run, 0, 1)
        else:
            _update_block(pv_slot, st_cur, o_run, 0, 0)
    if j == g:
        pl.store(
            out, o_run, [b, g * BQ + pl.get_subblock_idx() * HALF, h, 0], order=[1, 3]
        )


def _make_kernel(preload: int, name: str):
    """Build separately named main and serial entries with the same physical plan."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        q: pl.Tensor[[BATCH, SEQ, HEADS, D], FP16],
        k: pl.Tensor[[BATCH, SEQ, HEADS, D], FP16],
        v: pl.Tensor[[BATCH, SEQ, HEADS, D], FP16],
        out: pl.Tensor[[BATCH, SEQ, HEADS, D], FP32],
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
                shape=[BQ, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=Q_SLOTS,
        )
        k_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        v_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x38000,
            mutex_ids="auto",
            depth=KV_SLOTS,
        )
        l0a1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, BQ], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, BQ], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0a2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[BQ, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=2,
        )
        l0b2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, D], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=2,
        )
        acc2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BQ, D], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=2,
        )
        shared_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[BKV, HALF],
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
                shape=[BQ, BKV],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=0x60000,
            mutex_ids="auto",
            depth=P_SLOTS,
            fwd_ids=[4, 5, 6, 7],
            bwd_ids=[8, 9, 10, 11],
        )
        shared_pv = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF, D],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[12, 13],
            bwd_ids=[14, 15],
        )
        ptm_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=2,
        )
        ptb_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x24100,
            mutex_ids="auto",
            depth=2,
        )
        st_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[3, HALF], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x28200,
            mutex_ids="auto",
            depth=ST_SLOTS,
        )
        tri_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF, HALF], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x29100,
            mutex_ids="auto",
            depth=1,
        )
        o_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF, D], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x2D100,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_vector():
            tri0 = tri_g.current()
            _init_tri(tri0)

        # Q 槽组的转置视图：``pl.gemm(trans_b=True)`` 不接受运行期选槽的 tile，视图上的 ``q_zn[s]`` 就是 Q_s^T
        q_zn = pl.reinterpret(q_g, shape=[D, BQ], layout=pl.ZN)
        with pl.section_cube():
            # 核内首个 item 的 Q 与块 0 的 K/V：此后每个 item 的首块都由上一个 item 预取
            first_owner = begin // QB
            pl.load(
                q_g[begin % Q_SLOTS],
                q,
                [first_owner // HEADS, (begin % QB) * BQ, first_owner % HEADS, 0],
                order=[1, 3],
            )
            pl.load(
                k_g[0],
                k,
                [first_owner // HEADS, 0, first_owner % HEADS, 0],
                order=[1, 3],
            )
            pl.load(
                v_g[0],
                v,
                [first_owner // HEADS, 0, first_owner % HEADS, 0],
                order=[1, 3],
            )
        tick = 0
        for it in pl.range(begin, end):
            owner = it // QB
            g = it % QB
            b = owner // HEADS
            h = owner % HEADS
            fo = 0
            if g == 0:
                fo = 1
            if it == begin:
                fo = 1
            with pl.section_cube():
                # 下一个 item 的 Q 预取：放在循环体顶部（先于本 item 首个 qk stage 的阻塞式跨核 wait 发射），
                # 排在本 item 的 K/V 预取之前——搬进 stage 里发射会晚到无法覆盖
                if it + 1 < end:
                    nxt = it + 1
                    n_owner = nxt // QB
                    pl.load(
                        q_g[nxt % Q_SLOTS],
                        q,
                        [n_owner // HEADS, (nxt % QB) * BQ, n_owner % HEADS, 0],
                        order=[1, 3],
                    )
            has_next = 0
            if it + 1 < end:
                has_next = 1
            qs = it % Q_SLOTS
            for j in pl.range(g + 1):
                with pl.section_cube():
                    qk_block(
                        j,
                        g,
                        fo,
                        has_next,
                        b,
                        h,
                        qs,
                        k,
                        q_zn,
                        k_g,
                        l0a1,
                        l0b1,
                        acc1_g,
                        shared_s,
                    )
                with pl.section_vector():
                    softmax_block(
                        j, g, tick, shared_s, st_g, tri_g, ptm_g, ptb_g, shared_p
                    )
                with pl.section_cube():
                    pv_block(
                        j,
                        g,
                        fo,
                        has_next,
                        b,
                        h,
                        v,
                        v_g,
                        l0a2,
                        l0b2,
                        acc2_g,
                        shared_p,
                        shared_pv,
                    )
                with pl.section_vector():
                    update_block(j, g, b, h, tick, out, shared_pv, st_g, o_g)
                tick = tick + 1

    return kernel


flash_attention_mha = _make_kernel(preload=PRELOAD, name="flash_attention_mha")
flash_attention_mha_p0 = _make_kernel(preload=0, name="flash_attention_mha_p0")
