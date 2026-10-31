# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Single-head attention with online softmax and explicit physical resources.

Contract: O = softmax(Q @ K.T / sqrt(128)) @ V. Q, K and V are contiguous FP16 [1024,128]; O is FP32 [1024,128]. There
is no mask or dropout. Inputs are finite. Scores, running maxima/sums and output accumulation use FP32; unnormalized
probabilities narrow to FP16 before Cube consumes them.

Eight execution groups own disjoint 128-row query ranges; each AIV owns 64 query rows. A loop over eight 128-key blocks
calls four native stages: qk -> softmax -> pv -> update. For each block, m=max(m_old,rowmax(S)),
alpha=exp((m_old-m)/sqrt(128)), P=exp((S-m)/sqrt(128)), l=alpha*l_old+sum(P), and O=alpha*O_old+fp16(P)@V. Finalization
divides O by the unquantized FP32 l. No full score row or complete K/V matrix is materialized locally.

The score layout is transposed: S.T has key rows and query lanes. Two [65,64] FP16 NZ tiles pack the two key halves with
64 valid rows each. The shared P group is ZN [query,key], sharing the transposed NZ byte map. insert offsets [0,off] and
[KH,off], with off=subblock*64, use the NZ source coordinates.

Explicit byte ranges (ends exclusive):

| Space | Groups and ranges | Bytes/slot x depth |
| --- | --- | --- |
| Mat | q_g [0,0x8000), k_g [0x8000,0x10000) | 32768 x 1 each |
| Mat | v_g [0x10000,0x20000), shared_p [0x20000,0x28000) | 32768 x 2 / 1 |
| Left | l0a1 [0,0x8000), l0a2 [0x8000,0x10000) | 16384 x 2 each |
| Right | l0b1 [0,0x8000), l0b2 [0x8000,0x10000) | 16384 x 2 each |
| Acc | acc1_g [0,0x20000), acc2_g [0x20000,0x40000) | 65536 x 2 each |
| Vec | shared_s [0,0x10000), shared_pv [0x10000,0x20000) | 32768 x 2 each |
| Vec | ptm_g [0x20000,0x24100), ptb_g [0x24100,0x28200) | 8320 x 2 each |
| Vec | m_g [0x28200,0x28300), l_g [0x28300,0x28400) | 256 x 1 each |
| Vec | alpha_g [0x28400,0x28600), o_g [0x28600,0x30600) | 256 x 2 / 32768 x 1 |

All groups use automatic intra-core mutex IDs. shared_s uses forward [0,1] and reverse [2,3]; shared_pv uses forward
[6,7] and reverse [8,9]. Both are FIX -> V with V -> FIX release, and their VF consumers leave the shared inputs
unchanged. shared_p uses forward [4] (MTE3 -> MTE1) and reverse [5] (MTE1 -> MTE3).

The main entry uses PRELOAD=1, with stage delays 0/1/1/2. Score and product handovers need two slots; P needs one. The
same-core alpha ring needs PRELOAD+1 slots because softmax writes it before update consumes it. m/l/O retain their
separate persistent slots. The preload=0 control keeps the same allocation. Addresses and event lists describe these
fixed constants and must be replanned when geometry or depths change.

Mat uses 160 KiB and Vec uses 198144 bytes; Left/Right each use their full 64 KiB, and Acc uses its full 256 KiB. Tests
compare with an independent FP64 mathematical reference under the input-dependent error budget and assumptions in
reference.error_terms. They also cover extrema across blocks, uniform/one-hot inputs, partial launches, zero-tolerance
preload/repeat consistency, static bounded storage and five incorrect mathematical controls. Consistency is not a proof
of race freedom. This example makes no current performance claim.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M, N, D = 1024, 1024, 128  # Q[M,D], K[N,D], V[N,D], O[M,D]
NUM_CORES = 8
LAUNCH_BLOCK_DIM = (
    NUM_CORES  # mix kernel 的方括号数是执行组数（1c+2v/组），不是 AIV 线程数
)
ROWS_PER_CORE = M // NUM_CORES  # 128：一个核的 q 行数
HALF = ROWS_PER_CORE // 2  # 64：一个 AIV 的 q 行数 = 一条 FP32 寄存器的 lane 数
BLOCK_KV = 128  # Keys per block; both L0 pairs and accumulators fill their spaces.
KH = BLOCK_KV // 2  # 64：P^T 的一个 key 半区
NUM_BLOCKS = N // BLOCK_KV  # 8
SK = 64  # splitk 步长（两个 gemm 共用）
VL = 64  # 一条 FP32 寄存器的元素数
DCH = D // VL  # 一行 O 占的寄存器块数 = 2
NZ_PITCH = KH + 1  # 65：UB NZ 物理列高 = rows+1，奇数错开 bank
PRELOAD = 1
ALPHA_SLOTS = (
    PRELOAD + 1
)  # softmax(delay 1) -> update(delay 1+PRELOAD)：同核跨 stage，拍距 PRELOAD
P_SLOTS = (
    max(PRELOAD - 1, 0) + 1
)  # softmax(delay 1) -> pv(delay PRELOAD)：跨核，拍距 PRELOAD-1，槽数 = 拍距 + 1
SCALE = 0.08838834764831845  # 1/sqrt(128)
NEG_BIG = -1.0e30  # running 最大的初值（有限，不用 -inf）


@pl.vector_function
def _init_state(m_tile, l_tile, o_run):
    """``m = NEG_BIG``、``l = 0``、``O = 0``（本 AIV 的 64 个 q 行）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    negbig = vf.full(NEG_BIG, preg, dtype=FP32)
    zero = vf.full(0.0, preg, dtype=FP32)
    vf.store_align(m_tile, negbig, preg, 0)
    vf.store_align(l_tile, zero, preg, 0)
    for r in pl.range(HALF):
        for c in pl.range(DCH):
            vf.store_align(o_run, zero, preg, [r, c * VL])


@pl.vector_function
def _softmax_block(s_src, m_tile, l_tile, alpha_tile, pt_main, pt_back):
    """一个 KV 块的在线 softmax。``s_src``：``[BLOCK_KV, HALF]`` FP32（行 = key，lane = q 行）。

    产出：``m``/``l`` 就地更新；``alpha_tile`` 写本块的 rescale 系数（供 ``_update_block``）；
    ``pt_main`` / ``pt_back`` 分别接 key ``r`` 与 ``r + KH`` 的 P^T（FP16、NZ、row+1 pitch）。

    第一遍：四路累加器取块内最大（四条独立依赖链，掩住 ``vf.max`` 的延迟），再与旧 ``m`` 合并。
    第二遍：``exp((s - m) * c)``、行和累加、窄化写出。窄化配方同 ``mm_softmax_mm``：``astype(..., ZERO)`` 之后
    ``de_interleave`` 拼稠密；两个 key 一组，各自 ``ZERO`` 窄化，半掩码 ``MaskPattern.H`` 的 vsstb 游标散写。
    每个 VF 调用的结果先命名再复用，不嵌套（parser 约束）。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.H, dtype=FP16)
    a0 = vf.load_align(s_src, [0, 0])
    a1 = vf.load_align(s_src, [1, 0])
    a2 = vf.load_align(s_src, [2, 0])
    a3 = vf.load_align(s_src, [3, 0])
    for k in pl.range(1, BLOCK_KV // 4):
        x0 = vf.load_align(s_src, [4 * k, 0])
        x1 = vf.load_align(s_src, [4 * k + 1, 0])
        x2 = vf.load_align(s_src, [4 * k + 2, 0])
        x3 = vf.load_align(s_src, [4 * k + 3, 0])
        a0 = vf.max(a0, x0, preg)
        a1 = vf.max(a1, x1, preg)
        a2 = vf.max(a2, x2, preg)
        a3 = vf.max(a3, x3, preg)
    b0 = vf.max(a0, a1, preg)
    b1 = vf.max(a2, a3, preg)
    mblk = vf.max(b0, b1, preg)
    m_old = vf.load_align(m_tile, 0)
    m_new = vf.max(m_old, mblk, preg)
    dm = vf.sub(m_old, m_new, preg)
    dms = vf.muls(dm, SCALE, preg)
    alpha = vf.exp(dms, preg)
    vf.store_align(m_tile, m_new, preg, 0)
    vf.store_align(alpha_tile, alpha, preg, 0)
    sa = vf.full(0.0, preg, dtype=FP32)
    sb = vf.full(0.0, preg, dtype=FP32)
    for r in pl.range(KH):
        xa = vf.load_align(s_src, [r, 0])
        xb = vf.load_align(s_src, [r + KH, 0])
        da = vf.sub(xa, m_new, preg)
        db = vf.sub(xb, m_new, preg)
        ea = vf.muls(da, SCALE, preg)
        eb = vf.muls(db, SCALE, preg)
        pa = vf.exp(ea, preg)
        pb = vf.exp(eb, preg)
        sa = vf.add(sa, pa, preg)
        sb = vf.add(sb, pb, preg)
        ha = vf.astype(pa, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        hb = vf.astype(pb, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        dna, _a = vf.de_interleave(ha, ha)
        dnb, _b = vf.de_interleave(hb, hb)
        vf.store_align(
            pt_main,
            dna,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )
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
    l_old = vf.load_align(l_tile, 0)
    l_old = vf.mul_dst_add(alpha, stot, preg)  # l = l * alpha + rowsum(P)
    vf.store_align(l_tile, l_old, preg, 0)


@pl.vector_function
def _update_block(pv_src, alpha_tile, o_run):
    """``O = alpha * O + PV``：``alpha`` 逐行 BRC 广播（O 的 lane 是 d 列，alpha 的 lane 是 q 行）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(HALF):
        a = vf.load_align(alpha_tile, r, dist=pl.LoadDist.BRC_B32)
        for c in pl.range(DCH):
            o_row = vf.load_align(o_run, [r, c * VL])
            pv = vf.load_align(pv_src, [r, c * VL])
            o_row = vf.mul_dst_add(a, pv, preg)
            vf.store_align(o_run, o_row, preg, [r, c * VL])


@pl.vector_function
def _finalize(o_run, l_tile):
    """``O = O / l``（就地）。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(HALF):
        lr = vf.load_align(l_tile, r, dist=pl.LoadDist.BRC_B32)
        for c in pl.range(DCH):
            o_row = vf.load_align(o_run, [r, c * VL])
            q = vf.div(o_row, lr, preg)
            vf.store_align(o_run, q, preg, [r, c * VL])


@pl.pipeline.stage
def qk_block(j, k, q_g, k_g, l0a1, l0b1, acc1_g, shared_s):
    k_tile = k_g.next()
    pl.load(k_tile, k, [j * BLOCK_KV, 0])
    q_tile = q_g.current()
    acc1 = acc1_g.next()
    pl.gemm(
        acc1, k_tile, q_tile, l0a=l0a1, l0b=l0b1, trans_b=True, splitk=SK
    )  # S^T = K_j Q^T
    slot = shared_s.next()
    pl.move(slot, acc1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)


@pl.pipeline.stage
def softmax_block(j, shared_s, m_g, l_g, alpha_g, ptm_g, ptb_g, shared_p):
    s_slot = shared_s.next()
    m_tile = m_g.current()
    l_tile = l_g.current()
    alpha = alpha_g[j % ALPHA_SLOTS]
    ptm = ptm_g.next()
    ptb = ptb_g.next()
    _softmax_block(s_slot, m_tile, l_tile, alpha, ptm, ptb)
    pl.set_validshape(ptm, [KH, HALF])
    pl.set_validshape(ptb, [KH, HALF])
    off = pl.get_subblock_idx() * HALF  # 本 AIV 的 q 半区在 NZ 源坐标系的列偏移
    p_tile = shared_p.next()
    pl.insert(p_tile, ptm, [0, off])
    pl.insert(p_tile, ptb, [KH, off])


@pl.pipeline.stage
def pv_block(j, v, v_g, l0a2, l0b2, acc2_g, shared_p, shared_pv):
    v_tile = v_g.next()
    pl.load(v_tile, v, [j * BLOCK_KV, 0])
    p_tile = shared_p.next()
    acc2 = acc2_g.next()
    pl.gemm(acc2, p_tile, v_tile, l0a=l0a2, l0b=l0b2, splitk=SK)  # PV_j = P_j V_j
    slot = shared_pv.next()
    pl.move(slot, acc2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def update_block(j, shared_pv, alpha_g, o_g):
    pv_slot = shared_pv.next()
    alpha = alpha_g[j % ALPHA_SLOTS]
    o_run = o_g.current()
    _update_block(pv_slot, alpha, o_run)


def _make_kernel(preload: int, name: str):
    """Build separately named main and serial entries with the same physical plan."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        q: pl.Tensor[[M, D], FP16],
        k: pl.Tensor[[N, D], FP16],
        v: pl.Tensor[[N, D], FP16],
        out: pl.Tensor[[M, D], FP32],
    ):
        q_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[ROWS_PER_CORE, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        k_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCK_KV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x8000,
            mutex_ids="auto",
            depth=1,
        )
        v_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCK_KV, D], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=2,
        )
        l0a1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCK_KV, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, ROWS_PER_CORE],
                dtype=FP16,
                target_memory=pl.MemorySpace.Right,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCK_KV, ROWS_PER_CORE],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0a2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[ROWS_PER_CORE, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
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
                shape=[ROWS_PER_CORE, D], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=2,
        )
        shared_s = pl.make_tile_group(
            type=pl.TileType(
                shape=[BLOCK_KV, HALF],
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
                shape=[ROWS_PER_CORE, BLOCK_KV],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=P_SLOTS,
            fwd_ids=[4],
            bwd_ids=[5],
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
            fwd_ids=[6, 7],
            bwd_ids=[8, 9],
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
        m_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, HALF], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x28200,
            mutex_ids="auto",
            depth=1,
        )
        l_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, HALF], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x28300,
            mutex_ids="auto",
            depth=1,
        )
        alpha_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, HALF], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x28400,
            mutex_ids="auto",
            depth=ALPHA_SLOTS,
        )
        o_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF, D], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x28600,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_cube():
            core_id = pl.get_block_idx() // pl.get_subblock_num()
            q_tile = q_g.current()
            pl.load(q_tile, q, [core_id * ROWS_PER_CORE, 0])
        with pl.section_vector():
            m_t = m_g.current()
            l_t = l_g.current()
            o_t = o_g.current()
            _init_state(m_t, l_t, o_t)

        for j in pl.range(NUM_BLOCKS):
            with pl.section_cube():
                qk_block(j, k, q_g, k_g, l0a1, l0b1, acc1_g, shared_s)
            with pl.section_vector():
                softmax_block(j, shared_s, m_g, l_g, alpha_g, ptm_g, ptb_g, shared_p)
            with pl.section_cube():
                pv_block(j, v, v_g, l0a2, l0b2, acc2_g, shared_p, shared_pv)
            with pl.section_vector():
                update_block(j, shared_pv, alpha_g, o_g)

        with pl.section_vector():
            core_id = pl.get_block_idx() // pl.get_subblock_num()
            o_t = o_g.current()
            l_t = l_g.current()
            _finalize(o_t, l_t)
            row0 = core_id * ROWS_PER_CORE + pl.get_subblock_idx() * HALF
            pl.store(out, o_t, [row0, 0])

    return kernel


flash_attention = _make_kernel(preload=PRELOAD, name="flash_attention")
flash_attention_p0 = _make_kernel(preload=0, name="flash_attention_p0")
