/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * scatter_simt_common.h — Scatter 族 SIMT 范式代码（Ascend 950 / DAV_3510）
 *
 * 覆盖 [scatter_design.md](../scatter_design.md) 的实现 ①②③④：
 *   ScatterFiberVF    ① 纤维独占（免原子）
 *   ScatterPackArbVF  ② 打包键单趟仲裁
 *   ScatterWinVF / ScatterApplyVF   ③ 两趟胜者
 *   ScatterAtomicVF / ScatterAtomicMul  ④ 直接原子（含 CAS 补原子乘）
 *   scatter_copy_kernel / scatter_cast_kernel / scatter_pack_{init,unpack}_kernel
 *                     阶段一的 SIMD 顺序流式 pass
 *
 * 范式级代码：省略了 dtype × idx 宽度 × reduce 的全量派发宏与 extern "C" launcher 声明，
 * 落地时按自己的工程补。核心的索引算术、仲裁逻辑、同步粒度与实测调优结论都是原样的。
 *
 * 编译要点：含 SIMT VF 的 __global__ 核函数必须显式标 __vector__，否则 ld.lld 报
 * "auto derivate failed"。
 */

#ifndef SCATTER_SIMT_COMMON_H
#define SCATTER_SIMT_COMMON_H

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace scatter_ref {

constexpr int64_t PIPELINE_DEPTH = 2;
constexpr uint32_t SCATTER_THREADS = 1024;    // 实测吞吐 ∝ 总线程数，取 LAUNCH_BOUND 上限
constexpr uint32_t SCATTER_ADD_THREADS = 256; // add 路实测 256 优于 1024（原子冲突重试）
constexpr uint32_t SCATTER_PACK_THREADS = 256;
constexpr uint32_t SCATTER_WIN_THREADS = 256;
constexpr uint32_t SCATTER_CAST_TILE = 16384; // SIMD pass 的 tile 元素数

constexpr int32_t RED_NONE = 0;
constexpr int32_t RED_ADD = 1;
constexpr int32_t RED_MUL = 2;
constexpr int32_t RED_AMIN = 3;
constexpr int32_t RED_AMAX = 4;

// ─────────────────────────── 半精度 <-> float ───────────────────────────
// fp16/bf16 的累积型 reduce 在 float 域做：硬件 asc_atomic_add(__gm__ half*) 存在但每次
// 舍回 fp16，误差与软件版逐位相同，过不了精度。见 ../precision_and_traps.md §1。

template <typename T>
__simt_callee__ __aicore__ inline float ScatterToFloat(T x)
{
    if constexpr (AscendC::IsSameType<T, half>::value)
        return __half2float(x);
    else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value)
        return __bfloat162float(x);
    else
        return static_cast<float>(x);
}

template <typename T>
__simt_callee__ __aicore__ inline T ScatterFromFloat(float x)
{
    if constexpr (AscendC::IsSameType<T, half>::value)
        return __float2half_rn(x);
    else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value)
        return __float2bfloat16_rn(x);
    else
        return static_cast<T>(x);
}

template <typename T>
__simt_callee__ __aicore__ inline constexpr bool ScatterIsHalfLike()
{
    return AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, bfloat16_t>::value;
}

template <typename DST, typename SRC>
__simt_callee__ __aicore__ inline DST ScatterConvert(SRC x)
{
    if constexpr (AscendC::IsSameType<DST, SRC>::value)
        return x;
    else
        return ScatterFromFloat<DST>(ScatterToFloat<SRC>(x));
}

// ─────────────────────────────── reduce 组合 ───────────────────────────────
// include_self 恒为 true（torch scatter_reduce_(..., include_self=True)），不需要 hitCount 簿记。

template <int32_t RED, typename V>
__simt_callee__ __aicore__ inline V ScatterCombineRaw(V a, V b)
{
    if constexpr (RED == RED_ADD)
        return a + b;
    else if constexpr (RED == RED_MUL)
        return a * b;
    else if constexpr (RED == RED_AMIN)
        return (b < a) ? b : a;
    else if constexpr (RED == RED_AMAX)
        return (b > a) ? b : a;
    else
        return b; // RED_NONE：后写覆盖
}

template <int32_t RED, typename T>
__simt_callee__ __aicore__ inline T ScatterCombine(T a, T b)
{
    if constexpr (RED == RED_NONE)
        return b;
    else if constexpr (ScatterIsHalfLike<T>())
        return ScatterFromFloat<T>(ScatterCombineRaw<RED, float>(ScatterToFloat<T>(a), ScatterToFloat<T>(b)));
    else
        return ScatterCombineRaw<RED, T>(a, b);
}

// ══════════════════════ ① 纤维独占（免原子，天然后写覆盖）══════════════════════
// 线程 f 认领第 f 条纤维（非 dim 坐标固定），线程内沿 dim 递增遍历 j。
// 相邻线程吃相邻 inner 坐标 ⇒ 读 indices/updates 完全合并；不规则只体现在写侧。
// 各线程输出集合互不相交 ⇒ 全程无需原子；j 递增 ⇒ none 天然后写覆盖。
// ACC_T = 输出/累加器类型，UPD_T = updates 类型。二者相同即普通路径；
// fp16/bf16 的累积型 reduce 走 ACC_T=float、UPD_T=half/bfloat16_t。
// COMP_T = 索引运算宽度（所有线性下标 < 2^31 时可用 int32_t；本族实测零收益，保留形参以便复用）。

template <typename ACC_T, typename UPD_T, typename IDX, typename COMP_T, int32_t RED, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void ScatterFiberVF(__gm__ ACC_T* y, __gm__ IDX* indices,
                                                                        __gm__ UPD_T* updates, int64_t numFibers,
                                                                        int64_t inner, int64_t sIdx, int64_t sUpd,
                                                                        int64_t sVar)
{
    const COMP_T tid = static_cast<COMP_T>(blockIdx.x) * blockDim.x + threadIdx.x;
    const COMP_T stride = static_cast<COMP_T>(gridDim.x) * blockDim.x;
    const COMP_T cNumFibers = static_cast<COMP_T>(numFibers);
    const COMP_T cInner = static_cast<COMP_T>(inner);
    const COMP_T cSIdx = static_cast<COMP_T>(sIdx);
    const COMP_T cSVar = static_cast<COMP_T>(sVar);
    (void)sUpd; // 快路径约定 updates.shape == indices.shape

    for (COMP_T f = tid; f < cNumFibers; f += stride) {
        const COMP_T o = f / cInner;
        const COMP_T c = f - o * cInner;

        const COMP_T baseIdx = o * cSIdx * cInner + c; // indices / updates 共用
        const COMP_T baseVar = o * cSVar * cInner + c; // 输出 y

        for (COMP_T j = 0; j < cSIdx; ++j) {
            COMP_T ix = static_cast<COMP_T>(indices[baseIdx + j * cInner]);
            if (ix < 0)
                ix += cSVar; // 与 torch 一致地容忍负索引
            if (ix < 0 || ix >= cSVar)
                continue; // 非法索引跳过，不越界写

            const COMP_T dst = baseVar + ix * cInner;
            const ACC_T v = ScatterConvert<ACC_T, UPD_T>(updates[baseIdx + j * cInner]);
            if constexpr (RED == RED_NONE)
                y[dst] = v;
            else
                y[dst] = ScatterCombine<RED, ACC_T>(y[dst], v);
        }
    }
}

// ─────────────── 元素级并行（②③④）共用的坐标分解 ───────────────

#define SCATTER_EP_PROLOGUE() \
    const COMP_T tid = static_cast<COMP_T>(blockIdx.x) * blockDim.x + threadIdx.x; \
    const COMP_T stride = static_cast<COMP_T>(gridDim.x) * blockDim.x; \
    const COMP_T cTotal = static_cast<COMP_T>(total); \
    const COMP_T cOuter = static_cast<COMP_T>(outer); \
    const COMP_T cInner = static_cast<COMP_T>(inner); \
    const COMP_T cSIdx = static_cast<COMP_T>(sIdx); \
    const COMP_T cSVar = static_cast<COMP_T>(sVar); \
    const COMP_T idxPlane = cSIdx * cInner

// i（indices 线性下标）-> (o, j, c) -> dst；outer==1 时省掉一次除法。
// 注意：没有用 quick-div（GetUintDivMagicAndShift + Simt::UintDiv）。实测标量优化在本族
// 零收益，瓶颈在访存延迟不在标量流水。
#define SCATTER_EP_DECOMPOSE() \
    COMP_T o, j, c; \
    if (cOuter == 1) { \
        o = 0; \
        j = i / cInner; \
        c = i - j * cInner; \
    } else { \
        o = i / idxPlane; \
        const COMP_T rem = i - o * idxPlane; \
        j = rem / cInner; \
        c = rem - j * cInner; \
    } \
    COMP_T ix = static_cast<COMP_T>(indices[i]); \
    if (ix < 0) \
        ix += cSVar; \
    if (ix < 0 || ix >= cSVar) \
        continue; \
    const COMP_T dst = o * cSVar * cInner + ix * cInner + c

// ══════════════════ ② 打包键单趟仲裁（none + 2 字节值）══════════════════
//     key = (j + 1) << JSHIFT | bits(value)
// 高位是沿 dim 的下标，atomic_max 选出 j 最大者 = 后写覆盖。init pass 写 j=0，任何更新都能压过。
// 低位为什么不干扰胜负：dst = o*sVar*inner + ix*inner + c 是混合进制表示，(o,ix,c) ↔ dst
// 一一对应 ⇒ 同一 dst 只可能来自同一条纤维，纤维内 j 互不相同 ⇒ 高位永远分得出胜负。
// 只用 32 位键（KEY=uint32_t, VB=uint16_t, JSHIFT=16）：实测 32 位原子 11~18 G/s，
// 64 位原子只有 4.7~5.0 G/s。4 字节值改走 ③ 两趟 32 位反而更快。

template <typename KEY, typename VB, typename IDX, typename COMP_T, uint32_t JSHIFT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_PACK_THREADS) inline void ScatterPackArbVF(
    __gm__ KEY* ws, __gm__ IDX* indices, __gm__ VB* updates, int64_t total, int64_t outer, int64_t inner, int64_t sIdx,
    int64_t sUpd, int64_t sVar)
{
    SCATTER_EP_PROLOGUE();
    (void)sUpd;
    for (COMP_T i = tid; i < cTotal; i += stride) {
        SCATTER_EP_DECOMPOSE();
        const KEY key = (static_cast<KEY>(j + 1) << JSHIFT) | static_cast<KEY>(updates[i]);
        asc_atomic_max(ws + dst, key);
    }
}

// ══════════════════ ③ 两趟胜者（none + 键装不进 32 位）══════════════════
// 趟 1：atomic_max(win[dst], i+1)  —— 只裁决"谁最后写"，不搬数据
// 趟 2：win[dst] == i+1 才普通存储 —— 胜者唯一，无需原子
// win 需预置 0（表示"无人写"，保留 y 里已 clone 的 data）。
// 记 i+1 而不是 j+1：对固定 (o,c,dst)，线性 i 随 j 严格递增，同样能裁决；而且趟 2 可以
// 直接用 updates[winner-1] 取值，不必重算 dst、也不必再读 indices。

__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_THREADS) inline void ScatterFillZeroVF(__gm__ uint32_t* win, int64_t count)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = tid; i < count; i += stride)
        win[i] = 0u;
}

template <typename IDX, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_WIN_THREADS) inline void ScatterWinVF(__gm__ uint32_t* win,
                                                                                  __gm__ IDX* indices, int64_t total,
                                                                                  int64_t outer, int64_t inner,
                                                                                  int64_t sIdx, int64_t sUpd,
                                                                                  int64_t sVar)
{
    SCATTER_EP_PROLOGUE();
    (void)sUpd;
    for (COMP_T i = tid; i < cTotal; i += stride) {
        SCATTER_EP_DECOMPOSE();
        asc_atomic_max(win + dst, static_cast<uint32_t>(i + 1));
    }
}

// 趟 2 有两种遍历方式，host 侧按更新密度选（denseWinResolve = epTotal*4 >= numelData）：
//   密：outer==0 作标记，按 dst 遍历 numelData 个位置，winner!=0 就写 —— 省掉坐标分解
//   疏：按 i 遍历 epTotal 个更新，重算 dst 后比对 —— 避免扫过大量空位置
template <typename T, typename IDX, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCATTER_THREADS) inline void ScatterApplyVF(__gm__ T* y, __gm__ uint32_t* win,
                                                                                __gm__ IDX* indices, __gm__ T* updates,
                                                                                int64_t total, int64_t outer,
                                                                                int64_t inner, int64_t sIdx,
                                                                                int64_t sUpd, int64_t sVar)
{
    (void)indices;
    if (outer == 0) { // 稠密：按 dst 直接扫
        const COMP_T tid = static_cast<COMP_T>(blockIdx.x) * blockDim.x + threadIdx.x;
        const COMP_T stride = static_cast<COMP_T>(gridDim.x) * blockDim.x;
        const COMP_T cTotal = static_cast<COMP_T>(total);
        for (COMP_T d = tid; d < cTotal; d += stride) {
            const uint32_t winner = win[d];
            if (winner != 0)
                y[d] = updates[winner - 1];
        }
        return;
    }

    SCATTER_EP_PROLOGUE();
    (void)sUpd;
    for (COMP_T i = tid; i < cTotal; i += stride) {
        SCATTER_EP_DECOMPOSE();
        if (win[dst] == static_cast<uint32_t>(i + 1))
            y[dst] = updates[i];
    }
}

// ══════════════ ④ 直接原子（add / amin / amax / multiply）══════════════
// 上游须确认精度与确定性契约允许原子执行顺序；浮点 add/multiply 的舍入结果可能变化。
// 契约允许时无需裁决"谁最后写"，可直接铺满线程、冲突交给硬件原子。
// 原子比普通存储慢 2~5 倍，但比线程饥饿划算得多 —— 前提是纤维路径本来就铺不满核。

// 硬件没有原子乘 —— 用 CAS 循环补。随机索引下同一输出位置平均只被更新 ~1 次，竞争很轻，
// 循环几乎总是一次成功。实测把 multiply 踢出本路径退回纤维路径会从 3767us 劣化到 12602us。
// 但冲突密集时 CAS 会退化：判据应该是冲突度，不是"是不是 multiply"。
template <typename ACC_T>
__simt_callee__ __aicore__ inline void ScatterAtomicMul(__gm__ ACC_T* addr, ACC_T v)
{
    if constexpr (AscendC::IsSameType<ACC_T, float>::value) {
        __gm__ uint32_t* p = (__gm__ uint32_t*)addr;
        uint32_t assumed = *p;
        uint32_t old;
        do {
            old = assumed;
            const uint32_t desired = __float_as_uint(__uint_as_float(old) * v);
            assumed = asc_atomic_cas(p, old, desired);
        } while (assumed != old);
    } else {
        // int32 / int64 直接 CAS。half/bf16 没有标量 CAS（只有 half2），调用侧用
        // if constexpr (!ScatterIsHalfLike<ACC_T>()) 挡掉实例化，让它们走 fp32 累加器。
        ACC_T assumed = *addr;
        ACC_T old;
        do {
            old = assumed;
            assumed = asc_atomic_cas(addr, old, static_cast<ACC_T>(old * v));
        } while (assumed != old);
    }
}

template <typename ACC_T, typename UPD_T, typename IDX, int32_t RED, typename COMP_T, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void ScatterAtomicVF(__gm__ ACC_T* y, __gm__ IDX* indices,
                                                                         __gm__ UPD_T* updates, int64_t total,
                                                                         int64_t outer, int64_t inner, int64_t sIdx,
                                                                         int64_t sUpd, int64_t sVar)
{
    SCATTER_EP_PROLOGUE();
    (void)sUpd;
    for (COMP_T i = tid; i < cTotal; i += stride) {
        SCATTER_EP_DECOMPOSE();
        const ACC_T v = ScatterConvert<ACC_T, UPD_T>(updates[i]);
        if constexpr (RED == RED_ADD)
            asc_atomic_add(y + dst, v);
        else if constexpr (RED == RED_AMIN)
            asc_atomic_min(y + dst, v);
        else if constexpr (RED == RED_MUL)
            ScatterAtomicMul<ACC_T>(y + dst, v);
        else
            asc_atomic_max(y + dst, v);
    }
}

#undef SCATTER_EP_DECOMPOSE
#undef SCATTER_EP_PROLOGUE

// ══════════════ 阶段一：SIMD 顺序流式 pass（clone / cast / pack init+unpack）══════════════
// 这些都是规则的连续流量，必须走 SIMD 的 MTE 搬运 + Vector 指令。用 SIMT 标量循环做
// 实测慢一个数量级（某用例 1565us -> 89us）。

// y = data 整块拷贝。按定宽整型实例化（U = uint16_t / uint32_t），int64 按 2× 元素数走 4 字节版。
template <typename U>
__global__ __aicore__ __vector__ void scatter_copy_kernel(GM_ADDR src, GM_ADDR dst, int64_t totalLength,
                                                          int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> queue;
    pipe.InitBuffer(queue, PIPELINE_DEPTH, tileSize);

    const int64_t blockOffset = blockLength * AscendC::GetBlockIdx();
    int64_t remain = totalLength - blockOffset;
    if (remain <= 0)
        return;
    if (remain > blockLength)
        remain = blockLength;

    AscendC::GlobalTensor<U> srcGm, dstGm;
    srcGm.SetGlobalBuffer((__gm__ U*)src + blockOffset);
    dstGm.SetGlobalBuffer((__gm__ U*)dst + blockOffset);

    const int64_t elemsPerTile = static_cast<int64_t>(tileSize / sizeof(U));
    for (int64_t off = 0; off < remain; off += elemsPerTile) {
        const int64_t n = (remain - off < elemsPerTile) ? (remain - off) : elemsPerTile;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(n * sizeof(U)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<U> padParams{false, 0, 0, 0};

        auto local = queue.AllocTensor<U>();
        AscendC::DataCopyPad(local, srcGm[off], copyParams, padParams);
        queue.EnQue(local);
        local = queue.DeQue<U>();
        AscendC::DataCopyPad(dstGm[off], local, copyParams);
        queue.FreeTensor(local);
    }
}

// fp32 累加器的进出转换。bf16 必须 CAST_RINT（round-to-nearest-even）与 torch 对齐，
// CAST_ROUND 是 round-half-away-from-zero，会在 tie 上差 1 ULP 而挂精度。
template <typename DST, typename SRC>
__global__ __aicore__ __vector__ void scatter_cast_kernel(GM_ADDR dst, GM_ADDR src, int64_t count)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueue, 1, SCATTER_CAST_TILE * sizeof(SRC));
    pipe.InitBuffer(outQueue, 1, SCATTER_CAST_TILE * sizeof(DST));

    const int64_t blocks = AscendC::GetBlockNum();
    const int64_t blockLength = (count + blocks - 1) / blocks;
    const int64_t begin = blockLength * AscendC::GetBlockIdx();
    int64_t remain = count - begin;
    if (remain <= 0)
        return;
    if (remain > blockLength)
        remain = blockLength;

    AscendC::GlobalTensor<SRC> srcGm;
    AscendC::GlobalTensor<DST> dstGm;
    srcGm.SetGlobalBuffer((__gm__ SRC*)src + begin);
    dstGm.SetGlobalBuffer((__gm__ DST*)dst + begin);

    for (int64_t off = 0; off < remain; off += SCATTER_CAST_TILE) {
        const uint32_t n =
            static_cast<uint32_t>((remain - off < SCATTER_CAST_TILE) ? (remain - off) : SCATTER_CAST_TILE);
        AscendC::DataCopyExtParams inParams{1, static_cast<uint32_t>(n * sizeof(SRC)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<SRC> padParams{false, 0, 0, 0};

        auto inLocal = inQueue.AllocTensor<SRC>();
        AscendC::DataCopyPad(inLocal, srcGm[off], inParams, padParams);
        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<SRC>();

        auto outLocal = outQueue.AllocTensor<DST>();
        if constexpr (AscendC::IsSameType<DST, bfloat16_t>::value) {
            AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_RINT, n);
        } else {
            AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_NONE, n);
        }
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(inLocal);

        outLocal = outQueue.DeQue<DST>();
        AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(n * sizeof(DST)), 0, 0, 0};
        AscendC::DataCopyPad(dstGm[off], outLocal, outParams);
        outQueue.FreeTensor(outLocal);
    }
}

// ② 的 unpack：y[i] = ws[i] & 值掩码。掩码用 Duplicate 铺一次常量再 And。
// ⚠ 这一趟省不掉：曾用 as_strided(stride×2) 把 ws 低半字当输出返回，本地全绿、
// 平台因缺 StridedSlice 二进制直接崩。输出必须是连续张量（../precision_and_traps.md §3）。
template <typename KEY, typename VB, uint32_t JSHIFT>
__global__ __aicore__ __vector__ void scatter_pack_unpack_kernel(GM_ADDR y, GM_ADDR ws, int64_t count)
{
    constexpr uint32_t TILE = SCATTER_CAST_TILE;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;
    pipe.InitBuffer(inQueue, 1, TILE * sizeof(KEY));
    pipe.InitBuffer(outQueue, 1, TILE * sizeof(VB));
    pipe.InitBuffer(maskBuf, TILE * sizeof(KEY));

    auto maskLocal = maskBuf.Get<KEY>();
    AscendC::Duplicate(maskLocal, static_cast<KEY>((KEY{1} << JSHIFT) - 1), TILE);
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t blocks = AscendC::GetBlockNum();
    const int64_t blockLength = (count + blocks - 1) / blocks;
    const int64_t begin = blockLength * AscendC::GetBlockIdx();
    int64_t remain = count - begin;
    if (remain <= 0)
        return;
    if (remain > blockLength)
        remain = blockLength;

    AscendC::GlobalTensor<KEY> wsGm;
    AscendC::GlobalTensor<VB> yGm;
    wsGm.SetGlobalBuffer((__gm__ KEY*)ws + begin);
    yGm.SetGlobalBuffer((__gm__ VB*)y + begin);

    for (int64_t off = 0; off < remain; off += TILE) {
        const uint32_t n = static_cast<uint32_t>((remain - off < TILE) ? (remain - off) : TILE);
        AscendC::DataCopyExtParams inParams{1, static_cast<uint32_t>(n * sizeof(KEY)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<KEY> padParams{false, 0, 0, 0};

        auto inLocal = inQueue.AllocTensor<KEY>();
        AscendC::DataCopyPad(inLocal, wsGm[off], inParams, padParams);
        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<KEY>();

        AscendC::And(inLocal, inLocal, maskLocal, static_cast<int32_t>(n));
        AscendC::PipeBarrier<PIPE_V>();

        auto outLocal = outQueue.AllocTensor<VB>();
        AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_NONE, n);
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(inLocal);

        outLocal = outQueue.DeQue<VB>();
        AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(n * sizeof(VB)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], outLocal, outParams);
        outQueue.FreeTensor(outLocal);
    }
}

// ══════════════════════════ kernel 入口（派发骨架）══════════════════════════
// reduce 与线程数都是运行期标量，在核内 switch 分发到编译期特化的 VF：
//   · VF 内部保持 if constexpr 无分支（见 ../../simt/optimization-guide.md Rule 001）
//   · 又不会让 dtype × idx 宽度 × reduce × 线程数 爆出几十份 kernel 二进制
// LAUNCH_BOUND / dim3 要求编译期常量，所以线程数必须按候选值展开。

template <typename ACC_T, typename UPD_T, typename IDX>
__global__ __aicore__ __vector__ void scatter_kernel(GM_ADDR y, GM_ADDR indices, GM_ADDR updates, int64_t numFibers,
                                                     int64_t inner, int64_t sIdx, int64_t sUpd, int64_t sVar,
                                                     int32_t red, int32_t threads, int32_t narrowIdx)
{
    __gm__ ACC_T* yGm = (__gm__ ACC_T*)y;
    __gm__ IDX* idxGm = (__gm__ IDX*)indices;
    __gm__ UPD_T* updGm = (__gm__ UPD_T*)updates;

#define SCATTER_VF_CALL(RED_TAG, NTHREAD, CTYPE) \
    asc_vf_call<ScatterFiberVF<ACC_T, UPD_T, IDX, CTYPE, RED_TAG, NTHREAD>>(dim3(NTHREAD), yGm, idxGm, updGm, \
                                                                            numFibers, inner, sIdx, sUpd, sVar)

#define SCATTER_RED_SWITCH(NTHREAD, CTYPE) \
    do { \
        switch (red) { \
            case RED_ADD: \
                SCATTER_VF_CALL(RED_ADD, NTHREAD, CTYPE); \
                break; \
            case RED_MUL: \
                SCATTER_VF_CALL(RED_MUL, NTHREAD, CTYPE); \
                break; \
            case RED_AMIN: \
                SCATTER_VF_CALL(RED_AMIN, NTHREAD, CTYPE); \
                break; \
            case RED_AMAX: \
                SCATTER_VF_CALL(RED_AMAX, NTHREAD, CTYPE); \
                break; \
            default: \
                SCATTER_VF_CALL(RED_NONE, NTHREAD, CTYPE); \
                break; \
        } \
    } while (0)

#define SCATTER_THREAD_SWITCH(CTYPE) \
    do { \
        switch (threads) { \
            case 1024: \
                SCATTER_RED_SWITCH(1024, CTYPE); \
                break; \
            case 512: \
                SCATTER_RED_SWITCH(512, CTYPE); \
                break; \
            case 128: \
                SCATTER_RED_SWITCH(128, CTYPE); \
                break; \
            case 32: \
                SCATTER_RED_SWITCH(32, CTYPE); \
                break; \
            default: \
                SCATTER_RED_SWITCH(8, CTYPE); \
                break; \
        } \
    } while (0)

    if (narrowIdx != 0)
        SCATTER_THREAD_SWITCH(int32_t);
    else
        SCATTER_THREAD_SWITCH(int64_t);

#undef SCATTER_THREAD_SWITCH
#undef SCATTER_RED_SWITCH
#undef SCATTER_VF_CALL
}

} // namespace scatter_ref

#endif // SCATTER_SIMT_COMMON_H
