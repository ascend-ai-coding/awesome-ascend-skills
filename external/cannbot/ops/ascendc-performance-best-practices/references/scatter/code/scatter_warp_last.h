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
 * scatter_warp_last.h — 实现⑤ warp 段末写回（Ascend 950 / DAV_3510）
 *
 * 见 [warp_last_design.md](../warp_last_design.md)。
 *
 * 一条 warp 独占一条纤维，每轮并行读连续 32 个 j：把 (index, j) 压成 32 位键做 15 级
 * bitonic sort，同 index 的段末者就是本轮 j 最大者，只有该 lane 写回。各轮按 j 递增执行
 * 并在轮间做可见性同步，所以后轮覆盖前轮，严格等价于 CPU 的后写覆盖。
 *
 * 这是 reduce=none 里唯一既单趟、又免原子、又免 workspace 的实现，但适用面最窄：
 *   red==none && extentsMatch && starved && jBits+idxBits<=31 && esize==4 && numFibers>=1024
 * 门槛 1024 是实测定的，放宽到 256 会明显更慢（见设计文档 §5）。
 */

#ifndef SCATTER_WARP_LAST_H
#define SCATTER_WARP_LAST_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace scatter_ref {

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t WARP_THREADS = 1024;
constexpr uint32_t WARPS_PER_BLOCK = WARP_THREADS / WARP_SIZE;

// jBits 由 host 侧按 sIdx 算出：bitWidth(sIdx)，且 bitWidth(sIdx) + bitWidth(sVar) <= 31
// （预留 0xffffffff 作无效 lane 哨兵）。
template <typename T, typename IDX>
__simt_vf__ __aicore__ LAUNCH_BOUND(WARP_THREADS) inline void ScatterWarpLastVF(__gm__ T* y, __gm__ IDX* indices,
                                                                                __gm__ T* updates, int64_t numFibers,
                                                                                int64_t inner, int64_t sIdx,
                                                                                int64_t sUpd, int64_t sVar,
                                                                                int32_t jBits)
{
    (void)sUpd; // 快路径约定 updates.shape == indices.shape

    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warpInBlock = threadIdx.x / WARP_SIZE;
    const int64_t warpId = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warpInBlock;
    const int64_t warpStride = static_cast<int64_t>(gridDim.x) * WARPS_PER_BLOCK;
    const int64_t fiberRounds = (numFibers + warpStride - 1) / warpStride;

    for (int64_t round = 0; round < fiberRounds; ++round) {
        const int64_t f = warpId + round * warpStride;
        const bool activeFiber = f < numFibers;

        int64_t baseIdx = 0;
        int64_t baseUpd = 0;
        int64_t baseVar = 0;
        if (activeFiber) {
            const int64_t o = f / inner;
            const int64_t c = f - o * inner;
            baseIdx = o * sIdx * inner + c;
            baseUpd = baseIdx;
            baseVar = o * sVar * inner + c;
        }

        for (int64_t chunk = 0; chunk < sIdx; chunk += WARP_SIZE) {
            const int64_t originalJ = chunk + lane;

            // 无效 lane（纤维越界 / j 越界 / index 非法）统一用哨兵，不用分支 —— warp 内
            // 所有 lane 走同一条指令路径，无 divergence。哨兵是最大值，排序后被推到高位端。
            uint32_t safeIx = 0xffffffffu;
            if (activeFiber && originalJ < sIdx) {
                int64_t ix = static_cast<int64_t>(indices[baseIdx + originalJ * inner]);
                if (ix < 0)
                    ix += sVar;
                if (ix >= 0 && ix < sVar)
                    safeIx = static_cast<uint32_t>(ix);
            }

            uint32_t key = 0xffffffffu;
            if (safeIx != 0xffffffffu) {
                key = (safeIx << jBits) | static_cast<uint32_t>(originalJ);
            }

            // 15 级 bitonic 网络（log2(32)=5 个阶段，第 k 阶段 k 步）。MASK / WIDTH 必须是
            // 编译期常量，所以完全展开成直线代码。
#define SCATTER_BITONIC_STEP(MASK, WIDTH) \
    do { \
        const uint32_t other = asc_shfl_xor(key, MASK, WARP_SIZE); \
        const bool ascending = (lane & WIDTH) == 0; \
        const bool lowPartner = (lane & MASK) == 0; \
        const bool wantMin = lowPartner == ascending; \
        if ((wantMin && key > other) || (!wantMin && key < other)) \
            key = other; \
    } while (0)

            SCATTER_BITONIC_STEP(1, 2);
            SCATTER_BITONIC_STEP(2, 4);
            SCATTER_BITONIC_STEP(1, 4);
            SCATTER_BITONIC_STEP(4, 8);
            SCATTER_BITONIC_STEP(2, 8);
            SCATTER_BITONIC_STEP(1, 8);
            SCATTER_BITONIC_STEP(8, 16);
            SCATTER_BITONIC_STEP(4, 16);
            SCATTER_BITONIC_STEP(2, 16);
            SCATTER_BITONIC_STEP(1, 16);
            SCATTER_BITONIC_STEP(16, 32);
            SCATTER_BITONIC_STEP(8, 32);
            SCATTER_BITONIC_STEP(4, 32);
            SCATTER_BITONIC_STEP(2, 32);
            SCATTER_BITONIC_STEP(1, 32);
#undef SCATTER_BITONIC_STEP

            // 键高位是 index、低位是 j，升序排完后同一 index 的一段里 j 递增 ⇒ 段末即胜者。
            const uint32_t sortedIx = key >> jBits;
            const uint32_t nextIx = asc_shfl_down(key, 1, WARP_SIZE) >> jBits;
            const bool groupEnd = (lane == WARP_SIZE - 1) || (sortedIx != nextIx);

            if (activeFiber && groupEnd && sortedIx < static_cast<uint32_t>(sVar)) {
                const uint32_t jMask = (uint32_t{1} << jBits) - 1;
                const uint32_t winnerJ = key & jMask;
                y[baseVar + static_cast<int64_t>(sortedIx) * inner] =
                    updates[baseUpd + static_cast<int64_t>(winnerJ) * inner];
            }

            // 同一个输出位置在相邻 chunk 中可能由不同 lane 写，所以轮间要同步；但 warp 内
            // 本就锁步执行，只需保证本 warp 的 GM 写可见，不必让 1024 个线程互等。
            // 不同 warp 处理不同纤维，天然无交叉。
            asc_threadfence_block();
        }
    }
}

// 含 SIMT VF 的 __global__ 核函数必须显式标 __vector__，否则 ld.lld 报 auto derivate failed。
template <typename T, typename IDX>
__global__ __aicore__ __vector__ void scatter_warp_last_kernel(GM_ADDR y, GM_ADDR indices, GM_ADDR updates,
                                                               int64_t numFibers, int64_t inner, int64_t sIdx,
                                                               int64_t sUpd, int64_t sVar, int32_t jBits)
{
    asc_vf_call<ScatterWarpLastVF<T, IDX>>(dim3(WARP_THREADS), (__gm__ T*)y, (__gm__ IDX*)indices, (__gm__ T*)updates,
                                           numFibers, inner, sIdx, sUpd, sVar, jBits);
}

} // namespace scatter_ref

#endif // SCATTER_WARP_LAST_H
