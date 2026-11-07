/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this software except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file online_softmax_cross_core_epilogue.h
 * \brief Phase-2 epilogue: cross-core reduction + final softmax rescale (regbase).
 *
 * Consumes the GM workspace produced by OnlineSoftmaxPerTileEpilogue:
 *   onlineMax[cubeCoreNum * M]   — per-core per-row online max
 *   onlineSum[cubeCoreNum * M]   — per-core per-row online sum
 *   mHistory[M * numTiles]      — per-row per-tile max (for rescale)
 *   expWorkspace[M * expWsPitch] — per-row per-col exp(L - tileMax) (float, 32B aligned)
 *
 * Outputs:
 *   softmaxOut[M * N]            — final softmax result (float32)
 *
 * Two internal phases:
 *   Phase 1: ComputeMaxSum — load allMax/allSum, cross-core reduce maxFinal/sumFinal
 *   Phase 2: N-tile ping-pong rescale — load expTile per tile, rescale, MTE3 writeback
 *
 * N-tile ping-pong: dual expTile buffer, bufId = t & 1.
 *   MTE3 completion sets MTE3_MTE2(bufId) to release buffer for next MTE2 round.
 *   t=0,1 buffers are initially free, skip wait.
 */
#ifndef ONLINE_SOFTMAX_CROSS_CORE_EPILOGUE_H
#define ONLINE_SOFTMAX_CROSS_CORE_EPILOGUE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tensor_api/tensor.h"
#include "blaze_custom/utils/common_utils.h"

namespace Epilogue {

template <typename OutT>
class OnlineSoftmaxCrossCoreEpilogue {
public:
    using ComputeType = float;
    using OutputType = OutT;

    struct Params {
        GM_ADDR onlineMaxAddr{nullptr};
        GM_ADDR onlineSumAddr{nullptr};
        GM_ADDR mHistoryAddr{nullptr};
        GM_ADDR softmaxOutAddr{nullptr};
        GM_ADDR expWorkspaceAddr{nullptr};
        uint32_t cubeCoreNum{0};
        uint32_t vecCoreNum{0};
        uint32_t M{0};
        uint32_t N{0};
        uint32_t baseN{0};
    };

    __aicore__ inline void Init(const Params& p)
    {
        cubeCoreNum_ = p.cubeCoreNum;
        M_ = p.M;
        N_ = p.N;
        baseN_ = p.baseN;
        numTiles_ = (p.N + p.baseN - 1) / p.baseN;
        expWsPitch_ = ((p.N + ELM_PER_32B - 1) / ELM_PER_32B) * ELM_PER_32B;
        nAlignTile_ = ((baseN_ + ELM_PER_32B - 1) / ELM_PER_32B) * ELM_PER_32B;

        uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
        uint32_t perCore = p.M / p.vecCoreNum;
        uint32_t tail = p.M % p.vecCoreNum;
        myRows_ = perCore + (blk < tail ? 1 : 0);
        mStart_ = (blk < tail) ? (blk * (perCore + 1)) : (blk * perCore + tail);

        // UB layout (5 buffers, each 32B aligned padding):
        //   allMaxCore : cubeCoreNum * maxLoopNum_
        //   allSumCore : cubeCoreNum * maxLoopNum_
        //   mHist      : maxLoopNum_ * numTiles_
        //   expTile[0] : maxLoopNum_ * nAlignTile_ (ping)
        //   expTile[1] : maxLoopNum_ * nAlignTile_ (pong)
        maxLoopNum_ = (UB_SIZE - DATA_BLOCK * 5) /
                      (sizeof(ComputeType) * (2 * cubeCoreNum_ + numTiles_ + 2 * nAlignTile_));

        int64_t u = 0;
        ubOffMaxCore_  = u;
        u += CeilAlign(maxLoopNum_ * cubeCoreNum_ * sizeof(ComputeType), DATA_BLOCK);
        ubOffSumCore_  = u;
        u += CeilAlign(maxLoopNum_ * cubeCoreNum_ * sizeof(ComputeType), DATA_BLOCK);
        ubOffMHist_    = u;
        u += CeilAlign(maxLoopNum_ * numTiles_ * sizeof(ComputeType), DATA_BLOCK);
        ubOffExpTile_[0] = u;
        u += CeilAlign(maxLoopNum_ * nAlignTile_ * sizeof(ComputeType), DATA_BLOCK);
        ubOffExpTile_[1] = u;

        gmOnlineMax_    = reinterpret_cast<__gm__ ComputeType*>(p.onlineMaxAddr);
        gmOnlineSum_    = reinterpret_cast<__gm__ ComputeType*>(p.onlineSumAddr);
        gmMHistory_     = reinterpret_cast<__gm__ ComputeType*>(p.mHistoryAddr);
        gmExpWorkspace_ = reinterpret_cast<__gm__ ComputeType*>(p.expWorkspaceAddr);
        gmSoftmaxOut_   = reinterpret_cast<__gm__ OutputType*>(p.softmaxOutAddr);
    }

    __aicore__ inline void ReduceAll()
    {
        if (myRows_ == 0) { return; }

        uint32_t done = 0;
        while (done < myRows_) {
            uint32_t cur = myRows_ - done;
            if (cur > maxLoopNum_) { cur = maxLoopNum_; }
            uint32_t rowStart = mStart_ + done;
            ProcessBatch(rowStart, cur);
            done += cur;
        }
    }

    __aicore__ ~OnlineSoftmaxCrossCoreEpilogue() {}

private:
    static constexpr int64_t DATA_BLOCK = 32;
    static constexpr uint64_t UB_SIZE = 248 * 1024;
    static constexpr uint32_t ELM_PER_32B = DATA_BLOCK / sizeof(ComputeType);
    static constexpr uint32_t PHASE_1_FLAG_MAX_SUM = 2;
    static constexpr uint32_t PHASE_1_FLAG_MHIST = 3;

    static constexpr Reg::DivSpecificMode divMode = {
        Reg::MaskMergeMode::ZEROING, false};

    __gm__ ComputeType* gmOnlineMax_{nullptr};
    __gm__ ComputeType* gmOnlineSum_{nullptr};
    __gm__ ComputeType* gmMHistory_{nullptr};
    __gm__ ComputeType* gmExpWorkspace_{nullptr};
    __gm__ OutputType*  gmSoftmaxOut_{nullptr};

    uint32_t cubeCoreNum_{0};
    uint32_t maxLoopNum_{0};
    uint32_t M_{0};
    uint32_t N_{0};
    uint32_t baseN_{0};
    uint32_t numTiles_{0};
    uint32_t expWsPitch_{0};
    uint32_t myRows_{0};
    uint32_t mStart_{0};
    uint32_t nAlignTile_{0};

    int64_t ubOffMaxCore_{0};
    int64_t ubOffSumCore_{0};
    int64_t ubOffMHist_{0};
    int64_t ubOffExpTile_[2]{0, 0};

    template <typename T>
    __aicore__ inline static __ubuf__ T* UbAddr(int64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    template <typename T>
    __aicore__ inline static auto MakeContig1D(int64_t count)
    {
        using MakeLayout1d = Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>;
        return MakeLayout1d{}(1UL, count);
    }

    template <typename T>
    __aicore__ inline static auto MakeStrided2D(int64_t rows, int64_t cols, int64_t rowPitch)
    {
        auto shape = Te::MakeShape(
            Te::MakeShape(Std::Int<1>{}, rows),
            Te::MakeShape(Std::Int<1>{}, cols));
        auto stride = Te::MakeStride(
            Te::MakeStride(Std::Int<0>{}, rowPitch),
            Te::MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        return Te::MakePatternLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(shape, stride);
    }

    __aicore__ inline void CopyGmToUb1D(int64_t ubOff, __gm__ ComputeType* gmPtr, int64_t count)
    {
        auto layout = MakeContig1D<ComputeType>(count);
        auto ubT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOff), layout);
        auto gmT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmPtr), layout);
        Te::Copy(Te::MakeCopy(Te::CopyGM2UB{}), ubT, gmT);
    }

    __aicore__ inline void CopyExpTileToUb(
        int64_t ubOff, __gm__ ComputeType* gmPtr,
        uint32_t rows, uint32_t cols, uint32_t ubPitch, uint32_t gmPitch)
    {
        auto ubLayout = MakeStrided2D<ComputeType>(rows, cols, ubPitch);
        auto gmLayout = MakeStrided2D<ComputeType>(rows, cols, gmPitch);
        auto ubT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOff), ubLayout);
        auto gmT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmPtr), gmLayout);
        Te::Copy(Te::MakeCopy(Te::CopyGM2UB{}), ubT, gmT);
    }

    __aicore__ inline void CopyUbToGm2D(
        int64_t ubOff, __gm__ OutputType* gmPtr,
        uint32_t rows, uint32_t cols, uint32_t ubPitch, uint32_t gmPitch)
    {
        auto ubLayout = MakeStrided2D<OutputType>(rows, cols, ubPitch);
        auto gmLayout = MakeStrided2D<OutputType>(rows, cols, gmPitch);
        auto ubT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, OutputType>(ubOff), ubLayout);
        auto gmT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmPtr), gmLayout);
        Te::Copy(Te::MakeCopy(Te::CopyUB2GM{}), gmT, ubT);
    }

    __aicore__ inline void ProcessBatch(uint32_t rowStart, uint32_t cur)
    {
        constexpr uint32_t ELM_PER_32B = DATA_BLOCK / sizeof(ComputeType);

        // Phase 1: load allMax/allSum/mHist, V computes maxFinal/sumFinal
        CopyExpTileToUb(ubOffMaxCore_, gmOnlineMax_ + rowStart,
                        cubeCoreNum_, cur, cur, M_);
        CopyExpTileToUb(ubOffSumCore_, gmOnlineSum_ + rowStart,
                        cubeCoreNum_, cur, cur, M_);
        SetFlag<HardEvent::MTE2_V>(PHASE_1_FLAG_MAX_SUM);

        CopyGmToUb1D(ubOffMHist_, gmMHistory_ + rowStart * numTiles_, cur * numTiles_);
        SetFlag<HardEvent::MTE2_V>(PHASE_1_FLAG_MHIST);

        WaitFlag<HardEvent::MTE2_V>(PHASE_1_FLAG_MAX_SUM);
        ComputeMaxSum(cur);

        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(PHASE_1_FLAG_MHIST);

        // Phase 2: N-tile ping-pong rescale
        for (uint32_t t = 0; t < numTiles_; ++t) {
            uint32_t bufId = t & 1;
            uint32_t nStart = t * baseN_;
            uint32_t curN = (N_ - nStart < baseN_) ? (N_ - nStart) : baseN_;
            uint32_t nAlignCur = ((curN + ELM_PER_32B - 1) / ELM_PER_32B) * ELM_PER_32B;

            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(bufId);
            }

            CopyExpTileToUb(ubOffExpTile_[bufId],
                            gmExpWorkspace_ + rowStart * expWsPitch_ + nStart,
                            cur, nAlignCur, nAlignTile_, expWsPitch_);
            SetFlag<HardEvent::MTE2_V>(bufId);

            WaitFlag<HardEvent::MTE2_V>(bufId);
            RegbaseRescaleV(ubOffExpTile_[bufId], cur, curN, t);
            SetFlag<HardEvent::V_MTE3>(bufId);

            WaitFlag<HardEvent::V_MTE3>(bufId);
            CopyUbToGm2D(ubOffExpTile_[bufId],
                         gmSoftmaxOut_ + rowStart * N_ + nStart,
                         cur, curN, nAlignTile_, N_);
            SetFlag<HardEvent::MTE3_MTE2>(bufId);
        }

        // Drain: reclaim last MTE3_MTE2 for flag balance
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        if (numTiles_ >= 2) {
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }
    }

    __aicore__ inline void ComputeMaxSum(uint32_t cur)
    {
        __ubuf__ ComputeType* allMaxAddr = UbAddr<ComputeType>(ubOffMaxCore_);
        __ubuf__ ComputeType* allSumAddr = UbAddr<ComputeType>(ubOffSumCore_);

        uint16_t cubeCoreNum16 = static_cast<uint16_t>(cubeCoreNum_);
        uint16_t rows = static_cast<uint16_t>(cur);

        __VEC_SCOPE__
        {
            Reg::MaskReg allMask = Reg::CreateMask<ComputeType, Reg::MaskPattern::ALL>();

            for (uint16_t r = 0; r < rows; ++r)
            {
                // maxFinal = max over cores
                Reg::RegTensor<ComputeType> vregMaxF;
                Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                    vregMaxF, allMaxAddr + 0 * cur + r);

                for (uint16_t k = 1; k < cubeCoreNum16; ++k) {
                    Reg::RegTensor<ComputeType> vregMaxC;
                    Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                        vregMaxC, allMaxAddr + k * cur + r);
                    Reg::Max(vregMaxF, vregMaxF, vregMaxC, allMask);
                }

                // sumFinal = Σ_c sumCore * exp(maxCore - maxFinal)
                Reg::RegTensor<ComputeType> vregSumF;
                Reg::Duplicate(vregSumF, ComputeType(0), allMask);

                for (uint16_t k = 0; k < cubeCoreNum16; ++k) {
                    Reg::RegTensor<ComputeType> vregMaxC;
                    Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                        vregMaxC, allMaxAddr + k * cur + r);

                    Reg::RegTensor<ComputeType> vregDiff;
                    Reg::Sub(vregDiff, vregMaxC, vregMaxF, allMask);

                    Reg::RegTensor<ComputeType> vregScale;
                    Reg::Exp(vregScale, vregDiff, allMask);

                    Reg::RegTensor<ComputeType> vregSumC;
                    Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                        vregSumC, allSumAddr + k * cur + r);
                    Reg::Mul(vregSumC, vregSumC, vregScale, allMask);

                    Reg::Add(vregSumF, vregSumF, vregSumC, allMask);
                }

                // Write back maxFinal/sumFinal (overwrites core 0 data)
                Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    allMaxAddr + r, vregMaxF, allMask);
                Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    allSumAddr + r, vregSumF, allMask);
            }
        }
    }

    __aicore__ inline void RegbaseRescaleV(
        int64_t ubOffExp, uint32_t cur, uint32_t curN, uint32_t tileIdx)
    {
        __ubuf__ ComputeType* expAddr     = UbAddr<ComputeType>(ubOffExp);
        __ubuf__ ComputeType* mHistAddr   = UbAddr<ComputeType>(ubOffMHist_);
        __ubuf__ ComputeType* maxFinalAddr = UbAddr<ComputeType>(ubOffMaxCore_);
        __ubuf__ ComputeType* sumFinalAddr = UbAddr<ComputeType>(ubOffSumCore_);

        uint32_t VL = VECTOR_REG_WIDTH / sizeof(ComputeType);
        uint16_t vfN = static_cast<uint16_t>(
            (static_cast<uint32_t>(curN) + VL - 1) / VL);
        uint16_t rows = static_cast<uint16_t>(cur);

        __VEC_SCOPE__
        {
            Reg::MaskReg allMask = Reg::CreateMask<ComputeType, Reg::MaskPattern::ALL>();

            for (uint16_t r = 0; r < rows; ++r)
            {
                // Load maxFinal, sumFinal from UB
                Reg::RegTensor<ComputeType> vregMaxF;
                Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                    vregMaxF, maxFinalAddr + r);

                Reg::RegTensor<ComputeType> vregSumF;
                Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                    vregSumF, sumFinalAddr + r);

                // rescale: exp(mHist - maxFinal) / sumFinal
                Reg::RegTensor<ComputeType> vregMHist;
                Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(
                    vregMHist, mHistAddr + r * numTiles_ + tileIdx);

                Reg::RegTensor<ComputeType> vregDiff;
                Reg::Sub(vregDiff, vregMHist, vregMaxF, allMask);

                Reg::RegTensor<ComputeType> vregScale;
                Reg::Exp(vregScale, vregDiff, allMask);

                Reg::RegTensor<ComputeType> vregTotalScale;
                Reg::Div<ComputeType, &divMode>(vregTotalScale, vregScale, vregSumF, allMask);

                Reg::RegTensor<ComputeType> vregScaleBrc;
                Reg::Duplicate(vregScaleBrc, vregTotalScale, allMask);

                // Per VL chunk: rescale expTile in-place
                __ubuf__ ComputeType* rowExp = expAddr + r * nAlignTile_;

                Reg::MaskReg mask;
                for (uint16_t i = 0; i < vfN; ++i)
                {
                    uint32_t active = static_cast<uint32_t>(curN) - static_cast<uint32_t>(i) * VL;
                    active = (active > VL) ? VL : active;
                    mask = Reg::UpdateMask<ComputeType>(active);

                    Reg::RegTensor<ComputeType> vregExp;
                    Reg::LoadAlign(vregExp, rowExp + i * VL);
                    Reg::Mul(vregExp, vregExp, vregScaleBrc, mask);
                    Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_NORM_B32>(
                        rowExp + i * VL, vregExp, mask);
                }
            }
        }
    }
};

} // namespace Epilogue
#endif
