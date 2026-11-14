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
 * \file online_softmax_per_tile_epilogue.h
 * \brief Phase-1 epilogue: per-tile online softmax (regbase).
 *
 * Consumes BlockMmad L0C2UB output in UB, updates online max/sum per N-tile,
 * and writes exp values + mHistory to GM workspace for Phase-2 cross-core reduction.
 *
 * Adapts to the standard BlockEpilogue interface:
 *   Init(params, problemShape)     — params setup + InitSyncFlag
 *   operator()(blockShape, dstOffset, splitM, baseM, baseN, ubDB) — N-tile inner loop + CV sync
 *   ~destructor                    — CleanUpSyncFlag (guarded by initialized_)
 *
 * CV sync constants are hardcoded to match Blaze library BlockMmad,
 * values must be verified against Blaze source. Following the same pattern as BlockEpilogueFixpipe.
 */
#ifndef ONLINE_SOFTMAX_PER_TILE_EPILOGUE_H
#define ONLINE_SOFTMAX_PER_TILE_EPILOGUE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"
#include "blaze_custom/utils/common_utils.h"

using namespace AscendC;
using AscendC::Te::Get;

namespace Epilogue {

constexpr int64_t ALIGN_ELEM_F32 = 32 / sizeof(float);
constexpr int64_t DATA_BLOCK = 32;

template <typename MmOutT, typename OutT, bool MmOutFlag>
class OnlineSoftmaxPerTileEpilogue {
public:
    using L0CDataType = float;
    using ComputeType = float;
    using OutputType = MmOutT;

    using MakeLayoutND = Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<ComputeType>>;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t STAGE_EVENT_ID = 0;

    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;

    struct Params {
        GM_ADDR softmaxOutAddr{nullptr};
        GM_ADDR onlineMaxAddr{nullptr};
        GM_ADDR onlineSumAddr{nullptr};
        GM_ADDR mHistoryAddr{nullptr};
        GM_ADDR expWorkspaceAddr{nullptr};
        uint32_t cubeCoreNum{0};
        uint32_t m{0};
        uint32_t n{0};
    };

    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr uint64_t UB_SIZE = 248 * 1024;

    __aicore__ inline OnlineSoftmaxPerTileEpilogue() {}

    __aicore__ inline ~OnlineSoftmaxPerTileEpilogue()
    {
        if ASCEND_IS_AIV {
            if (initialized_) {
                CleanUpSyncFlag();
            }
        }
    }

    __aicore__ inline void Init(const Params& p, const ProblemShape& ps)
    {
        if ASCEND_IS_AIV {
            M_ = Get<MNK_M>(ps);
            N_ = Get<MNK_N>(ps);

            int64_t ration = static_cast<int64_t>(GetTaskRation());
            coreIdx_ = GetBlockIdx() / ration;

            VL_ = VECTOR_REG_WIDTH / sizeof(ComputeType);

            gmOnlineMaxPtr_     = reinterpret_cast<__gm__ ComputeType*>(p.onlineMaxAddr);
            gmOnlineSumPtr_     = reinterpret_cast<__gm__ ComputeType*>(p.onlineSumAddr);
            gmMHistoryPtr_      = reinterpret_cast<__gm__ ComputeType*>(p.mHistoryAddr);
            gmExpWorkspacePtr_  = reinterpret_cast<__gm__ ComputeType*>(p.expWorkspaceAddr);
            cubeCoreNum_        = p.cubeCoreNum;
            nAlignExp_          = CeilDiv(N_, ALIGN_ELEM_F32) * ALIGN_ELEM_F32;

            InitSyncFlag();
            initialized_ = true;
        }
    }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset = 0, bool splitM = false,
        int64_t baseM = 0, int64_t baseN = 0, uint64_t ubDB = 1)
    {
        if ASCEND_IS_AIV {
            Run(blockShape, dstOffset, splitM, baseM, baseN, ubDB);
        }
    }

    __aicore__ inline void InitSyncFlag()
    {
        SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);
        SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);
        SetFlag<HardEvent::MTE3_MTE2>(ZERO_FLAG);
    }

    __aicore__ inline void CleanUpSyncFlag()
    {
        WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
        WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);
        WaitFlag<HardEvent::MTE3_MTE2>(ZERO_FLAG);
    }

private:
    __aicore__ inline void Run(
        BlockShape const& blockShape, int64_t dstOffset, bool splitM,
        int64_t baseMParam, int64_t baseNParam, uint64_t ubDB)
    {
        baseM_ = baseMParam;
        baseN_ = baseNParam;

        nAlignL0C_ = CeilDiv(baseN_, ALIGN_ELEM_F32) * ALIGN_ELEM_F32;

        int64_t ration = static_cast<int64_t>(GetTaskRation());
        splitMRows_ = CeilDiv(baseM_, ration);
        matmulAreaBytes_ = splitMRows_ * nAlignL0C_ * sizeof(L0CDataType);

        ubOffMmData_ = 0;
        ubOffPreMax_ = matmulAreaBytes_;
        ubOffPreSum_ = ubOffPreMax_ + splitMRows_ * sizeof(ComputeType);
        ubOffMaxT_   = ubOffPreSum_ + splitMRows_ * sizeof(ComputeType);
        ubOffSumT_   = ubOffMaxT_   + splitMRows_ * sizeof(ComputeType);

        mmDataAddr_  = UbAddr<ComputeType>(ubOffMmData_);
        preMaxAddr_  = UbAddr<ComputeType>(ubOffPreMax_);
        preSumAddr_  = UbAddr<ComputeType>(ubOffPreSum_);
        maxTAddr_    = UbAddr<ComputeType>(ubOffMaxT_);
        sumTAddr_    = UbAddr<ComputeType>(ubOffSumT_);

        int64_t curM = Get<MNK_M>(blockShape);
        int64_t curN = Get<MNK_N>(blockShape);

        int64_t halfM = CeilDiv(curM, GetTaskRation());
        int64_t localRows = ((static_cast<uint64_t>(curM) & 1UL) > 0UL)
                              ? (halfM - GetSubBlockIdx())
                              : halfM;

        int64_t curBaseN = (baseN_ != 0) ? Min(curN, baseN_) : curN;
        int64_t nL1Iter = CeilDiv(curN, curBaseN);
        uint16_t cvPingPong = 0;

        for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
            int64_t tileN = (nIdx + 1 == nL1Iter) ? (curN - curBaseN * nIdx) : curBaseN;
            int64_t tileN0 = dstOffset % N_ + nIdx * curBaseN;
            uint16_t slot = 0;

            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);

            if (localRows > 0) {
                ProcessNTile(curM, tileN, tileN0, dstOffset, localRows, halfM);
            }

            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
            cvPingPong++;
        }
    }

    __aicore__ inline void ProcessNTile(
        int64_t curM, int64_t curN, int64_t tileN0, int64_t dstOffset,
        int64_t localRows, int64_t halfM)
    {
        int64_t tileM0 = dstOffset / N_;
        int64_t subM0  = tileM0 + GetSubBlockIdx() * halfM;
        int64_t wsOff  = coreIdx_ * M_ + subM0;
        int64_t numTilesN = CeilDiv(N_, baseN_);
        int64_t tileIdx = tileN0 / baseN_;

        int64_t gmTotalElems = cubeCoreNum_ * M_;
        auto gmMaxT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmOnlineMaxPtr_), MakeLayoutND{}(1UL, gmTotalElems));
        auto gmSumT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmOnlineSumPtr_), MakeLayoutND{}(1UL, gmTotalElems));
        auto gmMHistoryT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmMHistoryPtr_), MakeLayoutND{}(M_, numTilesN));
        auto gmExpWsT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(gmExpWorkspacePtr_), MakeLayoutND{}(M_, nAlignExp_));

        int64_t nAlignCur = CeilDiv(curN, ALIGN_ELEM_F32) * ALIGN_ELEM_F32;
        auto ubMmDataT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOffMmData_), MakeLayoutND{}(splitMRows_, nAlignCur));
        auto ubMaxT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOffMaxT_), MakeLayoutND{}(splitMRows_, 1UL));
        auto ubSumT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOffSumT_), MakeLayoutND{}(splitMRows_, 1UL));

        WaitFlag<HardEvent::MTE3_MTE2>(ZERO_FLAG);
        WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);

        CopyGmToUb1D(ubOffPreMax_, gmMaxT, wsOff, localRows);
        CopyGmToUb1D(ubOffPreSum_, gmSumT, wsOff, localRows);

        SetFlag<HardEvent::MTE2_V>(ZERO_FLAG);

        WaitFlag<HardEvent::MTE2_V>(ZERO_FLAG);
        WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);

        RegbaseSoftmax(localRows, curN);

        SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);
        SetFlag<HardEvent::V_MTE3>(ZERO_FLAG);
        WaitFlag<HardEvent::V_MTE3>(ZERO_FLAG);

        CopyUbToGm1D(gmMaxT, wsOff, ubOffMaxT_, localRows);
        CopyUbToGm1D(gmSumT, wsOff, ubOffSumT_, localRows);
        CopyUbToGm2D(gmExpWsT, subM0, tileN0, ubMmDataT, localRows, CeilDiv(curN, ALIGN_ELEM_F32) * ALIGN_ELEM_F32);
        CopyUbToGm2D(gmMHistoryT, subM0, tileIdx, ubMaxT, localRows, 1);

        SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);
        SetFlag<HardEvent::MTE3_MTE2>(ZERO_FLAG);
    }

    // ---- UB byte offsets ----
    int64_t ubOffMmData_{0};
    int64_t ubOffPreMax_{0};
    int64_t ubOffPreSum_{0};
    int64_t ubOffMaxT_{0};
    int64_t ubOffSumT_{0};

    // ---- UB raw pointers ----
    __ubuf__ ComputeType* mmDataAddr_{nullptr};
    __ubuf__ ComputeType* preMaxAddr_{nullptr};
    __ubuf__ ComputeType* preSumAddr_{nullptr};
    __ubuf__ ComputeType* maxTAddr_{nullptr};
    __ubuf__ ComputeType* sumTAddr_{nullptr};

    uint32_t VL_{0};
    bool initialized_{false};

    __gm__ ComputeType* gmOnlineMaxPtr_{nullptr};
    __gm__ ComputeType* gmOnlineSumPtr_{nullptr};
    __gm__ ComputeType* gmMHistoryPtr_{nullptr};
    __gm__ ComputeType* gmExpWorkspacePtr_{nullptr};

    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t splitMRows_{0};
    int64_t nAlignL0C_{0};
    int64_t M_{0};
    int64_t N_{0};
    int64_t coreIdx_{0};
    int64_t cubeCoreNum_{0};
    int64_t nAlignExp_{0};
    int64_t matmulAreaBytes_{0};

    template <typename T>
    __aicore__ inline static __ubuf__ T* UbAddr(int64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    template <typename GmTensor>
    __aicore__ inline void CopyGmToUb1D(int64_t ubByteOff, GmTensor& gmT,
                                        int64_t gmOffset, int64_t count)
    {
        auto ubT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubByteOff), MakeLayoutND{}(1UL, count));
        auto gmSlice = gmT.Slice(Te::MakeCoord(0UL, gmOffset), Te::MakeShape(1UL, count));
        Te::Copy(Te::MakeCopy(Te::CopyGM2UB{}), ubT, gmSlice);
    }

    template <typename GmTensor>
    __aicore__ inline void CopyUbToGm1D(GmTensor& gmT, int64_t gmOffset,
                                        int64_t ubByteOff, int64_t count)
    {
        auto ubT = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubByteOff), MakeLayoutND{}(1UL, count));
        auto gmSlice = gmT.Slice(Te::MakeCoord(0UL, gmOffset), Te::MakeShape(1UL, count));
        Te::Copy(Te::MakeCopy(Te::CopyUB2GM{}), gmSlice, ubT);
    }

    template <typename GmTensor, typename UbTensor>
    __aicore__ inline void CopyUbToGm2D(GmTensor& gmT,
                                        int64_t gmRowOffset, int64_t gmColOffset,
                                        UbTensor& ubT,
                                        int64_t rows, int64_t cols)
    {
        auto gmSlice = gmT.Slice(
            Te::MakeCoord(gmRowOffset, gmColOffset), Te::MakeShape(rows, cols));
        auto ubSlice = ubT.Slice(
            Te::MakeCoord(0UL, 0UL), Te::MakeShape(rows, cols));
        Te::Copy(Te::MakeCopy(Te::CopyUB2GM{}), gmSlice, ubSlice);
    }

    __aicore__ inline void ProcessRowPass1(
        __ubuf__ ComputeType* rowSrc, uint16_t r,
        uint16_t mainLoop, uint32_t tailActive,
        Reg::MaskReg& allMask, Reg::MaskReg& tailMask,
        Reg::RegTensor<ComputeType>& vregMaxBrc,
        Reg::RegTensor<ComputeType>& vregPreMax)
    {
        Reg::RegTensor<ComputeType> vregAcc;
        Reg::RegTensor<ComputeType> vregSrc;
        Reg::Duplicate(vregAcc, -__builtin_inff(), allMask);

        for (uint16_t i = 0; i < mainLoop; ++i) {
            Reg::LoadAlign(vregSrc, rowSrc + i * VL_);
            Reg::Max(vregAcc, vregSrc, vregAcc, allMask);
        }
        Reg::LoadAlign(vregSrc, rowSrc + mainLoop * VL_);
        Reg::Max<ComputeType, Reg::MaskMergeMode::MERGING>(vregAcc, vregSrc, vregAcc, tailMask);

        Reg::RegTensor<ComputeType> vregMaxReduced;
        Reg::Reduce<Reg::ReduceType::MAX>(vregMaxReduced, vregAcc, allMask);

        Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(vregPreMax, preMaxAddr_ + r);

        Reg::RegTensor<ComputeType> vregNewMax;
        Reg::Max(vregNewMax, vregMaxReduced, vregPreMax, allMask);

        Reg::Duplicate(vregMaxBrc, vregNewMax, allMask);

        Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            maxTAddr_ + r, vregNewMax, allMask);
    }

    __aicore__ inline void ProcessRowPass2(
        __ubuf__ ComputeType* rowSrc, uint16_t r,
        uint16_t mainLoop, uint32_t tailActive,
        Reg::MaskReg& allMask, Reg::MaskReg& tailMask,
        const Reg::RegTensor<ComputeType>& vregMaxBrc,
        const Reg::RegTensor<ComputeType>& vregPreMax)
    {
        Reg::RegTensor<ComputeType> vregSrc;
        Reg::RegTensor<ComputeType> vregSumAcc;
        Reg::Duplicate(vregSumAcc, ComputeType(0), allMask);

        for (uint16_t i = 0; i < mainLoop; ++i) {
            Reg::LoadAlign(vregSrc, rowSrc + i * VL_);
            Reg::Sub(vregSrc, vregSrc, vregMaxBrc, allMask);
            Reg::Exp(vregSrc, vregSrc, allMask);
            Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_NORM_B32>(
                rowSrc + i * VL_, vregSrc, allMask);
            Reg::Add(vregSumAcc, vregSrc, vregSumAcc, allMask);
        }
        Reg::LoadAlign(vregSrc, rowSrc + mainLoop * VL_);
        Reg::Sub(vregSrc, vregSrc, vregMaxBrc, tailMask);
        Reg::Exp(vregSrc, vregSrc, tailMask);
        Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_NORM_B32>(
            rowSrc + mainLoop * VL_, vregSrc, tailMask);
        Reg::Add<ComputeType, Reg::MaskMergeMode::MERGING>(vregSumAcc, vregSrc, vregSumAcc, tailMask);

        Reg::RegTensor<ComputeType> vregSumReduced;
        Reg::Reduce<Reg::ReduceType::SUM>(vregSumReduced, vregSumAcc, allMask);

        Reg::RegTensor<ComputeType> vregDiff;
        Reg::Sub(vregDiff, vregPreMax, vregMaxBrc, allMask);

        Reg::RegTensor<ComputeType> vregScale;
        Reg::Exp(vregScale, vregDiff, allMask);

        Reg::RegTensor<ComputeType> vregPreSum;
        Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(vregPreSum, preSumAddr_ + r);

        Reg::Mul(vregPreSum, vregPreSum, vregScale, allMask);
        Reg::Add(vregSumReduced, vregPreSum, vregSumReduced, allMask);

        Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            sumTAddr_ + r, vregSumReduced, allMask);
    }

    __aicore__ inline void RegbaseSoftmax(int64_t localRows, int64_t curN)
    {
        uint16_t rows = static_cast<uint16_t>(localRows);
        int64_t nAlignCur = CeilDiv(curN, ALIGN_ELEM_F32) * ALIGN_ELEM_F32;
        uint16_t vfN = static_cast<uint16_t>(CeilDiv(static_cast<uint64_t>(curN), VL_));
        uint16_t mainLoop = vfN - 1;
        uint32_t tailActive = static_cast<uint32_t>(curN) % VL_;
        if (tailActive == 0) { tailActive = VL_; }

        __VEC_SCOPE__
        {
            Reg::MaskReg allMask = Reg::CreateMask<ComputeType, Reg::MaskPattern::ALL>();
            Reg::MaskReg tailMask = Reg::UpdateMask<ComputeType>(tailActive);

            for (uint16_t r = 0; r < rows; ++r)
            {
                __ubuf__ ComputeType* rowSrc = mmDataAddr_ + r * nAlignCur;

                Reg::RegTensor<ComputeType> vregMaxBrc;
                Reg::RegTensor<ComputeType> vregPreMax;

                ProcessRowPass1(rowSrc, r, mainLoop, tailActive, allMask, tailMask, vregMaxBrc, vregPreMax);
                ProcessRowPass2(rowSrc, r, mainLoop, tailActive, allMask, tailMask, vregMaxBrc, vregPreMax);
            }
        }
    }
};

} // namespace Epilogue
#endif
