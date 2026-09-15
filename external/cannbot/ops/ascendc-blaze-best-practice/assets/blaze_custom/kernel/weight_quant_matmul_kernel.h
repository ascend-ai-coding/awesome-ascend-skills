/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

// ============================================================================
// Weight-Quant Matmul Kernel —— V+C fusion (prologue in this file)
// ----------------------------------------------------------------------------
// Data flow:
//   AIV (prologue): GM B(int8) → UB → RegBase dequant(int8→fp16→bf16→[Add]→Mul)
//                   → UB → L1 (bf16 B_dequant)
//   AIC (BlockMmad): GM A(bf16) → L1 → L0A; L1 B_dequant → L0B → MMAD → L0C → GM
//
// This file contains:
//   1. VF dequant function (__simd_vf__ execution domain)
//   2. WeightQuantMatmulPrologue class (AIV-side prologue: UB/L1 mgmt + K-loop + CV sync)
//   3. GemmUniversal specialization (orchestrates AIC BlockMmad + AIV Prologue)
//
// CV sync (AIV side):
//   Each K-L1 iteration: WaitFlag(AIC_SYNC_AIV_FLAG) → dequant+write L1 →
//   SetFlag(AIV_SYNC_AIC_FLAG).
//   After loop: WaitFlag consumes remaining AIC_SYNC_AIV_FLAG.
// ============================================================================

#ifndef WEIGHT_QUANT_MATMUL_KERNEL_H
#define WEIGHT_QUANT_MATMUL_KERNEL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

#include "../block/weight_quant_matmul_block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h"
#include "../epilogue/cv_sync_constants.h"

// ============================================================================
// VF dequant function (__simd_vf__ execution domain)
// ============================================================================

template <class BType, class DequantBType>
struct WeightDequantVfParams {
    __ubuf__ BType* bInPhyAddr;
    __ubuf__ DequantBType* bOutPhyAddr;
    __ubuf__ DequantBType* scalePhyAddr;
    __ubuf__ DequantBType* offsetPhyAddr;
    uint16_t vfNIterNum;
    uint16_t vfNStride;
    uint16_t vfKIterNum;
    uint16_t vfKStride;
    uint64_t innerAxisSizeAligned;
    uint64_t outerAxisSize;
};

template <class BType, class DequantBType,
          bool TransB, bool HasOffset,
          AscendC::Reg::LoadDist LOAD_TRAIT_SCALE>
__simd_vf__ inline void WeightDequantVf(WeightDequantVfParams<BType, DequantBType> params)
{
    AscendC::Reg::RegTensor<BType> regBIn;
    AscendC::Reg::RegTensor<DequantBType> regBOut;
    AscendC::Reg::RegTensor<DequantBType> regScale;
    AscendC::Reg::RegTensor<DequantBType> regOffset;
    AscendC::Reg::RegTensor<half> regFp16;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<DequantBType, AscendC::Reg::MaskPattern::ALL>();

    static constexpr AscendC::Reg::CastTrait castTraitInt8ToFP16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTraitFP16ToBF16 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr uint64_t BLOCK_CUBE = 16UL;

    for (uint16_t vfNIter = 0; vfNIter < params.vfNIterNum; ++vfNIter) {
        uint16_t vfNOffset = static_cast<uint16_t>(vfNIter * params.vfNStride);
        AscendC::Reg::LoadAlign<DequantBType, LOAD_TRAIT_SCALE>(regScale, params.scalePhyAddr + vfNOffset);
        if constexpr (HasOffset) {
            AscendC::Reg::LoadAlign<DequantBType, LOAD_TRAIT_SCALE>(regOffset, params.offsetPhyAddr + vfNOffset);
        }
        for (uint16_t vfKIter = 0; vfKIter < params.vfKIterNum; ++vfKIter) {
            uint16_t vfKOffset = static_cast<uint16_t>(vfKIter * params.vfKStride);
            uint16_t innerOffset;
            uint16_t outerIdx;
            if constexpr (TransB) {
                innerOffset = vfKOffset;
                outerIdx = vfNIter;
            } else {
                innerOffset = vfNOffset;
                outerIdx = vfKIter;
            }
            AscendC::Reg::LoadAlign<BType, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(
                regBIn, params.bInPhyAddr + outerIdx * params.innerAxisSizeAligned + innerOffset);
            if constexpr (AscendC::Std::is_same_v<DequantBType, bfloat16_t>) {
                AscendC::Reg::Cast<half, BType, castTraitInt8ToFP16>(regFp16, regBIn, mask);
                AscendC::Reg::Cast<DequantBType, half, castTraitFP16ToBF16>(regBOut, regFp16, mask);
            } else {
                AscendC::Reg::Cast<DequantBType, BType, castTraitInt8ToFP16>(regBOut, regBIn, mask);
            }
            if constexpr (HasOffset) {
                AscendC::Reg::Add(regBOut, regBOut, regOffset, mask);
            }
            AscendC::Reg::Mul(regBOut, regBOut, regScale, mask);
            AscendC::Reg::StoreAlign<DequantBType, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                params.bOutPhyAddr + params.outerAxisSize * innerOffset + BLOCK_CUBE * outerIdx,
                regBOut, params.outerAxisSize, mask);
        }
    }
}

// ============================================================================
// WeightQuantMatmulPrologue — AIV-side dequant prologue
// ----------------------------------------------------------------------------
// Responsibilities:
//   - UB buffer planning (bIn/bOut/scale/offset all ping-pong double buffered,
//     one set per half)
//   - GM→UB copy of B/scale/offset
//   - __simd_vf__ dequant invocation (cast→[Add offset]→Mul scale)
//   - UB→L1 copy of B_dequant
//   - AIV↔AIC CrossCore sync
// ============================================================================

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class BType_, class DequantBType_, bool TransB>
class WeightQuantMatmulPrologue {
public:
    using BType = BType_;
    using DequantBType = DequantBType_;
    using ScaleType = DequantBType_;
    using OffsetType = DequantBType_;
    static constexpr bool transB = TransB;

    static constexpr uint16_t CV_SYNC_MODE = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;

    using MakeLayoutDequantB = AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::Int<16>>;
    using MakeLayoutBIn = AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using MakeLayoutScale = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;

    static constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
    static constexpr uint64_t NUM_B16_IN_ONE_REG = VECTOR_REG_WIDTH / sizeof(int16_t);
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;

    static constexpr AscendC::Reg::LoadDist LOAD_TRAIT_SCALE = TransB ?
        AscendC::Reg::LoadDist::DIST_BRC_B16 : AscendC::Reg::LoadDist::DIST_NORM;

    struct Params {
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR offsetGmAddr{nullptr};
        uint32_t nUbSize{0};
        uint32_t kUbSize{0};
        uint8_t hasOffset{0};
    };

    __aicore__ inline WeightQuantMatmulPrologue() {}

    __aicore__ inline ~WeightQuantMatmulPrologue() {}

    __aicore__ inline void Init(uint64_t kL1, uint64_t kSize,
                                uint32_t nUbSize, uint32_t kUbSize, bool hasOffset)
    {
        nUbSize_ = nUbSize;
        kUbSize_ = kUbSize;
        hasOffset_ = hasOffset;
        kL1_ = kL1;
        kSize_ = kSize;
        iterKNum_ = CeilDiv(kSize_, kL1_);
        l1HalfSize_ = AscendC::TOTAL_L1_SIZE >> 1;

        bInOneBuffer_ = kUbSize_ * nUbSize_ * sizeof(BType);
        if constexpr (transB) {
            bOutOneBuffer_ = kUbSize_ * (nUbSize_ + 1) * sizeof(DequantBType);
        } else {
            bOutOneBuffer_ = (kUbSize_ + 1) * nUbSize_ * sizeof(DequantBType);
        }
        scaleBuffer_ = nUbSize_ * sizeof(ScaleType);
        offsetBuffer_ = nUbSize_ * sizeof(OffsetType);

        uint64_t halfUBSize = AscendC::TOTAL_UB_SIZE >> 1;
        bInBufferOffset_[0] = 0;
        bOutBufferOffset_[0] = bInOneBuffer_;
        scaleBufferOffset_[0] = bOutBufferOffset_[0] + bOutOneBuffer_;
        offsetBufferOffset_[0] = scaleBufferOffset_[0] + scaleBuffer_;

        bInBufferOffset_[1] = halfUBSize;
        bOutBufferOffset_[1] = halfUBSize + bInOneBuffer_;
        scaleBufferOffset_[1] = bOutBufferOffset_[1] + bOutOneBuffer_;
        offsetBufferOffset_[1] = scaleBufferOffset_[1] + scaleBuffer_;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FIRST_FLAG);
    }

    // ============ Stage 1: GM → UB ============
    template <typename TensorB, typename TensorScale, typename TensorOffset>
    __aicore__ inline void LoadGM2UB(
        uint64_t ubBufId, uint64_t iterK,
        uint64_t kSubOffset, uint64_t nSubOffset,
        uint64_t kUbLen, uint64_t nUbLen,
        uint64_t innerAxisSizeAligned,
        const TensorB& gmB, const TensorScale& gmScale, const TensorOffset& gmOffset,
        __ubuf__ BType*& bInPhyAddr)
    {
        uint64_t layoutRow = transB ? innerAxisSizeAligned : kUbLen;
        uint64_t layoutCol = transB ? nUbLen : innerAxisSizeAligned;
        auto layoutBIn = MakeLayoutBIn{}(layoutRow, layoutCol);
        auto tensorBIn = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, BType>(bInBufferOffset_[ubBufId]),
            layoutBIn);
        auto tensorBInReal = tensorBIn.Slice(
            AscendC::Te::MakeCoord(0UL, 0UL), AscendC::Te::MakeShape(kUbLen, nUbLen));
        auto tensorBlockBGm = gmB.Slice(
            AscendC::Te::MakeCoord(kSubOffset, nSubOffset), AscendC::Te::MakeShape(kUbLen, nUbLen));
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBInReal, tensorBlockBGm);
        bInPhyAddr = (__ubuf__ BType*)tensorBIn.Data().Get();

        if (iterK == 0 || iterK == 1) {
            auto layoutScaleUB = MakeLayoutScale{}(1UL, nUbLen);
            auto tensorScaleUB = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ScaleType>(scaleBufferOffset_[ubBufId]), layoutScaleUB);
            auto tensorBlockScaleGm = gmScale.Slice(
                AscendC::Te::MakeCoord(0UL, nSubOffset), AscendC::Te::MakeShape(1UL, nUbLen));
            AscendC::Te::Copy(copyGM2UB, tensorScaleUB, tensorBlockScaleGm);
            if (hasOffset_) {
                auto layoutOffsetUB = MakeLayoutScale{}(1UL, nUbLen);
                auto tensorOffsetUB = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, OffsetType>(offsetBufferOffset_[ubBufId]), layoutOffsetUB);
                auto tensorBlockOffsetGm = gmOffset.Slice(
                    AscendC::Te::MakeCoord(0UL, nSubOffset), AscendC::Te::MakeShape(1UL, nUbLen));
                AscendC::Te::Copy(copyGM2UB, tensorOffsetUB, tensorBlockOffsetGm);
            }
        }
    }

    // ============ Stage 2: VF dequant ============
    __aicore__ inline void DequantVF(
        uint64_t ubBufId,
        uint64_t kUbLen, uint64_t nUbLen,
        uint64_t innerAxisSizeAligned,
        __ubuf__ BType* bInPhyAddr,
        __ubuf__ DequantBType*& bOutPhyAddr)
    {
        auto layoutBOut = MakeLayoutDequantB{}(kUbLen, nUbLen);
        auto tensorBOut = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, DequantBType>
            (bOutBufferOffset_[ubBufId]), layoutBOut);
        bOutPhyAddr = (__ubuf__ DequantBType*)tensorBOut.Data().Get();

        WeightDequantVfParams<BType, DequantBType> vfParams;
        vfParams.bInPhyAddr = bInPhyAddr;
        vfParams.bOutPhyAddr = bOutPhyAddr;
        vfParams.scalePhyAddr = (__ubuf__ ScaleType*)(scaleBufferOffset_[ubBufId]);
        vfParams.offsetPhyAddr = (__ubuf__ OffsetType*)(offsetBufferOffset_[ubBufId]);
        vfParams.innerAxisSizeAligned = innerAxisSizeAligned;
        vfParams.outerAxisSize = (transB ? nUbLen : kUbLen) + 1;

        uint64_t innerAxisSize = transB ? kUbLen : nUbLen;
        if constexpr (transB) {
            vfParams.vfNIterNum = static_cast<uint16_t>(nUbLen);
            vfParams.vfNStride = 1;
            vfParams.vfKIterNum = static_cast<uint16_t>(CeilDiv(innerAxisSize, NUM_B16_IN_ONE_REG));
            vfParams.vfKStride = static_cast<uint16_t>(NUM_B16_IN_ONE_REG);
        } else {
            vfParams.vfNIterNum = static_cast<uint16_t>(CeilDiv(innerAxisSize, NUM_B16_IN_ONE_REG));
            vfParams.vfNStride = static_cast<uint16_t>(NUM_B16_IN_ONE_REG);
            vfParams.vfKIterNum = static_cast<uint16_t>(kUbLen);
            vfParams.vfKStride = 1;
        }

        if (hasOffset_) {
            asc_vf_call<WeightDequantVf<BType, DequantBType,
                                         transB, true, LOAD_TRAIT_SCALE>>(vfParams);
        } else {
            asc_vf_call<WeightDequantVf<BType, DequantBType,
                                         transB, false, LOAD_TRAIT_SCALE>>(vfParams);
        }
    }

    // ============ Stage 3: UB → L1 ============
    // Uses copy_ubuf_to_cbuf to copy bOut from UB to L1 with explicit srcGap=1
    // to avoid bank conflicts.
    // copy_ubuf_to_cbuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap)
    //   dst:      L1 destination address
    //   src:      UB source address
    //   sid:      stream id, fixed to 0
    //   nBurst:   outer-axis burst count (outer 16-group count)
    //   lenBurst: DataBlock count per burst (inner-axis contiguous DataBlocks, 32B each)
    //   srcGap:   DataBlock count to skip in UB between bursts (bank conflict padding = 1)
    //   dstGap:   DataBlock count to skip in L1 between bursts (L1 alignment padding)
    __aicore__ inline void StoreUB2L1(
        uint64_t ubBufId,
        uint64_t kUbLen, uint64_t nUbLen,
        uint64_t curKL1, uint64_t curN,
        uint64_t subBlockIdx, uint64_t kHalf, uint64_t nHalf,
        __ubuf__ DequantBType* bOutPhyAddr)
    {
        auto layoutBL1 = MakeLayoutDequantB{}(curKL1, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DequantBType>(l1HalfSize_ * ubBufId),
            layoutBL1);
        auto tensorBL1SliceCoord = transB ?
            AscendC::Te::MakeCoord(0UL, subBlockIdx * nHalf) :
            AscendC::Te::MakeCoord(subBlockIdx * kHalf, 0UL);
        auto tensorBL1Slice = tensorBL1.Slice(tensorBL1SliceCoord,
            AscendC::Te::MakeShape(kUbLen, nUbLen));

        if constexpr (!transB) {
            // NZ: outer=N, inner=K
            copy_ubuf_to_cbuf(
                reinterpret_cast<__cbuf__ DequantBType*>(tensorBL1Slice.Data().Get()),
                bOutPhyAddr, 0,
                static_cast<uint16_t>(CeilDiv(nUbLen, 16UL)),
                static_cast<uint16_t>(kUbLen),
                1,
                static_cast<uint16_t>(CeilDiv(curKL1, 16UL) * 16 - kUbLen));
        } else {
            // ZN: outer=K, inner=N
            copy_ubuf_to_cbuf(
                reinterpret_cast<__cbuf__ DequantBType*>(tensorBL1Slice.Data().Get()),
                bOutPhyAddr, 0,
                static_cast<uint16_t>(CeilDiv(kUbLen, 16UL)),
                static_cast<uint16_t>(nUbLen),
                1,
                static_cast<uint16_t>(CeilDiv(curN, 16UL) * 16 - nUbLen));
        }
    }

    template <typename TensorB, typename TensorScale, typename TensorOffset>
    __aicore__ inline void operator()(
        TensorB gmB, TensorScale gmScale, TensorOffset gmOffset, int64_t curNInt)
    {
        uint64_t curN = static_cast<uint64_t>(curNInt);
        uint64_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint64_t nHalf = (curN / 32) * BLOCK_CUBE;
        uint64_t nUbLen = transB ? (subBlockIdx == 0 ? nHalf : curN - nHalf) : curN;
        uint64_t nSubOffset = transB ? subBlockIdx * nHalf : 0UL;

        for (uint64_t iterK = 0; iterK < iterKNum_; ++iterK) {
            uint64_t ubBufId = ubLoopCnt_ & 0x1;
            uint64_t kL1Offset = iterK * kL1_;
            auto curKL1 = (iterK + 1 == iterKNum_) ? (kSize_ - kL1Offset) : kL1_;
            uint64_t kHalf = (curKL1 / 32) * BLOCK_CUBE;
            uint64_t kUbLen = transB ? curKL1 : (subBlockIdx == 0 ? kHalf : curKL1 - kHalf);
            auto kSubOffset = kL1Offset + (transB ? 0UL : subBlockIdx * kHalf);
            uint64_t innerAxisSizeAligned = Blaze::Gemm::CeilAlign(transB ? kUbLen : nUbLen, 32UL);

            __ubuf__ BType* bInPhyAddr = nullptr;
            __ubuf__ DequantBType* bOutPhyAddr = nullptr;

            // Stage 1: GM → UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubBufId);
            LoadGM2UB(ubBufId, iterK, kSubOffset, nSubOffset, kUbLen, nUbLen,
                      innerAxisSizeAligned, gmB, gmScale, gmOffset, bInPhyAddr);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubBufId);

            // Stage 2: VF dequant
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubBufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubBufId);
            DequantVF(ubBufId, kUbLen, nUbLen, innerAxisSizeAligned, bInPhyAddr, bOutPhyAddr);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubBufId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubBufId);

            // Stage 3: UB → L1
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubBufId);
            AscendC::CrossCoreWaitFlag<CV_SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + ubBufId);
            StoreUB2L1(ubBufId, kUbLen, nUbLen, curKL1, curN,
                       subBlockIdx, kHalf, nHalf, bOutPhyAddr);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubBufId);
            AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubBufId);

            ubLoopCnt_++;
        }
    }

    __aicore__ inline void Cleanup()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FIRST_FLAG);

        AscendC::CrossCoreWaitFlag<CV_SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreWaitFlag<CV_SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + 1);
    }

private:
    uint64_t nUbSize_{0};
    uint64_t kUbSize_{0};
    uint64_t kL1_{0};
    uint64_t kSize_{0};
    uint64_t iterKNum_{0};
    uint64_t ubLoopCnt_{0UL};
    uint64_t l1HalfSize_{0UL};
    bool hasOffset_{false};

    uint64_t bInOneBuffer_ = 0UL;
    uint64_t bOutOneBuffer_ = 0UL;
    uint64_t scaleBuffer_ = 0UL;
    uint64_t offsetBuffer_ = 0UL;
    uint64_t bInBufferOffset_[DOUBLE_BUFFER_NUM] = {0UL};
    uint64_t bOutBufferOffset_[DOUBLE_BUFFER_NUM] = {0UL};
    uint64_t scaleBufferOffset_[DOUBLE_BUFFER_NUM] = {0UL};
    uint64_t offsetBufferOffset_[DOUBLE_BUFFER_NUM] = {0UL};
};

} // namespace Kernel

// ============================================================================
// GemmUniversal specialization — orchestrates AIC BlockMmad + AIV Prologue
// ============================================================================

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<BlockEpilogue_, void> &&
        AscendC::Std::is_same_v<
            KernelWeightQuantMatmul,
            typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockSchedulerOp = BlockScheduler_;

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using DequantBType = typename BlockMmad::DequantBType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using ScaleType = AType;
    using OffsetType = AType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using Prologue = WeightQuantMatmulPrologue<BType, DequantBType, transB>;
    using PrologueParams = typename Prologue::Params;

    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    static constexpr uint64_t A_C0 = 32 / sizeof(AType);
    static constexpr uint64_t B_C0 = 32 / sizeof(BType);
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<A_C0>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<B_C0>>;
    using MakeLayoutScale = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;

    struct MatmulTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        MatmulTiling mmadTiling;
        PrologueParams prologueParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        BlockSchedulerOp bs(params.problemShape, params.schParams);

        int64_t m = AscendC::Te::Get<0>(params.problemShape);
        int64_t n = AscendC::Te::Get<1>(params.problemShape);
        int64_t k = AscendC::Te::Get<2>(params.problemShape);

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto layoutC = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(m, n);
        auto layoutScale = MakeLayoutScale{}(1UL, n);

        __gm__ AType* aGmPtr = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        __gm__ BType* bGmPtr = reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr);
        __gm__ CType* cGmPtr = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        __gm__ BiasType* biasGmPtr = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        __gm__ ScaleType* scaleGmPtr = reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.scaleGmAddr);
        __gm__ OffsetType* offsetGmPtr = reinterpret_cast<__gm__ OffsetType*>(params.prologueParams.offsetGmAddr);

        constexpr int64_t kPos = 0L;

        if ASCEND_IS_AIC {
            BlockMmad blockMmadOp;
            TupleShape mmadProblemShape = {m, n, k};
            TupleShape l0TileShape{
                params.mmadTiling.baseM, params.mmadTiling.baseN, params.mmadTiling.baseK};
            bool enableL0cPingPong = (params.mmadTiling.dbL0C > 1);
            bool hasBias = (params.mmadParams.biasGmAddr != nullptr);
            blockMmadOp.Init(mmadProblemShape, l0TileShape, params.l1Params, enableL0cPingPong, hasBias);

            auto gmA = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmPtr), layoutA);
            auto gmC = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmPtr), layoutC);
            auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, n);
            auto gmBiasFull = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmPtr), layoutBias);

            int64_t curBlockIdx = AscendC::GetBlockIdx();
            int64_t coreNums = AscendC::GetBlockNum();
            uint64_t tileNum = bs.GetTileCount();
            for (uint64_t tileIdx = static_cast<uint64_t>(curBlockIdx);
                 tileIdx < tileNum; tileIdx += static_cast<uint64_t>(coreNums)) {
                auto blockCoord = bs.GetBlockCoord(tileIdx);
                auto blockShape = bs.GetBlockShape(blockCoord);
                int64_t mPos = AscendC::Te::Get<Blaze::Gemm::MNK_M>(blockCoord);
                int64_t nPos = AscendC::Te::Get<Blaze::Gemm::MNK_N>(blockCoord);
                int64_t curM = AscendC::Te::Get<Blaze::Gemm::MNK_M>(blockShape);
                int64_t curN = AscendC::Te::Get<Blaze::Gemm::MNK_N>(blockShape);
                if (curM <= 0 || curN <= 0) { return; }

                auto gmBlockA = gmA.Slice(
                    AscendC::Te::MakeCoord(mPos, kPos),
                    AscendC::Te::MakeShape(curM, k));
                auto gmBlockC = gmC.Slice(
                    AscendC::Te::MakeCoord(mPos, nPos),
                    AscendC::Te::MakeShape(curM, curN));
                auto gmBias = gmBiasFull.Slice(
                    AscendC::Te::MakeCoord(0, nPos),
                    AscendC::Te::MakeShape(1UL, curN));
                blockMmadOp(gmBlockA, gmBias, gmBlockC, blockShape);
            }
        }

        if ASCEND_IS_AIV {
            Prologue prologueOp;
            prologueOp.Init(
                params.l1Params.kL1,
                static_cast<uint64_t>(k),
                params.prologueParams.nUbSize,
                params.prologueParams.kUbSize,
                params.prologueParams.hasOffset != 0);

            auto gmB = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmPtr), layoutB);
            auto gmScale = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleGmPtr), layoutScale);
            auto gmOffset = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(offsetGmPtr), layoutScale);

            int64_t curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
            int64_t coreNums = AscendC::GetBlockNum();
            uint64_t tileNum = bs.GetTileCount();
            for (uint64_t tileIdx = static_cast<uint64_t>(curBlockIdx);
                 tileIdx < tileNum; tileIdx += static_cast<uint64_t>(coreNums)) {
                auto blockCoord = bs.GetBlockCoord(tileIdx);
                auto blockShape = bs.GetBlockShape(blockCoord);
                int64_t nPos = AscendC::Te::Get<Blaze::Gemm::MNK_N>(blockCoord);
                int64_t curN = AscendC::Te::Get<Blaze::Gemm::MNK_N>(blockShape);
                if (curN <= 0) { return; }

                auto gmBlockB = gmB.Slice(
                    AscendC::Te::MakeCoord(kPos, nPos),
                    AscendC::Te::MakeShape(k, curN));
                auto gmBlockScale = gmScale.Slice(
                    AscendC::Te::MakeCoord(0, nPos),
                    AscendC::Te::MakeShape(1UL, curN));
                auto gmBlockOffset = gmOffset.Slice(
                    AscendC::Te::MakeCoord(0, nPos),
                    AscendC::Te::MakeShape(1UL, curN));
                prologueOp(gmBlockB, gmBlockScale, gmBlockOffset, curN);
            }
            prologueOp.Cleanup();
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#endif // WEIGHT_QUANT_MATMUL_KERNEL_H
