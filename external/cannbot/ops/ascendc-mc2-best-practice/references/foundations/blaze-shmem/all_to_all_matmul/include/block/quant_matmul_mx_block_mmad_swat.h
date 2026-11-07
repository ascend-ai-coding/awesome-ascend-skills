/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_mx_block_mmad_swat.h
 * \brief Block-level MX MMAD pipeline for the SWAT non-full-load path.
 */

#pragma once

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor_api/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../utils/constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/pad_mx_kl1.h"

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_,
    class LayoutBTuple_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<DispatchPolicy_, QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>>>> {
public:
    template <typename T>
    struct TypeUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct TypeUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    template <typename T>
    struct LayoutUnpack {
        using Data = T;
        using Scale = void;
    };

    template <typename T0, typename T1>
    struct LayoutUnpack<AscendC::Std::tuple<T0, T1>> {
        using Data = T0;
        using Scale = T1;
    };

    using AType = typename TypeUnpack<ATypeTuple_>::Data;
    using ScaleAType = typename TypeUnpack<ATypeTuple_>::Scale;
    using BType = typename TypeUnpack<BTypeTuple_>::Data;
    using ScaleBType = typename TypeUnpack<BTypeTuple_>::Scale;
    using CType = CType_;
    using LayoutA = typename LayoutUnpack<LayoutATuple_>::Data;
    using LayoutScaleA = typename LayoutUnpack<LayoutATuple_>::Scale;
    using LayoutB = typename LayoutUnpack<LayoutBTuple_>::Data;
    using LayoutScaleB = typename LayoutUnpack<LayoutBTuple_>::Scale;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = AscendC::IsSameType<LayoutA, AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>>::value;
	static constexpr bool transB = AscendC::IsSameType<LayoutB, AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>>::value;
    static constexpr bool isDTypeFp4 = AscendC::IsSameType<AType, fp4x2_e1m2_t>::value ||
        AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
    static constexpr uint64_t L1_BUFFER_NUM = DispatchPolicy::stages;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1;
    static constexpr uint64_t L1_BUFFER_GROUP_NUM = L1_BUFFER_NUM >> 1;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr int32_t L0C_C0 = 16;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t MXFP_GROUP_SIZE = 32UL;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t SCALE_BUFFER_NUM = 2;
    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{0UL};
    uint64_t scaleKL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t scaleLoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    bool enableL0cPingPong_{false};

    using MakeLayoutAL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<transA, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<transB, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::Int<C0_SIZE>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR localAGmAddr{nullptr};
        GM_ADDR localScaleAGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t originalM;
        uint64_t rankDim;
        uint64_t rankId;
    };

    __aicore__ inline BlockMmad()
    {
        // Prime all producer/consumer events so the first iteration can enter
        // the pipelined copy-and-compute loop without special-case branches.
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM_MX; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        // Drain every in-flight transfer before leaving so later blocks do not
        // observe stale event state from the previous pipeline instance.
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM_MX; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool enableL0cPingPong)
    {
        // Pre-compute all persistent buffer sizes and L1 offsets once per block
        // so the hot path only needs to switch between ping-pong slots.
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        enableL0cPingPong_ = enableL0cPingPong;
        constexpr uint64_t sizeShift = isDTypeFp4 ? 1UL : 0UL;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        aL1OneBuffer_ = (baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleL1Window_ = scaleKL1_ / kL1_;
        kL1ScaleSize_ = CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleKL1Group_ = CeilDiv(scaleKL1_, MXFP_GROUP_SIZE);
        scaleKL1ScaleSize_ = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|...|A1|B1|AScale1|BScale1|...
        // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|...|A1A3|B1B3|AScale1|BScale1|...
        uint64_t l1HalfSize = AscendC::TOTAL_L1_SIZE >> 1;
        #pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            uint64_t l1BufferGroup = bufferId >> 1;
            uint64_t l1HalfOffset = (bufferId & 1UL) * l1HalfSize;
            l1BufferAOffset_[bufferId] = l1HalfOffset + l1BufferGroup * aL1OneBuffer_;
            l1BufferBOffset_[bufferId] = l1HalfOffset + L1_BUFFER_GROUP_NUM * aL1OneBuffer_ +
                l1BufferGroup * bL1OneBuffer_;
        }
        #pragma unroll
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * L1_BUFFER_GROUP_NUM;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC, typename TensorL0C>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape,
        TensorL0C tensorL0C, int32_t splitKidx, bool enableAtomicAdd = false) //splitKidx有3种状态，0：第一块，-1最后一块儿，其他数字达标块数
    {
        // Non-full-load streams both A and B through L1/L0 in chunks. Scale
        // tensors advance in a coarser cadence that matches `scaleKL1_`.
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t scaleWindowIter = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t kL1Offset = iter0 * kL1_;
            auto curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            auto curPadKL1 = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
            auto curGmAKL1 = curGmBKL1;
            if (scaleWindowIter == 0) {
                // Scale fragments are refreshed only when the current K chunk
                // enters a new scale reuse window.
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);

                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }

                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
                auto layoutScaleAL1 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curM, scaleKL1Group_);
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(0, kL1Offset / MXFP_GROUP_SIZE),
                    AscendC::Te::MakeShape(
                        curM, CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

                auto layoutScaleBL1 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(scaleKL1Group_, curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(kL1Offset / MXFP_GROUP_SIZE, 0),
                    AscendC::Te::MakeShape(
                        CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = MakeLayoutAL1{}(curM, curPadKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kL1Offset),
                                AscendC::Te::MakeShape(curM, curGmAKL1));
            ::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = MakeLayoutBL1{}(curPadKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kL1Offset, 0),
                                AscendC::Te::MakeShape(curGmBKL1, curN));
            ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = CeilDiv(curGmBKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                // Each inner iteration slices the current L1 chunk into one
                // L0-sized MMAD tile and accumulates it into L0C.
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curPadKL1) ? (curPadKL1 - kL0Offset) : baseK_;
                uint64_t l0BufId = l0PingPong_ & 0x1;
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
                auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
                auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0);
                auto tensorAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1.Slice(AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN);
                auto tensorBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

                auto coordScaleKL1 = scaleWindowIter * kL1ScaleSize_;
                auto layoutScaleAL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto layoutScaleAL1 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curM, scaleKL1ScaleSize_);
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, coordScaleKL1),
                    AscendC::Te::MakeShape(curM, kL1ScaleSize_));
                auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleA3510, tensorScaleAL0, tensorBlockScaleAL1,
                    AscendC::Te::MakeCoord(0, kL0Offset));

                auto layoutScaleBL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
                auto tensorScaleBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, fp8_e8m0_t>(l0Offset), layoutScaleBL0);
                auto layoutScaleBL1 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(scaleKL1ScaleSize_, curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(coordScaleKL1, 0),
                    AscendC::Te::MakeShape(kL1ScaleSize_, curN));
                auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleB3510, tensorScaleBL0, tensorBlockScaleBL1,
                    AscendC::Te::MakeCoord(kL0Offset, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKidx == -1 || splitKidx == -2)) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0) &&
                    (splitKidx == 0 || (!enableAtomicAdd && splitKidx == -1) || (enableAtomicAdd && splitKidx == -2));
                AscendC::Te::MmadParams mmadParams;
                mmadParams.m = static_cast<uint16_t>(curM);
                mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
                mmadParams.n = static_cast<uint16_t>(curN);
                mmadParams.unitFlag = mmadUnitFlag;
                mmadParams.cmatrixInitVal = mmadCmatrixInitVal;
                AscendC::Te::Mmad(
                    AscendC::Te::MmadAtom<
                        AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitMX>>{}
                    .with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++;
            }

            // Release the current L1 slot only after every L0 slice derived
            // from it has completed its MMAD accumulation.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if (scaleWindowIter + 1 == scaleL1Window_ || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                scaleLoopCnt_++;
                scaleWindowIter = 0;
            } else {
                ++scaleWindowIter;
            }
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        if (splitKidx == -1 || splitKidx == -2) {
            if (enableAtomicAdd) {
                AscendC::DisableDmaAtomic();
                AscendC::SetAtomicAdd<CType>();
            }
            AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
            if (enableAtomicAdd) {
                AscendC::DisableDmaAtomic();
            }
        }
    }

private:
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t scaleL1Window_ = 0UL;
    uint64_t kL1ScaleSize_ = 0UL;
    uint64_t scaleKL1Group_ = 0UL;
    uint64_t scaleKL1ScaleSize_ = 0UL;
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleAOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleBOffset_[SCALE_BUFFER_NUM] = {0UL};
};
}  // namespace Block

