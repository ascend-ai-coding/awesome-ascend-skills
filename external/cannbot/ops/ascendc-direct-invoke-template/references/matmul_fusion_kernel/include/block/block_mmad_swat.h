/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================
// BlockMmad SWAT 偏特化 — 核心流水线实现
//
// [SCOPE] 本 skill 仅用于 mxfp8 + eltwise 融合算子。
//         单算子（L0C→GM）路径已移除，本文件 L0C 输出**恒定**为 Fixpipe→UB。
//
// 修改清单：
// [MODIFY 1] btBuffCtrl 参数 — 关联 transB（第 6 处转置修改点）
// [PATTERN]  GM→L1→L0→MMAD→L0C→UB 五级流水线逻辑，所有融合算子共用
// ============================================================

/*!
 * \file block_mmad_swat.h
 * \brief Block-level MX MMAD pipeline for the SWAT non-full-load path.
 */

#ifndef BLOCK_MMAD_SWAT_H
#define BLOCK_MMAD_SWAT_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "tensor.h"
#include "block_mmad.h"
#include "../policy/dispatch_policy.h"
#include "../utils/quant_matmul_constant.h"
#include "../utils/constants.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/copy_scale_gm_to_l1.h"
#include "../tile/pad_mx_kl1.h"

namespace Block {
using namespace AscendC;

// [MODIFY] SplitM 模式 FixpipeTrait — L0C→UB 按 M 维度拆分到双 AIV
constexpr AscendC::Te::FixpipeTrait FIXPIPE_TRAIT_SPLIT_M(
    AscendC::Te::RoundMode::DEFAULT,
    false,
    false,
    AscendC::Te::DualDstMode::DUAL_DST_SPLIT_M
);

template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<
            QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, DispatchPolicy_::stages>,
            DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr bool isDTypeFp4 = AscendC::IsSameType<AType, fp4x2_e1m2_t>::value ||
        AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
    static constexpr uint64_t L1_BUFFER_NUM = DispatchPolicy::stages;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1;
    static constexpr uint64_t L1_BUFFER_GROUP_NUM = L1_BUFFER_NUM >> 1;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
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
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};

    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    // Params 仅保留 matmul 必需的 5 个 GM 地址。
    // 注：cGmAddr 在融合模式下**不会被 BlockMmad 用于写回**，
    //     仅用于保持 operator() 调用签名 (传入 gmC)。
    //     Epilogue 会自己持有 outputGmAddr 负责实际写出。
    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
    };

    __aicore__ inline BlockMmad()
    {
        // Prime all producer/consumer events so the first iteration can enter
        // the pipelined copy-and-compute loop without special-case branches.
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
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
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape,
        const L1Params& l1Params, bool enableL0cPingPong)
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

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, BlockShape singleShape)
    {
        // Non-full-load streams both A and B through L1/L0 in chunks. Scale
        // tensors advance in a coarser cadence that matches `scaleKL1_`.
        (void)gmC;   // 融合模式下不直接写 GM，保留参数仅为签名兼容
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        // [FIX SPLIT_M ODD-M] Fixpipe DUAL_DST_SPLIT_M splits the L0C M-axis evenly
        // across the two AIVs, which requires the L0C view's M to be even.
        // For odd `curM` (e.g. M=1 tail tile), round the L0C / UB view up to
        // `curMPad` so SPLIT_M can dispatch (curMPad/2) rows per AIV.
        // MMAD still uses `curM` so only those rows hold valid matmul results.
        // The Epilogue's existing SubBlockIdx logic
        //   blockShapeM = (curM & 1) ? halfM - SubBlockIdx : halfM
        //   (with halfM = ceil(curM/2))
        // assigns AIV0 -> halfM rows (UB[0..halfM)) and AIV1 -> halfM-1 rows
        // (UB[0..halfM-1)). Since halfMPad = curMPad/2 = halfM, AIV1 reads the
        // first halfM-1 rows of its UB half, which still map to valid L0C rows
        // [halfM..curM-1]; the lone garbage padded row at L0C[curM] sits at the
        // tail of AIV1's UB half and is never copied out to GM.
        auto curMPad = (curM + 1L) & ~1L;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curMPad, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);
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

                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
                auto layoutScaleAL1 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, scaleKL1Group_);
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto gmBlockScaleA = gmScaleA(
                    AscendC::Te::MakeCoord(0, kL1Offset / MXFP_GROUP_SIZE),
                    AscendC::Te::MakeShape(
                        curM, CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

                auto layoutScaleBL1 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(scaleKL1Group_, curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB(
                    AscendC::Te::MakeCoord(kL1Offset / MXFP_GROUP_SIZE, 0),
                    AscendC::Te::MakeShape(
                        CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = MakeLayoutAL1{}(curM, curPadKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, kL1Offset), 
                                AscendC::Te::MakeShape(curM, curGmAKL1));
            if constexpr (!isDTypeFp4) {
                ::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            }
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(kL1Offset, 0), 
                                AscendC::Te::MakeShape(curGmBKL1, curN));
            if constexpr (!isDTypeFp4) {
                ::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            }
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

                auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);
                auto tensorAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1(AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
                auto tensorBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1(AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1);

                auto coordScaleKL1 = scaleWindowIter * kL1ScaleSize_;
                auto layoutScaleAL0 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto layoutScaleAL1 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, scaleKL1ScaleSize_);
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto tensorBlockScaleAL1 = tensorScaleAL1(
                    AscendC::Te::MakeCoord(0, coordScaleKL1),
                    AscendC::Te::MakeShape(curM, kL1ScaleSize_));
                auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleA3510, tensorScaleAL0, tensorBlockScaleAL1,
                    AscendC::Te::MakeCoord(0, kL0Offset));

                auto layoutScaleBL0 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
                auto tensorScaleBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<>((__cb__ fp8_e8m0_t*)(l0Offset)), layoutScaleBL0);
                auto layoutScaleBL1 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(scaleKL1ScaleSize_, curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto tensorBlockScaleBL1 = tensorScaleBL1(
                    AscendC::Te::MakeCoord(coordScaleKL1, 0),
                    AscendC::Te::MakeShape(kL1ScaleSize_, curN));
                auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
                AscendC::Te::Copy(
                    CopyL12L0MxScaleB3510, tensorScaleBL0, tensorBlockScaleBL1,
                    AscendC::Te::MakeCoord(kL0Offset, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                // [MODIFY] btBuffCtrl: 从硬编码 false 改为 transB，支持转置模式
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Tile::MmadMx>>{}.with(
                        static_cast<uint16_t>(curM),
                        static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE)),
                        static_cast<uint16_t>(curN), mmadUnitFlag, false, mmadCmatrixInitVal),
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

        // [PATTERN] L0C → UB (Fixpipe NZ→ND, SplitM 双 AIV)
        // 融合算子唯一输出路径；CV 同步由 Kernel 层统一管理。
        // gmC 参数仅用于签名兼容，本步骤不使用。
        // [FIX SPLIT_M ODD-M] UB 行数使用 curMPad 保证 fixpipe 内部
        // mSize = min(srcL0C_M, dstUB_M) 为偶数，从而 SPLIT_M 能等分到双 AIV。
        // [FIX ODD-N / N-ALIGN] UB 列数对齐到 32B（= 8 个 float），与 Epilogue 计算
        // nAlign = CeilDiv(curN, 32/sizeof(float)) * (32/sizeof(float)) 的 UB 行
        // stride 假设保持一致；否则 curN 非 8 的倍数（包括 N=1、N=odd、N=15、N=33 等）
        // 时 fixpipe 写 UB 用的 row stride = curN 与 Epilogue 读 UB 用的 row stride =
        // nAlign 不匹配，导致从第 1 行起所有数据错位。fixpipe 内部 nSize =
        // min(srcL0C_N=curN, dstUB_N=curNUbAlign) = curN，每行只填充 curN 列；尾部
        // (curNUbAlign - curN) 列保持未触碰，Epilogue 通过 DataCopyPad 用
        // rowBytes = curN * sizeof(float) 仅写出有效列，不会泄漏到 GM。
        constexpr int64_t UB_N_ALIGN_ELEM = 32L / static_cast<int64_t>(sizeof(float));
        auto curNUbAlign = ((curN + UB_N_ALIGN_ELEM - 1L) / UB_N_ALIGN_ELEM) * UB_N_ALIGN_ELEM;
        auto layoutUB = AscendC::Te::MakeNDLayout<float>(curMPad, curNUbAlign);
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeUBmemPtr<float>(0), layoutUB);
        AscendC::Te::Fixpipe<FIXPIPE_TRAIT_SPLIT_M>(ubTensor, tensorL0C,
            AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
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

#endif
