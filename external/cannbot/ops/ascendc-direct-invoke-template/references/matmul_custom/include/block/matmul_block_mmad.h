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
 * \file matmul_block_mmad.h
 * \brief Block-level MMAD copy/compute pipeline (GM->L1->L0A/L0B->L0C->GM).
 */

#ifndef MATMUL_BLOCK_MMAD_H
#define MATMUL_BLOCK_MMAD_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "block_mmad.h"
#include "../policy/dispatch_policy.h"
#include "../utils/matmul_constant.h"

// ============================================================================
// Matmul BlockMmad —— 单 block (baseM x baseN) 的数据搬运与 MMAD 流水
//
// 职责：
//   - L1 ping-pong 双缓冲（half-L1 = A|B 一组）
//   - GM -> L1 搬运 A/B（NZ / ZN 格式）
//   - L1 -> L0A/L0B 加载，按 baseK 切分
//   - 调用 tensor_api 的 `Mad()`，在 L0C 上累加（fp32 或 int32，由 `L0CType` 决定）
//   - 最后一次累加后 fixpipe 写回 GM（L0C -> CType，CType 决定 quantPre）
//
// [MODIFY] 新算子常见改点：
//   1. 如果要加入 Bias / Activation / Cast，需要在 fixpipe 调用处扩展 FixpipeParams
//      并在 L1 预留 bias table 空间（此处未支持）。
//   2. 如需 4-stage L1 流水，改 `L1_BUFFER_NUM = 4`，并在 dispatch policy 里加 STAGES 模板参数。
//   3. 如需不同 MMAD Trait（量化/MX），可替换 `AscendC::Te::MmadOperation` 为自定义
//      MmadTrait 特化（见 tensor_api/tile_mmad_*.h 示例）。
// ============================================================================

namespace Block {
using namespace AscendC;

template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>, DispatchPolicy_>>> {
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

    // [MODIFY] L1 buffer 数量——典型 2 (double-buffer) 或 4 (quad-buffer)。
    // 提高 L1_BUFFER_NUM 可加深流水掩盖 MTE2 搬运延迟，但会减少每个 buffer 容量。
    static constexpr uint64_t L1_BUFFER_NUM = 2UL;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1UL;
    static constexpr uint64_t L1_BUFFER_GROUP_NUM = L1_BUFFER_NUM >> 1;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    // [CONFIG] L0C 累加 dtype。bf16/fp16/fp8 输入用 fp32 累加；int8 输入用 int32 累加。
    // 硬件 MMAD/Fixpipe 静态检查会拒绝 int8 -> float 组合。
    using L0CType = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AType, int8_t>, int32_t, float>;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(L0CType);
    // [CONFIG] C0 cube granularity: bf16/fp16 = 16; int8/fp8 = 32; fp4 = 64。
    static constexpr uint64_t BLOCK_CUBE = 16UL;

    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};

    // A 按 NZ，B 按 ZN 存进 L1（Cube 的固定输入格式）。
    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        // [CONFIG] kL1 = baseK * stepK，代表单次 L1 流水覆盖的 K 长度。
        uint64_t kL1;
    };

    __aicore__ inline BlockMmad()
    {
        // 预置 MTE1<->MTE2 事件，使第一次迭代不必特殊分支。
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
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
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        enableL0cPingPong_ = enableL0cPingPong;
        // [PITFALL] MakeL1memPtr<T>(offset) 接受的是 **字节偏移**。
        // aL1OneBuffer_/bL1OneBuffer_ 必须显式乘 sizeof(AType)/sizeof(BType)。
        aL1OneBuffer_ = baseM_ * kL1_ * sizeof(AType);
        bL1OneBuffer_ = baseN_ * kL1_ * sizeof(BType);
        // L1 布局： [A0|B0] | [A1|B1] —— 上下两半，每半一组 ping/pong。
        uint64_t l1HalfSize = AscendC::TOTAL_L1_SIZE >> 1;
        #pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            uint64_t l1BufferGroup = bufferId >> 1;
            uint64_t l1HalfOffset = (bufferId & 1UL) * l1HalfSize;
            l1BufferAOffset_[bufferId] = l1HalfOffset + l1BufferGroup * aL1OneBuffer_;
            l1BufferBOffset_[bufferId] = l1HalfOffset + L1_BUFFER_GROUP_NUM * aL1OneBuffer_ +
                l1BufferGroup * bL1OneBuffer_;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorC gmC, BlockShape singleShape)
    {
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<L0CType>(l0cOffset), layoutL0C);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t kL1Offset = iter0 * kL1_;
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;

            // ---- GM -> L1 搬运 A, B ----
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, kL1Offset),
                                AscendC::Te::MakeShape(curM, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(kL1Offset, 0),
                                AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            // ---- 内层：按 baseK 切块，L1 -> L0A/L0B -> MMAD 累加 ----
            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curKL1) ? (curKL1 - kL0Offset) : baseK_;
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

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                // unitFlag 告诉硬件是否是整条 K 归并的最后一条 MMAD。
                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                // tensor_api 通用 Mmad 入口（非量化），非 mx 场景 m/k/n 直接传实际值，无需 16 对齐。
                AscendC::Te::MmadParams mmadParams{
                    static_cast<uint16_t>(curM),
                    static_cast<uint16_t>(curN),
                    static_cast<uint16_t>(curKL0),
                    mmadUnitFlag,
                    mmadCmatrixInitVal};
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{},
                    tensorL0C, tensorAL0, tensorBL0, mmadParams);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }

        // L0C -> GM，由 fixpipe 完成 L0C(fp32/int32) -> CType 的量化/cast。
        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
};
} // namespace Block

#endif // MATMUL_BLOCK_MMAD_H
