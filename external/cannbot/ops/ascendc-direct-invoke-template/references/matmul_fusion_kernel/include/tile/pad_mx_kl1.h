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
// [PATTERN] Tile 层 — 硬件指令封装
// 本文件封装底层 Cube 硬件指令，通常不需要修改。
// 所有 matmul 变体共用此层，通过上层模板参数自动适配。
// ============================================================

/*!
 * \file pad_mx_kl1.h
 * \brief Zero-pad A/B L1 buffers along K when GM slices are shorter than the L1-aligned layout.
 */
#ifndef TILE_PAD_KL1_H
#define TILE_PAD_KL1_H

#include "kernel_utils/common_utils.h"
#include "impl/atom/copy_traits_impl.h"

using AscendC::Te::AttrInfo;
using AscendC::Te::C0_SIZE;
using AscendC::Te::GetEleFromLayout;

namespace Tile {
struct PadMxKL1Base {
    template <typename T>
    __aicore__ inline static void PadZero(
        const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap)
    {
        create_cbuf_matrix((__cbuf__ half*)tensorL1.Data().Get(), (blockNum << 16) | (dstGap << 32) | repeatTimes, 0);
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp4()
    {
        using type = typename T::elementType;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp4x2_e1m2_t, __cbuf__ fp4x2_e2m1_t>;
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp8()
    {
        using type = typename T::elementType;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp8_e5m2_t, __cbuf__ fp8_e4m3fn_t>;
    }
};

struct PadMxKAL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only supports MXFP4/MXFP8 L1 tensors.");
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();
        auto kAxis = GetEleFromLayout<decltype(layoutGm), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutGm);
        auto kAxisL1Align = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(layoutL1) *
                            GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);

        if constexpr (AscendC::Te::IsNDFormat<U>::value) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }

            // ND2NZ already handles innermost-axis padding. When the remaining
            // K tail spans a full C0 block, clear that outer tail explicitly.
            auto mAlign = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 0>(layoutL1) *
                          GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(0, kAxisND2NZAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
        } else if constexpr (AscendC::Te::IsDNFormat<U>::value) {
            if (kAxis == kAxisL1Align) {
                return;
            }

            // DN2NZ can only zero-pad the innermost m0 axis. Clear the K-axis
            // tail across each outer m1 slice of the A-side NZ layout.
            auto m1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(0, kAxis));
            PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, kAxis);
        }
    }
};

struct PadMxKBL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only supports MXFP4/MXFP8 L1 tensors.");
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();

        auto kAxis = GetEleFromLayout<decltype(layoutGm), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutGm);
        auto kAxisL1Align = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 0>(layoutL1) *
                            GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::ROW, 1>(layoutL1);

        if constexpr (AscendC::Te::IsNDFormat<U>::value) {
            if (kAxis == kAxisL1Align) {
                return;
            }

            // ND2NZ can only zero-pad the innermost n0 axis. Clear the K-axis
            // tail across each outer n1 slice of the B-side NZ layout.
            auto n1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxis, 0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        } else if constexpr (AscendC::Te::IsDNFormat<U>::value) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }

            // For FP8 DN input, clear any full-C0 outer K tail from the
            // ND2NZ-aligned K boundary.
            auto nAlign = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(layoutL1) *
                          GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxisND2NZAlign, 0));
            PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0);
        } else if constexpr (AscendC::Te::IsNZFormat<U>::value) {
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::BLOCK_CUBE);
            if (kAxisND2NZAlign == kAxisL1Align) {
                return;
            }

            // NZ GM slices already expose blocked K coordinates. Clear the
            // remaining K-axis tail across each outer n1 slice.
            auto n1 = GetEleFromLayout<decltype(layoutL1), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layoutL1);
            auto sliceTensor = tensorL1(AscendC::Te::MakeCoord(kAxis, 0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        }
    }
};
} // namespace Tile

#endif // TILE_PAD_KL1_H
