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
 * \file pad_mx_kl1.h
 * \brief Zero-pad A/B L1 buffers along K when GM slices are shorter than the L1-aligned layout.
 */
#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
using AscendC::Te::C0_SIZE;

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
        auto kAxis = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(layoutGm);
        auto kAxisL1Align = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 0>(layoutL1) *
                            AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(layoutL1);

        if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<U, AscendC::Te::NDExtLayoutPtn>) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }
            auto mAlign = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 0>(layoutL1) *
                          AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>);
            auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(0, kAxisND2NZAlign), AscendC::Te::MakeShape(mAlign, kAxisL1Align - kAxisND2NZAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
        } else if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<U, AscendC::Te::DNExtLayoutPtn>) {
            if (kAxis == kAxisL1Align) {
                return;
            }

            // DN2NZ can only zero-pad the innermost m0 axis. Clear the K-axis
            // tail across each outer m1 slice of the A-side NZ layout.
            auto m1 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 1>(layoutL1);
            auto m0 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 0>(layoutL1);
            auto dstRowStride = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Stride, AscendC::Te::AttrInfo::Row, 1>(layoutL1);
 	        auto dstGap = (dstRowStride / C0_SIZE<T>) - kAxisL1Align + kAxis;
            auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(0, kAxis), AscendC::Te::MakeShape(m1 * m0, kAxisL1Align - kAxis));
            PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap);
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

        auto kAxis = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 1>(layoutGm);
        auto kAxisL1Align = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 0>(layoutL1) *
                            AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 1>(layoutL1);

        if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<U, AscendC::Te::NDExtLayoutPtn>) {
            if (kAxis == kAxisL1Align) {
                return;
            }
            // tail across each outer n1 slice of the B-side NZ layout.
            auto n1 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(layoutL1);
            auto n0 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 0>(layoutL1);
                        auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(kAxis, 0), AscendC::Te::MakeShape(kAxisL1Align - kAxis, n1 * n0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        } else if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<U, AscendC::Te::DNExtLayoutPtn>) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < C0_SIZE<T>) {
                return;
            }

            // For FP8 DN input, clear any full-C0 outer K tail from the
            auto nAlign = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 0>(layoutL1) *
                          AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, C0_SIZE<T>);
            auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(kAxisND2NZAlign, 0), AscendC::Te::MakeShape(kAxisL1Align - kAxisND2NZAlign, nAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0);
        } else if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<U, AscendC::Te::NZLayoutPtn>) {
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::BLOCK_CUBE);
            if (kAxisND2NZAlign == kAxisL1Align) {
                return;
            }

            // NZ GM slices already expose blocked K coordinates. Clear the
            // remaining K-axis tail across each outer n1 slice.
            auto n1 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(layoutL1);
            auto n0 = AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 0>(layoutL1);
                        auto sliceTensor = tensorL1.Slice(AscendC::Te::MakeCoord(kAxis, 0), AscendC::Te::MakeShape(kAxisL1Align - kAxis, n1 * n0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        }
    }
};
} // namespace Tile

