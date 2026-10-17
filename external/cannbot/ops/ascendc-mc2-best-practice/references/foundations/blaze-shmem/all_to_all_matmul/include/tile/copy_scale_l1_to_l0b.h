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
 * \file copy_scale_l1_to_l0b.h
 * \brief Tile helper that copies MX scaleB data from L1 to L0B.
 */

#pragma once
#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "../utils/constant.h"

namespace Tile {
struct CopyL12L0MxScaleB3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        // `coord` is expressed in the original K/N element space; the helper
        // converts it to the packed MX scale coordinates expected by the L0B
        // scale layout and issues one hardware MX load.
        // (n1, k/64, n0, 2)
        // shape ((2, k/64), (n0, n1))
        // stride ((2, k/64*n0*2), (1, n0*2))
        // Nn -> Nn
        uint16_t nStartPosition = CeilDiv(AscendC::Std::get<1>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = CeilDiv(AscendC::Std::get<0>(coord), MXFP_DIVISOR_SIZE);
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        asc_copy_l12l0b_mx(mxDstAddr, src.Data().Get(), nStartPosition, kStartPosition, nStep, kStep,
            srcStride, dstStride);
    }
};

// Expose this helper through TE's generic copy-trait interface.
} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleB3510>
    : public CopyTraits<
        ::Tile::CopyL12L0MxScaleB3510, CopyL12L0BTraitDefault, ::Tile::CopyL12L0MxScaleB3510,
        CopyL12L0BTraitDefault> {};

