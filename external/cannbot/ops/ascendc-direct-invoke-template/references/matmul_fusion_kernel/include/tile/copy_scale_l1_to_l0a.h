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
 * \file copy_scale_l1_to_l0a.h
 * \brief Tile helper that copies MX scaleA data from L1 to L0A.
 */

#ifndef MATMUL_TILE_DATAMOVE_COPY_L1_TO_L0A_H
#define MATMUL_TILE_DATAMOVE_COPY_L1_TO_L0A_H

#include "impl/atom/cube_datamove/copy_l12l0.h"
#include "kernel_utils/common_utils.h"
#include "../utils/quant_matmul_constant.h"

namespace Tile {
struct CopyL12L0MxScaleA3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        static_assert(
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<dstType, srcType>, AscendC::Std::tuple<__ca__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
        // `coord` is expressed in the original M/K element space; the helper
        // converts it to the packed MX scale coordinates expected by the L0A
        // scale layout and issues one hardware MX load.
        // (m1, k/64, m0, 2)
        // shape ((m0, m1), (2, k/64))
        // stride ((2, k/64*m0*2), (1, m0*2))
        // Zz -> Zz
        uint16_t mStartPosition = CeilDiv(AscendC::Std::get<0>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = CeilDiv(AscendC::Std::get<1>(coord), MXFP_DIVISOR_SIZE);
        auto mStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_ca_mx(
            mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()), mStartPosition, kStartPosition, mStep, kStep,
            srcStride, dstStride);
    }
};

// Expose this helper through TE's generic copy-trait interface.
} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleA3510>
    : public CopyTraits<
        ::Tile::CopyL12L0MxScaleA3510, LoadDataTraitDefault, ::Tile::CopyL12L0MxScaleA3510,
        LoadDataTraitDefault> {};

#endif
