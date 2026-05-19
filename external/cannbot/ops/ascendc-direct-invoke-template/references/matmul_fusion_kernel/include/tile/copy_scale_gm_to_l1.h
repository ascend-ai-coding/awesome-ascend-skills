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
 * \file copy_scale_gm_to_l1.h
 * \brief Scale-copy helpers that move MX scale tensors from GM to L1.
 */
#ifndef TILE_COPY_SCALE_GM_TO_L1_H
#define TILE_COPY_SCALE_GM_TO_L1_H

#include "utils/utils.h"
#include "impl/atom/copy_traits_impl.h"

using AscendC::Te::AttrInfo;
using AscendC::Te::C0_SIZE;
using AscendC::Te::GetCacheModeFromTensor;
using AscendC::Te::GetEleFromLayout;
using AscendC::Te::IsScaleANDFormat;
using AscendC::Te::IsScaleBNDFormat;
using AscendC::Te::IsZZFormat;
using AscendC::Te::MX_SCALE_K0;

namespace Tile {
// This helper normalizes all supported GM scale layouts into the L1 layouts
// expected by the shared MX block pipeline.
struct CopyScaleGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        if constexpr (IsZZFormat<T>::value) {
            if constexpr (IsScaleANDFormat<U>::value) {
                CopyScaleADn2nz<Tp, traits, T, U>(dst, src);
            } else {
                // The ND variant is only used by layouts that already expose
                // the expanded 64-element MX scale width on the GM side.
                CopyScaleANd2nz<Tp, traits, T, U>(dst, src);
            }
        } else {
            if constexpr (IsScaleBNDFormat<U>::value) {
                CopyScaleBNd2nz<Tp, traits, T, U>(dst, src);
            } else {
                CopyScaleBDn2nz<Tp, traits, T, U>(dst, src);
            }
        }
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleADn2nz(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // scaleA in DN format stores one logical MX scale value for every
        // 32-K group, while the destination expects an NZ-style packed layout.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 1,m,1,k/32, stride: 0,m,0,1

        uint16_t nValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufDn2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleANd2nz(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // The ND flavor already exposes the expanded 64-K divisor width, so
        // only the source stride interpretation changes during repacking.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 1,m,2,k/64, stride: 0,2,1,2*m

        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);

        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufNd2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleBDn2nz(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // scaleB flips the major axis relative to scaleA, so D/N extraction
        // mirrors the A-side helper even though the copy primitive is shared.
        auto dstLayout = dst.Layout(); // shape: 2,k/64,16,n/16, stride: 1,32,2,(k/32)*16
        auto srcLayout = src.Layout(); // shape: 1,k/32,1,n stride: 0,1,0,k/32

        uint16_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t nValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufDn2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyScaleBNd2nz(const T& dst, const U& src)
    {
        using type = typename U::elementType;
        static_assert(AscendC::Std::is_same_v<type, __gm__ fp8_e8m0_t>, "The data type is not supported.");

        // This is the B-side counterpart of `CopyScaleANd2nz`, with row/column
        // semantics swapped to match the filter-major scale layout.
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout(); // shape: 2,k/64,1,n, stride: 1,2*n,0,2

        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        uint64_t srcDValue =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) / MX_SCALE_K0;
        uint16_t dstNzC0Stride =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) / C0_SIZE<T>;
        CopyGmToCbufNd2nz<Tp, traits, T, U>(dst, src, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyGmToCbufDn2nz(
        const T& dst, const U& src, uint16_t nValue, uint32_t dValue, uint64_t srcDValue, uint16_t dstNzC0Stride)
    {
        uint16_t dnNum = 1;
        uint64_t srcDnMatrixStride = 0;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(half);
        uint64_t loop4SrcStride = srcDnMatrixStride * sizeof(half);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_SIZE<T>
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(half) / C0_SIZE<T>);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        // The hardware DN2NZ DMA expects a packed register describing the
        // destination NZ strides. The helper derives those fields once here.
        uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(dnNum);                        // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara); // CCE: store parameters for DN2NZ DMA instructions
        copy_gm_to_cbuf_multi_dn2nz(
            (__cbuf__ half*)dst.Data().Get(), (__gm__ half*)src.Data().Get(), 0, loop1SrcStride, cacheMode, nValue,
            dValue, loop4SrcStride, false);
    }

    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void CopyGmToCbufNd2nz(
        const T& dst, const U& src, uint16_t nValue, uint32_t dValue, uint64_t srcDValue, uint16_t dstNzC0Stride)
    {
        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;

        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(half);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(half);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_SIZE<T>
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(half) / C0_SIZE<T>);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        // The ND2NZ path uses the same register layout as DN2NZ; only the
        // source indexing logic differs in how ND tiles are walked in GM.
        uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        copy_gm_to_cbuf_multi_nd2nz(
            (__cbuf__ half*)dst.Data().Get(), (__gm__ half*)src.Data().Get(), 0, loop1SrcStride, cacheMode, nValue,
            dValue, loop4SrcStride, false);
    }
};
} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyScaleGM2L1>
    : public CopyTraits<
        ::Tile::CopyScaleGM2L1, AscendC::Te::LoadDataTraitDefault, ::Tile::CopyScaleGM2L1,
        AscendC::Te::LoadDataTraitDefault> {};

#endif // TILE_COPY_SCALE_GM_TO_L1_H
