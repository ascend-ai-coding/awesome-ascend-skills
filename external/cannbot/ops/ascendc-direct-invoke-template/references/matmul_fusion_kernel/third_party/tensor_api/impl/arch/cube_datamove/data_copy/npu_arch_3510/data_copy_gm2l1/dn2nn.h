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
 * \file dn2nn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_DN2NN_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_DN2NN_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufScaleBDN2Nn {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        DataCopyImpl<trait>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckDataTypeFor3510::CheckGm2L1ScaleDataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto srcRowShape = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcColShape = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint16_t srcBColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
        uint16_t dstBColStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint16_t dnNum = 1;
        uint16_t nValue = srcRowShape >> 1;  // use b16 for DN2NZ, so nValue = srcRowShape / 2
        uint16_t dValue = srcColShape;
        uint16_t dstNzNStride = 1;

        uint64_t loop1SrcStride = srcBColStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;                             // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstBColStride * sizeof(type) / C0_SIZE<>; // loop3_dst_stride = dst_nz_c0_Stride
        uint16_t loop4DstStride = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src);
        // fp8 scale use b16 for movement
        CopyGmToCbufMultiDn2nzInstr::CopyGmToCbufMultiDn2nz(
            (__cbuf__ half*)(dst.Data().Get()), (__gm__ half*)(src.Data().Get()), dnNum, loop2DstStride, loop3DstStride,
            loop4DstStride, loop1SrcStride, cacheMode, nValue, dValue, 0, false);
    }
};

} // namespace Te
} // namespace AscendC

#endif
