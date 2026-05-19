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
 * \file nn2nn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_NN2NN_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_NN2NN_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufScaleBNn2Nn {
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

        auto srcShapeColB = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        auto srcShapeColS = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
        auto srcShapeRowB = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcStrideColB = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
        auto srcStrideColS = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 0>(srcLayout);

        auto dstStrideColB = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint8_t cacheMode = GetCacheModeFromTensor(src);

        // lprp mode, dst_stride % C0_SIZE should be 0
        // multi rows copy, dst non-contiguous case
        uint32_t blockCount = srcShapeColB;
        uint32_t blockLen = srcShapeRowB * sizeof(type) * srcShapeColS * srcStrideColS;
        uint64_t srcStride = srcStrideColB * sizeof(type);
        uint32_t dstStride = dstStrideColB * sizeof(type);

        uint8_t leftPaddingCnt = 0;
        uint8_t rightPaddingCnt = 0;
        CopyGmToCbufAlignV2Base::CopyGmToCbufAlignV2((__cbuf__ half*)(dst.Data().Get()),
                                                     (__gm__ half*)(src.Data().Get()), blockCount, blockLen,
                                                     leftPaddingCnt, rightPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif
