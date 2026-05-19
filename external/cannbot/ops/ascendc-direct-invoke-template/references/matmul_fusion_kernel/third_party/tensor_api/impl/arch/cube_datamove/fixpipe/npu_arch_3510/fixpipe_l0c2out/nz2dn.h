/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "tensor_api/impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out/nz2dn.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nz2dn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_NZ2DN_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_NZ2DN_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class Fixpipe2OutNz2Dn3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename... Params>
    __aicore__ inline static void Run(const T& dst, const U& src, const Params&... params) {
        SetRegisterImpl<trait, T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U>(dst, src, params...);
    }

private:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckDNTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
        if constexpr (GetHardPos<T>() == Hardware::GM) {
            CheckDataTypeFor3510::CheckL0C2GmDataType<quantPre, T, U>();
        } else {
            CheckDataTypeFor3510::CheckL0C2UbDataType<quantPre, T, U>();
        }
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline static void SetRegisterImpl(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNzMatrixStride = 0;
        uint32_t dstDnMatrixStride = 0;
        uint32_t srcNzC0Stride = 1;
        SetRegister3510::SetRegister(dnNum, dstDnMatrixStride, srcNzMatrixStride, srcNzC0Stride);
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src, const FixpipeParams& params)
    {
        CheckTemplate<trait, quantPre, T, U>();
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint32_t nSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout));
        uint32_t mSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout));
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        if constexpr (GetHardPos<T>() == Hardware::GM) {
            uint8_t cacheMode = GetCacheModeFromTensor(dst);
            bool isChannelSplit = trait.enableChannelSplit;
            CopyMatrixCcToGm3510::DataCopy<quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride,
                                                                  cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn,
                                                                  nz2dnEn);
        } else {
            uint8_t dualDstCtl = trait.dualDstCtl;
            bool subBlockId = false;
            CopyMatrixCcToUb3510::DataCopy<quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride,
                                                                      dualDstCtl, reluEn, unitFlag, subBlockId, nz2ndEn,
                                                                      nz2dnEn);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_NZ2DN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
