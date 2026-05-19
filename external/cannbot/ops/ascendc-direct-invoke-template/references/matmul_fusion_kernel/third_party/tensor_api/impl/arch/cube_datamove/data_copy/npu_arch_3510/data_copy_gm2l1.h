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
    "tensor_api/impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy_gm2l1.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2nz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2zn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nd_onedim.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nd.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2zn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nz2nz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2zz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2zz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/zz2zz.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nd2nn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/dn2nn.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1/nn2nn.h"

namespace AscendC {
namespace Te {

class DataCopyGM2L13510 {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src)
    {
        Execute<trait>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Execute(const T& dst, const U& src)
    {
        if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            // ND2ND
            if constexpr (IsNDFormat<U>::oneDimValue) {
                // ND2ND with one dimension
                CopyGmToCbufAlignV2NDOneDim::Run<trait, T, U>(dst, src);
            } else {
                CopyGmToCbufAlignV2ND::Run<trait, T, U>(dst, src);
            }
        } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
            // ND2Nz
            CopyGmToCbufMultiND2Nz::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsZNFormat<T>::value) {
            // ND2Zn
            CopyGmToCbufMultiND2Zn::Run<trait, T, U>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsNZFormat<T>::value) {
            // DN2Nz
            CopyGmToCbufMultiDN2Nz::Run<trait, T, U>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsZNFormat<T>::value) {
            // DN2Zn
            CopyGmToCbufMultiDN2Zn::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            // Nz2Nz
            CopyGmToCbufAlignV2NZ::Run<trait, T, U>(dst, src);
        } else if constexpr (CheckArrangement<U>::IsScaleType()) {
            // ScaleA/ScaleB
            ExecuteScaleDataCopy<trait, T, U>(dst, src);
        } else {
            // assert error
            static_assert(Std::is_same_v<T, U>, "The data format is not supported.");
        }
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void ExecuteScaleDataCopy(const T& dst, const U& src)
    {
        if constexpr (IsScaleANDFormat<U>::value && IsZZFormat<T>::value) {
            // ScaleAND2Zz
            CopyGmToCbufScaleAND2Zz::Run<trait, T, U>(dst, src);
        } else if constexpr (IsScaleADNFormat<U>::value && IsZZFormat<T>::value) {
            // ScaleADN2Zz
            CopyGmToCbufScaleADN2Zz::Run<trait, T, U>(dst, src);
        } else if constexpr (IsZZFormat<U>::value && IsZZFormat<T>::value) {
            // ScaleAZz2Zz
            CopyGmToCbufScaleAZz2Zz::Run<trait, T, U>(dst, src);
        } else if constexpr (IsScaleBNDFormat<U>::value && IsNNFormat<T>::value) {
            // ScaleBND2NN
            CopyGmToCbufScaleBND2Nn::Run<trait, T, U>(dst, src);
        } else if constexpr (IsScaleBDNFormat<U>::value && IsNNFormat<T>::value) {
            // ScaleBDN2NN
            CopyGmToCbufScaleBDN2Nn::Run<trait, T, U>(dst, src);
        } else if constexpr (IsNNFormat<U>::value && IsNNFormat<T>::value) {
            // ScaleBNN2NN
            CopyGmToCbufScaleBNn2Nn::Run<trait, T, U>(dst, src);
        } else {
            // assert error
            static_assert(Std::is_same_v<T, U>, "The data format is not supported.");
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
