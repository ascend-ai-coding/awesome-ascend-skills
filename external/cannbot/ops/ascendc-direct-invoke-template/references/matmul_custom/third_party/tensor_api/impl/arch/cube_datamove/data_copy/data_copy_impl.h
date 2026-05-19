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
    "tensor_api/impl/arch/cube_datamove/data_copy/data_copy_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_IMPL_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_IMPL_H

#include "impl/arch/cube_datamove/data_copy/data_copy_routing.h"

namespace AscendC {
namespace Te {

constexpr DataCopyTrait DEFAULT_DATA_COPY_TRAIT;

template <typename T, typename U>
constexpr bool VerifyingDataCopyTemplate = (IsTileTensorV<T> && IsTileTensorV<U>);

template <typename T, typename U, typename Coord>
constexpr bool VerifyingDataCopyTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingDataCopyTemplate<T, U>;

template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplate<T, U>, void>::type
DataCopy(const T& dst, const U& src)
{
    constexpr Hardware dstTPos = GetHardPos<T>();
    constexpr Hardware srcTPos = GetHardPos<U>();
    using Tensor2Tensor = typename
        DataCopyTensor2Tensor<dstTPos, srcTPos, CURRENT_ARCH_VERSION>::type;
    Tensor2Tensor{}.template Run<trait, T, U>(dst, src);
}

template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplateWithCoord<T, U, Coord>, void>::type
DataCopy(const T& dst, const U& src, const Coord& coord)
{
    auto sliceTensor = src(coord, dst);
    DataCopy<trait, T, decltype(sliceTensor)>(dst, sliceTensor);
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
