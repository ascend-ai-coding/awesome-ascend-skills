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
    "tensor_api/impl/arch/cube_datamove/load_data/load_data_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file load_data_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_IMPL_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_IMPL_H

#include "impl/arch/cube_datamove/load_data/load_data_routing.h"

namespace AscendC {
namespace Te {

constexpr LoadDataTrait DEFAULT_LOAD_DATA_TRAIT;

template <typename T, typename U>
constexpr bool VerifyingLoadDataTemplate = IsTileTensorV<U> && IsTileTensorV<T>;

template <typename T, typename U, typename Coord>
constexpr bool VerifyingLoadDataTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingLoadDataTemplate<T, U>;

template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingLoadDataTemplate<T, U>, void>::type 
LoadData(const T& dst, const U& src)
{
    constexpr Hardware dstPos = GetHardPos<T>();
    constexpr Hardware srcPos = GetHardPos<U>();
    using Tensor2Tensor = typename LoadDataTensor2TensorNoCoord<dstPos, srcPos, CURRENT_ARCH_VERSION>::type;
    Tensor2Tensor{}.template Run<trait, T, U>(dst, src);
}

template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U, class Coord>
__aicore__ inline typename Std::enable_if<VerifyingLoadDataTemplateWithCoord<T, U, Coord>, void>::type 
LoadData(const T& dst, const U& src, const Coord& coord)
{
    constexpr Hardware dstPos = GetHardPos<T>();
    constexpr Hardware srcPos = GetHardPos<U>();
    using Tensor2Tensor = typename LoadDataTensor2Tensor<dstPos, srcPos, CURRENT_ARCH_VERSION>::type;
    Tensor2Tensor{}.template Run<trait, T, U, Coord>(dst, src, coord);
}
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
