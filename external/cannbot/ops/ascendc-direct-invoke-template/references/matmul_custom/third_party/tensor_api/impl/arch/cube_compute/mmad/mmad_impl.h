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
    "tensor_api/impl/arch/cube_compute/mmad/mmad_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H

#include "impl/arch/cube_compute/mmad/mmad_routing.h"

namespace AscendC {
namespace Te {

constexpr MmadTrait DEFAULT_MMAD_TRAIT;

constexpr MmadParams defaultMmadParams = {0, 0, 0, 0, true};

constexpr MmadParams defaultMmadWithBiasParams = {0, 0, 0, 0, false};

template <typename T, typename U, typename S>
static constexpr bool VerifyingMmadTemplate = (IsTileTensorV<T> && IsTileTensorV<U> 
    && IsTileTensorV<S>);

template <typename T, typename U, typename S, typename V>
static constexpr bool VerifyingMmadWithBiasTemplate = (IsTileTensorV<T> && IsTileTensorV<U> 
    && IsTileTensorV<S> && IsTileTensorV<V>);

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename Params>
__aicore__ inline typename Std::enable_if<VerifyingMmadTemplate<T, U, S>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const Params& params)
{
   constexpr Hardware dstPos = GetHardPos<T>();
   constexpr Hardware fmPos = GetHardPos<U>();
   constexpr Hardware filterPos = GetHardPos<S>();
   using Tensor2Tensor = typename MmadTensor2Tensor<dstPos, fmPos, filterPos, Hardware::MAX, 
      CURRENT_ARCH_VERSION>::type;
   Tensor2Tensor{}.template Run<trait>(dst, fm, filter, params);
}

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename V, typename Params>
__aicore__ inline typename Std::enable_if<VerifyingMmadWithBiasTemplate<T, U, S, V>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
{
   constexpr Hardware dstPos = GetHardPos<T>();
   constexpr Hardware fmPos = GetHardPos<U>();
   constexpr Hardware filterPos = GetHardPos<S>();
   constexpr Hardware biasPos = GetHardPos<V>();
   using Tensor2Tensor = typename MmadTensor2Tensor<dstPos, fmPos, filterPos, biasPos, 
      CURRENT_ARCH_VERSION>::type;
   Tensor2Tensor{}.template Run<trait>(dst, fm, filter, bias, params);
}
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
