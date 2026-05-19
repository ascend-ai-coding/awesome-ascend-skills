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
    "tensor_api/impl/tensor/layout_fractal.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_fractal.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_dispatch.h"
#include "impl/tensor/layout_static_fractal.h"

namespace AscendC {
namespace Te {
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::NZ, T>::apply(row, column);
    }
}

template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return L0CFormatLayout<row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::NZ, uint16_t>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNDLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeDNLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZnLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZNFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::ZN, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZzLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZZFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::ZZ, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNnLayout(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NNFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::NN, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDLayout(U row, S column) { // 不转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleANDFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::ND, Std::ignore_t>::apply(row, column); // (m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNLayout(U row, S column) { // 转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleADNFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column); // 转置(m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBNDLayout(U row, S column) { // 不转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleBNDFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBDNLayout(U row, S column) { // 转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleBDNFormatLayout<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return LayoutDispatcher<LayoutFormat::DN, Std::ignore_t>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, size_t... Is>
__aicore__ inline decltype(auto) MakeFractalShape(T originShape, U innerShape, Std::index_sequence<Is...>) {
    auto outerShape = Std::make_tuple(Std::ceil_division(Std::get<Is>(originShape), Std::get<Is>(innerShape))...);
    return MakeShape(MakeShape(Std::get<Is>(innerShape), Std::get<Is>(outerShape))...);
}

template <typename T, typename U>
__aicore__ inline decltype(auto) MakeFractalShape(T originShape, U innerShape) {
    static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<U>, "OriginShape and InnerShape must match");
    return MakeFractalShape(originShape, innerShape, Std::make_index_sequence<Std::tuple_size_v<U>>{});
}


} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
