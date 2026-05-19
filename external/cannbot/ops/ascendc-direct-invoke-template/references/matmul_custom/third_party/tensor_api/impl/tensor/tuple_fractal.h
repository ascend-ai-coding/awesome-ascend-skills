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
    "tensor_api/impl/tensor/tuple_fractal.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file tuple_fractal.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_TUPLE_FRACTAL_H
#define IMPL_TENSOR_API_TENSOR_TUPLE_FRACTAL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_dispatch.h"
#include "impl/tensor/layout_static_fractal.h"

namespace AscendC {
namespace Te {
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZShapeFormat<uint16_t, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Shape, uint16_t>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNDShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeDNShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZnShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZNShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZN, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZzShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZZShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNnShape(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNNShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NN, TupleFormat::Shape, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDShape(U row, S column) { // 不转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Shape, Std::ignore_t>::apply(row, column); // (m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNShape(U row, S column) { // 转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleDNShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Shape, T>::apply(row, column); // 转置(m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBNDShape(U row, S column) { // 不转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNDShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Shape, T>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBDNShape(U row, S column) { // 转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNShapeFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Shape, Std::ignore_t>::apply(row, column); // (scaleK, n)
    }
}

// make stride
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZStrideFormat<uint16_t, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Stride, uint16_t>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNDStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeDNStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZnStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZNStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZN, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZzStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZZStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNnStride(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNNStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NN, TupleFormat::Stride, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDStride(U row, S column) { // 不转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Stride, Std::ignore_t>::apply(row, column); // (m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNStride(U row, S column) { // 转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleDNStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Stride, T>::apply(row, column); // 转置(m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBNDStride(U row, S column) { // 不转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNDStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Stride, T>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBDNStride(U row, S column) { // 转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNStrideFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Stride, Std::ignore_t>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NZCoordFormat<uint16_t, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NZ, TupleFormat::Coord, uint16_t>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNDCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeDNCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNCoordormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZnCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZNCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZN, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZzCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ZZCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ZZ, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNnCoord(U row, S column) {
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNNCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::NN, TupleFormat::Coord, T>::apply(row, column);
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDCoord(U row, S column) { // 不转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return NDCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Coord, Std::ignore_t>::apply(row, column); // (m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNCoord(U row, S column) { // 转置(m, scaleK)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleDNCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Coord, T>::apply(row, column); // 转置(m, scaleK)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBNDCoord(U row, S column) { // 不转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return ScaleNDCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::ND, TupleFormat::Coord, T>::apply(row, column); // (scaleK, n)
    }
}

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBDNCoord(U row, S column) { // 转置(scaleK, n)
    if constexpr(IsIntegralConstantV<U> && IsIntegralConstantV<S>) {
        return DNCoordFormat<T, row, column>{};
    } else if(Std::is_integral_v<U> && Std::is_integral_v<S>){
        return TupleDispatcher<LayoutFormat::DN, TupleFormat::Coord, Std::ignore_t>::apply(row, column); // (scaleK, n)
    }
}
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_TUPLE_FRACTAL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
