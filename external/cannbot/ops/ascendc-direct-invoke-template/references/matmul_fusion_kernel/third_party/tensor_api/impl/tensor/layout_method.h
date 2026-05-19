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
    "tensor_api/impl/tensor/layout_method.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_method.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H

#include "impl/tensor/layout_definition.h"

namespace AscendC {
namespace Te {

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...> MakeTile(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
{
    return {t...};
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride)
{
    static_assert(Std::is_tuple_v<T> && Std::is_tuple_v<U>, "Shape or Stride is not tuple!");
    return Layout<T, U>(shape, stride);
}

// shape = ((x1, x2, ..., xn), (y1, y2, ..., yn))
// stride[0][0] = 1; stride[0][i] = shape[0][i-1]*shape[1][i-1]*stride[0][i-1]; stride[1][i] = shape[0][i]*stride[0][i]
template <size_t I, typename Row, typename Col>
struct StrideRowElem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col) {
        if constexpr (I == 0) {
            return Std::Int<1>{};
        } else {
            return Std::get<I - 1>(row) * Std::get<I - 1>(col) *
                StrideRowElem<I - 1, Row, Col>::value(row, col);
        }
    }
};

template <size_t I, typename Row, typename Col>
struct StrideColElem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col) {
        return Std::get<I>(row) * StrideRowElem<I, Row, Col>::value(row, col);
    }
};

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto BuildStrideRowImpl(const Row& row, const Col& col,
    Std::index_sequence<Is...>) {
    return MakeStride(StrideRowElem<Is, Row, Col>::value(row, col)...);
}

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto BuildStrideColImpl(const Row& row, const Col& col,
    Std::index_sequence<Is...>) {
    return MakeStride(StrideColElem<Is, Row, Col>::value(row, col)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto ComputeStride(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType> && Std::tuple_size_v<ShapeType> == 2,
        "ShapeType must be tuple of two tuples");
    const auto& row = Std::get<0>(shape);
    const auto& col = Std::get<1>(shape);
    static_assert(Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>> ==
        Std::tuple_size_v<Std::remove_cvref_t<decltype(col)>>,
        "ShapeType rows must have same length");
    constexpr size_t N = Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>>;
    using Row = Std::remove_cvref_t<decltype(row)>;
    using Col = Std::remove_cvref_t<decltype(col)>;
    auto stride0 = BuildStrideRowImpl(row, col, Std::make_index_sequence<N>{});
    auto stride1 = BuildStrideColImpl(row, col, Std::make_index_sequence<N>{});
    return MakeStride(stride0, stride1);
}

// shape = (x1, x2, x3, ..., xn) -> stride = (x2*x3*...*xn, ..., x_{n-1}*xn, xn, 1)
template <size_t I, typename ShapeType>
struct FlatStrideElem {
    __aicore__ static inline constexpr auto value(const ShapeType& shape) {
        constexpr size_t N = Std::tuple_size_v<ShapeType>;
        static_assert(N > 0, "ShapeType must not be empty");
        if constexpr (I == N - 1) {
            return Std::Int<1>{};
        } else {
            return FlatStrideElem<I + 1, ShapeType>::value(shape) * Std::get<I + 1>(shape);
        }
    }
};

template <typename ShapeType, size_t... Is>
__aicore__ inline constexpr auto BuildFlatStrideImpl(const ShapeType& shape,
    Std::index_sequence<Is...>) {
    return MakeStride(FlatStrideElem<Is, ShapeType>::value(shape)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto ComputeFlatStride(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType must be tuple");
    constexpr size_t N = Std::tuple_size_v<ShapeType>;
    return BuildFlatStrideImpl(shape, Std::make_index_sequence<N>{});
}

template <typename ShapeType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType is not tuple!");
    using ElemT = Std::remove_cvref_t<decltype(Std::get<0>(shape))>;
    if constexpr (Std::is_tuple_v<ElemT>) {
        return MakeLayout(shape, ComputeStride(shape));
    } else {
        return MakeLayout(shape, ComputeFlatStride(shape));
    }
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetShape(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Shape<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetShape(Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Shape<Is...>();
}

template <typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape)
{
    static_assert(Std::is_tuple_v<Tuple> || Std::is_integral_v<Tuple>, "shape is not a tuple or integer");
    return shape;
}

template <size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape)
{
    if constexpr (Std::is_tuple_v<Tuple>) {
        return GetShape<Is...>(Std::get<I>(shape));
    } else {
        return GetTuple<I, Is...>(shape);
    }
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetStride(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Stride<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetStride(Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Stride<Is...>();
}

struct CoshapeSum {
    template <typename... Args>
    __aicore__ inline constexpr auto operator()(const Args&... args) const {
        return (Std::Int<0>{} + ... + args);
    }
};

struct CoshapeCompute {
    template <typename T, typename U>
    __aicore__ inline constexpr auto operator()(const T& shape, const U& stride) const {
        if constexpr (Std::is_tuple_v<T> && Std::is_tuple_v<U>) {
            static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<U>, "Mismatched ranks");
            return TransformApply(shape, stride, CoshapeCompute{}, CoshapeSum{});
        } else {
            auto m1Shape = shape - Std::Int<1>{};
            auto absStride = stride < 0 ? -stride : stride;
            return m1Shape * absStride;
        }
    }
};

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Coshape(const Layout<ShapeType, StrideType>& layout)
{
    auto shape = GetShape<Is...>(layout);
    auto stride = GetStride<Is...>(layout);
    auto coCoord = CoshapeCompute{}(shape, stride);
    return coCoord + Std::Int<1>{};
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Cosize(const Layout<ShapeType, StrideType>& layout)
{
    return TupleSize(Coshape<Is...>(layout));
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Rank(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::tuple_size_v<ShapeType> == Std::tuple_size_v<StrideType>, "The dimensions of the ShapeType and StrideType are not the same.");
    return layout.template Rank<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Select(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return MakeLayout(SelectTuple<Is...>(layout.Shape()), SelectTuple<Is...>(layout.Stride()));
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Get(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return MakeLayout(GetTuple<Is...>(layout.Shape()), GetTuple<Is...>(layout.Stride()));
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Size(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Size<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Capacity(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.Capacity();
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
