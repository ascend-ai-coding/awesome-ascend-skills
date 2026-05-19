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
    "tensor_api/impl/tensor/tuple_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file tuple_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H
#define IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H

#include "impl/utils/utils_impl.h"

namespace AscendC {
namespace Te {

template <typename T, typename F, typename G, size_t... I>
__aicore__ inline constexpr auto TupleApply(T&& t, F&& f, G&& g, Std::index_sequence<I...>)
{
    return g(f(Std::get<I>(static_cast<T&&>(t)))...);
}

template <typename T0, typename T1, typename F, typename G, size_t... I>
__aicore__ inline constexpr auto TupleApply(T0&& t0, T1&& t1, F&& f, G&& g, Std::index_sequence<I...>)
{
    return g(f(Std::get<I>(static_cast<T0&&>(t0)),
                Std::get<I>(static_cast<T1&&>(t1)))...);
}

template <typename T, typename F, typename G>
__aicore__ inline constexpr auto TransformApply(T&& t, F&& f, G&& g)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<T>>) {
        return TupleApply(static_cast<T&&>(t), f, g, tuple_sequence<T>{});
    } else {
        return g(f(static_cast<T&&>(t)));
    }
}

template <typename T0, typename T1, typename F, typename G>
__aicore__ inline constexpr auto TransformApply(T0&& t0, T1&& t1, F&& f, G&& g)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<T0>>) {
        return TupleApply(static_cast<T0&&>(t0), static_cast<T1&&>(t1), f, g, tuple_sequence<T0>{});
    } else {
        return g(f(static_cast<T0&&>(t0), static_cast<T1&&>(t1)));
    }
}

struct MultipliesUnaryLeftFold {
    template <typename... T>
    __aicore__ inline constexpr auto operator()(T&&... t) const {
        return (... * t);
    }
};

struct Product {
    template <typename T>
    __aicore__ inline constexpr auto operator()(const T& intT) const
    {
        if constexpr (Std::is_tuple_v<T>) {
            if constexpr (Std::tuple_size_v<T> == 0) {
                return Std::Int<1>{};
            } else {
                return TransformApply(intT, Product{}, MultipliesUnaryLeftFold{});
            }
        } else if constexpr (Std::is_integral<T>::value) {
            return intT;
        } else {
            static_assert(sizeof(T) == 0, "Invalid Product parameters");
        }
    }
};


template <size_t I, typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    auto tt = Std::get<I>(static_cast<Tuple&&>(t));
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(tt)>>) {
        return tt;
    } else {
        return Std::make_tuple(tt);
    }
}

template <size_t I0, size_t I1, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return GetTuple<I1, Is...>(GetTuple<I0>(static_cast<Tuple&&>(t)));
}

template <typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return static_cast<Tuple&&>(t);
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetRank(const Tuple& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple!");
    if constexpr (sizeof...(Is) == 0) {
        return Std::Int<Std::tuple_size_v<Tuple>>{};
    } else {
        return GetRank(GetTuple<Is...>(t));
    }
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto TupleSize(const Tuple& t)
{
    if constexpr (sizeof...(Is) == 0) {
        return Product{}(t);
    } else {
        return TupleSize(GetTuple<Is...>(t));
    }
}

template <size_t I, typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    auto&& tt = Std::get<I>(static_cast<Tuple&&>(t));
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(tt)>>) {
        return tt;
    }else {
        return Std::make_tuple(tt);
    }
}

template <size_t I0, size_t I1, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return Std::make_tuple(Std::get<I0>(static_cast<Tuple&&>(t)), Std::get<I1>(static_cast<Tuple&&>(t)), Std::get<Is>(static_cast<Tuple&&>(t))...);
}

template <typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return static_cast<Tuple&&>(t);
}

template<size_t index, size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
{
    auto tupleEle = Std::get<index>(t);
    if constexpr(sizeof...(Is) == 0) {
        return Std::get<I>(tupleEle);
    } else {
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }
}

template<size_t index, typename Tuple>
__aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
{
    return Std::get<index>(t);
}

template<typename T0, typename... Ts>
__aicore__ inline constexpr auto GetMax(const T0& t0, const Ts&... ts)
{
    if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return Std::max(t0, GetMax(ts...));
    }
}

template<typename Shape, typename Stride>
__aicore__ inline constexpr auto GetCapacity(const Shape& shape, const Stride& stride);

template <typename Shape, typename Stride, size_t... Is>
__aicore__ inline constexpr auto GetCapacityImpl(const Shape& shape, const Stride& stride, Std::index_sequence<Is...>)
{
    return GetMax(GetCapacity(Std::get<Is>(shape), Std::get<Is>(stride))...);
}

template<typename Shape, typename Stride>
__aicore__ inline constexpr auto GetCapacity(const Shape& shape, const Stride& stride)
{
    if constexpr (Std::is_tuple_v<Shape> && Std::is_tuple_v<Stride>) {
        static_assert(Std::tuple_size_v<Shape> == Std::tuple_size_v<Stride>, "Mismatched ranks");
        return GetCapacityImpl(shape, stride, Std::make_index_sequence<Std::tuple_size_v<Shape>>{});
    } else {
        return shape * stride;
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
