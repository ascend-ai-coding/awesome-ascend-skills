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
    "tensor_api/impl/tensor/local_tensor_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file local_tensor_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LOCAL_TENSOR_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LOCAL_TENSOR_IMPL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_impl.h"
#include "impl/tensor/pointer_impl.h"

namespace AscendC {
namespace Te {

// struct LocalTensor
template <typename EngineType, typename LayoutType>
struct TensorAttribute {};

template <typename T>
struct LocalTensor {};

template <typename EngineType, typename LayoutType>
struct LocalTensor<TensorAttribute<EngineType, LayoutType>> {
    using iterator = typename EngineType::iterator;
    using valueType = typename EngineType::valueType;
    using elementType = typename EngineType::elementType;
    using reference = typename EngineType::reference;

    using engineType  = EngineType;
    using layoutType  = LayoutType;

    Std::tuple<layoutType, engineType> rep;

    __aicore__ inline constexpr LocalTensor() {}
    __aicore__ inline constexpr LocalTensor(const EngineType& engine, const LayoutType& layout) : rep(layout, engine) {}

    static constexpr int rank  = LayoutType::rank; // tuple size

    __aicore__ inline constexpr decltype(auto) Tensor() const {
        return *this;
    }

    __aicore__ inline constexpr decltype(auto) Engine() const {
        return Std::get<1>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Engine() {
        return Std::get<1>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Layout() const {
        return Std::get<0>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Data() const {
        return Engine().Begin();
    }

    __aicore__ inline constexpr decltype(auto) Data() {
        return Engine().Begin();
    }

    __aicore__ inline constexpr decltype(auto) Shape() const {
        return Layout().Shape();
    }

    __aicore__ inline constexpr decltype(auto) Stride() const {
        return Layout().Stride();
    }

    __aicore__ inline constexpr auto Size() const {
        return Layout().Size();
    }

    __aicore__ inline constexpr auto Capacity() const {
        return Layout().Capacity();
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) {
        return Data()[Layout()(coord)];
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) const {
        return Data()[Layout()(coord)];
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) {
        return operator()(coord, EmptyShape<LayoutType::depth>{});
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) const {
        return operator()(coord, EmptyShape<LayoutType::depth>{});
    }

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) {
        return operator()(MakeCoord(c0,c1,cs...));
    }

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) const {
        return operator()(MakeCoord(c0,c1,cs...));
    }

	template <typename Coord, typename InfoType>
 	__aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info) {
 	    auto iter = Data() + Layout()(coord);
 	    auto tileLayout = MakeTileLayout(coord, Layout(), info);
 	    return MakeTensor(iter, tileLayout);
 	}
 	 
 	template <typename Coord, typename InfoType>
 	__aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info) const {
 	    auto iter = Data() + Layout()(coord);
 	    auto tileLayout = MakeTileLayout(coord, Layout(), info);
 	    return MakeTensor(iter, tileLayout);
 	}

    template <typename... Layouts>
    __aicore__ inline constexpr auto Compose(const Layouts&... layouts) {
        return MakeTensor(Data(), Layout().Compose(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Compose(const Layouts&... layouts) const {
        return MakeTensor(Data(), Layout().Compose(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Tile(const Layouts&... layouts) {
        return MakeTensor(Data(), Layout().Tile(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Tile(const Layouts&... layouts) const {
        return MakeTensor(Data(), Layout().Tile(layouts...));
    }
};

template <typename T>
struct IsTileTensor : Std::false_type {};

template <typename Engine, typename Layout>
struct IsTileTensor<LocalTensor<TensorAttribute<Engine,Layout>>> : Std::true_type {};

template <typename T>
constexpr bool IsTileTensorV = IsTileTensor<T>::value;

// make_tensor.h
template <typename T, typename = void>
struct HasDereference : Std::false_type {};

template <typename T>
struct HasDereference<T, void_t<decltype(*Std::declval<T&>())>> : Std::true_type {};

template <typename T>
struct MakeLocalTensor {
template <typename Arg0, typename... Args>
__aicore__ inline constexpr auto operator()(const Arg0& arg0, const Args&... args) const {
    if constexpr (HasDereference<Arg0>::value) {
    using Engine = ViewEngine<Arg0>;
    if constexpr (sizeof...(Args) == 1 && (is_layout<Args>::value && ...)) {
        return LocalTensor<TensorAttribute<Engine, Args...>>{Engine{arg0}, args...};
    } else {
        return LocalTensor<TensorAttribute<Engine, decltype(MakeLayout(args...))>>{Engine{arg0}, MakeLayout(args...)};
    }
    }
}
};

template <typename Iterator, typename... Args>
__aicore__ inline constexpr auto MakeTensor(const Iterator& iter, const Args&... args)
{
    static_assert(HasDereference<Iterator>::value, "Expected iterator iter in MakeLocalTensor(iter, args...)");
    static_assert(!(HasDereference<Args>::value && ...), "Expected layout args... in MakeLocalTensor(iter, args...)");
    return MakeLocalTensor<Iterator>{}(iter, args...);
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LOCAL_TENSOR_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
