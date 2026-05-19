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
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file layout.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_LAYOUT_H
#define INCLUDE_TENSOR_API_TENSOR_LAYOUT_H

#include "impl/tensor/layout_impl.h"

namespace AscendC {
namespace Te {

// make_layout.h
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...>  MakeTile(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t);

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride);

template <typename T>
__aicore__ inline constexpr auto MakeLayout(const T& shape);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(Layout<Shape, Stride>& layout);

template <typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape);

template <size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Get(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capacity(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout);

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout);

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride);

// make_fractal.h
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzLayout(U row, S column);

template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeDNLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNDLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZnLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeZzLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNnLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBNDLayout(U row, S column);

template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleBDNLayout(U row, S column);

 template <typename Layout, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Layout& layout, const TileShape& tileShape);

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_TENSOR_LAYOUT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
