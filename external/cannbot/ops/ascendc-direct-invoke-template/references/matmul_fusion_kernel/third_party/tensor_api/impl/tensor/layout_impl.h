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
    "tensor_api/impl/tensor/layout_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_method.h"
#include "impl/tensor/coord_index.h"
#include "impl/tensor/layout_fractal.h"

namespace AscendC {
namespace Te {

template <typename T>
struct LocalTensor;

template <typename Coord, typename LayoutType, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const TileShape& tileShape) 
{
    using OriginShape = Std::remove_cvref_t<decltype(layout.Shape())>;
    if constexpr (nesting_depth_v<TileShape> == nesting_depth_v<OriginShape>	 
            && Std::tuple_size_v<TileShape> == Std::tuple_size_v<OriginShape>) {	 
        return MakeLayout(tileShape, layout.Stride());	 
    } else {
        static_assert(Std::is_tuple_v<TileShape>,"TileShape must be a tuple");
        static_assert(nesting_depth_v<TileShape> == TWO_DIM_DATA, "Only Support Two Dim TileShape");
        static_assert(nesting_depth_v<OriginShape> == FOUR_DIM_DATA, "Only Support Four Dim Layout");

        auto innerRow = Std::get<0>(GetShape<0>(layout));
        auto innerCol = Std::get<0>(GetShape<1>(layout));
    
        return MakeLayout(MakeFractalShape(tileShape, MakeShape(innerRow, innerCol)), layout.Stride()); 
    } 
}

template <typename Coord, typename LayoutType, typename TensorType>
 __aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const LocalTensor<TensorType>& tileTensor) 
{
    using TensorLayoutType = typename LocalTensor<TensorType>::layoutType;
    static_assert(TensorLayoutType::rank == LayoutType::rank, "Tensor Rank must be equal to Layout rank");

    auto innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
    auto innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

    auto srcRow = innerRow * Std::get<1>(Std::get<0>(layout.Shape())) - Std::get<0>(coord);
    auto srcCol = innerCol * Std::get<1>(Std::get<1>(layout.Shape())) - Std::get<1>(coord);

    auto dstRow = Std::get<0>(Std::get<0>(tileTensor.Shape())) * Std::get<1>(Std::get<0>(tileTensor.Shape()));
    auto dstCol = Std::get<0>(Std::get<1>(tileTensor.Shape())) * Std::get<1>(Std::get<1>(tileTensor.Shape()));

    auto realRow = Std::min(srcRow, dstRow);	 
    auto realCol = Std::min(srcCol, dstCol);

    return MakeLayout(MakeFractalShape(MakeShape(realRow, realCol), MakeShape(innerRow, innerCol)), layout.Stride()); 

}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
