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
    "tensor_api/impl/tensor/layout_static_fractal.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_static_fractal.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_STATIC_FRACTAL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_STATIC_FRACTAL_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_definition.h"

namespace AscendC {
namespace Te {
// coord
// NZ
template <typename T, size_t row, size_t column>
using NZCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>,
    Shape<Std::Int<0>, Std::Int<Std::ceil_division(column, C0_ELEMENT<T>)>>>;

// ZN
template <typename T, size_t  row, size_t  column>
using ZNCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<Std::ceil_division(row, C0_ELEMENT<T>)>>,
    Shape<Std::Int<0>, Std::Int<Std::ceil_division(column, FRACTAL_FIXED)>>>;

//  ScaleND
template <typename T, size_t row, size_t column>
using ScaleNDCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<row/2>>, 
                                    Shape<Std::Int<0>, Std::Int<column>>>;
//  ScaleDN
template <typename T, size_t row, size_t column>
using ScaleDNCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<row>>, 
                                    Shape<Std::Int<0>, Std::Int<column/MX_SCALE_K0>>>;

// ScaleNN
template <typename T, size_t row, size_t column>
using ScaleNNCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<row/MX_SCALE_K0>>, 
                                    Shape<Std::Int<0>, Std::Int<Std::ceil_division(column, FRACTAL_FIXED)>>>; 

// ND
template <typename T, size_t row, size_t column>
using NDCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<row>>, Shape<Std::Int<0>, Std::Int<column>>>;

// DN
template <typename T, size_t row, size_t column>
using DNCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<row>>, Shape<Std::Int<0>, Std::Int<column>>>;

// ZZ
template <typename T, size_t row, size_t column>
using ZZCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>, 
    Shape<Std::Int<0>, Std::Int<Std::ceil_division(column, C0_ELEMENT<T>)>>>; 

// scaleZZ
template <typename T, size_t row, size_t column>
using ScaleZZCoordFormat = Shape<Shape<Std::Int<0>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>,  
                                    Shape<Std::Int<0>, Std::Int<column/MX_SCALE_K0>>>;

// NZ
template <typename T, size_t row, size_t column>
using NZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>,
    Shape<Std::Int<C0_ELEMENT<T>>, Std::Int<Std::ceil_division(column, C0_ELEMENT<T>)>>>;

template <typename T, size_t row, size_t column>
using NZStrideFormat = Stride<Stride<Std::Int<C0_ELEMENT<T>>, Std::Int<C0_ELEMENT<T> * FRACTAL_FIXED>>,
    Stride<Std::Int<1>, Std::Int<C0_ELEMENT<T> * Std::ceil_align(row, FRACTAL_FIXED)>>>;

//  ScaleND
template <typename T, size_t row, size_t column>
using ScaleNDShapeFormat = Shape<Shape<Std::Int<MX_SCALE_K0>, Std::Int<row/2>>, 
                                    Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using ScaleNDStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<MX_SCALE_K0*column>>,
                                    Stride<Std::Int<0>, Std::Int<MX_SCALE_K0>>>;

//  ScaleDN
template <typename T, size_t row, size_t column>
using ScaleDNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, 
                                    Shape<Std::Int<MX_SCALE_K0>, Std::Int<column/MX_SCALE_K0>>>;

template <typename T, size_t row, size_t column>
using ScaleDNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<MX_SCALE_K0>>,
                                    Stride<Std::Int<1>, Std::Int<row*MX_SCALE_K0>>>;

// ScaleNN
template <typename T, size_t row, size_t column>
using ScaleNNShapeFormat = Shape<Shape<Std::Int<MX_SCALE_K0>, Std::Int<row/MX_SCALE_K0>>, 
                                    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<Std::ceil_division(column, FRACTAL_FIXED)>>>;

template <typename T, size_t row, size_t column>
using ScaleNNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_SIZE<>>>,
                                    Stride<Std::Int<MX_SCALE_K0>, Std::Int<row*FRACTAL_FIXED>>>;

// ND
template <typename T, size_t row, size_t column>
using NDShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using NDStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<column>>, Stride<Std::Int<0>, Std::Int<1>>>;

// DN
template <typename T, size_t row, size_t column>
using DNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using DNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<1>>, Stride<Std::Int<0>, Std::Int<row>>>;

// ZN
template <typename T, size_t  row, size_t  column>
using ZNShapeFormat = Shape<Shape<Std::Int<C0_ELEMENT<T>>, Std::Int<Std::ceil_division(row, C0_ELEMENT<T>)>>,
    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<Std::ceil_division(column, FRACTAL_FIXED)>>>; 

template <typename T, size_t  row, size_t  column>
using ZNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_ELEMENT<T> * Std::ceil_align(column, FRACTAL_FIXED)>>,
    Stride<Std::Int<C0_ELEMENT<T>>, Std::Int<C0_ELEMENT<T> * FRACTAL_FIXED>>>;

// ZZ
template <typename T, size_t row, size_t column>
using ZZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>,
    Shape<Std::Int<C0_ELEMENT<T>>, Std::Int<Std::ceil_division(column, C0_ELEMENT<T>)>>>;

template <typename T, size_t row, size_t column>
using ZZStrideFormat = Stride<Stride<Std::Int<C0_ELEMENT<T>>, Std::Int<FRACTAL_FIXED * Std::ceil_align(column, C0_ELEMENT<T>)>>,
    Stride<Std::Int<1>, Std::Int<C0_ELEMENT<T> * FRACTAL_FIXED>>>;

// scaleZZ
template <typename T, size_t row, size_t column>
using ScaleZZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<Std::ceil_division(row, FRACTAL_FIXED)>>, 
                                    Shape<Std::Int<MX_SCALE_K0>, Std::Int<column/MX_SCALE_K0>>>;

template <typename T, size_t row, size_t column>
using ScaleZZStrideFormat = Stride<Stride<Std::Int<MX_SCALE_K0>, Std::Int<FRACTAL_FIXED * column>>,
                                    Stride<Std::Int<1>, Std::Int<C0_SIZE<>>>>;

template <typename T, size_t row, size_t column>
using NDFormatLayout = Layout<NDShapeFormat<T, row, column>, NDStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using DNFormatLayout = Layout<DNShapeFormat<T, row, column>, DNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZNFormatLayout = Layout<ZNShapeFormat<T, row, column>, ZNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column, typename Enable = void>
struct NZLayoutFormatImpl;

template <typename T, size_t row, size_t column>
struct NZLayoutFormatImpl<T, row, column, typename Std::enable_if<!Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<NZShapeFormat<T, row, column>, NZStrideFormat<T, row, column>>;
};

template <typename T, size_t row, size_t column>
struct NZLayoutFormatImpl<T, row, column, typename Std::enable_if<Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<NZShapeFormat<uint16_t, row, column>, NZStrideFormat<uint16_t, row, column>>;
};

template <typename T, size_t row, size_t column>
using NZFormatLayout = typename NZLayoutFormatImpl<T, row, column>::type;

template <size_t row, size_t column>
using L0CFormatLayout = NZFormatLayout<Std::ignore_t, row, column>;

template <typename T, size_t row, size_t column>
struct ZZLayoutFormatImpl {
    using type = Layout<ScaleZZShapeFormat<T, row, column>, ScaleZZStrideFormat<T, row, column>>;
};

template <size_t row, size_t column>
struct ZZLayoutFormatImpl<fp8_e8m0_t, row, column> {
    using type = Layout<ScaleZZShapeFormat<fp8_e8m0_t, row, column>, ScaleZZStrideFormat<fp8_e8m0_t, row, column>>;
};

template <typename T, size_t row, size_t column>
using ZZFormatLayout = typename ZZLayoutFormatImpl<T, row, column>::type;

// NN
template <typename T, size_t row, size_t column>
using NNFormatLayout = Layout<ScaleNNShapeFormat<T, row, column>, ScaleNNStrideFormat<T, row, column>>;

// scaleAND
template <typename T, size_t row, size_t column>
using ScaleANDFormatLayout = Layout<NDShapeFormat<T, row, column>, NDStrideFormat<T, row, column>>;

// scaleADN
template <typename T, size_t row, size_t column>
using ScaleADNFormatLayout = Layout<ScaleDNShapeFormat<T, row, column>, ScaleDNStrideFormat<T, row, column>>;

// scaleBND
template <typename T, size_t row, size_t column>
using ScaleBNDFormatLayout = Layout<ScaleNDShapeFormat<T, row, column>, ScaleNDStrideFormat<T, row, column>>;

// scaleBDN
template <typename T, size_t row, size_t column>
using ScaleBDNFormatLayout = Layout<DNShapeFormat<T, row, column>, DNStrideFormat<T, row, column>>;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_STATIC_FRACTAL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
