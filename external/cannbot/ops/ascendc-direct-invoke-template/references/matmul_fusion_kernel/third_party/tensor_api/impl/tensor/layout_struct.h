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
    "tensor_api/impl/tensor/layout_struct.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_struct.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_STRUCT_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_STRUCT_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/layout_fractal.h"
#include "impl/tensor/layout_static_fractal.h"

namespace AscendC {
namespace Te {

template <typename T>
struct NzLayoutFormat {
    template <size_t row, size_t column>
    using type = NZFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeNzLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ZnLayoutFormat {
    template <size_t row, size_t column>
    using type = ZNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeZnLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct L0CLayoutFormat {
    template <size_t row, size_t column>
    using type = L0CFormatLayout<row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeL0CLayout<U, S>(row, column);
    }  
};

template <typename T>
struct DNLayoutFormat {
    template <size_t row, size_t column>
    using type = DNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeDNLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct NDLayoutFormat {
    template <size_t row, size_t column>
    using type = NDFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeNDLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ZzLayoutFormat {
    template <size_t row, size_t column>
    using type = ZZFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeZzLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct NnLayoutFormat {
    template <size_t row, size_t column>
    using type = NNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeNnLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ScaleANDLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleANDFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleANDLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ScaleADNLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleADNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleADNLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ScaleBNDLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleBNDFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleBNDLayout<T, U, S>(row, column);
    }  
};

template <typename T>
struct ScaleBDNLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleBDNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleBDNLayout<T, U, S>(row, column);
    }  
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_STRUCT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
