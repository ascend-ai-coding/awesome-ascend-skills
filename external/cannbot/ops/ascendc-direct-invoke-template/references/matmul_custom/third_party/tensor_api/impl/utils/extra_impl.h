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
    "tensor_api/impl/utils/extra_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file extra_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_UTILS_EXTRA_IMPL_H
#define IMPL_TENSOR_API_UTILS_EXTRA_IMPL_H

#include "impl/utils/constant_impl.h"

namespace AscendC {
namespace Te {
template <typename... Ts>
using void_t = void;

template <typename T, typename = void>
struct IterRef {
    using type = decltype(*Std::declval<T&>()); // type = T&
};

template <typename T>
struct IterRef<T, void_t<typename T::reference>> {
    using type = typename T::reference; 
};

template <typename T, typename = void>
struct IterEle {
    using type = Std::remove_reference_t<typename IterRef<T>::type>;
};

template <typename T>
struct IterEle<T,void_t<typename T::elementType>> {
    using type = typename T::elementType;
};

template <typename T, typename = void>
struct IterVal {
    using type = Std::remove_cv_t<typename IterEle<T>::type>;
};

template <typename T>
struct IterVal<T,void_t<typename T::valueType>> {
    using type = typename T::valueType;
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_UTILS_EXTRA_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
