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
 * \file pointer.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_POINTER_H
#define INCLUDE_TENSOR_API_TENSOR_POINTER_H

#include "impl/tensor/pointer_impl.h"

// algorithm
namespace AscendC {
namespace Te {

template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeUBmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeL1memPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0AmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0BmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0CmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter);

template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter);

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_TENSOR_POINTER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
