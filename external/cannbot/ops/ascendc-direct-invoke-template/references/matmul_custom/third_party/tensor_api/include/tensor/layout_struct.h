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
 * \file layout_struct.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_LAYOUT_STRUCT_H
#define INCLUDE_TENSOR_API_TENSOR_LAYOUT_STRUCT_H

#include "impl/tensor/layout_struct.h"

namespace AscendC {
namespace Te {

template <typename T>
struct NzLayoutFormat;

template <typename T>
struct ZnLayoutFormat;

template <typename T>
struct L0CLayoutFormat;

template <typename T>
struct NDLayoutFormat;

template <typename T>
struct DNLayoutFormat;

template <typename T>
struct ZzLayoutFormat;

template <typename T>
struct NnLayoutFormat;

template <typename T>
struct ScaleANDLayoutFormat;

template <typename T>
struct ScaleADNLayoutFormat;

template <typename T>
struct ScaleBNDLayoutFormat;

template <typename T>
struct ScaleBDNLayoutFormat;

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_TENSOR_LAYOUT_STRUCT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
