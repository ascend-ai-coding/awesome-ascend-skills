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
* \file copy.h
* \brief
*/
#ifndef INCLUDE_TENSOR_API_ALGORITHM_COPY_H
#define INCLUDE_TENSOR_API_ALGORITHM_COPY_H

#include "impl/algorithm/copy_impl.h"

namespace AscendC {
namespace Te {

template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params);

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params);

template <typename... Args>
__aicore__ inline auto MakeCopy(const Args& ...traits);

}
}

#endif // INCLUDE_TENSOR_API_ALGORITHM_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
