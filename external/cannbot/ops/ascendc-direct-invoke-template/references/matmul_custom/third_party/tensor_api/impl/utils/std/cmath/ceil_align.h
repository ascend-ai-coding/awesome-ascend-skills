/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ceil_align.h
 * \brief
 */
#ifndef IMPL_STD_CEIL_ALIGN_H
#define IMPL_STD_CEIL_ALIGN_H
#include "ceil_division.h"

namespace AscendC {
namespace Std {

template <typename T, typename U>
ASCENDC_HOST_AICORE inline constexpr auto ceil_align(const T& num1, const U& num2)
{
    return ceil_division(num1, num2) * num2;
}

}
}
#endif