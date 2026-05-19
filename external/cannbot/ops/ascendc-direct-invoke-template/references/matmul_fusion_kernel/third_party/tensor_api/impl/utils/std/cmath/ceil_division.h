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
 * \file ceil_division.h
 * \brief
 */
#ifndef IMPL_STD_CEIL_DIVISION_H
#define IMPL_STD_CEIL_DIVISION_H

namespace AscendC {
namespace Std {

template <typename T, typename U>
ASCENDC_HOST_AICORE inline constexpr auto ceil_division(const T& num1, const U& num2)
{
    return (num1 + num2 - Int<1>{}) / num2;
}

}
}
#endif