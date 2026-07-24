/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_AUX_C0_SIZE_H
#define UTILS_AUX_C0_SIZE_H

#include "kernel_operator.h"

namespace AscendC {

template <typename T>
constexpr int32_t AuxGetC0Size()
{
    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        return 16;
    } else if constexpr (std::is_same_v<T, float>) {
        return 8;
    } else if constexpr (std::is_same_v<T, fp8_e4m3fn_t> || std::is_same_v<T, int8_t>) {
        return 32;
    } else if constexpr (std::is_same_v<T, fp4x2_e2m1_t>) {
        return 64;
    } else {
        return 16;
    }
}

} // namespace AscendC

#endif
