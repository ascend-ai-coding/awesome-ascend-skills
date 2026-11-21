/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
uint16_t encode(float f, bool bf)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if (bf)
        return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    uint32_t sign = (u >> 16) & 0x8000;
    uint32_t exp = (u >> 23) & 255;
    uint32_t mant = u & 0x7fffff;
    if (exp == 255)
        return sign | 0x7c00 | (mant ? 0x200 : 0);
    if (exp > 142)
        return sign | 0x7c00;
    if (exp < 102)
        return sign;
    if (exp < 113) {
        mant |= 0x800000;
        unsigned shift = 126 - exp;
        uint32_t q = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        return sign | uint16_t(q + (rem > mid || (rem == mid && (q & 1))));
    }
    uint32_t base = ((exp - 112) << 10) | (mant >> 13);
    uint32_t rem = mant & 8191;
    return sign | uint16_t(base + (rem > 4096 || (rem == 4096 && (base & 1))));
}
float decode(uint16_t u, bool bf)
{
    if (bf) {
        uint32_t bits = uint32_t(u) << 16;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    unsigned e = (u >> 10) & 31;
    unsigned m = u & 1023;
    if (e == 31)
        return m ? std::numeric_limits<float>::quiet_NaN() :
                   ((u & 0x8000) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
    return std::ldexp(e ? float(1024 + m) : float(m), e ? int(e) - 25 : -24) * (u & 0x8000 ? -1 : 1);
}
