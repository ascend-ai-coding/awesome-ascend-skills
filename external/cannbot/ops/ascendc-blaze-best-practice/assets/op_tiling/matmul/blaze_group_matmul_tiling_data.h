/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef BLAZE_GROUP_MATMUL_TILING_DATA_H
#define BLAZE_GROUP_MATMUL_TILING_DATA_H

#include "blaze_matmul_tiling_data.h"
#ifndef __CCE_AICORE__
#include <cstddef>
#endif

#pragma pack(push, 8)
struct alignas(8) GroupMatmulTilingData {
    MatmulTilingData matmul;
    uint64_t groupListAddr{0};
    uint32_t groupNum{0};
    // 0: cumsum endpoints; 1: per-group row counts.
    uint8_t groupListType{0};
};
#pragma pack(pop)

#ifndef __CCE_AICORE__
static_assert(sizeof(GroupMatmulTilingData) == 88, "GroupMatmulTilingData ABI size changed");
static_assert(
    offsetof(GroupMatmulTilingData, groupListAddr) == 72, "GroupMatmulTilingData groupListAddr offset changed");
static_assert(offsetof(GroupMatmulTilingData, groupNum) == 80, "GroupMatmulTilingData groupNum offset changed");
#endif

#endif // BLAZE_GROUP_MATMUL_TILING_DATA_H
