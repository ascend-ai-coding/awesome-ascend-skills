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
 * \file add_example_tiling_data.h
 * \brief TilingData 结构体定义
 *
 * ✅ 必须使用标准 C++ struct 定义 TilingData
 * ❌ 禁止使用废弃的 BEGIN_TILING_DATA_DEF 宏
 *
 * 反模式（禁止）：
 *   BEGIN_TILING_DATA_DEF(AddExampleTilingData)
 *   TILING_DATA_FIELD_DEF(int64_t, totalNum);
 *   END_TILING_DATA_DEF;
 */

#ifndef _ADD_EXAMPLE_TILING_DATA_H_
#define _ADD_EXAMPLE_TILING_DATA_H_

struct AddExampleTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif
