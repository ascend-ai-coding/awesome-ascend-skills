/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_CV_SYNC_CONSTANTS_H
#define EPILOGUE_CV_SYNC_CONSTANTS_H

#include <cstdint>

namespace CvSync {

constexpr uint16_t MODE = 4;

constexpr int16_t AIC_TO_AIV_FLAG = 8;
constexpr int16_t AIV_TO_AIC_FLAG = 5;

constexpr int16_t COUNT_ID_MAX = 15;
constexpr int16_t COUNT_FLAG   = 3;

} // namespace CvSync

#endif // EPILOGUE_CV_SYNC_CONSTANTS_H
