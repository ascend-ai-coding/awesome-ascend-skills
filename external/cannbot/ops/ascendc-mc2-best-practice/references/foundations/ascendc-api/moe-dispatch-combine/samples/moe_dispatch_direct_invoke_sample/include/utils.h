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
 * \file utils.h
 * \brief Utility functions for MoeDispatch
 */
#ifndef MOE_DISPATCH_UTILS_H
#define MOE_DISPATCH_UTILS_H

#include "tiling_data.h"
#include "basic_api/kernel_basic_intf.h"

namespace MoeDispatchImpl {

using namespace AscendC;

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t FloorAlign(uint64_t a, uint32_t b)
{
    uint64_t bTemp = static_cast<uint64_t>(b);
    return (bTemp == 0) ? a : (a / bTemp) * bTemp;
}

__aicore__ inline uint64_t AlignUp(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return ((a + b - 1) / b) * b;
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID<event>();
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

} // namespace MoeDispatchImpl

#endif // MOE_DISPATCH_UTILS_H