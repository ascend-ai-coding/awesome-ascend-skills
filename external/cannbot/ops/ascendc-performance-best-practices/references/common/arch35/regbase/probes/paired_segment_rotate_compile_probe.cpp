/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Compile-only probe for paired_segment_rotate_vf.h. This file is not a runnable kernel.
#include "../code/paired_segment_rotate_vf.h"

using namespace AscendC;

template <typename T>
__aicore__ void InstantiatePairedSegmentRotateForType(LocalTensor<T>& local, GlobalTensor<T>& global,
                                                      const DataCopyPadExtParams<T>& pad)
{
    CopyInHalfMode<T>(local, global, 0, 1, 128 * sizeof(T), pad, 128, 128, 2);
    CopyOutHalfMode<T>(local, global, 0, 1, 128 * sizeof(T), 128, 128, 2);
    HalfAlignVF<T>(local, local, local, local, 128, 128, 1, 1);
}

__aicore__ void InstantiatePairedSegmentRotateFp32(LocalTensor<float>& local, GlobalTensor<float>& global,
                                                   const DataCopyPadExtParams<float>& pad)
{
    InstantiatePairedSegmentRotateForType(local, global, pad);
}

__aicore__ void InstantiatePairedSegmentRotateFp16(LocalTensor<half>& local, GlobalTensor<half>& global,
                                                   const DataCopyPadExtParams<half>& pad)
{
    InstantiatePairedSegmentRotateForType(local, global, pad);
}

__aicore__ void InstantiatePairedSegmentRotateBf16(LocalTensor<bfloat16_t>& local, GlobalTensor<bfloat16_t>& global,
                                                   const DataCopyPadExtParams<bfloat16_t>& pad)
{
    InstantiatePairedSegmentRotateForType(local, global, pad);
}
