/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "torch_kernel_helper.h"

#include "aclrtlaunch_helloworld.h"

namespace ascend_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y)
{
    /* create a result tensor */
    at::Tensor z = at::empty_like(x);

    /* define the block dim */
    uint32_t blockDim = 8;

    /* memory size */
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }

    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(helloworld, blockDim, x, y, z, totalLength);
    return z;
}

}  // namespace ascend_kernel
