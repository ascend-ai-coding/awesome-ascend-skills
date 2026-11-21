/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MATMUL_BLAZE_EXAMPLE_H_
#define ACLNN_MATMUL_BLAZE_EXAMPLE_H_
#include "aclnn/aclnn_base.h"
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnMatmulBlazeExampleGetWorkspaceSize(
    const aclTensor *a, const aclTensor *b,
    bool transA, bool transB,
    aclTensor *c, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnMatmulBlazeExample(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
