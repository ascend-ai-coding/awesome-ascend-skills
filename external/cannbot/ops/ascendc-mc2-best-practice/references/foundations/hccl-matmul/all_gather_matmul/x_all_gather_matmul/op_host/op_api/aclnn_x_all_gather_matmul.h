/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ALL_GATHER_MATMUL_
#define OP_API_INC_ALL_GATHER_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnXAllGatherMatmul 第一段接口：计算 workspace 大小。
 *        非量化、A2(Ascend910B) 场景：AllGather + Matmul 融合计算。
 *        仅保留 A2 路径（源仓 aclnn 中 A5(IsAscend910A5) 分支会委托 aclnnXAllGatherMatmulV2，已按需求剔除，
 *        同时去除了对 mc2/x_all_gather_matmul_v2 的依赖）。
 * @param [in] x1: matmul 左矩阵，float16 / bf16。
 * @param [in] x2: matmul 右矩阵，float16 / bf16。
 * @param [in] bias: 偏置，float16 / bf16（可选）。
 * @param [in] group: 通信域字符串。
 * @param [in] gatherIndex: gather 目标，0=左矩阵（当前仅支持 0）。
 * @param [in] commTurn: 通信切分数（当前仅支持 0）。
 * @param [in] streamMode: acl 流模式枚举（仅支持 1，即 STOP_ON_FAILURE）。
 * @param [out] output: 计算+通信结果，同输入 dtype。
 * @param [out] gatherOut: 仅 gather 通信结果，同输入 dtype。
 * @param [out] workspaceSize: device 侧 workspace 大小。
 * @param [out] executor: op 执行器。
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnXAllGatherMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                            const aclTensor* bias, const char* group,
                                                            int64_t gatherIndex, int64_t commTurn,
                                                            int64_t streamMode,
                                                            const aclTensor* output, const aclTensor* gatherOut,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnXAllGatherMatmul 第二段接口：执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnXAllGatherMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_MATMUL_
