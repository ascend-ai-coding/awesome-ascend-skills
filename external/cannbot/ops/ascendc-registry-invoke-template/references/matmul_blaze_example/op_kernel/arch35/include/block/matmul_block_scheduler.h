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
 * \file matmul_block_scheduler.h
 * \brief Row-major block scheduler for the matmul non-full-load path.
 */

#ifndef MATMUL_BLOCK_SCHEDULER_H
#define MATMUL_BLOCK_SCHEDULER_H

#include "kernel_utils/common_utils.h"

#include "./block_scheduler_policy.h"
#include "./block_scheduler_utils.h"

// ============================================================================
// Matmul BlockScheduler —— 把 [M/baseM][N/baseN] 的块格分配到各 AIC。
//
// 本脚手架采用最简单的 **行主序（row-major）** 平铺遍历：把 mCnt*nCnt 个块按
// 行优先编号，第 c 号核处理编号 c, c+blockNum, c+2*blockNum, ... 的块（round-robin）。
// M、N 方向的余数块尺寸由 GetBlockShape 用 min 直接算出（host tiling 不做 tail-split）。
//
// ============================================================================

namespace Block {

template <class ProblemShape_, bool TransA_, bool TransB_>
class MatmulBlockScheduler {
// 具体实现参考ascendc-blaze-best-practice/references/matmul_custom/include/block/matmul_block_scheduler.h
};

template <class ProblemShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, MatmulSwatScheduler<NO_FULL_LOAD_MODE>, TransA_, TransB_> {
    using SchedulerOp = MatmulBlockScheduler<ProblemShape_, TransA_, TransB_>;
};

} // namespace Block

#endif // MATMUL_BLOCK_SCHEDULER_H
