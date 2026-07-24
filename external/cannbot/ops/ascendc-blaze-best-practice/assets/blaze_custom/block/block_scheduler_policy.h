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
 * \file block_scheduler_policy.h
 * \brief Scheduler policy tags used by the matmul block pipeline.
 */

#ifndef BLOCK_SCHEDULER_POLICY_H
#define BLOCK_SCHEDULER_POLICY_H

#include <cstdint>

// ============================================================================
// [MODIFY] Scheduler 策略标签。`BlockSchedulerSelector`（block_scheduler_utils.h）
// 通过这个标签挑选具体的调度器实现（matmul_block_scheduler.h 中注册）。
// 单策略的 matmul 工程一般无需修改。
// ============================================================================

template <uint64_t FULL_LOAD_MODE_>
struct MatmulSwatScheduler {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

#endif
