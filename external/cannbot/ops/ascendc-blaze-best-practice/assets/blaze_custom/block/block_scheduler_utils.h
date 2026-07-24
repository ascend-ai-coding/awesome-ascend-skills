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
 * \file block_scheduler_utils.h
 * \brief Utility helpers shared by the SWAT block schedulers.
 */

#ifndef BLOCK_SCHEDULER_UTILS_H
#define BLOCK_SCHEDULER_UTILS_H

namespace Block {

// Base template for scheduler specialization.
//
// The concrete MX scheduler is selected later from problem shape, tile
// shape, transpose flags, and the SWAT mode encoded in the policy tag.
template <class ProblemShape, class BlockScheduler = void, bool TransA = false, bool TransB = false>
struct BlockSchedulerSelector;

} // namespace Block
#endif
