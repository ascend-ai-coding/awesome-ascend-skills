/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================
// [PATTERN] BlockMmad 基类模板
// 通过 DispatchPolicy 的类型 trait 选择偏特化实现。
// 不需要修改此文件。
// ============================================================

/*!
 * \file block_mmad.h
 * \brief Common block-level MMAD template declaration.
 */

#ifndef BLOCK_MMAD_H
#define BLOCK_MMAD_H

#include "kernel_utils/integral_constant.h"

namespace Block {
template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class Enable = void>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};
} // namespace Block

#endif
