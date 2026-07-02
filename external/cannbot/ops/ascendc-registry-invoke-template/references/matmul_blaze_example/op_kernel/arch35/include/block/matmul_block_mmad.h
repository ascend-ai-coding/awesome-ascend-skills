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
 * \file matmul_block_mmad.h
 * \brief Block-level MMAD copy/compute pipeline (GM->L1->L0A/L0B->L0C->GM).
 */

#ifndef MATMUL_BLOCK_MMAD_H
#define MATMUL_BLOCK_MMAD_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor_api/tensor.h"
#include "block_mmad.h"
#include "../policy/dispatch_policy.h"
#include "../utils/matmul_constant.h"

// ============================================================================
// Matmul BlockMmad —— 单 block (baseM x baseN) 的数据搬运与 MMAD 流水
//
// 通用模板（NO_FULL_LOAD_MODE）职责：
//   - L1 ping-pong 双缓冲（half-L1 = A|B 一组）
//   - GM -> L1 搬运 A/B（NZ / ZN 格式）
//   - L1 -> L0A/L0B 加载，按 baseK 切分
//   - 调用 tensor_api 的 `Mmad()`，在 L0C 上累加（fp32 或 int32，由 `L0CType` 决定）
//   - 最后一次累加后 fixpipe 写回 GM（L0C -> CType，CType 决定 quantPre）
// ============================================================================

namespace Block {
using namespace AscendC;

struct CopyL0C2UBSplitMTrait {
    using TraitType = AscendC::Te::CopyL0C2UBTrait;
    static constexpr const TraitType value{
        AscendC::Te::RoundMode::DEFAULT,
        false,
        false,
        AscendC::Te::DualDstMode::DUAL_DST_SPLIT_M
    };
};

template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>, DispatchPolicy_>>> {
//具体实现参考ascendc-blaze-best-practice/references/matmul_custom/include/block/matmul_block_mmad.h
//差异点：quant_matmul_gelu的输入矩阵dtype为int8，需要修改BLOCK_CUBE=32，而不能用默认的BLOCK_CUBE=16
//注意：BLOCK_CUBE_L0C固定为16，不能因为dtype改变而修改！

};
} // namespace Block

#endif // MATMUL_BLOCK_MMAD_H
