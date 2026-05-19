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
 * \file tiling_data.h
 * \brief Serialized tiling data passed from the host launcher to the kernel.
 */

// ============================================================
// Tiling Data — Host<->Device 共享的 Tiling 参数结构体
// 修改清单：
// [MODIFY 1] 基础 Shape 字段（根据算子需求调整）
// [EXTEND 1] Epilogue 私有 tiling 字段（dtype 大小、广播参数等；详见结构体末尾）
// ============================================================

#ifndef TILING_DATA_H
#define TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Serialized tiling result passed from host code to the kernel entry.
//
// The field order is part of the host-device contract, so layout stability is
// more important here than convenience of reordering members.
#pragma pack(push, 8)
struct alignas(8) QuantMatmulTilingData {
    // Original problem shape.
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};

    // Base tile shape selected by the tiling engine.
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};

    // Amount of K covered by one scale fragment staged in L1.
    uint32_t scaleKL1{0};

    // Tail split factors used in the final scheduling round.
    uint32_t mTailTile{1};
    uint32_t nTailTile{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};

    // Launch-time AIC count and buffering parameters.
    // Number of AICs launched for this kernel instance.
    uint32_t usedCoreNum{0};
    // Number of baseK tiles consumed before the rolling L1 buffers advance.
    uint8_t stepK{0};
    // Number of rolling A/B buffer slots reserved in L1.
    uint8_t nBufferNum{0};

    // Output buffering mode selected for the kernel implementation.
    uint8_t dbL0c{0};

    // [EXTEND] ===== 按需添加 Epilogue 私有 tiling 参数 =====
    //
    // 本 skill 仅用于 mxfp8 + eltwise 融合算子，L0C→UB 路径恒定，
    // **不存在**"融合开关"这类 bool 字段；Epilogue 类型通过
    // MatmulKernelFused 的模板参数注入，**禁止**使用 fusionOpType 枚举 +
    // if constexpr 分支选择算子。
    //
    // 这里可以添加的是 **tile 尺寸 / dtype 大小相关** 的 Epilogue 私有字段，例如：
    //   - uint64_t secondInputTypeSize{0};   // 第二路输入 dtype 大小（若非 float）
    //   - uint64_t outputTypeSize{0};        // 输出 dtype 大小（若非 float）
    //   - 其他 per-launch 参数
    // [EXTEND] ===== 扩展字段结束 =====
};
#pragma pack(pop)

#endif // TILING_DATA_H
