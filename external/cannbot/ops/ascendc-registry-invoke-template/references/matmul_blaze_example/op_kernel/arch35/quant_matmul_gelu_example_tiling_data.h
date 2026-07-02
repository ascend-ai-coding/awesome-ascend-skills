/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_gelu_example_tiling_data.h
 * \brief quant_matmul_gelu_example 算子 TilingData 结构体定义
 *
 * [MODIFY] 本文件定义 Host/Kernel 双侧共享的 TilingData 结构。
 * 开发新融合算子时，按以下步骤修改：
 *   1. 修改结构体名 QuantMatmulGeluExampleTilingData → 您的算子名
 *   2. Blaze 公共字段（m/n/k/baseM/baseN 等）必须保持布局与 MatmulTilingData 一致
 *   3. 如需添加自定义字段，追加在公共字段之后（不影响 Blaze memcpy）
 *
 *
 * 字段与 Blaze MatmulTilingData 完全一致，用于 Host/Kernel 双侧。
 * Host 侧通过 memcpy 把 Blaze MatmulTilingData 的公共字段拷入本结构，
 * 因此公共字段布局必须与 Blaze MatmulTilingData 保持一致。
 */

#ifndef _QUANT_MATMUL_GELU_EXAMPLE_TILING_DATA_H_
#define _QUANT_MATMUL_GELU_EXAMPLE_TILING_DATA_H_

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "kernel_tiling/kernel_tiling.h"

// ---- Blaze 公共 TilingData（原 matmul_tiling_data.h，已合并至本文件）----
// [MODIFY] host 端填写、device 端解包的 POD。新增 bias/scale 等输入时，在这里
// 增补地址或尺寸字段；字段顺序无关，但 total size 需保持 8 字节对齐。
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t mL1{0};
    uint32_t nL1{0};
    uint32_t kL1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t mTailCnt{0};
    uint32_t nTailCnt{0};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    uint32_t usedCoreNum{0};
    uint8_t l1BufferNum{0};
    uint8_t l0cDB{1};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) QuantMatmulGeluExampleTilingData {
    // ---- Blaze MatmulTilingData 公共字段（与 Blaze 结构体布局一致）----
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t mL1{0};
    uint32_t nL1{0};
    uint32_t kL1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t mTailCnt{0};
    uint32_t nTailCnt{0};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    uint32_t usedCoreNum{0};
    uint8_t l1BufferNum{0};
    uint8_t l0cDB{1};
    uint8_t reserved[6]{0}; // padding 对齐到 8 bytes

    // ========================================================================
    // [MODIFY] 自定义 Tiling 字段 — 追加在 Blaze 公共字段之后
    //   注意：自定义字段不影响 Blaze 公共字段的 memcpy，
    //   但需要同步修改 Host 侧 tiling.cpp 中的赋值逻辑。
    //
    //   泛化场景下的常见自定义字段示例（取消注释需要的字段）:
    // ========================================================================

    // [MODIFY] Epilogue 系数 — 如 Epilogue 需要运行时可配置参数
    //   例如 gelu 的 alpha/beta, sigmoid 的温度系数等
    // float geluAlpha{1.0f};         // gelu 缩放系数（ScaleGeluEpilogue 中为 scale）
    // float geluBeta{0.0f};          // gelu 偏移系数（ScaleGeluEpilogue 中为 bias）
    // float sigmoidTemperature{1.0f}; // sigmoid 温度参数: 1/(1+exp(-x/T))

    // [MODIFY] 形状信息 — 如 Epilogue 需要额外的 shape 元素（bias/scale/pertoken numel）
    //   当前 ScaleGeluEpilogueRegBase 不需要，因为 shape 信息通过 GM 地址 + matmul M/N 推导
    // uint32_t biasNumel{0};         // bias 元素数 = N
    // uint32_t scaleNumel{0};        // perchannel scale 元素数 = N
    // uint32_t pertokenNumel{0};     // pertoken scale 元素数 = M
};
#pragma pack(pop)

#endif // _QUANT_MATMUL_GELU_EXAMPLE_TILING_DATA_H_