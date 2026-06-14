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
 * \file layout_utils.h
 * \brief Layout tags and mapping traits used by TE tensor wrappers.
 */

#ifndef UTILS_LAYOUT_UTILS_H
#define UTILS_LAYOUT_UTILS_H

// Cube format definitions.
#include "matmul/matmul_config.h"
#include "include/tensor_api/tensor.h"
#include "./integral_constant.h"

// Map layout pattern (AscendC::Te::NDExtLayoutPtn / DNExtLayoutPtn) to the
// transpose flag consumed by BlockMmad / BlockScheduler. The launcher passes
// the pattern type directly; no legacy RowMajor/ColumnMajor wrapper needed.
template <typename T>
struct TagToTrans {
    static_assert(AscendC::Std::always_false_v<T>, "TagToTrans is not implemented for this layout pattern");
};

template <>
struct TagToTrans<AscendC::Te::NDExtLayoutPtn> {
    static constexpr bool value = false;  // ND = row-major, no transpose.
};

template <>
struct TagToTrans<AscendC::Te::DNExtLayoutPtn> {
    static constexpr bool value = true;   // DN = column-major, transposed.
};

template <>
struct TagToTrans<AscendC::Te::NZLayoutPtn> {
    static constexpr bool value = false;  // NZ = non-transposed fractal.
};

template <>
struct TagToTrans<AscendC::Te::ZNLayoutPtn> {
    static constexpr bool value = true;   // ZN = transposed fractal.
};

template <typename T>
struct IsNzOrZn {
    static constexpr bool value =
        AscendC::Std::is_same_v<T, AscendC::Te::NZLayoutPtn> ||
        AscendC::Std::is_same_v<T, AscendC::Te::ZNLayoutPtn>;
};

// L1LayoutHelper: 根据 GM pattern 自动选择 L1 layout。
// - NZ/ZN 输入：L1 与 GM 同 pattern（走块拷贝 CopyGmToCbufAlignV2NZ/ZN）
// - ND/DN 输入：按 trans 标志选 NZ/ZN（走硬件格式转换 ND→NZ / DN→ZN）
template <typename LayoutPtn, typename Type, bool TransVal>
struct L1LayoutHelper {
    static constexpr size_t C0 = 32 / sizeof(Type);
    using type = AscendC::Std::conditional_t<
        IsNzOrZn<LayoutPtn>::value,
        AscendC::Te::FrameLayoutFormat<LayoutPtn, AscendC::Std::Int<C0>>,
        AscendC::Std::conditional_t<
            TransVal,
            AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>,
            AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>>>;
};

#endif
