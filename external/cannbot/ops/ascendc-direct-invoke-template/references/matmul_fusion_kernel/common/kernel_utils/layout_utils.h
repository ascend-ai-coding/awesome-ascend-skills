/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [PATTERN] Layout 类型工具，不需要修改

/*!
 * \file layout_utils.h
 * \brief Layout tags and mapping traits used by TE tensor wrappers.
 */

#ifndef UTILS_LAYOUT_UTILS_H
#define UTILS_LAYOUT_UTILS_H

// Cube format definitions.
#include "matmul/matmul_config.h"
#include "./integral_constant.h"

namespace layout {
struct RowMajor {};
struct ColumnMajor {};
} // namespace layout

// Map layout tags to CubeFormat values.
template <typename T>
struct TagToFormat {
    static_assert(AscendC::Std::always_false_v<T>, "TagToFormat is not implemented for this layout");
};

template <>
struct TagToFormat<layout::RowMajor> {
    using tag = layout::RowMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <>
struct TagToFormat<layout::ColumnMajor> {
    using tag = layout::ColumnMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

// Map layout tags to transpose flags.
template <typename T>
struct TagToTrans {
    static_assert(AscendC::Std::always_false_v<T>, "TagToTrans is not implemented for this layout");
};

template <>
struct TagToTrans<layout::RowMajor> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::ColumnMajor> {
    static constexpr bool value = true;
};

#endif
