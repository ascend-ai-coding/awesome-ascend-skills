/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [PATTERN] Tuple 工具，不需要修改

/*!
 * \file tuple_utils.h
 * \brief Tuple access helpers for nested Ascend C tuple-like values.
 */

#ifndef UTILS_TUPLE_UTILS_H
#define UTILS_TUPLE_UTILS_H

#include "lib/std/tuple.h"
#include "./integral_constant.h"

// Base case: return the element addressed by one tuple index.
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive case: walk nested tuples by consuming one index at a time.
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

#endif
