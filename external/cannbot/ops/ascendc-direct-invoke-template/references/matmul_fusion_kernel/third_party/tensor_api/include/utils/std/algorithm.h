/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Algorithm helpers for Ascend C device-side code.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_STD_ALGORITHM_H
#define INCLUDE_TENSOR_API_UTILS_STD_ALGORITHM_H

namespace AscendC {
namespace Std {

// Minimal algorithm implementations for device-side code
template<typename T>
__aicore__ inline constexpr T min(T a, T b) {
    return a < b ? a : b;
}

template<typename T>
__aicore__ inline constexpr T max(T a, T b) {
    return a > b ? a : b;
}

} // namespace Std
} // namespace AscendC

#endif
