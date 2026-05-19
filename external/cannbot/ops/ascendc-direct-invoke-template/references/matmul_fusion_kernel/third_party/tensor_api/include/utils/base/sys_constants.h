/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * System constants for Ascend C tensor API.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_BASE_SYS_CONSTANTS_H
#define INCLUDE_TENSOR_API_UTILS_BASE_SYS_CONSTANTS_H

#include <cstdint>

namespace AscendC {
namespace Te {

// System constants
constexpr uint64_t MAX_TENSOR_DIMS = 8;
constexpr uint64_t DEFAULT_ALIGNMENT = 32;

} // namespace Te
} // namespace AscendC

#endif
