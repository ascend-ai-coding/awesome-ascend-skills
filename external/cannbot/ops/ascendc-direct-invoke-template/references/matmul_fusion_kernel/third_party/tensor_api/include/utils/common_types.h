/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_COMMON_TYPES_H
#define INCLUDE_TENSOR_API_UTILS_COMMON_TYPES_H

#ifndef __CCE_AICORE__
#include <cstdint>
#include <cstddef>
#endif

namespace AscendC {
namespace Te {

// Common type aliases for tensor API
using IndexType = int64_t;
using SizeType = size_t;

} // namespace Te
} // namespace AscendC

#endif
