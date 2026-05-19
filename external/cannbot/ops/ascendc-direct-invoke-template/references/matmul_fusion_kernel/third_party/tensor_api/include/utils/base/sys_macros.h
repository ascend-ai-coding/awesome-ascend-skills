/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * System macros for Ascend C tensor API.
 */
#ifndef INCLUDE_TENSOR_API_UTILS_BASE_SYS_MACROS_H
#define INCLUDE_TENSOR_API_UTILS_BASE_SYS_MACROS_H

// Macro helpers for the tensor API
#define ASCENDC_TENSOR_API_CONCAT_(a, b) a##b
#define ASCENDC_TENSOR_API_CONCAT(a, b) ASCENDC_TENSOR_API_CONCAT_(a, b)

#endif
