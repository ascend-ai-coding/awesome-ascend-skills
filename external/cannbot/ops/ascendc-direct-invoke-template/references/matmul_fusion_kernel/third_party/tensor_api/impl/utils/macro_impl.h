/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "tensor_api/impl/utils/macro_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file macro_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_UTILS_MACRO_IMPL_H
#define IMPL_TENSOR_API_UTILS_MACRO_IMPL_H
#include "utils/base/sys_macros.h"
#include "utils/base/sys_constants.h"

#if !defined(ASCENDC_CPU_DEBUG)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using fp4x2_e2m1_t = float4_e2m1x2_t;
    using fp4x2_e1m2_t = float4_e1m2x2_t;
    using fp8_e5m2_t = float8_e5m2_t;
    using fp8_e4m3fn_t = float8_e4m3_t;
    using fp8_e8m0_t = float8_e8m0_t;
#else
    using fp4x2_e2m1_t = uint8_t;
    using fp4x2_e1m2_t = uint8_t;
    using fp8_e5m2_t = uint8_t;
    using fp8_e4m3fn_t = uint8_t;
    using fp8_e8m0_t = uint8_t;
#endif
#endif

#if (__CCE__)
    #ifndef ASCENDC_HOST
        #define ASCENDC_HOST __host__
    #endif
    #ifndef ASCENDC_AICORE
        #define ASCENDC_AICORE __aicore__
    #endif
    #ifndef ASCENDC_HOST_AICORE
        #define ASCENDC_HOST_AICORE __host_aicore__
    #endif
#else
    #ifndef ASCENDC_HOST
        #define ASCENDC_HOST
    #endif
    #ifndef ASCENDC_AICORE
        #define ASCENDC_AICORE
    #endif
    #ifndef ASCENDC_HOST_AICORE
        #define ASCENDC_HOST_AICORE
    #endif
#endif

#endif //IMPL_TENSOR_API_UTILS_MACRO_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
