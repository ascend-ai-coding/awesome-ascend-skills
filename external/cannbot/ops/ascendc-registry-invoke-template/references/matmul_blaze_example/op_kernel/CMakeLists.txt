# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE MatmulBlazeExample
    KERNEL_DIR .
    KERNEL_FILE matmul_blaze_example_arch35.cpp
)

npu_op_kernel_options(ascendc_kernels MatmulBlazeExample
    OPTIONS
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/include/blaze
        -I${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_api
        -I${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_api/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_api/impl
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_host/arch35
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
