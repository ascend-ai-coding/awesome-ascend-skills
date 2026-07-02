# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Kernel sources - Arch35 only (Ascend950 / DAV_3510)
# [MODIFY] 如需支持更多架构，添加对应的 npu_op_kernel_sources 块
# =============================================================================
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE QuantMatmulGeluExample
    KERNEL_DIR .
    KERNEL_FILE quant_matmul_gelu_example_arch35.cpp
)

# =============================================================================
# Add Blaze/tensor_api include paths via kernel compile options
# =============================================================================
npu_op_kernel_options(ascendc_kernels QuantMatmulGeluExample
    OPTIONS
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch35/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch35/common
        -I${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensor_api
        -I${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensor_api/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensor_api/impl
)

# =============================================================================
# Build kernel library
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)