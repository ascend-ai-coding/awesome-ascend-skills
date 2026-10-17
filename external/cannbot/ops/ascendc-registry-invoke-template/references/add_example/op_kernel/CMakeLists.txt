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
# Kernel sources - Architecture-specific
# 根据 ASCEND_COMPUTE_UNIT 选择正确的 kernel 文件
# =============================================================================

# Arch35 芯片: ascend950
if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE AddExample
        KERNEL_DIR .
        KERNEL_FILE add_example_arch35.cpp
    )
else()
    # Arch32 芯片: ascend910b, ascend910_93, etc.
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE AddExample
        KERNEL_DIR .
        KERNEL_FILE add_example_arch22.cpp
    )
endif()

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
