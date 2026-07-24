# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(matmul_blaze_example_op_prj)
find_package(ASC REQUIRED)

set(ARCH35_COMPUTE_UNITS ascend950)
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH35_COMPUTE_UNITS})
endif()

set(package_name matmul_blaze_example_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")
