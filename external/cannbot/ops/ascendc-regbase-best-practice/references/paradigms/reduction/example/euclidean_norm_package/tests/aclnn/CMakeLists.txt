# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# =============================================================================
# euclidean_norm_package/tests/aclnn/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the ACLNN integration test of EuclideanNorm.
#        Compiles test_aclnn.cpp and links it against the installed vendor
#        operator libraries to verify end-to-end ACLNN API correctness.
#
# Operator name variants:
#   - PascalCase:   EuclideanNorm
#   - snake_case:   euclidean_norm
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(test_aclnn_euclidean_norm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CANN_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(CANN_HOME $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(CANN_HOME $ENV{ASCEND_TOOLKIT_HOME})
    else()
        message(FATAL_ERROR
            "CANN_HOME not set. Pass -DCANN_HOME=<cann>, or source <CANN>/set_env.sh"
            " to export ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME.")
    endif()
endif()

include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/include
    ${CANN_HOME}/opp/vendors/EuclideanNorm/op_api/include  # aclnn_euclidean_norm.h
)

link_directories(
    ${CANN_HOME}/lib64
    ${CANN_HOME}/opp/vendors/EuclideanNorm/op_api/lib  # libcust_opapi.so
)

add_executable(test_aclnn test_aclnn.cpp)

target_link_libraries(test_aclnn
    ascendcl
    nnopbase
    cust_opapi
)
