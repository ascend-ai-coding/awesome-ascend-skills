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
# xdivy_custom_package/tests/aclnn/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the ACLNN integration test of Xdivy.
#        Compiles test_aclnn.cpp and links it against the installed vendor
#        operator libraries to verify end-to-end ACLNN API correctness,
#        including broadcast scenarios.
#
# Key constraint:
#   The test uses INSTALLED headers and libraries under
#   ${CANN_HOME}/opp/vendors/Xdivy/, NOT the source/build tree.  This ensures
#   the test validates the installation package produced by build.sh.
#
# Operator name variants:
#   - PascalCase:   Xdivy
#   - snake_case:   xdivy
#   - UPPER_SNAKE:  XDIVY
#   - camelCase:    xdivy
# =============================================================================


# --- Minimum CMake version ---
cmake_minimum_required(VERSION 3.16)

# --- Project name ---
project(test_aclnn_xdivy VERSION 1.0.0 LANGUAGES CXX)

# --- C++ standard: must match the operator (C++17) ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CANN installation root ---
set(CANN_HOME /home/developer/Ascend/cann-9.1.0)

# --- Include directories ---
include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/include
    ${CANN_HOME}/opp/vendors/Xdivy/op_api/include  # aclnn_xdivy.h
)

# --- Link directories ---
link_directories(
    ${CANN_HOME}/lib64
    ${CANN_HOME}/opp/vendors/Xdivy/op_api/lib      # libcust_opapi.so
)

# --- Test executable ---
add_executable(test_aclnn test_aclnn.cpp)

# --- Link libraries ---
target_link_libraries(test_aclnn
    ascendcl
    nnopbase
    cust_opapi
)
