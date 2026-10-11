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
# xdivy_custom_package/tests/geir/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the GE (Graph Engine) IR integration test of
#        Xdivy.  Compiles test_geir.cpp and links it against the installed
#        vendor graph libraries to verify graph compilation and execution
#        (including broadcast scenarios).
#
# What this test validates:
#   - REG_OP(Xdivy) is correctly registered and discoverable.
#   - GE can build a graph containing Xdivy nodes with broadcast inputs.
#   - The graph can be compiled and executed via ge::Session.
#   - Output shape matches broadcast_max(x1.shape, x2.shape).
#   - Output values match CPU golden (x1==0) ? 0 : (x1/x2) with broadcast.
#
# Operator name variants:
#   - PascalCase:   Xdivy
#   - snake_case:   xdivy
#   - UPPER_SNAKE:  XDIVY
#   - camelCase:    xdivy
# =============================================================================


cmake_minimum_required(VERSION 3.16)
project(test_geir_xdivy VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN installation root.
set(CANN_HOME /home/developer/Ascend/cann-9.1.0)

# --- Include directories ---
#   opp/vendors/Xdivy/op_proto/inc: INSTALLED vendor op_proto.h containing
#     REG_OP(Xdivy).  Must use the installed version, not build/autogen.
include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/x86_64-linux/include/graph
    ${CANN_HOME}/x86_64-linux/include/ge
    ${CANN_HOME}/opp/built-in/op_proto/inc
    ${CANN_HOME}/opp/vendors/Xdivy/op_proto/inc
)

# --- Link directories ---
link_directories(
    ${CANN_HOME}/x86_64-linux/lib64
)

# --- Test executable ---
add_executable(test_geir test_geir.cpp)

# --- Link libraries ---
target_link_libraries(test_geir
    graph
    ge_runner
    graph_base
    ascendcl
)
