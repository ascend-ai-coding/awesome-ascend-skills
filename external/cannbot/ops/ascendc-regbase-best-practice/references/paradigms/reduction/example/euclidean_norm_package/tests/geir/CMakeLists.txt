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
# euclidean_norm_package/tests/geir/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the GE (Graph Engine) IR integration test of
#        EuclideanNorm.  Compiles test_geir.cpp and links it against the
#        installed vendor graph libraries to verify graph compilation and
#        execution with a value-dependent axes input.
#
# What this test validates:
#   - REG_OP(EuclideanNorm) is correctly registered and discoverable.
#   - GE can build a graph containing EuclideanNorm with Const axes.
#   - The graph can be compiled and executed via ge::Session.
#
# Operator name variants:
#   - PascalCase:   EuclideanNorm
#   - snake_case:   euclidean_norm
#   - UPPER_SNAKE:  EUCLIDEAN_NORM
# =============================================================================


cmake_minimum_required(VERSION 3.16)
project(test_geir_euclidean_norm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN SDK 路径：-DCANN_HOME > $ASCEND_HOME_PATH > $ASCEND_TOOLKIT_HOME > 报错
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

# --- Include directories ---
#   x86_64-linux/include: base architecture headers (acl/, graph/ base, ge/ base).
#   graph subdirectory: graph.h, types.h, tensor.h for GE graph objects.
#   ge subdirectory: ge_api.h, ge_error_codes.h for session API.
#   opp/built-in/op_proto/inc: built-in operators like op::Data (feed nodes).
#   opp/vendors/EuclideanNorm/op_proto/inc: INSTALLED vendor op_proto.h
#     containing REG_OP(EuclideanNorm).  Must use the installed version.
include_directories(
    ${CANN_HOME}/x86_64-linux/include
    ${CANN_HOME}/x86_64-linux/include/graph
    ${CANN_HOME}/x86_64-linux/include/ge
    ${CANN_HOME}/opp/built-in/op_proto/inc
    ${CANN_HOME}/opp/vendors/EuclideanNorm/op_proto/inc
)

# --- Link directories ---
link_directories(
    ${CANN_HOME}/x86_64-linux/lib64
)

# --- Test executable ---
add_executable(test_geir test_geir.cpp)

# --- Link libraries ---
#   graph:        GE graph IR library (graph construction, operator nodes).
#   ge_runner:    GE session runtime (graph compilation and execution).
#   graph_base:   GE tensor primitives (Shape, TensorDesc).
#   ascendcl:     ACL runtime for fp16 conversion utilities.
target_link_libraries(test_geir
    graph
    ge_runner
    graph_base
    ascendcl
)
