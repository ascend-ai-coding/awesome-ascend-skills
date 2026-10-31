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
# xdivy_custom_package/tests/tiling/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the tiling unit test of Xdivy.
#        Builds two targets:
#          1. xdivy_op_host_ut_lib (shared library) — compiles the operator
#             registration sources (OP_ADD, IMPL_OP_INFERSHAPE, TilingFunc,
#             graph infer) into a .so that is dlopen'd by the test.
#          2. test_tiling (executable) — loads the op_host .so at runtime,
#             then directly invokes the registered TilingFunc to verify tiling
#             calculations across various broadcast scenarios.
#
# Key design decisions (same as add_custom_package/tests/tiling):
#   - The op_host .so is SHARED so it can be dlopen'd via
#     OpImplSpaceRegistryV2::AddSoToRegistry (mirrors the CANN framework's own
#     loading mechanism, making the test representative).
#   - The test executable links only libopp_registry.so + libmetadef.so.
#   - _GLIBCXX_USE_CXX11_ABI=0 matches CANN's old C++ ABI.
#   - RPATH=$ORIGIN lets the test find the sibling .so without LD_LIBRARY_PATH.
#
# Operator name variants:
#   - PascalCase:   Xdivy
#   - snake_case:   xdivy
#   - UPPER_SNAKE:  XDIVY
#   - camelCase:    xdivy
# =============================================================================


cmake_minimum_required(VERSION 3.16)
project(test_tiling_xdivy VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# CANN_HOME resolution strategy (in priority order):
#   1. User-supplied: -DCANN_HOME=/path/to/cann
#   2. Environment: $ASCEND_HOME_PATH
#   3. Hardcoded default: /home/developer/Ascend/cann-9.1.0
if(NOT DEFINED CANN_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(CANN_HOME $ENV{ASCEND_HOME_PATH})
    else()
        set(CANN_HOME /home/developer/Ascend/cann-9.1.0)
    endif()
endif()


# ASCEND_HOST_ARCH: host architecture string (e.g. "x86_64", "aarch64").
if(NOT DEFINED ASCEND_HOST_ARCH)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE _UNAME_M OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ASCEND_HOST_ARCH ${_UNAME_M})
endif()
set(ASCEND_HOST_LINUX "${ASCEND_HOST_ARCH}-linux")


# CANN include and library paths:
set(CANN_INC ${CANN_HOME}/${ASCEND_HOST_LINUX}/include)
set(CANN_PKG_INC ${CANN_HOME}/${ASCEND_HOST_LINUX}/pkg_inc)
set(CANN_LIB ${CANN_HOME}/lib64)

message(STATUS "CANN_HOME: ${CANN_HOME}")
message(STATUS "CANN include: ${CANN_INC}")
message(STATUS "CANN pkg_inc: ${CANN_PKG_INC}")
message(STATUS "CANN lib: ${CANN_LIB}")


# PKG_DIR: root of the operator package (two levels up from this CMakeLists.txt).
set(PKG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)


# Compile options common to both targets.
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)


# CANN_COMMON_INCLUDE_DIRS: the two root include directories used by CANN headers.
set(CANN_COMMON_INCLUDE_DIRS
    ${CANN_INC}
    ${CANN_PKG_INC}
)


# =============================================================================
# Target 1: op_host shared library (xdivy_op_host_ut_lib.so)
# =============================================================================
#
# This .so contains the compiled operator registration sources.  Loaded at
# runtime via OpImplSpaceRegistryV2::AddSoToRegistry.
#
# Source files:
#   op_host/xdivy_def.cpp          - OP_ADD(Xdivy) operator definition.
#   op_host/xdivy_infershape.cpp   - IMPL_OP_INFERSHAPE(Xdivy).
#   op_host/arch35/xdivy_tiling_arch35.cpp - TilingFuncXdivy implementation.
#   op_graph/xdivy_graph_infer.cpp - IMPL_OP(Xdivy).InferDataType.

set(OP_HOST_SO xdivy_op_host_ut_lib)
add_library(${OP_HOST_SO} SHARED
    ${PKG_DIR}/op_host/xdivy_def.cpp
    ${PKG_DIR}/op_host/xdivy_infershape.cpp
    ${PKG_DIR}/op_host/arch35/xdivy_tiling_arch35.cpp
    ${PKG_DIR}/op_graph/xdivy_graph_infer.cpp
)
target_include_directories(${OP_HOST_SO} PRIVATE
    ${CANN_COMMON_INCLUDE_DIRS}
    ${PKG_DIR}/op_kernel/arch35    # xdivy_tiling_struct.h, xdivy_struct.h
    ${PKG_DIR}/op_graph            # xdivy_proto.h (pulled by graph_infer)
)

target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=Ascend950
    OP_TILING_LIB
    _GLIBCXX_USE_CXX11_ABI=0
)


# Link libraries for the op_host .so.
target_link_directories(${OP_HOST_SO} PRIVATE ${CANN_LIB})
target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed
    -Wl,--whole-archive
        ${CANN_LIB}/librt2_registry.a
        ${CANN_LIB}/libtiling_api.a
    -Wl,--no-whole-archive
    ${CANN_LIB}/libopp_registry.so
    ${CANN_LIB}/libregister.so
    ${CANN_LIB}/libc_sec.so
)


# RPATH for the .so.
set_target_properties(${OP_HOST_SO} PROPERTIES
    INSTALL_RPATH "${CANN_LIB}"
    BUILD_WITH_INSTALL_RPATH TRUE
)


# =============================================================================
# Target 2: tiling unit test executable (test_tiling)
# =============================================================================
#
# Directly constructs gert::TilingContext objects and invokes the registered
# TilingFunc, bypassing the ACL/GE runtime.  This allows isolated testing of
# tiling logic (including broadcast scenarios) without device hardware.

set(UT_EXE test_tiling)
add_executable(${UT_EXE} test_tiling.cpp)
target_include_directories(${UT_EXE} PRIVATE
    ${CANN_COMMON_INCLUDE_DIRS}
    ${PKG_DIR}/op_kernel/arch35    # xdivy_tiling_struct.h
)
target_compile_definitions(${UT_EXE} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_link_directories(${UT_EXE} PRIVATE ${CANN_LIB})
target_link_libraries(${UT_EXE} PRIVATE
    ${CANN_LIB}/libopp_registry.so   # Operator registry + tiling kernel lookup.
    ${CANN_LIB}/libmetadef.so        # Context builder + metadata types.
    ${CANN_LIB}/libc_sec.so          # Security library (dependency).
    dl                                # POSIX dlopen for loading the .so.
)


# RPATH for the test executable.
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${CANN_LIB}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
