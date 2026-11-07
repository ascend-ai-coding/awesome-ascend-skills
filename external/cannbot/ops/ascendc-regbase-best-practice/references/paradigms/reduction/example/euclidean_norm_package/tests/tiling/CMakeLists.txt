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
# euclidean_norm_package/tests/tiling/CMakeLists.txt
# =============================================================================
# Role: CMake build file for the tiling unit test of EuclideanNorm.
#        Builds two targets:
#          1. euclidean_norm_op_host_ut_lib (shared library) — compiles the
#             operator registration sources into a .so that is dlopen'd by the
#             test.
#          2. test_tiling (executable) — loads the .so, constructs TilingContext,
#             and directly invokes the registered TilingFunc.
#
# NOTE: EuclideanNorm requires additional include paths and link libraries
#       beyond add_custom_package (see main CMakeLists.txt for details):
#         - CANN op_common headers (log, util, infershape_reduce_util)
#         - ops-nn common/inc (tiling_util, tiling_templates_registry)
#         - asc/include/utils/tiling (platform_ascendc.h)
#         - libops_base (Ops::Base::* symbols)
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(test_tiling_euclidean_norm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CANN_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(CANN_HOME $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(CANN_HOME $ENV{ASCEND_TOOLKIT_HOME})
    else()
        message(FATAL_ERROR
            "CANN_HOME not set. Pass -DCANN_HOME=<cann>, or source <CANN>/set_env.sh")
    endif()
endif()

if(NOT DEFINED ASCEND_HOST_ARCH)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE _UNAME_M OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ASCEND_HOST_ARCH ${_UNAME_M})
endif()
set(ASCEND_HOST_LINUX "${ASCEND_HOST_ARCH}-linux")

set(CANN_INC ${CANN_HOME}/${ASCEND_HOST_LINUX}/include)
set(CANN_PKG_INC ${CANN_HOME}/${ASCEND_HOST_LINUX}/pkg_inc)
set(CANN_LIB ${CANN_HOME}/lib64)

message(STATUS "CANN_HOME: ${CANN_HOME}")

set(PKG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Extra include paths for EuclideanNorm reduce utilities
set(CANN_OP_COMMON_INC ${CANN_INC}/op_common)
set(CANN_OP_COMMON_HOST_INC ${CANN_INC}/op_common/op_host)

add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

set(CANN_COMMON_INCLUDE_DIRS
    ${CANN_INC}
    ${CANN_PKG_INC}
    ${CANN_OP_COMMON_INC}
    ${CANN_OP_COMMON_HOST_INC}
)

# =============================================================================
# Target 1: op_host shared library (euclidean_norm_op_host_ut_lib.so)
# =============================================================================
set(OP_HOST_SO euclidean_norm_op_host_ut_lib)
add_library(${OP_HOST_SO} SHARED
    ${PKG_DIR}/op_host/euclidean_norm_def.cpp
    ${PKG_DIR}/op_host/euclidean_norm_infershape.cpp
    ${PKG_DIR}/op_host/arch35/euclidean_norm_tiling_arch35.cpp
    ${PKG_DIR}/op_graph/euclidean_norm_proto.cc
)
target_include_directories(${OP_HOST_SO} PRIVATE
    ${CANN_COMMON_INCLUDE_DIRS}
    ${PKG_DIR}/op_kernel/arch35
    ${PKG_DIR}/op_graph
    ${PKG_DIR}
)
target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=Ascend950
    OP_TILING_LIB
    _GLIBCXX_USE_CXX11_ABI=0
)
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
    ${CANN_LIB}/libops_base.so
)
set_target_properties(${OP_HOST_SO} PROPERTIES
    INSTALL_RPATH "${CANN_LIB}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# =============================================================================
# Target 2: tiling unit test executable (test_tiling)
# =============================================================================
set(UT_EXE test_tiling)
add_executable(${UT_EXE} test_tiling.cpp)
target_include_directories(${UT_EXE} PRIVATE
    ${CANN_COMMON_INCLUDE_DIRS}
    ${PKG_DIR}/op_kernel/arch35
)
target_compile_definitions(${UT_EXE} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_link_directories(${UT_EXE} PRIVATE ${CANN_LIB})
target_link_libraries(${UT_EXE} PRIVATE
    ${CANN_LIB}/libopp_registry.so
    ${CANN_LIB}/libmetadef.so
    ${CANN_LIB}/libplatform.so
    ${CANN_LIB}/libc_sec.so
    dl
)
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${CANN_LIB}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
