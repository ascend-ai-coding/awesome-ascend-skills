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
project(add_example_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths - Use official ASCEND_HOME_PATH environment variable
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# =============================================================================
# Build Google Test with OLD ABI (Reference: ops-math/cmake/third_party/gtest.cmake)
# CRITICAL: Google Test MUST use old ABI to match CANN libraries (libplatform.so)
# =============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

set(NLOHMANN_JSON_VERSION "3.11.3")
set(NLOHMANN_JSON_TARBALL ${CMAKE_BINARY_DIR}/downloads/json-${NLOHMANN_JSON_VERSION}.tar.xz)
set(NLOHMANN_JSON_INSTALL_DIR ${CMAKE_BINARY_DIR}/3rd_party/nlohmann_json)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/downloads)
if(NOT EXISTS ${NLOHMANN_JSON_TARBALL})
    message(STATUS "Downloading nlohmann/json v${NLOHMANN_JSON_VERSION}...")
    execute_process(
        COMMAND curl -fsSL -o ${NLOHMANN_JSON_TARBALL}
            https://github.com/nlohmann/json/releases/download/v${NLOHMANN_JSON_VERSION}/json.tar.xz
        RESULT_VARIABLE _dl_result TIMEOUT 60
    )
    if(NOT _dl_result EQUAL 0)
        message(FATAL_ERROR "Failed to download nlohmann/json: ${_dl_result}")
    endif()
endif()
include(ExternalProject)
ExternalProject_Add(third_party_nlohmann_json
    URL ${NLOHMANN_JSON_TARBALL}
    CONFIGURE_COMMAND "" BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${NLOHMANN_JSON_INSTALL_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${NLOHMANN_JSON_INSTALL_DIR}/include
)
file(MAKE_DIRECTORY ${NLOHMANN_JSON_INSTALL_DIR}/include)
add_library(nlohmann_json INTERFACE IMPORTED GLOBAL)
set_target_properties(nlohmann_json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${NLOHMANN_JSON_INSTALL_DIR}/include)
add_dependencies(nlohmann_json third_party_nlohmann_json)

# Use our custom-built gtest instead of find_package(GTest)
# This ensures ABI compatibility with libplatform.so

# =============================================================================
# Common UT Framework Sources
# =============================================================================
set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# =============================================================================
# Common Include Directories
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

# =============================================================================
# Common Library Directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)

# =============================================================================
# Compiler Options
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

if(ENABLE_GCOV)
    add_compile_options(-fprofile-arcs -ftest-coverage)
    add_link_options(-fprofile-arcs -ftest-coverage)
endif()

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory(op_host)
add_subdirectory(op_api)
