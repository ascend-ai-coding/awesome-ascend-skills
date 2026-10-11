# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Configuration
# =============================================================================
set(OP_NAME add_example)
set(UT_EXE ${OP_NAME}_op_api_ut)

# Default SOC version for UT
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

message(STATUS "Building op_api UT for SOC_VERSION: ${SOC_VERSION}")

# =============================================================================
# Op API Source Files
# =============================================================================
set(OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)

set(OP_API_SRCS
    ${OP_API_DIR}/${OP_NAME}.cpp
    ${OP_API_DIR}/aclnn_${OP_NAME}.cpp
)

# =============================================================================
# UT Test Sources
# =============================================================================
set(UT_TEST_SRCS
    test_${OP_NAME}.cpp
    test_aclnn_${OP_NAME}.cpp
)

# =============================================================================
# UT Test Utilities
# =============================================================================
set(UT_UTIL_SRCS
    # No .cpp files for utilities (header-only)
)

# =============================================================================
# UT Test Main
# =============================================================================
set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_api_main.cpp)

# =============================================================================
# Build UT Executable
# =============================================================================
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${OP_API_SRCS}
    ${UT_UTIL_SRCS}
)

# =============================================================================
# Include Directories for UT
# =============================================================================
target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}/include
    ${OP_API_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35
)

# =============================================================================
# Compile Definitions for UT
# Key: Use old ABI to match CANN libraries (same as op_host)
# =============================================================================
target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

# =============================================================================
# Compile Options
# =============================================================================
target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

# =============================================================================
# Link Directories for UT
# =============================================================================
target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
)

# =============================================================================
# Link Libraries for UT
# Key: Link CANN metadef libraries and Google Test
# =============================================================================
target_link_libraries(${UT_EXE} PRIVATE
    # Google Test (built with old ABI)
    gtest
    gmock
    
    -Wl,--no-as-needed
    
    # CANN op_api layer libraries
    $ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopapi.so
    
    # CANN metadef libraries (required for op_api layer)
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
    
    # Standard libraries
    dl
    pthread
)

# =============================================================================
# Set RPATH for UT
# =============================================================================
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# =============================================================================
# Print Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== op_api UT Build Configuration ===")
message(STATUS "  UT Executable: ${UT_EXE}")
message(STATUS "  SOC Version: ${SOC_VERSION}")
message(STATUS "  Source Files:")
foreach(src ${OP_API_SRCS})
    message(STATUS "    - ${src}")
endforeach()
message(STATUS "  Test Files:")
foreach(test ${UT_TEST_SRCS})
    message(STATUS "    - ${test}")
endforeach()
message(STATUS "")
message(STATUS "  ABI Strategy:")
message(STATUS "    - ALL code uses OLD ABI (_GLIBCXX_USE_CXX11_ABI=0)")
message(STATUS "    - Matches CANN libraries (libplatform.so, libmetadef.so)")
message(STATUS "")
message(STATUS "  Test Coverage:")
message(STATUS "    - l0op::AddExample: Nullptr checks, invalid broadcast")
message(STATUS "    - aclnnAddExampleGetWorkspaceSize: Parameter validation, dtype/shape checks")
message(STATUS "=====================================================================")
message(STATUS "")
