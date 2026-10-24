# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Configuration (参考 ops-math 模式)
# =============================================================================
set(OP_NAME add_example)
set(UT_EXE ${OP_NAME}_op_host_ut)

# Default SOC version for UT
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

message(STATUS "Building UT for SOC_VERSION: ${SOC_VERSION}")

# =============================================================================
# Op Host Source Files (参考 ops-math：编译成 .so)
# =============================================================================
set(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host)

# 根据架构选择 tiling 源文件
if(SOC_VERSION STREQUAL "Ascend950")
    set(TILING_SRC ${OP_HOST_DIR}/arch35/${OP_NAME}_tiling.cpp)
    set(TILING_DATA_H ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35/${OP_NAME}_tiling_data.h)
else()
    set(TILING_SRC ${OP_HOST_DIR}/arch22/${OP_NAME}_tiling.cpp)
    set(TILING_DATA_H ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch22/${OP_NAME}_tiling_data.h)
endif()

set(OP_HOST_SRCS
    ${OP_HOST_DIR}/${OP_NAME}_def.cpp
    ${OP_HOST_DIR}/${OP_NAME}_infershape.cpp
    ${TILING_SRC}
)

# =============================================================================
# UT Test Sources
# =============================================================================
set(UT_TEST_SRCS
    test_${OP_NAME}_infershape.cpp
    test_${OP_NAME}_tiling.cpp
)

# =============================================================================
# UT Test Main
# =============================================================================
set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_host_main.cpp)

# =============================================================================
# Build op_host as SHARED library (参考 ops-math 模式)
# 关键：这个 .so 包含所有注册代码，通过 AddSoToRegistry() 加载
# =============================================================================
set(OP_HOST_SO ${OP_NAME}_op_host_ut_lib)

add_library(${OP_HOST_SO} SHARED
    ${OP_HOST_SRCS}
)

# Include Directories for .so
target_include_directories(${OP_HOST_SO} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${OP_HOST_DIR}/arch22
    ${OP_HOST_DIR}/arch35
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35
)

# Compile Definitions for .so
# 关键：使用旧 ABI 以匹配 libplatform.so 和 libtiling_api.a（参考 ops-math）
target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

# Link Libraries for .so (参考 ops-math)
target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed
    
    # 关键：whole-archive 链接静态库
    -Wl,--whole-archive
        $ENV{ASCEND_HOME_PATH}/lib64/librt2_registry.a
        $ENV{ASCEND_HOME_PATH}/lib64/libtiling_api.a
    -Wl,--no-whole-archive
    
    # 其他必需的库
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
)

# =============================================================================
# Build UT Executable (参考 ops-math 模式)
# 长期方案：所有代码使用旧 ABI，直接链接 libplatform.so
# =============================================================================
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

# Include Directories for UT
target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${OP_HOST_DIR}/arch22
    ${OP_HOST_DIR}/arch35
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35
)

# Compile Definitions for UT
# 关键：UT 可执行文件也使用旧 ABI（参考 ops-math）
target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

# Compile Options
target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

# Link Directories for UT
target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}  # 添加当前构建目录以找到 .so
)

# =============================================================================
# Link Libraries for UT (参考 ops-math)
# 关键：链接 libplatform.so，不再使用 stub
# =============================================================================
target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main
    nlohmann_json
    
    -Wl,--no-as-needed
    
    # 参考 ops-math 的库链接顺序
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
    
    dl
    pthread
)

# =============================================================================
# Set RPATH for UT (确保运行时能找到 .so)
# =============================================================================
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# =============================================================================
# Print Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== UT Build Configuration (ops-math mode - Long-term Solution) ===")
message(STATUS "  UT Executable: ${UT_EXE}")
message(STATUS "  OP Host Library: lib${OP_HOST_SO}.so")
message(STATUS "  SOC Version: ${SOC_VERSION}")
message(STATUS "  Tiling Source: ${TILING_SRC}")
message(STATUS "  Tiling Data Header: ${TILING_DATA_H}")
message(STATUS "")
message(STATUS "  ABI Strategy (ops-math unified ABI):")
message(STATUS "    - ALL code uses OLD ABI (_GLIBCXX_USE_CXX11_ABI=0)")
message(STATUS "    - Google Test: Built from source with old ABI")
message(STATUS "    - UT code: Old ABI (matches gtest and CANN libs)")
message(STATUS "    - op_host.so: Old ABI (matches libtiling_api.a)")
message(STATUS "")
message(STATUS "  Direct Linking (No Stub):")
message(STATUS "    - Link libplatform.so directly")
message(STATUS "    - No ABI mismatch issues")
message(STATUS "    - Platform infos stub DELETED")
message(STATUS "")
message(STATUS "  Reference: ops-math/cmake/third_party/gtest.cmake")
message(STATUS "=====================================================================")
message(STATUS "")
