# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# BuildGoogleTest.cmake
# 
# Purpose: Build Google Test from source with OLD ABI to match CANN libraries
# 
# Reference: ops-math/cmake/third_party/gtest.cmake
# ----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

# CRITICAL: Force build from source with OLD ABI to match CANN libraries
# We cannot use system Google Test because it uses new ABI by default
message(STATUS "")
message(STATUS "=== Building Google Test from source with OLD ABI ===")
message(STATUS "  Reason: System gtest uses new ABI, incompatible with libplatform.so")
message(STATUS "  Target ABI: _GLIBCXX_USE_CXX11_ABI=0 (old ABI)")
message(STATUS "")

set(GTEST_VERSION "1.14.0")
set(GTEST_INSTALL_DIR ${CMAKE_BINARY_DIR}/3rd_party/gtest)
set(GTEST_SOURCE_DIR ${CMAKE_BINARY_DIR}/3rd_party/gtest-src)

# Download URL (using gitcode mirror for China access)
set(GTEST_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v${GTEST_VERSION}/googletest-${GTEST_VERSION}.tar.gz")

# Compiler flags - CRITICAL: Use OLD ABI
set(GTEST_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -w")
set(GTEST_C_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -w")

include(ExternalProject)
ExternalProject_Add(
    third_party_gtest
    URL ${GTEST_URL}
    TLS_VERIFY OFF
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads
    SOURCE_DIR ${GTEST_SOURCE_DIR}
    INSTALL_DIR ${GTEST_INSTALL_DIR}
    
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${GTEST_C_FLAGS}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBUILD_SHARED_LIBS=OFF
        -Dgtest_build_tests=OFF
        -Dgtest_build_samples=OFF
        -Dgmock_build_tests=OFF
    
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)

# Create imported targets (matching ops-math)
set(GTEST_INCLUDE_DIR ${GTEST_INSTALL_DIR}/include)

# Ensure include directory exists for target_link_libraries
file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR})

# gtest
add_library(gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_DIR}/lib/libgtest.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)
add_dependencies(gtest third_party_gtest)

# gtest_main
add_library(gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_DIR}/lib/libgtest_main.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)
add_dependencies(gtest_main third_party_gtest)

# gmock
add_library(gmock STATIC IMPORTED GLOBAL)
set_target_properties(gmock PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_DIR}/lib/libgmock.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)
add_dependencies(gmock third_party_gtest)

# gmock_main
add_library(gmock_main STATIC IMPORTED GLOBAL)
set_target_properties(gmock_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_INSTALL_DIR}/lib/libgmock_main.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)
add_dependencies(gmock_main third_party_gtest)

# Create interface library for UT (matching ops-math intf_llt_pub_asan_cxx17)
add_library(intf_llt_pub_asan_cxx17 INTERFACE)
target_include_directories(intf_llt_pub_asan_cxx17 INTERFACE 
    ${GTEST_INCLUDE_DIR}
)
target_compile_definitions(intf_llt_pub_asan_cxx17 INTERFACE 
    _GLIBCXX_USE_CXX11_ABI=0 
    CFG_BUILD_DEBUG
)
target_compile_options(intf_llt_pub_asan_cxx17 INTERFACE 
    -g 
    --coverage 
    -fprofile-arcs 
    -ftest-coverage 
    -w 
    -std=c++17 
    -fPIC
)
target_link_options(intf_llt_pub_asan_cxx17 INTERFACE 
    -fprofile-arcs 
    -ftest-coverage
)
target_link_libraries(intf_llt_pub_asan_cxx17 INTERFACE 
    gcov 
    pthread
)

message(STATUS "")
message(STATUS "=== Google Test Build Configuration (ops-math mode) ===")
message(STATUS "  Version: ${GTEST_VERSION}")
message(STATUS "  Install Dir: ${GTEST_INSTALL_DIR}")
message(STATUS "  ABI: OLD (_GLIBCXX_USE_CXX11_ABI=0)")
message(STATUS "  Reason: Match CANN libraries (libplatform.so, libtiling_api.a)")
message(STATUS "=====================================================")
message(STATUS "")
