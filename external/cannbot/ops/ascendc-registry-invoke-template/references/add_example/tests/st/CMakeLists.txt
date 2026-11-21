# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for add_example ST test
cmake_minimum_required(VERSION 3.10)
project(aclnn_add_example_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MOCK "Use mock aclnn for testing without real operator" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK aclnn + CPU golden - full test validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden comparison")
endif()

# Set CANN paths (use official ASCEND_HOME_PATH)
if(NOT USE_MOCK)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH environment variable is not set! source set_env.sh in cann install directory first.")
    endif()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()
    
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
    set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)
    
    message(STATUS "CANN_ROOT: ${CANN_ROOT}")
endif()

add_executable(test_aclnn_add_example test_aclnn_add_example.cpp)

# Link Ascend libraries for Real mode
if(NOT USE_MOCK)
    # Find custom operator include directory
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_add_example.h
        PATHS 
            ${CANN_ROOT}/opp/vendors/add_example_custom/op_api/include
            /usr/local/Ascend/opp/vendors/add_example_custom/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/include
    )
    
    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found. Please install the operator first.")
    endif()
    
    target_include_directories(test_aclnn_add_example PRIVATE 
        ${ASCENDCL_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIRS}
        ${CUSTOM_OP_INCLUDE_DIR}
    )
    
    # Find custom operator library
    find_library(CUSTOM_OP_LIBRARY cust_opapi
        PATHS 
            ${CANN_ROOT}/opp/vendors/add_example_custom/op_api/lib
            /usr/local/Ascend/opp/vendors/add_example_custom/op_api/lib
            $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/lib
    )
    
    if(NOT CUSTOM_OP_LIBRARY)
        message(WARNING "cust_opapi library not found, using default linkage")
    endif()
    
    target_link_libraries(test_aclnn_add_example 
        ${ASCENDCL_LIBRARIES}
        ${CANN_LIBRARIES}
        ${CANN_ROOT}/lib64/libnnopbase.so
        ${CUSTOM_OP_LIBRARY}
    )
endif()

# Install target
install(TARGETS test_aclnn_add_example DESTINATION bin)
