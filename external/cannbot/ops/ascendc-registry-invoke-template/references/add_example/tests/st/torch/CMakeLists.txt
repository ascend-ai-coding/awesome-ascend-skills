# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for add_example ST test (PyTorch adapter)
cmake_minimum_required(VERSION 3.10)
project(aclnn_add_example_torch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find PyTorch
# ============================================================================
find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch path: ${TORCH_INSTALL_PREFIX}")

# ============================================================================
# Find CANN
# ============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH directory does not exist!")
endif()

set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)

message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# Find torch_npu
execute_process(
    COMMAND python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_NPU_ROOT)
    set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
endif()

if(TORCH_NPU_INCLUDE_DIR)
    message(STATUS "torch_npu include dir: ${TORCH_NPU_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "torch_npu include directory not found.")
endif()

# Find torch_npu library
find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS
        ${TORCH_NPU_ROOT}/lib
    PATH_SUFFIXES lib
)

if(TORCH_NPU_LIBRARY)
    message(STATUS "torch_npu library: ${TORCH_NPU_LIBRARY}")
else()
    message(FATAL_ERROR "torch_npu library not found.")
endif()

# Find custom operator
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_add_example.h
    PATHS
        ${CANN_ROOT}/opp/vendors/add_example_custom/op_api/include
        /usr/local/Ascend/opp/vendors/add_example_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/include
)

if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found.")
endif()

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/add_example_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/add_example_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/lib
)

if(CUSTOM_OP_LIBRARY)
    message(STATUS "Custom operator library: ${CUSTOM_OP_LIBRARY}")
else()
    message(FATAL_ERROR "cust_opapi library not found.")
endif()

# ============================================================================
# Build shared library
# ============================================================================
add_library(torch_adapter SHARED
    torch_adapter.cpp
)

target_include_directories(torch_adapter PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(torch_adapter PRIVATE
    ${TORCH_LIBRARIES}
    ${ASCENDCL_LIBRARIES}
    ${CANN_LIBRARIES}
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CUSTOM_OP_LIBRARY}
    ${TORCH_NPU_LIBRARY}
)

# Set rpath
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
get_filename_component(TORCH_NPU_LIB_DIR ${TORCH_NPU_LIBRARY} DIRECTORY)
target_link_options(torch_adapter PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

install(TARGETS torch_adapter DESTINATION lib)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build config:")
message(STATUS "  Output: libtorch_adapter.so")
message(STATUS "  Test:   python3 ../test.py --lib ./libtorch_adapter.so")
message(STATUS "========================================")
message(STATUS "")
