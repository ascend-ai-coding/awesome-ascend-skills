# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)
project(aclnn_matmul_blaze_example_test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH not set")
endif()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
add_executable(test_aclnn_matmul_blaze_example test_aclnn_matmul_blaze_example.cpp)
find_path(CUSTOM_OP_INCLUDE_DIR NAMES aclnn_matmul_blaze_example.h
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/matmul_blaze_example_custom/op_api/include)
if(NOT CUSTOM_OP_INCLUDE_DIR)
  message(FATAL_ERROR "Custom operator include directory not found")
endif()
find_library(CUSTOM_OP_LIBRARY cust_opapi
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../build/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/packages/vendors/matmul_blaze_example_custom/op_api/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/op_host)
target_include_directories(test_aclnn_matmul_blaze_example PRIVATE ${CANN_ROOT}/include ${CUSTOM_OP_INCLUDE_DIR})
target_link_libraries(test_aclnn_matmul_blaze_example
  ${CANN_ROOT}/lib64/libascendcl.so ${CANN_ROOT}/lib64/libopapi.so ${CANN_ROOT}/lib64/libnnopbase.so ${CUSTOM_OP_LIBRARY})
