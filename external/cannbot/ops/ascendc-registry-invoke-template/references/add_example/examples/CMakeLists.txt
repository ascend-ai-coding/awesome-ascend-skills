# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
# 设置工程名
project(ACLNN_EXAMPLE)

# 设置C++编译标准
add_compile_options(-std=c++11)

# 设置编译输出目录为当前目录下的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

# 设置调试和发布模式的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 添加可执行文件，指定算子调用的*.cpp文件
add_executable(test_aclnn_add_example
test_aclnn_add_example.cpp)

# ASCEND_PATH（CANN软件包目录，请根据实际路径修改）
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

# 查找自定义算子头文件（按头文件名精确匹配，避免多 vendor 目录误选）
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_add_example.h
    PATHS
        ${ASCEND_PATH}/opp/vendors/add_example_custom/op_api/include
        /usr/local/Ascend/opp/vendors/add_example_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/include
)

if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "未找到自定义算子头文件 aclnn_add_example.h，请先安装算子包")
endif()
message(STATUS "自定义算子头文件目录: ${CUSTOM_OP_INCLUDE_DIR}")

# 查找自定义算子库
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${ASCEND_PATH}/opp/vendors/add_example_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/add_example_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/add_example_custom/op_api/lib
)

if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "未找到自定义算子库 libcust_opapi.so，请先安装算子包")
endif()

# 设置头文件路径
include_directories(
    ${ASCEND_PATH}/include
    ${CUSTOM_OP_INCLUDE_DIR}
)

# 链接所需的动态库（libopapi 提供 l0op::ViewCopy/Contiguous 等符号）
target_link_libraries(test_aclnn_add_example PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi.so
    ${CUSTOM_OP_LIBRARY}
)
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_options(test_aclnn_add_example PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
)

# 安装目标文件到bin目录
install(TARGETS test_aclnn_add_example DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
