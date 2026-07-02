# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Host build configuration for quant_matmul_gelu_example operator (AIC+AIV MIX).
# NOTE: 本算子使用 Blaze/tensor_api，op_host CMakeLists 通过 -I 引入
#       op_host/arch35/include 和 arch35/common 中的 Blaze tiling/engine 头文件，
#       并保留 -I op_kernel/arch35 以引入 Host/Kernel 共享的 TilingData/TilingKey 头文件。
# =============================================================================

# Detect host architecture for CANN path resolution
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()
message(STATUS "Using ASCEND_ARCH_DIR: ${ASCEND_ARCH_DIR}")

# =============================================================================
# Host source files - Op Definition
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_gelu_example_def.cpp
)

# =============================================================================
# Host source files - InferShape
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_gelu_example_infershape.cpp
)

# =============================================================================
# Host source files - Tiling (arch35 only)
# =============================================================================
set(host_ops_tiling_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35/quant_matmul_gelu_example_tiling.cpp
)

# =============================================================================
# Combine all host sources
# =============================================================================
set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Run autogen
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -std=c++17
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/graph
        -I$ENV{ASCEND_HOME_PATH}/include
        -I$ENV{ASCEND_HOME_PATH}/asc/include/tiling
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch35
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/asc/include/tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
)

set(SECURE_COMPILE_OPTIONS -fPIC -fstack-protector-all)
set(SECURE_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

target_compile_options(cust_optiling PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_optiling PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# ACLNN library (hand-written)
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_quant_matmul_gelu_example.cpp
    ${op_api_dir}/quant_matmul_gelu_example.cpp
)
npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/aicpu
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

find_library(CANN_NNOPBASE_LIB nnopbase
    PATHS $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64 $ENV{ASCEND_HOME_PATH}/lib64)
find_library(CANN_OPAPI_LIB opapi
    PATHS $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64 $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(cust_opapi PRIVATE ${CANN_NNOPBASE_LIB} ${CANN_OPAPI_LIB})

# =============================================================================
# Graph proto library (auto-generated)
# =============================================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add libraries to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)