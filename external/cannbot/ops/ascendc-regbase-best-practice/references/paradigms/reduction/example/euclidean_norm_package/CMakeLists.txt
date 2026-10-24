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
# euclidean_norm_package/CMakeLists.txt
# =============================================================================
#
# ROLE: Top-level CMake build script for the EuclideanNorm vendor operator package.
#   Orchestrates the complete build pipeline for a CANN custom operator:
#   1. Compile host-side sources (tiling, graph, proto) into libascend_all_ops.so
#   2. Run op_build (CANN's operator build tool) to auto-generate stubs
#   3. Overwrite auto-generated op_proto stubs with hand-written implementations
#   4. Build object libraries (tiling, opapi, proto) separately
#   5. Package everything into an NPU vendor operator package via npu_op_package
#   6. Build Ascend C kernel sources via npu_op_kernel_library
#
# OPERATOR NAME VARIANTS:
#   PascalCase   : EuclideanNorm      (used in OP_TYPE, class names, package_name)
#   snake_case   : euclidean_norm     (used in filenames, function names, namespaces)
#   UPPER_SNAKE  : EUCLIDEAN_NORM     (used in header guards, TPL macros)
#   camelCase    : euclideanNorm      (unused in this package, shown for completeness)
# =============================================================================

cmake_minimum_required(VERSION 3.16.0)
project(custom_op_prj)

# find_package(ASC REQUIRED) loads the CANN ASC CMake module,
# which provides macros: npu_op_package, npu_op_library, npu_op_kernel_sources, etc.
find_package(ASC REQUIRED)

# Target hardware compute unit. 'ascend950' targets the Ascend 950 series NPU.
set(ASCEND_COMPUTE_UNIT ascend950)

# Package name in PascalCase. This must match the OP_TYPE registered in code.
set(package_name EuclideanNorm)

# CANN installation root. Contains headers, libraries, tools, and opp files.
# For a new environment, update this to the actual CANN installation path.
set(CANN_HOME /home/developer/Ascend/cann-9.1.0)
set(CANN_INC ${CANN_HOME}/include)
set(CANN_ARCH_INC ${CANN_HOME}/x86_64-linux/include)
set(CANN_LIB ${CANN_HOME}/lib64)

# Auto-generated files directory. op_build writes generated stubs here.
set(AUTO_GEN_DIR ${CMAKE_BINARY_DIR}/autogen)

# -----------------------------------------------------------------------------
# Additional include paths required by EuclideanNorm (reduce-class operator):
#   - op_common / op_common/op_host: CANN shared op utilities (log, infershape
#     reduce util, shape/math/platform util).
# -----------------------------------------------------------------------------
set(CANN_OP_COMMON_INC ${CANN_ARCH_INC}/op_common)
set(CANN_OP_COMMON_HOST_INC ${CANN_ARCH_INC}/op_common/op_host)

# =============================================================================
# SOURCE FILE COLLECTION
# =============================================================================

# host_tiling_srcs: Host-side tiling, shape inference, and operator definition.
# These are compiled into libascend_all_ops.so for op_build consumption.
# NOTE: euclidean_norm_infershape.cpp registers InferShape (and the axes
#   value-dependency) via IMPL_OP_INFERSHAPE;
file(GLOB host_tiling_srcs
    op_host/euclidean_norm_def.cpp
    op_host/euclidean_norm_infershape.cpp
    op_host/arch35/euclidean_norm_tiling_arch35.cpp
)

# host_graph_srcs: Graph-side inference registration (InferDataType) sources.
# Used by the graph engine (GE) at graph build time.  These sources are
# compiled into cust_proto_obj (see STEP 4 proto_srcs).
file(GLOB host_graph_srcs
    op_graph/euclidean_norm_graph_infer.cpp
)

# opapi_srcs: ACLNN (Ascend C Library Neural Network) API sources.
# Uses the op_build-generated stub at ${AUTO_GEN_DIR}/aclnn_euclidean_norm.cpp
#   which implements aclnnEuclideanNormGetWorkspaceSize / aclnnEuclideanNorm via
#   the Nnopbase* framework (no hand-written L0 needed).
set(opapi_srcs
    ${AUTO_GEN_DIR}/aclnn_euclidean_norm.cpp
)

# =============================================================================
# STEP 1: op_build — Auto-generates aic-*-ops-info.ini and stub files
# =============================================================================
file(MAKE_DIRECTORY ${AUTO_GEN_DIR})

# Compiler flags for host-side shared library.
set(HOST_OPS_CXX_FLAGS
    -g -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    -I ${CANN_INC}
    -I ${CANN_ARCH_INC}
    -I ${CANN_OP_COMMON_INC}
    -I ${CANN_OP_COMMON_HOST_INC}
    -I ${CANN_ARCH_INC}/pkg_inc
    -I ${CANN_HOME}/x86_64-linux/pkg_inc/base
    -D BUILD_SOC_VERSION=Ascend950
)

# Linker flags for host-side shared library.
#   -lops_base: Ops::Base::* utilities (SetUnknownRank, IsUnknownRank, etc.)
#     required by reduce-class operators.  Not needed by add_custom_package.
set(HOST_OPS_LINK_FLAGS
    -L ${CANN_LIB}
    -lexe_graph -lregister -ltiling_api -lc_sec -lops_base
    -Wl,-rpath,${CANN_LIB}
)

message(STATUS "Opbuild: compiling libascend_all_ops.so ...")
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} ${HOST_OPS_CXX_FLAGS}
            ${host_tiling_srcs} ${host_graph_srcs}
            ${HOST_OPS_LINK_FLAGS}
            -o ${AUTO_GEN_DIR}/libascend_all_ops.so
    RESULT_VARIABLE _build_ret
    OUTPUT_VARIABLE _build_out
    ERROR_VARIABLE  _build_err
)
if(NOT _build_ret EQUAL 0)
    message(FATAL_ERROR "Failed to compile libascend_all_ops.so: ${_build_err}")
endif()

message(STATUS "Opbuild: running op_build ...")
execute_process(
    COMMAND
        ${CMAKE_COMMAND} -E env
            "OPS_PRODUCT_NAME=${ASCEND_COMPUTE_UNIT}"
            "LD_LIBRARY_PATH=${CANN_LIB}:$ENV{LD_LIBRARY_PATH}"
        ${CANN_HOME}/tools/opbuild/op_build
            ${AUTO_GEN_DIR}/libascend_all_ops.so
            ${AUTO_GEN_DIR}
            "--compute_unit=${ASCEND_COMPUTE_UNIT}"
    RESULT_VARIABLE _opb_ret
    OUTPUT_VARIABLE _opb_out
    ERROR_VARIABLE  _opb_err
)
if(NOT _opb_ret EQUAL 0)
    message(FATAL_ERROR "op_build failed: ${_opb_err}")
endif()
message(STATUS "Opbuild done.")

# =============================================================================
# STEP 2: Overwrite auto-generated op_proto header with hand-written declaration
# (REG_OP lives in the header; the auto-generated op_proto.cc consumes it,
#  so no hand-written proto.cc translation unit is needed)
# =============================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/op_graph/euclidean_norm_proto.h
    ${AUTO_GEN_DIR}/op_proto.h
    COPYONLY
)

# =============================================================================
# STEP 3: Set global properties for npu_op_library macros
# =============================================================================
set_property(GLOBAL PROPERTY ASCENDC_AUTO_GEN_PATH ${AUTO_GEN_DIR})
set_property(GLOBAL PROPERTY _ASC_PKG_${package_name}_CODE_GEN_DIRS ${AUTO_GEN_DIR})

# =============================================================================
# STEP 4: Object libraries for separate compilation
# =============================================================================

# cust_optiling_obj: Tiling object library.
# Contains host tiling sources.
add_library(cust_optiling_obj OBJECT ${host_tiling_srcs})
target_include_directories(cust_optiling_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_OP_COMMON_INC}
    ${CANN_OP_COMMON_HOST_INC}
    ${CANN_ARCH_INC}/pkg_inc
    ${CANN_HOME}/x86_64-linux/pkg_inc/base
)
target_compile_options(cust_optiling_obj PRIVATE -fPIC)
target_compile_definitions(cust_optiling_obj PRIVATE
    BUILD_SOC_VERSION=Ascend950
    OP_TILING_LIB
    _GLIBCXX_USE_CXX11_ABI=0
)

# cust_opapi_obj: Operator API object library.
add_library(cust_opapi_obj OBJECT ${opapi_srcs})
target_include_directories(cust_opapi_obj PRIVATE
    ${AUTO_GEN_DIR}
    ${CANN_HOME}/include/aclnn
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_ARCH_INC}/aclnn
    ${CANN_ARCH_INC}/aclnnop
    ${CANN_HOME}/x86_64-linux/pkg_inc/aicpu
)
target_compile_options(cust_opapi_obj PRIVATE -fPIC)
target_compile_definitions(cust_opapi_obj PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

# cust_proto_obj: Graph proto object library.
# Compiles host_tiling_srcs + host_graph_srcs (euclidean_norm_graph_infer.cpp's
# InferDataType registration lives here via host_graph_srcs).
set(proto_srcs ${host_tiling_srcs} ${host_graph_srcs})
add_library(cust_proto_obj OBJECT ${proto_srcs})
target_include_directories(cust_proto_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_OP_COMMON_INC}
    ${CANN_OP_COMMON_HOST_INC}
    ${CANN_ARCH_INC}/pkg_inc
    ${CANN_HOME}/x86_64-linux/pkg_inc/base
)
target_compile_options(cust_proto_obj PRIVATE -fPIC)
target_compile_definitions(cust_proto_obj PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

# =============================================================================
# STEP 5: Vendor packaging via npu_op_package and npu_op_library
# =============================================================================
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

npu_op_library(cust_optiling TILING)
target_sources(cust_optiling PUBLIC $<TARGET_OBJECTS:cust_optiling_obj>)
# ops_base: provides Ops::Base::SetUnknownRank etc. used by infershape code
# (compiled into host_tiling_srcs).  Not needed by add_custom_package.
target_link_directories(cust_optiling PUBLIC ${CANN_LIB})
target_link_libraries(cust_optiling PUBLIC ops_base)

npu_op_library(cust_opapi ACLNN)
target_sources(cust_opapi PUBLIC $<TARGET_OBJECTS:cust_opapi_obj>)
target_include_directories(cust_opapi PRIVATE
    ${CANN_HOME}/include/aclnn
)
target_link_directories(cust_opapi PUBLIC ${CANN_LIB})
target_link_libraries(cust_opapi PUBLIC opapi)

npu_op_library(cust_op_proto GRAPH)
target_sources(cust_op_proto PUBLIC $<TARGET_OBJECTS:cust_proto_obj>)
target_link_directories(cust_op_proto PUBLIC ${CANN_LIB})
target_link_libraries(cust_op_proto PUBLIC ops_base)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)

# =============================================================================
# STEP 6: Kernel compilation
# =============================================================================
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE EuclideanNorm
    KERNEL_DIR op_kernel
    KERNEL_FILE arch35/euclidean_norm.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# Kernel-side include path for shared ascendc common headers (platform_util.h).
# These headers live in the OPP built-in implementation tree and are not in the
# kernel compiler's default system include paths.  Required by reduce-class
# operators that use Ops::Base device utilities.
npu_op_kernel_options(ascendc_kernels ALL OPTIONS
    -I$ENV{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
