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
# xdivy_custom_package/CMakeLists.txt
# =============================================================================
# Role: Top-level build script for the Xdivy vendor operator package.
#        Orchestrates the full operator build pipeline from source compilation
#        through op_build code generation, to final .run package archiving.
#
# Differences from add_custom_package/CMakeLists.txt:
#   - Xdivy is a BROADCAST operator (2 inputs, broadcast-compatible shapes).
#   - The kernel has a TPL parameter RANK (4 or 8) — two binary variants are
#     produced automatically via ASCENDC_TPL_SEL in op_kernel/arch35/xdivy_struct.h.
#   - Tiling uses IMPL_OP_OPTILING + TilingParse (not the simpler .SetTiling())
#     because the tiling function needs platform info (core count + UB size)
#     at compile time.
#   - The kernel entry (xdivy.cpp) uses GET_TILING_DATA_WITH_STRUCT
#     (not REGISTER_TILING_DEFAULT) because XdivyTilingData is a template.
#   - Three dtypes supported: BF16, FLOAT16, FLOAT (vs add_custom's two).
#
# Contents:
#   - CANN SDK path configuration (CANN_HOME, include/lib paths).
#   - Source file grouping for tiling, graph, and API layers.
#   - Step 1: Compile host-side sources into libascend_all_ops.so and run
#     op_build to generate aic-*-ops-info.ini and autogen stub files.
#   - Step 2: Overwrite op_build-generated stubs with hand-written
#     implementations (op_proto.h). The auto-generated ACLNN stub is used
#     as-is (no hand-written op_api layer needed).
#   - Step 3/4: Build CMake OBJECT libraries (cust_optiling_obj, cust_opapi_obj,
#     cust_proto_obj) for the tiling, API, and proto layers.
#   - Step 5: Package all libraries into a vendor .run archive via
#     npu_op_package / npu_op_library / npu_op_package_add.
#   - Step 6: Compile the Ascend C device kernel sources via
#     npu_op_kernel_sources / npu_op_kernel_library / npu_op_kernel_library.
#
# Operator name variants:
#   - PascalCase:   Xdivy
#   - snake_case:   xdivy
#   - UPPER_SNAKE:  XDIVY
#   - camelCase:    xdivy
#
# To create a new operator (e.g. FooBar), replace:
#   Xdivy → FooBar
#   xdivy → foo_bar
#   XDIVY → FOO_BAR
#   xdivy → fooBar
#   package_name from "Xdivy" to "FooBar"
# =============================================================================

# --- Minimum CMake version required for OBJECT libraries and modern targets ---
cmake_minimum_required(VERSION 3.16.0)

# --- Project declaration ---
project(custom_op_prj)

# --- MUST come before any ASC macro/function usage: loads Ascend CMake helpers ---
find_package(ASC REQUIRED)

# --- Target Ascend compute unit (chip architecture identifier) ---
set(ASCEND_COMPUTE_UNIT ascend950)

# --- Vendor package name (PascalCase), used in npu_op_package and install paths ---
set(package_name Xdivy)

# =============================================================================
# CANN SDK paths
# =============================================================================
# CANN_HOME: root of the CANN installation; change this when deploying on a
# different machine or CANN version.  The same path is hardcoded in test
# CMakeLists.txt files — keep them in sync when migrating.
set(CANN_HOME /home/developer/Ascend/cann-9.1.0)

# CANN_INC: base include directory (e.g. register/, exe_graph/, graph/).
set(CANN_INC ${CANN_HOME}/include)

# CANN_ARCH_INC: arch-specific include directory (e.g. aclnn/, acl/, ge/).
set(CANN_ARCH_INC ${CANN_HOME}/x86_64-linux/include)

# CANN_LIB: shared/static libraries directory.
set(CANN_LIB ${CANN_HOME}/lib64)

# AUTO_GEN_DIR: directory where op_build writes generated code.
set(AUTO_GEN_DIR ${CMAKE_BINARY_DIR}/autogen)

# =============================================================================
# Source file grouping
# =============================================================================

# host_tiling_srcs: tiling-related host sources.
#   xdivy_def.cpp          — OP_ADD(Xdivy) operator definition & AICore config.
#   xdivy_infershape.cpp   — IMPL_OP_INFERSHAPE: broadcast output shape inference.
#   xdivy_tiling_arch35.cpp — TilingFuncXdivy: broadcast multi-core tiling.
file(GLOB host_tiling_srcs
    op_host/xdivy_def.cpp
    op_host/xdivy_infershape.cpp
    op_host/arch35/xdivy_tiling_arch35.cpp
)

# host_graph_srcs: Graph-side inference registration (InferDataType) sources.
# Used by the graph engine (GE) at graph build time.  These sources are
# compiled into cust_proto_obj (see STEP 4 proto_srcs).
# NOTE: REG_OP(Xdivy) lives in op_graph/xdivy_proto.h; the auto-generated
#   op_proto.cc consumes it, so no hand-written proto.cc translation unit
#   is needed.
file(GLOB host_graph_srcs
    op_graph/xdivy_graph_infer.cpp
)

# opapi_srcs: ACLNN-level operator API sources.
# Uses the op_build-generated stub at ${AUTO_GEN_DIR}/aclnn_xdivy.cpp
#   which implements aclnnXdivyGetWorkspaceSize / aclnnXdivy via
#   the Nnopbase* framework (no hand-written L0 needed).
set(opapi_srcs
    ${AUTO_GEN_DIR}/aclnn_xdivy.cpp
)

# =============================================================================
# Step 1: op_build — generate aic-*-ops-info.ini and autogen stubs
# =============================================================================
# The CANN op_build tool takes a .so containing OP_ADD / IMPL_OP / REG_OP
# registrations and emits:
#   aic-<package>-ops-info.ini   — operator metadata (inputs/outputs, attrs).
#   op_proto.h / op_proto.cc     — auto-generated proto stubs.
#   aclnn_xdivy.h/.cpp           — auto-generated ACLNN API stubs (used as-is).
# The op_proto.h stub is then *overwritten* in Step 2 with hand-written code.
file(MAKE_DIRECTORY ${AUTO_GEN_DIR})

# Compiler flags for the host-side .so that op_build reads.
set(HOST_OPS_CXX_FLAGS
    -g -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    -I ${CANN_INC}
    -I ${CANN_ARCH_INC}
    -I ${CANN_ARCH_INC}/pkg_inc
    -D BUILD_SOC_VERSION=Ascend950
)

# Linker flags for the host-side .so.
set(HOST_OPS_LINK_FLAGS
    -L ${CANN_LIB}
    -lexe_graph -lregister -ltiling_api -lc_sec
    -Wl,-rpath,${CANN_LIB}
)

# Compile libascend_all_ops.so: aggregates all registration objects into one
# shared library that op_build can introspect.
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

# Run op_build: it introspects libascend_all_ops.so and writes generated files
# into AUTO_GEN_DIR.  OPS_PRODUCT_NAME sets the chip name embedded in .ini.
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
# Step 2: Overwrite autogen stubs with hand-written implementations
# =============================================================================

# Overwrite op_proto.h: REG_OP(Xdivy) with INPUT/OUTPUT declarations.
# (REG_OP lives in the header; the auto-generated op_proto.cc consumes it,
#  so no hand-written proto.cc translation unit is needed)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/op_graph/xdivy_proto.h
    ${AUTO_GEN_DIR}/op_proto.h
    COPYONLY
)

# =============================================================================
# Step 3: Global properties for npu_op_library
# =============================================================================

set_property(GLOBAL PROPERTY ASCENDC_AUTO_GEN_PATH ${AUTO_GEN_DIR})
set_property(GLOBAL PROPERTY _ASC_PKG_${package_name}_CODE_GEN_DIRS ${AUTO_GEN_DIR})

# =============================================================================
# Step 4: Object libraries
# =============================================================================

# --- cust_optiling_obj: tiling-layer sources ---
add_library(cust_optiling_obj OBJECT ${host_tiling_srcs})
target_include_directories(cust_optiling_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_ARCH_INC}/pkg_inc
)
target_compile_options(cust_optiling_obj PRIVATE -fPIC)
target_compile_definitions(cust_optiling_obj PRIVATE
    BUILD_SOC_VERSION=Ascend950
    OP_TILING_LIB
    _GLIBCXX_USE_CXX11_ABI=0
)

# --- cust_opapi_obj: ACLNN API-layer sources (op_build autogen stub) ---
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

# --- cust_proto_obj: proto registration sources (tiling + graph infer) ---
# Compiles host_tiling_srcs + host_graph_srcs (xdivy_graph_infer.cpp's
# InferDataType registration lives here via host_graph_srcs).
set(proto_srcs ${host_tiling_srcs} ${host_graph_srcs})
add_library(cust_proto_obj OBJECT ${proto_srcs})
target_include_directories(cust_proto_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CANN_INC}
    ${CANN_ARCH_INC}
    ${CANN_ARCH_INC}/pkg_inc
)
target_compile_options(cust_proto_obj PRIVATE -fPIC)
target_compile_definitions(cust_proto_obj PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

# =============================================================================
# Step 5: Vendor packaging (npu_op_package + npu_op_library + npu_op_package_add)
# =============================================================================

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# --- cust_optiling: tiling shared library ---
npu_op_library(cust_optiling TILING)
target_sources(cust_optiling PUBLIC $<TARGET_OBJECTS:cust_optiling_obj>)

# --- cust_opapi: ACLNN API shared library ---
npu_op_library(cust_opapi ACLNN)
target_sources(cust_opapi PUBLIC $<TARGET_OBJECTS:cust_opapi_obj>)
target_include_directories(cust_opapi PRIVATE
    ${CANN_HOME}/include/aclnn
)
target_link_directories(cust_opapi PUBLIC ${CANN_LIB})
target_link_libraries(cust_opapi PUBLIC opapi)

# --- cust_op_proto: graph proto shared library ---
npu_op_library(cust_op_proto GRAPH)
target_sources(cust_op_proto PUBLIC $<TARGET_OBJECTS:cust_proto_obj>)

# npu_op_package_add: registers the three libraries inside the .run package.
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)

# =============================================================================
# Step 6: Kernel — Ascend C device-side kernel compilation
# =============================================================================

# npu_op_kernel_sources: declares the device kernel source files.
#   OP_TYPE Xdivy: binds these kernel sources to the Xdivy operator.
#   KERNEL_DIR op_kernel: directory containing xdivy.cpp.
#   KERNEL_FILE xdivy.cpp: the kernel implementation entry point.
#
# Note: the kernel template parameter RANK (4 or 8) is handled automatically
# by the ASCENDC_TPL_SEL declaration in op_kernel/arch35/xdivy_struct.h — the
# build system emits a separate kernel binary for each selection, and the
# runtime dispatches to the right one based on SetTilingKey.
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Xdivy
    KERNEL_DIR op_kernel
    KERNEL_FILE xdivy.cpp
)

# npu_op_kernel_library: assembles compiled kernel objects into a kernel .so.
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# Add the kernel library to the vendor package.
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
