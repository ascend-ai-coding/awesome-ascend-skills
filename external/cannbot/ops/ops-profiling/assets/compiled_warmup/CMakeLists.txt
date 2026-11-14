# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(ops_profiling_warmup LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

if(NOT TORCH_NPU_PATH)
    message(FATAL_ERROR "TORCH_NPU_PATH is required")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR "$ENV{ASCEND_HOME_PATH}")
elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
    set(ASCEND_DIR "/usr/local/Ascend/ascend-toolkit/latest")
elseif(EXISTS "$ENV{HOME}/Ascend/ascend-toolkit/latest")
    set(ASCEND_DIR "$ENV{HOME}/Ascend/ascend-toolkit/latest")
else()
    message(FATAL_ERROR "Ascend toolkit not found; set ASCEND_HOME_PATH")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(SYSTEM_PREFIX "aarch64-linux")
else()
    set(SYSTEM_PREFIX "x86_64-linux")
endif()

set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng")
if(NOT EXISTS "${BISHENG}")
    message(FATAL_ERROR "Bisheng compiler not found: ${BISHENG}")
endif()

if(NPU_ARCH STREQUAL "ascend910b" OR NPU_ARCH STREQUAL "ascend910_93")
    set(BISHENG_NPU_ARCH "dav-2201")
elseif(NPU_ARCH STREQUAL "ascend950")
    set(BISHENG_NPU_ARCH "dav-3510")
else()
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}")
endif()

find_library(ASCENDC_RUNTIME_LIBRARY
    NAMES ascendc_runtime
    PATHS "${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64"
    NO_DEFAULT_PATH
    REQUIRED
)

set(ASCEND_INCLUDE_DIRS
    "${ASCEND_DIR}/include"
    "${ASCEND_DIR}/compiler/tikcpp/include"
    "${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl"
    "${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface"
    "${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl"
    "${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling"
    "${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api"
)

set(KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_kernel/compiled_warmup_kernel.cpp")
set(KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/compiled_warmup_kernel.o")
set(KERNEL_INCLUDE_ARGS "")
foreach(INCLUDE_DIR IN LISTS ASCEND_INCLUDE_DIRS)
    list(APPEND KERNEL_INCLUDE_ARGS "-I${INCLUDE_DIR}")
endforeach()

add_custom_command(
    OUTPUT "${KERNEL_OBJECT}"
    COMMAND "${BISHENG}" --npu-arch=${BISHENG_NPU_ARCH} -xasc -O3 -fPIC
            ${KERNEL_INCLUDE_ARGS} -c "${KERNEL_SOURCE}" -o "${KERNEL_OBJECT}"
    DEPENDS "${KERNEL_SOURCE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_kernel/compiled_warmup_launch.h"
    VERBATIM
)
set_source_files_properties("${KERNEL_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_library(_C SHARED
    csrc/extension.cpp
    csrc/op_plugin/compiled_warmup_plugin.cpp
    "${KERNEL_OBJECT}"
)
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "_C")
target_compile_options(_C PRIVATE -O3 ${TORCH_CXX_FLAGS_LIST})
target_include_directories(_C SYSTEM PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include"
    ${ASCEND_INCLUDE_DIRS}
)
target_link_directories(_C PRIVATE "${TORCH_NPU_PATH}/lib" "${ASCEND_DIR}/lib64")
target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    "${ASCENDC_RUNTIME_LIBRARY}"
)

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:_C>"
            "${CMAKE_CURRENT_SOURCE_DIR}/ops_profiling_warmup/$<TARGET_FILE_NAME:_C>"
    COMMENT "Copying local warmup extension into its Python package"
)
