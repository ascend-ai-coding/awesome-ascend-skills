# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CMake build for the matmul_kernel direct-invoke sample.
#
# [MODIFY] 创建新算子时修改：
#   1. project(<your_op>_custom ...) 与 add_executable 的目标名、源文件名
#   2. target_compile_options 中的 --npu-arch=dav-3510（按实际 NPU 型号调整）
#   3. 如果添加新的 .cpp，在同一 target 下 target_sources(...) 追加
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# ---- Ascend toolkit discovery ----------------------------------------------
if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(COMMAND uname -m
            OUTPUT_VARIABLE _ascend_processor
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    if("$ENV{USER}" STREQUAL "root")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Set ASCEND_HOME_PATH.")
        endif()
    else()
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()
message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")

set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng"
    CACHE FILEPATH "Path to Bisheng compiler")

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${BISHENG}" CACHE FILEPATH "C compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "CXX compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_LINKER)
    set(CMAKE_LINKER "${BISHENG}" CACHE FILEPATH "Linker" FORCE)
endif()

# [MODIFY] 工程名：matmul_custom → <your_op>_custom
project(matmul_custom LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/asc/include
    ${ASCEND_DIR}/compiler/asc/include
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)

add_library(matmul_ascend_base INTERFACE)
target_include_directories(matmul_ascend_base INTERFACE ${ASCEND_INCLUDE_DIRS})
target_link_directories(matmul_ascend_base INTERFACE ${ASCEND_DIR}/lib64)
target_link_libraries(matmul_ascend_base INTERFACE m dl platform tiling_api)

add_library(matmul_tensor_api INTERFACE)
target_include_directories(matmul_tensor_api INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensor_api)

# [CONFIG] Ascend C matmul 编译选项
#  -xasc             : Bisheng ASC 模式
#  --npu-arch=XXX    : dav-2201 (Atlas A2) / dav-3510 (Atlas A5)；见 npu-arch
#  -O3 / -w          : 优化等级 + 抑制 tensor_api 内部 warning
set(MATMUL_COMPILE_OPTS "-xasc" "--npu-arch=dav-3510" "-w" "-O3")

# [MODIFY] 目标名 matmul_custom → <your_op>_custom，源文件 matmul_custom.cpp → <your_op>.cpp
add_executable(matmul_custom matmul_custom.cpp)
target_include_directories(matmul_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(matmul_custom PRIVATE ${MATMUL_COMPILE_OPTS})
target_link_libraries(matmul_custom PRIVATE
    matmul_ascend_base
    matmul_tensor_api
)
