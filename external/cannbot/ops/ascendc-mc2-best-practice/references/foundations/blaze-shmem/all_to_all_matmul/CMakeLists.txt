
# ============================================================================
# MC2 通算融合算子构建脚本（all_to_all_matmul）
# ----------------------------------------------------------------------------
# 该 CMakeLists 是 [REUSE] 标记文件——常规 MC2 算子不需要改这里的内容。
# 仅在新增可执行目标时改两处：project() 名 + add_executable() 目标名 + 源文件。
#
# 关键设计：
#   1. NPU_ARCH=dav-3510 是硬约束（仅支持 Ascend 950）；dav-2201 仅做校验兼容，会发 WARNING；
#   2. cmake/ascend.cmake 必须在 project() 之前 include（设 BISHENG 编译器、ASCEND_HOME_PATH）；
#   3. shmem.cmake 和 tensor_api.cmake 用 REQUIRED（不是 OPTIONAL）——
#      OPTIONAL 在路径错时会静默跳过，导致 cann_samples::shmem / cann_samples::tensor_api 未定义，链接时报"target not found"；
#   4. tensor_api 来自 asc-devkit（gitcode.com/cann/asc-devkit），与 ascendc-blaze-best-practice skill 共享同一来源；
#   5. -xasc --npu-arch=dav-3510 是 bisheng 双路径编译的必需参数（host + kernel 同时编）。
#
# 进阶细节：references/codebase_map.md §4
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# NPU_ARCH 入参校验（与源工程父 CMakeLists 一致）
set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    message(FATAL_ERROR "NPU_ARCH is required. Example: cmake -S . -B build -DNPU_ARCH=dav-3510")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: dav-3510, dav-2201")
endif()
if(NOT NPU_ARCH STREQUAL "dav-3510")
    message(WARNING "Unsupported NPU_ARCH=${NPU_ARCH}, expected dav-3510")
endif()

# 关键：cmake/ascend.cmake 必须在 project() 之前 include（设置 BISHENG 编译器、ASCEND_HOME_PATH）
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake REQUIRED)

find_package(ASC)
project(all_to_all_matmul LANGUAGES C CXX ASC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# third_party: shmem + tensor_api（首次 cmake 配置时由各自 .cmake 自动 git clone，无需 submodule）
# 关键：用 REQUIRED 而非 OPTIONAL —— OPTIONAL 会在路径错误时静默跳过，
#       导致 cann_samples::shmem / cann_samples::tensor_api target 未定义，
#       后续 target_link_libraries 报 "target not found"，定位困难
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/shmem.cmake REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensor_api.cmake REQUIRED)

# Blaze 头文件路径：新 kernel (qbmm_mx_kernel.h) 依赖 blaze/gemm/block/block_mmad_qbmm_mx.h，
# 该头文件位于 CANN toolkit 的 ascendc/common 目录下（ops_nn 与 ops_transformer 内容一致）。
# 同时补齐 ASC toolkit 的 x86_64-linux 实际路径。
# tensor_api 头文件由 cmake/tensor_api.cmake 通过 cann_samples::tensor_api 提供（自动 clone asc-devkit），
# 不从 _BLAZE_COMMON_DIR/tensor_api 取（避免 toolkit 版与 asc-devkit 版混用）。
# 注意 ASCEND_HOME_PATH 是环境变量（由 ascend.cmake 设给 ASCEND_DIR CMake 变量），这里用 $ENV{} 兜底。
if(NOT DEFINED ASCEND_DIR)
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
endif()
set(_ASC_ARCH_DIR "${ASCEND_DIR}/x86_64-linux")
set(_BLAZE_COMMON_DIR "${ASCEND_DIR}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common")

add_executable(all_to_all_matmul src/all_to_all_matmul.cpp)
target_compile_definitions(all_to_all_matmul PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(all_to_all_matmul PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernel"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tiling"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/block"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tile"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/policy"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${_BLAZE_COMMON_DIR}"
    "${_ASC_ARCH_DIR}/asc/include"
    "${_ASC_ARCH_DIR}/asc"
)
# 新增 .cpp 必须保留 -xasc --npu-arch=dav-3510（bisheng 把 .cpp 同时走 host 编译 + kernel 编译双路径）
target_compile_options(all_to_all_matmul PRIVATE
    "-xasc"
    "--npu-arch=dav-3510"
    "-w"
    "-O3"
)
target_link_libraries(all_to_all_matmul
    PRIVATE
    cann_samples::tensor_api
    cann_samples::shmem
    dl
    platform
    tiling_api
    ascendcl
    runtime
    stdc++
)
