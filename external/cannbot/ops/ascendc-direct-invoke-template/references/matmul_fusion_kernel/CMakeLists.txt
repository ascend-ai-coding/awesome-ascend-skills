# ============================================================
# CMakeLists.txt — Matmul 融合算子构建配置
# [SCOPE]   本 skill 仅用于 mxfp8 + eltwise 融合算子，不再支持单算子 target。
# [MODIFY 1] project() — 修改工程名称
# [MODIFY 2] add_executable() — 修改 target 名称和源文件
# [CONFIG 1] --npu-arch — 编译目标架构（dav-3510 为 bisheng 编译架构）
# ============================================================

cmake_minimum_required(VERSION 3.16)

# 平台检测：自动适配 x86_64/aarch64 平台（参考 cann-samples/cmake/ascend.cmake）
if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _ascend_processor OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

# 设置 bisheng 编译器路径
# [CONFIG] ASCEND_HOME_PATH 环境变量自动检测，或手动指定
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_DIR "/home/developer/Ascend/cann-9.0.0")
endif()

# 使用平台前缀设置编译器路径（支持 x86_64-linux 和 aarch64-linux）
set(BISHENG_COMPILER "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng")
set(CMAKE_CXX_COMPILER "${BISHENG_COMPILER}" CACHE PATH "Ascend C compiler" FORCE)

find_package(ASC REQUIRED)

# [MODIFY] 工程名称
project(matmul_fusion_kernel LANGUAGES CXX ASC)

# [CONFIG] 编译选项
# --npu-arch=dav-3510 是 bisheng 编译器的编译目标架构，与运行时 NPU 无关
# 运行时目标通过 cannsim -s Ascend950 或实际 NPU 设备指定
set(ASCENDC_COMPILE_OPTS "-xasc" "--npu-arch=dav-3510" "-w" "-O3")

# 通用包含目录
# 1. 工程自定义目录
# 2. CANN 系统目录（参考 cann-samples/cmake/ascend.cmake 中的 ASCEND_INCLUDE_DIRS）
set(COMMON_INCLUDES
    # 工程目录
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensor_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensor_api
    # CANN 头文件路径（使用 SYSTEM_PREFIX 适配 x86_64/aarch64）
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/asc/include
    ${ASCEND_DIR}/asc/impl                           # 解决 kernel_macros.h 找不到问题
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)

# 通用链接库
set(COMMON_LIBS m dl platform tiling_api)

# ---- 融合算子 target ----
# [USER] 修改 target 名称和源文件
add_executable(matmul_fused_swat src/matmul_fused_swat.cpp)
target_include_directories(matmul_fused_swat PRIVATE ${COMMON_INCLUDES})
target_compile_options(matmul_fused_swat PRIVATE ${ASCENDC_COMPILE_OPTS})
target_link_libraries(matmul_fused_swat PRIVATE ${COMMON_LIBS})
