cmake_minimum_required(VERSION 3.16)
project(moe_dispatch_direct_invoke)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

set(SOC_VERSION "ascend950pr_957c" CACHE STRING "system on chip type")

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH not set, please configure CANN environment first")
endif()

set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    message(FATAL_ERROR "ascendc_kernel_cmake not found, check CANN at: ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ascendc_library(ascendc_kernels SHARED ${CMAKE_CURRENT_SOURCE_DIR}/moe_dispatch.cpp)
ascendc_include_directories(ascendc_kernels PRIVATE ${INCLUDE_DIRS})

add_executable(test_moe_dispatch ${CMAKE_CURRENT_SOURCE_DIR}/test/test_moe_dispatch.cpp)
target_compile_options(test_moe_dispatch PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(test_moe_dispatch PRIVATE SOC_VERSION="${SOC_VERSION}")
target_include_directories(test_moe_dispatch PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${INCLUDE_DIRS}
)
target_link_libraries(test_moe_dispatch PRIVATE
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hccl_fwk
    hcomm
)