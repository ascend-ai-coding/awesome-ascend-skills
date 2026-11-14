
if(TARGET cann_samples_tensor_api)
    return()
endif()

find_package(Git QUIET)

set(TENSOR_API_PATH "${PROJECT_SOURCE_DIR}/third_party/tensor_api")
set(TENSOR_API_GIT_URL "https://gitcode.com/cann/asc-devkit.git" CACHE STRING "tensor_api 仓库地址")
set(TENSOR_API_GIT_TAG "ad3d3bf04dddfb94370c534c39bdd305d8e38d88" CACHE STRING "tensor_api 仓库 commit (feature/tensor_api_from_9.0.0)")

if(NOT EXISTS "${TENSOR_API_PATH}/include/tensor_api/tensor.h" AND NOT GIT_FOUND)
    message(FATAL_ERROR
        "Git is required to initialize third_party/tensor_api automatically."
    )
endif()

# tensor_api 来源：gitcode.com/cann/asc-devkit（feature/tensor_api_from_9.0.0 分支，commit ad3d3bf），
# 与 ascendc-blaze-best-practice skill 共享同一来源。
# 不用 git submodule（避免根目录 .gitmodules），改为在 custom_target 里 git clone + checkout。
# 关键：include path 设为 clone 根目录，使代码中 #include "include/tensor_api/tensor.h"
#       解析到 ${TENSOR_API_PATH}/include/tensor_api/tensor.h；
#       impl/ 头文件也由该路径解析（${TENSOR_API_PATH}/impl/tensor_api/...）。
# 注意：tag 用 commit hash 而非分支名，因为 asc-devkit 的 tensor_api 在 feature 分支上，
#       无独立 tag；--depth 1 + 指定 commit 需要 fetch 该 commit，故不加 --depth。
set(TENSOR_API_PREPARE_COMMANDS)
if(GIT_FOUND AND NOT EXISTS "${TENSOR_API_PATH}/include/tensor_api/tensor.h")
    list(APPEND TENSOR_API_PREPARE_COMMANDS
        COMMAND ${GIT_EXECUTABLE} clone ${TENSOR_API_GIT_URL} "${TENSOR_API_PATH}"
        COMMAND ${GIT_EXECUTABLE} -C "${TENSOR_API_PATH}" checkout ${TENSOR_API_GIT_TAG}
    )
endif()

add_custom_target(cann_samples_tensor_api_dependencies
    ${TENSOR_API_PREPARE_COMMANDS}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Initializing third_party/tensor_api"
    VERBATIM
)

add_library(cann_samples_tensor_api INTERFACE)
add_library(cann_samples::tensor_api ALIAS cann_samples_tensor_api)
add_dependencies(cann_samples_tensor_api cann_samples_tensor_api_dependencies)

target_include_directories(cann_samples_tensor_api INTERFACE
    "${TENSOR_API_PATH}"
    "${TENSOR_API_PATH}/include"
)
