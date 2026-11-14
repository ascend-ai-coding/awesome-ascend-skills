if(TARGET cann_samples_shmem)
    return()
endif()

find_package(Git QUIET)

set(SHMEM_PATH "${PROJECT_SOURCE_DIR}/third_party/shmem")
set(SHMEM_GIT_URL "https://gitcode.com/cann/shmem.git" CACHE STRING "SHMEM 仓库地址")
set(SHMEM_GIT_TAG "v1.5.0" CACHE STRING "SHMEM 仓库版本标签")

if(NOT EXISTS "${SHMEM_PATH}/CMakeLists.txt" AND NOT GIT_FOUND)
    message(FATAL_ERROR
        "Git is required to initialize third_party/shmem automatically."
    )
endif()

# shmem 来源：gitcode.com/cann/shmem v1.5.0，由 cmake 首次配置时自动 clone。
# 不用 git submodule（避免根目录 .gitmodules），改为在 custom_target 里 git clone。
# 版本由 SHMEM_GIT_TAG 锁定；工程被 cp -r 到任意目录后首次 cmake 也能自动拉取。
set(SHMEM_PREPARE_COMMANDS)
if(GIT_FOUND AND NOT EXISTS "${SHMEM_PATH}/CMakeLists.txt")
    list(APPEND SHMEM_PREPARE_COMMANDS
        COMMAND ${GIT_EXECUTABLE} clone --branch ${SHMEM_GIT_TAG} --depth 1 ${SHMEM_GIT_URL} "${SHMEM_PATH}"
    )
endif()

add_custom_target(cann_samples_shmem_dependencies
    ${SHMEM_PREPARE_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E env
        ASCEND_HOME_PATH=${ASCEND_DIR}
        ASCEND_TOOLKIT_HOME=${ASCEND_DIR}
        cmake -S "${SHMEM_PATH}" -B "${SHMEM_PATH}/build" -DCMAKE_BUILD_TYPE=RELEASE -DSOC_TYPE=Ascend950 -DCMAKE_C_FLAGS=-w -DCMAKE_CXX_FLAGS=-w
    COMMAND ${CMAKE_COMMAND} --build "${SHMEM_PATH}/build" --target install --parallel
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Initializing and building third_party/shmem"
    VERBATIM
)

add_library(cann_samples_shmem_bootstrap_uid SHARED IMPORTED GLOBAL)
set_target_properties(cann_samples_shmem_bootstrap_uid PROPERTIES
    IMPORTED_LOCATION "${SHMEM_PATH}/install/shmem/lib/aclshmem_bootstrap_uid.so"
)
add_dependencies(cann_samples_shmem_bootstrap_uid cann_samples_shmem_dependencies)

add_library(cann_samples_shmem_bootstrap_config_store SHARED IMPORTED GLOBAL)
set_target_properties(cann_samples_shmem_bootstrap_config_store PROPERTIES
    IMPORTED_LOCATION "${SHMEM_PATH}/install/shmem/lib/aclshmem_bootstrap_config_store.so"
)
add_dependencies(cann_samples_shmem_bootstrap_config_store cann_samples_shmem_dependencies)

add_library(cann_samples_shmem INTERFACE)
add_library(cann_samples::shmem ALIAS cann_samples_shmem)
add_dependencies(cann_samples_shmem cann_samples_shmem_dependencies)

target_include_directories(cann_samples_shmem INTERFACE
    "${SHMEM_PATH}/install/shmem/include"
    "${SHMEM_PATH}/install/shmem/src"
    "${SHMEM_PATH}/install/shmem/src/device"
    "${SHMEM_PATH}/install/shmem/src/host_device"
)

target_link_directories(cann_samples_shmem INTERFACE
    "${SHMEM_PATH}/install/shmem/lib"
)

target_link_libraries(cann_samples_shmem INTERFACE
    shmem
    shmem_utils
    cann_samples_shmem_bootstrap_uid
    cann_samples_shmem_bootstrap_config_store
)

target_link_options(cann_samples_shmem INTERFACE
    "LINKER:-rpath,${SHMEM_PATH}/install/shmem/lib"
)
