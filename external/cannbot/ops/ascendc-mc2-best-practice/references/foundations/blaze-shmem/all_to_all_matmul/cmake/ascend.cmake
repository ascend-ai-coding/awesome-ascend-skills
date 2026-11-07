
# find ascend toolkit
if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE _ascend_processor
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    message(WARNING "Environment variable ASCEND_HOME_PATH is not set. Using default path.")
    if("$ENV{USER}" STREQUAL "root")
        message(STATUS "Running as root user, checking default root paths for Ascend toolkit.")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found in default root paths. Please set ASCEND_HOME_PATH.")
        endif()
    else()
        message(STATUS "Running as non-root user, checking default user paths for Ascend toolkit.")
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found in default user paths. Please set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

# Configure the default toolchain before project() initializes the languages.
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${BISHENG}" CACHE FILEPATH "C compiler" FORCE)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "CXX compiler" FORCE)
endif()

if(NOT DEFINED CMAKE_LINKER)
    set(CMAKE_LINKER "${BISHENG}" CACHE FILEPATH "Linker" FORCE)
endif()
