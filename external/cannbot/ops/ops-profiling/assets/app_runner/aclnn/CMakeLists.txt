cmake_minimum_required(VERSION 3.15)
project(aclnn_app_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CANN_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
    else()
        message(FATAL_ERROR "Pass -DCANN_HOME=<cann install> or source CANN setenv first")
    endif()
endif()

# Rename the adapter source to your target_adapter_<op>.cpp when adapting.
add_executable(aclnn_app_runner
    csrc/aclnn_app_runner.cpp
    csrc/target_adapter_add.cpp
)

target_include_directories(aclnn_app_runner PRIVATE
    "${CANN_HOME}/include"
)

target_link_libraries(aclnn_app_runner PRIVATE
    "${CANN_HOME}/lib64/libascendcl.so"
    "${CANN_HOME}/lib64/libnnopbase.so"
    "${CANN_HOME}/lib64/libopapi.so"
)

# Optional: vendor package providing the custom aclnn op.  The shipped Add
# example is a CANN built-in and needs no vendor; pass -DVENDOR_DIR when
# adapting to a custom op (build.sh --vendor-dir=...).
if(DEFINED VENDOR_DIR)
    if(NOT EXISTS "${VENDOR_DIR}/op_api/lib/libcust_opapi.so")
        message(FATAL_ERROR "VENDOR_DIR must contain op_api/lib/libcust_opapi.so: ${VENDOR_DIR}")
    endif()
    target_include_directories(aclnn_app_runner PRIVATE
        "${VENDOR_DIR}/op_api/include"
        "${VENDOR_DIR}/op_api/include/aclnnop"
    )
    target_link_libraries(aclnn_app_runner PRIVATE
        "${VENDOR_DIR}/op_api/lib/libcust_opapi.so"
    )
    set_target_properties(aclnn_app_runner PROPERTIES
        BUILD_RPATH "${CANN_HOME}/lib64;${VENDOR_DIR}/op_api/lib"
        INSTALL_RPATH "${CANN_HOME}/lib64;${VENDOR_DIR}/op_api/lib"
    )
else()
    set_target_properties(aclnn_app_runner PROPERTIES
        BUILD_RPATH "${CANN_HOME}/lib64"
        INSTALL_RPATH "${CANN_HOME}/lib64"
    )
endif()
