# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Verify proto header exists
# =============================================================================
file(GLOB proto_headers "${CMAKE_CURRENT_SOURCE_DIR}/*_proto.h")
if(NOT proto_headers)
    message(FATAL_ERROR "op_graph/ directory must contain at least one *_proto.h file with REG_OP definition")
endif()

# =============================================================================
# Add proto headers to GRAPH library include path
# =============================================================================
if(TARGET cust_op_proto)
    target_include_directories(cust_op_proto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
