#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# xdivy_custom_package/tests/tiling/run.sh
# =============================================================================
# Role: One-click build-and-run script for the Xdivy tiling unit test.
#        This test validates the TilingFunc logic directly (including broadcast
#        scenarios) without requiring device memory or NPU kernel execution.
#
# Workflow:
#   1. Source ${ASCEND_HOME_PATH}/set_env.sh to set up CANN environment.
#   2. Build the tiling unit test from this directory using CMake:
#        - xdivy_op_host_ut_lib.so (operator registration .so loaded at runtime)
#        - test_tiling (the test executable)
#   3. Set LD_LIBRARY_PATH to include CANN lib64.
#   4. Run ./test_tiling.
#
# Design notes:
#   - All paths are resolved relative to SCRIPT_DIR.
#   - 'set -e' makes the script fail-fast on any error.
#   - This test does NOT require build.sh to be run first (unlike aclnn/geir
#     tests) because it compiles sources directly and loads the .so from the
#     build directory, not from an installed vendor package.
#
# Operator name variants:
#   - PascalCase:   Xdivy
#   - snake_case:   xdivy
#   - UPPER_SNAKE:  XDIVY
#   - camelCase:    xdivy
# =============================================================================
set -e

# SCRIPT_DIR: directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the CANN environment if available.
SET_ENV="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.1.0}/set_env.sh"
if [ -f "${SET_ENV}" ]; then
    # shellcheck disable=SC1090
    . "${SET_ENV}"
fi

# --- Step 1: Build the tiling UT (op_host .so + test executable) ---
cd "${SCRIPT_DIR}"
mkdir -p build
cd build
cmake ..
make

# --- Step 2: Run the test ---
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"
./test_tiling
