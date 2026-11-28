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
# euclidean_norm_package/tests/geir/run.sh
# =============================================================================
# Role: One-click build-and-run script for the EuclideanNorm GE (Graph Engine)
#        integration test.
#
# Workflow:
#   1. Locate the build.sh script in the operators/ directory (3 levels up).
#   2. Run build.sh <package_name> to compile and install the vendor operator
#      package.  This installs the REG_OP(EuclideanNorm) proto header and the
#      ACLNN library that GE needs at runtime.
#   3. Build the GE test from this directory using CMake.
#   4. Set LD_LIBRARY_PATH to include the installed vendor library.
#   5. Run ./test_geir.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PKG_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PKG_NAME="$(basename "${PKG_DIR}")"

BUILD_SH="${SCRIPT_DIR}/../../../build.sh"

# --- Step 1: Build + install the custom operator package ---
"${BUILD_SH}" "${PKG_NAME}"

# --- Step 2: Build the GE test in this script's own build/ directory ---
cd "${SCRIPT_DIR}"
mkdir -p build
cd build
cmake ..
make

# --- Step 3: Run the test against the freshly installed operator ---
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/EuclideanNorm/op_api/lib/:${LD_LIBRARY_PATH}
./test_geir
