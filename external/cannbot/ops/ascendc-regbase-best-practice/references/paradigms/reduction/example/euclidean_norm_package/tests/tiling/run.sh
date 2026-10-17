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
# euclidean_norm_package/tests/tiling/run.sh
# =============================================================================
# Role: One-click build-and-run script for the EuclideanNorm tiling unit test.
#        This test validates the TilingFunc logic directly without requiring
#        device memory or NPU kernel execution.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_CANN_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [ -z "${_CANN_HOME}" ]; then
    echo "ERROR: ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME not set. Run: source <CANN>/set_env.sh" >&2
    exit 1
fi
SET_ENV="${_CANN_HOME}/set_env.sh"
if [ -f "${SET_ENV}" ]; then
    . "${SET_ENV}"
fi

cd "${SCRIPT_DIR}"
mkdir -p build
cd build
cmake ..
make

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH:-${_CANN_HOME}}/lib64:${LD_LIBRARY_PATH}"
./test_tiling
