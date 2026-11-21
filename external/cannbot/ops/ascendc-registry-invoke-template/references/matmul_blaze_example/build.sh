#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
set -e
SUPPORT_COMPUTE_UNITS=("ascend950")
export BASE_PATH=$(cd "$(dirname $0)"; pwd)
export BUILD_PATH="${BASE_PATH}/build"
CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if [ ${CORE_NUMS} -gt 8 ]; then CORE_NUMS=8; fi
usage(){ echo "Usage: bash build.sh --soc=ascend950 [-j8]"; echo "  --soc=soc_version   ascend950"; echo "  -j[n]               Compile threads"; echo "  --make_clean        Clean build"; }
COMPUTE_UNIT=""; THREAD_NUM=${CORE_NUMS}; ENABLE_CLEAN=FALSE
while [[ $# -gt 0 ]]; do case "$1" in --soc=*) COMPUTE_UNIT="${1#*=}"; shift ;; -j*) THREAD_NUM="${1:2}"; [ -z "$THREAD_NUM" ] && THREAD_NUM=${CORE_NUMS}; shift ;; --make_clean) ENABLE_CLEAN=TRUE; shift ;; *) usage; exit 1 ;; esac; done
if [ "$ENABLE_CLEAN" = "TRUE" ]; then rm -rf ${BUILD_PATH}/*; exit 0; fi
if [ -z "$COMPUTE_UNIT" ]; then echo "[ERROR] --soc parameter is required"; usage; exit 1; fi
COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | tr '[:upper:]' '[:lower:]')
mkdir -p "${BUILD_PATH}"
[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt
echo "[INFO] Configuring..."
cd "${BUILD_PATH}" && cmake -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT ..
echo "[INFO] Building with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j ${THREAD_NUM}
echo "[INFO] Build completed!"
