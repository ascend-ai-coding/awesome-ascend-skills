# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH 未设置，请先配置 CANN 环境" >&2; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || { echo "ERROR: set_env.sh 执行失败" >&2; exit 1; }

RANK_SIZE=${RANK_SIZE:-2}
TEST_CASES=("4 16" "8 16" "8 32")
DATA_BASE_DIR="${SCRIPT_DIR}/outputs"
BUILD_DIR="${SCRIPT_DIR}/build"
# Parse arguments: support --skip-build (also honor SKIP_BUILD env var)
SKIP_BUILD=${SKIP_BUILD:-0}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ "${SKIP_BUILD:-0}" -ne 1 ]; then
    echo "=== Building ==="
    bash build.sh
else
    echo "SKIP_BUILD=1 or --skip-build given, skipping build step"
fi
export LD_LIBRARY_PATH=${BUILD_DIR}/lib:$LD_LIBRARY_PATH

PASSED=0
FAILED=0

for test_case in "${TEST_CASES[@]}"; do
    BS=$(echo "$test_case" | awk '{print $1}')
    H=$(echo "$test_case" | awk '{print $2}')
    OUT_DIR="${DATA_BASE_DIR}/bs_${BS}_h_${H}"

    echo "Testing bs=${BS}, h=${H}, rank_size=${RANK_SIZE}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    if ! ${BUILD_DIR}/test_moe_dispatch \
        --rank_size ${RANK_SIZE} \
        --bs ${BS} \
        --h ${H} \
        --output_dir "${OUT_DIR}"; then
        echo "  FAIL: rank process failed"
        FAILED=$((FAILED + 1))
        continue
    fi

    if python3 ${SCRIPT_DIR}/scripts/verify_dispatch.py --data_dir "${OUT_DIR}" --rank_size ${RANK_SIZE} --bs ${BS} --h ${H}; then
        echo "  PASS"
        PASSED=$((PASSED + 1))
    else
        echo "  FAIL"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Summary: Passed=${PASSED}, Failed=${FAILED}"
[ $FAILED -eq 0 ] && echo "ALL PASSED" && exit 0 || echo "SOME FAILED" && exit 1