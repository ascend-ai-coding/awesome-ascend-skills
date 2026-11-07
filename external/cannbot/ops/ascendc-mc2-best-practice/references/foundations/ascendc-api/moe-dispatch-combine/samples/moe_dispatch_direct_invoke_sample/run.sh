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

RANK_SIZE="${RANK_SIZE:-2}"
BS="${BS:-4}"
H="${H:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/single_case}"
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
    bash build.sh
else
    echo "SKIP_BUILD=1 or --skip-build given, skipping build step"
fi
export LD_LIBRARY_PATH=${BUILD_DIR}/lib:$LD_LIBRARY_PATH

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

${BUILD_DIR}/test_moe_dispatch \
    --rank_size ${RANK_SIZE} \
    --bs ${BS} \
    --h ${H} \
    --output_dir "${OUTPUT_DIR}"

python3 ${SCRIPT_DIR}/scripts/verify_dispatch.py \
    --data_dir "${OUTPUT_DIR}" \
    --rank_size ${RANK_SIZE} \
    --bs ${BS} \
    --h ${H}