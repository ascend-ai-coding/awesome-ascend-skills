#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Build the pure-aclnn application runner.
#
# Usage:
#   bash build.sh                                # CANN built-in example (Add)
#   bash build.sh --vendor-dir /abs/path/to/vendors/<name>   # custom aclnn op
#
# Requirements: CANN env sourced (ASCEND_HOME_PATH) or pass --cann=/path;
# the per-op adapter lives in csrc/target_adapter_<op>.cpp.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cann_home="${ASCEND_HOME_PATH:-}"
vendor_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vendor-dir) vendor_dir="${2:?--vendor-dir needs a value}"; shift 2 ;;
        --vendor-dir=*) vendor_dir="${1#*=}"; shift ;;
        --cann) cann_home="${2:?--cann needs a value}"; shift 2 ;;
        --cann=*) cann_home="${1#*=}"; shift ;;
        *) echo "[ERROR] unsupported argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$cann_home" ]]; then
    echo "[ERROR] source CANN setenv first or pass --cann=<install>" >&2
    exit 2
fi

build_dir="$script_dir/build"
cmake -S "$script_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCANN_HOME="$cann_home"
if [[ -n "$vendor_dir" ]]; then
    cmake -S "$script_dir" -B "$build_dir" -DVENDOR_DIR="$vendor_dir"
fi
cmake --build "$build_dir" --config Release --parallel "${BUILD_JOBS:-4}"

test -x "$build_dir/aclnn_app_runner"
echo "$build_dir/aclnn_app_runner"
