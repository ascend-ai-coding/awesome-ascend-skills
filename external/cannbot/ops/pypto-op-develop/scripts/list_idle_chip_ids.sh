#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi not found" >&2
    exit 1
fi

npu-smi info | awk '
BEGIN {
    proc = 0
    expect_chip = 0
    mode = 1
}
/^\| NPU     Chip/ {
    proc = 1
    next
}
!proc && /^\| [0-9]+/ {
    if ($3 != "|" && $3 !~ /^[0-9]+$/) {
        current_npu = $2 + 0
        expect_chip = 1
        next
    }
    if (expect_chip) {
        if ($3 ~ /^[0-9]+$/) {
            mode = 2
            all[$3 + 0] = 1
        } else {
            all[current_npu] = 1
        }
        expect_chip = 0
    }
}
proc && /^\| [0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]+[0-9]+/ {
    npu = $2 + 0
    local_chip = $3 + 0
    gid = (mode == 2 ? npu * 2 + local_chip : npu)
    used[gid] = 1
}
END {
    first = 1
    found = 0
    for (i = 0; i < 32; i++) {
        if ((i in all) && !(i in used)) {
            if (!first) {
                printf " "
            }
            printf "%d", i
            first = 0
            found = 1
        }
    }
    if (found) {
        printf "\n"
    }
}
'
