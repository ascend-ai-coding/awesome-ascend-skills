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
# List idle NPU chip IDs.
#
# Design: avoids parsing npu-smi info main table.
#   1. npu-smi info -m      -> get NPU/Chip mapping
#   2. npu-smi info -t usages -i <id> -> check if NPU is busy
#   3. If no npu-smi, fallback to ps-based detection.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi not found" >&2
    exit 1
fi

# Step 1: Get all compute chips from mapping table (npu-smi info -m)
# Format:
#   NPU ID   Chip ID   Chip Logic ID   Chip Name
#   5        0         0               Ascend 910B3
#   5        1         -               Mcu
map_output=$(npu-smi info -m 2>/dev/null || true)
if [ -z "$map_output" ]; then
    echo "Failed to get NPU mapping" >&2
    exit 1
fi

# Parse mapping table: collect (npu_id, chip_id) pairs, skip MCU
all_chips=""
while IFS= read -r line; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//')
    [ -z "$line" ] && continue
    # Skip header
    [[ "$line" == *"NPU ID"* ]] && continue

    parts=($line)
    [ ${#parts[@]} -lt 4 ] && continue

    npu_id=${parts[0]}
    chip_id=${parts[1]}
    logic_id=${parts[2]}

    # Skip non-numeric or MCU chips
    if ! [[ "$npu_id" =~ ^[0-9]+$ ]]; then
        continue
    fi
    if ! [[ "$chip_id" =~ ^[0-9]+$ ]]; then
        continue
    fi
    # Skip MCU (Chip Logic ID == "-")
    if [ "$logic_id" = "-" ]; then
        continue
    fi

    all_chips="$all_chips $npu_id:$chip_id"
done <<< "$map_output"

if [ -z "$all_chips" ]; then
    exit 0
fi

# Step 2: Check usage for each unique NPU via structured subcommand
used_npus=""
for npu_chip in $all_chips; do
    npu_id=${npu_chip%:*}
    # Skip if already checked
    if echo "$used_npus" | grep -qw "$npu_id"; then
        continue
    fi

    # Query usages subcommand (key:value format, stable)
    usages_output=$(npu-smi info -t usages -i "$npu_id" 2>/dev/null || true)
    if [ -n "$usages_output" ]; then
        # Extract HBM Usage Rate and Aicore Usage Rate
        hbm_usage=$(echo "$usages_output" | grep "HBM Usage Rate" | head -1 | sed -n 's/.*:[[:space:]]*//p' | tr -d '%[:space:]')
        aicore_usage=$(echo "$usages_output" | grep "Aicore Usage Rate" | head -1 | sed -n 's/.*:[[:space:]]*//p' | tr -d '%[:space:]')

        # If either usage is significant (>1%), mark as used
        if [ -n "$hbm_usage" ] && [ "$hbm_usage" != "0" ] && [ "${hbm_usage%.*}" -gt 1 ] 2>/dev/null; then
            used_npus="$used_npus $npu_id"
            continue
        fi
        if [ -n "$aicore_usage" ] && [ "$aicore_usage" != "0" ] && [ "${aicore_usage%.*}" -gt 1 ] 2>/dev/null; then
            used_npus="$used_npus $npu_id"
            continue
        fi
    fi

done

# Step 3: Output idle chip IDs
first=1
found=0
for npu_chip in $all_chips; do
    npu_id=${npu_chip%:*}
    chip_id=${npu_chip#*:}

    # Skip if NPU is used
    if echo "$used_npus" | grep -qw "$npu_id"; then
        continue
    fi

    # Map to global chip ID: npu_id * max_chips + chip_id
    # For single-chip devices (chip_id=0), this equals npu_id
    gid=$((npu_id * 2 + chip_id))

    if [ $first -eq 1 ]; then
        printf "%d" "$gid"
        first=0
    else
        printf " %d" "$gid"
    fi
    found=1
done

if [ $found -eq 1 ]; then
    printf "\n"
fi
