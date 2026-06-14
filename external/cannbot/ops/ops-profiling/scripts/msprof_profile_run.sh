#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
# msprof runner - msprof 方案：环境没有 msopprof 时使用 msprof 采集 7 组 aic-metrics，
# 配合 msprof_perf_summary.py 生成 ops-profiling skill 需要的摘要。
#
# Usage:
#   bash msprof_profile_run.sh [--warm-up=N] [--output=<dir>] -- <executable> [args...]
#
# Example:
#   bash msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
#        ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096
#
# 产出:
#   <output_dir>/PROF_GROUP_<timestamp>/PROF_<ArithmeticUtilization|...>/ (7 个子目录)
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

WARM_UP=3
OUTPUT_DIR="./msprof_output"
APP_CMD=()

usage() {
    cat <<'EOF'
Usage: bash msprof_profile_run.sh [OPTIONS] -- <executable> [args...]

Options:
  --warm-up=N     在正式采集之前，先跑 N 次可执行文件（不采集）预热 DVFS。默认 3。
  --output=<dir>  msprof 结果根目录。默认 ./msprof_output。
  -h, --help      Show this help.

脚本会按顺序用 msprof 采集 7 个 aic-metrics:
  PipeUtilization, ArithmeticUtilization, Memory, MemoryL0, MemoryUB,
  L2Cache, ResourceConflictRatio
所有 PROF 目录放在同一个 PROF_GROUP_<timestamp>/ 下，便于 msprof_perf_summary.py 汇总。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-up=*) WARM_UP="${1#*=}"; shift ;;
        --output=*)  OUTPUT_DIR="${1#*=}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; APP_CMD=("$@"); break ;;
        -*)          echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
        *)           APP_CMD=("$@"); break ;;
    esac
done

if [[ ${#APP_CMD[@]} -eq 0 ]]; then
    echo "ERROR: 缺少可执行文件参数 (使用 -- 分隔 msprof 选项与 app 命令)" >&2
    usage
    exit 1
fi

if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: 未找到 msprof，请先 source CANN set_env.sh / setenv.bash" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
GROUP_DIR="${OUTPUT_DIR}/PROF_GROUP_${TS}"
mkdir -p "$GROUP_DIR"

echo "=== msprof profiling ==="
echo "App      : ${APP_CMD[*]}"
echo "Output   : ${GROUP_DIR}"
echo "Warm-up  : ${WARM_UP}"

if [[ "$WARM_UP" -gt 0 ]]; then
    echo "--- Warm-up x${WARM_UP} ---"
    for i in $(seq 1 "$WARM_UP"); do
        echo "  warm-up ${i}/${WARM_UP}"
        "${APP_CMD[@]}" >/dev/null 2>&1 || true
    done
fi

METRICS=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

for M in "${METRICS[@]}"; do
    SUB="${GROUP_DIR}/PROF_${M}"
    mkdir -p "$SUB"
    echo "--- aic-metrics=${M} -> ${SUB} ---"
    msprof --output="$SUB" \
           --ai-core=on \
           --aic-metrics="$M" \
           --task-time=on \
           --ascendcl=on \
           "${APP_CMD[@]}" >"${SUB}/msprof.log" 2>&1 || {
        echo "ERROR: msprof run failed for ${M}. See ${SUB}/msprof.log" >&2
        tail -20 "${SUB}/msprof.log" >&2 || true
        exit 1
    }
done

# 追加 sample-based 采集一份，用于逐核负载均衡分析 (task_cyc in aicore.db)
SAMPLE_SUB="${GROUP_DIR}/PROF_Sample"
mkdir -p "$SAMPLE_SUB"
echo "--- sample-based (per-core load balance) -> ${SAMPLE_SUB} ---"
msprof --output="$SAMPLE_SUB" \
       --ai-core=on \
       --aic-metrics=PipeUtilization \
       --aic-mode=sample-based \
       --aic-freq=100 \
       --task-time=on \
       --ascendcl=on \
       "${APP_CMD[@]}" >"${SAMPLE_SUB}/msprof.log" 2>&1 || {
    echo "WARN: sample-based run failed, per-core analysis will be skipped. See ${SAMPLE_SUB}/msprof.log" >&2
    tail -20 "${SAMPLE_SUB}/msprof.log" >&2 || true
}

echo ""
echo "=== Done. PROF group = ${GROUP_DIR} ==="
echo "Next:"
echo "  python3 $(dirname "$(readlink -f "$0")")/msprof_perf_summary.py ${GROUP_DIR} <ops_dir>"
