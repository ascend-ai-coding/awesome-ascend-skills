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
# msprof runner - 统一性能采集入口
#
# 支持三种模式：
#   1. 标准模式 (默认): 对单个可执行文件采集 7 组 aic-metrics + sample-based
#   2. --compare 模式:  对算子目录下的 model.py vs model_new_ascendc.py 做对比测试
#   3. --batch 模式:    批量扫描目录下的多个算子子目录，并行执行对比测试
#
# Usage (标准模式):
#   bash msprof_profile_run.sh [--warm-up=N] [--output=<dir>] -- <executable> [args...]
#
# Usage (--compare 模式):
#   bash msprof_profile_run.sh --compare --output-dir=/path/to/op_dir [--warm-up=N] [--device=N]
#
# Usage (--batch 模式):
#   bash msprof_profile_run.sh --batch --base-dir=/path/to/output_dir [--max-jobs=N] [--device-start=N]
#
# Example (标准):
#   bash msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
#        ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096
#
# Example (对比):
#   bash msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
#
# Example (批量):
#   bash msprof_profile_run.sh --batch --base-dir=./output_performance --max-jobs=7 --device-start=1
#
# 产出:
#   标准模式: <output_dir>/PROF_GROUP_<timestamp>/PROF_<Metric>/ (7 个子目录 + 1 个 Sample)
#   对比模式: <output_dir>/performance.json + performance.log + perf_report.md
#   批量模式: <base_dir>/batch_performance.log + 各子目录 performance.json
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
WARM_UP=3
OUTPUT_DIR="./msprof_output"
APP_CMD=()

# 模式开关
MODE="standard"   # standard | compare | batch
OUTPUT_DIR_ARG=""  # --compare 模式用的算子目录
DEVICE_ID=""
BASE_DIR=""
MAX_JOBS=7
DEVICE_START=1

usage() {
    cat <<'EOF'
Usage: bash msprof_profile_run.sh [OPTIONS] -- <executable> [args...]
       bash msprof_profile_run.sh --compare --output-dir=<dir> [OPTIONS]
       bash msprof_profile_run.sh --batch --base-dir=<dir> [OPTIONS]

通用选项:
  --warm-up=N     在正式采集之前，先跑 N 次可执行文件（不采集）预热 DVFS。默认 3。
  --output=<dir>  msprof 结果根目录。默认 ./msprof_output。
  -h, --help      Show this help.

标准模式 (默认):
  bash msprof_profile_run.sh [通用选项] -- <executable> [args...]

对比模式 (--compare):
  --compare              启用对比模式：测试 model.py vs model_new_ascendc.py
  --output-dir=<dir>     算子输出目录（包含 model.py, model_new_ascendc.py, .jsonl）
  --device=N             指定 NPU 设备 ID；未指定时自动选择空闲卡
  --repeats=N            重复采集次数（默认 1）
  --retry=N              单 case 解析失败重试次数（默认 2）
  --keep-prof            保留 msprof 原始 PROF 目录（用于深度分析）

批量模式 (--batch):
  --batch                启用批量模式：扫描 base-dir 下所有子目录并行测试
  --base-dir=<dir>       包含多个算子输出子目录的根目录
  --max-jobs=N           最大并发数（默认 7）
  --device-start=N       起始 NPU 设备 ID（默认 1）

脚本会按顺序用 msprof 采集 7 个 aic-metrics:
  PipeUtilization, ArithmeticUtilization, Memory, MemoryL0, MemoryUB,
  L2Cache, ResourceConflictRatio
所有 PROF 目录放在同一个 PROF_GROUP_<timestamp>/ 下，便于 msprof_perf_summary.py 汇总。
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-up=*) WARM_UP="${1#*=}"; shift ;;
        --output=*)  OUTPUT_DIR="${1#*=}"; shift ;;
        --output-dir=*) OUTPUT_DIR_ARG="${1#*=}"; shift ;;
        --device=*)  DEVICE_ID="${1#*=}"; shift ;;
        --repeats=*) REPEATS="${1#*=}"; shift ;;
        --retry=*)   RETRY="${1#*=}"; shift ;;
        --keep-prof) KEEP_PROF=1; shift ;;
        --compare)   MODE="compare"; shift ;;
        --batch)     MODE="batch"; shift ;;
        --base-dir=*) BASE_DIR="${1#*=}"; shift ;;
        --max-jobs=*) MAX_JOBS="${1#*=}"; shift ;;
        --device-start=*) DEVICE_START="${1#*=}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; APP_CMD=("$@"); break ;;
        -*)          echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
        *)           APP_CMD=("$@"); break ;;
    esac
done

# 检查 msprof
if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: 未找到 msprof，请先 source CANN set_env.sh / setenv.bash" >&2
    exit 1
fi

# ============================================================================
# 标准模式：对单个可执行文件采集
# ============================================================================
run_standard() {
    if [[ ${#APP_CMD[@]} -eq 0 ]]; then
        echo "ERROR: 缺少可执行文件参数 (使用 -- 分隔 msprof 选项与 app 命令)" >&2
        usage
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

    # 追加 sample-based 采集一份，用于逐核负载均衡分析
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
    echo "  python3 ${SCRIPT_DIR}/msprof_perf_summary.py ${GROUP_DIR} <ops_dir>"
}

# ============================================================================
# 对比模式：model.py vs model_new_ascendc.py
# ============================================================================
run_compare() {
    if [[ -z "$OUTPUT_DIR_ARG" ]]; then
        echo "ERROR: --compare 模式必须指定 --output-dir=<dir>" >&2
        usage
        exit 1
    fi

    OUT_DIR="$(cd "$OUTPUT_DIR_ARG" && pwd)"

    # 调用 msprof_perf_summary.py --compare（已融合 kernel_perf.py 功能）
    local extra_args=()
    [[ -n "$DEVICE_ID" ]] && extra_args+=("--device" "$DEVICE_ID")
    [[ -n "${REPEATS:-}" ]] && extra_args+=("--repeats" "$REPEATS")
    [[ -n "${RETRY:-}" ]] && extra_args+=("--retry" "$RETRY")
    [[ "${KEEP_PROF:-0}" == "1" ]] && extra_args+=("--keep-prof")

    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --compare \
        --output-dir "$OUT_DIR" \
        --warmup "$WARM_UP" \
        "${extra_args[@]}"
}

# ============================================================================
# 批量模式：多 NPU 并行执行多个算子
# ============================================================================
run_batch() {
    if [[ -z "$BASE_DIR" ]]; then
        echo "ERROR: --batch 模式必须指定 --base-dir=<dir>" >&2
        usage
        exit 1
    fi

    BASE_DIR="$(cd "$BASE_DIR" && pwd)"
    LOG_FILE="${BASE_DIR}/batch_performance.log"
    > "$LOG_FILE"

    LOCK_DIR="/tmp/ascend_locks_$$"
    mkdir -p "$LOCK_DIR"

    device_id=$DEVICE_START

    for dir in "$BASE_DIR"/*; do
        if [[ -d "$dir" ]]; then
            folder_name=$(basename "$dir")

            # 控制总并发数
            while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
                sleep 0.5
            done

            # 寻找空闲设备
            while true; do
                lock_file="$LOCK_DIR/device_${device_id}.lock"
                if ! ( set -o noclobber; flock -n 200 ) 200>"$lock_file" 2>/dev/null; then
                    device_id=$(( (device_id % 7) + 1 ))
                    sleep 0.2
                else
                    break
                fi
            done

            echo ">>> 锁定设备 $device_id 并启动算子: $folder_name" | tee -a "$LOG_FILE"

            # 启动后台任务
            (
                lock_file="$LOCK_DIR/device_${device_id}.lock"
                exec 200>"$lock_file"
                flock 200

                export ASCEND_RT_VISIBLE_DEVICES=$device_id

                # 调用对比模式
                bash "$0" --compare --output-dir="$dir" --warm-up="$WARM_UP" --device="$device_id" 2>&1 | \
                    grep -v "tiling struct \[MC2MatmulV3TilingData\] is conflict" | \
                    grep -v "tiling struct \[TileInfo\] is conflict" \
                    >> "$LOG_FILE"

                echo ">>> 算子 $folder_name 在设备 $device_id 上执行完毕" >> "$LOG_FILE"
            ) &

            device_id=$(( (device_id % 7) + 1 ))
        fi
    done

    wait
    rm -rf "$LOCK_DIR"

    echo "所有并行任务已执行完毕，日志已保存至 $LOG_FILE"

    # 生成批量汇总报告
    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --batch "$BASE_DIR" \
        --output-md "${BASE_DIR}/batch_report.md" \
        --output-json "${BASE_DIR}/batch_summary.json"
}

# ============================================================================
# 主入口
# ============================================================================
case "$MODE" in
    standard)
        run_standard
        ;;
    compare)
        run_compare
        ;;
    batch)
        run_batch
        ;;
    *)
        echo "ERROR: 未知模式: $MODE" >&2
        usage
        exit 1
        ;;
esac
