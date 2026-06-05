#!/bin/bash
# ============================================================================
# mssanitizer 内存检测主脚本
#
# 使用方法:
#   bash run_mssanitizer.sh <test_script.py> [cann_root]
#
# 参数:
#   $1 — 测试脚本路径（必须），如 gelu_custom_mssanitizer_test.py
#   $2 — CANN 安装根路径（可选），默认取 ASCEND_HOME_PATH
#
# 本脚本会:
#   1. 锁定 CANN 环境，避免多版本污染
#   2. 定位 mssanitizer 工具
#   3. 依次执行 memcheck(device-heap) → memcheck(cann-heap) → racecheck → initcheck → synccheck
#   4. 生成汇总报告
# ============================================================================

set -euo pipefail

# ── 参数解析 ────────────────────────────────────────────────────────────────
TEST_SCRIPT="${1:?用法: bash run_mssanitizer.sh <test_script.py> [cann_root]}"
TEST_SCRIPT="$(cd "$(dirname "$TEST_SCRIPT")" && pwd)/$(basename "$TEST_SCRIPT")"

CANN_ROOT="${2:-${ASCEND_HOME_PATH:-}}"
if [ -z "$CANN_ROOT" ]; then
    echo "错误: 请通过第二个参数或 ASCEND_HOME_PATH 指定 CANN 安装根路径"
    exit 1
fi

# ── 锁定 CANN 环境 ─────────────────────────────────────────────────────────
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export LD_LIBRARY_PATH="${CANN_ROOT}/lib64:${CANN_ROOT}/lib64/plugin/opskernel:${CANN_ROOT}/lib64/plugin/nnengine:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$TEST_SCRIPT")")"
LOG_DIR="${PROJECT_DIR}/mssanitizer_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARSE_TOOL="${SCRIPT_DIR}/parse_mssanitizer_log.py"

echo "========================================"
echo "mssanitizer 内存检测"
echo "========================================"
echo "CANN 根路径:   ${CANN_ROOT}"
echo "测试脚本:      ${TEST_SCRIPT}"
echo "项目目录:      ${PROJECT_DIR}"
echo "日志目录:      ${LOG_DIR}"
echo "时间戳:        ${TIMESTAMP}"
echo ""

# ── 定位 mssanitizer ───────────────────────────────────────────────────────
MSSANITIZER_PATH="${CANN_ROOT}/tools/mssanitizer/bin/mssanitizer"
if [ ! -x "$MSSANITIZER_PATH" ]; then
    MSSANITIZER_PATH=$(find "$CANN_ROOT" -name mssanitizer -type f 2>/dev/null | head -1)
fi
if [ -z "$MSSANITIZER_PATH" ]; then
    echo "错误: 未找到 mssanitizer (CANN_ROOT=$CANN_ROOT)"
    exit 1
fi
echo "使用 mssanitizer: ${MSSANITIZER_PATH}"
mkdir -p "${LOG_DIR}"

# ── 通用执行函数 ───────────────────────────────────────────────────────────
run_check() {
    local label="$1"; shift
    local log_file="${LOG_DIR}/${label}_${TIMESTAMP}.log"
    local json_file="${LOG_DIR}/${label}_report_${TIMESTAMP}.json"

    echo "========================================"
    echo ">>> ${label}"
    echo "========================================"

    "$MSSANITIZER_PATH" "$@" \
        --log-file="${log_file}" \
        -- python3 "${TEST_SCRIPT}" --output "${json_file}" || true

    echo "${label} 完成. 日志: ${log_file}"
    echo ""
}

# ── 1. memcheck — device-heap ──────────────────────────────────────────────
run_check "memcheck_device" -t memcheck --leak-check=yes --check-device-heap=yes

# ── 2. memcheck — cann-heap ───────────────────────────────────────────────
run_check "memcheck_cann" -t memcheck --leak-check=yes --check-cann-heap=yes

# ── 3. racecheck ──────────────────────────────────────────────────────────
run_check "racecheck" -t racecheck

# ── 4. initcheck ──────────────────────────────────────────────────────────
run_check "initcheck" -t initcheck

# ── 5. synccheck ──────────────────────────────────────────────────────────
run_check "synccheck" -t synccheck

# ── 6. 生成汇总报告（含解析） ──────────────────────────────────────────────
echo "========================================"
echo ">>> 生成汇总报告"
echo "========================================"

SUMMARY_FILE="${LOG_DIR}/mssanitizer_summary_${TIMESTAMP}.md"
{
    echo "# mssanitizer 内存检测汇总报告"
    echo ""
    echo "**生成时间**: $(date)"
    echo "**CANN 路径**: ${CANN_ROOT}"
    echo "**测试脚本**: ${TEST_SCRIPT}"
    echo ""

    echo "## 检测工具执行结果总览"
    echo ""
    echo "| 检测工具 | 错误数 | 状态 |"
    echo "|---------|--------|------|"

    total_errors=0
    for check_type in memcheck_device memcheck_cann racecheck initcheck synccheck; do
        log_file="${LOG_DIR}/${check_type}_${TIMESTAMP}.log"
        if [ -f "${log_file}" ]; then
            error_count=$(grep -c "ERROR" "${log_file}" 2>/dev/null | head -1 || echo "0")
            error_count=$(echo "${error_count}" | grep -o '[0-9]*' | head -1)
            error_count=${error_count:-0}
        else
            error_count=0
        fi
        total_errors=$((total_errors + error_count))

        if [ "${error_count}" -eq 0 ]; then
            status="🟢 通过"
        elif [ "${error_count}" -lt 3 ]; then
            status="🟡 中等"
        else
            status="🔴 严重"
        fi
        echo "| **${check_type}** | ${error_count} | ${status} |"
    done

    echo ""
    if [ "${total_errors}" -eq 0 ]; then
        echo "**整体评级**: 🟢 **良好 (GOOD)** — 未检测到内存错误"
    elif [ "${total_errors}" -lt 5 ]; then
        echo "**整体评级**: 🟡 **中等 (MODERATE)** — 检测到 ${total_errors} 个内存问题"
    else
        echo "**整体评级**: 🔴 **严重 (CRITICAL)** — 检测到 ${total_errors} 个内存问题"
    fi
    echo ""
    echo "---"
    echo ""

    for check_type in memcheck_device memcheck_cann racecheck initcheck synccheck; do
        log_file="${LOG_DIR}/${check_type}_${TIMESTAMP}.log"

        echo "## ${check_type}"
        echo ""

        if [ ! -f "${log_file}" ]; then
            echo "日志文件不存在，跳过。"
            echo ""
            echo "---"
            echo ""
            continue
        fi

        error_count=$(grep -c "ERROR" "${log_file}" 2>/dev/null | head -1 || echo "0")
        error_count=$(echo "${error_count}" | grep -o '[0-9]*' | head -1)
        error_count=${error_count:-0}

        echo "- 检测到错误数: ${error_count}"
        echo ""

        if [ -s "${log_file}" ] && [ -f "${PARSE_TOOL}" ]; then
            echo "### 解析报告"
            echo ""
            python3 "${PARSE_TOOL}" "${log_file}" --no-title --heading-offset 2 || true
            echo ""
        fi

        if [ -s "${log_file}" ]; then
            echo "### 原始日志"
            echo ""
            echo '```'
            cat "${log_file}"
            echo '```'
            echo ""
        elif [ "${error_count}" -eq 0 ]; then
            echo "日志为空，未检测到问题。"
            echo ""
        fi

        echo "---"
        echo ""
    done

    echo "## 检测输出文件清单"
    echo ""
    echo "所有检测日志和报告保存在 \`${LOG_DIR}\` 目录下："
    echo ""
    echo "| 文件 | 说明 |"
    echo "|------|------|"
    for f in "${LOG_DIR}"/*_${TIMESTAMP}.*; do
        fname=$(basename "${f}")
        case "${fname}" in
            *summary*)  desc="汇总报告（本文件）" ;;
            *.log)       desc="原始日志" ;;
            *report*.json) desc="测试结果 JSON" ;;
            *)           desc="其他" ;;
        esac
        echo "| \`${fname}\` | ${desc} |"
    done
} > "${SUMMARY_FILE}"

echo "汇总报告: ${SUMMARY_FILE}"

echo ""
echo "========================================"
echo "所有 mssanitizer 检测完成!"
echo "========================================"
ls -la "${LOG_DIR}"/*_${TIMESTAMP}.*
