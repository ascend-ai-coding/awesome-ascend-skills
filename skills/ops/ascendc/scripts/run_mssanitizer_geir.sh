#!/bin/bash
# ============================================================================
# mssanitizer 内存检测脚本 — C++ 模式（ops 算子仓专用）
#
# 使用方法:
#   bash run_mssanitizer_geir.sh <project_dir> [cann_root]
#
# 参数:
#   $1 — 算子工程目录路径（必须），如 /home/rcz/ops-nn/activation/gelu_quant
#   $2 — CANN 安装根路径（可选），默认取 ASCEND_HOME_PATH
#
# 本脚本会:
#   1. 自动检测算子工程结构，定位 examples 目录下的测试源码
#      - 优先查找 test_geir_*.cpp（GE IR 模式）
#      - 其次查找 test_aclnn_*.cpp（aclnn 模式）
#   2. 自动构建测试可执行文件（若尚未构建）
#   3. 依次执行 memcheck(device-heap) → memcheck(cann-heap) → racecheck → initcheck → synccheck
#   4. 生成汇总报告（含日志解析）
# ============================================================================

set -euo pipefail

# ── 参数解析 ────────────────────────────────────────────────────────────────
PROJECT_DIR="${1:?用法: bash run_mssanitizer_geir.sh <project_dir> [cann_root]}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

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
LOG_DIR="${PROJECT_DIR}/mssanitizer_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARSE_TOOL="${SCRIPT_DIR}/parse_mssanitizer_log.py"
BUILD_DIR="${PROJECT_DIR}/mssanitizer_test/build"

echo "========================================"
echo "mssanitizer 内存检测 (C++ 模式)"
echo "========================================"
echo "算子工程目录: ${PROJECT_DIR}"
echo "CANN 根路径:  ${CANN_ROOT}"
echo "日志目录:     ${LOG_DIR}"
echo "时间戳:       ${TIMESTAMP}"
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

# ── 定位测试可执行文件 ────────────────────────────────────────────────────
TEST_EXEC=""
OP_NAME=""
TEST_MODE=""

OP_NAME=$(basename "$PROJECT_DIR")

# 检查是否已有构建好的可执行文件（优先 geir，其次 aclnn）
if [ -f "${BUILD_DIR}/test_geir_${OP_NAME}" ]; then
    TEST_EXEC="${BUILD_DIR}/test_geir_${OP_NAME}"
    TEST_MODE="GE IR"
    echo "找到已有测试可执行文件: ${TEST_EXEC}"
elif [ -f "${PROJECT_DIR}/build/test_geir_${OP_NAME}" ]; then
    TEST_EXEC="${PROJECT_DIR}/build/test_geir_${OP_NAME}"
    TEST_MODE="GE IR"
    echo "找到已有测试可执行文件: ${TEST_EXEC}"
elif [ -f "${BUILD_DIR}/test_aclnn_${OP_NAME}" ]; then
    TEST_EXEC="${BUILD_DIR}/test_aclnn_${OP_NAME}"
    TEST_MODE="aclnn"
    echo "找到已有测试可执行文件: ${TEST_EXEC}"
elif [ -f "${PROJECT_DIR}/build/test_aclnn_${OP_NAME}" ]; then
    TEST_EXEC="${PROJECT_DIR}/build/test_aclnn_${OP_NAME}"
    TEST_MODE="aclnn"
    echo "找到已有测试可执行文件: ${TEST_EXEC}"
else
    # 尝试自动构建
    echo "未找到测试可执行文件，尝试自动构建..."

    # 优先查找 test_geir_*.cpp
    EXAMPLE_SRC=$(find "${PROJECT_DIR}/examples" -name "test_geir_*.cpp" 2>/dev/null | head -1)
    if [ -n "$EXAMPLE_SRC" ]; then
        TEST_MODE="GE IR"
        echo "找到 GE IR 测试源文件: ${EXAMPLE_SRC}"
    else
        # 其次查找 test_aclnn_*.cpp
        EXAMPLE_SRC=$(find "${PROJECT_DIR}/examples" -name "test_aclnn_*.cpp" 2>/dev/null | head -1)
        if [ -n "$EXAMPLE_SRC" ]; then
            TEST_MODE="aclnn"
            echo "找到 aclnn 测试源文件: ${EXAMPLE_SRC}"
        fi
    fi

    if [ -z "$EXAMPLE_SRC" ]; then
        echo "错误: 未在 ${PROJECT_DIR}/examples/ 下找到 test_geir_*.cpp 或 test_aclnn_*.cpp 测试源文件"
        echo "请确认算子工程目录结构正确（ops 算子仓应有 examples/ 目录）"
        exit 1
    fi

    EXAMPLE_BASENAME=$(basename "$EXAMPLE_SRC" .cpp)
    mkdir -p "${BUILD_DIR}"

    if [ "$TEST_MODE" = "GE IR" ]; then
        # ── GE IR 模式构建 ──
        OP_GRAPH_DIR="${PROJECT_DIR}/op_graph"
        if [ ! -d "$OP_GRAPH_DIR" ]; then
            echo "错误: 未找到 op_graph 目录: ${OP_GRAPH_DIR}"
            exit 1
        fi

        cat > "${BUILD_DIR}/CMakeLists.txt" << CMAKE_EOF
cmake_minimum_required(VERSION 3.16)
project(${EXAMPLE_BASENAME})

set(CMAKE_CXX_STANDARD 17)

find_package(ASCEND REQUIRED)

add_executable(${EXAMPLE_BASENAME}
    ${EXAMPLE_SRC}
)

target_include_directories(${EXAMPLE_BASENAME} PRIVATE
    \${ASCEND_INC_DIR}
    ${OP_GRAPH_DIR}
)

target_link_directories(${EXAMPLE_BASENAME} PRIVATE
    \${ASCEND_LIB_DIR}
)

target_link_libraries(${EXAMPLE_BASENAME}
    ascendcl
    ge_runner
    graph
    register
)
CMAKE_EOF

        echo "开始构建 ${EXAMPLE_BASENAME} (GE IR 模式)..."
        cd "${BUILD_DIR}"
        cmake . 2>&1 | tail -3
        make -j$(nproc) 2>&1 | tail -5

    else
        # ── aclnn 模式构建 ──
        # aclnn 测试文件使用 AscendCL + opapi 库，无需 op_graph
        CANN_LIB_DIR="${CANN_ROOT}/lib64"
        CANN_ARCH_LIB_DIR="${CANN_ROOT}/aarch64-linux/lib64"
        [ ! -d "$CANN_ARCH_LIB_DIR" ] && CANN_ARCH_LIB_DIR="${CANN_ROOT}/$(uname -m)-linux/lib64"

        cat > "${BUILD_DIR}/CMakeLists.txt" << CMAKE_EOF
cmake_minimum_required(VERSION 3.16)
project(${EXAMPLE_BASENAME})

set(CMAKE_CXX_STANDARD 17)

add_executable(${EXAMPLE_BASENAME}
    ${EXAMPLE_SRC}
)

target_include_directories(${EXAMPLE_BASENAME} PRIVATE
    ${CANN_ROOT}/include
)

target_link_directories(${EXAMPLE_BASENAME} PRIVATE
    ${CANN_LIB_DIR}
    ${CANN_ARCH_LIB_DIR}
)

target_link_libraries(${EXAMPLE_BASENAME}
    ascendcl
    opapi
    nnopbase
)
CMAKE_EOF

        echo "开始构建 ${EXAMPLE_BASENAME} (aclnn 模式)..."
        cd "${BUILD_DIR}"
        cmake . 2>&1 | tail -3
        make -j$(nproc) 2>&1 | tail -5
    fi

    if [ -f "${BUILD_DIR}/${EXAMPLE_BASENAME}" ]; then
        TEST_EXEC="${BUILD_DIR}/${EXAMPLE_BASENAME}"
        echo "构建成功: ${TEST_EXEC}"
    else
        echo "错误: 构建失败"
        exit 1
    fi
fi

if [ ! -x "$TEST_EXEC" ]; then
    echo "错误: 测试可执行文件不可执行: ${TEST_EXEC}"
    exit 1
fi

echo "测试可执行文件: ${TEST_EXEC}"
echo "检测模式: ${TEST_MODE}"
echo ""

mkdir -p "${LOG_DIR}"

# ── 通用执行函数 ───────────────────────────────────────────────────────────
run_check() {
    local label="$1"; shift
    local log_file="${LOG_DIR}/geir_${label}_${TIMESTAMP}.log"

    echo "========================================"
    echo ">>> ${label}"
    echo "========================================"

    if [ "$TEST_MODE" = "GE IR" ]; then
        "$MSSANITIZER_PATH" "$@" \
            --log-file="${log_file}" \
            -- "${TEST_EXEC}" float || true
    else
        "$MSSANITIZER_PATH" "$@" \
            --log-file="${log_file}" \
            -- "${TEST_EXEC}" || true
    fi

    echo "${label} 完成. 日志: ${log_file}"
    echo ""
}

# ── 1. memcheck (device-heap) ──────────────────────────────────────────────
run_check "memcheck_device" -t memcheck --leak-check=yes --check-device-heap=yes

# ── 2. memcheck (cann-heap) ────────────────────────────────────────────────
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

SUMMARY_FILE="${LOG_DIR}/geir_mssanitizer_summary_${TIMESTAMP}.md"
{
    echo "# mssanitizer 内存检测汇总报告 (${TEST_MODE} 模式)"
    echo ""
    echo "**生成时间**: $(date)"
    echo "**CANN 路径**: ${CANN_ROOT}"
    echo "**算子工程**: ${PROJECT_DIR}"
    echo "**测试可执行文件**: ${TEST_EXEC}"
    echo "**检测模式**: ${TEST_MODE} (C++ 可执行文件)"
    echo ""

    echo "## 检测工具执行结果总览"
    echo ""
    echo "| 检测工具 | 错误数 | 状态 |"
    echo "|---------|--------|------|"

    total_errors=0
    for check_type in memcheck_device memcheck_cann racecheck initcheck synccheck; do
        log_file="${LOG_DIR}/geir_${check_type}_${TIMESTAMP}.log"
        if [ -f "${log_file}" ] && [ -s "${log_file}" ]; then
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
        log_file="${LOG_DIR}/geir_${check_type}_${TIMESTAMP}.log"

        echo "## ${check_type}"
        echo ""

        if [ ! -f "${log_file}" ]; then
            echo "日志文件不存在，跳过。"
            echo ""
            echo "---"
            echo ""
            continue
        fi

        if [ -s "${log_file}" ]; then
            error_count=$(grep -c "ERROR" "${log_file}" 2>/dev/null | head -1 || echo "0")
            error_count=$(echo "${error_count}" | grep -o '[0-9]*' | head -1)
            error_count=${error_count:-0}
        else
            error_count=0
        fi

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
    for f in "${LOG_DIR}"/geir_*_${TIMESTAMP}.*; do
        [ -e "$f" ] || continue
        fname=$(basename "${f}")
        case "${fname}" in
            *summary*)  desc="汇总报告（本文件）" ;;
            *analysis*) desc="解析报告" ;;
            *.log)       desc="原始日志" ;;
            *)           desc="其他" ;;
        esac
        echo "| \`${fname}\` | ${desc} |"
    done
} > "${SUMMARY_FILE}"

echo "汇总报告: ${SUMMARY_FILE}"

echo ""
echo "========================================"
echo "所有 mssanitizer 检测完成! (${TEST_MODE} 模式)"
echo "========================================"
ls -la "${LOG_DIR}"/geir_*_${TIMESTAMP}.* 2>/dev/null || true
