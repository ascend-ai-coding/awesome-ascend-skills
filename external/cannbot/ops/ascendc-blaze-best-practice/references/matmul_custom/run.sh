#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Build + generate data + run + verify for matmul direct-invoke sample.
#
# Usage:
#   bash run.sh                      # 默认 M=K=N=256, transA=false, transB=true
#   bash run.sh M K N                # 指定 shape
#   bash run.sh M K N TA TB          # 指定 shape + transA/transB (true/false)
#   bash run.sh --skip-build ...     # 复用已有编译产物（代码审查阶段用）
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# [MODIFY] 替换为实际的算子二进制名称
OP_NAME="matmul_custom"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
    shift
fi

# 默认形状，可通过 CLI 覆盖
M="${1:-256}"
K="${2:-256}"
N="${3:-256}"
TRANS_A="${4:-false}"
TRANS_B="${5:-true}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先配置 CANN 环境"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" || die "cmake 配置失败"
    cmake --build "${SCRIPT_DIR}/build" --parallel "$(nproc 2>/dev/null || echo 4)" || die "编译失败"
fi

echo "=== [3/4] 生成测试数据 (M=${M} K=${K} N=${N} transB=${TRANS_B}) ==="
cd build
python3 ../scripts/gen_data.py "${M}" "${K}" "${N}" "${TRANS_A}" "${TRANS_B}" || die "gen_data.py 执行失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" "${M}" "${K}" "${N}" "${TRANS_A}" "${TRANS_B}" || die "Kernel 运行失败 (exit $?)"
[ -f output/output.bin ] || die "Kernel 运行后 output/output.bin 不存在"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py "${M}" "${N}" || die "精度验证失败"

echo "=== 完成 ==="
exit 0
