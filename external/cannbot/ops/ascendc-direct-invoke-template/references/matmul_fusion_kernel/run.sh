#!/bin/bash
# ============================================================
# run.sh — 一键编译并测试（NPU 实机执行）
# [MODIFY] OP_NAME — 修改为你的算子名称
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# [MODIFY] 算子名称（影响二进制文件名）
OP_NAME="matmul_fused_swat"

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: bash run.sh [--skip-build] [M K N]

Options:
  --skip-build   跳过编译，直接复用 build/${OP_NAME}
  -h, --help     显示帮助

Examples:
  bash run.sh
  bash run.sh 256 512 128
  bash run.sh --skip-build 128 128 128
EOF
}

SKIP_BUILD=0
POSITIONAL_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL_ARGS[@]}" -eq 0 ]; then
    M=128
    K=128
    N=128
elif [ "${#POSITIONAL_ARGS[@]}" -eq 3 ]; then
    M="${POSITIONAL_ARGS[0]}"
    K="${POSITIONAL_ARGS[1]}"
    N="${POSITIONAL_ARGS[2]}"
else
    usage
    die "M/K/N 参数必须同时提供（共 3 个）"
fi

for dim in "$M" "$K" "$N"; do
    [[ "$dim" =~ ^[0-9]+$ ]] || die "M/K/N 必须是正整数"
    [ "$dim" -gt 0 ] || die "M/K/N 必须大于 0"
done

echo "================================================================"
echo "[1/4] CANN 环境设置"
echo "================================================================"
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    die "ASCEND_HOME_PATH 未设置，请先 source set_env.sh"
fi

command -v python3 >/dev/null 2>&1 || die "python3 不可用"

if [ "$SKIP_BUILD" -eq 0 ]; then
    echo
    echo "================================================================"
    echo "[2/4] 编译"
    echo "================================================================"
    mkdir -p build
    cd build
    cmake ..
    make -j4
    cd "$SCRIPT_DIR"
    echo "编译完成: build/${OP_NAME}"
else
    [ -x "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在"
    echo "[2/4] 跳过编译 (--skip-build)"
fi

echo
echo "================================================================"
echo "[3/4] 生成测试数据"
echo "================================================================"
cd build
rm -rf ./input/ ./output/
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 ../scripts/gen_data.py "$M" "$K" "$N"

echo
echo "================================================================"
echo "[4/4] NPU 执行与精度验证"
echo "================================================================"

"./${OP_NAME}" "$M" "$K" "$N"
python3 ../scripts/verify_result.py "$M" "$N"

echo
echo "测试完成。"
