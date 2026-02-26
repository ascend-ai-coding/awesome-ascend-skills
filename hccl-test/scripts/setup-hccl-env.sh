#!/bin/bash
#
# HCCL Test Environment Setup Script - 一键配置 HCCL Test 环境
# Usage: ./setup-hccl-env.sh [--mpi-type=mpich|openmpi] [--mpi-path=PATH] [--cann-path=PATH]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MPI_TYPE="mpich"
MPI_PATH=""
CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
HCCL Test Environment Setup Script

Usage: $0 [options]

Options:
    --mpi-type=TYPE      MPI 类型: mpich 或 openmpi (默认: mpich)
    --mpi-path=PATH      MPI 安装路径 (默认: /usr/local/mpich 或 /usr/local/openmpi)
    --cann-path=PATH     CANN 安装路径 (默认: /usr/local/Ascend/ascend-toolkit/latest)
    -h, --help           显示帮助信息

Examples:
    $0                                    # 使用默认 MPICH 配置
    $0 --mpi-type=openmpi                 # 使用 Open MPI
    $0 --mpi-path=/opt/mpich --cann-path=/opt/ascend-toolkit/latest
EOF
}

parse_args() {
    for arg in "$@"; do
        case $arg in
            --mpi-type=*)  MPI_TYPE="${arg#*=}" ;;
            --mpi-path=*)  MPI_PATH="${arg#*=}" ;;
            --cann-path=*) CANN_PATH="${arg#*=}" ;;
            -h|--help)     show_help; exit 0 ;;
            *)             log_error "Unknown argument: $arg"; show_help; exit 1 ;;
        esac
    done
}

check_product_model() {
    log_info "检查产品型号..."
    if command -v dmidecode &> /dev/null; then
        PRODUCT=$(dmidecode -t system 2>/dev/null | head -20 | grep Product || true)
        [[ -n "$PRODUCT" ]] && log_info "产品型号: $PRODUCT" || log_warn "无法获取产品型号（可能需要 root 权限）"
    else
        log_warn "dmidecode 命令不可用"
    fi
}

check_mpi_installation() {
    log_info "检查 MPI 安装..."

    [[ -z "$MPI_PATH" ]] && MPI_PATH=$([[ "$MPI_TYPE" == "mpich" ]] && echo "/usr/local/mpich" || echo "/usr/local/openmpi")

    [[ ! -d "$MPI_PATH" ]] && { log_error "MPI 安装路径不存在: $MPI_PATH"; exit 1; }
    [[ ! -f "$MPI_PATH/bin/mpirun" ]] && { log_error "mpirun 不存在: $MPI_PATH/bin/mpirun"; exit 1; }

    log_info "MPI 类型: $MPI_TYPE"
    log_info "MPI 路径: $MPI_PATH"
    MPI_VERSION=$("$MPI_PATH/bin/mpirun" --version 2>&1 | head -1 || true)
    log_info "MPI 版本: $MPI_VERSION"
}

check_cann_installation() {
    log_info "检查 CANN 安装..."

    [[ ! -d "$CANN_PATH" ]] && { log_error "CANN 安装路径不存在: $CANN_PATH"; exit 1; }
    [[ ! -d "$CANN_PATH/tools/hccl_test" ]] && { log_error "HCCL Test 目录不存在: $CANN_PATH/tools/hccl_test"; exit 1; }

    log_info "CANN 路径: $CANN_PATH"
    log_info "HCCL Test 路径: $CANN_PATH/tools/hccl_test"
}

setup_environment() {
    log_info "配置环境变量..."

    export INSTALL_DIR="$CANN_PATH"
    export PATH="$MPI_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$MPI_PATH/lib:${CANN_PATH}/lib64:${LD_LIBRARY_PATH:-}"

    log_info "已设置以下环境变量:"
    echo "  export INSTALL_DIR=$CANN_PATH"
    echo "  export PATH=$MPI_PATH/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=$MPI_PATH/lib:${CANN_PATH}/lib64:\$LD_LIBRARY_PATH"
}

print_env_commands() {
    echo ""
    echo "=========================================="
    echo "请在当前 shell 运行以下命令设置环境变量:"
    echo "=========================================="
    echo ""
    echo "export INSTALL_DIR=$CANN_PATH"
    echo "export PATH=$MPI_PATH/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=$MPI_PATH/lib:${CANN_PATH}/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "cd ${CANN_PATH}/tools/hccl_test"
    echo "make MPI_HOME=$MPI_PATH ASCEND_DIR=\$INSTALL_DIR"
    echo ""
}

main() {
    parse_args "$@"
    echo "=========================================="
    echo "HCCL Test Environment Setup"
    echo "=========================================="
    echo ""
    check_product_model
    echo ""
    check_mpi_installation
    echo ""
    check_cann_installation
    echo ""
    setup_environment
    print_env_commands
}

main "$@"
