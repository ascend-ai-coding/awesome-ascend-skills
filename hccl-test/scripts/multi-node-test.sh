#!/bin/bash
#
# HCCL Multi-Node Test Script - 多机测试完整流程
# Usage: ./multi-node-test.sh --nodes <ip1,ip2,...\u003e --npus <n\u003e [options]
# Example: ./multi-node-test.sh --nodes 175.99.1.2,175.99.1.3 --npus 8 --mode full
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCCL_TEST_DIR="${INSTALL_DIR:-/usr/local/Ascend/ascend-toolkit/latest}/tools/hccl_test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# 默认配置
NODES=""
NPUS=8
MODE="quick"  # quick 或 full
MPI_TYPE="mpich"
SSH_USER="root"
CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"
SKIP_CHECK=false

show_help() {
    cat << EOF
HCCL Multi-Node Test Script

一键执行多机 HCCL 测试，包含前置检查、测试执行和结果汇总。

Usage: $0 --nodes <ip1,ip2,...\u003e --npus <n\u003e [options]

Required Arguments:
    --nodes <ip1,ip2,...\u003e    节点 IP 列表（逗号分隔）
    --npus <n\u003e               每节点 NPU 数量

Options:
    --mode <quick|full\u003e      测试模式 (默认: quick)
                               quick: 快速连通性测试 (-b 8K -e 64M)
                               full: 完整性能测试 (-b 8K -e 1G)
    --mpi-type <mpich|openmpi\u003e  MPI 类型 (默认: mpich)
    --ssh-user <user\u003e        SSH 登录用户 (默认: root)
    --cann-path <path\u003e       CANN 安装路径
    --skip-check              跳过前置检查
    -h, --help                显示帮助信息

Examples:
    # 快速连通性测试（双机 16 卡）
    $0 --nodes 175.99.1.2,175.99.1.3 --npus 8

    # 完整性能测试
    $0 --nodes 175.99.1.2,175.99.1.3 --npus 8 --mode full

    # 使用 Open MPI
    $0 --nodes 175.99.1.2,175.99.1.3 --npus 8 --mpi-type openmpi

    # 跳过前置检查（已知环境正常）
    $0 --nodes 175.99.1.2,175.99.1.3 --npus 8 --mode full --skip-check
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --nodes)
                NODES="$2"
                shift 2
                ;;
            --npus)
                NPUS="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                shift 2
                ;;
            --mpi-type)
                MPI_TYPE="$2"
                shift 2
                ;;
            --ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            --cann-path)
                CANN_PATH="$2"
                shift 2
                ;;
            --skip-check)
                SKIP_CHECK=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证必需参数
    if [[ -z "$NODES" ]]; then
        log_error "错误: 必须指定 --nodes 参数"
        show_help
        exit 1
    fi
    
    if [[ -z "$NPUS" ]]; then
        log_error "错误: 必须指定 --npus 参数"
        show_help
        exit 1
    fi
}

# 设置 MPI 环境
setup_mpi_env() {
    if [[ "$MPI_TYPE" == "mpich" ]]; then
        export MPI_HOME="${MPI_HOME:-/usr/local/mpich}"
        export PATH="$MPI_HOME/bin:$PATH"
    else
        export MPI_HOME="${MPI_HOME:-/usr/local/openmpi}"
        export PATH="$MPI_HOME/bin:$PATH"
    fi
    export INSTALL_DIR="$CANN_PATH"
    export LD_LIBRARY_PATH="$MPI_HOME/lib:${CANN_PATH}/lib64:${LD_LIBRARY_PATH:-}"
}

# 执行前置检查
run_pre_check() {
    if [[ "$SKIP_CHECK" == true ]]; then
        log_warn "跳过前置检查"
        return 0
    fi
    
    log_step "执行前置检查..."
    
    # 解析节点列表
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    
    # 调用前置检查脚本
    if [[ -f "$SCRIPT_DIR/pre-test-check.sh" ]]; then
        SSH_USER="$SSH_USER" CANN_PATH="$CANN_PATH" "$SCRIPT_DIR/pre-test-check.sh" "${NODE_ARRAY[@]}"
        if [[ $? -ne 0 ]]; then
            log_error "前置检查失败，请解决错误后重试"
            exit 1
        fi
    else
        log_warn "前置检查脚本不存在，跳过"
    fi
}

# 创建 hostfile
create_hostfile() {
    log_step "创建 hostfile..."
    
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    
    local hostfile="/tmp/hccl_hostfile_$$"
    > "$hostfile"
    
    for node in "${NODE_ARRAY[@]}"; do
        if [[ "$MPI_TYPE" == "mpich" ]]; then
            echo "$node:$NPUS" >> "$hostfile"
        else
            # Open MPI 格式需要主机名
            local hostname=$(ssh -o StrictHostKeyChecking=no ${SSH_USER}@$node "hostname" 2>/dev/null || echo "$node")
            echo "$hostname slots=$NPUS" >> "$hostfile"
        fi
    done
    
    echo "$hostfile"
}

# 清理残余进程
cleanup_residual() {
    log_step "清理残余进程..."
    
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    local total_npus=$((${#NODE_ARRAY[@]} * NPUS))
    
    local hostfile=$(create_hostfile)
    
    if [[ "$MPI_TYPE" == "mpich" ]]; then
        mpirun -f "$hostfile" -n $total_npus pkill -9 -f "all_reduce_test|all_gather_test|mpirun" 2>/dev/null || true
    else
        mpirun --hostfile "$hostfile" -n $total_npus pkill -9 -f "all_reduce_test|all_gather_test|openmpi" 2>/dev/null || true
    fi
    
    sleep 2
    log_info "残余进程清理完成"
}

# 运行单个测试
run_test() {
    local test_name="$1"
    local test_binary="$2"
    local extra_args="${3:-}"
    
    log_test "正在测试: $test_name"
    
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    local total_npus=$((${#NODE_ARRAY[@]} * NPUS))
    local hostfile=$(create_hostfile)
    
    # 根据模式设置数据量
    local data_args
    if [[ "$MODE" == "full" ]]; then
        data_args="-b 8K -e 1G -f 2"
    else
        data_args="-b 8K -e 64M -f 2"
    fi
    
    local cmd
    if [[ "$MPI_TYPE" == "mpich" ]]; then
        cmd="mpirun -f $hostfile -n $total_npus $HCCL_TEST_DIR/bin/$test_binary -p $NPUS $data_args $extra_args"
    else
        cmd="mpirun --prefix $MPI_HOME --hostfile $hostfile -x LD_LIBRARY_PATH -x HCCL_SOCKET_FAMILY --allow-run-as-root -n $total_npus $HCCL_TEST_DIR/bin/$test_binary -p $NPUS $data_args $extra_args"
    fi
    
    local log_file="/tmp/hccl_${test_name}_$$.log"
    
    if $cmd > "$log_file" 2&<1; then
        # 检查结果中是否有 success 或 NULL
        if grep -qE "(success|NULL)" "$log_file" 2>/dev/null; then
            # 提取带宽数据
            local max_bw=$(grep -v "^#" "$log_file" | awk '{print $3}' | grep -E '^[0-9.]+$' | sort -n | tail -1 || echo "N/A")
            log_info "✅ $test_name 通过 (最高带宽: ${max_bw} GB/s)"
            return 0
        else
            log_warn "⚠️  $test_name 输出异常"
            return 1
        fi
    else
        log_error "❌ $test_name 失败"
        log_error "   日志: $log_file"
        return 1
    fi
}

# 运行所有测试
run_all_tests() {
    log_step "开始 HCCL 测试 (模式: $MODE)..."
    
    local passed=0
    local failed=0
    
    cd "$HCCL_TEST_DIR"
    
    # 核心算子（必测）
    if run_test "all_reduce_test" "all_reduce_test" "-d fp32 -o sum"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    if run_test "all_gather_test" "all_gather_test" "-d fp32"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # 可选算子
    if run_test "broadcast_test" "broadcast_test" "-d fp32"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    if run_test "alltoall_test" "alltoall_test" "-d fp32"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    if run_test "reduce_scatter_test" "reduce_scatter_test" "-d fp32 -o sum"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # 输出汇总
    echo ""
    echo "=========================================="
    echo "测试汇总"
    echo "=========================================="
    echo -e "${GREEN}通过: $passed${NC}"
    echo -e "${RED}失败: $failed${NC}"
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}✅ 所有测试通过！${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ 部分测试失败，请检查日志文件: /tmp/hccl_*_$$.log${NC}"
        return 1
    fi
}

# 输出测试配置信息
show_config() {
    log_step "测试配置"
    
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    local total_npus=$((${#NODE_ARRAY[@]} * NPUS))
    
    echo "  节点列表: ${NODES}"
    echo "  每节点 NPU: $NPUS"
    echo "  总 NPU 数: $total_npus"
    echo "  测试模式: $MODE"
    echo "  MPI 类型: $MPI_TYPE"
    echo "  HCCL Test 路径: $HCCL_TEST_DIR"
    
    if [[ "$MODE" == "full" ]]; then
        echo "  数据量范围: 8K ~ 1G"
    else
        echo "  数据量范围: 8K ~ 64M"
    fi
    echo ""
}

# 主函数
main() {
    parse_args "$@"
    setup_mpi_env
    
    echo "=========================================="
    echo "HCCL Multi-Node Test"
    echo "=========================================="
    echo ""
    
    show_config
    run_pre_check
    cleanup_residual
    run_all_tests
}

main "$@"
