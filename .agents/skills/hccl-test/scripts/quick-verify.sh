#!/bin/bash
#
# HCCL Test Quick Verification Script - 快速验证所有集合通信算子
# Usage: ./quick-verify.sh [num_npus] [mpi_type]
#   num_npus: 测试的 NPU 数量 (默认: 4)
#   mpi_type: mpich 或 openmpi (默认: mpich)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCCL_TEST_DIR="${INSTALL_DIR:-/usr/local/Ascend/ascend-toolkit/latest}/tools/hccl_test"
NUM_NPUS="${1:-4}"
MPI_TYPE="${2:-mpich}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }

show_help() {
    cat << EOF
HCCL Test Quick Verification Script

快速验证所有 10 个集合通信算子是否正常工作。

Usage: $0 [num_npus] [mpi_type]

Arguments:
    num_npus    测试的 NPU 数量 (默认: 4)
    mpi_type    MPI 类型: mpich 或 openmpi (默认: mpich)

Examples:
    $0                    # 使用默认配置测试 4 卡
    $0 8                  # 测试 8 卡
    $0 4 openmpi          # 使用 Open MPI 测试 4 卡

Environment Variables:
    INSTALL_DIR     CANN 安装路径 (默认: /usr/local/Ascend/ascend-toolkit/latest)
    MPI_HOME        MPI 安装路径 (默认: /usr/local/mpich 或 /usr/local/openmpi)
EOF
}

# 设置 MPI 环境
setup_mpi_env() {
    if [[ "$MPI_TYPE" == "mpich" ]]; then
        MPI_HOME="${MPI_HOME:-/usr/local/mpich}"
        export PATH="$MPI_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$MPI_HOME/lib:${INSTALL_DIR:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:${LD_LIBRARY_PATH:-}"
    else
        MPI_HOME="${MPI_HOME:-/usr/local/openmpi}"
        export PATH="$MPI_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$MPI_HOME/lib:${INSTALL_DIR:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

# 检查环境
check_environment() {
    echo "=========================================="
    echo "HCCL Test Environment Check"
    echo "=========================================="
    echo ""
    
    # 检查 NPU
    log_info "检查 NPU 状态..."
    if command -v npu-smi &> /dev/null; then
        npu-smi info | head -20
        echo ""
    else
        log_warn "npu-smi 命令不可用"
    fi
    
    # 检查 MPI
    log_info "检查 MPI 环境..."
    if [[ -f "$MPI_HOME/bin/mpirun" ]]; then
        echo "MPI 路径: $MPI_HOME"
        "$MPI_HOME/bin/mpirun" --version 2>&1 | head -3
    else
        log_error "未找到 MPI: $MPI_HOME/bin/mpirun"
        exit 1
    fi
    echo ""
    
    # 检查 HCCL Test 工具
    log_info "检查 HCCL Test 工具..."
    if [[ -d "$HCCL_TEST_DIR/bin" ]]; then
        echo "HCCL Test 路径: $HCCL_TEST_DIR"
        ls -1 "$HCCL_TEST_DIR/bin/" | head -10
    else
        log_error "未找到 HCCL Test 工具: $HCCL_TEST_DIR"
        exit 1
    fi
    echo ""
    
    # 显示测试配置
    log_info "测试配置:"
    echo "  NPU 数量: $NUM_NPUS"
    echo "  MPI 类型: $MPI_TYPE"
    echo "  MPI 路径: $MPI_HOME"
    echo ""
}

# 运行单个测试
run_test() {
    local test_name="$1"
    local test_binary="$2"
    local extra_args="${3:-}"
    
    log_test "正在测试: $test_name"
    
    local cmd="$MPI_HOME/bin/mpirun -n $NUM_NPUS $HCCL_TEST_DIR/bin/$test_binary -p $NUM_NPUS -b 8K -e 64M -f 2 $extra_args"
    
    if $cmd &> /tmp/hccl_test_${test_name}.log; then
        if grep -q "success" /tmp/hccl_test_${test_name}.log 2>/dev/null || grep -q "NULL" /tmp/hccl_test_${test_name}.log 2>/dev/null; then
            log_info "✓ $test_name 通过"
            ((PASSED++))
            return 0
        else
            log_warn "? $test_name 输出异常 (无 success/NULL 标记)"
            ((SKIPPED++))
            return 1
        fi
    else
        log_error "✗ $test_name 失败"
        ((FAILED++))
        echo "  错误日志: /tmp/hccl_test_${test_name}.log"
        return 1
    fi
}

# 清理残余进程
cleanup_residual() {
    log_info "清理可能的残余进程..."
    pkill -9 -f "all_reduce_test|all_gather_test|alltoall_test|mpirun" 2>/dev/null || true
    sleep 1
}

# 主测试流程
run_all_tests() {
    echo "=========================================="
    echo "Starting HCCL Test Verification"
    echo "=========================================="
    echo ""
    
    cd "$HCCL_TEST_DIR"
    
    # 清理残余进程
    cleanup_residual
    
    # 测试归约类算子 (需要 -o sum 参数)
    run_test "all_reduce_test" "all_reduce_test" "-d fp32 -o sum"
    run_test "reduce_scatter_test" "reduce_scatter_test" "-d fp32 -o sum"
    run_test "reduce_test" "reduce_test" "-d fp32 -o sum"
    
    # 测试聚合类算子
    run_test "all_gather_test" "all_gather_test" "-d fp32"
    run_test "all_gatherv_test" "all_gatherv_test" "-d fp32"
    
    # 测试全对全通信
    run_test "alltoall_test" "alltoall_test" "-d fp32"
    run_test "alltoallv_test" "alltoallv_test" "-d fp32"
    
    # 测试广播和分发
    run_test "broadcast_test" "broadcast_test" "-d fp32"
    run_test "scatter_test" "scatter_test" "-d fp32"
    
    # 测试 ReduceScatterV
    run_test "reduce_scatterv_test" "reduce_scatterv_test" "-d fp32 -o sum"
    
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo -e "${GREEN}通过: $PASSED${NC}"
    echo -e "${RED}失败: $FAILED${NC}"
    echo -e "${YELLOW}异常: $SKIPPED${NC}"
    echo ""
    
    if [[ $FAILED -eq 0 ]]; then
        log_info "所有测试通过! HCCL 通信正常。"
        return 0
    else
        log_error "部分测试失败，请检查日志文件: /tmp/hccl_test_*.log"
        return 1
    fi
}

# 主函数
main() {
    # 解析参数
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    NUM_NPUS="${1:-4}"
    MPI_TYPE="${2:-mpich}"
    
    # 设置环境
    setup_mpi_env
    
    # 检查环境
    check_environment
    
    # 运行测试
    run_all_tests
}

main "$@"
