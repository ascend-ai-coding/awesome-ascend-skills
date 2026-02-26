#!/bin/bash
#
# HCCL Test Pre-test Check Script - 多机测试前置检查
# Usage: ./pre-test-check.sh <node1_ip> <node2_ip> ...
# Example: ./pre-test-check.sh 175.99.1.2 175.99.1.3
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_check() { echo -e "${BLUE}[CHECK]${NC} $1"; }

show_help() {
    cat << EOF
HCCL Test Pre-test Check Script

多机测试前置检查脚本，检查 SSH、CANN 版本、NPU 健康状态和网络连通性。

Usage: $0 <node1_ip> <node2_ip> ...

Arguments:
    node_ip    节点 IP 地址（可指定多个）

Examples:
    $0 175.99.1.2                    # 单机检查
    $0 175.99.1.2 175.99.1.3         # 双机检查
    $0 175.99.1.{2..5}               # 四机检查（bash 扩展）

Environment Variables:
    CANN_PATH    CANN 安装路径 (默认: /usr/local/Ascend/ascend-toolkit/latest)
    SSH_USER     SSH 登录用户 (默认: root)
    SSH_TIMEOUT  SSH 连接超时时间 (默认: 5 秒)
EOF
}

# 解析参数
if [[ $# -eq 0 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

NODES=("$@")
CANN_PATH="${CANN_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SSH_USER="${SSH_USER:-root}"
SSH_TIMEOUT="${SSH_TIMEOUT:-5}"

ERRORS=0
WARNINGS=0

echo "=========================================="
echo "HCCL Test Pre-test Check"
echo "=========================================="
echo ""
echo "检查节点: ${NODES[*]}"
echo "CANN 路径: $CANN_PATH"
echo "SSH 用户: $SSH_USER"
echo ""

# 1. SSH 免密登录检查
check_ssh() {
    log_check "检查 SSH 免密登录..."
    
    for node in "${NODES[@]}"; do
        if ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no ${SSH_USER}@$node "echo 'SSH_OK'" > /dev/null 2&<1; then
            log_info "✅ $node SSH 免密登录正常"
        else
            log_error "❌ $node SSH 免密登录失败"
            log_error "   请执行: ssh-copy-id -i ~/.ssh/id_rsa.pub ${SSH_USER}@$node"
            ((ERRORS++))
        fi
    done
    echo ""
}

# 2. CANN 版本一致性检查
check_cann_version() {
    log_check "检查 CANN 版本一致性..."
    
    local versions=()
    local version_info=""
    
    for node in "${NODES[@]}"; do
        version_info=$(ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no ${SSH_USER}@$node \
            "cat ${CANN_PATH}/version.cfg 2>/dev/null | grep runtime_running_version || echo 'NOT_FOUND'" 2>/dev/null || echo "SSH_FAILED")
        
        if [[ "$version_info" == "SSH_FAILED" ]]; then
            log_error "❌ $node 无法连接"
            ((ERRORS++))
            continue
        fi
        
        if [[ "$version_info" == "NOT_FOUND" ]]; then
            log_error "❌ $node 无法读取 CANN 版本"
            ((ERRORS++))
            continue
        fi
        
        versions+=("$node: $version_info")
        log_info "  $node: $version_info"
    done
    
    # 检查版本是否一致
    if [[ ${#versions[@]} -gt 1 ]]; then
        local first_version=$(echo "${versions[0]}" | awk -F': ' '{print $2}')
        local consistent=true
        
        for v in "${versions[@]}"; do
            local node_version=$(echo "$v" | awk -F': ' '{print $2}')
            if [[ "$node_version" != "$first_version" ]]; then
                consistent=false
                break
            fi
        done
        
        if [[ "$consistent" == true ]]; then
            log_info "✅ CANN 版本一致: $first_version"
        else
            log_error "❌ CANN 版本不一致！"
            log_error "   请统一所有节点的 CANN 版本"
            ((ERRORS++))
        fi
    fi
    echo ""
}

# 3. NPU 健康状态检查
check_npu_health() {
    log_check "检查 NPU 健康状态..."
    
    for node in "${NODES[@]}"; do
        local has_error=false
        local output
        
        # 获取 NPU 数量
        local npu_count=$(ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no ${SSH_USER}@$node \
            "npu-smi info -t info 2>/dev/null | grep -c 'NPU Name'" 2>/dev/null || echo "0")
        
        if [[ "$npu_count" == "0" ]] || [[ -z "$npu_count" ]]; then
            log_warn "⚠️  $node 无法获取 NPU 信息"
            ((WARNINGS++))
            continue
        fi
        
        for ((i=0; i<npu_count; i++)); do
            output=$(ssh -o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no ${SSH_USER}@$node \
                "npu-smi info -t health -i $i 2>/dev/null" 2>/dev/null || echo "CHECK_FAILED")
            
            if [[ "$output" == "CHECK_FAILED" ]]; then
                log_warn "⚠️  $node NPU $i 无法检查健康状态"
                has_error=true
                continue
            fi
            
            local health=$(echo "$output" | grep "Health" | awk '{print $2}' || echo "UNKNOWN")
            
            case "$health" in
                "OK")
                    # 正常，不输出（减少信息噪音）
                    ;;
                "Alarm")
                    log_error "❌ $node NPU $i 状态异常: Alarm"
                    has_error=true
                    ((ERRORS++))
                    ;;
                "Offline")
                    log_error "❌ $node NPU $i 状态异常: Offline"
                    has_error=true
                    ((ERRORS++))
                    ;;
                *)
                    log_warn "⚠️  $node NPU $i 状态未知: $health"
                    has_error=true
                    ((WARNINGS++))
                    ;;
            esac
        done
        
        if [[ "$has_error" == false ]]; then
            log_info "✅ $node 所有 NPU 状态正常 ($npu_count 张卡)"
        fi
    done
    echo ""
}

# 4. 网络连通性检查
check_network() {
    log_check "检查网络连通性..."
    
    for node in "${NODES[@]}"; do
        if ping -c 3 -W 3 $node > /dev/null 2&<1; then
            local avg_time=$(ping -c 3 -W 3 $node 2>/dev/null | tail -1 | awk -F'/' '{print $5}')
            log_info "✅ $node 网络正常 (延迟: ${avg_time}ms)"
        else
            log_error "❌ $node 网络不通"
            ((ERRORS++))
        fi
    done
    echo ""
}

# 5. MPI 环境检查（本地）
check_mpi() {
    log_check "检查本地 MPI 环境..."
    
    if command -v mpirun > /dev/null 2&<1; then
        local mpi_version=$(mpirun --version 2>&1 | head -1)
        log_info "✅ MPI 已安装: $mpi_version"
    else
        log_error "❌ 未找到 mpirun，请先安装 MPI"
        log_error "   参考: https://www.mpich.org/downloads/"
        ((ERRORS++))
    fi
    
    # 检查 HCCL Test 工具
    if [[ -d "${CANN_PATH}/tools/hccl_test" ]]; then
        log_info "✅ HCCL Test 工具目录存在"
    else
        log_error "❌ HCCL Test 工具目录不存在: ${CANN_PATH}/tools/hccl_test"
        log_error "   请检查 CANN 安装路径"
        ((ERRORS++))
    fi
    echo ""
}

# 主函数
main() {
    check_ssh
    check_cann_version
    check_npu_health
    check_network
    check_mpi
    
    # 输出汇总
    echo "=========================================="
    echo "检查完成"
    echo "=========================================="
    
    if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
        echo -e "${GREEN}✅ 所有检查通过，可以执行 HCCL 测试${NC}"
        exit 0
    elif [[ $ERRORS -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  检查通过，但有 $WARNINGS 个警告${NC}"
        echo -e "${YELLOW}   建议解决警告后再进行测试${NC}"
        exit 0
    else
        echo -e "${RED}❌ 检查发现 $ERRORS 个错误，$WARNINGS 个警告${NC}"
        echo -e "${RED}   请解决错误后再进行测试${NC}"
        exit 1
    fi
}

main
