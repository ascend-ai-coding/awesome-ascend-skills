#!/bin/bash
# preflight_check.sh - 训练前综合检查

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ==========================================
# 解析参数
# ==========================================
MODEL_SIZE="8B"
MODEL_PATH=""
TRAIN_DATA=""
VAL_DATA=""
CONTAINER_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --train) TRAIN_DATA="$2"; shift 2 ;;
        --val) VAL_DATA="$2"; shift 2 ;;
        --container) CONTAINER_NAME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ==========================================
# 开始检查
# ==========================================
echo "=========================================="
echo "训练前综合检查"
echo "=========================================="

# 1. NPU 检测
echo ""
echo "--- NPU 检测 ---"
if command -v npu-smi &>/dev/null; then
    NPU_INFO=$(npu-smi info 2>/dev/null || echo "")
    NPU_COUNT=$(echo "$NPU_INFO" | grep -c "910B\|Ascend 910" || echo "0")
    
    if [[ "$NPU_INFO" == *"910B"* ]]; then
        ok "NPU 类型: A2 (910B), 检测到 ${NPU_COUNT} 张"
    elif [[ "$NPU_INFO" == *"Ascend 910"* ]]; then
        ok "NPU 类型: A3, 检测到 ${NPU_COUNT} 张"
    else
        warn "无法检测 NPU 类型"
    fi
else
    warn "npu-smi 不可用，跳过 NPU 检测"
fi

# 2. 路径验证
echo ""
if [ -z "$MODEL_PATH" ]; then
    case "$MODEL_SIZE" in
        0.6B|0.6) MODEL_PATH="$DEFAULT_MODEL_06B" ;;
        *)        MODEL_PATH="$DEFAULT_MODEL_8B" ;;
    esac
fi
[ -z "$TRAIN_DATA" ] && TRAIN_DATA="$DEFAULT_TRAIN_DATA"
[ -z "$VAL_DATA" ] && VAL_DATA="$DEFAULT_VAL_DATA"

validate_all_paths "$MODEL_PATH" "$TRAIN_DATA" "$VAL_DATA"
PATH_ERRORS=$?

# 3. Docker 检查
echo ""
echo "--- Docker 检查 ---"
if command -v docker &>/dev/null; then
    # 查找 verl 容器
    CONTAINERS=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "verl|jins" || echo "")
    
    if [ -n "$CONTAINER_NAME" ]; then
        if docker ps --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
            ok "指定容器存在: $CONTAINER_NAME"
        else
            fail "指定容器不存在: $CONTAINER_NAME"
        fi
    elif [ -n "$CONTAINERS" ]; then
        CONTAINER_COUNT=$(echo "$CONTAINERS" | wc -l)
        if [ "$CONTAINER_COUNT" -eq 1 ]; then
            ok "唯一容器: $(echo $CONTAINERS)"
        else
            warn "多个容器: $(echo $CONTAINERS | tr '\n' ' ')"
            info "使用 --container 指定容器"
        fi
    else
        warn "未找到 verl 容器，需要创建"
    fi
else
    warn "Docker 不可用"
fi

# ==========================================
# 结果
# ==========================================
echo ""
echo "=========================================="
if [ "$PATH_ERRORS" -eq 0 ]; then
    ok "所有检查通过! 可以启动训练"
else
    fail "有 ${PATH_ERRORS} 个路径错误，请检查后重试"
fi
echo "=========================================="