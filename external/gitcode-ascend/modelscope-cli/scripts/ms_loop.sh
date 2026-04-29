#!/bin/bash

# ModelScope 循环重试脚本
# 失败时等待后自动重试，直到成功
#
# 使用方法:
#   bash ms_loop.sh <script_path> [retry_interval]
#
# 参数:
#   script_path    - 要执行的脚本路径
#   retry_interval - 重试间隔（秒），默认 5

SCRIPT_PATH=$1
RETRY_INTERVAL=${2:-5}

if [ -z "$SCRIPT_PATH" ]; then
    echo "用法: bash ms_loop.sh <script_path> [retry_interval]"
    echo ""
    echo "示例:"
    echo "  bash ms_loop.sh run_ms_model_download.sh"
    echo "  bash ms_loop.sh run_ms_model_download.sh 10  # 10秒后重试"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

echo "========================================"
echo "ModelScope 循环重试模式"
echo "========================================"
echo ""
echo "目标脚本: $SCRIPT_PATH"
echo "重试间隔: ${RETRY_INTERVAL}s"
echo ""

RETRY_COUNT=0
MAX_RETRY=100  # 最大重试次数

while true; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    echo "========================================"
    echo "第 ${RETRY_COUNT} 次尝试 ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "========================================"
    echo ""
    
    # 执行脚本
    set +e
    bash "$SCRIPT_PATH"
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "✅ 执行成功！"
        echo "========================================"
        echo "总尝试次数: $RETRY_COUNT"
        exit 0
    fi
    
    echo ""
    echo "❌ 执行失败，退出码: $EXIT_CODE"
    
    if [ $RETRY_COUNT -ge $MAX_RETRY ]; then
        echo ""
        echo "⚠️  已达到最大重试次数 ($MAX_RETRY)，退出"
        exit 1
    fi
    
    echo "将在 ${RETRY_INTERVAL}s 后重试..."
    echo ""
    
    sleep $RETRY_INTERVAL
done
