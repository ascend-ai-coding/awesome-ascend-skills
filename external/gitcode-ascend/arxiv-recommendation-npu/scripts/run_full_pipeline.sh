#!/bin/bash
# 完整流水线脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SKILL_DIR"

# 默认搜索最近3天，可通过第一个参数指定天数
DAYS="${1:-3}"

echo "========================================"
echo "arxiv-recommendation-npu 流水线"
echo "搜索范围: 最近 ${DAYS} 天"
echo "========================================"

# 设置 NPU 环境
export ASCEND_VISIBLE_DEVICES=2,3
export ASCEND_RT_VISIBLE_DEVICES=2,3

# 执行流水线
python scripts/main.py --days "$DAYS"

echo "========================================"
echo "流水线执行完成"
echo "========================================"