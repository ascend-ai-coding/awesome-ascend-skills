#!/bin/bash

# SHA256 校验脚本
# 用于校验已下载 ModelScope 模型或数据集的完整性

set -e

DIR=${1:-""}

# 检查参数
if [ -z "$DIR" ]; then
    echo "用法: $0 <目录路径>"
    echo "示例: $0 ./Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp"
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
    echo "错误: 目录不存在: $DIR"
    exit 1
fi

SHA256_FILE="${DIR}/.sha256sum"

# 检查是否已存在校验文件
if [ ! -f "$SHA256_FILE" ]; then
    echo "校验文件不存在，正在生成..."
    echo "生成位置: $SHA256_FILE"
    sha256sum "$DIR"/* > "$SHA256_FILE"
    echo "校验文件已生成，包含 $(wc -l < "$SHA256_FILE") 个文件的校验值"
fi

# 执行校验
echo "开始校验文件完整性..."
echo "校验目录: $DIR"
echo "校验文件: $SHA256_FILE"
echo "----------------------------------------"

if sha256sum -c "$SHA256_FILE" 2>/dev/null; then
    echo "----------------------------------------"
    echo "✅ 所有文件校验通过！"
    exit 0
else
    echo "----------------------------------------"
    echo "❌ 部分文件校验失败，请检查下载完整性"
    exit 1
fi
