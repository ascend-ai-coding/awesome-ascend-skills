#!/bin/bash
# ============================================================================
# pre_check.sh — 环境预检查（简化版）
# 检测 NPU、Docker 镜像、模型权重、数据集、网络信息
# 输出检测摘要，供 SKILL.md 的 Claude 读取
# ============================================================================

set -euo pipefail

echo "=========================================="
echo "Verl 训练环境预检查"
echo "=========================================="

# ---- NPU ----
echo ""
echo "[NPU 信息]"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>/dev/null | head -30 || echo "  NPU 检测失败"
    NPU_COUNT=$(npu-smi info 2>/dev/null | grep -c "NPU" || echo "0")
    echo "  检测到 $NPU_COUNT 张 NPU"
else
    echo "  npu-smi 命令不可用（可能不在 NPU 机器上）"
fi

# ---- Docker ----
echo ""
echo "[Docker 信息]"
if command -v docker >/dev/null 2>&1; then
    echo "  已有 Verl 容器:"
    docker ps -a 2>/dev/null | grep -E "verl|ascend" || echo "    (无)"
    echo ""
    echo "  本地 Verl 镜像:"
    docker images 2>/dev/null | grep -E "verl|ascend" || echo "    (无)"
else
    echo "  docker 命令不可用"
fi

# ---- 模型权重 ----
echo ""
echo "[模型权重]"
for dir in /mnt/public /mnt2 /mnt/project; do
    if [ -d "$dir" ]; then
        found=$(find "$dir" -maxdepth 4 -type d -name "Qwen*" 2>/dev/null | head -10)
        if [ -n "$found" ]; then
            echo "$found"
        fi
    fi
done

# ---- 数据集 ----
echo ""
echo "[数据集 (.parquet)]"
for dir in /mnt /mnt2; do
    if [ -d "$dir" ]; then
        find "$dir" -name "*.parquet" 2>/dev/null | head -10 || true
    fi
done

# ---- 网络信息 ----
echo ""
echo "[网络信息]"
echo "  本机 IP: $(hostname -I 2>/dev/null | awk '{print $1}')"
echo "  默认网卡: $(grep -Po '^(\S+)(?=\s+00000000)' /proc/net/route 2>/dev/null | head -1)"

echo ""
echo "=========================================="
echo "预检查完成"
echo "=========================================="
