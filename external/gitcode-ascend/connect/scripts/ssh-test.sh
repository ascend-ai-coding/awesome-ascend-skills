#!/usr/bin/env bash
# ssh-test.sh — 测试 SSH 连接
# 用法: bash ssh-test.sh <profile_name>
# 依赖: $PYTHON, ssh, sshpass(密码认证时)
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
PROFILE="${1:?用法: bash ssh-test.sh <profile_name>}"

# 解析配置
CONFIG_JSON=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE")
SSH_OPTS=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE" --ssh-opts)
PASSWORD=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('password',''))")

# 确保 ControlPath 目录存在
mkdir -p ~/.ssh/sockets

# 构造 SSH 命令（不使用 ControlMaster，避免复用干扰测试）
SSH_CMD="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ControlMaster=no $SSH_OPTS"
if [ -n "$PASSWORD" ]; then
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=accept-new -o ControlMaster=no $SSH_OPTS"
fi

echo "=== SSH 连接测试: $PROFILE ==="
echo "--- 连通性测试 ---"
if eval "$SSH_CMD echo 'SSH_OK'" 2>&1 | grep -q 'SSH_OK'; then
    echo "✓ 连接成功"
else
    echo "✗ 连接失败" >&2
    exit 1
fi

echo "--- 远程环境信息 ---"
eval "$SSH_CMD 'uname -a && echo \"---\" && df -h / && echo \"---\" && free -m 2>/dev/null || true'"

echo ""
echo "=== 测试完成 ==="
