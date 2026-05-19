#!/usr/bin/env bash
# ssh-exec.sh — 远程执行命令
# 用法: bash ssh-exec.sh <profile_name> <command>
# 容器场景自动包装为 docker exec / kubectl exec
# 返回: 远程命令的退出码

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
PROFILE="${1:?用法: bash ssh-exec.sh <profile_name> <command>}"
shift
COMMAND="$*"

CONFIG_JSON=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE")
SSH_OPTS=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE" --ssh-opts)
PASSWORD=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('password',''))")
CONTAINER=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('container',''))")
RUNTIME=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('container_runtime','docker'))")

mkdir -p ~/.ssh/sockets

# 容器场景：包装命令（使用 printf %q 安全转义，避免引号嵌套问题）
if [ -n "$CONTAINER" ]; then
    ESCAPED_CMD=$(printf '%q' "$COMMAND")
    if [ "$RUNTIME" = "kubectl" ]; then
        COMMAND="kubectl exec -i $CONTAINER -- bash -c $ESCAPED_CMD"
    else
        COMMAND="docker exec -i $CONTAINER bash -c $ESCAPED_CMD"
    fi
fi

# 构造并执行
if [ -n "$PASSWORD" ]; then
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=accept-new $SSH_OPTS "$COMMAND"
else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $SSH_OPTS "$COMMAND"
fi
