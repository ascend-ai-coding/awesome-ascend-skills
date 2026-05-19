#!/usr/bin/env bash
# ssh-upload.sh — 上传文件/目录到远程
# 用法: bash ssh-upload.sh <profile_name> <local_path> <remote_path>
# 容器场景：先传到宿主机 /tmp，再 docker cp 进容器
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
PROFILE="${1:?用法: bash ssh-upload.sh <profile_name> <local_path> <remote_path>}"
LOCAL_PATH="${2:?缺少 local_path}"
REMOTE_PATH="${3:?缺少 remote_path}"

CONFIG_JSON=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE")
SSH_OPTS=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE" --ssh-opts)
PASSWORD=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('password',''))")
CONTAINER=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('container',''))")
RUNTIME=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('container_runtime','docker'))")
HOST=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('host',''))")
PORT=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('port',22))")
USERNAME=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('username',''))")
IDENTITY=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('identity_file',''))")

mkdir -p ~/.ssh/sockets

# 在宿主机上执行命令（处理密码认证）
run_host_ssh() {
    if [ -n "$PASSWORD" ]; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=accept-new $SSH_OPTS "$@"
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $SSH_OPTS "$@"
    fi
}

# 文件传输：优先 rsync，回退 scp
do_upload() {
    local src="$1" dst_remote="$2"
    local ssh_transport="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
    [ -n "$IDENTITY" ] && ssh_transport="$ssh_transport -i $IDENTITY"

    if command -v rsync &>/dev/null; then
        local rsync_cmd="rsync -avz -e \"$ssh_transport\""
        [ -n "$PASSWORD" ] && { export SSHPASS="$PASSWORD"; rsync_cmd="sshpass -e $rsync_cmd"; }
        eval "$rsync_cmd \"$src\" \"$USERNAME@$HOST:$dst_remote\""
    else
        local scp_cmd="scp -P $PORT -o StrictHostKeyChecking=accept-new"
        [ -n "$IDENTITY" ] && scp_cmd="$scp_cmd -i $IDENTITY"
        [ -d "$src" ] && scp_cmd="$scp_cmd -r"
        [ -n "$PASSWORD" ] && { export SSHPASS="$PASSWORD"; scp_cmd="sshpass -e $scp_cmd"; }
        eval "$scp_cmd \"$src\" \"$USERNAME@$HOST:$dst_remote\""
    fi
}

if [ -n "$CONTAINER" ]; then
    TMP_REMOTE="/tmp/_ssh_upload_$(basename "$LOCAL_PATH")"
    do_upload "$LOCAL_PATH" "$TMP_REMOTE"
    if [ "$RUNTIME" = "kubectl" ]; then
        run_host_ssh "kubectl cp $TMP_REMOTE $CONTAINER:$REMOTE_PATH && rm -rf $TMP_REMOTE"
    else
        run_host_ssh "docker cp $TMP_REMOTE $CONTAINER:$REMOTE_PATH && rm -rf $TMP_REMOTE"
    fi
else
    do_upload "$LOCAL_PATH" "$REMOTE_PATH"
fi

echo "上传完成: $LOCAL_PATH → $REMOTE_PATH"
