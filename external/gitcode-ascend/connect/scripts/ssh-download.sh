#!/usr/bin/env bash
# ssh-download.sh — 从远程下载文件/目录
# 用法: bash ssh-download.sh <profile_name> <remote_path> <local_path>
# 容器场景：先从容器 cp 到宿主机，再下载
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
PROFILE="${1:?用法: bash ssh-download.sh <profile_name> <remote_path> <local_path>}"
REMOTE_PATH="${2:?缺少 remote_path}"
LOCAL_PATH="${3:?缺少 local_path}"

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
do_download() {
    local src_remote="$1" dst="$2"
    local ssh_transport="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
    [ -n "$IDENTITY" ] && ssh_transport="$ssh_transport -i $IDENTITY"

    if command -v rsync &>/dev/null; then
        local rsync_cmd="rsync -avz -e \"$ssh_transport\""
        [ -n "$PASSWORD" ] && { export SSHPASS="$PASSWORD"; rsync_cmd="sshpass -e $rsync_cmd"; }
        eval "$rsync_cmd \"$USERNAME@$HOST:$src_remote\" \"$dst\""
    else
        local scp_cmd="scp -P $PORT -o StrictHostKeyChecking=accept-new"
        [ -n "$IDENTITY" ] && scp_cmd="$scp_cmd -i $IDENTITY"
        scp_cmd="$scp_cmd -r"  # always use -r for safety
        [ -n "$PASSWORD" ] && { export SSHPASS="$PASSWORD"; scp_cmd="sshpass -e $scp_cmd"; }
        eval "$scp_cmd \"$USERNAME@$HOST:$src_remote\" \"$dst\""
    fi
}

if [ -n "$CONTAINER" ]; then
    TMP_REMOTE="/tmp/_ssh_download_$(basename "$REMOTE_PATH")"
    if [ "$RUNTIME" = "kubectl" ]; then
        run_host_ssh "kubectl cp $CONTAINER:$REMOTE_PATH $TMP_REMOTE"
    else
        run_host_ssh "docker cp $CONTAINER:$REMOTE_PATH $TMP_REMOTE"
    fi
    do_download "$TMP_REMOTE" "$LOCAL_PATH"
    run_host_ssh "rm -rf $TMP_REMOTE"
else
    do_download "$REMOTE_PATH" "$LOCAL_PATH"
fi

echo "下载完成: $REMOTE_PATH → $LOCAL_PATH"
