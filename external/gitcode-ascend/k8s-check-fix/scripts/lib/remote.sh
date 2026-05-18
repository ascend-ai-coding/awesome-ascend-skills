# lib/remote.sh — 远程 SSH 预检和连接测试

check_remote_connectivity() {
    local ssh_cmd=$(build_ssh_cmd)
    if ! $ssh_cmd "echo ok" &>/dev/null; then
        err "无法连接到远程主机 $REMOTE_HOST"
        exit 1
    fi
    if ! $ssh_cmd "command -v kubectl" &>/dev/null; then
        err "远程主机 $REMOTE_HOST 上未找到 kubectl"
        exit 127
    fi
}