# lib/k8s_utils.sh — kubectl 执行封装，支持本地和远程模式，RBAC 错误捕获

# 构建 SSH 命令字符串
build_ssh_cmd() {
    local ssh_cmd="ssh"
    [[ -n "$REMOTE_USER" ]] && ssh_cmd="$ssh_cmd $REMOTE_USER@$REMOTE_HOST" || ssh_cmd="$ssh_cmd $REMOTE_HOST"
    [[ -n "$REMOTE_KEY" ]] && ssh_cmd="$ssh_cmd -i $REMOTE_KEY"
    echo "$ssh_cmd"
}

# 执行 kubectl（本地或远程）
run_kubectl() {
    if [[ "$REMOTE_MODE" == "true" ]]; then
        local ssh_cmd=$(build_ssh_cmd)
        local remote_cmd="kubectl"
        for arg in "$@"; do
            remote_cmd="$remote_cmd $(printf "%q" "$arg")"
        done
        $ssh_cmd "$remote_cmd"
    else
        kubectl "$@"
    fi
}

# 安全执行 kubectl，捕获 RBAC 等错误并输出 JSON
safe_kc() {
    local output
    if output=$("$@" 2>&1); then
        echo "$output"
    else
        local rc=$?
        if echo "$output" | grep -qi "forbidden\|unauthorized\|RBAC"; then
            jq -n --arg msg "RBAC 权限不足: $output" '{rbac_error: $msg}'
        else
            jq -n --arg msg "$output" --argjson rc "$rc" '{error: $msg, exit_code: $rc}'
        fi
    fi
}

# 构建 kubectl 基础命令（带 context/namespace）
kc() {
    local cmd=(run_kubectl)
    [[ -n "$CONTEXT" ]] && cmd+=(--context "$CONTEXT")
    [[ -n "$NAMESPACE" ]] && cmd+=(--namespace "$NAMESPACE")
    "${cmd[@]}" "$@"
}

# 类似 kc，但未指定 namespace 时自动使用 --all-namespaces
kc_all() {
    local cmd=(run_kubectl)
    [[ -n "$CONTEXT" ]] && cmd+=(--context "$CONTEXT")
    if [[ -n "$NAMESPACE" ]]; then
        cmd+=(--namespace "$NAMESPACE")
    else
        cmd+=(--all-namespaces)
    fi
    "${cmd[@]}" "$@"
}