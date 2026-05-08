# lib/preflight.sh — 本地/远程环境检查

preflight() {
    if [[ "$REMOTE_MODE" == "true" ]]; then
        if ! command -v ssh &>/dev/null; then
            err "本地未找到 ssh 命令，无法使用远程模式"
            exit 127
        fi
        check_remote_connectivity
    else
        if ! command -v kubectl &>/dev/null; then
            err "未找到 kubectl，请安装 https://kubernetes.io/docs/tasks/tools/"
            exit 127
        fi
        if ! command -v jq &>/dev/null; then
            err "未找到 jq，请安装 https://jqlang.github.io/jq/download/"
            exit 127
        fi
        if ! kubectl ${CONTEXT:+--context "$CONTEXT"} cluster-info --request-timeout=5s &>/dev/null 2>&1; then
            err "无法连接到 Kubernetes 集群，请检查 kubeconfig 和上下文"
            exit 1
        fi
    fi
}