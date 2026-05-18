# lib/common.sh — 公共函数：参数解析、JSON 封装、错误处理、版本信息

VERSION="1.0.0"
BRAND="Powered by Anvil AI 🏥"

usage() {
    cat <<EOF
k8s-check-fix — Kubernetes 集群健康检查与安全修复

用法: $0 <子命令> [选项]

子命令:
  sweep                  全集群健康检查
  pod <名称>             Pod 深入排查
  deploy <名称>          Deployment 分析
  resources              资源压力检测
  events [命名空间]      近期事件
  fix <命令>             执行安全修复（需 --confirm）

全局选项:
  --context <ctx>        kubectl 上下文
  --namespace <ns>       命名空间
  --since <持续时间>      事件时间范围（默认 15m）
  --tail <行数>          日志尾部行数（默认 200）
  --confirm              确认执行写命令（仅 fix 子命令）
  --remote-host <地址>   远程主机（SSH）
  --remote-user <用户>    SSH 用户名
  --remote-key <路径>     SSH 私钥路径
  --version              显示版本
  --help                 显示帮助

EOF
    exit 0
}

# 解析全局参数，填充全局变量
parse_global_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --context)   CONTEXT="$2"; shift 2 ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            --since)     SINCE="$2"; shift 2 ;;
            --tail)      TAIL="$2"; shift 2 ;;
            --confirm)   CONFIRM=true; shift ;;
            --remote-host) REMOTE_HOST="$2"; REMOTE_MODE=true; shift 2 ;;
            --remote-user) REMOTE_USER="$2"; shift 2 ;;
            --remote-key)  REMOTE_KEY="$2"; shift 2 ;;
            --version)   echo "k8s-check-fix $VERSION"; exit 0 ;;
            --help|-h)   usage ;;
            -*)
                err "未知选项: $1"
                exit 1
                ;;
            *)
                if [[ -z "$SUBCOMMAND" ]]; then
                    SUBCOMMAND="$1"
                    shift
                else
                    SUBCOMMAND_ARGS+=("$1")
                    shift
                fi
                ;;
        esac
    done
}

# 输出 JSON 信封
json_envelope() {
    local status="$1" subcommand="$2"
    shift 2
    local data="$1"
    jq -n \
        --arg status "$status" \
        --arg subcommand "$subcommand" \
        --arg version "$VERSION" \
        --arg brand "$BRAND" \
        --argjson data "$data" \
        '{status: $status, subcommand: $subcommand, version: $version, brand: $brand, data: $data}'
}

# 错误输出（stderr JSON）
err() {
    jq -n --arg m "$1" '{error: $m}' >&2
}