#!/usr/bin/env bash
# k8s-check-fix.sh — 主入口，解析全局参数并路由到子命令

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/preflight.sh"
source "$SCRIPT_DIR/lib/remote.sh"
source "$SCRIPT_DIR/lib/k8s_utils.sh"

# 全局变量
CONTEXT=""
NAMESPACE=""
SINCE="15m"
TAIL="200"
CONFIRM=false
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_KEY=""
SUBCOMMAND=""
SUBCOMMAND_ARGS=()

# 解析参数
parse_global_args "$@"

if [[ -z "$SUBCOMMAND" ]]; then
    usage
    exit 0
fi

# 执行预检
preflight

# 路由到子命令
case "$SUBCOMMAND" in
    sweep)     source "$SCRIPT_DIR/subcommands/sweep.sh"; cmd_sweep ;;
    pod)       source "$SCRIPT_DIR/subcommands/pod.sh"; cmd_pod ;;
    deploy)    source "$SCRIPT_DIR/subcommands/deploy.sh"; cmd_deploy ;;
    resources) source "$SCRIPT_DIR/subcommands/resources.sh"; cmd_resources ;;
    events)    source "$SCRIPT_DIR/subcommands/events.sh"; cmd_events ;;
    fix)       source "$SCRIPT_DIR/subcommands/fix.sh"; cmd_fix ;;
    *)
        err "未知子命令: $SUBCOMMAND"
        exit 1
        ;;
esac