# subcommands/fix.sh — 安全修复执行（严格白名单 + 用户确认）

ALLOWLIST=(
    "rollout undo"
    "rollout restart"
    "scale"
    "delete pod"
    "cordon"
    "uncordon"
)

cmd_fix() {
    if [[ ${#SUBCOMMAND_ARGS[@]} -eq 0 ]]; then
        err "用法: $0 fix '<kubectl 命令>' [--confirm]"
        exit 1
    fi
    local cmd_str="${SUBCOMMAND_ARGS[0]}"

    if [[ "$CONFIRM" != "true" ]]; then
        err "写操作需要 --confirm 标志。请向用户展示命令并获得确认后重试。"
        exit 1
    fi

    # 禁止 shell 元字符
    if [[ "$cmd_str" =~ [;|&$`(] ]]; then
        err "命令包含非法字符"
        exit 1
    fi

    # 解析命令
    local cmd_parts=()
    read -ra cmd_parts <<< "$cmd_str"
    if [[ "${cmd_parts[0]}" != "kubectl" ]]; then
        err "只允许执行 kubectl 命令"
        exit 1
    fi

    local verb="${cmd_parts[1]:-}"
    local resource="${cmd_parts[2]:-}"
    local allowed=false
    for pattern in "${ALLOWLIST[@]}"; do
        if [[ "${verb} ${resource}" == "$pattern" ]]; then
            allowed=true
            break
        fi
        # 特殊处理 scale（后面可跟 deployment）
        if [[ "$verb" == "scale" ]] && [[ "$resource" == "deployment" ]]; then
            allowed=true
            break
        fi
    done

    if [[ "$allowed" != "true" ]]; then
        err "命令 '$verb $resource' 不在白名单中。允许的操作: ${ALLOWLIST[*]}"
        exit 1
    fi

    # 执行
    local output
    if [[ "$REMOTE_MODE" == "true" ]]; then
        output=$(run_kubectl "${cmd_parts[@]:1}" 2>&1) || true
    else
        output=$("${cmd_parts[@]}" 2>&1) || true
    fi

    local data
    data=$(jq -n --arg cmd "$cmd_str" --arg output "$output" '{command_executed: $cmd, output: $output}')
    json_envelope "ok" "fix" "$data"
}