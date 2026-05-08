#!/usr/bin/env bash
# scripts/lib/config_loader.sh — 配置文件读取与写入

# 设置 config.json 路径（假设位于技能根目录，即 ../.. 相对于此脚本）
CONFIG_LOADER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$CONFIG_LOADER_SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$SKILL_ROOT/config.json"

# 默认配置值（用于缺失时填充）
DEFAULT_CONFIG='{
  "default_context": "",
  "default_namespace": "",
  "readonly_mode": false,
  "auto_confirm": false,
  "ask_before_dangerous": true,
  "remote_defaults": {
    "host": "",
    "user": "",
    "key_path": ""
  }
}'

# 检查 config.json 是否存在，若不存在则创建默认配置
ensure_config_exists() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
        echo "已创建默认配置文件: $CONFIG_FILE" >&2
    fi
}

# 读取整个配置并输出为 JSON 字符串
read_config() {
    ensure_config_exists
    jq '.' "$CONFIG_FILE"
}

# 获取配置中的特定字段（支持 jq 路径）
# 用法: get_config_value ".default_context"
get_config_value() {
    local path="$1"
    ensure_config_exists
    jq -r "$path" "$CONFIG_FILE"
}

# 检查配置是否完整（必要字段不为空或符合预期）
# 必要字段: default_context, readonly_mode, auto_confirm, ask_before_dangerous
check_config_complete() {
    ensure_config_exists
    local missing=()
    # 检查 default_context 是否为空字符串（允许空）
    local default_context
    default_context=$(get_config_value ".default_context")
    # 检查其他布尔字段是否存在
    if ! jq -e '.readonly_mode' "$CONFIG_FILE" >/dev/null 2>&1; then
        missing+=("readonly_mode")
    fi
    if ! jq -e '.auto_confirm' "$CONFIG_FILE" >/dev/null 2>&1; then
        missing+=("auto_confirm")
    fi
    if ! jq -e '.ask_before_dangerous' "$CONFIG_FILE" >/dev/null 2>&1; then
        missing+=("ask_before_dangerous")
    fi
    # 如果 default_context 缺失或为空，不视为不完整，因为用户可能愿意使用当前上下文
    # 但可以提示
    if [[ -z "$default_context" ]]; then
        echo "WARN: default_context 未设置，将使用当前上下文" >&2
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "配置不完整，缺少字段: ${missing[*]}" >&2
        return 1
    fi
    return 0
}

# 写入完整配置（接受 JSON 字符串）
# 用法: write_config '{"default_context":"prod","readonly_mode":false,...}'
write_config() {
    local config_json="$1"
    if ! echo "$config_json" | jq . >/dev/null 2>&1; then
        echo "错误: 提供的配置不是有效的 JSON" >&2
        return 1
    fi
    echo "$config_json" > "$CONFIG_FILE"
    echo "配置文件已更新: $CONFIG_FILE" >&2
}

# 更新特定字段（jq 路径和值）
# 用法: update_config_field ".default_context" "prod"
# 注意：值会被当作 JSON 值处理，字符串需加引号，数字和布尔值不加。
update_config_field() {
    local path="$1"
    local value="$2"
    ensure_config_exists
    local new_config
    new_config=$(jq --argjson val "$value" "$path = \$val" "$CONFIG_FILE")
    if [[ $? -eq 0 ]]; then
        echo "$new_config" > "$CONFIG_FILE"
        echo "已更新 $path = $value" >&2
    else
        echo "更新失败: jq 错误" >&2
        return 1
    fi
}

# 导出配置为环境变量（方便主脚本使用）
# 用法: load_config_env
load_config_env() {
    ensure_config_exists
    export K8S_DEFAULT_CONTEXT=$(get_config_value ".default_context")
    export K8S_DEFAULT_NAMESPACE=$(get_config_value ".default_namespace")
    export K8S_READONLY_MODE=$(get_config_value ".readonly_mode")
    export K8S_AUTO_CONFIRM=$(get_config_value ".auto_confirm")
    export K8S_ASK_BEFORE_DANGEROUS=$(get_config_value ".ask_before_dangerous")
    export K8S_REMOTE_HOST=$(get_config_value ".remote_defaults.host")
    export K8S_REMOTE_USER=$(get_config_value ".remote_defaults.user")
    export K8S_REMOTE_KEY=$(get_config_value ".remote_defaults.key_path")
}