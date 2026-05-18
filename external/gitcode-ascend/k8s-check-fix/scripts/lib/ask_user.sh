#!/usr/bin/env bash
# scripts/lib/ask_user.sh — 结构化提问辅助函数
#
# 本文件提供用于生成结构化提问的函数，供 AI 模型在需要收集用户信息时调用。
# 这些函数输出符合 AskUserQuestion 工具要求的 JSON 格式，模型可将其作为工具调用参数。
#
# 注意：这些函数不会直接执行提问，而是生成提问请求的 JSON 描述。
# 实际调用 AskUserQuestion 工具由模型在运行时完成。

# 构建单选/多选问题
# 用法: ask_choice <question_id> <title> <description> <multiselect> <option1> [option2 ...]
# 参数:
#   question_id   - 问题唯一标识
#   title         - 问题标题（简短）
#   description   - 问题详细描述
#   multiselect   - 是否多选 (true/false)
#   后续参数      - 选项列表
# 输出: JSON 格式的提问对象
ask_choice() {
    local id="$1"
    local title="$2"
    local description="$3"
    local multiselect="$4"
    shift 4
    local options=("$@")

    # 构建选项数组 JSON
    local options_json="["
    local first=true
    for opt in "${options[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
            options_json+="\"$opt\""
        else
            options_json+=", \"$opt\""
        fi
    done
    options_json+="]"

    jq -n \
        --arg id "$id" \
        --arg title "$title" \
        --arg description "$description" \
        --argjson multiselect "$multiselect" \
        --argjson options "$options_json" \
        '{id: $id, title: $title, description: $description, multiselect: $multiselect, options: $options}'
}

# 构建文本输入问题
# 用法: ask_text <question_id> <title> <description> <placeholder> [default]
# 参数:
#   question_id   - 问题唯一标识
#   title         - 问题标题
#   description   - 问题详细描述
#   placeholder   - 输入框占位符
#   default       - 默认值（可选）
# 输出: JSON 格式的提问对象
ask_text() {
    local id="$1"
    local title="$2"
    local description="$3"
    local placeholder="$4"
    local default="${5:-}"

    if [[ -n "$default" ]]; then
        jq -n \
            --arg id "$id" \
            --arg title "$title" \
            --arg description "$description" \
            --arg placeholder "$placeholder" \
            --arg default "$default" \
            '{id: $id, title: $title, description: $description, placeholder: $placeholder, default: $default}'
    else
        jq -n \
            --arg id "$id" \
            --arg title "$title" \
            --arg description "$description" \
            --arg placeholder "$placeholder" \
            '{id: $id, title: $title, description: $description, placeholder: $placeholder}'
    fi
}

# 构建一组问题的提问请求
# 用法: ask_user <question_json1> [question_json2 ...]
# 输出: 符合 AskUserQuestion 工具的 JSON 请求
ask_user() {
    local questions=()
    for q in "$@"; do
        questions+=("$q")
    done
    jq -n --argjson questions "$(printf '%s\n' "${questions[@]}" | jq -s '.')" '{questions: $questions}'
}

# 常用预设问题模板（用于快速集成）

# 询问默认 Kubernetes 上下文
# 参数: contexts_json - 从 kubectl config get-contexts -o name 获取的 JSON 数组
ask_default_context() {
    local contexts_json="$1"
    local options=()
    while IFS= read -r ctx; do
        options+=("$ctx")
    done <<< "$(echo "$contexts_json" | jq -r '.[]')"
    ask_choice "default_context" "默认 Kubernetes 上下文" "选择默认使用的集群上下文" false "${options[@]}"
}

# 询问默认命名空间
ask_default_namespace() {
    ask_text "default_namespace" "默认命名空间" "指定默认操作的命名空间（留空表示所有命名空间）" "例如: default, production" ""
}

# 询问只读模式
ask_readonly_mode() {
    ask_choice "readonly_mode" "只读模式" "开启后将禁止所有写操作（fix 命令）" false "开启（推荐）" "关闭"
}

# 询问自动确认修复（危险）
ask_auto_confirm() {
    ask_choice "auto_confirm" "自动确认修复" "开启后，所有写操作将自动执行，不再询问用户（极危险）" false "关闭（强烈推荐）" "开启"
}

# 询问危险操作前是否二次确认
ask_ask_before_dangerous() {
    ask_choice "ask_before_dangerous" "危险操作前确认" "执行可能影响集群的操作（如删除节点、驱逐 Pod）前，是否再次询问用户" false "是（推荐）" "否"
}

# 询问远程执行默认配置
ask_remote_defaults() {
    local remote_host=$(ask_text "remote_host" "远程主机地址" "用于远程执行 kubectl 的 SSH 主机地址（格式：user@host）" "user@bastion.example.com")
    local remote_user=$(ask_text "remote_user" "SSH 用户名" "SSH 登录用户名（如果已在主机地址中包含，可留空）" "yourname")
    local remote_key=$(ask_text "remote_key" "SSH 私钥路径" "SSH 私钥文件路径" "~/.ssh/id_rsa")
    # 返回一个组合的 JSON 对象，实际使用时可分别调用，这里仅示意
    echo "$remote_host"
    echo "$remote_user"
    echo "$remote_key"
}

# 示例：生成完整的初始配置提问
# 该函数生成一组问题，用于首次运行时收集用户配置
ask_initial_config() {
    local contexts_json="$1"  # 期望是一个 JSON 数组，如 `["context1","context2"]`
    local q1 q2 q3 q4 q5
    q1=$(ask_default_context "$contexts_json")
    q2=$(ask_default_namespace)
    q3=$(ask_readonly_mode)
    q4=$(ask_auto_confirm)
    q5=$(ask_ask_before_dangerous)
    ask_user "$q1" "$q2" "$q3" "$q4" "$q5"
}

# 使用示例（模型可以这样调用）：
#   contexts=$(kubectl config get-contexts -o name | jq -R -s 'split("\n")[:-1]')
#   ask_initial_config "$contexts"