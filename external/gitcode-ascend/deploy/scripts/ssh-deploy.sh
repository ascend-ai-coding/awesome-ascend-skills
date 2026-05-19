#!/usr/bin/env bash
# ssh-deploy.sh — 部署本地项目到远程服务器
# 用法:
#   bash ssh-deploy.sh deploy          # 执行部署
#   bash ssh-deploy.sh rollback        # 回滚到上一版本
# 依赖: $PYTHON, rsync, ssh-connect 套件
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
CONFIG_FILE="$SUITE_DIR/config.yaml"

ACTION="${1:-deploy}"

# 读取部署配置
read_config() {
    $PYTHON -c "
import yaml, json, sys
config = yaml.safe_load(open('$CONFIG_FILE', encoding='utf-8'))
deploy = config.get('deploy', {})
print(json.dumps(deploy, ensure_ascii=False))
"
}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE" >&2
    echo "请让 agent 引导你配置部署信息" >&2
    exit 1
fi

DEPLOY_JSON=$(read_config)
PROFILE=$(echo "$DEPLOY_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['profile'])")
LOCAL_PATH=$(echo "$DEPLOY_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['local_path'])")
REMOTE_PATH=$(echo "$DEPLOY_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['remote_path'])")
SYNC_MODE=$(echo "$DEPLOY_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('sync_mode','incremental'))")

run_remote() {
    bash "$SUITE_DIR/connect/scripts/ssh-exec.sh" "$PROFILE" "$@"
}

run_commands() {
    local stage="$1"
    local commands
    commands=$(echo "$DEPLOY_JSON" | $PYTHON -c "
import sys, json
cmds = json.load(sys.stdin).get('$stage', [])
for c in cmds:
    print(c)
")
    if [ -z "$commands" ]; then return 0; fi

    echo "--- $stage ---"
    while IFS= read -r cmd; do
        echo "执行: $cmd"
        if ! run_remote "$cmd"; then
            echo "✗ 命令失败: $cmd" >&2
            exit 1
        fi
    done <<< "$commands"
}

case "$ACTION" in
    deploy)
        echo "=== 部署开始: $PROFILE → $REMOTE_PATH ==="
        run_commands "pre_deploy"

        echo "--- 文件同步 ($SYNC_MODE) ---"
        bash "$SUITE_DIR/connect/scripts/ssh-upload.sh" "$PROFILE" "$LOCAL_PATH" "$REMOTE_PATH"

        run_commands "post_deploy"
        echo "=== 部署完成 ==="
        ;;

    rollback)
        echo "=== 回滚开始: $PROFILE ==="
        run_commands "rollback"
        echo "=== 回滚完成 ==="
        ;;

    *)
        echo "未知操作: $ACTION (可用: deploy, rollback)" >&2
        exit 1
        ;;
esac
