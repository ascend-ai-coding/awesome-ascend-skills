#!/usr/bin/env bash
# ssh-tunnel.sh — SSH 通道管理
# 用法:
#   bash ssh-tunnel.sh forward <profile> <local_port>:<remote_host>:<remote_port>
#   bash ssh-tunnel.sh reverse <profile> <remote_port>:<local_host>:<local_port>
#   bash ssh-tunnel.sh socks <profile> <local_port>
#   bash ssh-tunnel.sh proxy <profile> [--local-proxy-port 7890]
#   bash ssh-tunnel.sh list
#   bash ssh-tunnel.sh stop <tunnel_id>
#   bash ssh-tunnel.sh stop-all
# 依赖: $PYTHON, ssh, sshpass(密码认证时)
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
TUNNELS_FILE="$HOME/.ssh/tunnels.json"

# ── 工具函数 ──────────────────────────────────────────────

usage() {
    sed -n '3,10p' "$0" | sed 's/^# *//' >&2
    exit 1
}

die() { echo "错误: $*" >&2; exit 1; }

# 确保 tunnels.json 存在
ensure_tunnels_file() {
    mkdir -p "$(dirname "$TUNNELS_FILE")"
    [ -f "$TUNNELS_FILE" ] || echo '[]' > "$TUNNELS_FILE"
}

# 解析 profile，输出 JSON
resolve_profile() {
    local profile="$1"
    $PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$profile"
}

# 解析 profile，输出 ssh 命令行参数
resolve_ssh_opts() {
    local profile="$1"
    $PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$profile" --ssh-opts
}

# 从 profile JSON 提取 password
get_password() {
    local config_json="$1"
    echo "$config_json" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('password',''))"
}

# ── 运行 SSH（处理 sshpass 包装）────────────────────────

run_ssh() {
    local password="$1"
    shift
    # $@ = ssh 完整参数列表

    mkdir -p ~/.ssh/sockets

    if [ -n "$password" ]; then
        if ! command -v sshpass &>/dev/null; then
            die "profile 使用密码认证但未安装 sshpass"
        fi
        sshpass -p "$password" ssh -o StrictHostKeyChecking=accept-new "$@"
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@"
    fi
}

# ── 生成隧道 ID ──────────────────────────────────────────

generate_tunnel_id() {
    local type="$1" profile="$2"
    echo "${type}-${profile}-$$"
}

# ── 隧道记录管理（Python 单行）────────────────────────────

add_tunnel_record() {
    local tid="$1" type="$2" profile="$3" pid="$4" detail="$5"
    ensure_tunnels_file
    $PYTHON -c "
import json, sys
rec = {'id': sys.argv[1], 'type': sys.argv[2], 'profile': sys.argv[3],
       'pid': int(sys.argv[4]), 'detail': sys.argv[5]}
data = json.load(open(sys.argv[6]))
data.append(rec)
json.dump(data, open(sys.argv[6], 'w'), indent=2)
" "$tid" "$type" "$profile" "$pid" "$detail" "$TUNNELS_FILE"
}

remove_tunnel_record() {
    local tid="$1"
    ensure_tunnels_file
    $PYTHON -c "
import json, sys
data = json.load(open(sys.argv[2]))
data = [r for r in data if r['id'] != sys.argv[1]]
json.dump(data, open(sys.argv[2], 'w'), indent=2)
" "$tid" "$TUNNELS_FILE"
}

list_tunnel_records() {
    ensure_tunnels_file

    # 在 bash 层做 PID 存活检查（跨平台可靠），生成存活 PID 列表
    local alive_pids=""
    local all_pids
    all_pids=$($PYTHON -c "
import json, sys
data = json.load(open(sys.argv[1]))
for r in data:
    print(r['pid'])
" "$TUNNELS_FILE")

    for pid in $all_pids; do
        if kill -0 "$pid" 2>/dev/null; then
            alive_pids="$alive_pids $pid"
        fi
    done

    # Python 过滤 + 格式化显示，传入存活 PID 集合
    $PYTHON -c "
import json, sys
alive_set = set(int(p) for p in sys.argv[2].split() if p.strip())
data = json.load(open(sys.argv[1]))
if not data:
    print('当前无活跃隧道')
    sys.exit(0)
alive = [r for r in data if r['pid'] in alive_set]
json.dump(alive, open(sys.argv[1], 'w'), indent=2)
if not alive:
    print('当前无活跃隧道')
    sys.exit(0)
fmt = '  {:<28s} {:<10s} {:<14s} {:>7s}  {}'
print(fmt.format('ID', 'TYPE', 'PROFILE', 'PID', 'DETAIL'))
print('  ' + '-' * 78)
for r in alive:
    print(fmt.format(r['id'], r['type'], r['profile'], str(r['pid']), r['detail']))
" "$TUNNELS_FILE" "$alive_pids"
}

# ── 启动隧道并记录 PID ───────────────────────────────────
# 使用 ssh 不带 -f，手动后台运行以捕获 PID

start_tunnel() {
    local type="$1" profile="$2" detail="$3"
    shift 3
    # $@ = 隧道专用 ssh 参数（如 -L, -R, -D）

    local config_json ssh_opts password
    config_json=$(resolve_profile "$profile")
    ssh_opts=$(resolve_ssh_opts "$profile")
    password=$(get_password "$config_json")

    local tid
    tid=$(generate_tunnel_id "$type" "$profile")

    # 不使用 ssh -f，而是手动 & 后台运行以便捕获 PID
    # shellcheck disable=SC2086
    run_ssh "$password" -N "$@" $ssh_opts &
    local pid=$!

    # 简单等待确认进程启动
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        die "隧道启动失败（进程已退出）"
    fi

    add_tunnel_record "$tid" "$type" "$profile" "$pid" "$detail"
    echo "隧道已建立: id=$tid pid=$pid ($detail)"
}

# ── 子命令实现 ────────────────────────────────────────────

cmd_forward() {
    [ $# -ge 2 ] || die "用法: forward <profile> <local_port>:<remote_host>:<remote_port>"
    local profile="$1" spec="$2"

    local local_port remote_host remote_port
    IFS=':' read -r local_port remote_host remote_port <<< "$spec"
    [ -n "$local_port" ] && [ -n "$remote_host" ] && [ -n "$remote_port" ] \
        || die "格式错误，期望 local_port:remote_host:remote_port"

    start_tunnel "local" "$profile" "L ${local_port}→${remote_host}:${remote_port}" \
        -L "${local_port}:${remote_host}:${remote_port}"
}

cmd_reverse() {
    [ $# -ge 2 ] || die "用法: reverse <profile> <remote_port>:<local_host>:<local_port>"
    local profile="$1" spec="$2"

    local remote_port local_host local_port
    IFS=':' read -r remote_port local_host local_port <<< "$spec"
    [ -n "$remote_port" ] && [ -n "$local_host" ] && [ -n "$local_port" ] \
        || die "格式错误，期望 remote_port:local_host:local_port"

    start_tunnel "reverse" "$profile" "R ${remote_port}→${local_host}:${local_port}" \
        -R "${remote_port}:${local_host}:${local_port}"
}

cmd_socks() {
    [ $# -ge 2 ] || die "用法: socks <profile> <local_port>"
    local profile="$1" local_port="$2"

    start_tunnel "socks" "$profile" "D ${local_port}" \
        -D "$local_port"
}

cmd_proxy() {
    local profile="" local_proxy_port="7890"

    while [ $# -gt 0 ]; do
        case "$1" in
            --local-proxy-port)
                local_proxy_port="${2:?--local-proxy-port 需要参数}"
                shift 2 ;;
            -*)
                die "未知选项: $1" ;;
            *)
                [ -z "$profile" ] && profile="$1" || die "多余参数: $1"
                shift ;;
        esac
    done
    [ -n "$profile" ] || die "用法: proxy <profile> [--local-proxy-port 7890]"

    # 获取远程端口（默认与本地一致）
    local remote_proxy_port="$local_proxy_port"

    start_tunnel "reverse_proxy" "$profile" \
        "R ${remote_proxy_port}→localhost:${local_proxy_port}" \
        -R "${remote_proxy_port}:localhost:${local_proxy_port}"

    echo ""
    echo "在远程服务器上执行以下命令以启用代理:"
    echo "  export http_proxy=http://127.0.0.1:${remote_proxy_port}"
    echo "  export https_proxy=http://127.0.0.1:${remote_proxy_port}"
    echo "  export HTTP_PROXY=http://127.0.0.1:${remote_proxy_port}"
    echo "  export HTTPS_PROXY=http://127.0.0.1:${remote_proxy_port}"
}

cmd_stop() {
    [ $# -ge 1 ] || die "用法: stop <tunnel_id>"
    local tid="$1"
    ensure_tunnels_file

    local pid
    pid=$($PYTHON -c "
import json, sys
data = json.load(open(sys.argv[2]))
matches = [r for r in data if r['id'] == sys.argv[1]]
if not matches:
    print('', end='')
else:
    print(matches[0]['pid'], end='')
" "$tid" "$TUNNELS_FILE")

    [ -n "$pid" ] || die "隧道 '$tid' 不存在"

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        echo "已停止隧道: id=$tid pid=$pid"
    else
        echo "隧道进程已不存在: id=$tid pid=$pid"
    fi

    remove_tunnel_record "$tid"
}

cmd_stop_all() {
    ensure_tunnels_file

    # 读取所有隧道记录（id 和 pid）
    local records
    records=$($PYTHON -c "
import json, sys
data = json.load(open(sys.argv[1]))
if not data:
    print('当前无活跃隧道', file=sys.stderr)
    sys.exit(1)
for r in data:
    print(r['id'] + ' ' + str(r['pid']))
" "$TUNNELS_FILE") || { echo "当前无活跃隧道"; return 0; }

    # 在 bash 层逐个 kill（兼容 MSYS2 进程空间）
    while read -r tid pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "已停止: $tid (pid=$pid)"
        else
            echo "已失效: $tid (pid=$pid)"
        fi
    done <<< "$records"

    # 清空隧道记录
    echo '[]' > "$TUNNELS_FILE"
    echo "所有隧道已清理"
}

# ── 主入口 ────────────────────────────────────────────────

SUB="${1:-}"
[ -n "$SUB" ] || usage
shift

case "$SUB" in
    forward)  cmd_forward "$@" ;;
    reverse)  cmd_reverse "$@" ;;
    socks)    cmd_socks "$@" ;;
    proxy)    cmd_proxy "$@" ;;
    list)     list_tunnel_records ;;
    stop)     cmd_stop "$@" ;;
    stop-all) cmd_stop_all ;;
    *)        die "未知子命令: $SUB"; usage ;;
esac
