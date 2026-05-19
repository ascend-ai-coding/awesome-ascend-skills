#!/usr/bin/env bash
# ssh-job.sh — 远程后台任务管理
# 用法:
#   bash ssh-job.sh start <profile> <command>                    → 启动后台任务
#   bash ssh-job.sh status <profile> <job_id>                    → 查看任务状态
#   bash ssh-job.sh output <profile> <job_id> [--tail N] [--head N] [--grep pattern]
#   bash ssh-job.sh kill <profile> <job_id>                      → 终止任务
#   bash ssh-job.sh list <profile>                               → 列出所有任务
#   bash ssh-job.sh stream <profile> <job_id> [local_file]              → 流式输出到本地文件
# 依赖: $PYTHON, ssh-connect 套件
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SUITE_DIR/_lib/_python.sh"
ACTION="${1:?用法: bash ssh-job.sh <action> <profile> [args...]}"
shift

run_remote() {
    local profile="$1"; shift
    bash "$SCRIPT_DIR/ssh-exec.sh" "$profile" "$@"
}

case "$ACTION" in
    start)
        PROFILE="${1:?缺少 profile}"; shift
        COMMAND="$*"
        JOB_ID="job-$(date +%Y%m%d-%H%M%S)-$$"
        JOB_DIR="~/.ssh-jobs/$JOB_ID"

        # 使用 base64 编码传递命令，避免引号嵌套问题
        ENCODED_CMD=$(printf '%s' "$COMMAND" | base64 -w0 2>/dev/null || printf '%s' "$COMMAND" | base64)
        run_remote "$PROFILE" "mkdir -p $JOB_DIR && echo '$ENCODED_CMD' | base64 -d > $JOB_DIR/cmd.txt"
        run_remote "$PROFILE" "nohup bash -c \"\$(echo '$ENCODED_CMD' | base64 -d)\" > $JOB_DIR/stdout.log 2> $JOB_DIR/stderr.log & echo \$! > $JOB_DIR/pid.txt && wait \$! 2>/dev/null; echo \$? > $JOB_DIR/exit_code.txt" &
        disown

        echo "任务已启动: $JOB_ID"
        echo "查看状态: bash ssh-job.sh status $PROFILE $JOB_ID"
        echo "查看输出: bash ssh-job.sh output $PROFILE $JOB_ID --tail 50"
        ;;

    status)
        PROFILE="${1:?缺少 profile}"
        JOB_ID="${2:?缺少 job_id}"
        JOB_DIR="~/.ssh-jobs/$JOB_ID"

        if ! run_remote "$PROFILE" "test -d $JOB_DIR" 2>/dev/null; then
            echo "任务不存在: $JOB_ID" >&2; exit 1
        fi

        if run_remote "$PROFILE" "test -f $JOB_DIR/exit_code.txt" 2>/dev/null; then
            EXIT_CODE=$(run_remote "$PROFILE" "cat $JOB_DIR/exit_code.txt")
            if [ "$EXIT_CODE" = "0" ]; then
                echo "状态: completed (退出码: 0)"
            else
                echo "状态: failed (退出码: $EXIT_CODE)"
            fi
        else
            PID=$(run_remote "$PROFILE" "cat $JOB_DIR/pid.txt 2>/dev/null || echo ''")
            if [ -n "$PID" ] && run_remote "$PROFILE" "kill -0 $PID 2>/dev/null"; then
                echo "状态: running (PID: $PID)"
            else
                echo "状态: unknown (进程可能已异常退出)"
            fi
        fi

        STDOUT_SIZE=$(run_remote "$PROFILE" "wc -c < $JOB_DIR/stdout.log 2>/dev/null || echo 0")
        STDERR_SIZE=$(run_remote "$PROFILE" "wc -c < $JOB_DIR/stderr.log 2>/dev/null || echo 0")
        echo "stdout: ${STDOUT_SIZE}B, stderr: ${STDERR_SIZE}B"
        ;;

    output)
        PROFILE="${1:?缺少 profile}"
        JOB_ID="${2:?缺少 job_id}"
        JOB_DIR="~/.ssh-jobs/$JOB_ID"
        shift 2

        LOG_FILE="$JOB_DIR/stdout.log"
        FILTER_CMD="cat"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --tail) FILTER_CMD="tail -n $2"; shift 2 ;;
                --head) FILTER_CMD="head -n $2"; shift 2 ;;
                --grep) FILTER_CMD="grep -n '$2'"; shift 2 ;;
                --stderr) LOG_FILE="$JOB_DIR/stderr.log"; shift ;;
                *) shift ;;
            esac
        done

        run_remote "$PROFILE" "$FILTER_CMD $LOG_FILE"
        ;;

    kill)
        PROFILE="${1:?缺少 profile}"
        JOB_ID="${2:?缺少 job_id}"
        JOB_DIR="~/.ssh-jobs/$JOB_ID"
        PID=$(run_remote "$PROFILE" "cat $JOB_DIR/pid.txt 2>/dev/null || echo ''")
        if [ -n "$PID" ]; then
            run_remote "$PROFILE" "kill $PID 2>/dev/null || true"
            echo "已终止任务: $JOB_ID (PID: $PID)"
        else
            echo "未找到 PID: $JOB_ID" >&2; exit 1
        fi
        ;;

    list)
        PROFILE="${1:?缺少 profile}"
        run_remote "$PROFILE" "
if [ ! -d ~/.ssh-jobs ]; then echo '无任务'; exit 0; fi
for d in ~/.ssh-jobs/job-*/; do
    [ -d \"\$d\" ] || continue
    id=\$(basename \$d)
    cmd=\$(cat \$d/cmd.txt 2>/dev/null | head -c 60)
    if [ -f \$d/exit_code.txt ]; then
        code=\$(cat \$d/exit_code.txt)
        [ \"\$code\" = '0' ] && status='completed' || status='failed'
    elif [ -f \$d/pid.txt ] && kill -0 \$(cat \$d/pid.txt) 2>/dev/null; then
        status='running'
    else
        status='unknown'
    fi
    printf '%-30s %-12s %s\n' \"\$id\" \"\$status\" \"\$cmd\"
done
"
        ;;

    stream)
        PROFILE="${1:?缺少 profile}"
        JOB_ID="${2:?缺少 job_id}"
        LOCAL_FILE="${3:-job-$JOB_ID-stream.log}"
        JOB_DIR="~/.ssh-jobs/$JOB_ID"

        if ! run_remote "$PROFILE" "test -f $JOB_DIR/stdout.log" 2>/dev/null; then
            echo "任务日志不存在: $JOB_ID" >&2; exit 1
        fi

        echo "开始流式输出到: $LOCAL_FILE"
        echo "按 Ctrl+C 停止"

        CONFIG_JSON=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE")
        SSH_OPTS=$($PYTHON "$SUITE_DIR/_lib/ssh-config.py" "$PROFILE" --ssh-opts)
        PASSWORD=$(echo "$CONFIG_JSON" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('password',''))")

        if [ -n "$PASSWORD" ]; then
            sshpass -p "$PASSWORD" ssh $SSH_OPTS "tail -f $JOB_DIR/stdout.log" > "$LOCAL_FILE"
        else
            ssh $SSH_OPTS "tail -f $JOB_DIR/stdout.log" > "$LOCAL_FILE"
        fi
        ;;

    *)
        echo "未知操作: $ACTION (可用: start, status, output, kill, list, stream)" >&2
        exit 1
        ;;
esac
