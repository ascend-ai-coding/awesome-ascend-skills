#!/usr/bin/env bash
# checkpoint.sh — 长耗时任务checkpoint管理
# 用法:
#   bash checkpoint.sh write <profile> <job_id> --task "..." --duration "..." --next "..." --context "..."
#   bash checkpoint.sh read <profile> <job_id>
# 返回: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ACTION="${1:?用法: checkpoint.sh write|read <profile> <job_id> [options]}"
PROFILE="${2:?缺少 profile}"
JOB_ID="${3:?缺少 job_id}"
shift 3

run_remote() {
    bash "$SUITE_DIR/connect/scripts/ssh-exec.sh" "$PROFILE" "$@"
}

case "$ACTION" in
    write)
        TASK=""
        DURATION=""
        NEXT=""
        CONTEXT=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --task) TASK="$2"; shift 2 ;;
                --duration) DURATION="$2"; shift 2 ;;
                --next) NEXT="$2"; shift 2 ;;
                --context) CONTEXT="$2"; shift 2 ;;
                *) shift ;;
            esac
        done

        CHECKPOINT_CONTENT="job_id: $JOB_ID
profile: $PROFILE
task: \"$TASK\"
started: $(date -u +"%Y-%m-%dT%H:%M:%S")
expected_duration: $DURATION
next_steps:"

        IFS=',' read -ra STEPS <<< "$NEXT"
        for step in "${STEPS[@]}"; do
            CHECKPOINT_CONTENT="$CHECKPOINT_CONTENT
  - $step"
        done

        CHECKPOINT_CONTENT="$CHECKPOINT_CONTENT
context: \"$CONTEXT\""

        run_remote "cat > ~/.ssh-jobs/$JOB_ID/checkpoint.md << 'CHECKPOINT_EOF'
$CHECKPOINT_CONTENT
CHECKPOINT_EOF"

        echo "✓ Checkpoint已写入: ~/.ssh-jobs/$JOB_ID/checkpoint.md"
        ;;

    read)
        if ! run_remote "test -f ~/.ssh-jobs/$JOB_ID/checkpoint.md" 2>/dev/null; then
            echo "✗ Checkpoint不存在: $JOB_ID" >&2
            exit 1
        fi

        run_remote "cat ~/.ssh-jobs/$JOB_ID/checkpoint.md"
        ;;

    *)
        echo "未知操作: $ACTION (可用: write, read)" >&2
        exit 1
        ;;
esac
