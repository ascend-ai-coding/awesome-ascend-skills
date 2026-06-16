#!/bin/bash
# 自动化 profiling 测试脚本（支持 warmup）
# 使用方法:
#   ./run_profiling_test.sh <output_dir> <pid> warmup     # 仅 warmup
#   ./run_profiling_test.sh <output_dir> <pid> test        # 仅正式测试
#   ./run_profiling_test.sh <output_dir> <pid> full        # warmup + 正式测试

set -e

OUTPUT_DIR=$1
XLLM_PID=$2
MODE=${3:-full}
BATCH_SIZE=${4:-16}
NUM_BATCHES=${5:-1}
WARMUP_BATCHES=${6:-1}

if [ -z "$OUTPUT_DIR" ] || [ -z "$XLLM_PID" ]; then
    echo "Usage: $0 <output_dir> <xllm_pid> [mode] [batch_size] [num_batches] [warmup_batches]"
    echo "  mode: warmup | test | full (default: full)"
    exit 1
fi

echo "=== Profiling Test Configuration ==="
echo "Output Dir: $OUTPUT_DIR"
echo "XLLM PID: $XLLM_PID"
echo "Mode: $MODE"
echo "Batch Size: $BATCH_SIZE"
echo "Test Batches: $NUM_BATCHES"
echo "Warmup Batches: $WARMUP_BATCHES"
echo "===================================="

mkdir -p "$OUTPUT_DIR"

# 创建 FIFO 管道
PIPE_FILE="/tmp/msprof_pipe_$$"
mkfifo "$PIPE_FILE"

# 启动 msprof
echo "Starting msprof in attach mode..."
msprof \
  --dynamic=on \
  --output="$OUTPUT_DIR" \
  --model-execution=on \
  --runtime-api=on \
  --aicpu=on \
  --pid="$XLLM_PID" < "$PIPE_FILE" &

MSPROF_PID=$!
exec 3>"$PIPE_FILE"

echo "msprof started with PID: $MSPROF_PID"
sleep 2

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_warmup() {
    echo ""
    echo "========== WARMUP PHASE (Profiling OFF) =========="
    python3 "$SCRIPT_DIR/multibatch_test.py" "$BATCH_SIZE" "$WARMUP_BATCHES" 2>&1 | sed 's/^/[WARMUP] /'
    sleep 3
}

run_test() {
    echo ""
    echo "========== FORMAL TEST PHASE (Profiling ON) =========="
    echo "start" >&3
    sleep 2
    python3 "$SCRIPT_DIR/multibatch_test.py" "$BATCH_SIZE" "$NUM_BATCHES" 2>&1 | sed 's/^/[TEST] /'
    echo "stop" >&3
    sleep 3
}

case "$MODE" in
    warmup)
        run_warmup
        ;;
    test)
        run_test
        ;;
    full)
        run_warmup
        echo ""
        echo "Waiting 2 seconds before formal test..."
        sleep 2
        run_test
        ;;
esac

# 退出 msprof
echo "quit" >&3
sleep 2

# 清理
exec 3>&-
rm -f "$PIPE_FILE"
wait $MSPROF_PID 2>/dev/null || true

# 查找最新的 profiling 目录
LATEST_PROF=$(ls -td "$OUTPUT_DIR"/PROF_* 2>/dev/null | head -1)

echo ""
echo "=== Profiling completed ==="
echo "Raw data directory: $OUTPUT_DIR"
echo "Latest profiling dir: $LATEST_PROF"

# 解析 profiling 数据
if [ -n "$LATEST_PROF" ]; then
    echo ""
    echo "=== Exporting profiling data ==="
    msprof --export=on --output="$LATEST_PROF" 2>&1 | tail -10

    # 查找生成的报告目录
    REPORT_DIR="$LATEST_PROF/mindstudio_profiler_output"
    if [ -d "$REPORT_DIR" ]; then
        echo ""
        echo "=== Profiling report generated ==="
        echo "Report directory: $REPORT_DIR"
        ls -la "$REPORT_DIR"/
    else
        echo "[Warning] Report directory not found"
    fi
fi
