#!/usr/bin/env bash
# run_bench.sh — Execute a single vllm bench serve command with error handling
# Usage: ./run_bench.sh "full vllm bench serve command" [--timeout SECONDS]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"vllm bench serve --base-url ... --model ... [options]\" [--timeout SECONDS]"
  exit 1
fi

BENCH_CMD="$1"
BENCH_TIMEOUT="${2:-}"
# Parse optional --timeout flag
if [[ "$BENCH_TIMEOUT" == "--timeout" && -n "${3:-}" ]]; then
  BENCH_TIMEOUT="$3"
elif [[ "$BENCH_TIMEOUT" =~ ^[0-9]+$ ]]; then
  : # numeric value passed directly as $2
else
  BENCH_TIMEOUT=""
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Quick environment check: verify vllm bench serve is available
if ! command -v vllm &>/dev/null; then
  echo "ERROR: 'vllm' command not found. Please install vLLM first."
  echo "  Run: ${SCRIPT_DIR}/check_bench_env.sh for detailed checks."
  exit 1
fi

# Extract result-dir from command if present
RESULT_DIR=$(echo "$BENCH_CMD" | grep -oP '(?<=--result-dir\s)\S+' || echo "./bench_results")

# Create result directory
mkdir -p "$RESULT_DIR"

# Create log directory
LOG_DIR="${RESULT_DIR%/}/../logs"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp/bench_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bench_${TIMESTAMP}.log"

echo "=== Benchmark Execution ==="
echo "  Time:    $(date)"
echo "  Log:     $LOG_FILE"
echo "  Results: $RESULT_DIR"
echo ""
echo "  Command:"
echo "    $BENCH_CMD"
echo ""

# Execute benchmark (use bash -c instead of eval to isolate execution)
EXIT_CODE=0
if [[ -n "$BENCH_TIMEOUT" ]]; then
  echo "  Timeout: ${BENCH_TIMEOUT}s"
  timeout "$BENCH_TIMEOUT" bash -c "$BENCH_CMD" 2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?
  if [[ $EXIT_CODE -eq 124 ]]; then
    echo "  ERROR: Benchmark timed out after ${BENCH_TIMEOUT}s"
  fi
else
  bash -c "$BENCH_CMD" 2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?
fi

echo ""
echo "=== Execution Complete ==="
echo "  Exit code: $EXIT_CODE"
echo "  Log: $LOG_FILE"

# Check for result files by extracting expected filename from command
EXPECTED_FILENAME=$(echo "$BENCH_CMD" | grep -oP '(?<=--result-filename\s)\S+' || echo "")
if [[ $EXIT_CODE -eq 0 ]]; then
  if [[ -n "$EXPECTED_FILENAME" && -f "$RESULT_DIR/$EXPECTED_FILENAME" ]]; then
    echo "  Result file: $RESULT_DIR/$EXPECTED_FILENAME"
  else
    # Fallback: look for any JSON files in result dir
    RESULT_FILES=$(find "$RESULT_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | head -5)
    if [[ -n "$RESULT_FILES" ]]; then
      echo "  Result files:"
      echo "$RESULT_FILES" | while read -r f; do echo "    $f"; done
    else
      echo "  Warning: No result JSON files found in $RESULT_DIR"
    fi
  fi
else
  echo "  ERROR: Benchmark failed with exit code $EXIT_CODE"
  echo "  Check log for details: $LOG_FILE"
fi

exit $EXIT_CODE
