#!/usr/bin/env bash
# run_batch.sh — Execute a batch of benchmark cases (sequentially or in parallel)
# Usage: ./run_batch.sh --config batch_cases.jsonl --result-dir DIR [--common-args "..."] [--parallel N] [--timeout S]
#
# batch_cases.jsonl format (one JSON object per line):
#   {"case_name": "c1", "max_concurrency": 1, "num_prompts": 50}
#   {"case_name": "c8", "max_concurrency": 8, "num_prompts": 400}
#
# Common args are shared across all cases. Per-case args override common args.
set -euo pipefail

CONFIG=""
RESULT_DIR=""
COMMON_ARGS=""
CASE_TIMEOUT=""
PARALLEL=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --common-args) COMMON_ARGS="$2"; shift 2 ;;
    --timeout) CASE_TIMEOUT="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  echo "Usage: $0 --config batch.jsonl --result-dir DIR [--common-args \"...\"]"
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: Config file not found: $CONFIG"
  exit 1
fi

# Default result dir
if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="./bench_results/batch/batch_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULT_DIR"

echo "=== Batch Benchmark Execution ==="
echo "  Config:     $CONFIG"
echo "  Result Dir: $RESULT_DIR"
echo "  Common Args: $COMMON_ARGS"
if [[ "$PARALLEL" -gt 1 ]]; then
  echo "  Parallel:   $PARALLEL (WARNING: concurrent benchmarks may interfere with each other)"
fi
echo ""

TOTAL=$(wc -l < "$CONFIG")
CURRENT=0
PASSED=0
FAILED=0
FAILED_CASES=()
PIDS=()

while IFS= read -r line; do
  ((CURRENT++))
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == "#"* ]] && continue

  CASE_NAME=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('case_name', f'case_{$CURRENT:03d}'))" 2>/dev/null || echo "case_${CURRENT}")

  echo ""
  echo "--- Case $CURRENT/$TOTAL: $CASE_NAME ---"

  # Build per-case args from JSON
  CASE_ARGS=$(echo "$line" | python3 -c "
import sys, json
d = json.load(sys.stdin)
args = []
skip_keys = {'case_name'}
for k, v in d.items():
    if k in skip_keys:
        continue
    flag = '--' + k.replace('_', '-')
    if isinstance(v, bool):
        if v:
            args.append(flag)
    elif v is not None:
        args.append(f'{flag} {v}')
print(' '.join(args))
" 2>/dev/null) || CASE_ARGS=""

  # Build filename
  CASE_FILENAME="case_${CURRENT}_${CASE_NAME}.json"

  # Build full command
  FULL_CMD="vllm bench serve $COMMON_ARGS $CASE_ARGS --save-result --save-detailed --result-dir $RESULT_DIR --result-filename $CASE_FILENAME"

  echo "  Command: $FULL_CMD"

  if [[ "$PARALLEL" -gt 1 ]]; then
    # Parallel execution: run in background, log to file
    (
      _EC=0
      if [[ -n "$CASE_TIMEOUT" ]]; then
        timeout "$CASE_TIMEOUT" bash -c "$FULL_CMD" > "$RESULT_DIR/${CASE_NAME}.log" 2>&1 || _EC=$?
      else
        bash -c "$FULL_CMD" > "$RESULT_DIR/${CASE_NAME}.log" 2>&1 || _EC=$?
      fi
      echo "$_EC" > "$RESULT_DIR/${CASE_NAME}.exitcode"
    ) &
    PIDS+=($!)
    # Throttle: wait if we've hit the parallel limit
    while [[ $(jobs -rp | wc -l) -ge "$PARALLEL" ]]; do
      wait -n 2>/dev/null || true
    done
  else
    # Sequential execution
    EXIT_CODE=0
    if [[ -n "$CASE_TIMEOUT" ]]; then
      timeout "$CASE_TIMEOUT" bash -c "$FULL_CMD" 2>&1 | tee "$RESULT_DIR/${CASE_NAME}.log" || EXIT_CODE=$?
      if [[ $EXIT_CODE -eq 124 ]]; then
        echo "  TIMEOUT after ${CASE_TIMEOUT}s"
      fi
    else
      bash -c "$FULL_CMD" 2>&1 | tee "$RESULT_DIR/${CASE_NAME}.log" || EXIT_CODE=$?
    fi

    if [[ $EXIT_CODE -eq 0 ]]; then
      ((PASSED++))
      echo "  Result: PASS"
    else
      ((FAILED++))
      FAILED_CASES+=("$CASE_NAME")
      echo "  Result: FAIL (exit code $EXIT_CODE)"
    fi
  fi

done < "$CONFIG"

# Wait for all parallel jobs and collect results
if [[ "$PARALLEL" -gt 1 ]]; then
  echo ""
  echo "  Waiting for parallel jobs to complete..."
  wait
  # Collect results from exit code files
  for ec_file in "$RESULT_DIR"/*.exitcode; do
    [[ -f "$ec_file" ]] || continue
    case_base=$(basename "$ec_file" .exitcode)
    ec=$(cat "$ec_file")
    if [[ "$ec" -eq 0 ]]; then
      ((PASSED++))
      echo "  $case_base: PASS"
    else
      ((FAILED++))
      FAILED_CASES+=("$case_base")
      echo "  $case_base: FAIL (exit code $ec)"
    fi
    rm -f "$ec_file"
  done
fi

echo ""
echo "=== Batch Summary ==="
echo "  Total:  $CURRENT"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
if [[ ${#FAILED_CASES[@]} -gt 0 ]]; then
  echo "  Failed cases: ${FAILED_CASES[*]}"
fi
echo "  Results in: $RESULT_DIR"

# Run aggregation if aggregate script exists
if [[ -f "$SCRIPT_DIR/aggregate_results.py" ]]; then
  echo ""
  echo "=== Aggregating Results ==="
  python3 "$SCRIPT_DIR/aggregate_results.py" --result-dir "$RESULT_DIR" --format markdown
fi

exit $( [[ $FAILED -gt 0 ]] && echo 1 || echo 0 )
