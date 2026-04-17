#!/usr/bin/env bash
# check_bench_env.sh — Verify vllm bench serve execution environment
# Usage: ./check_bench_env.sh [--base-url URL]
set -euo pipefail

BASE_URL=""
PASS=0
FAIL=0
WARN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

check_pass() { echo "  [PASS] $1"; ((PASS++)); }
check_fail() { echo "  [FAIL] $1"; ((FAIL++)); }
check_warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "=== vllm bench serve Environment Check ==="
echo ""

# 1. Python
echo "1. Python:"
if command -v python3 &>/dev/null; then
  PY_VER=$(python3 --version 2>&1)
  check_pass "$PY_VER"
else
  check_fail "python3 not found"
fi

# 2. vLLM installation
echo "2. vLLM package:"
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null) && \
  check_pass "vllm $VLLM_VER" || \
  check_fail "vllm not installed"

# 3. vllm bench serve subcommand
echo "3. vllm bench serve:"
if vllm bench serve --help &>/dev/null; then
  check_pass "vllm bench serve available"
else
  # Fallback check
  if python3 -m vllm.benchmarks.serve --help &>/dev/null; then
    check_warn "vllm bench serve not found, but python3 -m vllm.benchmarks.serve works"
  else
    check_fail "vllm bench serve not available"
  fi
fi

# 4. aiohttp
echo "4. aiohttp:"
AIOHTTP_VER=$(python3 -c "import aiohttp; print(aiohttp.__version__)" 2>/dev/null) && \
  check_pass "aiohttp $AIOHTTP_VER" || \
  check_fail "aiohttp not installed (required for async benchmark)"

# 5. Service reachability (if URL provided)
if [[ -n "$BASE_URL" ]]; then
  echo "5. Service reachability ($BASE_URL):"
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$BASE_URL/health" 2>/dev/null) || HTTP_CODE="000"
  if [[ "$HTTP_CODE" == "200" ]]; then
    check_pass "Service healthy (HTTP 200)"
  elif [[ "$HTTP_CODE" == "000" ]]; then
    check_fail "Cannot connect to $BASE_URL"
  else
    check_warn "Service responded with HTTP $HTTP_CODE"
  fi
fi

# 6. Writable directory
echo "6. Writable directory:"
TEST_DIR="./bench_results"
if mkdir -p "$TEST_DIR" 2>/dev/null && touch "$TEST_DIR/.write_test" 2>/dev/null; then
  rm -f "$TEST_DIR/.write_test"
  check_pass "$TEST_DIR is writable"
elif touch /tmp/bench_write_test 2>/dev/null; then
  rm -f /tmp/bench_write_test
  check_warn "./bench_results not writable, but /tmp is available"
else
  check_fail "No writable directory found"
fi

# Summary
echo ""
echo "=== Summary ==="
echo "  PASS: $PASS  |  WARN: $WARN  |  FAIL: $FAIL"

if [[ $FAIL -gt 0 ]]; then
  echo "  Status: NOT READY — fix FAILed checks before running benchmarks"
  exit 1
elif [[ $WARN -gt 0 ]]; then
  echo "  Status: READY (with warnings)"
  exit 0
else
  echo "  Status: READY"
  exit 0
fi
