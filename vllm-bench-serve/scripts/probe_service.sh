#!/usr/bin/env bash
# probe_service.sh — Probe a running vLLM service for health, models, and backends
# Usage: ./probe_service.sh --base-url http://ip:port [--timeout 10]
# Output: JSON with service info
set -euo pipefail

BASE_URL=""
TIMEOUT=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$BASE_URL" ]]; then
  echo '{"error": "missing --base-url"}' >&2
  exit 1
fi

# Remove trailing slash
BASE_URL="${BASE_URL%/}"

HEALTHY=false
MODELS="[]"
BACKENDS="[]"

# 1. Health check
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" "$BASE_URL/health" 2>/dev/null) || HTTP_CODE="000"
if [[ "$HTTP_CODE" == "200" ]]; then
  HEALTHY=true
fi

# 2. Get model list (id = served name, root = weight path for tokenizer)
MODELS_RAW=$(curl -s --connect-timeout "$TIMEOUT" "$BASE_URL/v1/models" 2>/dev/null) || MODELS_RAW=""
MODELS="[]"
MODEL_DETAILS="[]"
if [[ -n "$MODELS_RAW" ]]; then
  read -r MODELS MODEL_DETAILS < <(echo "$MODELS_RAW" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ids = [m['id'] for m in d.get('data', [])]
    details = [{'id': m['id'], 'root': m.get('root', m['id'])} for m in d.get('data', [])]
    print(json.dumps(ids), json.dumps(details))
except:
    print('[] []')
" 2>/dev/null) || { MODELS="[]"; MODEL_DETAILS="[]"; }
fi

# 3. Detect available backends
DETECTED_BACKENDS=()

# Check completions endpoint
COMP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" \
  -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"hi","max_tokens":1}' 2>/dev/null) || COMP_CODE="000"
if [[ "$COMP_CODE" != "000" && "$COMP_CODE" != "404" ]]; then
  DETECTED_BACKENDS+=("openai")
fi

# Check chat completions endpoint
CHAT_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null) || CHAT_CODE="000"
if [[ "$CHAT_CODE" != "000" && "$CHAT_CODE" != "404" ]]; then
  DETECTED_BACKENDS+=("openai-chat")
fi

# Check embeddings endpoint
EMB_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" \
  -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","input":"hi"}' 2>/dev/null) || EMB_CODE="000"
if [[ "$EMB_CODE" != "000" && "$EMB_CODE" != "404" ]]; then
  DETECTED_BACKENDS+=("openai-embeddings")
fi

# Check rerank endpoint
RERANK_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" \
  -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","query":"hi","documents":["test"]}' 2>/dev/null) || RERANK_CODE="000"
if [[ "$RERANK_CODE" != "000" && "$RERANK_CODE" != "404" ]]; then
  DETECTED_BACKENDS+=("vllm-rerank")
fi

# Build and output final JSON in a single Python call
python3 -c "
import json, sys

# Build backends list from detected backends
detected = []
for b in sys.argv[1:]:
    if b.strip():
        detected.append(b.strip())

result = {
    'healthy': $(  [[ $HEALTHY == true ]] && echo 'True' || echo 'False' ),
    'models': $MODELS,
    'model_details': $MODEL_DETAILS,
    'backends': detected,
    'base_url': '$BASE_URL'
}
print(json.dumps(result, indent=2))
" ${DETECTED_BACKENDS[@]+"${DETECTED_BACKENDS[@]}"}
