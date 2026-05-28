#!/usr/bin/env bash
# Detect verl rollout inference backend: vllm | sglang | unknown
set -euo pipefail

SCRIPT="${1:-}"

detect_from_text() {
  local text="$1"
  if echo "$text" | grep -qiE 'rollout\.name\s*=\s*sglang|rollout:\s*\n\s*name:\s*sglang'; then
    echo "sglang"; return 0
  fi
  if echo "$text" | grep -qiE 'rollout\.name\s*=\s*vllm|rollout:\s*\n\s*name:\s*vllm'; then
    echo "vllm"; return 0
  fi
  if echo "$text" | grep -qiE 'rollout\.name\s*=\s*trtllm'; then
    echo "trtllm"; return 0
  fi
  if echo "$text" | grep -qiE 'rollout\.name\s*=\s*hf'; then
    echo "hf"; return 0
  fi
  return 1
}

if [[ -z "$SCRIPT" || ! -f "$SCRIPT" ]]; then
  echo "usage: detect-rollout-backend.sh <training.sh>" >&2
  exit 2
fi

content="$(cat "$SCRIPT")"
if detect_from_text "$content"; then
  exit 0
fi

while IFS= read -r yaml; do
  if [[ -f "$yaml" ]] && detect_from_text "$(cat "$yaml")"; then
    exit 0
  fi
done < <(grep -oE '[a-zA-Z0-9_./-]+\.ya?ml' "$SCRIPT" | sort -u)

echo "unknown"
exit 1
