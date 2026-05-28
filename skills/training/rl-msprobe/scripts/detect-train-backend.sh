#!/usr/bin/env bash
# Detect verl training backend (fsdp|megatron) from a shell training script.
set -euo pipefail

SCRIPT="${1:-}"
if [[ -z "$SCRIPT" || ! -f "$SCRIPT" ]]; then
  echo "usage: detect-train-backend.sh <training.sh>" >&2
  exit 2
fi

content="$(cat "$SCRIPT")"

detect_from_text() {
  local text="$1"
  if echo "$text" | grep -qiE 'strategy[= ]megatron|backend[= ]megatron|\.strategy=megatron'; then
    echo "megatron"
    return 0
  fi
  if echo "$text" | grep -qiE 'strategy[= ]fsdp2?|backend[= ]fsdp2?|\.strategy=fsdp'; then
    echo "fsdp"
    return 0
  fi
  return 1
}

if detect_from_text "$content"; then
  exit 0
fi

# Follow sourced yaml references (best effort)
while IFS= read -r yaml; do
  if [[ -f "$yaml" ]] && detect_from_text "$(cat "$yaml")"; then
    exit 0
  fi
done < <(grep -oE '[a-zA-Z0-9_./-]+\.ya?ml' "$SCRIPT" | sort -u)

echo "unknown"
exit 1
