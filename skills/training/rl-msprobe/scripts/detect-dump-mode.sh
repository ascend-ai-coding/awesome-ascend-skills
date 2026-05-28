#!/usr/bin/env bash
# Detect verl msprobe dump mode: training | inference | consistency
set -euo pipefail

SCRIPT="${1:-}"
USER_HINT="${2:-}"

score_training=0
score_inference=0
score_consistency=0

bump() {
  local mode="$1" pts="$2"
  case "$mode" in
    training) score_training=$((score_training + pts)) ;;
    inference) score_inference=$((score_inference + pts)) ;;
    consistency) score_consistency=$((score_consistency + pts)) ;;
  esac
}

case "${USER_HINT,,}" in
  training|train|普通训练|训练采集|profiler|precision_debugger|actor_compute_log_prob|ref_compute_log_prob|actor_update) bump training 10 ;;
  inference|infer|推理|rollout|generate|vllm|sglang) bump inference 10 ;;
  consistency|训推|一致性|prefill|compare|训推一致) bump consistency 10 ;;
esac

if [[ -n "$SCRIPT" && -f "$SCRIPT" ]]; then
  content="$(cat "$SCRIPT")"
  echo "$content" | grep -qiE 'global_profiler\.tool=precision_debugger|tool:\s*precision_debugger' \
    && bump training 8
  echo "$content" | grep -qiE 'actor_compute_log_prob|ref_compute_log_prob|actor_update' \
    && bump training 4
  echo "$content" | grep -qiE 'precision_debugger\.stages|profiler\.enable=True' \
    && bump training 3
  echo "$content" | grep -qiE 'dump_config_path|msprobe_dump_config|msprobe-dump-config' \
    && bump inference 6
  echo "$content" | grep -qiE 'enforce_eager=True|enforce_eager:\s*True' \
    && bump inference 2
  echo "$content" | grep -qiE 'DUMP_ON|PROMPTS_ONLY' \
    && bump consistency 8
  echo "$content" | grep -qiE 'dispatch_log|update_actor_log' \
    && bump consistency 5
  echo "$content" | grep -qiE '训推一致|consistency|prefill.*对齐' \
    && bump consistency 5
  # consistency usually combines training engine patch + inference dump
  if echo "$content" | grep -qiE 'DUMP_ON' \
    && echo "$content" | grep -qiE 'dump_config_path|PROMPTS_ONLY'; then
    bump consistency 5
  fi
fi

best="unknown"
best_score=-1
for mode in training inference consistency; do
  eval "s=\$score_${mode}"
  if [[ $s -gt $best_score ]]; then
    best_score=$s
    best=$mode
  fi
done

if [[ $best_score -le 0 ]]; then
  echo "unknown"
  exit 1
fi

echo "$best"
exit 0
