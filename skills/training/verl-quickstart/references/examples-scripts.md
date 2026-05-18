# examples 脚本（参考）

## 列表与过滤

```bash
export WORKSPACE="$(pwd)"
export EXAMPLES_DIR="${EXAMPLES_DIR:-$WORKSPACE/examples}"
export SCRIPT_QUERY="${SCRIPT_QUERY:-}"

echo "[candidates] all:"
ls -1 "$EXAMPLES_DIR"/*/run_*.sh 2>/dev/null || true

if [ -n "$SCRIPT_QUERY" ]; then
  echo "[candidates] filtered by: $SCRIPT_QUERY"
  CANDIDATES="$(ls -1 "$EXAMPLES_DIR"/*/run_*.sh 2>/dev/null || true)"
  for kw in $SCRIPT_QUERY; do
    CANDIDATES="$(printf "%s\n" "$CANDIDATES" | rg -i "$kw" || true)"
  done
  printf "%s\n" "$CANDIDATES"
fi
```

## 修改后冒烟（必须）

```bash
RS="$(readlink -f "$RUN_SCRIPT" 2>/dev/null || echo "$RUN_SCRIPT")"
export WORKSPACE="$(cd "$(dirname "$RS")/../.." && pwd)"
cd "$WORKSPACE"

export DATA_DIR="${DATA_DIR:-$WORKSPACE/data/gsm8k}"
export MODEL_PATH="${MODEL_PATH:?本地权重目录}"
export LOG="${LOG:-verl_smoke.log}"

nohup env MODEL_PATH="$MODEL_PATH" DATA_DIR="$DATA_DIR" LOG="$LOG" \
  bash "$RS" trainer.total_epochs=1 trainer.test_freq=1 </dev/null &

echo "训练/冒烟日志：$(readlink -f "$WORKSPACE/$LOG")"
echo "tail -f $(readlink -f "$WORKSPACE/$LOG")"
```
