#!/bin/bash
set -e
CUR="$(cd "$(dirname "$0")" && pwd)"
INI="$CUR/single_server_config.ini"

ROUNDS=$(awk -F= '/^rounds/{gsub(/ /,"",$2);print $2}' "$INI")
PASS=0; FAIL=0
K_CHOICES=(1024 2048 3072 12288)
N_CHOICES=(1024 2048 3072 3904)

for ((i=0; i<ROUNDS; i++)); do
  M=$((RANDOM % 64 + 1))
  K=${K_CHOICES[$((RANDOM % 4))]}
  N=${N_CHOICES[$((RANDOM % 4))]}
  DT=$((i % 2))      # 0=fp16, 1=bf16
  HB=$((i % 2))      # alternate bias on/off
  IGO=1
  sed -i "s/^m = .*/m = ${M}/" "$INI"
  sed -i "s/^k = .*/k = ${K}/" "$INI"
  sed -i "s/^n = .*/n = ${N}/" "$INI"
  sed -i "s/^dataType = .*/dataType = ${DT}/" "$INI"
  sed -i "s/^hasBias = .*/hasBias = ${HB}/" "$INI"
  sed -i "s/^isGatherOut = .*/isGatherOut = ${IGO}/" "$INI"
  echo "=== Round $i: M=$M K=$K N=$N dtype=$DT bias=$HB gatherOut=$IGO ==="
  python3 "$CUR/single_server_gen_data.py"
  python3 "$CUR/single_server_run.py" || { echo "run failed"; FAIL=$((FAIL+1)); continue; }
  if python3 "$CUR/single_server_check_result.py"; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
  fi
done
echo "==== PASSED=$PASS FAILED=$FAIL ===="
