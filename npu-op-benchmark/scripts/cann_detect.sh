#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  CANDIDATE_ROOTS=("$1")
else
  CANDIDATE_ROOTS=(
    "/usr/local/Ascend/ascend-toolkit"
    "/usr/local/Ascend/cann"
  )
fi

for root in "${CANDIDATE_ROOTS[@]}"; do
  echo "cann_root=${root}"
  if [[ -f "${root}/set_env.sh" ]]; then
    echo "set_env_exists=true"
  else
    echo "set_env_exists=false"
  fi

  if [[ -e "${root}/latest" ]]; then
    echo "latest_ll=$(ls -ld "${root}/latest")"
    if [[ -L "${root}/latest/compiler" ]]; then
      echo "compiler_ll=$(ls -ld "${root}/latest/compiler")"
    else
      echo "compiler_ll=missing_or_not_symlink"
    fi
  else
    echo "latest_ll=missing"
    echo "compiler_ll=missing"
  fi

  find "${root}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | sed 's/^/version_dir=/' || true
  echo "===="
done
