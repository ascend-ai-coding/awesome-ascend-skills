#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found" >&2
  exit 1
fi

for c in $(docker ps --format '{{.Names}}'); do
  echo "container_name=${c}"
  docker exec "$c" /bin/sh -lc 'pip list 2>/dev/null | grep -E "^(torch|torch-npu|torch_npu)" || true; echo ---; ls -ld /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || true; echo ---; find /usr/local/Ascend/ascend-toolkit -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort || true'
  echo "===="
done
