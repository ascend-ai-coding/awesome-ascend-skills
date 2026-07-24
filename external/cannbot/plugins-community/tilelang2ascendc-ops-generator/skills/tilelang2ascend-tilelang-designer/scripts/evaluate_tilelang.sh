#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_workdir() {
  if [[ -n "${WORKDIR:-}" ]]; then
    echo "${WORKDIR}"
    return 0
  fi

  local candidate="${SCRIPT_DIR}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/.claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py" ]]; then
      echo "${candidate}"
      return 0
    fi
    candidate="$(cd "${candidate}/.." && pwd)"
  done

  return 1
}

WORKDIR="$(find_workdir)" || {
  echo "Unable to locate repository root containing .claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py" >&2
  exit 1
}

SSH_TARGET="${SSH_TARGET:-ascend-box}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-}"

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/root/tilelang_eval}"
CONTAINER_NAME="${CONTAINER_NAME:-zyy_cann}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/home/z00893531/tilelang-ascend}"
REMOTE_EVAL_WORKDIR="${REMOTE_EVAL_WORKDIR:-workdir_remote_eval}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-3}"

usage() {
  cat <<'EOF'
Usage: bash skills/ascendc/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh [task]

Arguments:
  task    Task directory to verify. Defaults to current_task.

Environment overrides:
  SSH_TARGET            SSH host or ~/.ssh/config alias
  REMOTE_PORT           Optional SSH port override
  SSH_KEY               Optional SSH identity file override
  REMOTE_BASE_DIR       Host path used to store uploaded workdir
  CONTAINER_NAME        Target docker container name
  CONTAINER_WORKDIR     Project root inside the container
  REMOTE_EVAL_WORKDIR   Working directory name used inside the container
  ASCEND_RT_VISIBLE_DEVICES  Device id used inside the container

Examples:
  bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh
  bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh matmul_add
  SSH_TARGET=ascend-box bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh matmul_add
  REMOTE_EVAL_WORKDIR=workdir_remote_eval_wzz bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh current_task
  REMOTE_BASE_DIR=/data/eval bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh current_task
  ASCEND_RT_VISIBLE_DEVICES=3 bash <path-to-tilelang2ascend-tilelang-designer>/scripts/evaluate_tilelang.sh current_task
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${1:-current_task}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="workdir_${TIMESTAMP}.tar.gz"
LOCAL_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_SESSION_DIR="${REMOTE_BASE_DIR}/${TIMESTAMP}"

PYTHONPATH_PREFIX="${WORKDIR}"
if [[ -d "${WORKDIR}/archive_tasks" ]]; then
  PYTHONPATH_PREFIX="${WORKDIR}/archive_tasks:${PYTHONPATH_PREFIX}"
fi

if [[ ! -f "${WORKDIR}/.claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py" ]]; then
  echo "Missing verification script: ${WORKDIR}/.claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py" >&2
  exit 1
fi

# Support both relative and absolute paths for TASK
if [[ "${TASK}" == /* ]]; then
  TASK_DIR="${TASK}"
else
  TASK_DIR="${WORKDIR}/${TASK}"
fi

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "Task directory not found: ${TASK_DIR}" >&2
  exit 1
fi

if python -c 'import tilelang; import torch; import torch_npu' >/dev/null 2>&1; then
  echo "Detected local TileLang-Ascend environment, running local verification"
  cd "${WORKDIR}"
  PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
    python .claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py "${TASK_DIR}"
  exit 0
fi

SSH_OPTS=()
SCP_OPTS=()
if [[ -n "${REMOTE_PORT}" ]]; then
  SSH_OPTS+=(-p "${REMOTE_PORT}")
  SCP_OPTS+=(-P "${REMOTE_PORT}")
fi

if [[ -n "${SSH_KEY}" ]]; then
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_OPTS+=(-i "${SSH_KEY}")
  SCP_OPTS+=(-i "${SSH_KEY}")
fi

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
  SSH_CMD+=("${SSH_OPTS[@]}")
fi
if [[ ${#SCP_OPTS[@]} -gt 0 ]]; then
  SCP_CMD+=("${SCP_OPTS[@]}")
fi

cleanup() {
  rm -f "${LOCAL_ARCHIVE}"
}
trap cleanup EXIT

echo "[1/4] Packaging ${WORKDIR}"
tar \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".ruff_cache" \
  -C "${WORKDIR}" \
  -czf "${LOCAL_ARCHIVE}" \
  .

echo "[2/4] Uploading archive to ${SSH_TARGET}:${REMOTE_ARCHIVE}"
"${SCP_CMD[@]}" \
  "${LOCAL_ARCHIVE}" \
  "${SSH_TARGET}:${REMOTE_ARCHIVE}"

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail
cleanup_remote() {
  rm -rf "${REMOTE_SESSION_DIR}"
}
trap cleanup_remote EXIT

mkdir -p "${REMOTE_SESSION_DIR}"
tar -xzf "${REMOTE_ARCHIVE}" -C "${REMOTE_SESSION_DIR}"
rm -f "${REMOTE_ARCHIVE}"

docker exec "${CONTAINER_NAME}" /bin/bash -lc 'mkdir -p "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"'

docker cp "${REMOTE_SESSION_DIR}/." "${CONTAINER_NAME}:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"

docker exec "${CONTAINER_NAME}" /bin/bash -lc '
set -euo pipefail
cd "${CONTAINER_WORKDIR}"
source set_env.sh
cd "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"
PYTHONPATH="${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}/archive_tasks:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}\${PYTHONPATH:+:\${PYTHONPATH}}" \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
python .claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py "${TASK}"
'
EOF

echo "[3/4] Running verification inside container ${CONTAINER_NAME}"
"${SSH_CMD[@]}" \
  "${SSH_TARGET}" \
  "${REMOTE_SCRIPT}"

echo "[4/4] Verification completed"
