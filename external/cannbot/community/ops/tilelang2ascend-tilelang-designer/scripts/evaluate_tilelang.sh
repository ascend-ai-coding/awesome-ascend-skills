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
SKILL_NAME="tilelang2ascend-tilelang-designer"
VERIFY_REL="scripts/verification_tilelang.py"

# 工具中立锚点优先，旧工具目录依次保底。
# 顺序即优先级：老工程没有 .cannbot 时退回 .claude，行为与改造前完全一致。
CANNBOT_SKILL_SUBDIRS=(
  ".cannbot/skills"
  ".claude/skills"
  ".opencode/skills"
  ".agents/skills"
  ".dsh/skills"
)

find_skills_subdir() {
  local candidate="$1" sub
  for sub in "${CANNBOT_SKILL_SUBDIRS[@]}"; do
    if [[ -f "${candidate}/${sub}/${SKILL_NAME}/${VERIFY_REL}" ]]; then
      echo "${sub}"
      return 0
    fi
  done
  return 1
}

find_workdir() {
  if [[ -n "${CANNBOT_WORKDIR:-}" ]]; then
    echo "${CANNBOT_WORKDIR}"
    return 0
  fi

  if [[ -n "${WORKDIR:-}" ]]; then
    echo "${WORKDIR}"
    return 0
  fi

  local candidate="${SCRIPT_DIR}" sub
  while [[ "${candidate}" != "/" ]]; do
    if sub="$(find_skills_subdir "${candidate}")"; then
      SKILLS_SUBDIR="${sub}"
      echo "${candidate}"
      return 0
    fi
    candidate="$(cd "${candidate}/.." && pwd)"
  done

  return 1
}

WORKDIR="$(find_workdir)" || {
  echo "Unable to locate repository root containing <anchor>/skills/${SKILL_NAME}/${VERIFY_REL}" >&2
  echo "  tried anchors: ${CANNBOT_SKILL_SUBDIRS[*]}" >&2
  echo "  set CANNBOT_WORKDIR=<project-root> to specify it explicitly" >&2
  exit 1
}

# WORKDIR 由环境变量直接给出时，SKILLS_SUBDIR 尚未确定，这里补一次探测
if [[ -z "${SKILLS_SUBDIR:-}" ]]; then
  SKILLS_SUBDIR="$(find_skills_subdir "${WORKDIR}")" || {
    echo "Missing verification script under ${WORKDIR}/${SKILL_NAME}/${VERIFY_REL}" >&2
    echo "  tried anchors: ${CANNBOT_SKILL_SUBDIRS[*]}" >&2
    exit 1
  }
fi
SKILLS_ROOT="${WORKDIR}/${SKILLS_SUBDIR}"

SSH_TARGET="${SSH_TARGET:-ascend-box}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-}"

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/root/tilelang_eval}"
CONTAINER_NAME="${CONTAINER_NAME:-zyy_cann}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/home/z00893531/tilelang-ascend}"
REMOTE_EVAL_WORKDIR="${REMOTE_EVAL_WORKDIR:-workdir_remote_eval}"
# 远端 docker 容器是外部既有环境，其 skills 目录不由本插件安装：
# 默认沿用 .claude/skills，避免改坏当前可跑的远端环境；需要时用环境变量覆盖。
REMOTE_SKILLS_SUBDIR="${REMOTE_SKILLS_SUBDIR:-.claude/skills}"
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
  REMOTE_SKILLS_SUBDIR  Skills directory inside the container (default: .claude/skills)
  ASCEND_RT_VISIBLE_DEVICES  Device id used inside the container
  CANNBOT_WORKDIR       Project root; default: auto-detect by walking up from the script,
                        trying .cannbot/skills, .claude/skills, .opencode/skills,
                        .agents/skills, .dsh/skills in that order

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

if [[ ! -f "${SKILLS_ROOT}/${SKILL_NAME}/${VERIFY_REL}" ]]; then
  echo "Missing verification script: ${SKILLS_ROOT}/${SKILL_NAME}/${VERIFY_REL}" >&2
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
    python "${SKILLS_ROOT}/${SKILL_NAME}/${VERIFY_REL}" "${TASK_DIR}"
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
python "${REMOTE_SKILLS_SUBDIR}/${SKILL_NAME}/${VERIFY_REL}" "${TASK}"
'
EOF

echo "[3/4] Running verification inside container ${CONTAINER_NAME}"
"${SSH_CMD[@]}" \
  "${SSH_TARGET}" \
  "${REMOTE_SCRIPT}"

echo "[4/4] Verification completed"
