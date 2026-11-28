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
SKILL_NAME="tilelang2ascend-translator"
VERIFY_REL="scripts/verification_ascendc.py"

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

ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-3}"
ASCENDC_CLEAN_BUILD="${ASCENDC_CLEAN_BUILD:-1}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

usage() {
  cat <<'EOF'
Usage: bash <path-to-tilelang2ascend-translator>/scripts/evaluate_ascendc.sh [task] [--non-compute]

Arguments:
  task          Task directory to verify. Defaults to current_task.
  --non-compute Non-compute operator (Cast / Memcpy), use bit-exact comparison.

Environment overrides:
  ASCENDC_SOC_VERSION        SoC version (default: Ascend910B3)
  ASCEND_RT_VISIBLE_DEVICES  Device id (default: 3)
  ASCENDC_CLEAN_BUILD        Clean build before compiling (default: 1)
  BUILD_TYPE                 CMake build type (default: Release)
  CANNBOT_WORKDIR            Project root; default: auto-detect by walking up from the script,
                             trying .cannbot/skills, .claude/skills, .opencode/skills,
                             .agents/skills, .dsh/skills in that order

Examples:
  bash <path-to-tilelang2ascend-translator>/scripts/evaluate_ascendc.sh
  bash <path-to-tilelang2ascend-translator>/scripts/evaluate_ascendc.sh current_task
  bash <path-to-tilelang2ascend-translator>/scripts/evaluate_ascendc.sh current_task --non-compute
  ASCENDC_SOC_VERSION=Ascend910B3 bash <path-to-tilelang2ascend-translator>/scripts/evaluate_ascendc.sh current_task
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${1:-current_task}"
NON_COMPUTE_FLAG=""
if [[ "${2:-}" == "--non-compute" ]]; then
  NON_COMPUTE_FLAG="--non-compute"
fi

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

if [[ ! -d "${TASK_DIR}/kernel" ]]; then
  echo "Task kernel directory not found: ${TASK_DIR}/kernel" >&2
  exit 1
fi

echo "Building kernel and running local verification"

KERNEL_DIR="${TASK_DIR}/kernel"
BUILD_DIR="${KERNEL_DIR}/build"

# Source CANN environment
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
fi

# Clean build
if [[ "${ASCENDC_CLEAN_BUILD}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake "${KERNEL_DIR}" \
  -DSOC_VERSION="${ASCENDC_SOC_VERSION}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

make -j$(nproc)

# WHL pack & install
cd "${KERNEL_DIR}"
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

cd "${WORKDIR}"

PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
  python "${SKILLS_ROOT}/${SKILL_NAME}/${VERIFY_REL}" "${TASK_DIR}" ${NON_COMPUTE_FLAG}
