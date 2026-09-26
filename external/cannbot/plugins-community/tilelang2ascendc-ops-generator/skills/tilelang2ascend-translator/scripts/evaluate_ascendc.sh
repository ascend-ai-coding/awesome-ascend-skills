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
    if [[ -f "${candidate}/.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" ]]; then
      echo "${candidate}"
      return 0
    fi
    candidate="$(cd "${candidate}/.." && pwd)"
  done

  return 1
}

WORKDIR="$(find_workdir)" || {
  echo "Unable to locate repository root containing .claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" >&2
  exit 1
}

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

if [[ ! -f "${WORKDIR}/.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" ]]; then
  echo "Missing verification script: ${WORKDIR}/.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" >&2
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
  python .claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py "${TASK_DIR}" ${NON_COMPUTE_FLAG}
