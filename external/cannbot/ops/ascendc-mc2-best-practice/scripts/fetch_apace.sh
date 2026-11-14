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
# fetch_apace.sh — 获取官网 ops-transformer 仓 apace 子树的最新代码
#
# 用途：ascendc-mc2-best-practice skill 不持有代码快照。文档引用的样例代码
#       通过本脚本从官网仓（gitcode.com/cann/ops-transformer）现取现读。
#
# 用法：
#   ./fetch_apace.sh                # 更新本地仓并输出 apace 子树路径
#   ./fetch_apace.sh --ref <commit> # 锚定到指定 commit（复现用）
#   APACE_REPO=/path/to/ops-transformer ./fetch_apace.sh  # 使用已有本地克隆
#   APACE_PIN_REF=<commit> ./fetch_apace.sh               # 等价 --ref（环境变量形式）
#
# ⚠️ master 漂移警示：官网 master 的 apace 目录结构会演进（如 block/→core/ 迁移、
#   kernel/fusions/ 层级、新增算子）。skill 文档以 pin 的已验证快照为结构基准；
#   拉取 master 仅用于契约核对，使用前必须 diff 校验并在文档引用失效时更新文档。
#
# 行为：
#   1. APACE_REPO 已设置且含 apace 子树 → git fetch 更新该仓（零重复克隆）
#   2. 否则 sparse clone（--filter=blob:none --sparse，仅 apace 子树）到缓存目录
#      默认缓存：~/.cache/apace-reference/ops-transformer
#   3. 输出：APACE_ROOT=<本地 apace 子树绝对路径>
#   4. 写入 manifest：<缓存或仓>/.apace_fetch_manifest.json

set -euo pipefail

REPO_URL="https://gitcode.com/cann/ops-transformer.git"
APACE_SUBPATH="mc2/common/op_kernel/apace"
CACHE_DIR="${APACE_CACHE_DIR:-$HOME/.cache/apace-reference/ops-transformer}"
REF="${APACE_PIN_REF:-master}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[fetch_apace] $*" >&2; }

if [[ -n "${APACE_REPO:-}" ]]; then
    if [[ ! -d "${APACE_REPO}/${APACE_SUBPATH}" ]]; then
        log "错误：APACE_REPO=${APACE_REPO} 下不存在 ${APACE_SUBPATH}"
        exit 1
    fi
    REPO_DIR="${APACE_REPO}"
    log "使用已有本地克隆：${REPO_DIR}"
    if [[ "${REF}" == "master" ]]; then
        log "git fetch 更新中..."
        git -C "${REPO_DIR}" fetch origin master --quiet || log "警告：fetch 失败（可能离线），使用本地现有状态"
    fi
else
    REPO_DIR="${CACHE_DIR}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "缓存已存在：${REPO_DIR}，更新中..."
        git -C "${REPO_DIR}" fetch origin master --quiet || log "警告：fetch 失败（可能离线），使用本地现有状态"
    else
        log "sparse clone 官网仓（仅 ${APACE_SUBPATH} 子树）到 ${REPO_DIR} ..."
        mkdir -p "$(dirname "${REPO_DIR}")"
        git clone --filter=blob:none --sparse --quiet "${REPO_URL}" "${REPO_DIR}"
        git -C "${REPO_DIR}" sparse-checkout set "${APACE_SUBPATH}"
    fi
fi

if [[ "${REF}" != "master" ]]; then
    log "锚定到 ${REF}"
    git -C "${REPO_DIR}" checkout --quiet "${REF}"
else
    git -C "${REPO_DIR}" checkout --quiet origin/master 2>/dev/null || git -C "${REPO_DIR}" checkout --quiet master
fi

APACE_ROOT="${REPO_DIR}/${APACE_SUBPATH}"
COMMIT=$(git -C "${REPO_DIR}" rev-parse --short HEAD)
DATE=$(date -Iseconds)

cat > "${REPO_DIR}/.apace_fetch_manifest.json" <<EOF
{
  "source_repo": "${REPO_URL}",
  "subpath": "${APACE_SUBPATH}",
  "commit": "${COMMIT}",
  "ref": "${REF}",
  "fetched_at": "${DATE}",
  "apace_root": "${APACE_ROOT}"
}
EOF

log "完成：commit=${COMMIT}"
echo "APACE_ROOT=${APACE_ROOT}"
