#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# ascendc-evolution-strategies — 知识仓显式绑定 + 本地知识库初始化
#
# 用法:
#   bash init.sh [--project <dir>] [--knowledge-root <cannbot-knowledge 路径>] [--check]
#
# 行为:
#   1. 解析知识仓根（CLI > CANNBOT_KNOWLEDGE_ROOT > .cannbot/knowledge.env > 兄弟检出）
#   2. 写入 <project>/.cannbot/knowledge.env（含 root / mode / commit）
#   3. 初始化 <project>/.cannbot/knowledge-local/{staging,promoted,rejected}
#   4. 跑一次 knowledge_smoke.py，证明知识库确实被加载
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT="$PWD"
KNOWLEDGE_ROOT=""
CHECK_ONLY=0

show_help() {
    cat <<'EOF'
ascendc-evolution-strategies init

用法:
  bash init.sh [选项]

选项:
  --project <dir>         项目根（默认当前目录）；写入 .cannbot/knowledge.env
  --knowledge-root <dir>  cannbot-knowledge 仓根（显式绑定，优先级最高）
  --check                 只做校验 + smoke，不写配置
  --help                  显示帮助

解析优先级:
  --knowledge-root > CANNBOT_KNOWLEDGE_ROOT > 项目向上 .cannbot/knowledge.env > 兄弟检出
  显式来源配置无效时直接报错退出（不会静默换仓）。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)         PROJECT="$2"; shift 2 ;;
        --knowledge-root)  KNOWLEDGE_ROOT="$2"; shift 2 ;;
        --check)           CHECK_ONLY=1; shift ;;
        --help|-h)         show_help; exit 0 ;;
        *) echo "Error: unknown argument '$1'" >&2; show_help; exit 2 ;;
    esac
done

PY_ARGS=()
[[ -n "$KNOWLEDGE_ROOT" ]] && PY_ARGS+=(--knowledge-root "$KNOWLEDGE_ROOT")

# 旧布局（正文仍在本地 references/playbooks）用 local corpus 校验；
# 新布局（Playbook 已迁知识仓）用默认 knowledge corpus。
CORPUS_ARGS=()
if [[ -d "$SCRIPT_DIR/references/playbooks" && -d "$SCRIPT_DIR/references/cards" ]]; then
    CORPUS_ARGS=(--corpus local)
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/knowledge_smoke.py" "${PY_ARGS[@]+"${PY_ARGS[@]}"}" "${CORPUS_ARGS[@]+"${CORPUS_ARGS[@]}"}"
fi

echo "[1/4] 解析并校验知识仓..."
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/knowledge_root.py" "${PY_ARGS[@]+"${PY_ARGS[@]}"}" --validate

echo "[2/4] 写入项目配置: $PROJECT/.cannbot/knowledge.env"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/knowledge_root.py" "${PY_ARGS[@]+"${PY_ARGS[@]}"}" \
    --write-env --project "$PROJECT" --mode consumer

echo "[3/4] 初始化本地知识库: $PROJECT/.cannbot/knowledge-local"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/knowledge_local.py" init --project "$PROJECT"

echo "[4/4] 知识库加载 smoke..."
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/knowledge_smoke.py" "${PY_ARGS[@]+"${PY_ARGS[@]}"}" "${CORPUS_ARGS[@]+"${CORPUS_ARGS[@]}"}"

echo "OK: knowledge root bound to $(grep -m1 CANNBOT_KNOWLEDGE_ROOT "$PROJECT/.cannbot/knowledge.env")"
