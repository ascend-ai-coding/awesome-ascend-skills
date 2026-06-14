#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKS_DIR="${SCRIPT_DIR}/../checks"

required_headers=(
  "## 输入"
  "## 命令"
  "## 通过标准"
  "## 失败回流"
)

if [[ ! -d "${CHECKS_DIR}" ]]; then
  echo "[ERROR] checks 目录不存在: ${CHECKS_DIR}" >&2
  exit 1
fi

shopt -s nullglob
check_files=("${CHECKS_DIR}"/*.md)
shopt -u nullglob

if [[ ${#check_files[@]} -eq 0 ]]; then
  echo "[ERROR] 未找到 checks/*.md 文件" >&2
  exit 1
fi

failed=0

for file in "${check_files[@]}"; do
  echo "[CHECK] ${file}"
  prev_line=0

  for header in "${required_headers[@]}"; do
    line="$(awk -v h="${header}" '$0==h{print NR; exit}' "${file}")"

    if [[ -z "${line}" ]]; then
      echo "  [FAIL] 缺少标题: ${header}" >&2
      failed=1
      continue
    fi

    if [[ "${line}" -le "${prev_line}" ]]; then
      echo "  [FAIL] 标题顺序错误: ${header}" >&2
      failed=1
      continue
    fi

    echo "  [OK] ${header} (line ${line})"
    prev_line="${line}"
  done
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[RESULT] checks 布局校验失败" >&2
  exit 1
fi

echo "[RESULT] checks 布局校验通过"
