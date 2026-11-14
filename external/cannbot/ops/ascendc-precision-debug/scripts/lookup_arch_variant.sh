#!/bin/bash
# lookup_arch_variant.sh — 确认目标芯片实际命中的 impl 变体（NpuArch 宏分支 / dav_* 目录）
#
# 背景：$ASCEND_HOME_PATH 下 dav_* 头文件按 __NPU_ARCH__ 宏分芯片变体并存，
#       grep 命中的变体文件可能不是当前芯片实际编译的变体——把非生效变体
#       （如 dav_c100）的断言误当事实，曾导致多轮"某模式不支持"误判。
#       证据链：ini 的 SoC_version(芯片型号) → NpuArch(宏值) / CCEC_AIV_version(dav 变体)。
#
# 用法：lookup_arch_variant.sh [SoC型号]   # 如 Ascend950PR_950z；缺省尝试 npu-smi 探测
# 输出：NpuArch（__NPU_ARCH__ 宏分支）与生效变体目录名；读 impl/断言只看该变体。
set -uo pipefail

SOC="${1:-}"
if [ -z "$SOC" ]; then
  SOC="$(npu-smi info 2>/dev/null | grep -oE 'Ascend[0-9A-Za-z]+' | head -1 || true)"
  [ -z "$SOC" ] && { echo "ERROR: 自动探测失败，请显式传参：$0 <SoC型号>（如 Ascend950PR_950z）" >&2; exit 2; }
  echo "# 自动探测 SoC=$SOC（不精确时请显式传参）" >&2
fi

[ -z "${ASCEND_HOME_PATH:-}" ] && { echo "ERROR: ASCEND_HOME_PATH 未设置" >&2; exit 2; }
INI_DIR="$ASCEND_HOME_PATH/x86_64-linux/data/platform_config"
[ -d "$INI_DIR" ] || { echo "ERROR: 未找到 $INI_DIR" >&2; exit 2; }

# 第一优先级：SoC_version 精确匹配（禁止文件名通配/前缀模糊——910B 与 910B1 的 NpuArch 不同）
INI="$(grep -l "^SoC_version=${SOC}$" "$INI_DIR"/*.ini 2>/dev/null | head -1 || true)"
MATCH="exact"
# 回退：型号不带后缀时做"型号_变体"前缀匹配（如 Ascend950PR → Ascend950PR_950z），多候选须报全
if [ -z "$INI" ]; then
  CAND="$(grep -lE "^SoC_version=${SOC}[_.]" "$INI_DIR"/*.ini 2>/dev/null || true)"
  N="$(echo "$CAND" | grep -c . || true)"
  if [ "$N" -ge 1 ]; then
    INI="$(echo "$CAND" | head -1)"; MATCH="prefix"
    if [ "$N" -gt 1 ]; then
      echo "WARN: $SOC 前缀匹配到 $N 个 ini，请核对 NpuArch 是否一致：" >&2
      for f in $CAND; do echo "  $(basename "$f"): $(grep '^NpuArch=' "$f" || echo 'NpuArch 缺失')" >&2; done
    fi
  fi
fi
if [ -z "$INI" ]; then
  echo "ERROR: $INI_DIR 下无 SoC_version 匹配 $SOC 的 ini" >&2
  HINT="$(grep -h "^SoC_version=${SOC}" "$INI_DIR"/*.ini 2>/dev/null | cut -d= -f2 | sort -u || true)"
  [ -n "$HINT" ] && echo "相近的 SoC_version（请精确指定其一）："$'\n'"$HINT" >&2
  exit 1
fi

ARCH="$(grep '^NpuArch=' "$INI" | head -1 | cut -d= -f2 | tr -d '[:space:]')"
AIV="$(grep '^CCEC_AIV_version=' "$INI" | head -1 | cut -d= -f2 | tr -d '[:space:]')"   # 如 dav-c310-vec
VAR="$(echo "$AIV" | sed -E 's/^dav-([a-z0-9]+)-vec$/dav_\1/')"

echo "SoC=$SOC (match=$MATCH)"
echo "ini=$INI"
echo "NpuArch=$ARCH      # 读头文件只走 __NPU_ARCH__ == $ARCH 宏分支"
echo "AIV=$AIV"
echo "variant_dir=$VAR   # 读 impl/断言只看 **/$VAR/ 下的文件，其他变体目录的断言不构成证据"
