#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# msnpureport 环境自检（只读，不修改任何 Device 配置）
#
# 用法:
#   bash preflight.sh [导出目录]
#
# 检查项: 工具可用性 / 版本 / 用户权限 / 容器场景 / 目录加锁与权限 / 并发进程

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TARGET_DIR="${1:-$PWD}"
WARN_COUNT=0
FAIL_COUNT=0

ok()   { printf "${GREEN}✓${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$1"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { printf "${RED}✗${NC} %s\n" "$1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
info() { printf "  %s\n" "$1"; }

printf "${CYAN}=== msnpureport 环境自检 ===${NC}\n\n"

# ---------- 1. 工具可用性 ----------
printf "${CYAN}[1] 工具可用性${NC}\n"
if command -v msnpureport >/dev/null 2>&1; then
    ok "msnpureport 已在 PATH 中: $(command -v msnpureport)"
else
    DEFAULT_TOOL_DIR="/usr/local/Ascend/driver/tools"
    if [ -x "${DEFAULT_TOOL_DIR}/msnpureport" ]; then
        warn "msnpureport 不在 PATH 中，但存在于 ${DEFAULT_TOOL_DIR}"
        info "执行: export PATH=${DEFAULT_TOOL_DIR}:\$PATH"
    else
        fail "未找到 msnpureport，请确认昇腾驱动包已部署"
        info "默认路径: {Driver安装目录}/driver/tools/msnpureport"
    fi
fi

if command -v msnpureport >/dev/null 2>&1; then
    VERSION_OUT=$(msnpureport version 2>&1 | head -1)
    [ -n "$VERSION_OUT" ] && info "版本: ${VERSION_OUT}"
fi

# 连续导出脚本（可选）
if [ -f "/usr/local/Ascend/driver/tools/msnpureport_auto_export.sh" ]; then
    ok "msnpureport_auto_export.sh 可用（脚本方式连续导出）"
else
    info "msnpureport_auto_export.sh 未在默认路径找到（推荐用 report --permanent 替代）"
fi
echo ""

# ---------- 2. 用户权限 ----------
printf "${CYAN}[2] 用户权限${NC}\n"
if [ "$(id -u)" -eq 0 ]; then
    ok "当前为 root 用户，导出与 config --set 类命令可执行"
else
    warn "当前为非 root 用户（$(id -un)）"
    info "导出类命令与所有 config --set 类命令仅支持 root 执行"
    info "查询类命令需驱动以 --install-for-all 安装才具备 Device 侧权限"
fi
echo ""

# ---------- 3. 容器场景 ----------
printf "${CYAN}[3] 运行环境${NC}\n"
IN_CONTAINER=0
if [ -f /.dockerenv ]; then
    IN_CONTAINER=1
elif [ -r /proc/1/cgroup ] && grep -qE 'docker|kubepods|containerd' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
fi

if [ "$IN_CONTAINER" -eq 1 ]; then
    warn "检测到容器环境"
    info "命令需添加 --docker 参数，并配置 PATH 环境变量"
    info "非特权容器：单次导出仅支持 Host 侧驱动内核日志，连续导出不支持"
    info "容器内无法查看 syslog，建议 msnpureport report --print 1"
else
    ok "非容器环境（裸机/Host 侧）"
fi
echo ""

# ---------- 4. 导出目录 ----------
printf "${CYAN}[4] 导出目录: %s${NC}\n" "$TARGET_DIR"
if [ ! -d "$TARGET_DIR" ]; then
    fail "目录不存在，请先创建（如 mkdir -p /var/log/npu/report）"
else
    if [ -r "$TARGET_DIR" ] && [ -w "$TARGET_DIR" ] && [ -x "$TARGET_DIR" ]; then
        ok "当前用户具备读、写、执行权限"
    else
        fail "当前用户缺少读/写/执行权限"
    fi

    # 加锁目录检查（i 属性）
    if command -v lsattr >/dev/null 2>&1; then
        ATTR=$(lsattr -d "$TARGET_DIR" 2>/dev/null | awk '{print $1}')
        if printf '%s' "$ATTR" | grep -q 'i'; then
            fail "目录已加锁（lsattr 含 i 属性），无法在此执行"
            info "如需解锁: chattr -i ${TARGET_DIR}；用后恢复: chattr +i ${TARGET_DIR}"
            info "为安全起见不建议在加锁目录中执行"
        else
            ok "目录未加锁"
        fi
    else
        info "lsattr 不可用，跳过加锁检查"
    fi

    # 普通用户可访问性（安全风险）
    DIR_PERM=$(stat -c '%a' "$TARGET_DIR" 2>/dev/null || stat -f '%Lp' "$TARGET_DIR" 2>/dev/null)
    if [ -n "${DIR_PERM:-}" ]; then
        OTHER_PERM="${DIR_PERM: -1}"
        if [ "$OTHER_PERM" != "0" ]; then
            warn "目录对其他用户开放权限（mode ${DIR_PERM}）"
            info "存在日志被恶意删除或系统信息泄露风险，建议 chmod 700"
        else
            ok "目录对普通用户无访问权限（mode ${DIR_PERM}）"
        fi
    fi

    # 剩余空间
    AVAIL=$(df -h "$TARGET_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
    [ -n "${AVAIL:-}" ] && info "可用空间: ${AVAIL}"
fi
echo ""

# ---------- 5. 并发进程 ----------
printf "${CYAN}[5] 并发检查${NC}\n"
RUNNING=$(ps -elf 2>/dev/null | grep -c "[m]snpureport" || true)
if [ "${RUNNING:-0}" -gt 0 ]; then
    warn "检测到 ${RUNNING} 个 msnpureport 相关进程正在运行"
    info "同路径并发会造成时间戳目录名冲突；单 Device 不支持并发连续导出"
    info "查看: ps -elf | grep msnpureport"
    info "终止连续导出: kill -15 <pid>"
else
    ok "无正在运行的 msnpureport 进程"
fi
echo ""

# ---------- 汇总 ----------
printf "${CYAN}=== 自检结果 ===${NC}\n"
if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    ok "全部检查通过"
elif [ "$FAIL_COUNT" -eq 0 ]; then
    printf "${YELLOW}通过，但有 %d 项警告${NC}\n" "$WARN_COUNT"
else
    printf "${RED}%d 项失败，%d 项警告${NC}\n" "$FAIL_COUNT" "$WARN_COUNT"
fi

printf "\n后续步骤: cd %s && msnpureport report\n" "$TARGET_DIR"

[ "$FAIL_COUNT" -eq 0 ] || exit 1
exit 0
