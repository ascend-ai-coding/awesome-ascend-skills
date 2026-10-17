#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Locate the asys executable and report whether its prerequisites are in place.
# Read-only: probes paths and runs help queries only. Never modifies the environment.

set -uo pipefail

ASYS_BIN=""
EXIT_CODE=0

note() { printf '%s\n' "$*"; }
fail() { printf '[MISSING] %s\n' "$*"; EXIT_CODE=1; }

note "=== asys 定位 ==="

# set_env.sh puts asys on PATH; fall back to known install layouts.
if command -v asys >/dev/null 2>&1; then
    ASYS_BIN=$(command -v asys)
    note "[OK] PATH 中找到 asys: ${ASYS_BIN}"
else
    note "[INFO] PATH 中未找到 asys，尝试 CANN 安装目录"
    CANDIDATES=()
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        CANDIDATES+=("${ASCEND_HOME_PATH}/tools/ascend_system_advisor/asys/asys")
    fi
    CANDIDATES+=(
        "/usr/local/Ascend/cann/tools/ascend_system_advisor/asys/asys"
        "/usr/local/Ascend/ascend-toolkit/latest/tools/ascend_system_advisor/asys/asys"
        "${HOME}/Ascend/ascend-toolkit/latest/tools/ascend_system_advisor/asys/asys"
    )
    for candidate in "${CANDIDATES[@]}"; do
        if [ -x "$candidate" ]; then
            ASYS_BIN="$candidate"
            note "[OK] 找到 asys: ${ASYS_BIN}"
            break
        fi
    done
fi

if [ -z "$ASYS_BIN" ]; then
    fail "未找到 asys 可执行文件"
    note "  检查项:"
    note "    1. 是否已 source \${INSTALL_DIR}/set_env.sh（设置后可直接使用 asys 命令）"
    note "    2. CANN Toolkit 是否已安装"
    note "    3. ASCEND_HOME_PATH 当前值: ${ASCEND_HOME_PATH:-未设置}"
else
    note ""
    note "=== 帮助信息 ==="
    if ! "$ASYS_BIN" --help 2>&1 | head -20; then
        note "[WARN] asys --help 执行失败，工具可能不完整"
    fi
fi

note ""
note "=== 形态检查 ==="
note "asys 仅支持在 Ascend EP 形态下使用，RC 形态不支持。"
if [ -e /dev/davinci_manager ] || [ -e /dev/davinci0 ]; then
    note "[OK] 检测到 davinci 设备节点"
else
    note "[WARN] 未检测到 /dev/davinci* 设备节点，确认驱动已安装且当前用户有访问权限"
fi

note ""
note "=== 外部依赖 ==="
check_dep() {
    local cmd="$1" purpose="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        note "[OK] ${cmd} — ${purpose}"
    else
        fail "${cmd} 缺失 — ${purpose}"
    fi
}
check_dep gdb "analyze -r=coredump 解析 core 文件"
check_dep readelf "analyze -r=stackcore 获取文件信息"
check_dep addr2line "analyze -r=stackcore / -r=coretrace 解析函数名和行号"

note ""
note "=== 权限与环境 ==="
if [ "$(id -u)" -eq 0 ]; then
    note "[OK] 当前为 root，可执行 diagnose / config（还需物理机），"
    note "     且可收集 Device 侧固件日志、系统日志、黑匣子、stackcore、coretrace"
else
    note "[WARN] 当前非 root（$(id -un)）。diagnose / config 必须在物理机且 root 用户下执行；"
    note "       Device 侧固件日志、系统日志、黑匣子、stackcore、coretrace 将收集不到"
fi

ATRACE_DIR="${HOME}/ascend/atrace"
if [ -d "$ATRACE_DIR" ]; then
    trace_count=$(find "$ATRACE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    note "[INFO] trace 日志目录 ${ATRACE_DIR} 下有 ${trace_count} 个文件"
    if [ "$trace_count" -gt 1000 ]; then
        note "[WARN] 文件较多。asys 会检索该目录，可能导致执行时间长，建议先清理"
    fi
else
    note "[INFO] trace 日志目录 ${ATRACE_DIR} 不存在（未产生过 trace 日志）"
fi

note ""
note "=== 影响收集范围的环境变量 ==="
note "执行 asys 时这些变量的取值需与业务运行时一致，否则收集到的信息可能不准确:"
for var in ASCEND_PROCESS_LOG_PATH NPU_COLLECT_PATH DUMP_GRAPH_PATH \
           ASCEND_WORK_PATH ASCEND_CACHE_PATH ASCEND_CUSTOM_OPP_PATH \
           ASCEND_OPP_PATH ASCEND_COREDUMP_SIGNAL; do
    note "  ${var}=${!var:-未设置}"
done

if [ "${ASCEND_COREDUMP_SIGNAL:-}" = "none" ]; then
    note ""
    note "[WARN] ASCEND_COREDUMP_SIGNAL=none：trace 处理的信号集已关闭。"
    note "       实时堆栈导出需向进程发送信号 35，此配置下会终止卡住进程且无法导出堆栈。"
    note "       导出前需将该变量设为非 none 的值或取消设置"
fi

note ""
if [ "$EXIT_CODE" -eq 0 ]; then
    note "=== 结论: asys 可用，依赖齐备 ==="
else
    note "=== 结论: 存在缺失项，见上方 [MISSING] ==="
fi
exit "$EXIT_CODE"
