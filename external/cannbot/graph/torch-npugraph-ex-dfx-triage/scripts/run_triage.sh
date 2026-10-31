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

# npugraph_ex DFX 分诊 5-step 执行器
#
# 设计目标：把分诊里的确定性部分（目录建立、env 快照、5-step 顺序执行、
# 失败即停、日志完整性检查）抽离出来，agent 只负责生成 step 脚本和调用。
#
# 用法:
#   run_triage.sh <ROOT> <CMD_TEMPLATE>
#
# 参数:
#   ROOT          本次分诊根目录（必须已存在，agent 提前 mkdir）。
#                 推荐: torch-npugraph-ex-triage-logs/<YYYYMMDD-HHMMSS>-pid<SHELL_PID>
#   CMD_TEMPLATE  原始启动命令模板，必须包含占位符 {SCRIPT}。
#                 示例:
#                   "python {SCRIPT}"
#                   "python {SCRIPT} --batch 32 --device npu:0"
#                   "torchrun --nproc_per_node=8 {SCRIPT} --config cfg.yaml"
#
# 前提（由 agent 准备）:
#   <ROOT>/scripts/ 下放置 5 个 step 脚本（缺哪个就跳过哪个，不视为失败）:
#     step1-eager.py
#     step2-compile-eager.py
#     step3-compile-aot_eager.py
#     step4-npugraph_ex-force_eager.py
#     step5-npugraph_ex.py
#   可选: original.py（用户原始脚本备份）
#
# 输出:
#   <ROOT>/env.txt                          版本快照
#   <ROOT>/<step>/stdout_stderr.log         合并打屏日志
#   <ROOT>/<step>/ascend_plog/              Ascend plog 直接落盘
#   <ROOT>/<step>/torch_compile_debug/      step4/5 才有
#   <ROOT>/<step>/SUCCESS                   该 step 通过的标记
#   <ROOT>/first_failure.txt                首个失败 step + exit code
#   <ROOT>/warnings.txt                     日志缺失/空告警
#
# 退出码:
#   0    全部 step 通过（或全部缺失）
#   1-5  首个失败 step 编号
#   64   参数错误

set -uo pipefail

usage() { sed -n '2,40p' "$0" >&2; }

if [ "$#" -ne 2 ]; then
  usage
  exit 64
fi

ROOT=$1
CMD_TEMPLATE=$2

if [ ! -d "$ROOT" ]; then
  echo "[run_triage] ROOT 目录不存在: $ROOT" >&2
  exit 64
fi
if [ ! -d "$ROOT/scripts" ]; then
  echo "[run_triage] 未找到 $ROOT/scripts/，请先把 step 脚本放进去" >&2
  exit 64
fi
case "$CMD_TEMPLATE" in
  *"{SCRIPT}"*) ;;
  *)
    echo "[run_triage] CMD_TEMPLATE 必须包含占位符 {SCRIPT}" >&2
    exit 64
    ;;
esac

# ---------- env 快照 ----------
{
  python -c "import torch; print('torch', torch.__version__)" 2>&1 || true
  python -c "import torch_npu; print('torch_npu', torch_npu.__version__)" 2>&1 || true
  python -c "import torchair; print('torchair', torchair.__version__)" 2>&1 || true
  if [ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]; then
    echo "--- CANN version.cfg ---"
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
  else
    echo "CANN version.cfg not found at /usr/local/Ascend/ascend-toolkit/latest/"
  fi
} > "$ROOT/env.txt"

# ---------- step 执行 ----------
run_step () {
  local name=$1
  local script="$ROOT/scripts/$name.py"
  local dir="$ROOT/$name"

  if [ ! -f "$script" ]; then
    echo "[run_triage] $name 跳过：$script 不存在" >&2
    return 0
  fi

  mkdir -p "$dir/ascend_plog"
  local cmd=${CMD_TEMPLATE//\{SCRIPT\}/$script}
  local status=0

  echo "[run_triage] ==> $name" >&2
  echo "[run_triage]     cmd: $cmd" >&2

  case "$name" in
    step1-*|step2-*|step3-*)
      TORCH_LOGS="+all" \
      ASCEND_PROCESS_LOG_PATH="$dir/ascend_plog" \
      ASCEND_GLOBAL_LOG_LEVEL=1 \
      ASCEND_SLOG_PRINT_TO_STDOUT=0 \
      bash -c "$cmd" > "$dir/stdout_stderr.log" 2>&1
      status=$?
      ;;
    step4-*|step5-*)
      mkdir -p "$dir/torch_compile_debug"
      TORCH_COMPILE_DEBUG=1 \
      TORCH_COMPILE_DEBUG_DIR="$dir/torch_compile_debug" \
      ASCEND_PROCESS_LOG_PATH="$dir/ascend_plog" \
      ASCEND_GLOBAL_LOG_LEVEL=1 \
      ASCEND_SLOG_PRINT_TO_STDOUT=0 \
      bash -c "$cmd" > "$dir/stdout_stderr.log" 2>&1
      status=$?
      ;;
    *)
      echo "[run_triage] 未知 step 名: $name" >&2
      return 64
      ;;
  esac

  if [ "$status" -eq 0 ]; then
    touch "$dir/SUCCESS"
  elif [ ! -f "$ROOT/first_failure.txt" ]; then
    printf '%s exit_code=%s\n' "$name" "$status" > "$ROOT/first_failure.txt"
  fi

  if [ ! -s "$dir/stdout_stderr.log" ]; then
    echo "$name stdout_stderr.log missing or empty" >> "$ROOT/warnings.txt"
  fi
  if [ ! -d "$dir/ascend_plog" ] || [ -z "$(ls -A "$dir/ascend_plog" 2>/dev/null)" ]; then
    echo "$name ascend_plog missing or empty" >> "$ROOT/warnings.txt"
  fi

  return "$status"
}

# ---------- 主流程：失败即停 ----------
STEPS=(
  "step1-eager:1"
  "step2-compile-eager:2"
  "step3-compile-aot_eager:3"
  "step4-npugraph_ex-force_eager:4"
  "step5-npugraph_ex:5"
)

for entry in "${STEPS[@]}"; do
  name=${entry%:*}
  idx=${entry##*:}
  if ! run_step "$name"; then
    rc=$?
    if [ "$rc" -eq 64 ]; then
      exit 64
    fi
    echo "[run_triage] $name 失败 (exit=$rc)，停止后续 step" >&2
    exit "$idx"
  fi
done

echo "[run_triage] 全部 step 通过" >&2
exit 0
