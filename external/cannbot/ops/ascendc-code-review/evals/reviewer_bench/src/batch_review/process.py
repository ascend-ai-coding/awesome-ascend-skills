#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""进程管理模块 — 子进程生命周期管理（启动 → 监控 → 清理）"""
import logging
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .config import TargetConfig
from .prompt import build_prompt, build_command, build_settings_json, build_environment
from .timeout import FixedTimeout, idle_timeout_monitor, terminate_process_group, is_alive
from .state import TaskState

import sys
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


# 所有已知的 provider API key 环境变量名
ALL_PROVIDER_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_API_KEY",
]


def _ensure_report_at_expected_path(target: TargetConfig, expected_report: Path, output_dir: Path = None) -> bool:
    """确保报告文件存在于期望路径。

    ascendc-code-review skill 的 report-write 步骤会将报告写到
    ``operators/pr-{n}/{n}_review_summary.md`` 或
    ``operators/{op}/{file}_review_summary.md``，
    而非 prompt 中指定的 ``review_report.md``。
    此函数在任务完成后检查期望路径，若不存在则从工作目录搜索并复制报告文件。
    """
    if expected_report.exists() and expected_report.stat().st_size > 0:
        return True

    search_dir = Path(target.path) if target.path else (output_dir if output_dir else None)
    if not search_dir:
        return False

    operators_dir = search_dir / "operators"
    if not operators_dir.exists():
        return False

    candidates = sorted(
        (f for f in operators_dir.rglob("*review_summary*.md") if f.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (f for f in operators_dir.rglob("*review*.md") if f.stat().st_size > 0),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        return False

    try:
        shutil.copy2(candidates[0], expected_report)
        logging.info(f"[REPORT] 报告已从 {candidates[0]} 复制到 {expected_report}")
        return True
    except (OSError, shutil.Error) as e:
        logging.error(f"⚠ 复制报告失败: {e}")
        return False


def _extract_report_from_log(log_file: Path, report_file: Path) -> bool:
    """从 opencode.log 提取 Agent 直接输出到终端的报告内容。

    当 Agent 没有使用 Write 工具写文件，而是直接在回复中输出 Markdown 时，
    opencode.log 会记录这些内容。此函数去除 ANSI 转义码后提取有效行写入 report_file。
    """
    if not log_file.exists():
        return False

    raw = log_file.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    content_lines = []
    capturing = False
    for line in lines:
        line = line.replace("\x1b", "")
        line = re.sub(r'\[[0-9;]*m', '', line)
        stripped = line.strip()

        if not stripped:
            if capturing:
                content_lines.append("")
            continue
        if stripped.startswith("> build") or stripped.startswith("$ "):
            if capturing:
                break
            continue
        if stripped.startswith("\u2192") or stripped.startswith("\u2190") or stripped.startswith("\u2713") or stripped.startswith("\u2022"):
            continue
        if stripped.startswith("[") and len(stripped) < 10:
            continue
        capturing = True
        content_lines.append(stripped)

    clean = [l for l in content_lines if l.strip()]
    if len(clean) < 3:
        return False

    try:
        report_file.write_text("\n".join(content_lines).strip() + "\n", encoding="utf-8")
        logging.info(f"[REPORT] 从日志提取报告写入 {report_file}")
        return True
    except OSError as e:
        logging.error(f"⚠ 从日志提取报告失败: {e}")
        return False


class ProcessResult:
    """子进程执行结果"""

    def __init__(self):
        self.returncode: Optional[int] = None
        self.timed_out: bool = False
        self.timeout_type: str = ""  # "fixed_timeout" | "idle_timeout" | ""
        self.error: str = ""
        self.pid: Optional[int] = None
        self.report_file: str = ""  # 检视报告文件路径
        self.attempts: int = 0       # 实际尝试次数
        self.success: bool = False   # 是否最终成功
        # Claude Code JSON 输出相关
        self.session_id: str = ""
        self.total_cost_usd: float = 0.0
        self.claude_result: str = ""


def run_task(
    target: TargetConfig,
    output_dir: Path,
    timeout_sec: int,
    idle_timeout_sec: int,
    dry_run: bool = False,
    max_retries: int = 0,
    engine: str = "opencode",
) -> ProcessResult:
    """执行单个检视任务（带超时重试）"""
    result = ProcessResult()

    # 报告文件路径
    report_file = output_dir / "review_report.md"
    result.report_file = str(report_file)

    # 构建 prompt（自动追加报告写入指令）
    prompt_text = build_prompt(target, output_dir, report_file)
    prompt_file = output_dir / "PROMPT.md"

    # 根据 engine 构建命令
    settings_file = None
    if engine == "claude":
        settings_file = build_settings_json(target, output_dir)

    cmd = build_command(target, prompt_text, output_dir, engine, settings_file)
    child_env = build_environment(target, output_dir, prompt_file, report_file, engine)

    # 日志文件命名根据 engine 区分
    if engine == "claude":
        log_file = output_dir / "claude.log"
        stderr_file = output_dir / "claude_stderr.log"
    else:
        log_file = output_dir / "opencode.log"
        stderr_file = None

    # dry-run 模式：仅打印命令
    if dry_run:
        logging.info(f"\n[DRY-RUN] 任务: {target.name}")
        logging.info(f"  引擎: {engine}")
        logging.info(f"  命令: {' '.join(cmd)}")
        if engine == "claude" and settings_file:
            logging.info(f"  Settings: {settings_file}")
        logging.info(f"  API Key 来源: {target.api_key_env}")
        logging.info(f"  输出目录: {output_dir}")
        logging.info(f"  报告文件: {report_file}")
        return result

    # 初始化任务状态
    task_state = TaskState(output_dir, target.name)
    task_state.update(report_file=str(report_file), log_file=str(log_file.name))

    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        result.attempts = attempt
        logging.info(f"\n▶ 任务 {target.name}: 第 {attempt}/{total_attempts} 次尝试...")

        # 重试前清理上次的部分报告和日志
        if attempt > 1:
            if report_file.exists():
                report_file.unlink()

            # 日志备份
            files_to_backup = [log_file]
            if engine == "claude":
                files_to_backup.append(stderr_file)

            for f in files_to_backup:
                if f and f.exists():
                    backup = output_dir / f"{f.stem}_attempt_{attempt-1}{f.suffix}"
                    if backup.exists():
                        backup.unlink()
                    f.rename(backup)

        # 启动子进程
        # 统一计算 cwd：有 path 用 path，否则用 output_dir
        cwd = str(Path(target.path).resolve()) if target.path else str(output_dir.resolve())

        if engine == "claude":
            # claude: stdout 和 stderr 分离
            stdout_fh = open(log_file, "w", encoding="utf-8", errors="replace")
            stderr_fh = open(stderr_file, "w", encoding="utf-8", errors="replace")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    start_new_session=True,
                    env=child_env,
                    cwd=cwd,
                )
            except FileNotFoundError:
                result.error = f"{engine} 可执行文件未找到"
                task_state.fail(result.error)
                stdout_fh.close()
                stderr_fh.close()
                break
            except Exception as e:
                result.error = str(e)
                task_state.fail(result.error)
                stdout_fh.close()
                stderr_fh.close()
                break
        else:
            # opencode: stderr 合并到 stdout
            log_fh = open(log_file, "w", encoding="utf-8", errors="replace")
            try:
                popen_kwargs = dict(
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    cwd=cwd,
                )
                if platform.system() == "Windows":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(
                    cmd,
                    **popen_kwargs,
                )
            except FileNotFoundError:
                result.error = f"{engine} 可执行文件未找到"
                task_state.fail(result.error)
                log_fh.close()
                break
            except Exception as e:
                result.error = str(e)
                task_state.fail(result.error)
                log_fh.close()
                break

        result.pid = proc.pid
        task_state.start(proc.pid, target.model, target.api_key_env, engine)

        # 超时回调
        timeout_event = threading.Event()
        timeout_type_holder = [""]

        def on_timeout(t_type: str):
            timeout_type_holder[0] = t_type
            timeout_event.set()
            terminate_process_group(proc.pid)

        # 启动固定超时
        fixed_timer = FixedTimeout(proc.pid, timeout_sec, on_timeout)
        fixed_timer.start()

        # 启动空闲超时监控
        idle_thread = idle_timeout_monitor(
            proc.pid, log_file, idle_timeout_sec, on_timeout
        )

        # 等待进程结束
        try:
            proc.wait()
        except KeyboardInterrupt:
            terminate_process_group(proc.pid)
            proc.wait()

        # 取消固定超时
        fixed_timer.cancel()

        # 关闭文件句柄
        if engine == "claude":
            stdout_fh.close()
            stderr_fh.close()
        else:
            log_fh.close()

        # 收集结果
        result.returncode = proc.returncode

        if timeout_event.is_set():
            result.timed_out = True
            result.timeout_type = timeout_type_holder[0]
            result.error = f"超时 ({result.timeout_type})"
            logging.warning(f"⚠ 任务 {target.name}: 第 {attempt} 次尝试超时 ({result.timeout_type})")
            task_state.timeout()
            if attempt < total_attempts:
                time.sleep(5)
                continue
            else:
                break
        else:
            task_state.complete(proc.returncode)
            if proc.returncode == 0:
                if engine == "claude":
                    # 解析 Claude JSON 输出
                    _parse_claude_output(result, log_file)
                    task_state.update(
                        session_id=result.session_id,
                        total_cost_usd=result.total_cost_usd,
                    )

                # 报告提取 fallback（opencode 引擎）
                if not report_file.exists() or report_file.stat().st_size == 0:
                    if _ensure_report_at_expected_path(target, report_file, output_dir):
                        task_state.update(output_file=str(report_file))
                    elif engine == "opencode" and _extract_report_from_log(log_file, report_file):
                        task_state.update(output_file=str(report_file))

                # 统一成功判断：报告文件存在且非空
                if report_file.exists() and report_file.stat().st_size > 0:
                    result.success = True
                    result.error = ""
                    cost_info = f" (cost: ${result.total_cost_usd:.4f})" if engine == "claude" else ""
                    logging.info(f"✅ 任务 {target.name}: 第 {attempt} 次尝试成功{cost_info}")
                    break
                else:
                    result.error = f"{engine} 退出码 0 但报告文件未生成"
                    logging.warning(f"⚠ 任务 {target.name}: {result.error}")
                    if attempt < total_attempts:
                        time.sleep(5)
                        continue
                    else:
                        break
            else:
                result.error = f"{engine} 退出码: {proc.returncode}"
                logging.error(f"⚠ 任务 {target.name}: 第 {attempt} 次尝试失败 (退出码 {proc.returncode})")
                if attempt < total_attempts:
                    time.sleep(5)
                    continue
                else:
                    break

    return result

def _parse_claude_output(result: ProcessResult, log_file: Path):
    """解析 claude --output-format stream-json 的输出（逐行 JSONL）

    逐行解析每个 event，只提取 type == "result" 的最终结果行。
    单行解析失败会跳过，不影响其他行；整体异常也不影响成功判断
    （只要报告文件存在即可），只在 error 中记录。
    """
    try:
        if not log_file.exists():
            return
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 非法行（比如被截断的最后一行）直接跳过

                if data.get("type") == "result":
                    result.session_id = data.get("session_id", "")
                    result.total_cost_usd = data.get("total_cost_usd", 0.0)
                    result.claude_result = data.get("result", "")
                    if data.get("is_error"):
                        result.error = f"claude 返回业务错误: {data.get('result', '')}"
    except Exception as e:
        result.error = f"claude 输出 JSON 解析失败: {e}"