#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Prompt 构建模块 — 生成 PROMPT.md + 构建执行命令（opencode/claude 双引擎）"""
import json
import os
from pathlib import Path
from typing import List, Optional

from .config import TargetConfig, _infer_provider_key


def build_prompt(target: TargetConfig, output_dir: Path, report_file: Path) -> str:
    """为单个任务构建完整 prompt 并生成 PROMPT.md
    
    自动追加报告写入指令，确保 skill 将检视结果输出到指定文件。
    """
    prompt_text = target.prompt
    
    # 追加报告写入指令
    report_instruction = f"\n\n检视完成后，必须将报告写入 {report_file.absolute()}（不要写到其他位置）。"
    full_prompt = prompt_text + report_instruction
    
    # 如果有 extra_files，在 prompt 中告知路径（转为绝对路径）
    if target.extra_files:
        files_str = "\n".join(f"  - {Path(f).absolute()}" for f in target.extra_files)
        full_prompt += f"\n\n请额外参考以下文件：\n{files_str}"
    
    # 写入 PROMPT.md（供调试/复盘）
    prompt_file = output_dir / "PROMPT.md"
    prompt_file.write_text(full_prompt, encoding="utf-8")
    
    return full_prompt


def build_command(
    target: TargetConfig,
    prompt_text: str,
    output_dir: Path,
    engine: str,
    settings_file: Optional[Path] = None,
) -> List[str]:
    """构建执行命令（根据 engine 分派）"""
    if engine == "claude":
        return _build_claude_command(target, prompt_text, settings_file)
    else:
        return _build_opencode_command(target, prompt_text, output_dir)


def _build_opencode_command(
    target: TargetConfig,
    prompt_text: str,
    output_dir: Path,
) -> List[str]:
    """构建 opencode run 命令参数列表"""
    import shutil
    opencode_bin = shutil.which("opencode") or "opencode"
    
    # npm 安装的 opencode 可能是 .CMD/.ps1 包装器，传递含换行符的长参数时会截断
    # 优先直接使用 .exe，避免 PowerShell/cmd 包装器的参数传递问题
    if opencode_bin.endswith(('.CMD', '.ps1')):
        exe_candidate = Path(opencode_bin).parent / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
        if exe_candidate.exists():
            opencode_bin = str(exe_candidate)
    
    cmd = [opencode_bin, "run", "--dangerously-skip-permissions"]
    
    if target.agent:
        cmd.extend(["--agent", target.agent])
    if target.model:
        cmd.extend(["-m", target.model])
    
    # 设置 --dir：有 path 用 path，无 path（如 eval 任务）用 output_dir
    if target.path:
        dir_path = Path(target.path).resolve()
    else:
        dir_path = output_dir.resolve()
    cmd.extend(["--dir", str(dir_path)])
    
    for extra_file in target.extra_files:
        if target.path:
            extra_path = Path(target.path) / extra_file
        else:
            extra_path = Path(extra_file)
        if extra_path.exists():
            cmd.extend(["-f", str(extra_path.absolute())])
    
    cmd.append("--")
    cmd.append(prompt_text)
    return cmd


def _build_claude_command(
    target: TargetConfig,
    prompt_text: str,
    settings_file: Path,
) -> List[str]:
    """构建 claude headless 命令"""
    cmd = [
        "claude",
        "-p",
        prompt_text,
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep",
        "--permission-mode", "acceptEdits",
        "--settings", str(settings_file.absolute()),
    ]

    if target.model:
        cmd.extend(["--model", target.model])

    if target.agent:
        cmd.extend(["--append-system-prompt", f"You are acting as: {target.agent}"])

    return cmd


def build_settings_json(
    target: TargetConfig,
    output_dir: Path,
) -> Path:
    """为 claude 引擎生成独立 settings.json（Key 隔离）
    
    返回 settings.json 的路径
    """
    api_key_value = os.environ.get(target.api_key_env, "")
    
    settings = {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": api_key_value,
        }
    }
    
    if target.base_url:
        settings["env"]["ANTHROPIC_BASE_URL"] = target.base_url
    
    settings_file = output_dir / "settings.json"
    settings_file.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return settings_file


def build_environment(
    target: TargetConfig,
    output_dir: Path,
    prompt_file: Path,
    report_file: Path,
    engine: str,
) -> dict:
    """构建子进程环境变量"""
    child_env = os.environ.copy()
    
    # 共通部分：辅助环境变量
    child_env["REPORT_FILE"] = str(report_file.absolute())
    child_env["BENCHMARK_PROMPT_FILE"] = str(prompt_file.absolute())
    child_env["BENCHMARK_OUTPUT_DIR"] = str(output_dir.absolute())
    child_env["BENCHMARK_REPORT_FILE"] = str(report_file.absolute())
    if target.path:
        child_env["BENCHMARK_WORKDIR"] = target.path
    
    if engine == "opencode":
        # Key 隔离逻辑：注入 provider key + 清空其他
        api_key_value = os.environ.get(target.api_key_env, "")
        provider_key = target.provider_key or _infer_provider_key(target.api_key_env)
        
        if provider_key:
            child_env[provider_key] = api_key_value
        
        all_provider_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY",
            "AZURE_API_KEY",
        ]
        for key in all_provider_keys:
            if key != provider_key:
                child_env[key] = ""
    # claude 引擎：Key 隔离由 settings.json 负责，环境变量不注入
    
    return child_env