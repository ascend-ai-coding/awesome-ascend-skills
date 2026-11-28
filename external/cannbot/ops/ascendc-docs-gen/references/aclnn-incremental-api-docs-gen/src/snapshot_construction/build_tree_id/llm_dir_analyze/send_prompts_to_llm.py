#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
调用 LLM API 处理组合 prompt，将响应拆分为每个目录独立的 JSON 文件。

原有方式：每个目录单独调用 LLM API，大量 HTTP 请求效率低。
当前方式：读取组合 prompt 一次性调用 LLM，解析 JSON 响应，拆分成多个响应文件
（保持与 Step 4 兼容的输出格式和目录结构）。
"""

import json
import re
from pathlib import Path
from typing import Optional

from src.llm.client import LLMClient

from src.snapshot_construction.pub_ref import PubRefConfig

from src.logger import info, debug, warning, error


def extract_json_from_response(response_text: str) -> Optional[dict]:
    """
    从 LLM 响应中提取 JSON 对象。
    支持多种格式：纯 JSON、代码块包裹的 JSON、大括号包裹的 JSON。

    Args:
        response_text: LLM 原始响应文本

    Returns:
        解析成功的字典，或 None
    """
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json ... ``` 或 ``` ... ``` 代码块中提取
    code_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
    )
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试从外层大括号中提取
    brace_match = re.search(r"\{[\s\S]*\}", response_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def main(out_path):
    """主函数：读取组合 prompt，一次性调用 LLM，拆分响应"""
    prompts_dir = out_path / "new_prompts"
    output_dir = out_path / "llm_responses"

    # 查找组合 prompt 文件
    prompt_files = list(prompts_dir.glob("*.txt"))
    if not prompt_files:
        error(f"错误: 在 {prompts_dir} 中未找到 prompt 文件")
        return

    combined_prompt_file = prompt_files[0]
    info(f"读取组合 prompt: {combined_prompt_file}")

    with open(combined_prompt_file, "r", encoding="utf-8") as f:
        prompt_content = f.read()

    info(f"Prompt 长度: {len(prompt_content)} 字符")

    # 创建 LLMClient（内置重试逻辑）
    llm_client = LLMClient(
        api_key=PubRefConfig.INCREMENTAL_LLM_KEY,
        base_url=PubRefConfig.INCREMENTAL_LLM_URL,
        model=PubRefConfig.MODEL,
    )

    # 发起一次 LLM 调用
    info("调用 LLM API（一次性请求，包含所有目录）...")
    try:
        response = llm_client.chat([{"role": "user", "content": prompt_content}])
    except Exception as e:
        error(f"LLM API 调用失败: {e}")
        return

    info("LLM API 调用成功，解析响应...")

    # 解析 JSON 响应
    dir_descriptions = extract_json_from_response(response)

    if not dir_descriptions:
        error("错误: 无法从 LLM 响应中解析 JSON")
        error(f"原始响应前 500 字符:\n{response[:500]}")
        # 保存原始响应以供调试
        debug_file = output_dir / "_raw_response.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(response)
        info(f"原始响应已保存到: {debug_file}")
        return

    info(f"成功解析 {len(dir_descriptions)} 个目录描述")

    # 拆分为每个目录独立的 JSON 文件（保持与 Step 4 兼容的格式）
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for dir_path, description in dir_descriptions.items():
        # dir_path 如 "objdetect/roi_align/op_host"
        parts = Path(dir_path).parts
        dir_name = parts[-1] if parts else "root"

        # 创建输出路径: llm_responses/objdetect/roi_align/op_host/op_host.json
        output_subdir = output_dir / dir_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{dir_name}.json"

        # prompt_path 用于 Step 4 提取目录路径（Step 4 会去掉文件名得到目录路径）
        prompt_path = f"{dir_path}/{dir_name}.txt"

        response_data = {
            "prompt_path": prompt_path,
            "response": description,
            "success": True,
            "error": None,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

        saved_count += 1
        info(f"  已保存: {output_file.relative_to(output_dir)}")

    info(f"完成: 已保存 {saved_count} 个目录描述文件到 {output_dir}")


if __name__ == "__main__":
    main()
