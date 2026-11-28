# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Style summarizer - extracts core topic and key points from document"""

import json
from pathlib import Path
from typing import Optional

from src.llm.client import LLMClient

from ..core.models import StyleSummary


class StyleSummarizer:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        style_file: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.style_file = style_file

    @staticmethod
    def get_from_file(file_path: Path) -> StyleSummary:
        """
        直接从 JSON 文件读取 StyleSummary。
        支持结构降级：允许 "style" 嵌套字段，或直接是 StyleSummary 结构。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"风格文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 支持两种结构：
        # 1. 根节点就是 StyleSummary
        # 2. 在 "style" 字段下
        summary_data = data.get("style", data)  # 降级支持

        required_keys = {"core_topic", "key_points"}
        if not required_keys.intersection(summary_data.keys()):
            raise ValueError(f"文件格式错误：缺少必要字段 {required_keys}")

        return StyleSummary(
            core_topic=summary_data.get("core_topic", ""),
            key_points=summary_data.get("key_points", []),
            context_dependencies=summary_data.get("context_dependencies", []),
        )

    def summarize(self, doc_content: str) -> StyleSummary:
        if self.style_file and self.style_file.exists():
            with open(self.style_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StyleSummary(**data)

        if self.llm is None:
            return StyleSummary(
                core_topic="",
                key_points=[],
                context_dependencies=[],
            )

        prompt = f"""分析以下文档，提取核心主题和关键信息点：

文档内容:
{doc_content[:10000]}

输出 JSON:
{{
    "core_topic": "核心主题（一句话）",
    "key_points": ["关键点1", "关键点2"],
    "context_dependencies": ["需要的前置知识"]
}}"""

        result = self.llm.chat_with_json(
            [
                {"role": "system", "content": "你是一个文档分析助手"},
                {"role": "user", "content": prompt},
            ]
        )
        return StyleSummary(**result)

    def summarize_and_save(self, doc_content: str, output_path: str, force: bool = False):
        output_path = Path(output_path)
        if output_path.exists() and not force:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summarize(doc_content)
        output_path.write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def gen_doc_summarize(doc_content: str, output_path: str, force: bool = False):
    from src.llm.client import LLMClient

    llm_client = LLMClient()
    summarizer = StyleSummarizer(llm_client=llm_client)
    summarizer.summarize_and_save(doc_content, output_path, force)
