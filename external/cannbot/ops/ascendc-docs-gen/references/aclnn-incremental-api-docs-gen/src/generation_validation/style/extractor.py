# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Style extractor - extracts writing style from document"""

import json
from pathlib import Path
from typing import Optional

from src.llm.client import LLMClient

from ..core.models import StyleExtraction


class StyleExtractor:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        style_file: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.style_file = style_file

    @staticmethod
    def get_from_file(file_path: Path) -> StyleExtraction:
        """
        直接从 JSON 文件读取 StyleExtraction。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"style 文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return StyleExtraction(
            terminology=data.get("terminology", {}),
            structure=data.get("structure", {}),
            format=data.get("format", {}),
            punctuation=data.get("punctuation", {}),
            tone=data.get("tone", {}),
        )

    def extract(self, doc_content: str) -> StyleExtraction:
        if self.style_file and self.style_file.exists():
            with open(self.style_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StyleExtraction(**data)

        if self.llm is None:
            return StyleExtraction(
                terminology={},
                structure={},
                format={},
                punctuation={},
                tone={},
            )

        prompt = f"""分析以下文档的写作风格，特别关注格式细节：

文档内容:
{doc_content[:3000]}

输出 JSON（注意：format字段要详细描述格式特点，不是简单的是否使用）:
{{
    "terminology": {{
        "high_freq_terms": ["术语1"],
        "synonyms": {{"常用词": "同义词"}}
    }},
    "structure": {{
        "avg_sentence_length": 20,
        "short_long_ratio": "3:7"
    }},
    "format": {{
        "table_style": "表格样式描述，如：2列对比表、产品支持矩阵",
        "code_style": "代码块样式，如：函数原型、完整调用示例、片段示例",
        "heading_structure": "标题层级及用途，如：[2级=功能名, 3级=子功能]",
        "list_style": "列表样式，如：短横线列表、编号列表、混合使用",
        "special_markers": ["特殊标记1", "特殊标记2"]
    }},
    "punctuation": {{
        "preferred_marks": [";", "："]
    }},
    "tone": {{
        "person": "third",
        "tense": "present"
    }}
}}"""

        result = self.llm.chat_with_json(
            [
                {"role": "system", "content": "你是一个写作风格分析助手"},
                {"role": "user", "content": prompt},
            ]
        )
        return StyleExtraction(**result)
