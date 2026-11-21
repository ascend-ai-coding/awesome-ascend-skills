# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Template loader - loads document templates from YAML files"""

from pathlib import Path
from typing import Optional

import yaml


class TemplateLoader:
    def __init__(self, template_file: Optional[Path] = None):
        self.template_file = template_file
        self._templates = None

    @staticmethod
    def from_file(file_path: Path) -> "TemplateLoader":
        return TemplateLoader(template_file=file_path)

    def get_template(self, doc_part: str, sub_key: Optional[str] = None) -> dict:
        templates = self._load()
        doc_parts = templates.get("doc_parts", {})
        part_data = doc_parts.get(doc_part, {})

        if not part_data:
            return {}

        if sub_key:
            return part_data.get(sub_key, {})

        if "label" not in part_data:
            return part_data

        return part_data

    def get_label(self, doc_part: str, sub_key: Optional[str] = None) -> str:
        template = self.get_template(doc_part, sub_key)
        if sub_key:
            return template.get("label", sub_key or doc_part)
        return template.get("label", doc_part)

    def get_requirement(self, doc_part: str, sub_key: Optional[str] = None) -> str:
        template = self.get_template(doc_part, sub_key)
        return template.get("requirement", "")

    def get_template_example(self, doc_part: str, sub_key: Optional[str] = None) -> str:
        template = self.get_template(doc_part, sub_key)
        return template.get("template", "")

    def has_nested(self, doc_part: str) -> bool:
        templates = self._load()
        doc_parts = templates.get("doc_parts", {})
        part = doc_parts.get(doc_part, {})
        if not part:
            return False
        return "label" not in part

    def get_nested_keys(self, doc_part: str) -> list:
        templates = self._load()
        doc_parts = templates.get("doc_parts", {})
        part = doc_parts.get(doc_part, {})
        if self.has_nested(doc_part):
            return list(part.keys())
        return []

    def _load(self) -> dict:
        if self._templates is not None:
            return self._templates

        if not self.template_file or not self.template_file.exists():
            return {}

        with open(self.template_file, "r", encoding="utf-8") as f:
            self._templates = yaml.safe_load(f)

        return self._templates
