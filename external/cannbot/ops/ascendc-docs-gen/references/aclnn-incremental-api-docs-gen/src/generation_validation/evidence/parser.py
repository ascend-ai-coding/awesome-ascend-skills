# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Evidence parser - rule-based parsing"""

import json
from typing import Optional

from src.llm.client import LLMClient

from ..core.models import (
    ParsedResult,
    RawEvidence,
    RawEvidenceChainInput,
)
from .extractor import EvidenceExtractor


class EvidenceParser:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.extractor = EvidenceExtractor(llm_client)

    @staticmethod
    def parse_anchor(anchor) -> list:
        """Parse a single anchor to extract evidence items"""
        evidences = []
        for ev in anchor.evidence:
            content = (ev.evidence_information or {}).copy()
            evidences.append(
                {
                    "evidence_id": ev.evidence_id,
                    "type": ev.source,
                    "target": content.get("target"),
                    "content": content.get("content", ""),
                    "confidence": ev.confidence,
                    "category": ev.category,
                }
            )
        return evidences

    @staticmethod
    def to_json_output(result: ParsedResult) -> list[dict]:
        """Convert parsed result to target JSON format"""
        return result.evidences

    @staticmethod
    def _get_target_groups(target, doc_part: str):
        if target is None:
            return [None]
        if isinstance(target, str):
            return [target]
        if isinstance(target, list) and doc_part == "sub_function_hash":
            first_level_keys = set()
            for t in target:
                if isinstance(t, list) and len(t) > 0:
                    first_level_keys.add(t[0])
            return list(first_level_keys)
        if isinstance(target, list):
            groups = []
            for t in target:
                if isinstance(t, list) and len(t) > 0:
                    groups.append(t[0])
                elif isinstance(t, str):
                    groups.append(t)
            if groups:
                return groups
        return [str(target)]

    def parse_raw_input(self, raw_input: RawEvidenceChainInput, deduplicate: bool = False) -> ParsedResult:
        raw_evidences = self._extract_raw_evidences(raw_input, deduplicate=deduplicate)
        doc_original_parsed = dict(raw_input.doc.sub_part) if isinstance(raw_input.doc.sub_part, dict) else {}
        doc_original_md = raw_input.doc.md_format

        evidences = []
        evidences_by_target = {}

        for ev in raw_evidences:
            content = ev.content

            target = ev.gene_rule.get("target") if ev.gene_rule else None
            target_groups = self._get_target_groups(target, raw_input.doc.doc_part)

            ev_data = {
                "evidence_id": ev.evidence_id,
                "type": ev.source,
                "target": target,
                "content": content,
                "support_chips": ev.support_chips if ev.support_chips else {},
                "dep_class": ev.dep_class,
            }
            evidences.append(ev_data)

            ev_data_for_group = {
                "evidence_id": ev.evidence_id,
                "type": ev.source,
                "content": content,
                "support_chips": ev.support_chips if ev.support_chips else {},
                "dep_class": ev.dep_class,
            }
            for target_group in target_groups:
                if target_group not in evidences_by_target:
                    evidences_by_target[target_group] = []
                evidences_by_target[target_group].append(ev_data_for_group.copy())

        # 上游锚定文件的 sub_part 常为空 dict，导致提取 prompt 中"旧版文档信息"缺失；
        # 按 sub_function_N 前缀与 md_format 小节顺序对齐，回填对应小节旧文档
        if raw_input.doc.doc_part == "sub_function_hash":
            import re as _re

            for group_key in evidences_by_target:
                m = _re.match(r"(sub_function_\d+)_(param_matrix|errorcode)", group_key)
                if not m:
                    continue
                idx = int(m.group(1).rsplit("_", 1)[1]) - 1
                section_md = doc_original_md[idx] if 0 <= idx < len(doc_original_md) else ""
                matrix_key = "param_matrix" if m.group(2) == "param_matrix" else "errorcode_matrix"
                doc_original_parsed.setdefault(matrix_key, {})
                doc_original_parsed[matrix_key].setdefault(group_key, section_md)

        return ParsedResult(
            doc_part=raw_input.doc.doc_part,
            evidences=evidences,
            doc_original_parsed=doc_original_parsed,
            doc_original_md=doc_original_md,
            total_evidences=len(evidences),
            evidences_by_target=evidences_by_target,
        )

    def _extract_raw_evidences(self, raw_input: RawEvidenceChainInput, deduplicate: bool = False) -> list[RawEvidence]:
        """Extract raw evidences from anchors"""
        return self.extractor.extract_raw_evidences(raw_input, deduplicate=deduplicate)
