# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Core data models for the document auto-editor system"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class DocPart(str, Enum):
    SUPPORT = "support"
    SUMMARY = "summary"
    FUNCTION_PROTO = "function_proto"
    SUB_FUNCTION_HASH = "sub_function_hash"
    CONSTRAINT = "constraint"
    SUB_FUNCTION_SCATTER = "sub_function_scatter"


class EvidenceCategory(str, Enum):
    FACT = "fact"
    DESC = "desc"


class EvidenceSource(str, Enum):
    CODE_DIFF = "code_diff"
    CODE_COMMENT = "code_comment"
    LLM_GENERATED = "llm_generated"


class AnchorType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    NOTE = "note"
    SECTION = "section"
    FILE_DEF = "file_def"


class Evidence(BaseModel):
    evidence_id: str
    content: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: EvidenceCategory

    def get_evidence_id(self) -> str:
        return self.evidence_id


class Anchor(BaseModel):
    anchor_id: str
    note: str
    anchor_type: AnchorType
    anchor_information: dict
    evidence: list[Evidence]


class DocBlock(BaseModel):
    doc_name: str
    doc_path: str
    doc_part: str
    sub_part: dict
    content: str = ""


class EvidenceChainInput(BaseModel):
    doc: DocBlock
    anchors: list[Anchor]


class EvidenceChainOutput(BaseModel):
    evidence_chain: list[Evidence]
    validation_results: list[dict]
    summary: dict


class RawEvidence(BaseModel):
    evidence_id: str
    source: str
    content: dict
    gene_rule: Optional[dict] = None
    support_chips: Optional[dict] = None
    dep_class: str = "original"


class ParsedEvidence(BaseModel):
    evidence_id: str
    content: str
    target_field: str
    confidence: float = 1.0
    category: EvidenceCategory = EvidenceCategory.FACT
    anchor_type: AnchorType = AnchorType.SECTION
    gene_rule: Optional[dict] = None
    extracted_data: Optional[dict] = None


class ParsedResult(BaseModel):
    doc_part: str
    evidences: list[dict]
    doc_original_parsed: Optional[dict] = None
    doc_original_md: list[str] = []
    total_evidences: int = 0
    evidences_by_target: dict = {}


class StyleSummary(BaseModel):
    core_topic: str
    key_points: list[str]
    context_dependencies: list[str]


class StyleExtraction(BaseModel):
    terminology: dict
    structure: dict
    format: dict
    punctuation: dict
    tone: dict


class ModificationResult(BaseModel):
    generated_text: str
    evidence_usage: list[dict]
    modification_summary: dict


class ValidationResult(BaseModel):
    consistency_passed: bool = True
    accuracy_rate: float = 1.0
    consistency_scores: dict = {}
    total_evidence: int = 0
    total_batches: int = 0
    issues: list[dict] = []

    metric_passed: Optional[str] = None
    suggestion: Optional[str] = None
    edit_distance_ratio: Optional[float] = None
    avg_confidence: Optional[float] = None
    passed: Optional[bool] = None


class RawEvidenceInput(BaseModel):
    evidence_id: Optional[str] = None
    envidence_id: Optional[str] = None
    evidence_information: dict
    gene_rule: dict
    support_chips: Optional[dict] = None

    def get_evidence_id(self) -> str:
        return self.evidence_id or self.envidence_id or ""


class RawAnchorInput(BaseModel):
    anchor_id: str
    note: str
    anchor_information: dict
    evidence: list[RawEvidenceInput]


class RawDocInput(BaseModel):
    doc_name: str
    doc_path: str
    doc_part: str
    sub_part: dict
    md_format: list[str]


class RawEvidenceChainInput(BaseModel):
    doc: RawDocInput
    anchor_tuple: list[RawAnchorInput]
    summary: Optional[dict] = None
