# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Pipeline stages - evidence parsing, style processing, generation, and validation."""

import json
from pathlib import Path

from src.llm import LLMClient
from src.logger import info

from .. import config
from ..core.models import (
    RawAnchorInput,
    RawDocInput,
    RawEvidenceChainInput,
    StyleExtraction,
    StyleSummary,
)
from ..evidence.extractor import EvidenceExtractor
from ..evidence.parser import EvidenceParser
from ..evidence.ranker import EvidenceRanker
from ..generator.generator import DocumentGenerator
from ..generator.writer import TextWriter
from ..style.extractor import StyleExtractor
from ..style.loader import TemplateLoader
from ..style.summarizer import StyleSummarizer
from ..validator.consistency import ConsistencyChecker
from ..validator.metric import ModificationMetric
from ..validator.validator import DocumentValidator

SUB_FUNCTION_MAX = 4


class EvidenceStage:
    """Evidence parsing, ranking, and extraction stage."""

    def __init__(self, llm_client: LLMClient = None):
        self.parser = EvidenceParser()
        self.ranker = EvidenceRanker()
        self.extractor = EvidenceExtractor(llm_client)

    @staticmethod
    def load_evidence_file(json_path: Path) -> RawEvidenceChainInput:
        """Load evidence from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RawEvidenceChainInput(
            doc=RawDocInput(**data["doc"]),
            anchor_tuple=[RawAnchorInput(**anchor) for anchor in data["anchor_tuple"]],
        )

    def parse_and_rank(self, raw_input: RawEvidenceChainInput, deduplicate: bool = False):
        """Parse raw evidence and rank by confidence."""
        parsed = self.parser.parse_raw_input(raw_input, deduplicate=deduplicate)
        ranked = self.ranker.rank_parsed_result(parsed)
        return parsed, ranked

    def extract_evidence(
        self,
        ranked,
        doc_part: str,
        doc_original: dict,
        progressive: bool = True,
        evidence_confidence_threshold: float = 0.8,
        modified_files: list = None,
    ) -> tuple[dict, list]:
        """Extract structured evidence by doc_part using progressive extraction.
        Returns (extracted_dict, source_conflicts).
        """
        from pathlib import Path

        modified_file_basenames = {Path(f).name for f in modified_files} if modified_files else None

        extracted, source_conflicts = self.extractor.extract_by_doc_part(
            doc_part=doc_part,
            evidences_by_target=ranked.evidences_by_target,
            doc_original=doc_original,
            progressive=progressive,
            threshold=evidence_confidence_threshold,
            modified_file_basenames=modified_file_basenames,
        )
        return extracted, source_conflicts


class StyleStage:
    """Style summarization and extraction stage."""

    def __init__(self, llm_client: LLMClient = None, style_dir: Path = None):
        self.summarizer = StyleSummarizer(llm_client, style_file=style_dir / "summary.json" if style_dir else None)
        self.extractor = StyleExtractor(llm_client, style_file=config.RESOURCES_DIR / "soul.json")
        self.template_loader = TemplateLoader.from_file(config.RESOURCES_DIR / "templates.yaml")
        self.style_dir = style_dir

    @property
    def template_loader(self):
        return self._template_loader

    @template_loader.setter
    def template_loader(self, val):
        self._template_loader = val

    def get_style_summary(self) -> StyleSummary:
        """Get style summary from pre-processed file, LLM, or pre-loaded dict."""
        if self.style_dir:
            summary_file = Path(self.style_dir)
            if summary_file.exists():
                return self.summarizer.get_from_file(summary_file)
        return StyleSummary(core_topic="N/A", key_points=[], context_dependencies=[])

    def get_style_extraction(self) -> StyleExtraction:
        """Get style extraction from pre-processed file."""
        soul_path = config.RESOURCES_DIR / "soul.json"
        if soul_path.exists():
            return self.extractor.get_from_file(soul_path)
        return StyleExtraction(terminology={}, structure={}, format={}, punctuation={}, tone={})


class GenerationStage:
    """Document generation stage."""

    def __init__(self, llm_client: LLMClient = None, template_loader: TemplateLoader = None):
        self.generator = DocumentGenerator(llm_client, template_loader)
        self.writer = TextWriter(llm_client, template_loader)

    def generate(
        self,
        original_content: dict,
        evidences: dict,
        style_summary: StyleSummary,
        style_extraction: StyleExtraction,
        doc_part: str,
        sub_key: str = "",
        previous_issues: list = None,
        generated_doc: str = "",
        passed_indices: list = None,
        change_reason: dict = None,
    ):
        """Generate modified document sections."""
        results = self.generator.generate(
            original_content=original_content,
            evidences=evidences,
            style_summary=style_summary,
            style_extraction=style_extraction,
            doc_part=doc_part,
            sub_key=sub_key,
            previous_issues=previous_issues,
            generated_doc=generated_doc,
            passed_indices=passed_indices,
            change_reason=change_reason,
        )
        return results


class ValidationStage:
    """Validation stage - consistency check + metric check."""

    def __init__(self, llm_client: LLMClient = None):
        self.validator = DocumentValidator(llm_client)
        self.consistency = ConsistencyChecker(llm_client)
        self.metric = ModificationMetric()

    @staticmethod
    def format_result(result) -> dict:
        """Format validation result to dict for collection."""
        return {
            "consistency_passed": result.consistency_passed,
            "accuracy_rate": result.accuracy_rate,
            "metric_passed": result.metric_passed,
            "edit_distance_ratio": result.edit_distance_ratio,
            "avg_confidence": result.avg_confidence,
            "issues": result.issues,
            "suggestion": result.suggestion,
        }

    @staticmethod
    def print_result(result):
        """Print validation result."""
        info(f"  consistency_passed: {result.consistency_passed}")
        info(f"  accuracy_rate: {result.accuracy_rate}")
        info(f"  metric_passed: {result.metric_passed}")
        info(f"  edit_distance_ratio: {result.edit_distance_ratio}")
        info(f"  avg_confidence: {result.avg_confidence}")
        if result.issues:
            info(f"  issues ({len(result.issues)}):")
            for issue in result.issues:
                info(f"    - [{issue.get('id')}] {issue.get('reason', '')}")
        if result.suggestion:
            info(f"  suggestion: {result.suggestion}")

    @staticmethod
    def get_sub_function_indices(parsed) -> list:
        """Get valid sub-function indices from parsed evidence targets."""
        return [
            i
            for i in range(1, SUB_FUNCTION_MAX + 1)
            if any(key.startswith(f"sub_function_{i}") for key in parsed.evidences_by_target.keys())
        ]

    def validate(
        self,
        doc_part: str,
        doc_original: dict,
        evidences: list,
        doc_generated: str = "",
        evidences_extracted: dict = None,
        check_metric: bool = True,
        idx: int = None,
    ):
        """Run full validation pipeline."""
        if doc_part == "sub_function_hash" and idx is not None:
            prefix = f"sub_function_{idx}"
            evidences_function_level = [
                ev
                for ev in evidences
                if any(
                    isinstance(t, list) and len(t) > 0 and isinstance(t[0], str) and t[0].startswith(prefix)
                    for t in ev.get("target", [])
                )
            ]
            extracted_function_level = {k: v for k, v in (evidences_extracted or {}).items() if k.startswith(prefix)}
            doc_original = doc_original
            evidences = evidences_function_level
            evidences_extracted = extracted_function_level
        else:
            doc_original = str(doc_original) if isinstance(doc_original, dict) else doc_original

        return self.validator.validate(
            doc_original=doc_original,
            evidences=evidences,
            check_metric=check_metric,
            doc_part=doc_part,
            doc_generated=doc_generated,
            evidences_extracted=evidences_extracted,
        )
