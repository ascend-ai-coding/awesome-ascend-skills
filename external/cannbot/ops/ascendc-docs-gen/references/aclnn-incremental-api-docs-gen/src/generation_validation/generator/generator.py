# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Document generator - main entry for document generation"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

from rich.console import Console

from src.llm.client import LLMClient
from src.logger import debug, info, log_result

from .. import config
from ..core.models import (
    ModificationResult,
    StyleExtraction,
    StyleSummary,
)
from ..dashboard import (
    dashboard_log,
    get_current_doc_part,
    get_dashboard,
    set_dashboard_context,
)
from ..style.loader import TemplateLoader
from .writer import TextWriter


class DocumentGenerator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        template_loader: Optional[TemplateLoader] = None,
    ):
        self.writer = TextWriter(llm_client, template_loader)

    @staticmethod
    def _clean_evidences(evidences: dict) -> dict:
        """Remove category field from evidences dict"""
        if not isinstance(evidences, dict):
            return evidences

        cleaned = {}
        for field_name, items in evidences.items():
            if isinstance(items, dict):
                cleaned[field_name] = {}
                for sub_field, sub_items in items.items():
                    if isinstance(sub_items, list):
                        cleaned[field_name][sub_field] = [
                            {k: v for k, v in item.items() if k != "category"}
                            for item in sub_items
                            if isinstance(item, dict)
                        ]
                    else:
                        cleaned[field_name][sub_field] = sub_items
            elif isinstance(items, list):
                # fmt: off
                cleaned[field_name] = [
                    {k: v for k, v in item.items() if k != "category"}
                    for item in items
                    if isinstance(item, dict)
                ]
                # fmt: on
            else:
                cleaned[field_name] = items
        return cleaned

    def generate(
        self,
        original_content: dict,
        evidences: dict = None,
        style_summary: StyleSummary = None,
        style_extraction: StyleExtraction = None,
        doc_part: str = "",
        sub_key: str = "",
        previous_issues: list = None,
        generated_doc: Union[str, dict] = "",
        passed_indices: list = None,
        change_reason: Optional[dict] = None,
    ) -> List[ModificationResult]:
        if doc_part == "sub_function_hash":
            return self._generate_sub_function_hash(
                original_content=original_content,
                evidences=evidences,
                style_summary=style_summary,
                style_extraction=style_extraction,
                previous_issues=previous_issues,
                generated_doc=generated_doc,
                passed_indices=passed_indices or [],
                change_reason=change_reason,
            )

        md_list = original_content.get("md", [])
        doc_original_md_str = md_list[0] if md_list else ""
        console = Console()

        if get_dashboard():
            dashboard_log("  > 生成文档模块")
        else:
            console.print("  > 生成文档模块")
        cleaned_evidences = self._clean_evidences(evidences) if evidences else None
        gen_doc_str = ""
        if isinstance(generated_doc, dict):
            gen_doc_str = next(iter(generated_doc.values()), "") if generated_doc else ""
        else:
            gen_doc_str = generated_doc
        result = self.writer.generate(
            original=doc_original_md_str,
            evidences=cleaned_evidences,
            style_summary=style_summary,
            style_extraction=style_extraction,
            doc_part=doc_part,
            sub_key=sub_key,
            previous_issues=previous_issues,
            generated_doc=gen_doc_str,
            change_reason=change_reason,
        )

        changes = result.modification_summary.get("changes", [])
        result.modification_summary = {
            "sub_key": sub_key,
            "changes_identified": len(changes),
            "evidence_used": len(changes),
            # 保留 writer 的 analysis（need_change/evidence_conflict/conflict_details），
            # 上层重试环依赖它判断证据冲突与无需修改
            "analysis": result.modification_summary.get("analysis"),
        }

        log_result("Result")
        debug(f"[modification_summary]:\n{json.dumps(result.modification_summary, ensure_ascii=False, indent=2)}")
        log_result(f"[changes]:\n{json.dumps(changes, ensure_ascii=False, indent=2)}")
        log_result(f"[generated_text]:\n```markdown\n{result.generated_text}\n```\n")

        return [result]

    def _generate_sub_function_hash(
        self,
        original_content: dict,
        evidences: dict,
        style_summary: StyleSummary,
        style_extraction: StyleExtraction,
        previous_issues: list = None,
        generated_doc: Union[str, dict] = "",
        passed_indices: list = None,
        change_reason: Optional[dict] = None,
    ) -> List[ModificationResult]:
        """Generate for sub_function_hash - handle param_matrix and errorcode separately"""
        doc_original_parsed = original_content.get("parsed", {})
        if not isinstance(doc_original_parsed, dict):
            doc_original_parsed = {}
        doc_original_md = original_content.get("md", [])

        results = []

        param_matrix_keys = list(doc_original_parsed.get("param_matrix", {}).keys())
        template_mapping = {
            "sub_function_1_param_matrix": "aclnnXxxGetWorkspaceSize_param_and_return",
            "sub_function_2_param_matrix": "aclnnXxx_param_and_return",
            "sub_function_3_param_matrix": "aclnnXxxGetWorkspaceSize_param_and_return",
            "sub_function_4_param_matrix": "aclnnXxx_param_and_return",
            "sub_function_1_errorcode": "aclnnXxxGetWorkspaceSize_param_and_return",
            "sub_function_2_errorcode": "aclnnXxx_param_and_return",
            "sub_function_3_errorcode": "aclnnXxxGetWorkspaceSize_param_and_return",
            "sub_function_4_errorcode": "aclnnXxx_param_and_return",
        }

        passed_set = set(passed_indices or [])

        def _generate_single_sub_function(idx, sub_key_name):
            safe_md = doc_original_md[idx] if idx < len(doc_original_md) else ""

            if idx in passed_set:
                info(f"  Skipping {sub_key_name} (already passed validation)")
                return idx, ModificationResult(
                    generated_text=safe_md,
                    evidence_usage=[],
                    modification_summary={"sub_key": sub_key_name, "skipped": True},
                )

            func_num = sub_key_name.split("_")[2]
            errorcode_key = f"sub_function_{func_num}_errorcode"

            sub_key_og_content = safe_md

            sub_key_evidences = {}
            if sub_key_name in evidences:
                sub_key_evidences["param_matrix"] = evidences[sub_key_name]
            if errorcode_key in evidences:
                sub_key_evidences["errorcode"] = evidences[errorcode_key]
            console = Console()

            if get_dashboard():
                dashboard_log(f"  > 为 {sub_key_name} 生成文档")
            else:
                console.print(f"  > 为 {sub_key_name} 生成文档")
            cleaned_evidences = self._clean_evidences(sub_key_evidences)
            filtered_issues = []
            if previous_issues:
                filtered_issues = [iss for iss in previous_issues if iss.get("sub_key", "") == sub_key_name]
            sub_gen_doc = ""
            if isinstance(generated_doc, dict):
                sub_gen_doc = generated_doc.get(sub_key_name, "")
            else:
                sub_gen_doc = generated_doc
            result = self.writer.generate(
                original=sub_key_og_content,
                evidences=cleaned_evidences,
                style_summary=style_summary,
                style_extraction=style_extraction,
                doc_part="sub_function_hash",
                sub_key=template_mapping.get(sub_key_name),
                previous_issues=filtered_issues if filtered_issues else None,
                generated_doc=sub_gen_doc,
                change_reason=change_reason,
            )

            changes = result.modification_summary.get("changes", [])
            result.modification_summary = {
                "changes_identified": len(changes),
                "evidence_used": len(changes),
                "sub_key": sub_key_name,
                # 保留 writer 的 analysis（need_change/evidence_conflict/conflict_details），
                # 上层重试环依赖它判断证据冲突与无需修改
                "analysis": result.modification_summary.get("analysis"),
            }

            log_result(f"Result for {result.modification_summary.get('sub_key', 'N/A')}")
            debug(f"[Modification Summary]:\n{json.dumps(result.modification_summary, ensure_ascii=False, indent=2)}")
            log_result(f"[Editing Points]:\n{json.dumps(changes, ensure_ascii=False, indent=2)}")
            log_result(f"\n[Generated Text]:\n```markdown\n{result.generated_text}\n```")

            return idx, result

        current_dashboard = get_dashboard()
        current_doc_part = get_current_doc_part()

        def _run_with_context(idx, sub_key_name):
            if current_dashboard:
                tokens = set_dashboard_context(current_dashboard, current_doc_part)
                try:
                    return _generate_single_sub_function(idx, sub_key_name)
                finally:
                    for t in reversed(tokens):
                        t.var.reset(t)
            else:
                return _generate_single_sub_function(idx, sub_key_name)

        with ThreadPoolExecutor(max_workers=config.GENERATION_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_run_with_context, idx, sub_key_name): idx
                for idx, sub_key_name in enumerate(param_matrix_keys)
            }
            indexed_results = {}
            for future in as_completed(futures):
                idx, result = future.result()
                indexed_results[idx] = result

        results = [indexed_results[i] for i in range(len(param_matrix_keys))]
        return results
