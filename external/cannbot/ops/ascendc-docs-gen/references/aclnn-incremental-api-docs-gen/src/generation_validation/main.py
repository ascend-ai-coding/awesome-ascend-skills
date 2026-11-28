# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Main entry point - orchestrates the full document editing pipeline."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.llm import LLMClient
from src.logger import debug, highlight, info, log_result, section, setup_logger

from . import config
from .core import ModificationResult
from .pipeline import (
    GenerationStage,
    StyleStage,
    ValidationStage,
)
from .pipeline.deterministic_checks import (
    build_sig_corpus,
    collect_issues,
    resolve_orig_doc_full,
)
from .pipeline.helpers import get_output_dir, save_json, setup_pipeline
from .ui.cli import _is_validation_passed

setup_logger(level=logging.INFO)


def run_full_pipeline(evidence_json: str, doc_part: str, style_dir: str = None, change_reason: dict = None) -> dict:
    """Run the complete pipeline: evidence -> style -> generation -> validation."""

    # Step 0
    section("Step 0: Configuration for File Path & LLM Client")

    style_dir_path = (
        (Path(style_dir) if Path(style_dir).is_absolute() else config.PROJECT_ROOT / style_dir)
        if style_dir
        else config.RESOURCES_DIR
    )
    llm = LLMClient()

    # Step 1: Load & parse evidence
    evidence_stage, raw_input, parsed, ranked, _, evidence_path = setup_pipeline(evidence_json, llm)
    info(f"[Original Markdown Doc]\n```markdown\n{parsed.doc_original_md[0] if parsed.doc_original_md else ''}\n```")

    # Step 2
    section("步骤 2: 编排代码证据")
    output_dir = get_output_dir()
    extracted_path = output_dir / f"extracted_{Path(evidence_json).stem}.json"

    read_cache_for_step_2 = False
    if read_cache_for_step_2:
        info(f"  Loading cached extracted data from {extracted_path}")
        with open(extracted_path, "r", encoding="utf-8") as f:
            extracted = json.load(f)
    else:
        extracted, _ = evidence_stage.extract_evidence(
            ranked=ranked,
            doc_part=doc_part,
            doc_original=parsed.doc_original_parsed,
            progressive=True,
            evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
        )
        save_json(extracted, extracted_path)
        highlight(f"  Saved extracted data to {extracted_path}")

    highlight(f"  extracted fields: {', '.join(extracted.keys())}")

    # Step 3
    section("步骤 3: 加载领域信息")
    style_stage = StyleStage(llm, style_dir_path)
    style_summary = style_stage.get_style_summary()  # 即时生成 doc_path=doc_full_path；否则读取预处理 summary.json 文件
    style_extraction = style_stage.get_style_extraction()  # 读取预处理 soul.json 文件

    highlight("[Doc Summary]\n" + json.dumps(style_summary.__dict__, ensure_ascii=False, indent=4))
    debug("[Style Extraction]\n" + json.dumps(style_extraction.__dict__, ensure_ascii=False, indent=4))

    # Step 4: Generate
    section("步骤 4: 生成文档模块")
    validation_results_list = []
    final_results = []

    if doc_part == "support":
        support_matrix = extracted.get("support_matrix", {})
        supported = support_matrix.get("supported")
        md_content = support_matrix.get("md_content")
        highlight(f"supported: {supported}\nmd_content:\n```markdown\n{md_content}\n```\n")

        section("步骤 5: 验证文档模块")
        validation_stage = ValidationStage(llm)
        parsed_dedup = evidence_stage.parser.parse_raw_input(raw_input, deduplicate=True)
        doc_original_for_validation = parsed.doc_original_md[0] if parsed.doc_original_md else ""
        validation_result = validation_stage.validate(
            doc_part=doc_part,
            doc_original=doc_original_for_validation,
            evidences=parsed_dedup.evidences,
            doc_generated=md_content,
            evidences_extracted=extracted,
            check_metric=True,
        )
        ValidationStage.print_result(validation_result)
        result_dict = ValidationStage.format_result(validation_result)
        result_dict["passed"] = _is_validation_passed(result_dict)
        validation_results_list.append(result_dict)
        final_results.append(
            {
                "sub_key": "support_matrix",
                "generated_text": md_content,
                "modification_summary": {},
            }
        )
    else:
        gen_result_path = output_dir / f"gen_result_{Path(evidence_json).stem}.json"

        generation_stage = GenerationStage(llm, style_stage.template_loader)
        validation_stage = ValidationStage(llm)

        previous_issues = []
        previous_generated_doc = ""
        all_passed = False
        passed_indices = []

        for attempt in range(config.MAX_REGENERATION_RETRIES):
            if attempt > 0:
                info(f"--- Regeneration attempt {attempt + 1}/{config.MAX_REGENERATION_RETRIES} ---")
                if passed_indices:
                    info(f"  Skipping {len(passed_indices)} already-passed section(s): {passed_indices}")

            read_cache_for_step_4 = False
            if read_cache_for_step_4:
                info(f"  Loading cached generated results from {gen_result_path}")
                with open(gen_result_path, "r", encoding="utf-8") as f:
                    results = [ModificationResult(**r) for r in json.load(f)]
            else:
                results = generation_stage.generate(
                    original_content={
                        "parsed": parsed.doc_original_parsed,
                        "md": parsed.doc_original_md,
                    },
                    evidences=extracted,
                    style_summary=style_summary,
                    style_extraction=style_extraction,
                    doc_part=doc_part,
                    previous_issues=previous_issues,
                    generated_doc=previous_generated_doc,
                    passed_indices=passed_indices,
                    change_reason=change_reason,
                )
                save_json(
                    [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
                    gen_result_path,
                )
                highlight(f"  Saved results data to {gen_result_path}")

            section(f"步骤 5: 验证文档模块 (attempt {attempt + 1})")

            validation_results_list.clear()
            final_results.clear()
            all_passed = True
            current_issues = []
            current_generated_doc = ""
            current_passed_indices = []

            # 签名语料：本 doc_part 全部原章节 + 证据链原文 + 编排后证据（供参数放行判断；
            # changed_only 证据被过滤掉 aclnn 签名时由辅助函数回补兄弟全量证据）
            sig_corpus = build_sig_corpus(parsed.doc_original_md, raw_input, extracted, evidence_json)
            # 原文档全文（跨 doc_part），供约束节参数越界检查取到其他小节的参数名；
            # anchor 的 doc_path 可能是目录（须拼 doc_name），为相对目录时经证据路径回溯
            orig_doc_full = resolve_orig_doc_full(raw_input.doc.doc_path, raw_input.doc.doc_name, evidence_json)

            for i, result in enumerate(results):
                info(f"Validate [{i + 1}/{len(results)}]")
                generated_text = result.generated_text
                validation_result = validation_stage.validate(
                    doc_part=doc_part,
                    doc_original=parsed.doc_original_md[i] if i < len(parsed.doc_original_md) else "",
                    evidences=parsed.evidences,
                    doc_generated=generated_text,
                    evidences_extracted=extracted,
                    check_metric=True,
                    idx=i + 1,
                )
                result_dict = ValidationStage.format_result(validation_result)
                ValidationStage.print_result(validation_result)

                orig_section = parsed.doc_original_md[i] if i < len(parsed.doc_original_md) else ""

                # 确定性巡检：链接改写 / 参数表越界 → 计入 issues 触发重试再生成
                new_issues = collect_issues(orig_section, generated_text, doc_part, sig_corpus, orig_doc_full)
                if new_issues:
                    result_dict["consistency_passed"] = False
                    result_dict["issues"] = result_dict.get("issues", []) + new_issues
                    for issue in new_issues:
                        info(f"  [{issue['id']}] {issue['reason']}")

                validation_failed = not _is_validation_passed(result_dict)

                if validation_failed:
                    all_passed = False
                    current_issues.extend(result_dict.get("issues", []))
                    current_generated_doc += generated_text + "\n"
                else:
                    current_passed_indices.append(i)

                validation_results_list.append(result_dict)
                final_results.append(
                    {
                        "sub_key": result.modification_summary.get("sub_key"),
                        "generated_text": result.generated_text,
                        "modification_summary": result.modification_summary,
                    }
                )

            if all_passed:
                info("All validations passed.")
                break
            elif attempt < config.MAX_REGENERATION_RETRIES - 1:
                previous_issues = current_issues
                previous_generated_doc = current_generated_doc
                passed_indices = current_passed_indices
                info(f"Validation failed. Collecting {len(previous_issues)} issues for regeneration...")
            else:
                highlight("Max regeneration attempts reached. Proceeding with last results.")

    return {
        "parsed": {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
            "evidences_by_target": ranked.evidences_by_target,
            "doc_original": parsed.doc_original_md,
        },
        "extracted": extracted,
        "style_summary": {
            "core_topic": style_summary.core_topic,
            "key_points": style_summary.key_points,
            "context_dependencies": style_summary.context_dependencies,
        },
        "style_extraction": {
            "terminology": style_extraction.terminology,
            "structure": style_extraction.structure,
            "format": style_extraction.format,
        },
        "generated": final_results,
        "validation": validation_results_list,
    }


def validate_doc_with_evidences(evidence_json: str, doc_part: str) -> dict:
    """Run the partial pipeline: evidence -> validation."""

    # Step 0 & 1: Setup pipeline and load evidence
    section("Step 0: Configuration for File Path & LLM Client")

    evidence_stage, raw_input, parsed, ranked, llm, evidence_path = setup_pipeline(evidence_json)

    # Step 2
    section("步骤 2: 验证文档模块")

    validation_stage = ValidationStage(llm)
    validation_results_list = []

    extracted = None
    if doc_part == "support":
        extracted, _ = evidence_stage.extract_evidence(
            ranked=ranked,
            doc_part=doc_part,
            doc_original=parsed.doc_original_md[0] if parsed.doc_original_md else "",
            progressive=True,
            evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
        )
        indices = [None]
    elif doc_part == "sub_function_hash":
        # --- sub-function-hash 情况下多个子函数分别验证 ---
        indices = validation_stage.get_sub_function_indices(parsed)
    else:
        indices = [None]

    for idx in indices:
        if isinstance(idx, int):
            doc_orig = parsed.doc_original_md[idx - 1] if idx - 1 < len(parsed.doc_original_md) else ""
        else:
            doc_orig = parsed.doc_original_md[0] if parsed.doc_original_md else ""
        validation_result = validation_stage.validate(
            doc_part=doc_part,
            doc_original=doc_orig,
            evidences=parsed.evidences,
            evidences_extracted=extracted,
            check_metric=False,
            idx=idx,
        )
        result_dict = ValidationStage.format_result(validation_result)
        ValidationStage.print_result(validation_result)
        validation_results_list.append(result_dict)

    return {
        "parsed": {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
            "evidences_by_target": ranked.evidences_by_target,
            "doc_original": parsed.doc_original_md,
        },
        "extracted": {},
        "style_summary": {},
        "style_extraction": {},
        "generated": [],
        "validation": validation_results_list,
    }


def run_pipeline(subparts, evidence_json, mode="full"):
    start = datetime.now(tz=timezone.utc)
    if mode == "full":
        result = run_full_pipeline(evidence_json=evidence_json, doc_part=subparts, style_dir="resources")
    elif mode == "validate_only":
        result = validate_doc_with_evidences(evidence_json=evidence_json, doc_part=subparts)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'full' or 'validate_only'.")
    end = datetime.now(tz=timezone.utc)

    if mode == "full":
        debug("FULL PIPELINE RESULT SUMMARY")
    else:
        debug("VALIDATION RESULT SUMMARY")
    debug("\n" + json.dumps(result, ensure_ascii=False, indent=2))

    if mode == "full":
        generated_list = result.get("generated", [])
        for item in generated_list:
            text = item.get("generated_text", "")
            if text:
                info(f"生成的文档内容：\n{text}")

    total_seconds = (end - start).total_seconds()
    total_minutes = total_seconds / 60
    highlight(f"runtime: {total_seconds:.2f} s ({total_minutes:.2f} min)")


def run_generate_and_validate(subparts, evidence_json):
    run_pipeline(subparts, evidence_json, mode="full")


def run_validate_only(subparts, evidence_json):
    run_pipeline(subparts, evidence_json, mode="validate_only")


if __name__ == "__main__":
    subparts = config.DOC_PARTS[2]
    evidence_json = f"data/evidences/example/aclnnAbs.md-{subparts}.json"
    run_generate_and_validate(subparts, evidence_json)
