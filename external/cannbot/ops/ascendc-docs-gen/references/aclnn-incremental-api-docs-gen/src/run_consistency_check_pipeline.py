# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Consistency Check Pipeline - 一致性检测流水线 (validate_only 模式)。

参照 run_all_cli.py 的 validate_only 模式, 融合为单一非交互脚本,
供 VSCode 扩展以子进程方式调用。workspace_dir 为绝对路径, 其余路径均为相对路径。

输出标记:
  - ==result== / ==/result== (全量一致性检测结果 JSON)
  - DONE 文件 (成功) / FAILED 文件 (异常); 运行开始时清理旧标记, 结束后出现的标记必属本次运行
"""

import argparse
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

# 设置路径: 添加 cann-document-generation/（仓库根目录）到搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import apply_settings, load_settings
from src.snapshot_construction.anchor_construction.common.doc_selector import (
    select_doc_file,
)
from src.snapshot_construction.pub_ref import PubRefConfig


def _configure_ssl_cert() -> None:
    """SSL 证书配置: 如果证书不存在则禁用 SSL 验证（原 import 期全局改写，移入函数保留语义）。"""
    try:
        from src.llm import config as llm_config
    except ImportError:
        return
    if llm_config.SSL_CERT is not None and not Path(str(llm_config.SSL_CERT)).exists():
        print(f"[run_consistency_check_pipeline] SSL cert not found: {llm_config.SSL_CERT}, disabling")
        llm_config.SSL_CERT = None


def _vget(v, key, default=None):
    """dict/对象双态取值: validation 元素可能是 dict (run_full_pipeline 非 example 分支)
    或 Pydantic 对象 (run_validate_only / example 分支), getattr 对 dict 不读键, 需分流。"""
    if isinstance(v, dict):
        return v.get(key, default)
    return getattr(v, key, default)


def _serialize_result(r) -> dict:
    """将 PipelineResult 序列化为可JSON化的dict, 处理Pydantic模型"""
    d = asdict(r)
    d["validation"] = [v.model_dump() if hasattr(v, "model_dump") else v for v in r.validation]
    return d


def _unlink_quietly(path: Path) -> None:
    """删除哑文件, 不存在时静默跳过。"""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _clear_sentinel_files(workspace_dir: str, data_path: str) -> Path:
    """运行开始时清理旧 DONE/FAILED 哑文件, 返回解析后的 data_path。

    哑文件语义: 运行期间无标记 = 进行中; 运行结束后出现的标记必属本次运行。
    """
    data_path_resolved = (Path(workspace_dir).resolve() / data_path).resolve()
    data_path_resolved.mkdir(parents=True, exist_ok=True)
    _unlink_quietly(data_path_resolved / "DONE")
    _unlink_quietly(data_path_resolved / "FAILED")
    return data_path_resolved


def _output_result(results: dict, data_path_resolved: Path):
    """输出结果标记并写入 DONE 文件"""
    print("\n=== Consistency Check Pipeline Complete ===")
    print("==result==")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("==/result==")

    _unlink_quietly(data_path_resolved / "FAILED")  # 终态互斥: 删除相反状态旧标记
    done_file = data_path_resolved / "DONE"
    done_file.write_text(
        json.dumps({"completedAt": datetime.now().isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"DONE file written: {done_file}")


def _build_inconsistency(
    doc_part: str,
    evidence_dir: Path,
    selected_file: Path,
    snapshot_op_dir: Path,
    issue: dict,
) -> dict:
    """从 validation issue 构造 InconsistencyDetail"""
    doc_path = ""
    code_path = ""

    # 从 evidence anchor 中提取文件路径 (如果 issue 中有)
    location = issue.get("location", "")
    if location:
        code_path = location

    # 文档路径基于 selected_file
    if selected_file:
        doc_path = (
            str(selected_file.relative_to(snapshot_op_dir))
            if selected_file.is_relative_to(snapshot_op_dir)
            else selected_file.name
        )

    # severity 推断: accuracy_rate低于0.7为high, 0.7-0.9为medium, 其他为low
    severity = issue.get("severity", "medium")

    return {
        "docPart": doc_part,
        "docPath": doc_path,
        "codePath": code_path,
        "description": issue.get("reason", issue.get("description", str(issue))),
        "severity": severity,
    }


def run_pipeline(
    snapshot_id: str,
    workspace_dir: str,
    original_op_dir: str,
    data_path: str,
    operator_version: str = "base",
    verbose: bool = False,
):
    """执行一致性检测流水线 (validate_only 模式)"""
    from src.change_detection.doc_part_change_detection.docpart_file_extractor import (
        DocpartFileExtractor,
    )
    from src.generation_validation.ui.cli import (
        EvidenceFile,
        LogCapture,
        _is_validation_passed,
        resolve_evidence_file,
        run_full_pipeline,
        run_validate_only,
    )
    from src.logger import setup_logger

    setup_logger(level=logging.INFO, console=False)
    _configure_ssl_cert()

    # ── 路径解析 ─────────────────────────────────────────────────
    workspace = Path(workspace_dir).resolve()
    data_path_resolved = (workspace / data_path).resolve()
    data_path_resolved.mkdir(parents=True, exist_ok=True)
    original_op_dir_resolved = (workspace / original_op_dir).resolve()

    selected_op = original_op_dir_resolved.name
    print(f"  Workspace:  {workspace}")
    print(f"  Operator:   {selected_op}")
    print(f"  Data path:  {data_path_resolved}")

    results = {
        "snapshotId": snapshot_id,
        "version": operator_version,
        "operatorName": selected_op,
        "docParts": [],
        "results": {},
        "detailedResults": {},
        "inconsistencies": [],
        "backfillMapPostValFix": {},
        "summary": {
            "totalParts": 0,
            "passedParts": 0,
            "failedParts": 0,
            "overallPassed": False,
        },
        "generatedAt": None,
    }

    md_stems_dir = data_path_resolved / selected_op

    if not md_stems_dir.exists():
        raise Exception(f"算子快照目录不存在: {md_stems_dir}")

    # ── 选择算子 md_stem ─────────────────────────────────────────
    available_md_stems = sorted(d.name for d in md_stems_dir.iterdir() if d.is_dir())
    if not available_md_stems:
        raise Exception(f"未找到可用的算子 md_stem: {md_stems_dir}")

    if len(available_md_stems) == 1:
        selected_md_stem = available_md_stems[0]
    elif operator_version in available_md_stems:
        selected_md_stem = operator_version
    else:
        print(f"  md_stem '{operator_version}' not in {available_md_stems}, fallback to first")
        selected_md_stem = available_md_stems[0]
    print(f"  md_stem:   {selected_md_stem}")

    # ── 配置各阶段路径 ───────────────────────────────────────────
    version_dir = md_stems_dir / selected_md_stem
    snapshot_op_dir = version_dir
    evidence_dir = version_dir / "anchor"
    source_docs = snapshot_op_dir / "docs"
    doc_json_dir = version_dir.parent / "doc_json"

    if not evidence_dir.exists():
        raise Exception(f"锚点证据目录不存在: {evidence_dir}")

    # ── 选择文档文件 ─────────────────────────────────────────────
    selected_file = None

    if source_docs.exists():
        doc_files = sorted(source_docs.glob("*.md"))
        if doc_files:
            selected_file = select_doc_file(doc_files, selected_md_stem=selected_md_stem)
            if selected_file is None:
                raise Exception(f"未找到以 aclnn 开头的文档文件: {source_docs}")
            Path(selected_file).read_text(encoding="utf-8")
        else:
            print(f"  Warning: No .md files found in {source_docs}")
    else:
        print(f"  Warning: Docs directory not found: {source_docs}")

    # ===== Step 1: DocPart提取 (validate_only模式入口) =====
    print("\n=== Step 1: Extract DocPart Mappings ===")
    print(f"  Evidence dir: {evidence_dir}")

    extractor = DocpartFileExtractor(evidence_dir)
    mappings = extractor.extract_fine_grained_mappings()
    doc_subpart_todo_list = [subpart for subpart, files in mappings.items() if files]
    examples_dir = snapshot_op_dir / "examples"
    if examples_dir.exists() and "example" not in doc_subpart_todo_list:
        doc_subpart_todo_list.append("example")

    if not doc_subpart_todo_list:
        print("\nNo doc parts found with evidence, pipeline complete with empty results")
        results["generatedAt"] = datetime.now().isoformat()
        _output_result(results, data_path_resolved)
        return results

    print(f"  Todo list: {doc_subpart_todo_list}")
    results["docParts"] = doc_subpart_todo_list

    # ===== Step 2: 一致性验证 =====
    print("\n=== Step 2: Consistency Validation ===")

    Console()
    passed_parts = 0
    failed_parts = 0

    def _process_doc_part(doc_part: str):
        print(f"\nProcessing doc_part: {doc_part}")

        evidence_file = resolve_evidence_file(evidence_dir, doc_part)

        if evidence_file is None and doc_part != "example":
            print(f"  Warning: No evidence file for {doc_part}, skipping")
            return (
                doc_part,
                {
                    "consistencyPassed": False,
                    "accuracyRate": 0.0,
                    "passed": False,
                    "issues": [{"description": f"No evidence file found for {doc_part}"}],
                    "suggestion": None,
                    "totalEvidence": 0,
                    "avgConfidence": None,
                },
                None,
                [],
            )
        # example doc_part: 需要额外的 original_op_dir 和 doc_json_dir
        if doc_part == "example":
            example_doc_json_dir = ""
            if selected_file:
                example_doc_json_dir = str(doc_json_dir / f"{selected_md_stem}.json")

            result = run_validate_only(
                evidence_file,
                doc_part,
                LogCapture(),
                original_op_dir=str(snapshot_op_dir),
                doc_json_dir=example_doc_json_dir,
            )
        else:
            result = run_validate_only(
                evidence_file,
                doc_part,
                LogCapture(),
                original_op_dir=str(snapshot_op_dir),
                doc_json_dir=str(doc_json_dir / f"{selected_md_stem}.json") if selected_file else "",
            )

        # 序列化全量结果
        serialized = _serialize_result(result)
        # 提取验证汇总信息
        if result.validation:
            # 复用生成模块统一校验谓词 (consistency_passed 且 metric_passed∈{None,N/A,PASS})。
            # 旧写法 passed OR consistency_passed 会把 metric FAIL 导致的最终失败
            # (passed=False, consistency_passed=True) 误判为通过
            all_passed = all(_is_validation_passed(v) for v in result.validation)
            avg_accuracy = 0.0
            avg_confidence = None
            total_evidence = result.parsed.get("total_evidences", 0)
            all_issues = []
            inconsistencies = []

            for v in result.validation:
                v_dict = v.model_dump() if hasattr(v, "model_dump") else v
                if isinstance(v_dict, dict):
                    avg_accuracy += v_dict.get("accuracy_rate", 0.0)
                    conf = v_dict.get("avg_confidence")
                    if conf is not None:
                        avg_confidence = conf
                    issues = v_dict.get("issues", [])
                    for issue in issues:
                        if isinstance(issue, dict):
                            all_issues.append(issue)
                            # 构造inconsistency
                            inconsistencies.append(
                                _build_inconsistency(
                                    doc_part,
                                    evidence_dir,
                                    selected_file,
                                    snapshot_op_dir,
                                    issue,
                                )
                            )

            avg_accuracy = avg_accuracy / len(result.validation) if result.validation else 0.0

            # 提取suggestion (取第一个非空的)
            suggestion = None
            for v in result.validation:
                v_dict = v.model_dump() if hasattr(v, "model_dump") else v
                if isinstance(v_dict, dict):
                    s = v_dict.get("suggestion")
                    if s and str(s) != "None":
                        suggestion = s
                        break

            part_result = {
                "consistencyPassed": all_passed,
                "accuracyRate": avg_accuracy,
                "passed": all_passed,
                "issues": all_issues,
                "suggestion": suggestion,
                "totalEvidence": total_evidence,
                "avgConfidence": avg_confidence,
            }

            if all_passed:
                print(f"  PASS: {doc_part}")
            else:
                print(f"  FAIL: {doc_part}")

            return doc_part, part_result, serialized, inconsistencies
        else:
            print(f"  FAIL: {doc_part} (no validation results)")
            return (
                doc_part,
                {
                    "consistencyPassed": False,
                    "accuracyRate": 0.0,
                    "passed": False,
                    "issues": [{"description": "No validation results"}],
                    "suggestion": None,
                    "totalEvidence": 0,
                    "avgConfidence": None,
                },
                serialized,
                [],
            )

    max_workers = PubRefConfig.DOC_PART_MAX_WORKERS
    print(f"  Processing {len(doc_subpart_todo_list)} doc_parts in parallel (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_doc_part, dp): dp for dp in doc_subpart_todo_list}
        for future in as_completed(futures):
            dp = futures[future]
            try:
                doc_part, part_result, serialized, inconsistencies = future.result()
                results["results"][doc_part] = part_result
                if serialized is not None:
                    results["detailedResults"][doc_part] = serialized
                results["inconsistencies"].extend(inconsistencies)
                if part_result.get("passed", False):
                    passed_parts += 1
                else:
                    failed_parts += 1
            except Exception as e:
                print(f"  Error processing doc_part {dp}: {e}")
                failed_parts += 1
                results["results"][dp] = {
                    "consistencyPassed": False,
                    "accuracyRate": 0.0,
                    "passed": False,
                    "issues": [{"description": f"Processing error: {e}"}],
                    "suggestion": None,
                    "totalEvidence": 0,
                    "avgConfidence": None,
                }

    # ===== Step 3: Post-Validation Fix (对失败项重新生成) =====
    failed_doc_parts = [dp for dp, r in results["results"].items() if not r.get("passed", False)]
    if failed_doc_parts:
        print(f"\n=== Step 3: Post-Validation Fix ({len(failed_doc_parts)} failed) ===")
        print(f"  Failed doc_parts: {failed_doc_parts}")

        backfill_map_post_val_fix: dict[str, str] = {}

        def _fix_doc_part(doc_part: str):
            print(f"\n  Fixing doc_part: {doc_part}")
            evidence_file = resolve_evidence_file(evidence_dir, doc_part)
            if evidence_file is None and doc_part != "example":
                print(f"    Warning: No evidence file for {doc_part}, skipping fix")
                return doc_part, None, None, None

            log_capture = LogCapture()
            result = run_full_pipeline(
                evidence_file,
                doc_part,
                log_capture,
                modified_op_dir=str(snapshot_op_dir),
                operator_name=selected_op,
            )
            generated_text = "".join(item["generated_text"] for item in result.generated)
            serialized = _serialize_result(result)
            return doc_part, generated_text, serialized, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fix_doc_part, dp): dp for dp in failed_doc_parts}
            for future in as_completed(futures):
                dp = futures[future]
                try:
                    doc_part, generated_text, serialized, fix_result = future.result()
                    if generated_text is not None:
                        backfill_map_post_val_fix[doc_part] = generated_text
                    if serialized is not None:
                        results["detailedResults"][doc_part] = serialized
                    if fix_result is not None and fix_result.validation:
                        # 与 Step 2 同一谓词, 避免修复后统计再次误判
                        all_passed = all(_is_validation_passed(v) for v in fix_result.validation)
                        if all_passed:
                            results["results"][doc_part]["passed"] = True
                            results["results"][doc_part]["consistencyPassed"] = True
                            passed_parts += 1
                            failed_parts -= 1
                except Exception as e:
                    print(f"    Error fixing doc_part {dp}: {e}")

        results["backfillMapPostValFix"] = backfill_map_post_val_fix
        print(f"\n  Post-validation fix complete: {len(backfill_map_post_val_fix)} parts fixed")
    else:
        print("\n=== Step 3: Post-Validation Fix (skipped, all passed) ===")

    # ── 汇总统计 ─────────────────────────────────────────────────
    results["summary"] = {
        "totalParts": len(doc_subpart_todo_list),
        "passedParts": passed_parts,
        "failedParts": failed_parts,
        "overallPassed": failed_parts == 0,
    }
    results["generatedAt"] = datetime.now().isoformat()

    print(f"\n  Summary: {passed_parts} passed, {failed_parts} failed out of {len(doc_subpart_todo_list)}")
    if results["backfillMapPostValFix"]:
        fix_items = ", ".join(f"{k!r}: {v[:50]!r}..." for k, v in results["backfillMapPostValFix"].items())
        print(f"  backfillMapPostValFix: {{{fix_items}}}")
    _output_result(results, data_path_resolved)
    return results


def main():
    parser = argparse.ArgumentParser(description="Consistency Check Pipeline - 一致性检测流水线 (validate_only 模式)")
    parser.add_argument("--snapshot-id", required=True, help="快照 ID")
    parser.add_argument("--workspace-dir", required=True, help="工作区根目录 (绝对路径)")
    parser.add_argument("--original-op-dir", required=True, help="算子目录 (相对于 workspace-dir)")
    parser.add_argument("--data-path", required=True, help="数据输出路径 (相对于 workspace-dir)")
    parser.add_argument("--operator-version", default="base", help="算子版本 (默认: base)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    # 运行开始清理旧哑文件 (DONE/FAILED), 避免上次运行残留标记被误判为本次状态
    _clear_sentinel_files(args.workspace_dir, args.data_path)

    try:
        run_pipeline(
            snapshot_id=args.snapshot_id,
            workspace_dir=args.workspace_dir,
            original_op_dir=args.original_op_dir,
            data_path=args.data_path,
            operator_version=args.operator_version,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nPipeline FAILED: {e}")
        traceback.print_exc()

        # 写入 FAILED 文件 (终态互斥: 先删除旧 DONE)
        workspace = Path(args.workspace_dir).resolve()
        data_path_resolved = (workspace / args.data_path).resolve()
        data_path_resolved.mkdir(parents=True, exist_ok=True)
        _unlink_quietly(data_path_resolved / "DONE")
        failed_file = data_path_resolved / "FAILED"
        failed_file.write_text(
            json.dumps(
                {"error": str(e), "failedAt": datetime.now().isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"FAILED file written: {failed_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
